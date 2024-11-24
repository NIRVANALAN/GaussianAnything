import torch.nn as nn
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from pdb import set_trace as st

from ldm.modules.attention import MemoryEfficientCrossAttention
from .dit_models_xformers import *
# from apex.normalization import FusedLayerNorm as LayerNorm
from torch.nn import LayerNorm
# from apex.normalization import FusedRMSNorm as RMSNorm

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except:
    from dit.norm import RMSNorm

from timm.models.vision_transformer import Mlp

from vit.vit_triplane import XYZPosEmbed

from .dit_trilatent import DiT_PCD_PixelArt


class DiT_I23D(DiT):
    # DiT with 3D_aware operations
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        pooling_ctx_dim=768,
        roll_out=False,
        vit_blk=ImageCondDiTBlock,
        final_layer_blk=T2IFinalLayer,
        enable_rope=False,
    ):
        # st()
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                         mixed_prediction, context_dim, roll_out, vit_blk,
                         T2IFinalLayer, enable_rope=enable_rope)

        assert self.roll_out

        # if context_dim is not None:
        # self.dino_proj = CaptionEmbedder(context_dim,
        self.clip_ctx_dim = 1024 # vit-l
        # self.dino_proj = CaptionEmbedder(self.clip_ctx_dim, # ! dino-vitl/14 here, for img-cond
        self.dino_proj = CaptionEmbedder(context_dim, # ! dino-vitb/14 here, for MV-cond. hard coded for now...
        # self.dino_proj = CaptionEmbedder(1024, # ! dino-vitb/14 here, for MV-cond. hard coded for now...
                                                hidden_size,
                                                act_layer=approx_gelu)

        self.clip_spatial_proj = CaptionEmbedder(1024, # clip_I-L
                                                hidden_size,
                                                act_layer=approx_gelu)

    def init_PE_3D_aware(self):

        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.plane_n * self.x_embedder.num_patches, self.embed_dim),
                                      requires_grad=False)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        p = int(self.x_embedder.num_patches**0.5)
        D = self.pos_embed.shape[-1]
        grid_size = (self.plane_n, p * p)  # B n HW C

        pos_embed = get_2d_sincos_pos_embed(D, grid_size).reshape(
            self.plane_n * p * p, D)  # H*W, D

        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

    def initialize_weights(self):
        super().initialize_weights()

        # ! add 3d-aware PE
        self.init_PE_3D_aware()

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)
        # context = self.clip_text_proj(context)
        clip_cls_token = self.clip_text_proj(context['vector'])
        clip_spatial_token, dino_spatial_token = context['crossattn'][..., :self.clip_ctx_dim], self.dino_proj(context['crossattn'][..., self.clip_ctx_dim:])

        t = self.t_embedder(timesteps) + clip_cls_token  # (N, D)
        # ! todo, return spatial clip features.

        # if self.roll_out:  # !
        x = rearrange(x, 'b (c n) h w->(b n) c h w',
                      n=3)  # downsample with same conv
        x = self.x_embedder(x)  # (b n) c h/f w/f

        x = rearrange(x, '(b n) l c -> b (n l) c', n=3)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # if self.roll_out:  # ! roll-out in the L dim, not B dim. add condition to all tokens.
        # x = rearrange(x, '(b n) l c ->b (n l) c', n=3)

        # assert context.ndim == 2
        # if isinstance(context, dict):
        #     context = context['crossattn']  # sgm conditioner compat


        # c = t + context
        # else:
        # c = t  # BS 1024

        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t, dino_spatial_token=dino_spatial_token, clip_spatial_token=clip_spatial_token)  # (N, T, D)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, 'b (n l) c ->(b n) l c', n=3)

        x = self.unpatchify(x)  # (N, out_channels, H, W)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, '(b n) c h w -> b (c n) h w', n=3)
            # x = rearrange(x, 'b n) c h w -> b (n c) h w', n=3)

        # cast to float32 for better accuracy
        x = x.to(torch.float32).contiguous()

        return x

    # ! compat issue
    def forward_with_cfg(self, x, t, context, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # half = x[: len(x) // 2]
        # combined = torch.cat([half, half], dim=0)
        eps = self.forward(x, t, context)
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps




class DiT_I23D_PixelArt(DiT_I23D):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        pooling_ctx_dim=768,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArtRMSNorm,
        final_layer_blk=FinalLayer,
        create_cap_embedder=True,
        enable_rope=False,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                        #  mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArt,
                         mixed_prediction, context_dim, pooling_ctx_dim, roll_out, vit_blk,
                         final_layer_blk, 
                         enable_rope=enable_rope)

        # ! a shared one
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # ! single
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        del self.clip_text_proj
        if create_cap_embedder:
            self.cap_embedder = nn.Sequential( # TODO, init with zero here.
                LayerNorm(pooling_ctx_dim),
                nn.Linear(
                    pooling_ctx_dim,
                    hidden_size,
                ),
            )

            nn.init.constant_(self.cap_embedder[-1].weight, 0)
            nn.init.constant_(self.cap_embedder[-1].bias, 0)
        else:
            self.cap_embedder = nn.Identity() # placeholder

        print(self) # check model arch

        self.attention_y_norm = RMSNorm(
            1024, eps=1e-5
        )  # https://github.com/Alpha-VLLM/Lumina-T2X/blob/0c8dd6a07a3b7c18da3d91f37b1e00e7ae661293/lumina_t2i/models/model.py#L570C9-L570C61


    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)
        # context = self.clip_text_proj(context)
        clip_cls_token = self.cap_embedder(context['vector'])
        clip_spatial_token, dino_spatial_token = context['crossattn'][..., :self.clip_ctx_dim], self.dino_proj(context['crossattn'][..., self.clip_ctx_dim:])
        clip_spatial_token = self.attention_y_norm(clip_spatial_token) # avoid re-normalization in each blk

        t = self.t_embedder(timesteps) + clip_cls_token  # (N, D)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        # if self.roll_out:  # !
        x = rearrange(x, 'b (c n) h w->(b n) c h w',
                      n=3)  # downsample with same conv
        x = self.x_embedder(x)  # (b n) c h/f w/f

        x = rearrange(x, '(b n) l c -> b (n l) c', n=3)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # if self.roll_out:  # ! roll-out in the L dim, not B dim. add condition to all tokens.
        # x = rearrange(x, '(b n) l c ->b (n l) c', n=3)

        # assert context.ndim == 2
        # if isinstance(context, dict):
        #     context = context['crossattn']  # sgm conditioner compat


        # c = t + context
        # else:
        # c = t  # BS 1024

        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t0, dino_spatial_token=dino_spatial_token, clip_spatial_token=clip_spatial_token)  # (N, T, D)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, 'b (n l) c ->(b n) l c', n=3)

        x = self.unpatchify(x)  # (N, out_channels, H, W)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, '(b n) c h w -> b (c n) h w', n=3)
            # x = rearrange(x, 'b n) c h w -> b (n c) h w', n=3)

        # cast to float32 for better accuracy
        x = x.to(torch.float32).contiguous()

        return x

class DiT_I23D_PCD_PixelArt(DiT_I23D_PixelArt):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        pooling_ctx_dim=768,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArtRMSNorm,
        final_layer_blk=FinalLayer,
        create_cap_embedder=True,
        use_clay_ca=False,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                        #  mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArt,
                         mixed_prediction, context_dim, pooling_ctx_dim, roll_out, vit_blk,
                         final_layer_blk)

        self.x_embedder = Mlp(in_features=in_channels,
                          hidden_features=hidden_size,
                          out_features=hidden_size,
                          act_layer=approx_gelu,
                          drop=0)
        del self.pos_embed
        self.use_clay_ca = use_clay_ca
        if use_clay_ca:
            del self.dino_proj # no prepending required.

        # add ln_pred and ln_post, as in point-e. (does not help, worse performance)
        # self.ln_pre = LayerNorm(hidden_size)
        # self.ln_post = LayerNorm(hidden_size)
    @staticmethod
    def precompute_freqs_cis(
        dim: int,
        end: int,
        theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        ntk_factor: float = 1.0,
    ):
        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex
                exponentials.
        """

        theta = theta * ntk_factor

        print(f"theta {theta} rope scaling {rope_scaling_factor} ntk {ntk_factor}")
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float().cuda() / dim))
        t = torch.arange(end, device=freqs.device, dtype=torch.float)  # type: ignore
        t = t / rope_scaling_factor
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)

        # dino_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        # t = self.t_embedder(timesteps)

        # global condition
        if 'caption_vector' in context:
            clip_cls_token = self.cap_embedder(context['caption_vector'])
        elif 'img_vector' in context:
            clip_cls_token = self.cap_embedder(context['img_vector'])
        else:
            clip_cls_token = 0

        # spatial condition
        clip_spatial_token, dino_spatial_token = context['img_crossattn'][..., :self.clip_ctx_dim], context['img_crossattn'][..., self.clip_ctx_dim:]
        if not self.use_clay_ca:
            dino_spatial_token=self.dino_proj(dino_spatial_token)

        t = self.t_embedder(timesteps) + clip_cls_token  # (N, D)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        x = self.x_embedder(x)

        # add a norm layer here, as in point-e
        # x = self.ln_pre(x)

        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t0, dino_spatial_token=dino_spatial_token, clip_spatial_token=clip_spatial_token, clip_caption_token=context.get('caption_crossattn'))

        # add a norm layer here, as in point-e
        # x = self.ln_post(x)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        x = x.to(torch.float32).contiguous()

        return x

# dino only version
class DiT_I23D_PCD_PixelArt_noclip(DiT_I23D_PixelArt):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        pooling_ctx_dim=768,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArtRMSNormNoClip,
        final_layer_blk=FinalLayer,
        create_cap_embedder=True,
        use_clay_ca=False,
        has_caption=False,
        # has_rope=False,
        rope_scaling_factor: float = 1.0,
        ntk_factor: float = 1.0,
        enable_rope=False,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                        #  mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArt,
                         mixed_prediction, context_dim, pooling_ctx_dim, roll_out, vit_blk,
                         final_layer_blk, enable_rope=enable_rope)

        self.x_embedder = Mlp(in_features=in_channels,
                          hidden_features=hidden_size,
                          out_features=hidden_size,
                          act_layer=approx_gelu,
                          drop=0)
        del self.pos_embed
        del self.dino_proj

        self.enable_rope = enable_rope
        if self.enable_rope: # implementation copied from Lumina-T2X code base
            self.freqs_cis = DiT_I23D_PCD_PixelArt.precompute_freqs_cis(
            hidden_size // num_heads,
            40000,
            rope_scaling_factor=rope_scaling_factor,
            ntk_factor=ntk_factor,
        )
        else:
            self.freqs_cis = None

        self.rope_scaling_factor = rope_scaling_factor
        self.ntk_factor = ntk_factor

        self.use_clay_ca = use_clay_ca

        self.has_caption = has_caption
        pooled_vector_dim = context_dim
        if has_caption:
            pooled_vector_dim += 768

        self.pooled_vec_embedder = nn.Sequential( # TODO, init with zero here.
            LayerNorm(pooled_vector_dim),
            nn.Linear(
                pooled_vector_dim,
                hidden_size,
            ),
        )
        nn.init.constant_(self.pooled_vec_embedder[-1].weight, 0)
        nn.init.constant_(self.pooled_vec_embedder[-1].bias, 0)

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)

        # dino_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        # t = self.t_embedder(timesteps)

        # clip_cls_token = self.cap_embedder(context['vector'])
        # clip_spatial_token, dino_spatial_token = context['crossattn'][..., :self.clip_ctx_dim], self.dino_proj(context['crossattn'][..., self.clip_ctx_dim:])
        # dino_spatial_token = context['crossattn']
        dino_spatial_token = context['img_crossattn']
        dino_pooled_vector = context['img_vector']
        if self.has_caption:
            clip_caption_token = context.get('caption_crossattn')
            pooled_vector = torch.cat([dino_pooled_vector, context.get('caption_vector')], -1) # concat dino_vector
        else:
            clip_caption_token = None
            pooled_vector = dino_pooled_vector
        

        t = self.t_embedder(timesteps) + self.pooled_vec_embedder(pooled_vector)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        x = self.x_embedder(x)

        freqs_cis = None
        if self.enable_rope:
            freqs_cis=self.freqs_cis[: x.size(1)]

        # add a norm layer here, as in point-e
        # x = self.ln_pre(x)

        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t0, dino_spatial_token=dino_spatial_token, clip_caption_token=clip_caption_token, freqs_cis=freqs_cis)

        # add a norm layer here, as in point-e
        # x = self.ln_post(x)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        x = x.to(torch.float32).contiguous()

        return x


# xyz-diff


# xyz-cond tex diff
class DiT_I23D_PCD_PixelArt_xyz_cond_kl_diff(DiT_I23D_PCD_PixelArt):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        pooling_ctx_dim=768,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArtRMSNorm,
        final_layer_blk=FinalLayer,
        create_cap_embedder=True,
        use_pe_cond=False,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                        #  mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArt,
                         mixed_prediction, context_dim, pooling_ctx_dim, roll_out, vit_blk,
                         final_layer_blk)

        self.use_pe_cond = use_pe_cond
        self.x_embedder = Mlp(in_features=in_channels+3*(1-use_pe_cond),
                          hidden_features=hidden_size,
                          out_features=hidden_size,
                          act_layer=approx_gelu,
                          drop=0)

        if use_pe_cond:
            self.xyz_pos_embed = XYZPosEmbed(hidden_size)

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)

        # dino_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        # t = self.t_embedder(timesteps)

        clip_cls_token = self.cap_embedder(context['vector'])
        clip_spatial_token, dino_spatial_token = context['crossattn'][..., :self.clip_ctx_dim], self.dino_proj(context['crossattn'][..., self.clip_ctx_dim:])

        fps_xyz = context['fps-xyz']

        t = self.t_embedder(timesteps) + clip_cls_token  # (N, D)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        if self.use_pe_cond:
            x = self.x_embedder(x) + self.xyz_pos_embed(fps_xyz) # point-wise addition
        else: # use concat to add info
            x = torch.cat([fps_xyz, x], dim=-1)
            x = self.x_embedder(x)

        # add a norm layer here, as in point-e
        # x = self.ln_pre(x)

        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t0, dino_spatial_token=dino_spatial_token, clip_spatial_token=clip_spatial_token)

        # add a norm layer here, as in point-e
        # x = self.ln_post(x)

        x = self.final_layer(x, t) # no loss on the xyz side 

        x = x.to(torch.float32).contiguous()

        return x

# xyz-cond tex diff, but clay
class DiT_I23D_PCD_PixelArt_noclip_clay_stage2(DiT_I23D_PCD_PixelArt_noclip):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        pooling_ctx_dim=768,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArtRMSNorm,
        final_layer_blk=FinalLayer,
        create_cap_embedder=True,
        use_pe_cond=False,
        has_caption=False,
        use_clay_ca=False,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                        #  mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArt,
                         mixed_prediction, context_dim, pooling_ctx_dim, roll_out, vit_blk,
                         final_layer_blk, use_clay_ca=use_clay_ca, has_caption=has_caption)

        self.has_caption = False
        self.use_pe_cond = use_pe_cond
        self.x_embedder = Mlp(in_features=in_channels+3*(1-use_pe_cond),
                          hidden_features=hidden_size,
                          out_features=hidden_size,
                          act_layer=approx_gelu,
                          drop=0)

        if use_pe_cond:
            self.xyz_pos_embed = XYZPosEmbed(hidden_size)

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)

        dino_spatial_token = context['img_crossattn']
        dino_pooled_vector = context['img_vector']
        if self.has_caption:
            clip_caption_token = context.get('caption_crossattn')
            pooled_vector = torch.cat([dino_pooled_vector, context.get('caption_vector')], -1) # concat dino_vector
        else:
            clip_caption_token = None
            pooled_vector = dino_pooled_vector
        
        t = self.t_embedder(timesteps) + self.pooled_vec_embedder(pooled_vector)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        fps_xyz = context['fps-xyz']
        if self.use_pe_cond:
            x = self.x_embedder(x) + self.xyz_pos_embed(fps_xyz) # point-wise addition
        else: # use concat to add info
            x = torch.cat([fps_xyz, x], dim=-1)
            x = self.x_embedder(x)

        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t0, dino_spatial_token=dino_spatial_token, clip_caption_token=clip_caption_token)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        x = x.to(torch.float32).contiguous()

        return x


class DiT_I23D_PixelArt_MVCond(DiT_I23D_PixelArt):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        pooling_ctx_dim=768,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArt,
        final_layer_blk=FinalLayer,
        create_cap_embedder=False,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                        #  mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArt,
                         mixed_prediction, context_dim,
                         pooling_ctx_dim, roll_out, ImageCondDiTBlockPixelArtRMSNorm,
                         final_layer_blk, create_cap_embedder=create_cap_embedder)


        # support multi-view img condition
        # DINO handles global pooling here; clip takes care of camera-cond with ModLN
        # Input DINO concat also + global pool. InstantMesh adopts DINO (but CA).
        # expected: support dynamic numbers of frames? since CA, shall be capable of. Any number of context window size.
        del self.dino_proj

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)

        # st()
        # (Pdb) p context.keys()
        # dict_keys(['crossattn', 'vector', 'concat'])
        # (Pdb) p context['vector'].shape
        # torch.Size([2, 768])
        # (Pdb) p context['crossattn'].shape
        # torch.Size([2, 256, 1024])
        # (Pdb) p context['concat'].shape
        # torch.Size([2, 4, 256, 768]) # mv dino spatial features

        # ! clip spatial tokens for append self-attn, thus add a projection layer (self.dino_proj)
        # DINO features sent via crossattn, thus no proj required (already KV linear layers in crossattn blk)
        clip_cls_token, clip_spatial_token = self.cap_embedder(context['vector']), self.clip_spatial_proj(context['crossattn']) # no norm here required? QK norm is enough, since self.ln_post(x) in vit
        dino_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        t = self.t_embedder(timesteps) + clip_cls_token  # (N, D)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        # if self.roll_out:  # !
        x = rearrange(x, 'b (c n) h w->(b n) c h w',
                      n=3)  # downsample with same conv
        x = self.x_embedder(x)  # (b n) c h/f w/f

        x = rearrange(x, '(b n) l c -> b (n l) c', n=3)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        for blk_idx, block in enumerate(self.blocks):
            # x = block(x, t0, dino_spatial_token=dino_spatial_token, clip_spatial_token=clip_spatial_token)  # (N, T, D)
            # ! DINO tokens for CA, CLIP tokens for append here.
            x = block(x, t0, dino_spatial_token=clip_spatial_token, clip_spatial_token=dino_spatial_token)  # (N, T, D)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, 'b (n l) c ->(b n) l c', n=3)

        x = self.unpatchify(x)  # (N, out_channels, H, W)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, '(b n) c h w -> b (c n) h w', n=3)

        x = x.to(torch.float32).contiguous()

        return x


class DiT_I23D_PixelArt_MVCond_noClip(DiT_I23D_PixelArt):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        pooling_ctx_dim=768,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArt,
        final_layer_blk=FinalLayer,
        create_cap_embedder=False,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                        #  mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArt,
                         mixed_prediction, context_dim,
                         pooling_ctx_dim, roll_out, 
                         ImageCondDiTBlockPixelArtRMSNormNoClip,
                         final_layer_blk, 
                         create_cap_embedder=create_cap_embedder)


        # support multi-view img condition
        # DINO handles global pooling here; clip takes care of camera-cond with ModLN
        # Input DINO concat also + global pool. InstantMesh adopts DINO (but CA).
        # expected: support dynamic numbers of frames? since CA, shall be capable of. Any number of context window size.

        del self.dino_proj
        del self.clip_spatial_proj, self.cap_embedder # no clip required

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)

        # st()
        # (Pdb) p context.keys()
        # dict_keys(['crossattn', 'vector', 'concat'])
        # (Pdb) p context['vector'].shape
        # torch.Size([2, 768])
        # (Pdb) p context['crossattn'].shape
        # torch.Size([2, 256, 1024])
        # (Pdb) p context['concat'].shape
        # torch.Size([2, 4, 256, 768]) # mv dino spatial features

        # ! clip spatial tokens for append self-attn, thus add a projection layer (self.dino_proj)
        # DINO features sent via crossattn, thus no proj required (already KV linear layers in crossattn blk)
        # clip_cls_token, clip_spatial_token = self.cap_embedder(context['vector']), self.clip_spatial_proj(context['crossattn']) # no norm here required? QK norm is enough, since self.ln_post(x) in vit
        dino_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        # t = self.t_embedder(timesteps) + clip_cls_token  # (N, D)
        t = self.t_embedder(timesteps)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        # if self.roll_out:  # !
        x = rearrange(x, 'b (c n) h w->(b n) c h w',
                      n=3)  # downsample with same conv
        x = self.x_embedder(x)  # (b n) c h/f w/f

        x = rearrange(x, '(b n) l c -> b (n l) c', n=3)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        for blk_idx, block in enumerate(self.blocks):
            # x = block(x, t0, dino_spatial_token=dino_spatial_token, clip_spatial_token=clip_spatial_token)  # (N, T, D)
            # ! DINO tokens for CA, CLIP tokens for append here.
            x = block(x, t0, dino_spatial_token=dino_spatial_token)  # (N, T, D)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, 'b (n l) c ->(b n) l c', n=3)

        x = self.unpatchify(x)  # (N, out_channels, H, W)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, '(b n) c h w -> b (c n) h w', n=3)

        x = x.to(torch.float32).contiguous()

        return x





# pcd-structured latent ddpm

class DiT_pcd_I23D_PixelArt_MVCond(DiT_I23D_PixelArt_MVCond_noClip):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        pooling_ctx_dim=768,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArt,
        final_layer_blk=FinalLayer,
        create_cap_embedder=False,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                        #  mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArt,
                         mixed_prediction, context_dim,
                         pooling_ctx_dim,
                          roll_out, ImageCondDiTBlockPixelArtRMSNorm,
                         final_layer_blk,
                         create_cap_embedder=create_cap_embedder)
        # ! first, normalize xyz from [-0.45,0.45] to [-1,1]
        # Then, encode xyz with point fourier feat + MLP projection, serves as PE here.
        # a separate MLP for the KL feature
        # add them together in the feature space
        # use a single MLP (final_layer) to map them back to 16 + 3 dims.
        self.x_embedder = Mlp(in_features=in_channels,
                          hidden_features=hidden_size,
                          out_features=hidden_size,
                          act_layer=approx_gelu,
                          drop=0)
        del self.pos_embed


    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)

        dino_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        t = self.t_embedder(timesteps)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        x = self.x_embedder(x)

        for blk_idx, block in enumerate(self.blocks):
            # x = block(x, t0, dino_spatial_token=dino_spatial_token, clip_spatial_token=clip_spatial_token)  # (N, T, D)
            # ! DINO tokens for CA, CLIP tokens for append here.
            x = block(x, t0, dino_spatial_token=dino_spatial_token)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        x = x.to(torch.float32).contiguous()

        return x

class DiT_pcd_I23D_PixelArt_MVCond_clay(DiT_PCD_PixelArt):
    # fine-tune the mv model from text conditioned model
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        pooling_ctx_dim=768,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArt,
        final_layer_blk=FinalLayer,
        create_cap_embedder=False, **kwargs
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                        #  mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArt,
                         mixed_prediction, context_dim,
                        #  pooling_ctx_dim,
                          roll_out, vit_blk,
                         final_layer_blk,)
                        #  create_cap_embedder=create_cap_embedder)

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert context is not None

        clip_cls_token = self.cap_embedder(context['caption_vector']) # pooled
        t = self.t_embedder(timesteps) + clip_cls_token  # (N, D)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        x = self.x_embedder(x)

        # ! spatial tokens
        dino_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        # assert context.ndim == 2
        # if isinstance(context, dict):
            # context = context['caption_crossattn']  # sgm conditioner compat

        # loop dit block
        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t0, clip_caption_token=context['caption_crossattn'], 
                             dino_spatial_token=dino_spatial_token)  # (N, T, D)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        # cast to float32 for better accuracy
        x = x.to(torch.float32).contiguous()

        return x


# single-img pretrained clay

class DiT_pcd_I23D_PixelArt_MVCond_clay_i23dpt(DiT_I23D_PCD_PixelArt_noclip):
    # fine-tune the mv model from text conditioned model
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        pooling_ctx_dim=768,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArt,
        final_layer_blk=FinalLayer,
        create_cap_embedder=False, **kwargs
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                         mixed_prediction, context_dim,
                         pooling_ctx_dim,
                          roll_out, vit_blk,
                         final_layer_blk,)

        self.has_caption = False

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)

        # dino_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        # t = self.t_embedder(timesteps)

        # clip_cls_token = self.cap_embedder(context['vector'])
        # clip_spatial_token, dino_spatial_token = context['crossattn'][..., :self.clip_ctx_dim], self.dino_proj(context['crossattn'][..., self.clip_ctx_dim:])
        # dino_spatial_token = context['crossattn']
        # st()
        dino_spatial_token = context['img_crossattn']
        dino_pooled_vector = context['img_vector']
        dino_mv_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        if self.has_caption:
            clip_caption_token = context.get('caption_crossattn')
            pooled_vector = torch.cat([dino_pooled_vector, context.get('caption_vector')], -1) # concat dino_vector
        else:
            clip_caption_token = None
            pooled_vector = dino_pooled_vector
        

        t = self.t_embedder(timesteps) + self.pooled_vec_embedder(pooled_vector)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        x = self.x_embedder(x)

        # add a norm layer here, as in point-e
        # x = self.ln_pre(x)

        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t0, dino_spatial_token=dino_spatial_token, dino_mv_spatial_token=dino_mv_spatial_token)

        # add a norm layer here, as in point-e
        # x = self.ln_post(x)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        x = x.to(torch.float32).contiguous()

        return x

# stage 2
class DiT_pcd_I23D_PixelArt_MVCond_clay_i23dpt_stage2(DiT_I23D_PCD_PixelArt_noclip):
    # fine-tune the mv model from text conditioned model
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        pooling_ctx_dim=768,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArt,
        final_layer_blk=FinalLayer,
        create_cap_embedder=False, **kwargs
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                         mixed_prediction, context_dim,
                         pooling_ctx_dim,
                          roll_out, vit_blk,
                         final_layer_blk,)

        self.has_caption = False

        self.use_pe_cond = True
        self.x_embedder = Mlp(in_features=in_channels+3*(1-self.use_pe_cond),
                          hidden_features=hidden_size,
                          out_features=hidden_size,
                          act_layer=approx_gelu,
                          drop=0)

        if self.use_pe_cond:
            self.xyz_pos_embed = XYZPosEmbed(hidden_size)

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)

        # dino_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        # t = self.t_embedder(timesteps)

        # clip_cls_token = self.cap_embedder(context['vector'])
        # clip_spatial_token, dino_spatial_token = context['crossattn'][..., :self.clip_ctx_dim], self.dino_proj(context['crossattn'][..., self.clip_ctx_dim:])
        # dino_spatial_token = context['crossattn']
        # st()
        dino_spatial_token = context['img_crossattn']
        dino_pooled_vector = context['img_vector']
        dino_mv_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        if self.has_caption:
            clip_caption_token = context.get('caption_crossattn')
            pooled_vector = torch.cat([dino_pooled_vector, context.get('caption_vector')], -1) # concat dino_vector
        else:
            clip_caption_token = None
            pooled_vector = dino_pooled_vector
        

        t = self.t_embedder(timesteps) + self.pooled_vec_embedder(pooled_vector)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        # x = self.x_embedder(x)

        fps_xyz = context['fps-xyz']
        if self.use_pe_cond:
            x = self.x_embedder(x) + self.xyz_pos_embed(fps_xyz) # point-wise addition
        else: # use concat to add info
            x = torch.cat([fps_xyz, x], dim=-1)
            x = self.x_embedder(x)


        # add a norm layer here, as in point-e
        # x = self.ln_pre(x)

        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t0, dino_spatial_token=dino_spatial_token, dino_mv_spatial_token=dino_mv_spatial_token)

        # add a norm layer here, as in point-e
        # x = self.ln_post(x)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        x = x.to(torch.float32).contiguous()

        return x




class DiT_pcd_I23D_PixelArt_MVCond_clay_i23dpt_noi23d(DiT_I23D_PCD_PixelArt_noclip):
    # fine-tune the mv model from text conditioned model
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        pooling_ctx_dim=768,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArt,
        final_layer_blk=FinalLayer,
        create_cap_embedder=False, **kwargs
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                         mixed_prediction, context_dim,
                         pooling_ctx_dim,
                          roll_out, vit_blk,
                         final_layer_blk,)

        self.has_caption = False
        del self.pooled_vec_embedder

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)

        # dino_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        # t = self.t_embedder(timesteps)

        # clip_cls_token = self.cap_embedder(context['vector'])
        # clip_spatial_token, dino_spatial_token = context['crossattn'][..., :self.clip_ctx_dim], self.dino_proj(context['crossattn'][..., self.clip_ctx_dim:])
        # dino_spatial_token = context['crossattn']
        # st()
        # dino_spatial_token = context['img_crossattn']
        # dino_pooled_vector = context['img_vector']
        dino_mv_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        if self.has_caption:
            clip_caption_token = context.get('caption_crossattn')
            pooled_vector = torch.cat([dino_pooled_vector, context.get('caption_vector')], -1) # concat dino_vector
        else:
            clip_caption_token = None
            # pooled_vector = dino_pooled_vector
            pooled_vector = None
        

        # t = self.t_embedder(timesteps) + self.pooled_vec_embedder(pooled_vector)
        t = self.t_embedder(timesteps) 
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        x = self.x_embedder(x)

        # add a norm layer here, as in point-e
        # x = self.ln_pre(x)

        for blk_idx, block in enumerate(self.blocks):
            # x = block(x, t0, dino_spatial_token=dino_spatial_token, dino_mv_spatial_token=dino_mv_spatial_token)
            x = block(x, t0, dino_mv_spatial_token=dino_mv_spatial_token)

        # add a norm layer here, as in point-e
        # x = self.ln_post(x)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        x = x.to(torch.float32).contiguous()

        return x


#################################################################################
#                                   DiT_I23D Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT_I23D(depth=28,
                         hidden_size=1152,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)


def DiT_L_2(**kwargs):
    return DiT_I23D(depth=24,
                         hidden_size=1024,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)


def DiT_B_2(**kwargs):
    return DiT_I23D(depth=12,
                         hidden_size=768,
                         patch_size=2,
                         num_heads=12,
                         **kwargs)


def DiT_B_1(**kwargs):
    return DiT_I23D(depth=12,
                         hidden_size=768,
                         patch_size=1,
                         num_heads=12,
                         **kwargs)


def DiT_L_Pixelart_2(**kwargs):
    return DiT_I23D_PixelArt(depth=24,
                         hidden_size=1024,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)


def DiT_B_Pixelart_2(**kwargs):
    return DiT_I23D_PixelArt(depth=12,
                         hidden_size=768,
                         patch_size=2,
                         num_heads=12,
                         **kwargs)

def DiT_L_Pixelart_MV_2(**kwargs):
    return DiT_I23D_PixelArt_MVCond(depth=24,
                         hidden_size=1024,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)

def DiT_L_Pixelart_MV_2_noclip(**kwargs):
    return DiT_I23D_PixelArt_MVCond_noClip(depth=24,
                         hidden_size=1024,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)

def DiT_XL_Pixelart_MV_2(**kwargs):
    return DiT_I23D_PixelArt_MVCond(depth=28,
                         hidden_size=1152,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)



def DiT_B_Pixelart_MV_2(**kwargs):
    return DiT_I23D_PixelArt_MVCond(depth=12,
                         hidden_size=768,
                         patch_size=2,
                         num_heads=12,
                         **kwargs)

# pcd latent 

def DiT_L_Pixelart_MV_pcd(**kwargs):
    return DiT_pcd_I23D_PixelArt_MVCond(depth=24,
                         hidden_size=1024,
                         patch_size=1, # no spatial compression here
                         num_heads=16,
                         **kwargs)

# raw gs i23d
def DiT_L_Pixelart_pcd(**kwargs):
    return DiT_I23D_PCD_PixelArt(depth=24,
    # return DiT_I23D_PCD_PixelArt_noclip(depth=24,
                         hidden_size=1024,
                         patch_size=1, # no spatial compression here
                         num_heads=16,
                         **kwargs)

def DiT_L_Pixelart_clay_pcd(**kwargs):
    return DiT_I23D_PCD_PixelArt_noclip(depth=24,
                        vit_blk=ImageCondDiTBlockPixelArtRMSNormClayLRM,
                        use_clay_ca=True,
                         hidden_size=1024,
                         patch_size=1, # no spatial compression here
                         num_heads=16,
                         enable_rope=False,
                         **kwargs)

def DiT_XL_Pixelart_clay_pcd(**kwargs):
    return DiT_I23D_PCD_PixelArt_noclip(depth=28,
                        vit_blk=ImageCondDiTBlockPixelArtRMSNormClayLRM,
                        use_clay_ca=True,
                         hidden_size=1152,
                         patch_size=1, # no spatial compression here
                         num_heads=16,
                         enable_rope=False,
                         **kwargs)


def DiT_B_Pixelart_clay_pcd(**kwargs):
    return DiT_I23D_PCD_PixelArt_noclip(depth=12,
                        vit_blk=ImageCondDiTBlockPixelArtRMSNormClayLRM,
                        use_clay_ca=True,
                         hidden_size=768,
                         patch_size=1, # no spatial compression here
                         num_heads=12,
                         **kwargs)

def DiT_L_Pixelart_clay_pcd_stage2(**kwargs):
    return DiT_I23D_PCD_PixelArt_noclip_clay_stage2(depth=24,
                        vit_blk=ImageCondDiTBlockPixelArtRMSNormClayLRM,
                        use_clay_ca=True,
                         hidden_size=1024,
                         patch_size=1, # no spatial compression here
                         num_heads=16,
                         use_pe_cond=True,
                         **kwargs)

def DiT_B_Pixelart_clay_pcd_stage2(**kwargs):
    return DiT_I23D_PCD_PixelArt_noclip_clay_stage2(depth=12,
                        vit_blk=ImageCondDiTBlockPixelArtRMSNormClayLRM,
                        use_clay_ca=True,
                         hidden_size=768,
                         patch_size=1, # no spatial compression here
                         num_heads=12,
                         use_pe_cond=True,
                         **kwargs)



def DiT_L_Pixelart_clay_tandi_pcd(**kwargs):
    return DiT_I23D_PCD_PixelArt_noclip(depth=24,
                        vit_blk=ImageCondDiTBlockPixelArtRMSNormClayText,
                        use_clay_ca=True,
                         hidden_size=1024,
                         patch_size=1, # no spatial compression here
                         num_heads=16,
                         has_caption=True,
                         **kwargs)

def DiT_B_Pixelart_clay_tandi_pcd(**kwargs):
    return DiT_I23D_PCD_PixelArt_noclip(depth=12,
                        vit_blk=ImageCondDiTBlockPixelArtRMSNormClayText,
                        use_clay_ca=True,
                         hidden_size=768,
                         patch_size=1, # no spatial compression here
                         num_heads=12,
                         has_caption=True,
                         **kwargs)


def DiT_B_Pixelart_pcd(**kwargs):
    return DiT_I23D_PCD_PixelArt(depth=12,
                         hidden_size=768,
                         patch_size=1, # no spatial compression here
                         num_heads=12,
                         **kwargs)

def DiT_B_Pixelart_pcd_cond_diff(**kwargs):
    return DiT_I23D_PCD_PixelArt_xyz_cond_kl_diff(depth=12,
                         hidden_size=768,
                         patch_size=1, # no spatial compression here
                         num_heads=12,
                         **kwargs)

def DiT_B_Pixelart_pcd_cond_diff_pe(**kwargs):
    return DiT_I23D_PCD_PixelArt_xyz_cond_kl_diff(depth=12,
                         hidden_size=768,
                         patch_size=1, # no spatial compression here
                         vit_blk=ImageCondDiTBlockPixelArtRMSNormClayLRM,
                         num_heads=12,
                         use_pe_cond=True,
                         **kwargs)

def DiT_L_Pixelart_pcd_cond_diff_pe(**kwargs):
    return DiT_I23D_PCD_PixelArt_xyz_cond_kl_diff(depth=24,
                         hidden_size=1024,
                         patch_size=1, # no spatial compression here
                         vit_blk=ImageCondDiTBlockPixelArtRMSNormClayLRM,
                         num_heads=16,
                         use_pe_cond=True,
                         **kwargs)

# mv version

def DiT_L_Pixelart_clay_mv_pcd(**kwargs):
    return DiT_pcd_I23D_PixelArt_MVCond_clay(depth=24,
                        vit_blk=ImageCondDiTBlockPixelArtRMSNormClayText,
                        use_clay_ca=True,
                         hidden_size=1024,
                         patch_size=1, # no spatial compression here
                         num_heads=16,
                         **kwargs)


def DiT_L_Pixelart_clay_mv_i23dpt_pcd(**kwargs):
    return DiT_pcd_I23D_PixelArt_MVCond_clay_i23dpt(depth=24,
                        vit_blk=ImageCondDiTBlockPixelArtRMSNormClayMV,
                        use_clay_ca=True,
                         hidden_size=1024,
                         patch_size=1, # no spatial compression here
                         num_heads=16,
                         **kwargs)


def DiT_L_Pixelart_clay_mv_i23dpt_pcd_noi23d(**kwargs):
    return DiT_pcd_I23D_PixelArt_MVCond_clay_i23dpt_noi23d(depth=24,
                        vit_blk=ImageCondDiTBlockPixelArtRMSNormClayMV_noi23d,
                        use_clay_ca=True,
                         hidden_size=1024,
                         patch_size=1, # no spatial compression here
                         num_heads=16,
                         **kwargs)


def DiT_L_Pixelart_clay_mv_i23dpt_pcd_stage2(**kwargs):
    return DiT_pcd_I23D_PixelArt_MVCond_clay_i23dpt_stage2(depth=24,
                        vit_blk=ImageCondDiTBlockPixelArtRMSNormClayMV,
                        use_clay_ca=True,
                         hidden_size=1024,
                         patch_size=1, # no spatial compression here
                         num_heads=16,
                         **kwargs)




DiT_models = {
    'DiT-XL/2': DiT_XL_2,
    'DiT-L/2': DiT_L_2,
    'DiT-B/2': DiT_B_2,
    'DiT-B/1': DiT_B_1,
    'DiT-PixArt-L/2': DiT_L_Pixelart_2,
    'DiT-PixArt-MV-XL/2': DiT_XL_Pixelart_MV_2,
    # 'DiT-PixArt-MV-L/2': DiT_L_Pixelart_MV_2,
    'DiT-PixArt-MV-L/2': DiT_L_Pixelart_MV_2_noclip,
    'DiT-PixArt-MV-PCD-L': DiT_L_Pixelart_MV_pcd,
    # raw xyz cond
    'DiT-PixArt-PCD-L': DiT_L_Pixelart_pcd,
    'DiT-PixArt-PCD-CLAY-XL': DiT_XL_Pixelart_clay_pcd,
    'DiT-PixArt-PCD-CLAY-L': DiT_L_Pixelart_clay_pcd,
    'DiT-PixArt-PCD-CLAY-B': DiT_B_Pixelart_clay_pcd,
    'DiT-PixArt-PCD-CLAY-stage2-B': DiT_B_Pixelart_clay_pcd_stage2,
    'DiT-PixArt-PCD-CLAY-stage2-L': DiT_L_Pixelart_clay_pcd_stage2,
    'DiT-PixArt-PCD-CLAY-TandI-L': DiT_L_Pixelart_clay_tandi_pcd,
    'DiT-PixArt-PCD-CLAY-TandI-B': DiT_B_Pixelart_clay_tandi_pcd,
    'DiT-PixArt-PCD-B': DiT_B_Pixelart_pcd,
    # xyz-conditioned KL feature diffusion
    'DiT-PixArt-PCD-cond-diff-B': DiT_B_Pixelart_pcd_cond_diff,
    'DiT-PixArt-PCD-cond-diff-pe-B': DiT_B_Pixelart_pcd_cond_diff_pe,
    'DiT-PixArt-PCD-cond-diff-pe-L': DiT_L_Pixelart_pcd_cond_diff_pe,
    'DiT-PixArt-MV-B/2': DiT_B_Pixelart_MV_2,
    'DiT-PixArt-B/2': DiT_B_Pixelart_2,

    # ! mv version following clay
    'DiT-PixArt-PCD-MV-L': DiT_L_Pixelart_clay_mv_pcd,
    'DiT-PixArt-PCD-MV-I23Dpt-L': DiT_L_Pixelart_clay_mv_i23dpt_pcd,
    'DiT-PixArt-PCD-MV-I23Dpt-L-noI23D': DiT_L_Pixelart_clay_mv_i23dpt_pcd_noi23d,
    'DiT-PixArt-PCD-MV-I23Dpt-L-stage2': DiT_L_Pixelart_clay_mv_i23dpt_pcd_stage2,
}
