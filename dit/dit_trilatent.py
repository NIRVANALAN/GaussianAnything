import torch.nn as nn
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from pdb import set_trace as st
from timm.models.vision_transformer import Mlp

from ldm.modules.attention import MemoryEfficientCrossAttention
from .dit_models_xformers import DiT, get_2d_sincos_pos_embed, DiTBlock, FinalLayer, t2i_modulate, PixelArtTextCondDiTBlock, T2IFinalLayer, approx_gelu

from torch.nn import LayerNorm
from vit.vit_triplane import XYZPosEmbed

class DiT_TriLatent(DiT):
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
        roll_out=False,
        vit_blk=DiTBlock,
        final_layer_blk=FinalLayer,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                         mixed_prediction, context_dim, roll_out, vit_blk,
                         final_layer_blk)

        assert self.roll_out

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
        assert context is not None

        t = self.t_embedder(timesteps)  # (N, D)

        # if self.roll_out:  # !
        x = rearrange(x, 'b (c n) h w->(b n) c h w',
                      n=3)  # downsample with same conv
        x = self.x_embedder(x)  # (b n) c h/f w/f

        x = rearrange(x, '(b n) l c -> b (n l) c', n=3)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # if self.roll_out:  # ! roll-out in the L dim, not B dim. add condition to all tokens.
        # x = rearrange(x, '(b n) l c ->b (n l) c', n=3)

        # assert context.ndim == 2
        if isinstance(context, dict):
            context = context['crossattn']  # sgm conditioner compat

        context = self.clip_text_proj(context)

        # c = t + context
        # else:
        # c = t  # BS 1024

        for blk_idx, block in enumerate(self.blocks):
            # if self.roll_out:
            if False:
                if blk_idx % 2 == 0:  # with-in plane self attention
                    x = rearrange(x, 'b (n l) c -> (b n) l c', n=3)
                    x = block(x, repeat(t, 'b c -> (b n) c ', n=3), # TODO, calculate once
                              repeat(context, 'b l c -> (b n) l c ', n=3))  # (N, T, D)

                else:  # global attention
                    x = rearrange(x, '(b n) l c -> b (n l) c ', n=self.plane_n)
                    x = block(x, t, context)  # (N, T, D)
            else:
                x = block(x, t, context)  # (N, T, D)

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
        # st()

        return x


class DiT_TriLatent_PixelArt(DiT_TriLatent):
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
        roll_out=False,
        vit_blk=DiTBlock,
        final_layer_blk=FinalLayer,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                         mixed_prediction, context_dim, roll_out, vit_blk,
                         final_layer_blk)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        del self.clip_text_proj
        self.cap_embedder = nn.Sequential( # TODO, init with zero here.
            LayerNorm(context_dim),
            nn.Linear(
                context_dim,
                hidden_size,
            ),
        )
        nn.init.constant_(self.cap_embedder[-1].weight, 0)
        nn.init.constant_(self.cap_embedder[-1].bias, 0)


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

        clip_cls_token = self.cap_embedder(context['vector']) # pooled
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
        if isinstance(context, dict):
            context = context['crossattn']  # sgm conditioner compat

        # context = self.clip_text_proj(context) # ! with rmsnorm here for 

        # c = t + context
        # else:
        # c = t  # BS 1024

        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t0, context)  # (N, T, D)

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
        # st()

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


# PCD, general single-stage model.
class DiT_PCD_PixelArt(DiT_TriLatent_PixelArt):
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
        roll_out=False,
        vit_blk=PixelArtTextCondDiTBlock,
        final_layer_blk=FinalLayer,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                         mixed_prediction, context_dim, roll_out, vit_blk,
                         final_layer_blk)
        # an MLP to transform the input 19-dim feature to high-dim.
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
        assert context is not None

        clip_cls_token = self.cap_embedder(context['caption_vector']) # pooled
        t = self.t_embedder(timesteps) + clip_cls_token  # (N, D)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        x = self.x_embedder(x)

        # assert context.ndim == 2
        if isinstance(context, dict):
            context = context['caption_crossattn']  # sgm conditioner compat

        # loop dit block
        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t0, context)  # (N, T, D)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        # cast to float32 for better accuracy
        x = x.to(torch.float32).contiguous()

        return x

# ! two-stage version, the second-stage here, for text pretraining.
class DiT_PCD_PixelArt_tofeat(DiT_PCD_PixelArt):
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
        roll_out=False,
        vit_blk=DiTBlock,
        final_layer_blk=FinalLayer,
        use_pe_cond=True,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                         mixed_prediction, context_dim, roll_out, PixelArtTextCondDiTBlock,
                         final_layer_blk)

        self.use_pe_cond = use_pe_cond
        if use_pe_cond:
            self.xyz_pos_embed = XYZPosEmbed(hidden_size)
        else:
            self.x_embedder = Mlp(in_features=in_channels+3,
                            hidden_features=hidden_size,
                            out_features=hidden_size,
                            act_layer=approx_gelu,
                            drop=0)


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

        clip_cls_token = self.cap_embedder(context['caption_vector']) # pooled
        caption_crossattn, fps_xyz = context['caption_crossattn'], context['fps-xyz']

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
            x = block(x, t0, caption_crossattn)

        # add a norm layer here, as in point-e
        # x = self.ln_post(x)

        x = self.final_layer(x, t) # no loss on the xyz side 

        x = x.to(torch.float32).contiguous()

        return x



#################################################################################
#                                   DiT_TriLatent Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT_TriLatent(depth=28,
                         hidden_size=1152,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)


def DiT_L_2(**kwargs):
    return DiT_TriLatent(depth=24,
                         hidden_size=1024,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)


def DiT_B_2(**kwargs):
    return DiT_TriLatent(depth=12,
                         hidden_size=768,
                         patch_size=2,
                         num_heads=12,
                         **kwargs)


def DiT_B_1(**kwargs):
    return DiT_TriLatent(depth=12,
                         hidden_size=768,
                         patch_size=1,
                         num_heads=12,
                         **kwargs)


def DiT_B_Pixelart_2(**kwargs):
    return DiT_TriLatent_PixelArt(depth=12,
                         hidden_size=768,
                         patch_size=2,
                         num_heads=12,
                        #  vit_blk=PixelArtTextCondDiTBlock,
                         final_layer_blk=T2IFinalLayer,
                         **kwargs)

def DiT_L_Pixelart_2(**kwargs):
    return DiT_TriLatent_PixelArt(depth=24,
                         hidden_size=1024,
                         patch_size=2,
                         num_heads=16,
                        #  vit_blk=PixelArtTextCondDiTBlock,
                         final_layer_blk=T2IFinalLayer,
                         **kwargs)


# PCD-DiT
def DiT_PCD_B(**kwargs):

    return DiT_PCD_PixelArt(depth=12,
                         hidden_size=768,
                         patch_size=1,
                         num_heads=12,
                         **kwargs)

def DiT_PCD_L(**kwargs):

    return DiT_PCD_PixelArt(depth=24,
                         hidden_size=1024,
                         patch_size=1,
                         num_heads=16,
                         **kwargs)

def DiT_PCD_B_tofeat(**kwargs):

    return DiT_PCD_PixelArt_tofeat(depth=12,
                         hidden_size=768,
                         patch_size=1,
                         num_heads=12,
                         **kwargs)

def DiT_PCD_L_tofeat(**kwargs):

    return DiT_PCD_PixelArt_tofeat(depth=24,
                         hidden_size=1024,
                         patch_size=1,
                         num_heads=16,
                         **kwargs)

def DiT_PCD_XL_tofeat(**kwargs):

    return DiT_PCD_PixelArt_tofeat(depth=28,
                         hidden_size=1152,
                         patch_size=1,
                         num_heads=16,
                         **kwargs)




DiT_models = {
    'DiT-XL/2': DiT_XL_2,
    'DiT-L/2': DiT_L_2,
    'DiT-PixelArt-L/2': DiT_L_Pixelart_2,
    'DiT-PixelArt-B/2': DiT_B_Pixelart_2,
    'DiT-B/2': DiT_B_2,
    'DiT-B/1': DiT_B_1,
    'DiT-PCD-B': DiT_PCD_B,
    'DiT-PCD-L': DiT_PCD_L,
    'DiT-PCD-B-stage2-xyz2feat': DiT_PCD_B_tofeat,
    'DiT-PCD-L-stage2-xyz2feat': DiT_PCD_L_tofeat,
    'DiT-PCD-XL-stage2-xyz2feat': DiT_PCD_XL_tofeat,
    # 'DiT-PCD-L-stage1-text': DiT_PCD_L_tofeat,
}
