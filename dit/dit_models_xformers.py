# https://github.com/facebookresearch/DiT

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
# from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.models.vision_transformer import PatchEmbed, Mlp
from einops import rearrange
from pdb import set_trace as st

# support flash attention and xformer acceleration
from ldm.modules.attention import CrossAttention
from vit.vision_transformer import MemEffAttention as Attention
# import apex
# from apex.normalization import FusedRMSNorm as RMSNorm

from torch.nn import LayerNorm

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except:
    from dit.norm import RMSNorm

# from torch.nn import LayerNorm
# from xformers import triton
# import xformers.triton

if torch.cuda.is_available():
    # from xformers.triton import FusedLayerNorm as LayerNorm # compat issue
    from xformers.components.activations import build_activation, Activation
    from xformers.components.feedforward import fused_mlp
    # from xformers.components.feedforward import mlp

from ldm.modules.attention import MemoryEfficientCrossAttention, JointMemoryEfficientCrossAttention


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def t2i_modulate(x, shift, scale):  # for pix-art arch
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """
    # from torch.nn import LayerNorm

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        # self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size,
                                patch_size * patch_size * out_channels,
                                bias=True)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5)
        self.adaLN_modulation = None
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2,
                                                                         dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32) /
            half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding,
                                            hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0],
                                  device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class ClipProjector(nn.Module):

    def __init__(self, transformer_width, embed_dim, tx_width, *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        '''a CLIP text encoder projector, adapted from CLIP.encode_text
        '''

        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim))
        nn.init.normal_(self.text_projection, std=tx_width**-0.5)

    def forward(self, clip_text_x):
        return clip_text_x @ self.text_projection


def approx_gelu():
    return nn.GELU(approximate="tanh")


class CaptionEmbedder(nn.Module):
    """
    copied from https://github.com/hpcaitech/Open-Sora

    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self,
                 in_channels,
                 hidden_size,
                 act_layer=nn.GELU(approximate="tanh"),
                 token_num=120):
        super().__init__()

        self.y_proj = Mlp(in_features=in_channels,
                          hidden_features=hidden_size,
                          out_features=hidden_size,
                          act_layer=act_layer,
                          drop=0)
        # self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5))
        # self.uncond_prob = uncond_prob

    # def token_drop(self, caption, force_drop_ids=None):
    #     """
    #     Drops labels to enable classifier-free guidance.
    #     """
    #     if force_drop_ids is None:
    #         drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
    #     else:
    #         drop_ids = force_drop_ids == 1
    #     caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
    #     return caption

    def forward(self, caption, **kwargs):
        # if train:
        #     assert caption.shape[2:] == self.y_embedding.shape
        # use_dropout = self.uncond_prob > 0
        # if (train and use_dropout) or (force_drop_ids is not None):
        #     caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self,
                 hidden_size,
                 num_heads,
                 mlp_ratio=4.0,
                 context_dim=None,
                #  enable_rmsnorm=False,
                 norm_type='layernorm',
                 qk_norm=True,
                #  enable_rope=False,
                 **block_kwargs):
        super().__init__()
        # st()
        assert qk_norm
        # nn.LayerNorm
        if norm_type == 'layernorm':
            self.norm1 = LayerNorm(
                hidden_size,
                # affine=False,
                  elementwise_affine=False,
                eps=1e-6)
            self.norm2 = LayerNorm(
                hidden_size,
                # affine=False,
                elementwise_affine=False,
                eps=1e-6)
        else:
            assert norm_type == 'rmsnorm'  # more robust to bf16 training.
            self.norm1 = RMSNorm(hidden_size, eps=1e-5)
            self.norm2 = RMSNorm(hidden_size, eps=1e-5)

        # st()
        self.attn = Attention(hidden_size,
                              num_heads=num_heads,
                              qkv_bias=True,
                              qk_norm=qk_norm,
                              **block_kwargs)
        # mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")

        # self.mlp = Mlp(in_features=hidden_size,
        #                hidden_features=mlp_hidden_dim,
        #                act_layer=approx_gelu,
        #                drop=0)

        self.mlp = fused_mlp.FusedMLP(
            dim_model=hidden_size,
            dropout=0,
            activation=Activation.GeLU,
            hidden_layer_multiplier=int(mlp_ratio),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class TextCondDiTBlock(DiTBlock):
    # https://github.com/hpcaitech/Open-Sora/blob/68b8f60ff0ff4b3a3b63fe1d8cb17d66b7845ef7/opensora/models/stdit/stdit.py#L69
    def __init__(self, hidden_size, num_heads, mlp_ratio=4, **block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio, **block_kwargs)
        self.cross_attn = MemoryEfficientCrossAttention(query_dim=hidden_size,
                                                        heads=num_heads)
        # self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, x, t, context):
        # B, N, C = x.shape

        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
        #    self.scale_shift_table[None] + t.reshape(B,6,-1)).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            t).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa))

        # add text embedder via pre-norm cross attention
        x = x + self.cross_attn(x, context)

        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class PixelArtTextCondDiTBlock(DiTBlock):
    # 1. add shared AdaLN
    # 2. add return pooled vector token (in the outer loop already)
    def __init__(self,
                 hidden_size,
                 num_heads,
                 mlp_ratio=4,
                 context_dim=None,
                qk_norm=True,
                 **block_kwargs):
        super().__init__(hidden_size,
                         num_heads,
                         mlp_ratio,
                         norm_type='rmsnorm',
                         qk_norm=qk_norm,
                         **block_kwargs)
        # super().__init__(hidden_size, num_heads, mlp_ratio, norm_type='layernorm', **block_kwargs)
        self.cross_attn = MemoryEfficientCrossAttention(
            query_dim=hidden_size, context_dim=context_dim, heads=num_heads, qk_norm=qk_norm) # ! force QK_Norm
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5)
        self.adaLN_modulation = None
        self.attention_y_norm = RMSNorm(
            context_dim, eps=1e-5
        )  # https://github.com/Alpha-VLLM/Lumina-T2X/blob/0c8dd6a07a3b7c18da3d91f37b1e00e7ae661293/lumina_t2i/models/model.py#L570C9-L570C61

        self.prenorm_ca_text = RMSNorm(hidden_size, eps=1e-5)

    def forward(self, x, t, context):
        B, N, C = x.shape

        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
        #    self.scale_shift_table[None] + t.reshape(B,6,-1)).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
        #     t).chunk(6, dim=1)

        x = x + gate_msa * self.attn(
            t2i_modulate(self.norm1(x), shift_msa, scale_msa))

        # add text embedder via cross attention
        x = x + self.cross_attn(self.prenorm_ca_text(x), self.attention_y_norm(context))

        x = x + gate_mlp * self.mlp(
            t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class MMTextCondDiTBlock(DiTBlock):
    # follow SD-3
    def __init__(self, hidden_size, num_heads, mlp_ratio=4, **block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio, **block_kwargs)
        self.cross_attn = MemoryEfficientCrossAttention(query_dim=hidden_size,
                                                        heads=num_heads)
        # self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        self.adaLN_modulation_img = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.mlp_img = fused_mlp.FusedMLP(
            dim_model=hidden_size,
            dropout=0,
            activation=Activation.GeLU,
            hidden_layer_multiplier=int(mlp_ratio),
        )

    def forward(self, x, t, context):
        # B, N, C = x.shape

        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
        #    self.scale_shift_table[None] + t.reshape(B,6,-1)).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            t).chunk(6, dim=1)

        # TODO, batch inference with above
        shift_msa_img, scale_msa_img, gate_msa_img, shift_mlp_img, scale_mlp_img, gate_mlp_img = self.adaLN_modulation_img(
            t).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa))

        # add text embedder via cross attention
        x = x + self.cross_attn(x, context)

        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


# for image condition


class ImageCondDiTBlock(DiTBlock):
    # follow EMU and SVD, concat + cross attention. Also adopted by concurrent work Direct3D.
    def __init__(self,
                 hidden_size,
                 num_heads,
                 context_dim,
                 mlp_ratio=4,
                #  enable_rmsnorm=False,
                 qk_norm=True,
                 enable_rope=False,
                 **block_kwargs):
        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         context_dim=context_dim,
                        #  enable_rmsnorm=enable_rmsnorm,
                         qk_norm=qk_norm,
                         enable_rope=enable_rope,
                         **block_kwargs)
        assert qk_norm
        self.cross_attn = MemoryEfficientCrossAttention(
            query_dim=hidden_size,
            context_dim=context_dim, # ! mv-cond
            # context_dim=1280,  # clip vit-G, adopted by SVD.
            # context_dim=1024,  # clip vit-L
            heads=num_heads,
            # enable_rmsnorm=enable_rmsnorm, 
            qk_norm=qk_norm, 
            enable_rope=enable_rope)
        assert qk_norm
        # self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        self.attention_y_norm = RMSNorm(
            1024, eps=1e-5
        )  # https://github.com/Alpha-VLLM/Lumina-T2X/blob/0c8dd6a07a3b7c18da3d91f37b1e00e7ae661293/lumina_t2i/models/model.py#L570C9-L570C61

    def forward(self, x, t, dino_spatial_token, clip_spatial_token):
        # B, N, C = x.shape
        # assert isinstance(context, dict)
        # assert isinstance(context, dict) # clip + dino.

        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
        #    self.scale_shift_table[None] + t.reshape(B,6,-1)).chunk(6, dim=1)

        # TODO t is t + [clip_cls] here. update in base class.

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            t).chunk(6, dim=1)

        post_modulate_selfattn_feat = torch.cat([
            modulate(self.norm1(x), shift_msa, scale_msa), dino_spatial_token
        ],
                                                dim=1)  # concat in L dim

        x = x + gate_msa.unsqueeze(1) * self.attn(
            post_modulate_selfattn_feat
        )[:, :x.shape[1]]  # remove dino-feat to maintain unchanged dimension.

        # add clip_i spatial embedder via cross attention
        x = x + self.cross_attn(x, self.attention_y_norm(clip_spatial_token))

        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class ImageCondDiTBlockPixelArt(ImageCondDiTBlock):
    # follow EMU and SVD, concat + cross attention. Also adopted by concurrent work Direct3D.
    def __init__(self,
                 hidden_size,
                 num_heads,
                 context_dim,
                 mlp_ratio=4,
                #  enable_rmsnorm=False,
                 qk_norm=False,
                    norm_type='rmsnorm',
                 **block_kwargs):
        # super().__init__(hidden_size, num_heads, mlp_ratio, enable_rmsnorm=True, **block_kwargs)
        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         context_dim=context_dim,
                        #  enable_rmsnorm=False,
                         qk_norm=True, # otherwise AMP fail
                         norm_type=norm_type,
                         **block_kwargs)
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5)
        self.adaLN_modulation = None  # single-adaLN
        # self.attention_y_norm = RMSNorm(
        #     1024, eps=1e-5
        # )  # https://github.com/Alpha-VLLM/Lumina-T2X/blob/0c8dd6a07a3b7c18da3d91f37b1e00e7ae661293/lumina_t2i/models/model.py#L570C9-L570C61

    def forward(self, x, t, dino_spatial_token, clip_spatial_token):
        B, N, C = x.shape
        # assert isinstance(context, dict)
        # assert isinstance(context, dict) # clip + dino.

        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
        #    self.scale_shift_table[None] + t.reshape(B,6,-1)).chunk(6, dim=1)

        # TODO t is t + [clip_cls] here. update in base class.

        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
        # t).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # st()

        post_modulate_selfattn_feat = torch.cat([
            t2i_modulate(self.norm1(x), shift_msa, scale_msa),
            dino_spatial_token
        ],
                                                dim=1)  # concat in L dim

        # x = x + gate_msa.unsqueeze(1) * self.attn(
        x = x + gate_msa * self.attn(post_modulate_selfattn_feat)[:, :x.shape[
            1]]  # remove dino-feat to maintain unchanged dimension.

        # add clip_i spatial embedder via cross attention
        x = x + self.cross_attn(x, clip_spatial_token) # attention_y_norm not required, since x_norm_patchtokens?
        # x = x + self.cross_attn(x, self.attention_y_norm(clip_spatial_token)) # attention_y_norm not required, since x_norm_patchtokens?

        x = x + gate_mlp * self.mlp(
            t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

class ImageCondDiTBlockPixelArtNoclip(ImageCondDiTBlock):
    # follow EMU and SVD, concat + cross attention. Also adopted by concurrent work Direct3D.
    def __init__(self,
                 hidden_size,
                 num_heads,
                 context_dim,
                 mlp_ratio=4,
                #  enable_rmsnorm=False,
                 qk_norm=False,
                 **block_kwargs):
        # super().__init__(hidden_size, num_heads, mlp_ratio, enable_rmsnorm=True, **block_kwargs)
        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         context_dim=context_dim,
                        #  enable_rmsnorm=False,
                         qk_norm=True, # otherwise AMP fail
                         **block_kwargs)
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5)
        self.adaLN_modulation = None  # single-adaLN
        # self.attention_y_norm = RMSNorm(
        #     1024, eps=1e-5
        # )  # https://github.com/Alpha-VLLM/Lumina-T2X/blob/0c8dd6a07a3b7c18da3d91f37b1e00e7ae661293/lumina_t2i/models/model.py#L570C9-L570C61

    # def _forward_cross_attention(self, x, tokens):
    #     x = x + self.cross_attn(x, tokens) # attention_y_norm not required, since x_norm_patchtokens?
    #     return x

    # def forward(self, x, t, dino_spatial_token, clip_spatial_token):
    def forward(self, x, t, dino_spatial_token):
        B, N, C = x.shape
        # assert isinstance(context, dict)
        # assert isinstance(context, dict) # clip + dino.

        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
        #    self.scale_shift_table[None] + t.reshape(B,6,-1)).chunk(6, dim=1)

        # TODO t is t + [clip_cls] here. update in base class.

        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
        # t).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # st()

        post_modulate_selfattn_feat = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

            # t2i_modulate(self.norm1(x), shift_msa, scale_msa),
            # dino_spatial_token
        # ],
                                                # dim=1)  # concat in L dim

        # x = x + gate_msa.unsqueeze(1) * self.attn(
        x = x + gate_msa * self.attn(post_modulate_selfattn_feat) # remove dino-feat to maintain unchanged dimension.

        # add clip_i spatial embedder via cross attention
        x = x + self.cross_attn(x, dino_spatial_token) # attention_y_norm not required, since x_norm_patchtokens?
        # x = self._forward_cross_attention(x, dino_spatial_token)
        # x = x + self.cross_attn(x, self.attention_y_norm(clip_spatial_token)) # attention_y_norm not required, since x_norm_patchtokens?

        x = x + gate_mlp * self.mlp(
            t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class ImageCondDiTBlockPixelArtRMSNorm(ImageCondDiTBlockPixelArt):
    # follow EMU and SVD, concat + cross attention. Also adopted by concurrent work Direct3D.
    def __init__(self,
                 hidden_size,
                 num_heads,
                 context_dim,
                 mlp_ratio=4,
                # enable_rmsnorm=False,
                norm_type='rmsnorm',
                 **block_kwargs):
        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         context_dim=context_dim,
                        #  enable_rmsnorm=False,
                         norm_type=norm_type,
                         **block_kwargs)


class ImageCondDiTBlockPixelArtRMSNormNoClip(ImageCondDiTBlockPixelArtNoclip):
    # follow EMU and SVD, concat + cross attention. Also adopted by concurrent work Direct3D.
    def __init__(self,
                 hidden_size,
                 num_heads,
                 context_dim,
                 mlp_ratio=4,
                 **block_kwargs):
        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         context_dim=context_dim,
                         enable_rmsnorm=False,
                         norm_type='rmsnorm',
                         **block_kwargs)

# support more conditions
class ImageCondDiTBlockPixelArtRMSNormClay(ImageCondDiTBlockPixelArtRMSNorm):
    # follow EMU and SVD, concat + cross attention. Also adopted by concurrent work Direct3D.
    def __init__(self,
                 hidden_size,
                 num_heads,
                 context_dim,
                 mlp_ratio=4,
                 qk_norm=True,
                 norm_type='rmsnorm',
                 enable_text_ca=False,
                 **block_kwargs):
        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         context_dim=context_dim,
                        #  enable_rmsnorm=False,
                         norm_type=norm_type,
                         qk_norm=qk_norm,
                         **block_kwargs)

        # ca for text
        self.cross_attn_dino = MemoryEfficientCrossAttention(
            query_dim=hidden_size,
            context_dim=context_dim,
            heads=num_heads,
            # enable_rmsnorm=enable_rmsnorm, 
            qk_norm=qk_norm, 
            dim_head=hidden_size//num_heads)
        # if enable_text_ca:
        #     self.cross_attn_text = MemoryEfficientCrossAttention(
        #         query_dim=hidden_size,
        #         context_dim=context_dim,
        #         heads=num_heads,
        #         # enable_rmsnorm=enable_rmsnorm, 
        #         qk_norm=qk_norm)
        # else:
        self.cross_attn_text = lambda x, context: torch.zeros_like(x) # always return 0
        
    # def _forward_cross_attention(self, x, tokens):
    #     x = x + self.cross_attn(x, tokens) # attention_y_norm not required, since x_norm_patchtokens?
    #     return x
        
    def forward(self, x, t, dino_spatial_token, clip_spatial_token, clip_caption_token):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

        post_modulate_selfattn_feat = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        x = x + gate_msa * self.attn(post_modulate_selfattn_feat) # remove dino-feat to maintain unchanged dimension.

        # can be accelerated by parallel attention
        x = x + self.cross_attn(x, clip_spatial_token) + self.cross_attn_dino(x, dino_spatial_token) + self.cross_attn_text(x, clip_caption_token)

        x = x + gate_mlp * self.mlp(
            t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

# lrm-like, CA-SA-FFN
class ImageCondDiTBlockPixelArtRMSNormClayLRM(ImageCondDiTBlockPixelArtRMSNorm):
    # follow EMU and SVD, concat + cross attention. Also adopted by concurrent work Direct3D.
    def __init__(self,
                 hidden_size,
                 num_heads,
                 context_dim,
                 mlp_ratio=4,
                 qk_norm=True,
                 norm_type='rmsnorm',
                 enable_text_ca=False,
                 enable_rope=False,
                 **block_kwargs):
        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         context_dim=context_dim,
                        #  enable_rmsnorm=False,
                         norm_type=norm_type,
                         qk_norm=qk_norm,
                         enable_rope=enable_rope,
                         **block_kwargs)

        # ca for text
        self.cross_attn_dino = MemoryEfficientCrossAttention(
            query_dim=hidden_size,
            context_dim=context_dim,
            heads=num_heads,
            # enable_rmsnorm=enable_rmsnorm, 
            qk_norm=qk_norm,
            dim_head=hidden_size//num_heads, 
            enable_rope=enable_rope)
        self.prenorm_ca_dino = RMSNorm(hidden_size, eps=1e-5)
        # if enable_text_ca:
        #     self.cross_attn_text = MemoryEfficientCrossAttention(
        #         query_dim=hidden_size,
        #         context_dim=context_dim,
        #         heads=num_heads,
        #         # enable_rmsnorm=enable_rmsnorm, 
        #         qk_norm=qk_norm)
        # else:
        self.cross_attn_text = lambda x, context, freqs_cis: torch.zeros_like(x) # always return 0
        self.prenorm_ca_text = nn.Identity()
        del self.cross_attn # not for text now
        
    # def _forward_cross_attention(self, x, tokens):
    #     x = x + self.cross_attn(x, tokens) # attention_y_norm not required, since x_norm_patchtokens?
    #     return x
        
    def forward(self, x, t, dino_spatial_token, clip_caption_token=None, freqs_cis=None, **kwargs):
        B, N, C = x.shape

        # ! prepare adaLN
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

        # post_modulate_selfattn_feat = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        # ! CA for spaital info (img, mv-img .etc.)
        x = x + self.cross_attn_dino(self.prenorm_ca_dino(x), dino_spatial_token, freqs_cis=freqs_cis)  # ! pre-norm & residual, as in CLAY

        # ! SA
        x = x + gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis=freqs_cis) # remove dino-feat to maintain unchanged dimension.

        # # ! CA for text
        x = x + self.cross_attn_text(self.prenorm_ca_text(x), clip_caption_token, freqs_cis=freqs_cis)

        # ! FFN
        x = x + gate_mlp * self.mlp(
            t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

# support more conditions
class ImageCondDiTBlockPixelArtRMSNormClayText(PixelArtTextCondDiTBlock):
    # follow EMU and SVD, concat + cross attention. Also adopted by concurrent work Direct3D.
    def __init__(self,
                 hidden_size,
                 num_heads,
                 context_dim,
                 mlp_ratio=4,
                 qk_norm=True,
                #  norm_type='rmsnorm',
                 enable_text_ca=False,
                 **block_kwargs):
        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         context_dim=context_dim,
                        #  enable_rmsnorm=False,
                        #  norm_type=norm_type,
                         qk_norm=qk_norm,
                         **block_kwargs)

        # dino_hidden_size = 768
        dino_hidden_size = 384 # a dit-s as in clay
        self.cross_attn_dino = MemoryEfficientCrossAttention(
            query_dim=hidden_size,
            context_dim=dino_hidden_size,
            heads=num_heads,
            # enable_rmsnorm=enable_rmsnorm, 
            qk_norm=qk_norm)
        self.prenorm_ca_dino = RMSNorm(hidden_size, eps=1e-5)


    def forward(self, x, t, dino_spatial_token, clip_caption_token=None, **kwargs):
        B, N, C = x.shape

        # ! prepare adaLN
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

        # post_modulate_selfattn_feat = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        # ! SA
        x = x + gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)) # remove dino-feat to maintain unchanged dimension.

        # # ! CA for text
        # x = x + self.cross_attn(self.prenorm_ca_text(x), self.attention_y_norm(clip_caption_token))

        # ! add more conditions through residuals.
        x = x + self.cross_attn(self.prenorm_ca_text(x), self.attention_y_norm(clip_caption_token)) + self.cross_attn_dino(self.prenorm_ca_dino(x), dino_spatial_token)

        # # ! CA for spaital info (img, mv-img .etc.)
        # x = x + self.cross_attn_dino(self.prenorm_ca_dino(x), dino_spatial_token)  # ! pre-norm & residual, as in CLAY

        # ! FFN
        x = x + gate_mlp * self.mlp(
            t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


# support more conditions
class ImageCondDiTBlockPixelArtRMSNormClayMV(ImageCondDiTBlockPixelArtRMSNormClayLRM):
    # follow EMU and SVD, concat + cross attention. Also adopted by concurrent work Direct3D.
    def __init__(self,
                 hidden_size,
                 num_heads,
                 context_dim,
                 mlp_ratio=4,
                 qk_norm=True,
                #  norm_type='rmsnorm',
                 enable_text_ca=False,
                 **block_kwargs):
        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         context_dim=context_dim,
                        #  enable_rmsnorm=False,
                        #  norm_type=norm_type,
                         qk_norm=qk_norm,
                         **block_kwargs)

        # dino_hidden_size = 768
        dino_hidden_size = 384 # a dit-s as in clay
        self.cross_attn_dino_mv = MemoryEfficientCrossAttention(
            query_dim=hidden_size,
            context_dim=dino_hidden_size,
            heads=num_heads,
            # enable_rmsnorm=enable_rmsnorm, 
            qk_norm=qk_norm)
        self.prenorm_ca_dino_mv = RMSNorm(hidden_size, eps=1e-5)
        # self.mv_alpha = nn.Parameter(torch.zeros(1)-2.5) # per-layer blending into the original parameters, like zero-conv.
        # self.mv_alpha = nn.Parameter(torch.zeros(1)+2.5) # per-layer blending into the original parameters, like zero-conv.
        self.mv_alpha = nn.Parameter(torch.zeros(1)) # per-layer blending into the original parameters, like zero-conv.


    def forward(self, x, t, dino_spatial_token, dino_mv_spatial_token=None, **kwargs):
        B, N, C = x.shape

        # ! prepare adaLN
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

        # post_modulate_selfattn_feat = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        # ! CA for spaital info (img, mv-img .etc.)
        x = x + self.cross_attn_dino(self.prenorm_ca_dino(x), dino_spatial_token) + self.cross_attn_dino_mv(self.prenorm_ca_dino_mv(x), dino_mv_spatial_token) * (torch.nn.functional.tanh(self.mv_alpha))

        # ! SA
        x = x + gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)) # remove dino-feat to maintain unchanged dimension.

        # # ! CA for text
        # x = x + self.cross_attn_text(self.prenorm_ca_text(x), clip_caption_token)

        # ! FFN
        x = x + gate_mlp * self.mlp(
            t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

# no i23d ca
class ImageCondDiTBlockPixelArtRMSNormClayMV_noi23d(ImageCondDiTBlockPixelArtRMSNormClayLRM):
    # follow EMU and SVD, concat + cross attention. Also adopted by concurrent work Direct3D.
    def __init__(self,
                 hidden_size,
                 num_heads,
                 context_dim,
                 mlp_ratio=4,
                 qk_norm=True,
                #  norm_type='rmsnorm',
                 enable_text_ca=False,
                 **block_kwargs):
        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         context_dim=context_dim,
                        #  enable_rmsnorm=False,
                        #  norm_type=norm_type,
                         qk_norm=qk_norm,
                         **block_kwargs)

        # del self.cross_attn_dino, self.prenorm_ca_dino
        del self.cross_attn_dino
        self.prenorm_ca_dino_mv = self.prenorm_ca_dino
        # dino_hidden_size = 768
        dino_hidden_size = 384 # a dit-s as in clay
        self.cross_attn_dino_mv = MemoryEfficientCrossAttention(
            query_dim=hidden_size,
            context_dim=dino_hidden_size,
            heads=num_heads,
            # enable_rmsnorm=enable_rmsnorm, 
            qk_norm=qk_norm)
        # self.prenorm_ca_dino_mv = RMSNorm(hidden_size, eps=1e-5)
        # self.mv_alpha = nn.Parameter(torch.zeros(1)-2.5) # per-layer blending into the original parameters, like zero-conv.
        # self.mv_alpha = nn.Parameter(torch.zeros(1)+2.5) # per-layer blending into the original parameters, like zero-conv.
        # self.mv_alpha = nn.Parameter(torch.zeros(1)) # per-layer blending into the original parameters, like zero-conv.


    # def forward(self, x, t, dino_spatial_token, dino_mv_spatial_token=None, **kwargs):
    def forward(self, x, t, dino_mv_spatial_token, **kwargs):
        B, N, C = x.shape

        # ! prepare adaLN
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

        # post_modulate_selfattn_feat = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        # ! CA for spaital info (img, mv-img .etc.)
        # x = x + self.cross_attn_dino(self.prenorm_ca_dino(x), dino_spatial_token) + self.cross_attn_dino_mv(self.prenorm_ca_dino_mv(x), dino_mv_spatial_token) * (torch.nn.functional.tanh(self.mv_alpha))
        x = x + self.cross_attn_dino_mv(self.prenorm_ca_dino_mv(x), dino_mv_spatial_token)

        # ! SA
        x = x + gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)) # remove dino-feat to maintain unchanged dimension.

        # # ! CA for text
        # x = x + self.cross_attn_text(self.prenorm_ca_text(x), clip_caption_token)

        # ! FFN
        x = x + gate_mlp * self.mlp(
            t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x




class DiTBlockRollOut(DiTBlock):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4, **block_kwargs):
        super().__init__(hidden_size * 3, num_heads, mlp_ratio, **block_kwargs)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
# 

class FinalLayer(nn.Module):
    """
    The final layer of DiT, basically the decoder_pred in MAE with adaLN.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        # self.norm_final = nn.LayerNorm(hidden_size,
        self.norm_final = LayerNorm(
            hidden_size,
               elementwise_affine=False,#  apex or nn kernel
            # affine=False,
            eps=1e-6)
        self.linear = nn.Linear(hidden_size,
                                patch_size * patch_size * out_channels,
                                bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        roll_out=False,
        vit_blk=DiTBlock,
        # vit_blk=TextCondDiTBlock,
        final_layer_blk=FinalLayer,
        enable_rope=False,
    ):
        super().__init__()
        self.plane_n = 3
        # st()

        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.embed_dim = hidden_size

        self.x_embedder = PatchEmbed(input_size,
                                     patch_size,
                                     in_channels,
                                     hidden_size,
                                     bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if num_classes > 0:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size,
                                            class_dropout_prob)
        else:
            self.y_embedder = None

        if context_dim is not None:
            self.clip_text_proj = CaptionEmbedder(context_dim,
                                                  hidden_size,
                                                  act_layer=approx_gelu)

        else:
            self.clip_text_proj = None

        self.roll_out = roll_out

        num_patches = self.x_embedder.num_patches  # 14*14*3
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size),
                                      requires_grad=False)

        # if not self.roll_out:
        self.blocks = nn.ModuleList([
            vit_blk(hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    context_dim=context_dim, 
                    enable_rope=enable_rope) for _ in range(depth)
        ])
        # else:
        #     self.blocks = nn.ModuleList([
        #         DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) if idx % 2 == 0 else
        #         DiTBlockRollOut(hidden_size, num_heads, mlp_ratio=mlp_ratio)
        #         for idx in range(depth)
        #     ])

        self.final_layer = final_layer_blk(hidden_size, patch_size,
                                           self.out_channels)
        self.initialize_weights()

        # self.mixed_prediction = mixed_prediction  # This enables mixed prediction
        # if self.mixed_prediction:
        #     if self.roll_out:
        #         logit_ch = in_channels * 3
        #     else:
        #         logit_ch = in_channels
        #     init = mixing_logit_init * torch.ones(
        #         size=[1, logit_ch, 1, 1])  # hard coded for now
        #     self.mixing_logit = torch.nn.Parameter(init, requires_grad=True)

    # def len(self):
    #     return len(self.blocks)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        # st()
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            if block.adaLN_modulation is not None:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if self.final_layer.adaLN_modulation is not None:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        # p = self.x_embedder.patch_size[0]
        p = self.patch_size
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    # def forward(self, x, t, y=None, get_attr=''):
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

        if get_attr != '':  # not breaking the forward hooks
            return getattr(self, get_attr)

        t = self.t_embedder(timesteps)  # (N, D)

        if self.roll_out:  # !
            x = rearrange(x, 'b (c n) h w->(b n) c h w', n=3)

        x = self.x_embedder(
            x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        if self.roll_out:  # ! roll-out in the L dim, not B dim. add condition to all tokens.
            x = rearrange(x, '(b n) l c ->b (n l) c', n=3)

        # if self.y_embedder is not None:
        #     assert y is not None
        #     y = self.y_embedder(y, self.training)  # (N, D)
        #     c = t + y  # (N, D)

        assert context is not None

        # assert context.ndim == 2
        if isinstance(context, dict):
            context = context['crossattn']  # sgm conditioner compat
        context = self.clip_text_proj(context)

        # c = t + context
        # else:
        # c = t  # BS 1024

        for blk_idx, block in enumerate(self.blocks):
            # if self.roll_out:
            #     if blk_idx % 2 == 0: # with-in plane self attention
            #         x = rearrange(x, 'b (n l) c -> b l (n c) ', n=3)
            #         x = block(x, torch.repeat_interleave(c, 3, 0))  # (N, T, D)
            #     else: # global attention
            #         # x = rearrange(x, '(b n) l c -> b (n l) c ', n=3)
            #         x = rearrange(x, 'b l (n c) -> b (n l) c ', n=3)
            #         x = block(x, c)  # (N, T, D)
            # else:
            # st()
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
        x = x.to(torch.float32)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[:len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward_with_cfg_unconditional(self, x, t, y=None, cfg_scale=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # half = x[:len(x) // 2]
        # combined = torch.cat([half, half], dim=0)
        combined = x
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        # cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        # half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        # eps = torch.cat([half_eps, half_eps], dim=0)
        # return torch.cat([eps, rest], dim=1)
        # st()
        return model_out


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim,
                            grid_size,
                            cls_token=False,
                            extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, tuple):
        grid_size_h, grid_size_w = grid_size
        grid_h = np.arange(grid_size_h, dtype=np.float32)
        grid_w = np.arange(grid_size_w, dtype=np.float32)
    else:
        grid_size_h = grid_size_w = grid_size
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(depth=28,
               hidden_size=1152,
               patch_size=2,
               num_heads=16,
               **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28,
               hidden_size=1152,
               patch_size=4,
               num_heads=16,
               **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28,
               hidden_size=1152,
               patch_size=8,
               num_heads=16,
               **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24,
               hidden_size=1024,
               patch_size=2,
               num_heads=16,
               **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24,
               hidden_size=1024,
               patch_size=4,
               num_heads=16,
               **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24,
               hidden_size=1024,
               patch_size=8,
               num_heads=16,
               **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_B_16(**kwargs):  # ours cfg
    return DiT(depth=12,
               hidden_size=768,
               patch_size=16,
               num_heads=12,
               **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,
    'DiT-XL/4': DiT_XL_4,
    'DiT-XL/8': DiT_XL_8,
    'DiT-L/2': DiT_L_2,
    'DiT-L/4': DiT_L_4,
    'DiT-L/8': DiT_L_8,
    'DiT-B/2': DiT_B_2,
    'DiT-B/4': DiT_B_4,
    'DiT-B/8': DiT_B_8,
    'DiT-B/16': DiT_B_16,
    'DiT-S/2': DiT_S_2,
    'DiT-S/4': DiT_S_4,
    'DiT-S/8': DiT_S_8,
}
