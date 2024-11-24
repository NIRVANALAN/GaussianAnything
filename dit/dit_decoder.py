import torch
import torch.nn as nn
import numpy as np
import math

from einops import rearrange
from pdb import set_trace as st

# from .dit_models import DiT, DiTBlock, DiT_models, get_2d_sincos_pos_embed, modulate, FinalLayer

from .dit_models_xformers import DiT, DiTBlock, DiT_models, get_2d_sincos_pos_embed, modulate, FinalLayer
# from .dit_models import DiT, DiTBlock, DiT_models, get_2d_sincos_pos_embed, modulate, FinalLayer


def modulate2(x, shift, scale):
    return x * (1 + scale) + shift


class DiTBlock2(DiTBlock):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4, **block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio, **block_kwargs)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=-1)
        # st()
        x = x + gate_msa * self.attn(
            modulate2(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(
            modulate2(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer2(FinalLayer):
    """
    The final layer of DiT, basically the decoder_pred in MAE with adaLN.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__(hidden_size, patch_size, out_channels)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate2(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT2(DiT):
    # a conditional ViT
    def __init__(self,
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
                 plane_n=3,
                 return_all_layers=False,
                 in_plane_attention=True,
                 vit_blk=...):
        super().__init__(input_size,
                         patch_size,
                         in_channels,
                         hidden_size,
                         depth,
                         num_heads,
                         mlp_ratio,
                         class_dropout_prob,
                         num_classes,
                         learn_sigma,
                         mixing_logit_init,
                         mixed_prediction,
                         context_dim,
                         roll_out,
                         vit_blk=DiTBlock2,
                         final_layer_blk=FinalLayer2)
        # st()

        # no t and x embedder
        del self.x_embedder
        del self.t_embedder
        del self.final_layer
        torch.cuda.empty_cache()
        self.clip_text_proj = None
        self.plane_n = plane_n
        self.return_all_layers = return_all_layers
        self.in_plane_attention = in_plane_attention

    def forward(self, c, *args, **kwargs):
        # return super().forward(x, timesteps, context, y, get_attr, **kwargs)
        """
        Forward pass of DiT.
        c: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        """

        x = self.pos_embed.repeat(
            c.shape[0], 1, 1).to(c.dtype)  # (N, T, D), where T = H * W / patch_size ** 2

        if self.return_all_layers:
            all_layers = []

        # if context is not None:
        # c = context # B 3HW C

        for blk_idx, block in enumerate(self.blocks):
            if self.roll_out:
                if self.in_plane_attention: # plane-wise output
                    if blk_idx % 2 == 0:  # with-in plane self attention
                        x = rearrange(x, 'b (n l) c -> (b n) l c ', n=self.plane_n)
                        x = block(x,
                                rearrange(c,
                                            'b (n l) c -> (b n) l c ',
                                            n=self.plane_n))  # (N, T, D)
                        # st()
                        if self.return_all_layers:

                            all_layers.append(
                                rearrange(x,
                                        '(b n) l c -> b (n l) c',
                                        n=self.plane_n))

                            # all_layers.append(x)
                    else:  # global attention
                        x = rearrange(x, '(b n) l c -> b (n l) c ', n=self.plane_n)
                        x = block(x, c)  # (N, T, D)
                        # st()
                        if self.return_all_layers:
                            #  all merged into B dim
                            all_layers.append(x)
                            # all_layers.append(
                            #     rearrange(x,
                            #               'b (n l) c -> (b n) l c',
                            #               n=self.plane_n))
                else:

                    # ! already b (n l) c
                    # if blk_idx == 0: # rearrange once
                    #     x = rearrange(x, '(b n) l c -> b (n l) c ', n=self.plane_n)
                    x = block(x, c)  # (N, T, D)
                    if self.return_all_layers:
                        #  all merged into B dim
                        all_layers.append(x)

            else:
                x = block(x, c)  # (N, T, D)

        # x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)

        # if self.roll_out:  # move n from L to B axis
        # x = rearrange(x, 'b (n l) c ->(b n) l c', n=3)

        # x = self.unpatchify(x)  # (N, out_channels, H, W)

        # if self.roll_out:  # move n from L to B axis
        #     x = rearrange(x, '(b n) c h w -> b (n c) h w', n=3)
        # st()

        if self.return_all_layers:
            return all_layers
        else:
            # return x.to(torch.float32)
            return x


# class DiT2_DPT(DiT2):
#     def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4, class_dropout_prob=0.1, num_classes=1000, learn_sigma=True, mixing_logit_init=-3, mixed_prediction=True, context_dim=False, roll_out=False, plane_n=3, vit_blk=...):
#         super().__init__(input_size, patch_size, in_channels, hidden_size, depth, num_heads, mlp_ratio, class_dropout_prob, num_classes, learn_sigma, mixing_logit_init, mixed_prediction, context_dim, roll_out, plane_n, vit_blk)
#         self.return_all_layers = True

#################################################################################
#                                   DiT2 Configs                                  #
#################################################################################


def DiT2_XL_2(**kwargs):
    return DiT2(depth=28,
                hidden_size=1152,
                patch_size=2,
                num_heads=16,
                **kwargs)


def DiT2_XL_2_half(**kwargs):
    return DiT2(depth=28 // 2,
                hidden_size=1152,
                patch_size=2,
                num_heads=16,
                **kwargs)


def DiT2_XL_4(**kwargs):
    return DiT2(depth=28,
                hidden_size=1152,
                patch_size=4,
                num_heads=16,
                **kwargs)


def DiT2_XL_8(**kwargs):
    return DiT2(depth=28,
                hidden_size=1152,
                patch_size=8,
                num_heads=16,
                **kwargs)


def DiT2_L_2(**kwargs):
    return DiT2(depth=24,
                hidden_size=1024,
                patch_size=2,
                num_heads=16,
                **kwargs)


def DiT2_L_2_stage1(**kwargs):
    return DiT2(depth=24-6,
                hidden_size=1024,
                patch_size=2,
                num_heads=16,
                **kwargs)

def DiT2_L_2_stage1(**kwargs):
    return DiT2(depth=24-6,
                hidden_size=1024,
                patch_size=2,
                num_heads=16,
                **kwargs)

def DiT2_L_2_half(**kwargs):
    return DiT2(depth=24//2,
                hidden_size=1024,
                patch_size=2,
                num_heads=16,
                **kwargs)

def DiT2_L_2_half_ninelayer(**kwargs):
    return DiT2(depth=9,
                hidden_size=1024,
                patch_size=2,
                num_heads=16,
                **kwargs)



def DiT2_L_4(**kwargs):
    return DiT2(depth=24,
                hidden_size=1024,
                patch_size=4,
                num_heads=16,
                **kwargs)


def DiT2_L_8(**kwargs):
    return DiT2(depth=24,
                hidden_size=1024,
                patch_size=8,
                num_heads=16,
                **kwargs)


def DiT2_B_2(**kwargs):
    return DiT2(depth=12,
                hidden_size=768,
                patch_size=2,
                num_heads=12,
                **kwargs)


def DiT2_B_2_stage1(**kwargs):
    return DiT2(depth=12, # ! just 12, stage-2 3 layers afterwards.
                hidden_size=768,
                patch_size=2,
                num_heads=12,
                **kwargs)

def DiT2_B_4(**kwargs):
    return DiT2(depth=12,
                hidden_size=768,
                patch_size=4,
                num_heads=12,
                **kwargs)


def DiT2_B_8(**kwargs):
    return DiT2(depth=12,
                hidden_size=768,
                patch_size=8,
                num_heads=12,
                **kwargs)


def DiT2_B_16(**kwargs):  # ours cfg
    return DiT2(depth=12,
                hidden_size=768,
                patch_size=16,
                num_heads=12,
                **kwargs)


def DiT2_S_2(**kwargs):
    return DiT2(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT2_S_4(**kwargs):
    return DiT2(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT2_S_8(**kwargs):
    return DiT2(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT2_models = {
    'DiT2-XL/2': DiT2_XL_2,
    'DiT2-XL/2/half': DiT2_XL_2_half,
    'DiT2-XL/4': DiT2_XL_4,
    'DiT2-XL/8': DiT2_XL_8,
    'DiT2-L/2': DiT2_L_2,
    'DiT2-L/2/S1': DiT2_L_2_stage1,
    'DiT2-L/2/S1-v2': DiT2_L_2_stage1,
    'DiT2-B/2/S1': DiT2_B_2_stage1,
    'DiT2-L/4': DiT2_L_4,
    'DiT2-L/2-half': DiT2_L_2_half,
    'DiT2-L/2-ninelayer': DiT2_L_2_half_ninelayer,
    'DiT2-L/8': DiT2_L_8,
    'DiT2-B/2': DiT2_B_2,
    'DiT2-B/4': DiT2_B_4,
    'DiT2-B/8': DiT2_B_8,
    'DiT2-B/16': DiT2_B_16,
    'DiT2-S/2': DiT2_S_2,
    'DiT2-S/4': DiT2_S_4,
    'DiT2-S/8': DiT2_S_8,
}
