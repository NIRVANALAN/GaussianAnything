import torch.nn as nn
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from pdb import set_trace as st

from ldm.modules.attention import MemoryEfficientCrossAttention
from .dit_decoder import DiT2

class DiT3D(DiT2):
    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4, class_dropout_prob=0.1, num_classes=1000, learn_sigma=True, mixing_logit_init=-3, mixed_prediction=True, context_dim=False, roll_out=False, plane_n=3, return_all_layers=False, in_plane_attention=True, vit_blk=...):
        super().__init__(input_size, patch_size, in_channels, hidden_size, depth, num_heads, mlp_ratio, class_dropout_prob, num_classes, learn_sigma, mixing_logit_init, mixed_prediction, context_dim, roll_out, plane_n, return_all_layers, in_plane_attention, vit_blk)
        # follow point infinity, add "write" CA block per 6 blocks

        # 25/4/2024, cascade a "read&write" block after the DiT base model.
        self.read_ca = MemoryEfficientCrossAttention(hidden_size, context_dim)
        self.point_infinity_blocks = nn.ModuleList([
            vit_blk(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(2)
        ])
    
    def initialize_weights(self):
        super().initialize_weights()

        # Zero-out adaLN modulation layers in DiT blocks:
        # ! no final layer anymore
        for block in self.point_infinity_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                                                      
        
    
    def forward(self, c, *args, **kwargs):
        x_base = super().forward(c, *args, **kwargs) # base latent
        # add read&write block



#################################################################################
#                                   DiT3D Configs                                  #
#################################################################################


def DiT3DXL_2(**kwargs):
    return DiT3D(depth=28,
                hidden_size=1152,
                patch_size=2,
                num_heads=16,
                **kwargs)


def DiT3DXL_2_half(**kwargs):
    return DiT3D(depth=28 // 2,
                hidden_size=1152,
                patch_size=2,
                num_heads=16,
                **kwargs)


def DiT3DXL_4(**kwargs):
    return DiT3D(depth=28,
                hidden_size=1152,
                patch_size=4,
                num_heads=16,
                **kwargs)


def DiT3DXL_8(**kwargs):
    return DiT3D(depth=28,
                hidden_size=1152,
                patch_size=8,
                num_heads=16,
                **kwargs)


def DiT3DL_2(**kwargs):
    return DiT3D(depth=24,
                hidden_size=1024,
                patch_size=2,
                num_heads=16,
                **kwargs)


def DiT3DL_2_half(**kwargs):
    return DiT3D(depth=24 // 2,
                hidden_size=1024,
                patch_size=2,
                num_heads=16,
                **kwargs)


def DiT3DL_4(**kwargs):
    return DiT3D(depth=24,
                hidden_size=1024,
                patch_size=4,
                num_heads=16,
                **kwargs)


def DiT3DL_8(**kwargs):
    return DiT3D(depth=24,
                hidden_size=1024,
                patch_size=8,
                num_heads=16,
                **kwargs)


def DiT3DB_2(**kwargs):
    return DiT3D(depth=12,
                hidden_size=768,
                patch_size=2,
                num_heads=12,
                **kwargs)


def DiT3DB_4(**kwargs):
    return DiT3D(depth=12,
                hidden_size=768,
                patch_size=4,
                num_heads=12,
                **kwargs)


def DiT3DB_8(**kwargs):
    return DiT3D(depth=12,
                hidden_size=768,
                patch_size=8,
                num_heads=12,
                **kwargs)


def DiT3DB_16(**kwargs):  # ours cfg
    return DiT3D(depth=12,
                hidden_size=768,
                patch_size=16,
                num_heads=12,
                **kwargs)


def DiT3DS_2(**kwargs):
    return DiT3D(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT3DS_4(**kwargs):
    return DiT3D(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT3DS_8(**kwargs):
    return DiT3D(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT3Dmodels = {
    'DiT3D-XL/2': DiT3DXL_2,
    'DiT3D-XL/2/half': DiT3DXL_2_half,
    'DiT3D-XL/4': DiT3DXL_4,
    'DiT3D-XL/8': DiT3DXL_8,
    'DiT3D-L/2': DiT3DL_2,
    'DiT3D-L/2/half': DiT3DL_2_half,
    'DiT3D-L/4': DiT3DL_4,
    'DiT3D-L/8': DiT3DL_8,
    'DiT3D-B/2': DiT3DB_2,
    'DiT3D-B/4': DiT3DB_4,
    'DiT3D-B/8': DiT3DB_8,
    'DiT3D-B/16': DiT3DB_16,
    'DiT3D-S/2': DiT3DS_2,
    'DiT3D-S/4': DiT3DS_4,
    'DiT3D-S/8': DiT3DS_8,
}

