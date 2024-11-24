import torch
import torch.nn as nn
import numpy as np
import math

from pdb import set_trace as st

from .dit_models import DiT, DiTBlock, DiT_models, get_2d_sincos_pos_embed


class DiT_Triplane_V1(DiT):
    """
    1. merge the 3*H*W as L, and 8 as C only
    2. pachify, flat into 224*(224*3) with 8 channels for pachify
    3. unpachify accordingly
    """

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
                 learn_sigma=False):

        input_size = (input_size, input_size*3)
        super().__init__(input_size, patch_size, in_channels//3, hidden_size, # type: ignore
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma)
    
    def initialize_weights(self):
        """all the same except the PE part
        """
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.x_embedder.grid_size)
        # st()
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        # ! untouched below
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
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        # TODO
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] # type: ignore
        h = w = int((x.shape[1]//3)**0.5)
        assert h * w * 3 == x.shape[1] # merge triplane 3 dims with hw

        x = x.reshape(shape=(x.shape[0], h, w, 3, p, p, c))
        x = torch.einsum('nhwzpqc->nczhpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c*3, h * p, h * p)) # type: ignore
        return imgs # B 8*3 H W

    def forward(self, x, t, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        # ! merge tri-channel into w chanenl for 3D-aware TX
        x = x.reshape(x.shape[0], -1, 3, x.shape[2], x.shape[3]) # B 8 3 H W
        x = x.permute(0,1,3,4,2).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1]*3) # B 8 H W83

        x = self.x_embedder(
            x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)

        if self.y_embedder is not None:
            assert y is not None
            y = self.y_embedder(y, self.training)  # (N, D)
            c = t + y  # (N, D)
        else:
            c = t

        for block in self.blocks:
            x = block(x, c)  # (N, T, D)

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        
        return x




class DiT_Triplane_V1_learnedPE(DiT_Triplane_V1):
    """
    1. learned PE, default cos/sin wave
    """

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
                 learn_sigma=True):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma)


class DiT_Triplane_V1_fixed3DPE(DiT_Triplane_V1):
    """
    1. 3D aware PE, fixed
    """

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
                 learn_sigma=True):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma)


class DiT_Triplane_V1_learned3DPE(DiT_Triplane_V1):
    """
    1. init with 3D aware PE, learnable
    """

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
                 learn_sigma=True):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma)

def V1_Triplane_DiT_S_2(**kwargs):
    return DiT_Triplane_V1(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def V1_Triplane_DiT_S_4(**kwargs):
    return DiT_Triplane_V1(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def V1_Triplane_DiT_S_8(**kwargs):
    return DiT_Triplane_V1(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def V1_Triplane_DiT_B_8(**kwargs):
    return DiT_Triplane_V1(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def V1_Triplane_DiT_B_16(**kwargs): # ours cfg
    return DiT_Triplane_V1(depth=12, hidden_size=768, patch_size=16, num_heads=12, **kwargs)

DiT_models.update({
    'v1-T-DiT-S/2': V1_Triplane_DiT_S_2,
    'v1-T-DiT-S/4': V1_Triplane_DiT_S_4,
    'v1-T-DiT-S/8': V1_Triplane_DiT_S_8,
    'v1-T-DiT-B/8': V1_Triplane_DiT_B_8,
    'v1-T-DiT-B/16': V1_Triplane_DiT_B_16,
})