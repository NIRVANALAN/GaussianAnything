import torch
import torch.nn as nn
import numpy as np

import math
from einops import rearrange
from vit.vision_transformer import MemEffAttention, Attention
# from xformers.triton import FusedLayerNorm as LayerNorm
from torch.nn import LayerNorm
from xformers.components.feedforward import fused_mlp
# from xformers.components.feedforward import mlp
from xformers.components.activations import build_activation, Activation


class PositionalEncoding(nn.Module):

    def __init__(self, num_octaves=8, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

    def forward(self, coords, rays=None):
        embed_fns = []
        batch_size, num_points, dim = coords.shape

        octaves = torch.arange(self.start_octave,
                               self.start_octave + self.num_octaves)
        octaves = octaves.float().to(coords)
        multipliers = 2**octaves * math.pi
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers

        sines = torch.sin(scaled_coords).reshape(batch_size, num_points,
                                                 dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(batch_size, num_points,
                                                   dim * self.num_octaves)

        result = torch.cat((sines, cosines), -1)
        return result


class RayEncoder(nn.Module):

    def __init__(self,
                 pos_octaves=8,
                 pos_start_octave=0,
                 ray_octaves=4,
                 ray_start_octave=0):
        super().__init__()
        self.pos_encoding = PositionalEncoding(num_octaves=pos_octaves,
                                               start_octave=pos_start_octave)
        self.ray_encoding = PositionalEncoding(num_octaves=ray_octaves,
                                               start_octave=ray_start_octave)

    def forward(self, pos, rays):
        if len(rays.shape) == 4:
            batchsize, height, width, dims = rays.shape
            pos_enc = self.pos_encoding(pos.unsqueeze(1))
            pos_enc = pos_enc.view(batchsize, pos_enc.shape[-1], 1, 1)
            pos_enc = pos_enc.repeat(1, 1, height, width)
            rays = rays.flatten(1, 2)

            ray_enc = self.ray_encoding(rays)
            ray_enc = ray_enc.view(batchsize, height, width, ray_enc.shape[-1])
            ray_enc = ray_enc.permute((0, 3, 1, 2))
            x = torch.cat((pos_enc, ray_enc), 1)
        else:
            pos_enc = self.pos_encoding(pos)
            ray_enc = self.ray_encoding(rays)
            x = torch.cat((pos_enc, ray_enc), -1)

        return x


# Transformer implementation based on ViT
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim,
                                           dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0., selfatt=True, kv_dim=None):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim=-1)
#         if selfatt:
#             self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#         else:
#             self.to_q = nn.Linear(dim, inner_dim, bias=False)
#             self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x, z=None):
#         if z is None:
#             qkv = self.to_qkv(x).chunk(3, dim=-1)
#         else:
#             q = self.to_q(x)
#             k, v = self.to_kv(z).chunk(2, dim=-1)
#             qkv = (q, k, v)

#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)


class Transformer(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 dropout=0.,
                 selfatt=True,
                 kv_dim=None, 
                 no_flash_op=False,):
        super().__init__()

        # if no_flash_op:
        #     attn_cls = Attention # raw torch attention
        # else:
        attn_cls = MemEffAttention

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim,
                            attn_cls(
                                dim,
                                num_heads=heads,
                                qkv_bias=True,
                                qk_norm=True, # as in vit-22B
                                no_flash_op=no_flash_op, 
                            )),
                    PreNorm(
                        dim,
                        fused_mlp.FusedMLP(dim,
                        # mlp.MLP(dim,
                                           hidden_layer_multiplier=mlp_dim //
                                           dim,
                                           dropout=dropout,
                                           activation=Activation.GeLU))
                ]))

    def forward(self, x):
        for attn, ff in self.layers: # type: ignore
            x = attn(x) + x
            x = ff(x) + x
        return x
