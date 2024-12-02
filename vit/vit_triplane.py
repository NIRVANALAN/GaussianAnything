import math
from pathlib import Path
# from pytorch3d.ops import create_sphere
import torchvision
import point_cloud_utils as pcu
from tqdm import trange
import random
import einops
from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from functools import partial
from torch.profiler import profile, record_function, ProfilerActivity

from nsr.networks_stylegan2 import Generator as StyleGAN2Backbone
from nsr.volumetric_rendering.renderer import ImportanceRenderer, ImportanceRendererfg_bg
from nsr.volumetric_rendering.ray_sampler import RaySampler
from nsr.triplane import OSGDecoder, Triplane, Triplane_fg_bg_plane
# from nsr.losses.helpers import ResidualBlock
from utils.dust3r.heads.dpt_head import create_dpt_head_ln3diff
from utils.nerf_utils import get_embedder
from vit.vision_transformer import TriplaneFusionBlockv4_nested, TriplaneFusionBlockv4_nested_init_from_dino_lite, TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout, VisionTransformer, TriplaneFusionBlockv4_nested_init_from_dino

from .vision_transformer import Block, VisionTransformer
from .utils import trunc_normal_

from guided_diffusion import dist_util, logger

from pdb import set_trace as st

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from torch_utils.components import PixelShuffleUpsample, ResidualBlock, Upsample, PixelUnshuffleUpsample, Conv3x3TriplaneTransformation
from torch_utils.distributions.distributions import DiagonalGaussianDistribution
from nsr.superresolution import SuperresolutionHybrid2X, SuperresolutionHybrid4X

from torch.nn.parameter import Parameter, UninitializedParameter, UninitializedBuffer

from nsr.common_blks import ResMlp
from timm.models.vision_transformer import PatchEmbed, Mlp
from .vision_transformer import *

from dit.dit_models import get_2d_sincos_pos_embed
from dit.dit_decoder import DiTBlock2
from torch import _assert
from itertools import repeat
import collections.abc

from nsr.srt.layers import Transformer as SRT_TX
from nsr.srt.layers import PreNorm

# from diffusers.models.upsampling import Upsample2D

from torch_utils.components import NearestConvSR
from timm.models.vision_transformer import PatchEmbed

from utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply, build_rotation

# from nsr.gs import GaussianRenderer

from utils.dust3r.heads import create_dpt_head

from ldm.modules.attention import MemoryEfficientCrossAttention, CrossAttention

# from nsr.geometry.camera.perspective_camera import PerspectiveCamera
# from nsr.geometry.render.neural_render import NeuralRender
# from nsr.geometry.rep_3d.flexicubes_geometry import FlexiCubesGeometry
# from utils.mesh_util import xatlas_uvmap


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


def approx_gelu():
    return nn.GELU(approximate="tanh")


def init_gaussian_prediction(gaussian_pred_mlp):

    # https://github.com/szymanowiczs/splatter-image/blob/98b465731c3273bf8f42a747d1b6ce1a93faf3d6/configs/dataset/chairs.yaml#L15

    out_channels = [3, 1, 3, 4, 3]  # xyz, opacity, scale, rotation, rgb
    scale_inits = [  # ! avoid affecting final value (offset) 
        0,  #xyz_scale
        0.0,  #cfg.model.opacity_scale, 
        # 0.001,  #cfg.model.scale_scale,
        0,  #cfg.model.scale_scale,
        1,  # rotation
        0
    ]  # rgb

    bias_inits = [
        0.0,  # cfg.model.xyz_bias, no deformation here
        0,  # cfg.model.opacity_bias, sigmoid(0)=0.5 at init
        -2.5,  # scale_bias
        0.0,  # rotation
        0.5
    ]  # rgb

    start_channels = 0

    # for out_channel, b, s in zip(out_channels, bias, scale):
    for out_channel, b, s in zip(out_channels, bias_inits, scale_inits):
        # nn.init.xavier_uniform_(
        #     self.superresolution['conv_sr'].dpt.head[-1].weight[
        #         start_channels:start_channels + out_channel, ...], s)
        nn.init.constant_(
            gaussian_pred_mlp.weight[start_channels:start_channels +
                                     out_channel, ...], s)
        nn.init.constant_(
            gaussian_pred_mlp.bias[start_channels:start_channels +
                                   out_channel], b)
        start_channels += out_channel


class PatchEmbedTriplane(nn.Module):
    """ GroupConv patchembeder on triplane
    """

    def __init__(
        self,
        img_size=32,
        patch_size=2,
        in_chans=4,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        plane_n=3,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.plane_n = plane_n
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans,
                              embed_dim * self.plane_n,
                              kernel_size=patch_size,
                              stride=patch_size,
                              bias=bias,
                              groups=self.plane_n)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # st()
        B, C, H, W = x.shape
        _assert(
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        )
        _assert(
            W == self.img_size[1],
            f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        )
        x = self.proj(x)  # B 3*C token_H token_W

        x = x.reshape(B, x.shape[1] // self.plane_n, self.plane_n, x.shape[-2],
                      x.shape[-1])  # B C 3 H W

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BC3HW -> B 3HW C
        x = self.norm(x)
        return x


# https://github.com/facebookresearch/MCC/blob/main/mcc_model.py#L81
class XYZPosEmbed(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, embed_dim, multires=10):
        super().__init__()
        self.embed_dim = embed_dim
        # no [cls] token here.

        # ! use fixed PE here
        self.embed_fn, self.embed_input_ch = get_embedder(multires)
        # st()

        # self.two_d_pos_embed = nn.Parameter(
        #     # torch.zeros(1, 64 + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        #     torch.zeros(1, 64, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.win_size = 8

        self.xyz_projection = nn.Linear(self.embed_input_ch, embed_dim)

        # self.blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads=12, mlp_ratio=2.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        #     for _ in range(1)
        # ])

        # self.invalid_xyz_token = nn.Parameter(torch.zeros(embed_dim,))

        # self.initialize_weights()

    # def initialize_weights(self):
    #     # torch.nn.init.normal_(self.cls_token, std=.02)

    #     two_d_pos_embed = get_2d_sincos_pos_embed(self.two_d_pos_embed.shape[-1], 8, cls_token=False)
    #     self.two_d_pos_embed.data.copy_(torch.from_numpy(two_d_pos_embed).float().unsqueeze(0))

    #     torch.nn.init.normal_(self.invalid_xyz_token, std=.02)

    def forward(self, xyz):
        xyz = self.embed_fn(xyz)  # PE encoding
        xyz = self.xyz_projection(xyz)  # linear projection
        return xyz


class gaussian_prediction(nn.Module):

    def __init__(
        self,
        query_dim,
    ) -> None:
        super().__init__()
        self.gaussian_pred = nn.Sequential(
            nn.SiLU(), nn.Linear(query_dim, 14,
                                 bias=True))  # TODO, init require

        self.init_gaussian_prediction()

    def init_gaussian_prediction(self):

        # https://github.com/szymanowiczs/splatter-image/blob/98b465731c3273bf8f42a747d1b6ce1a93faf3d6/configs/dataset/chairs.yaml#L15

        out_channels = [3, 1, 3, 4, 3]  # xyz, opacity, scale, rotation, rgb
        scale_inits = [  # ! avoid affecting final value (offset) 
            0,  #xyz_scale
            0.0,  #cfg.model.opacity_scale, 
            # 0.001,  #cfg.model.scale_scale,
            0,  #cfg.model.scale_scale,
            1.0,  # rotation
            0
        ]  # rgb

        bias_inits = [
            0.0,  # cfg.model.xyz_bias, no deformation here
            0,  # cfg.model.opacity_bias, sigmoid(0)=0.5 at init
            -2.5,  # scale_bias
            0.0,  # rotation
            0.5
        ]  # rgb

        start_channels = 0

        # for out_channel, b, s in zip(out_channels, bias, scale):
        for out_channel, b, s in zip(out_channels, bias_inits, scale_inits):
            # nn.init.xavier_uniform_(
            #     self.superresolution['conv_sr'].dpt.head[-1].weight[
            #         start_channels:start_channels + out_channel, ...], s)
            nn.init.constant_(
                self.gaussian_pred[1].weight[start_channels:start_channels +
                                             out_channel, ...], s)
            nn.init.constant_(
                self.gaussian_pred[1].bias[start_channels:start_channels +
                                           out_channel], b)
            start_channels += out_channel

    def forward(self, x):

        return self.gaussian_pred(x)


class surfel_prediction(nn.Module):
    # for 2dgs

    def __init__(
        self,
        query_dim,
    ) -> None:
        super().__init__()
        self.gaussian_pred = nn.Sequential(
            nn.SiLU(), nn.Linear(query_dim, 13,
                                 bias=True))  # TODO, init require

        self.init_gaussian_prediction()

    def init_gaussian_prediction(self):

        # https://github.com/szymanowiczs/splatter-image/blob/98b465731c3273bf8f42a747d1b6ce1a93faf3d6/configs/dataset/chairs.yaml#L15

        out_channels = [3, 1, 2, 4, 3]  # xyz, opacity, scale, rotation, rgb
        scale_inits = [  # ! avoid affecting final value (offset) 
            0,  #xyz_scale
            0.0,  #cfg.model.opacity_scale, 
            # 0.001,  #cfg.model.scale_scale,
            0,  #cfg.model.scale_scale,
            1.0,  # rotation
            0
        ]  # rgb

        bias_inits = [
            0.0,  # cfg.model.xyz_bias, no deformation here
            0,  # cfg.model.opacity_bias, sigmoid(0)=0.5 at init
            -2.5,  # scale_bias
            0,  # scale bias, also 0
            0.0,  # rotation
            0.5
        ]  # rgb

        start_channels = 0

        # for out_channel, b, s in zip(out_channels, bias, scale):
        for out_channel, b, s in zip(out_channels, bias_inits, scale_inits):
            # nn.init.xavier_uniform_(
            #     self.superresolution['conv_sr'].dpt.head[-1].weight[
            #         start_channels:start_channels + out_channel, ...], s)
            nn.init.constant_(
                self.gaussian_pred[1].weight[start_channels:start_channels +
                                             out_channel, ...], s)
            nn.init.constant_(
                self.gaussian_pred[1].bias[start_channels:start_channels +
                                           out_channel], b)
            start_channels += out_channel

    def forward(self, x):

        return self.gaussian_pred(x)


class pointInfinityWriteCA(gaussian_prediction):

    def __init__(self,
                 query_dim,
                 context_dim,
                 heads=8,
                 dim_head=64,
                 dropout=0.0) -> None:
        super().__init__(query_dim=query_dim)
        self.write_ca = MemoryEfficientCrossAttention(query_dim, context_dim,
                                                      heads, dim_head, dropout)

    def forward(self, x, z, return_x=False):
        # x: point to write
        # z: extracted latent
        x = self.write_ca(x, z)  # write from z to x
        if return_x:
            return self.gaussian_pred(x), x  # ! integrate it into dit?
        else:
            return self.gaussian_pred(x)  # ! integrate it into dit?


class pointInfinityWriteCA_cascade(pointInfinityWriteCA):
    # gradually (in 6 times) add deformation offsets to the initialized canonical pts, follow PI
    def __init__(self,
                 vit_depth,
                 query_dim,
                 context_dim,
                 heads=8,
                 dim_head=64,
                 dropout=0) -> None:
        super().__init__(query_dim, context_dim, heads, dim_head, dropout)

        del self.write_ca
        # query_dim = 384 # to speed up CA compute
        write_ca_interval = 12 // 4
        # self.deform_pred = nn.Sequential( # to-gaussian layer
        #     nn.SiLU(), nn.Linear(query_dim, 3, bias=True)) # TODO, init require

        # query_dim = 384 here
        self.write_ca_blocks = nn.ModuleList([
            MemoryEfficientCrossAttention(query_dim, context_dim,
                                          heads=heads)  # make it lite
            for _ in range(write_ca_interval)
            # for _ in range(write_ca_interval)
        ])
        self.hooks = [3, 7, 11]  # hard coded for now
        # [(vit_depth * 1 // 3) - 1, (vit_depth * 2 // 4) - 1, (vit_depth * 3 // 4) - 1,
        #             vit_depth - 1]

    def forward(self, x: torch.Tensor, z: list):
        # x is the canonical point
        # z: extracted latent (for different layers), all layers in dit
        # TODO, optimize memory, no need to return all layers?
        # st()

        z = [z[hook] for hook in self.hooks]
        # st()

        for idx, ca_blk in enumerate(self.write_ca_blocks):
            x = x + ca_blk(x, z[idx])  # learn residual feature

        return self.gaussian_pred(x)


def create_sphere(radius, num_points):
    # Generate spherical coordinates
    phi = torch.linspace(0, 2 * torch.pi, num_points)
    theta = torch.linspace(0, torch.pi, num_points)
    phi, theta = torch.meshgrid(phi, theta, indexing='xy')

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)

    # Stack x, y, z coordinates
    points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

    return points


class GS_Adaptive_Write_CA(nn.Module):

    def __init__(
            self,
            query_dim,
            context_dim,
            f=4,  # upsampling ratio
            heads=8,
            dim_head=64,
            dropout=0.0) -> None:
        super().__init__()

        self.f = f
        self.write_ca = MemoryEfficientCrossAttention(query_dim, context_dim,
                                                      heads, dim_head, dropout)
        self.gaussian_residual_pred = nn.Sequential(
            nn.SiLU(),
            nn.Linear(query_dim, 14,
                      bias=True))  # predict residual, before activations

        # ! hard coded
        self.scene_extent = 0.9  # g-buffer, [-0.45, 0.45]
        self.percent_dense = 0.01  # 3dgs official value
        self.residual_offset_act = lambda x: torch.tanh(
            x) * self.scene_extent * 0.015  # avoid large deformation

        init_gaussian_prediction(self.gaussian_residual_pred[1])

    # def densify_and_split(self, gaussians_base, base_gaussian_xyz_embed):

    def forward(self,
                gaussians_base,
                gaussian_base_pre_activate,
                gaussian_base_feat,
                xyz_embed_fn,
                shrink_scale=True):
        # gaussians_base: xyz_base after activations and deform offset
        # xyz_base: original features (before activations)

        # ! use point embedder, or other features?
        # base_gaussian_xyz_embed = xyz_embed_fn(gaussians_base[..., :3])

        # x = self.densify_and_split(gaussians_base, base_gaussian_xyz_embed)

        # ! densify
        B, N = gaussians_base.shape[:2]  # gaussians upsample factor
        # n_init_points = self.get_xyz.shape[0]

        pos, opacity, scaling, rotation = gaussians_base[
            ..., 0:3], gaussians_base[..., 3:4], gaussians_base[
                ..., 4:7], gaussians_base[..., 7:11]

        # ! filter clone/densify based on scaling range
        split_mask = scaling.max(
            dim=-1
        )[0] > self.scene_extent * self.percent_dense  # shape: B 4096
        # clone_mask = ~split_mask

        stds = scaling.repeat_interleave(self.f, dim=1)  #  0 0 1 1 2 2...
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)  # B f*N 3

        # rots = build_rotation(rotation).repeat(N, 1, 1)
        # rots = rearrange(build_rotation(rearrange(rotation, 'B N ... -> (B N) ...')), '(B N) ... -> B N ...', B=B, N=N)
        # rots = rots.repeat_interleave(self.f, dim=1) # B f*N 3 3

        # torch.bmm only supports ndim=3 Tensor
        # new_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + pos.repeat_interleave(self.f, dim=1)
        new_xyz = samples + pos.repeat_interleave(
            self.f, dim=1)  # ! no rotation for now
        # new_xyz: B f*N 3

        # ! new points to features
        new_xyz_embed = xyz_embed_fn(new_xyz)
        new_gaussian_embed = self.write_ca(
            new_xyz_embed, gaussian_base_feat)  # write from z to x

        # ! predict gaussians residuals
        gaussian_residual_pre_activate = self.gaussian_residual_pred(
            new_gaussian_embed)

        # ! add back. how to deal with new rotations? check the range first.
        # scaling and rotation.
        if shrink_scale:
            gaussian_base_pre_activate[split_mask][
                4:7] -= 1  # reduce scale for those points

        gaussian_base_pre_activate_repeat = gaussian_base_pre_activate.repeat_interleave(
            self.f, dim=1)

        # new scaling
        # ! pre-activate scaling value, shall be negative? since more values are 0.1 before softplus.
        # TODO wrong here, shall get new scaling before repeat
        gaussians = gaussian_residual_pre_activate + gaussian_base_pre_activate_repeat  # learn the residual

        new_gaussians_pos = new_xyz + self.residual_offset_act(
            gaussians[..., :3])

        return gaussians, new_gaussians_pos  # return positions independently


class GS_Adaptive_Read_Write_CA(nn.Module):

    def __init__(
            self,
            query_dim,
            context_dim,
            mlp_ratio,
            vit_heads,
            f=4,  # upsampling ratio
            heads=8,
            dim_head=64,
            dropout=0.0,
            depth=2,
            vit_blk=DiTBlock2) -> None:
        super().__init__()

        self.f = f
        self.read_ca = MemoryEfficientCrossAttention(query_dim, context_dim,
                                                     heads, dim_head, dropout)

        # more dit blocks
        self.point_infinity_blocks = nn.ModuleList([
            vit_blk(context_dim, num_heads=vit_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)  # since dit-b here
        ])

        self.write_ca = MemoryEfficientCrossAttention(query_dim, context_dim,
                                                      heads, dim_head, dropout)

        self.gaussian_residual_pred = nn.Sequential(
            nn.SiLU(),
            nn.Linear(query_dim, 14,
                      bias=True))  # predict residual, before activations

        # ! hard coded
        self.scene_extent = 0.9  # g-buffer, [-0.45, 0.45]
        self.percent_dense = 0.01  # 3dgs official value
        self.residual_offset_act = lambda x: torch.tanh(
            x) * self.scene_extent * 0.015  # avoid large deformation

        self.initialize_weights()

    def initialize_weights(self):
        init_gaussian_prediction(self.gaussian_residual_pred[1])

        for block in self.point_infinity_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    # def densify_and_split(self, gaussians_base, base_gaussian_xyz_embed):

    def forward(self, gaussians_base, gaussian_base_pre_activate,
                gaussian_base_feat, latent_from_vit, vae_latent, xyz_embed_fn):
        # gaussians_base: xyz_base after activations and deform offset
        # xyz_base: original features (before activations)

        # ========= START read CA ========
        latent_from_vit = self.read_ca(latent_from_vit,
                                       gaussian_base_feat)  # z_i -> z_(i+1)

        for blk_idx, block in enumerate(self.point_infinity_blocks):
            latent_from_vit = block(latent_from_vit,
                                    vae_latent)  # vae_latent: c

        # ========= END read CA ========

        # ! use point embedder, or other features?
        # base_gaussian_xyz_embed = xyz_embed_fn(gaussians_base[..., :3])

        # x = self.densify_and_split(gaussians_base, base_gaussian_xyz_embed)

        # ! densify
        B, N = gaussians_base.shape[:2]  # gaussians upsample factor
        # n_init_points = self.get_xyz.shape[0]

        pos, opacity, scaling, rotation = gaussians_base[
            ..., 0:3], gaussians_base[..., 3:4], gaussians_base[
                ..., 4:7], gaussians_base[..., 7:11]

        # ! filter clone/densify based on scaling range
        split_mask = scaling.max(
            dim=-1
        )[0] > self.scene_extent * self.percent_dense  # shape: B 4096
        # clone_mask = ~split_mask

        stds = scaling.repeat_interleave(self.f, dim=1)  #  0 0 1 1 2 2...
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)  # B f*N 3

        rots = build_rotation(rotation).repeat(N, 1, 1)
        rots = rearrange(build_rotation(
            rearrange(rotation, 'B N ... -> (B N) ...')),
                         '(B N) ... -> B N ...',
                         B=B,
                         N=N)
        rots = rots.repeat_interleave(self.f, dim=1)  # B f*N 3 3

        # torch.bmm only supports ndim=3 Tensor
        new_xyz = torch.matmul(
            rots, samples.unsqueeze(-1)).squeeze(-1) + pos.repeat_interleave(
                self.f, dim=1)
        # new_xyz = samples + pos.repeat_interleave(
        #     self.f, dim=1)  # ! no rotation for now
        # new_xyz: B f*N 3

        # ! new points to features
        new_xyz_embed = xyz_embed_fn(new_xyz)
        new_gaussian_embed = self.write_ca(
            new_xyz_embed, latent_from_vit
        )  # ! use z_(i+1), rather than gaussian_base_feat here

        # ! predict gaussians residuals
        gaussian_residual_pre_activate = self.gaussian_residual_pred(
            new_gaussian_embed)

        # ! add back. how to deal with new rotations? check the range first.
        # scaling and rotation.
        gaussian_base_pre_activate[split_mask][
            4:7] -= 1  # reduce scale for those points
        gaussian_base_pre_activate_repeat = gaussian_base_pre_activate.repeat_interleave(
            self.f, dim=1)

        # new scaling
        # ! pre-activate scaling value, shall be negative? since more values are 0.1 before softplus.
        # TODO wrong here, shall get new scaling before repeat
        gaussians = gaussian_residual_pre_activate + gaussian_base_pre_activate_repeat  # learn the residual

        new_gaussians_pos = new_xyz + self.residual_offset_act(
            gaussians[..., :3])

        return gaussians, new_gaussians_pos, latent_from_vit, new_gaussian_embed  # return positions independently


class GS_Adaptive_Read_Write_CA_adaptive(GS_Adaptive_Read_Write_CA):

    def __init__(self,
                 query_dim,
                 context_dim,
                 mlp_ratio,
                 vit_heads,
                 f=4,
                 heads=8,
                 dim_head=64,
                 dropout=0,
                 depth=2,
                 vit_blk=DiTBlock2) -> None:
        super().__init__(query_dim, context_dim, mlp_ratio, vit_heads, f,
                         heads, dim_head, dropout, depth, vit_blk)

        # assert self.f == 6

    def forward(self, gaussians_base, gaussian_base_pre_activate,
                gaussian_base_feat, latent_from_vit, vae_latent, xyz_embed_fn):
        # gaussians_base: xyz_base after activations and deform offset
        # xyz_base: original features (before activations)

        # ========= START read CA ========
        latent_from_vit = self.read_ca(latent_from_vit,
                                       gaussian_base_feat)  # z_i -> z_(i+1)

        for blk_idx, block in enumerate(self.point_infinity_blocks):
            latent_from_vit = block(latent_from_vit,
                                    vae_latent)  # vae_latent: c

        # ========= END read CA ========

        # ! use point embedder, or other features?
        # base_gaussian_xyz_embed = xyz_embed_fn(gaussians_base[..., :3])

        # x = self.densify_and_split(gaussians_base, base_gaussian_xyz_embed)

        # ! densify
        B, N = gaussians_base.shape[:2]  # gaussians upsample factor
        # n_init_points = self.get_xyz.shape[0]

        pos, opacity, scaling, rotation = gaussians_base[
            ..., 0:3], gaussians_base[..., 3:4], gaussians_base[
                ..., 4:7], gaussians_base[..., 7:11]

        # ! filter clone/densify based on scaling range

        split_mask = scaling.max(
            dim=-1
        )[0] > self.scene_extent * self.percent_dense  # shape: B 4096

        # clone_mask = ~split_mask

        # stds = scaling.repeat_interleave(self.f, dim=1)  #  B 13824 3
        # stds = scaling.unsqueeze(1).repeat_interleave(self.f, dim=1)  #  B 6 13824 3
        stds = scaling  #  B 13824 3

        # TODO, in mat form. axis aligned creation.
        samples = torch.zeros(B, N, 3, 3).to(stds.device)

        samples[..., 0, 0] = stds[..., 0]
        samples[..., 1, 1] = stds[..., 1]
        samples[..., 2, 2] = stds[..., 2]

        eye_mat = torch.cat([torch.eye(3), -torch.eye(3)],
                            0)  # 6 * 3, to put gaussians along the axis
        eye_mat = eye_mat.reshape(1, 1, 6, 3).repeat(B, N, 1,
                                                     1).to(stds.device)
        samples = (eye_mat @ samples).squeeze(-1)

        # st()
        # means = torch.zeros_like(stds)
        # samples = torch.normal(mean=means, std=stds)  # B f*N 3

        rots = rearrange(build_rotation(
            rearrange(rotation, 'B N ... -> (B N) ...')),
                         '(B N) ... -> B N ...',
                         B=B,
                         N=N)
        rots = rots.unsqueeze(2).repeat_interleave(self.f, dim=2)  # B f*N 3 3

        # torch.bmm only supports ndim=3 Tensor
        # new_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + pos.repeat_interleave(self.f, dim=1)
        # st()

        # new_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + pos.repeat_interleave(self.f, dim=1)
        new_xyz = (rots @ samples.unsqueeze(-1)).squeeze(-1) + pos.unsqueeze(
            2).repeat_interleave(self.f, dim=2)  # B N 6 3
        new_xyz = rearrange(new_xyz, 'b n f c -> b (n f) c')

        # ! not considering rotation here
        # new_xyz = samples + pos.repeat_interleave(
        #     self.f, dim=1)  # ! no rotation for now
        # new_xyz: B f*N 3

        # ! new points to features
        new_xyz_embed = xyz_embed_fn(new_xyz)
        new_gaussian_embed = self.write_ca(
            new_xyz_embed, latent_from_vit
        )  # ! use z_(i+1), rather than gaussian_base_feat here

        # ! predict gaussians residuals
        gaussian_residual_pre_activate = self.gaussian_residual_pred(
            new_gaussian_embed)

        # ! add back. how to deal with new rotations? check the range first.
        # scaling and rotation.
        # gaussian_base_pre_activate[split_mask][
        #     4:7] -= 1  # reduce scale for those points

        gaussian_base_pre_activate_repeat = gaussian_base_pre_activate.repeat_interleave(
            self.f, dim=1)

        # new scaling
        # ! pre-activate scaling value, shall be negative? since more values are 0.1 before softplus.
        # TODO wrong here, shall get new scaling before repeat
        gaussians = gaussian_residual_pre_activate + gaussian_base_pre_activate_repeat  # learn the residual

        # new_gaussians_pos = new_xyz + self.residual_offset_act(
        #     gaussians[..., :3])

        return gaussians, new_xyz, latent_from_vit, new_gaussian_embed  # return positions independently


class GS_Adaptive_Read_Write_CA_adaptive_f14_prepend(
        GS_Adaptive_Read_Write_CA_adaptive):

    def __init__(self,
                 query_dim,
                 context_dim,
                 mlp_ratio,
                 vit_heads,
                 f=4,
                 heads=8,
                 dim_head=64,
                 dropout=0,
                 depth=2,
                 vit_blk=DiTBlock2, 
                 no_flash_op=False,) -> None:
        super().__init__(query_dim, context_dim, mlp_ratio, vit_heads, f,
                         heads, dim_head, dropout, depth, vit_blk)

        # corner_mat = torch.empty(8,3)
        # counter = 0
        # for i in range(-1,3,2):
        #     for j in range(-1,3,2):
        #         for k in range(-1,3,2):
        #             corner_mat[counter] = torch.Tensor([i,j,k])
        #             counter += 1

        # self.corner_mat=corner_mat.contiguous().to(dist_util.dev()).reshape(1,1,8,3)

        del self.read_ca, self.write_ca
        del self.point_infinity_blocks

        # ? why not saved to checkpoint
        # self.latent_embedding = nn.Parameter(torch.randn(1, f, query_dim)).to(
        #     dist_util.dev())

        # ! not .cuda() here
        self.latent_embedding = nn.Parameter(torch.randn(1, f, query_dim),
                                             requires_grad=True)

        self.transformer = SRT_TX(
            context_dim,  # 12 * 64 = 768
            depth=depth,
            heads=context_dim // 64,  # vit-b default.
            mlp_dim=4 * context_dim,  # 1536 by default
            no_flash_op=no_flash_op,
        )

        # self.offset_act = lambda x: torch.tanh(x) * (self.scene_range[
        #     1]) * 0.5  # regularize small offsets

    def forward(self, gaussians_base, gaussian_base_pre_activate,
                gaussian_base_feat, latent_from_vit, vae_latent, xyz_embed_fn,
                offset_act):
        # gaussians_base: xyz_base after activations and deform offset
        # xyz_base: original features (before activations)

        # ========= START read CA ========
        # latent_from_vit = self.read_ca(latent_from_vit,
        #                                gaussian_base_feat)  # z_i -> z_(i+1)

        # for blk_idx, block in enumerate(self.point_infinity_blocks):
        #     latent_from_vit = block(latent_from_vit,
        #                             vae_latent)  # vae_latent: c

        # ========= END read CA ========

        # ! use point embedder, or other features?
        # base_gaussian_xyz_embed = xyz_embed_fn(gaussians_base[..., :3])

        # x = self.densify_and_split(gaussians_base, base_gaussian_xyz_embed)

        # ! densify
        B, N = gaussians_base.shape[:2]  # gaussians upsample factor
        # n_init_points = self.get_xyz.shape[0]

        pos, opacity, scaling, rotation = gaussians_base[
            ..., 0:3], gaussians_base[..., 3:4], gaussians_base[
                ..., 4:7], gaussians_base[..., 7:11]

        # ! filter clone/densify based on scaling range
        """

        # split_mask = scaling.max(
        #     dim=-1
        # )[0] > self.scene_extent * self.percent_dense  # shape: B 4096

        stds = scaling  #  B 13824 3

        # TODO, in mat form. axis aligned creation.
        samples = torch.zeros(B, N, 3, 3).to(stds.device) 

        samples[..., 0,0] = stds[..., 0]
        samples[..., 1,1] = stds[..., 1]
        samples[..., 2,2] = stds[..., 2]

        eye_mat = torch.cat([torch.eye(3), -torch.eye(3)], 0) # 6 * 3, to put gaussians along the axis
        eye_mat = eye_mat.reshape(1,1,6,3).repeat(B, N, 1, 1).to(stds.device)
        samples = (eye_mat @ samples).squeeze(-1) # B N 6 3

        # ! create corner
        samples_corner = stds.clone().unsqueeze(-2).repeat(1,1,8,1) # B N 8 3

        # ! optimize with matmul, register to self
        samples_corner = torch.mul(samples_corner,self.corner_mat)

        samples = torch.cat([samples, samples_corner], -2)

        rots = rearrange(build_rotation(rearrange(rotation, 'B N ... -> (B N) ...')), '(B N) ... -> B N ...', B=B, N=N)
        rots = rots.unsqueeze(2).repeat_interleave(self.f, dim=2) # B f*N 3 3

        new_xyz = (rots @ samples.unsqueeze(-1)).squeeze(-1) + pos.unsqueeze(2).repeat_interleave(self.f, dim=2) # B N 6 3
        new_xyz = rearrange(new_xyz, 'b n f c -> b (n f) c')
        
        # ! new points to features
        new_xyz_embed = xyz_embed_fn(new_xyz)
        new_gaussian_embed = self.write_ca(
            new_xyz_embed, latent_from_vit
        )  # ! use z_(i+1), rather than gaussian_base_feat here

        """

        # ! [global_emb, local_emb, learnable_query_emb] self attention -> fetch last K tokens as the learned query -> add to base

        # ! query from local point emb
        global_local_query_emb = torch.cat(
            [
                # rearrange(latent_from_vit.unsqueeze(1).expand(-1,N,-1,-1), 'B N L C -> (B N) L C'), # 8, 768, 1024. expand() returns a new view.
                rearrange(gaussian_base_feat,
                          'B N C -> (B N) 1 C'),  # 8, 2304, 1024 -> 8*2304 1 C
                self.latent_embedding.repeat(B * N, 1,
                                             1)  # 1, 14, 1024 -> B*N 14 1024
            ],
            dim=1)  # OOM if prepend global feat
        global_local_query_emb = self.transformer(
            global_local_query_emb)  # torch.Size([18432, 15, 1024])
        # st() # do self attention

        # ! query from global shape emb
        # new_gaussian_embed = self.write_ca(
        #     global_local_query_emb,
        #     rearrange(latent_from_vit.unsqueeze(1).expand(-1,N,-1,-1), 'B N L C -> (B N) L C'),
        # )  # ! use z_(i+1), rather than gaussian_base_feat here

        # ! predict gaussians residuals
        gaussian_residual_pre_activate = self.gaussian_residual_pred(
            global_local_query_emb[:, 1:, :])

        gaussian_residual_pre_activate = rearrange(
            gaussian_residual_pre_activate, '(B N) L C -> B N L C', B=B,
            N=N)  # B 2304 14 C
        # TODO here
        # ? new_xyz from where
        offsets = offset_act(gaussian_residual_pre_activate[..., 0:3])
        new_xyz = offsets + pos.unsqueeze(2).repeat_interleave(
            self.f, dim=2)  # B N F 3
        new_xyz = rearrange(new_xyz, 'b n f c -> b (n f) c')

        gaussian_base_pre_activate_repeat = gaussian_base_pre_activate.unsqueeze(
            -2).expand(-1, -1, self.f, -1)  # avoid new memory allocation
        gaussians = rearrange(gaussian_residual_pre_activate +
                              gaussian_base_pre_activate_repeat,
                              'B N F C -> B (N F) C',
                              B=B,
                              N=N)  # learn the residual in the feature space

        # return gaussians, new_xyz, latent_from_vit, new_gaussian_embed  # return positions independently
        # return gaussians, latent_from_vit, new_gaussian_embed  # return positions independently
        return gaussians, new_xyz


class GS_Adaptive_Read_Write_CA_adaptive_2dgs(
        GS_Adaptive_Read_Write_CA_adaptive_f14_prepend):

    def __init__(self,
                 query_dim,
                 context_dim,
                 mlp_ratio,
                 vit_heads,
                 f=16,
                 heads=8,
                 dim_head=64,
                 dropout=0,
                 depth=2,
                 vit_blk=DiTBlock2, 
                 no_flash_op=False,
                 cross_attention=False,) -> None:
        super().__init__(query_dim, context_dim, mlp_ratio, vit_heads, f,
                         heads, dim_head, dropout, depth, vit_blk, no_flash_op)

        # del self.gaussian_residual_pred # will use base one

        self.cross_attention = cross_attention
        if cross_attention: # since much efficient than self attention, linear complexity
            # del self.transformer
            self.sr_ca = CrossAttention(query_dim, context_dim, # xformers fails large batch size: https://github.com/facebookresearch/xformers/issues/845
                                                      heads, dim_head, dropout, 
                                                      no_flash_op=no_flash_op)

        # predict residual over base (features)
        self.gaussian_residual_pred = PreNorm(  # add prenorm since using pre-norm TX as the sr module
            query_dim, nn.Linear(query_dim, 13, bias=True))

        # init as full zero, since predicting residual here
        nn.init.constant_(self.gaussian_residual_pred.fn.weight, 0)
        nn.init.constant_(self.gaussian_residual_pred.fn.bias, 0)

    def forward(self,
                latent_from_vit,
                base_gaussians,
                skip_weight,
                offset_act,
                gs_pred_fn,
                gs_act_fn,
                gaussian_base_pre_activate=None):
        B, N, C = latent_from_vit.shape  # e.g., B 768 768

        if not self.cross_attention:
            # ! query from local point emb
            global_local_query_emb = torch.cat(
                [
                    rearrange(latent_from_vit,
                            'B N C -> (B N) 1 C'),  # 8, 2304, 1024 -> 8*2304 1 C
                    self.latent_embedding.repeat(B * N, 1, 1).to(
                        latent_from_vit)  # 1, 14, 1024 -> B*N 14 1024
                ],
                dim=1)  # OOM if prepend global feat

            global_local_query_emb = self.transformer(
                global_local_query_emb)  # torch.Size([18432, 15, 1024])

            # ! add residuals to the base features
            global_local_query_emb = rearrange(global_local_query_emb[:, 1:],
                                            '(B N) L C -> B N L C',
                                            B=B,
                                            N=N)  # B N C f
        else:

            # st()
            # for xformers debug
            # global_local_query_emb = self.sr_ca( self.latent_embedding.repeat(B, 1, 1).to( latent_from_vit).contiguous(), latent_from_vit[:, 0:1, :],)
            # st()

            # self.sr_ca( self.latent_embedding.repeat(B * N, 1, 1).to(latent_from_vit)[:8000], rearrange(latent_from_vit, 'B N C -> (B N) 1 C')[:8000],).shape
            global_local_query_emb = self.sr_ca( self.latent_embedding.repeat(B * N, 1, 1).to(latent_from_vit), rearrange(latent_from_vit, 'B N C -> (B N) 1 C'),)

            global_local_query_emb = self.transformer(
                global_local_query_emb)  # torch.Size([18432, 15, 1024])

            # ! add residuals to the base features
            global_local_query_emb = rearrange(global_local_query_emb,
                                            '(B N) L C -> B N L C',
                                            B=B,
                                            N=N)  # B N C f

        # * predict residual features
        gaussian_residual_pre_activate = self.gaussian_residual_pred(
            global_local_query_emb)

        # ! directly add xyz offsets 
        offsets = offset_act(gaussian_residual_pre_activate[..., :3])

        gaussians_upsampled_pos = offsets + einops.repeat(
            base_gaussians[..., :3], 'B N C -> B N F C',
            F=self.f)  # ! reasonable init

        # ! add residual features
        gaussian_residual_pre_activate = gaussian_residual_pre_activate + einops.repeat(
            gaussian_base_pre_activate, 'B N C -> B N F C', F=self.f)

        gaussians_upsampled = gs_act_fn(pos=gaussians_upsampled_pos,
                                        x=gaussian_residual_pre_activate)

        gaussians_upsampled = rearrange(gaussians_upsampled,
                                        'B N F C -> B (N F) C')

        return gaussians_upsampled, (rearrange(
            gaussian_residual_pre_activate, 'B N F C -> B (N F) C'
        ), rearrange(
            global_local_query_emb, 'B N F C -> B (N F) C'
        ))


class ViTTriplaneDecomposed(nn.Module):

    def __init__(
        self,
        vit_decoder,
        triplane_decoder: Triplane,
        cls_token=False,
        decoder_pred_size=-1,
        unpatchify_out_chans=-1,
        sr_ratio=2,
    ) -> None:
        super().__init__()
        self.superresolution = None

        self.decomposed_IN = False

        self.decoder_pred_3d = None
        self.transformer_3D_blk = None
        self.logvar = None

        self.cls_token = cls_token
        self.vit_decoder = vit_decoder
        self.triplane_decoder = triplane_decoder
        # triplane_sr_ratio = self.triplane_decoder.triplane_size / self.vit_decoder.img_size
        # self.decoder_pred = nn.Linear(self.vit_decoder.embed_dim,
        #                               self.vit_decoder.patch_size**2 *
        #                               self.triplane_decoder.out_chans,
        #                               bias=True)  # decoder to pat

        # self.patch_size = self.vit_decoder.patch_embed.patch_size
        self.patch_size = 14  # TODO, hard coded here
        if isinstance(self.patch_size, tuple):  # dino-v2
            self.patch_size = self.patch_size[0]

        # self.img_size = self.vit_decoder.patch_embed.img_size
        self.img_size = None  # TODO, hard coded
        if decoder_pred_size == -1:
            decoder_pred_size = self.patch_size**2 * self.triplane_decoder.out_chans

        if unpatchify_out_chans == -1:
            self.unpatchify_out_chans = self.triplane_decoder.out_chans
        else:
            self.unpatchify_out_chans = unpatchify_out_chans

        self.decoder_pred = nn.Linear(
            self.vit_decoder.embed_dim,
            decoder_pred_size,
            #   self.patch_size**2 *
            #   self.triplane_decoder.out_chans,
            bias=True)  # decoder to pat
        # st()

    def triplane_decode(self, latent, c):
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})
        return ret_dict

    def triplane_renderer(self, latent, coordinates, directions):

        planes = latent.view(len(latent), 3,
                             self.triplane_decoder.decoder_in_chans,
                             latent.shape[-2],
                             latent.shape[-1])  # BS 96 256 256

        ret_dict = self.triplane_decoder.renderer.run_model(
            planes, self.triplane_decoder.decoder, coordinates, directions,
            self.triplane_decoder.rendering_kwargs)  # triplane latent -> imgs
        # ret_dict.update({'latent': latent})
        return ret_dict

        # * increase encoded encoded latent dim to match decoder

    def forward_vit_decoder(self, x, img_size=None):
        # latent: (N, L, C) from DINO/CLIP ViT encoder

        # * also dino ViT
        # add positional encoding to each token
        if img_size is None:
            img_size = self.img_size

        if self.cls_token:
            x = x + self.vit_decoder.interpolate_pos_encoding(
                x, img_size, img_size)[:, :]  # B, L, C
        else:
            x = x + self.vit_decoder.interpolate_pos_encoding(
                x, img_size, img_size)[:, 1:]  # B, L, C

        for blk in self.vit_decoder.blocks:
            x = blk(x)
        x = self.vit_decoder.norm(x)

        return x

    def unpatchify(self, x, p=None, unpatchify_out_chans=None):
        """
        x: (N, L, patch_size**2 * self.out_chans)
        imgs: (N, self.out_chans, H, W)
        """
        # st()
        if unpatchify_out_chans is None:
            unpatchify_out_chans = self.unpatchify_out_chans
        # p = self.vit_decoder.patch_size
        if self.cls_token:  # TODO, how to better use cls token
            x = x[:, 1:]

        if p is None:  # assign upsample patch size
            p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, unpatchify_out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], unpatchify_out_chans, h * p,
                                h * p))
        return imgs

    def forward(self, latent, c, img_size):
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        if self.cls_token:
            # latent, cls_token = latent[:, 1:], latent[:, :1]
            cls_token = latent[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        # st()
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # TODO 2D convolutions -> Triplane
        # * triplane rendering
        # ret_dict = self.forward_triplane_decoder(latent,
        #                                          c)  # triplane latent -> imgs
        ret_dict = self.triplane_decoder(planes=latent, c=c)
        ret_dict.update({'latent': latent, 'cls_token': cls_token})

        return ret_dict


# merged above class into a single class

class vae_3d(nn.Module):
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            ldm_z_channels,
            ldm_embed_dim,
            plane_n=1,
            vae_dit_token_size=16,
            **kwargs) -> None:
        super().__init__()

        self.reparameterization_soft_clamp = True  # some instability in training VAE

        # st()
        self.plane_n = plane_n
        self.cls_token = cls_token
        self.vit_decoder = vit_decoder
        self.triplane_decoder = triplane_decoder

        self.patch_size = 14  # TODO, hard coded here
        if isinstance(self.patch_size, tuple):  # dino-v2
            self.patch_size = self.patch_size[0]

        self.img_size = None  # TODO, hard coded

        self.ldm_z_channels = ldm_z_channels
        self.ldm_embed_dim = ldm_embed_dim
        self.vae_p = 4  # resolution = 4 * 16
        self.token_size = vae_dit_token_size  # use dino-v2 dim tradition here
        self.vae_res = self.vae_p * self.token_size

        self.superresolution = nn.ModuleDict({}) # put all the stuffs here
        self.embed_dim = vit_decoder.embed_dim

        # placeholder for compat issue
        self.decoder_pred = None
        self.decoder_pred_3d = None
        self.transformer_3D_blk = None
        self.logvar = None
        self.register_buffer('w_avg', torch.zeros([512]))



    def init_weights(self):
        # ! init (learnable) PE for DiT
        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1, self.vit_decoder.embed_dim,
                        self.vit_decoder.embed_dim),
            requires_grad=True)  # token_size = embed_size by default.
        trunc_normal_(self.vit_decoder.pos_embed, std=.02)


# the base class
class pcd_structured_latent_space_vae_decoder(vae_3d):
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token, **kwargs)
        # from splatting_dit_v4_PI_V1_trilatent_sphere
        self.D_roll_out_input = False

        # ! renderer
        self.gs = triplane_decoder  # compat

        self.rendering_kwargs = self.gs.rendering_kwargs
        self.scene_range = [
            self.rendering_kwargs['sampler_bbox_min'],
            self.rendering_kwargs['sampler_bbox_max']
        ]

        # hyper parameters
        self.skip_weight = torch.tensor(0.1).to(dist_util.dev())

        self.offset_act = lambda x: torch.tanh(x) * (self.scene_range[
            1]) * 0.5  # regularize small offsets

        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1,
                        self.plane_n * (self.token_size**2 + self.cls_token),
                        vit_decoder.embed_dim))
        self.init_weights()  # re-init weights after re-writing token_size

        self.output_size = {
            'gaussians_base': 128,
        }

        # activations
        self.rot_act = lambda x: F.normalize(x, dim=-1)  # as fixed in lgm
        self.scene_extent = self.rendering_kwargs['sampler_bbox_max'] * 0.01
        scaling_factor = (self.scene_extent /
                          F.softplus(torch.tensor(0.0))).to(dist_util.dev())
        self.scale_act = lambda x: F.softplus(
            x
        ) * scaling_factor  # make sure F.softplus(0) is the average scale size
        self.rgb_act = lambda x: 0.5 * torch.tanh(
            x) + 0.5  # NOTE: may use sigmoid if train again
        self.pos_act = lambda x: x.clamp(-0.45, 0.45)
        self.opacity_act = lambda x: torch.sigmoid(x)


        self.superresolution.update(
            dict(
                conv_sr=surfel_prediction(query_dim=vit_decoder.embed_dim),
                quant_conv=Mlp(in_features=2 * self.ldm_z_channels,
                               out_features=2 * self.ldm_embed_dim,
                               act_layer=approx_gelu,
                               drop=0),
                post_quant_conv=Mlp(in_features=self.ldm_z_channels,
                                    out_features=vit_decoder.embed_dim,
                                    act_layer=approx_gelu,
                                    drop=0),
                ldm_upsample=nn.Identity(),
                xyz_pos_embed=nn.Identity(),
            ))

        # for gs prediction
        self.superresolution.update(  # f=14 here
            dict(
                ada_CA_f4_1=GS_Adaptive_Read_Write_CA_adaptive_2dgs(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    # depth=vit_decoder.depth // 6,
                    depth=vit_decoder.depth // 6 if vit_decoder.depth==12 else 2,
                    # f=16,  # 
                    f=8,  # 
                    heads=8),  # write
            ))


    def vae_reparameterization(self, latent, sample_posterior):
        # latent: B 24 32 32

        # assert self.vae_p > 1

        # ! do VAE here
        posterior = self.vae_encode(latent)  # B self.ldm_z_channels 3 L

        assert sample_posterior
        if sample_posterior:
            # torch.manual_seed(0)
            # np.random.seed(0)
            kl_latent = posterior.sample()
        else:
            kl_latent = posterior.mode()  # B C 3 L

        ret_dict = dict(
            latent_normalized=rearrange(kl_latent, 'B C L -> B L C'),
            posterior=posterior,
            query_pcd_xyz=latent['query_pcd_xyz'],
        )

        return ret_dict

    # from pcd_structured_latent_space_lion_learnoffset_surfel_sr_noptVAE.vae_encode
    def vae_encode(self, h):
        # * smooth convolution before triplane
        # B, L, C = h.shape #
        h, query_pcd_xyz = h['h'], h['query_pcd_xyz']
        moments = self.superresolution['quant_conv'](
            h)  # Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), groups=3)

        moments = rearrange(moments,
                            'B L C -> B C L')  # for sd vae code compat

        posterior = DiagonalGaussianDistribution(
            moments, soft_clamp=self.reparameterization_soft_clamp)

        return posterior

    # from pcd_structured_latent_space_lion_learnoffset_surfel_novaePT._get_base_gaussians
    def _get_base_gaussians(self, ret_after_decoder, c=None):
        x = ret_after_decoder['gaussian_base_pre_activate']
        B, N, C = x.shape  # B C D H W, 14-dim voxel features
        assert C == 13  # 2dgs

        offsets = self.offset_act(x[..., 0:3])  # ! model prediction
        # st()
        # vae_sampled_xyz = ret_after_decoder['latent_normalized'][..., :3] # B L C

        vae_sampled_xyz = ret_after_decoder['query_pcd_xyz'].to(
            x.dtype)  # ! directly use fps pcd as "anchor points"

        pos = offsets * self.skip_weight + vae_sampled_xyz  # ! reasonable init

        opacity = self.opacity_act(x[..., 3:4])

        scale = self.scale_act(x[..., 4:6])

        rotation = self.rot_act(x[..., 6:10])
        rgbs = self.rgb_act(x[..., 10:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        return gaussians

    # from pcd_structured_latent_space
    def vit_decode_backbone(self, latent, img_size):
        # assert x.ndim == 3  # N L C
        if isinstance(latent, dict):
            latent = latent['latent_normalized']  # B, C*3, H, W

        latent = self.superresolution['post_quant_conv'](
            latent)  # to later dit embed dim

        # ! directly feed to vit_decoder
        return {
            'latent': latent,
            'latent_from_vit': self.forward_vit_decoder(latent, img_size)
        }  # pred_vit_latent

    # from pcd_structured_latent_space_lion_learnoffset_surfel_sr
    def _gaussian_pred_activations(self, pos, x):
        # if pos is None:
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:6])
        rotation = self.rot_act(x[..., 6:10])
        rgbs = self.rgb_act(x[..., 10:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        return gaussians.float()

    # from pcd_structured_latent_space_lion_learnoffset_surfel_sr
    def vis_gaussian(self, gaussians, file_name_base):
        # gaussians = ret_after_decoder['gaussians']
        # gaussians = ret_after_decoder['latent_after_vit']['gaussians_base']
        B = gaussians.shape[0]
        pos, opacity, scale, rotation, rgbs = gaussians[..., 0:3], gaussians[
            ..., 3:4], gaussians[..., 4:6], gaussians[...,
                                                      6:10], gaussians[...,
                                                                       10:13]
        file_path = Path(logger.get_dir())

        for b in range(B):
            file_name = f'{file_name_base}-{b}'

            np.save(file_path / f'{file_name}_opacity.npy',
                    opacity[b].float().detach().cpu().numpy())
            np.save(file_path / f'{file_name}_scale.npy',
                    scale[b].float().detach().cpu().numpy())
            np.save(file_path / f'{file_name}_rotation.npy',
                    rotation[b].float().detach().cpu().numpy())

            pcu.save_mesh_vc(str(file_path / f'{file_name}.ply'),
                             pos[b].float().detach().cpu().numpy(),
                             rgbs[b].float().detach().cpu().numpy())

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict, return_upsampled_residual=False):
        # from ViT_decode_backbone()

        # latent_from_vit = latent_from_vit['latent_from_vit']
        # vae_sampled_xyz = ret_dict['query_pcd_xyz'].to(latent_from_vit.dtype) # ! directly use fps pcd as "anchor points"
        gaussian_base_pre_activate = self.superresolution['conv_sr'](
            latent_from_vit['latent_from_vit'])  # B 14 H W

        gaussians_base = self._get_base_gaussians(
            {
                # 'latent_from_vit': latent_from_vit,  # latent (vae latent), latent_from_vit (dit)
                # 'ret_dict': ret_dict,
                **ret_dict,
                'gaussian_base_pre_activate':
                gaussian_base_pre_activate,
            }, )

        gaussians_upsampled, (gaussian_upsampled_residual_pre_activate, upsampled_global_local_query_emb) = self.superresolution['ada_CA_f4_1'](
            latent_from_vit['latent_from_vit'],
            gaussians_base,
            skip_weight=self.skip_weight,
            gs_pred_fn=self.superresolution['conv_sr'],
            gs_act_fn=self._gaussian_pred_activations,
            offset_act=self.offset_act,
            gaussian_base_pre_activate=gaussian_base_pre_activate)

        ret_dict.update({
            'gaussians_upsampled': gaussians_upsampled,
            'gaussians_base': gaussians_base
        })  #

        if return_upsampled_residual:
            return ret_dict, (gaussian_upsampled_residual_pre_activate, upsampled_global_local_query_emb)
        else:
            return ret_dict

    def vit_decode(self, latent, img_size, sample_posterior=True, c=None):

        ret_dict = self.vae_reparameterization(latent, sample_posterior)

        latent = self.vit_decode_backbone(ret_dict, img_size)
        ret_after_decoder = self.vit_decode_postprocess(latent, ret_dict)

        return self.forward_gaussians(ret_after_decoder, c=c)

    # from pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr.forward_gaussians
    def forward_gaussians(self, ret_after_decoder, c=None):

        # ! currently, only using upsampled gaussians for training.

        # if True:
        if False:
            ret_after_decoder['gaussians'] = torch.cat([
                ret_after_decoder['gaussians_base'],
                ret_after_decoder['gaussians_upsampled'],
            ],
                                                       dim=1)
        else:  # only adopt SR
            # ! random drop out requires
            ret_after_decoder['gaussians'] = ret_after_decoder[
                'gaussians_upsampled']
            # ret_after_decoder['gaussians'] = ret_after_decoder['gaussians_base']
            pass  # directly use base. vis first.

        ret_after_decoder.update({
            'gaussians': ret_after_decoder['gaussians'],
            'pos': ret_after_decoder['gaussians'][..., :3],
            'gaussians_base_opa': ret_after_decoder['gaussians_base'][..., 3:4]
        })

        # st()
        # self.vis_gaussian(ret_after_decoder['gaussians'], 'sr-8')
        # self.vis_gaussian(ret_after_decoder['gaussians_base'], 'sr-8-base')
        # pcu.save_mesh_v(f'{Path(logger.get_dir())}/anchor-fps-8.ply',ret_after_decoder['query_pcd_xyz'][0].float().detach().cpu().numpy())
        # st()

        # ! render at L:8414 triplane_decode()
        return ret_after_decoder

    def forward_vit_decoder(self, x, img_size=None):
        return self.vit_decoder(x)

    # from pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr_cascade.triplane_decode
    def triplane_decode(self,
                        ret_after_gaussian_forward,
                        c,
                        bg_color=None, 
                        render_all_scale=False,
                        **kwargs):
        # ! render multi-res img with different gaussians

        def render_gs(gaussians, c_data, output_size):

            results = self.gs.render(
                gaussians,  #  type: ignore
                c_data['cam_view'],
                c_data['cam_view_proj'],
                c_data['cam_pos'],
                tanfov=c_data['tanfov'],
                bg_color=bg_color,
                output_size=output_size,
            )

            results['image_raw'] = results[
                'image'] * 2 - 1  # [0,1] -> [-1,1], match tradition
            results['image_depth'] = results['depth']
            results['image_mask'] = results['alpha']

            return results

        cascade_splatting_results = {}

        # for gaussians_key in ('gaussians_base', 'gaussians_upsampled'):
        all_keys_to_render = list(self.output_size.keys())

    
        if self.rand_base_render and not render_all_scale:
            keys_to_render = [random.choice(all_keys_to_render[:-1])] + [all_keys_to_render[-1]]
        else:
            keys_to_render = all_keys_to_render

        for gaussians_key in keys_to_render:
            cascade_splatting_results[gaussians_key] = render_gs(ret_after_gaussian_forward[gaussians_key], c, self.output_size[gaussians_key])

        return cascade_splatting_results


class pcd_structured_latent_space_vae_decoder_cascaded(pcd_structured_latent_space_vae_decoder):
    # for 2dgs

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token, **kwargs)

        self.output_size.update(
            {
                'gaussians_upsampled': 256,
                'gaussians_upsampled_2': 384,
                'gaussians_upsampled_3': 512,
            }
        ) 
                        
        self.rand_base_render = True

        # further x8 up-sampling.
        self.superresolution.update(  
            dict(
                ada_CA_f4_2=GS_Adaptive_Read_Write_CA_adaptive_2dgs(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    # depth=vit_decoder.depth // 6,
                    depth=1,
                    f=4,  # 
                    heads=8, 
                    no_flash_op=True,  # fails when bs>1
                    cross_attention=False),  # write
                ada_CA_f4_3=GS_Adaptive_Read_Write_CA_adaptive_2dgs(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    # depth=vit_decoder.depth // 6,
                    depth=1,
                    f=3,  # 
                    heads=8, 
                    no_flash_op=True, 
                    cross_attention=False),  # write
            ),
        )



    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        # further x8 using upper class
        # TODO, merge this into ln3diff open sourced code.
        ret_dict, (gaussian_upsampled_residual_pre_activate, upsampled_global_local_query_emb) = super().vit_decode_postprocess(latent_from_vit, ret_dict, return_upsampled_residual=True)

        gaussians_upsampled_2, (gaussian_upsampled_residual_pre_activate_2, upsampled_global_local_query_emb_2) = self.superresolution['ada_CA_f4_2'](
            upsampled_global_local_query_emb,
            ret_dict['gaussians_upsampled'],
            skip_weight=self.skip_weight,
            gs_pred_fn=self.superresolution['conv_sr'],
            gs_act_fn=self._gaussian_pred_activations,
            offset_act=self.offset_act,
            gaussian_base_pre_activate=gaussian_upsampled_residual_pre_activate)


        gaussians_upsampled_3, _ = self.superresolution['ada_CA_f4_3'](
            upsampled_global_local_query_emb_2,
            gaussians_upsampled_2,
            skip_weight=self.skip_weight,
            gs_pred_fn=self.superresolution['conv_sr'],
            gs_act_fn=self._gaussian_pred_activations,
            offset_act=self.offset_act,
            gaussian_base_pre_activate=gaussian_upsampled_residual_pre_activate_2)


        ret_dict.update({
            'gaussians_upsampled_2': gaussians_upsampled_2,
            'gaussians_upsampled_3': gaussians_upsampled_3,
        }) 

        return ret_dict

