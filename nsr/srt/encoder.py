from calendar import c
import torchvision
import random
# import einops
import kornia
import einops
import numpy as np
import torch
import torch.nn as nn
from .layers import RayEncoder, Transformer, PreNorm
from pdb import set_trace as st

from pathlib import Path
import math
from ldm.modules.attention import MemoryEfficientCrossAttention
from timm.models.vision_transformer import PatchEmbed
from ldm.modules.diffusionmodules.model import Encoder
from guided_diffusion import dist_util, logger
import point_cloud_utils as pcu

import pytorch3d.ops

from pytorch3d.ops.utils import masked_gather
from timm.models.vision_transformer import PatchEmbed, Mlp

from vit.vit_triplane import XYZPosEmbed

from utils.geometry import index, perspective


def approx_gelu():
    return nn.GELU(approximate="tanh")


class SRTConvBlock(nn.Module):

    def __init__(self, idim, hdim=None, odim=None):
        super().__init__()
        if hdim is None:
            hdim = idim

        if odim is None:
            odim = 2 * hdim

        conv_kwargs = {'bias': False, 'kernel_size': 3, 'padding': 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs), nn.ReLU(),
            nn.Conv2d(hdim, odim, stride=2, **conv_kwargs), nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class SRTEncoder(nn.Module):
    """ Scene Representation Transformer Encoder, as presented in the SRT paper at CVPR 2022 (caveats below)"""

    def __init__(self,
                 num_conv_blocks=4,
                 num_att_blocks=10,
                 pos_start_octave=0,
                 scale_embeddings=False):
        super().__init__()
        self.ray_encoder = RayEncoder(pos_octaves=15,
                                      pos_start_octave=pos_start_octave,
                                      ray_octaves=15)

        conv_blocks = [SRTConvBlock(idim=183, hdim=96)]
        cur_hdim = 192
        for i in range(1, num_conv_blocks):
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=None))
            cur_hdim *= 2

        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.per_patch_linear = nn.Conv2d(cur_hdim, 768, kernel_size=1)

        # Original SRT initializes with stddev=1/math.sqrt(d).
        # But model initialization likely also differs between torch & jax, and this worked, so, eh.
        embedding_stdev = (1. / math.sqrt(768)) if scale_embeddings else 1.
        self.pixel_embedding = nn.Parameter(
            torch.randn(1, 768, 15, 20) * embedding_stdev)
        self.canonical_camera_embedding = nn.Parameter(
            torch.randn(1, 1, 768) * embedding_stdev)
        self.non_canonical_camera_embedding = nn.Parameter(
            torch.randn(1, 1, 768) * embedding_stdev)

        # SRT as in the CVPR paper does not use actual self attention, but a special type:
        # the current features in the Nth layer don't self-attend, but they
        # always attend into the initial patch embedding (i.e., the output of
        # the CNN). SRT further used post-normalization rather than
        # pre-normalization.  Since then though, in OSRT, pre-norm and regular
        # self-attention was found to perform better overall.  So that's what
        # we do here, though it may be less stable under some circumstances.
        self.transformer = Transformer(768,
                                       depth=num_att_blocks,
                                       heads=12,
                                       dim_head=64,
                                       mlp_dim=1536,
                                       selfatt=True)

    def forward(self, images, camera_pos, rays):
        """
        Args:
            images: [batch_size, num_images, 3, height, width].
                Assume the first image is canonical - shuffling happens in the data loader.
            camera_pos: [batch_size, num_images, 3]
            rays: [batch_size, num_images, height, width, 3]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """

        batch_size, num_images = images.shape[:2]

        x = images.flatten(0, 1)
        camera_pos = camera_pos.flatten(0, 1)
        rays = rays.flatten(0, 1)

        canonical_idxs = torch.zeros(batch_size, num_images)
        canonical_idxs[:, 0] = 1
        canonical_idxs = canonical_idxs.flatten(
            0, 1).unsqueeze(-1).unsqueeze(-1).to(x)
        camera_id_embedding = canonical_idxs * self.canonical_camera_embedding + \
                (1. - canonical_idxs) * self.non_canonical_camera_embedding

        ray_enc = self.ray_encoder(camera_pos, rays)
        x = torch.cat((x, ray_enc), 1)
        x = self.conv_blocks(x)
        x = self.per_patch_linear(x)
        height, width = x.shape[2:]
        x = x + self.pixel_embedding[:, :, :height, :width]
        x = x.flatten(2, 3).permute(0, 2, 1)
        x = x + camera_id_embedding

        patches_per_image, channels_per_patch = x.shape[1:]
        x = x.reshape(batch_size, num_images * patches_per_image,
                      channels_per_patch)

        x = self.transformer(x)

        return x


class ImprovedSRTEncoder(nn.Module):
    """
    Scene Representation Transformer Encoder with the improvements from Appendix A.4 in the OSRT paper.
    """

    def __init__(self,
                 num_conv_blocks=3,
                 num_att_blocks=5,
                 pos_start_octave=0):
        super().__init__()
        self.ray_encoder = RayEncoder(pos_octaves=15,
                                      pos_start_octave=pos_start_octave,
                                      ray_octaves=15)

        conv_blocks = [SRTConvBlock(idim=183, hdim=96)]
        cur_hdim = 192
        for i in range(1, num_conv_blocks):
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=None))
            cur_hdim *= 2

        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.per_patch_linear = nn.Conv2d(cur_hdim, 768, kernel_size=1)

        self.transformer = Transformer(768,
                                       depth=num_att_blocks,
                                       heads=12,
                                       dim_head=64,
                                       mlp_dim=1536,
                                       selfatt=True)

    def forward(self, images, camera_pos, rays):
        """
        Args:
            images: [batch_size, num_images, 3, height, width]. Assume the first image is canonical.
            camera_pos: [batch_size, num_images, 3]
            rays: [batch_size, num_images, height, width, 3]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """

        batch_size, num_images = images.shape[:2]

        x = images.flatten(0, 1)
        camera_pos = camera_pos.flatten(0, 1)
        rays = rays.flatten(0, 1)

        ray_enc = self.ray_encoder(camera_pos, rays)
        x = torch.cat((x, ray_enc), 1)
        x = self.conv_blocks(x)
        x = self.per_patch_linear(x)
        x = x.flatten(2, 3).permute(0, 2, 1)

        patches_per_image, channels_per_patch = x.shape[1:]
        x = x.reshape(batch_size, num_images * patches_per_image,
                      channels_per_patch)

        x = self.transformer(x)

        return x


class ImprovedSRTEncoderVAE(nn.Module):
    """
    Modified from ImprovedSRTEncoder
    1. replace conv_blocks to timm embedder 
    2. replace ray_PE with Plucker coordinate
    3. add xformers/flash for transformer attention
    """

    def __init__(
            self,
            *,
            ch,
            out_ch,
            ch_mult=(1, 2, 4, 8),
            num_res_blocks,
            attn_resolutions,
            dropout=0.0,
            resamp_with_conv=True,
            in_channels,
            resolution,
            z_channels,
            double_z=True,
            num_frames=4,
            num_att_blocks=5,
            tx_dim=768,
            num_heads=12,
            mlp_ratio=2,  # denoted by srt
            patch_size=16,
            decomposed=False,
            **kwargs):
        super().__init__()
        # self.ray_encoder = RayEncoder(pos_octaves=15, pos_start_octave=pos_start_octave,
        #                               ray_octaves=15)

        # conv_blocks = [SRTConvBlock(idim=183, hdim=96)]
        # cur_hdim = 192
        # for i in range(1, num_conv_blocks):
        #     conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=None))
        #     cur_hdim *= 2
        self.num_frames = num_frames
        self.embed_dim = tx_dim
        self.embedder = PatchEmbed(
            img_size=256,
            patch_size=patch_size,
            # patch_size=8, # compare the performance
            in_chans=in_channels,
            embed_dim=self.embed_dim,
            norm_layer=None,
            flatten=True,
            bias=True,
        )  # downsample f=16 here.

        # same configuration as vit-B
        if not decomposed:
            self.transformer = Transformer(
                self.embed_dim,  # 12 * 64 = 768
                depth=num_att_blocks,
                heads=num_heads,
                mlp_dim=mlp_ratio * self.embed_dim,  # 1536 by default
            )
        else:
            self.transformer_selfattn = Transformer(
                self.embed_dim,  # 12 * 64 = 768
                depth=1,
                heads=num_heads,
                mlp_dim=mlp_ratio * self.embed_dim,  # 1536 by default
            )
            self.transformer = Transformer(
                self.embed_dim,  # 12 * 64 = 768
                # depth=num_att_blocks-1,
                depth=num_att_blocks,
                heads=num_heads,
                mlp_dim=mlp_ratio * self.embed_dim,  # 1536 by default
            )

        # to a compact latent, with CA
        # query_dim = 4*(1+double_z)
        query_dim = 12 * (1 + double_z
                          )  # for high-quality 3D encoding, follow direct3D
        self.latent_embedding = nn.Parameter(
            torch.randn(1, 32 * 32 * 3, query_dim))
        self.readout_ca = MemoryEfficientCrossAttention(
            query_dim,
            self.embed_dim,
        )

    def forward_tx(self, x):
        x = self.transformer(x)  # B VL C

        # ? 3DPE
        x = self.readout_ca(self.latent_embedding.repeat(x.shape[0], 1, 1), x)

        # ! reshape to 3D latent here. how to make the latent 3D-aware? Later. Performance first.
        x = einops.rearrange(x, 'B (N H W) C -> B C (N H) W', H=32, W=32, N=3)
        return x

    def forward(self, x, **kwargs):
        """
        Args:
            images: [batch_size, num_images, 3, height, width]. Assume the first image is canonical.
            camera_pos: [batch_size, num_images, 3]
            rays: [batch_size, num_images, height, width, 3]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """

        x = self.embedder(x)  # B L C
        x = einops.rearrange(x, '(B V) L C -> B (V L) C', V=self.num_frames)
        x = self.forward_tx(x)

        return x


# ! ablation the srt design
class ImprovedSRTEncoderVAE_K8(ImprovedSRTEncoderVAE):

    def __init__(self, **kwargs):
        super().__init__(patch_size=8, **kwargs)


class ImprovedSRTEncoderVAE_L6(ImprovedSRTEncoderVAE):

    def __init__(self, **kwargs):
        super().__init__(num_att_blocks=6, **kwargs)


class ImprovedSRTEncoderVAE_L5_vitl(ImprovedSRTEncoderVAE):

    def __init__(self, **kwargs):
        super().__init__(num_att_blocks=5, tx_dim=1024, num_heads=16, **kwargs)


class ImprovedSRTEncoderVAE_mlp_ratio4(ImprovedSRTEncoderVAE
                                       ):  # ! by default now

    def __init__(self, **kwargs):
        super().__init__(mlp_ratio=4, **kwargs)


class ImprovedSRTEncoderVAE_mlp_ratio4_decomposed(
        ImprovedSRTEncoderVAE_mlp_ratio4):

    def __init__(self, **kwargs):
        super().__init__(decomposed=True, **kwargs)  # just decompose tx

    def forward(self, x, **kwargs):
        """
        Args:
            images: [batch_size, num_images, 3, height, width]. Assume the first image is canonical.
            camera_pos: [batch_size, num_images, 3]
            rays: [batch_size, num_images, height, width, 3]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """

        x = self.embedder(x)  # B L C
        # x = einops.rearrange(x, '(B V) L C -> B (V L) C', V=self.num_frames)
        x = self.transformer_selfattn(x)
        x = einops.rearrange(x, '(B V) L C -> B (V L) C', V=self.num_frames)
        x = self.forward_tx(x)

        return x


class ImprovedSRTEncoderVAE_mlp_ratio4_f8(ImprovedSRTEncoderVAE):

    def __init__(self, **kwargs):
        super().__init__(mlp_ratio=4, patch_size=8, **kwargs)


class ImprovedSRTEncoderVAE_mlp_ratio4_f8_L6(ImprovedSRTEncoderVAE):

    def __init__(self, **kwargs):
        super().__init__(mlp_ratio=4, patch_size=8, num_att_blocks=6, **kwargs)


class ImprovedSRTEncoderVAE_mlp_ratio4_L6(ImprovedSRTEncoderVAE):

    def __init__(self, **kwargs):
        super().__init__(mlp_ratio=4, num_att_blocks=6, **kwargs)


# ! an SD VAE with one SRT attention + one CA attention for KL
class HybridEncoder(Encoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # st()
        self.srt = ImprovedSRTEncoderVAE(
            **kwargs,
            #  num_frames=4,
            num_att_blocks=1,  # only one layer required
            tx_dim=self.conv_out.weight.shape[1],
            num_heads=8,  # 256  / 64
            mlp_ratio=4,  # denoted by srt
            #  patch_size=16,
        )
        del self.srt.embedder  # use original
        self.conv_out = nn.Identity()

    def forward(self, x, **kwargs):
        x = super().forward(x)
        x = einops.rearrange(x,
                             '(B V) C H W -> B (V H W) C',
                             V=self.srt.num_frames)
        x = self.srt.forward_tx(x)
        return x


class ImprovedSRTEncoderVAE_mlp_ratio4_heavyPatchify(ImprovedSRTEncoderVAE):

    def __init__(self, **kwargs):
        super().__init__(mlp_ratio=4, **kwargs)
        del self.embedder

        conv_blocks = [SRTConvBlock(idim=10, hdim=48)]  # match the ViT-B dim
        cur_hdim = 48 * 2
        for i in range(1,
                       4):  # f=16 still. could reduce attention layers by one?
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=None))
            cur_hdim *= 2

        self.embedder = nn.Sequential(*conv_blocks)

    def forward(self, x, **kwargs):
        """
        Args:
            images: [batch_size, num_images, 3, height, width]. Assume the first image is canonical.
            camera_pos: [batch_size, num_images, 3]
            rays: [batch_size, num_images, height, width, 3]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """

        x = self.embedder(x)  # B C H W
        x = einops.rearrange(x,
                             '(B V) C H W -> B (V H W) C',
                             V=self.num_frames)
        x = self.transformer(x)  # B VL C

        # ? 3DPE
        x = self.readout_ca(self.latent_embedding.repeat(x.shape[0], 1, 1), x)

        # ! reshape to 3D latent here. how to make the latent 3D-aware? Later. Performance first.
        x = einops.rearrange(x, 'B (N H W) C -> B C (N H) W', H=32, W=32, N=3)

        return x


class HybridEncoderPCDStructuredLatent(Encoder):

    def __init__(self, num_frames, latent_num=768, **kwargs):
        super().__init__(**kwargs)
        # st()
        self.num_frames = num_frames
        tx_dim = self.conv_out.weight.shape[1]  # after encoder mid_layers
        self.srt = ImprovedSRTEncoderVAE(
            **kwargs,
            #  num_frames=4,
            num_att_blocks=3,  # only one layer required
            tx_dim=tx_dim,
            num_heads=8,  # 256  / 64
            mlp_ratio=4,  # denoted by srt
        )
        del self.srt.embedder, self.srt.readout_ca, self.srt.latent_embedding  # use original

        # self.box_pool2d = kornia.filters.BlurPool2D(kernel_size=(8,8), stride=8)
        self.box_pool2d = kornia.filters.BlurPool2D(kernel_size=(8, 8),
                                                    stride=8)
        # self.pool2d = kornia.filters.MedianBlur(kernel_size=(8,8), stride=8)
        self.agg_ca = MemoryEfficientCrossAttention(
            tx_dim,
            tx_dim,
            qk_norm=True,  # as in vit-22B
        )
        self.spatial_token_reshape = lambda x: einops.rearrange(
            x, '(B V) C H W -> B (V H W) C', V=self.num_frames)
        self.latent_num = latent_num  # 768 * 3 by default
        self.xyz_pos_embed = XYZPosEmbed(tx_dim)

        # ! VAE part
        self.conv_out = nn.Identity()
        self.Mlp_out = PreNorm(
            tx_dim,  # ! add PreNorm before VAE reduction, stablize training.
            Mlp(
                in_features=tx_dim,  # reduce dim
                hidden_features=tx_dim,
                out_features=self.z_channels * 2,  # double_z
                act_layer=approx_gelu,
                drop=0))
        self.ca_no_pcd = False
        self.pixel_aligned_query = False

    # def _process_token_xyz(self, token_xyz, h):
    #     # pad zero xyz points to reasonable value.

    #     nonzero_mask = (token_xyz != 0).all(dim=2)  # Shape: (B, N)
    #     non_zero_token_xyz = token_xyz[nonzero_mask]
    #     non_zero_token_h = h[nonzero_mask]

    #     # for loop to get foreground points of each instance
    #     # TODO, accelerate with vmap
    #     # No, directly use sparse pcd as input as surface points? fps sampling 768 from 4096 points.
    #     # All points here should not have 0 xyz.
    #     # fg_token_xyz = []
    #     # for idx in range(token_xyz.shape[1]):

    #     fps_xyz, fps_idx = pytorch3d.ops.sample_farthest_points(
    #         non_zero_token_xyz, K=self.latent_num) # B self.latent_num
    #     # pcu.save_mesh_v(f'xyz.ply', xyz[0].float().detach().permute(1,2,0).reshape(-1,3).cpu().numpy(),) # check result first, before fps sampling
    #     # pcu.save_mesh_v(f'fps_xyz.ply', fps_xyz[0].float().detach().reshape(-1,3).cpu().numpy(),) # check result first, before fps sampling
    #     pcu.save_mesh_v(f'token_xyz3.ply', token_xyz[0].float().detach().reshape(-1,3).cpu().numpy(),)
    #     # xyz = self.spatial_token_reshape(xyz)
    #     # pcu.save_mesh_v(f'xyz_new.ply', xyz[0].float().detach().reshape(-1,3).cpu().numpy(),)

    #     st()
    #     query_h = masked_gather(non_zero_token_h, fps_idx) # torch.gather with dim expansion

    #     return query_h, fps_xyz

    def _process_token_xyz(self, pcd, pcd_h):
        # ! 16x uniform downsample before FPS.
        # rand_start_pt = random.randint(0,16)
        # query_pcd_xyz, fps_idx = pytorch3d.ops.sample_farthest_points(
        #     pcd[:, rand_start_pt::16], K=self.latent_num, random_start_point=True) # B self.latent_num
        # query_pcd_h = masked_gather(pcd_h[:, rand_start_pt::16], fps_idx) # torch.gather with dim expansion

        # ! fps very slow on high-res pcd
        query_pcd_xyz, fps_idx = pytorch3d.ops.sample_farthest_points(
            pcd, K=self.latent_num,
            # random_start_point=False)  # B self.latent_num
            random_start_point=True)  # B self.latent_num
        query_pcd_h = masked_gather(pcd_h,
                                    fps_idx)  # torch.gather with dim expansion

        # pcu.save_mesh_v(f'xyz.ply', xyz[0].float().detach().permute(1,2,0).reshape(-1,3).cpu().numpy(),) # check result first, before fps sampling
        # pcu.save_mesh_v(f'fps_xyz.ply', fps_xyz[0].float().detach().reshape(-1,3).cpu().numpy(),) # check result first, before fps sampling
        # pcu.save_mesh_v(f'query_pcd_xyz.ply', query_pcd_xyz[0].float().detach().reshape(-1,3).cpu().numpy(),)
        # pcu.save_mesh_v(f'pcd_xyz.ply', pcd[0].float().detach().reshape(-1,3).cpu().numpy(),)
        # xyz = self.spatial_token_reshape(xyz)
        # pcu.save_mesh_v(f'xyz_new.ply', xyz[0].float().detach().reshape(-1,3).cpu().numpy(),)

        return query_pcd_h, query_pcd_xyz

    def forward(self, x, pcd, **kwargs):

        # def forward(self, x, num_frames=None):
        assert x.shape[1] == 15  # rgb(3),normal(3),plucker_ray(6),xyz(3)
        xyz = x[:, -3:, ...]  # for fps downsampling

        # 0. retrieve VAE tokens
        h = super().forward(
            x, num_frames=self.num_frames
        )  # ! support data augmentation, different FPS different latent corresponding to the same instance?

        # st()
        # pcu.save_mesh_v(f'{Path(logger.get_dir())}/anchor_all.ply',pcd[0].float().detach().cpu().numpy())

        # ! add 3D PE.
        # 1. unproj 2D tokens to 3D
        token_xyz = xyz[..., 4::8, 4::8]

        if self.pixel_aligned_query:

            # h = self.spatial_token_reshape(h) # V frames merge to a single latent here.
            # h = h + self.xyz_pos_embed(token_xyz) # directly add PE to h here.

            # # ! PE over surface fps-pcd
            # pcd_h = self.xyz_pos_embed(pcd) # directly add PE to h here.

            # 2. fps sampling surface as pcd-structured latent.
            h, query_pcd_xyz = self._process_token_xyz(
                pcd, token_xyz, h, c=kwargs.get('c'),
                x=x)  # aggregate with pixel-aligned operation.

        else:
            token_xyz = self.spatial_token_reshape(token_xyz)
            h = self.spatial_token_reshape(
                h)  # V frames merge to a single latent here.
            h = h + self.xyz_pos_embed(token_xyz)  # directly add PE to h here.

            # ! PE over surface fps-pcd
            pcd_h = self.xyz_pos_embed(pcd)  # directly add PE to h here.

            # 2. fps sampling surface as pcd-structured latent.
            query_pcd_h, query_pcd_xyz = self._process_token_xyz(pcd, pcd_h)

            # 2.5 Cross attention to aggregate from all tokens.
            if self.ca_no_pcd:
                h = self.agg_ca(query_pcd_h, h)
            else:
                h = self.agg_ca(
                    query_pcd_h, torch.cat([h, pcd_h], dim=1)
                )  # cross attend to aggregate info from both vae-h and pcd-h

        # 3. add vit TX (5 layers, concat xyz-PE)
        # h = h + self.xyz_pos_embed(fps_xyz) # TODO, add PE of query pts. directly add to h here.
        h = self.srt.transformer(h)  # B L C

        h = self.Mlp_out(h)  # equivalent to conv_out, 256 -> 8 in sd-VAE
        # h = einops.rearrange(h, 'B L C -> B C L') # for VAE compat

        return {
            'h': h,
            'query_pcd_xyz': query_pcd_xyz
        }  # h_0, point cloud-structured latent space. For VAE later.


class HybridEncoderPCDStructuredLatentUniformFPS(
        HybridEncoderPCDStructuredLatent):

    def __init__(self, num_frames, latent_num=768, **kwargs):
        super().__init__(num_frames, latent_num, **kwargs)
        self.ca_no_pcd = True  # check speed up ratio

    def _process_token_xyz(self, pcd, pcd_h):
        # ! 16x uniform downsample before FPS.
        rand_start_pt = random.randint(0, 16)
        # rand_start_pt = 0
        query_pcd_xyz, fps_idx = pytorch3d.ops.sample_farthest_points(
            # pcd[:, rand_start_pt::16], K=self.latent_num, random_start_point=False) # B self.latent_num
            pcd[:, rand_start_pt::16],
            K=self.latent_num,
            random_start_point=True)  # B self.latent_num
        query_pcd_h = masked_gather(pcd_h[:, rand_start_pt::16],
                                    fps_idx)  # torch.gather with dim expansion
        # st()

        # ! fps very slow on high-res pcd
        # query_pcd_xyz, fps_idx = pytorch3d.ops.sample_farthest_points(
        #     pcd, K=self.latent_num, random_start_point=True) # B self.latent_num
        # query_pcd_h = masked_gather(pcd_h, fps_idx) # torch.gather with dim expansion

        # pcu.save_mesh_v(f'xyz.ply', xyz[0].float().detach().permute(1,2,0).reshape(-1,3).cpu().numpy(),) # check result first, before fps sampling
        # pcu.save_mesh_v(f'fps_xyz.ply', fps_xyz[0].float().detach().reshape(-1,3).cpu().numpy(),) # check result first, before fps sampling
        # pcu.save_mesh_v(f'query_pcd_xyz.ply', query_pcd_xyz[0].float().detach().reshape(-1,3).cpu().numpy(),)
        # pcu.save_mesh_v(f'pcd_xyz.ply', pcd[0].float().detach().reshape(-1,3).cpu().numpy(),)
        # xyz = self.spatial_token_reshape(xyz)
        # pcu.save_mesh_v(f'xyz_new.ply', xyz[0].float().detach().reshape(-1,3).cpu().numpy(),)

        return query_pcd_h, query_pcd_xyz


class HybridEncoderPCDStructuredLatentSNoPCD(HybridEncoderPCDStructuredLatent):

    def __init__(self, num_frames, latent_num=768, **kwargs):
        super().__init__(num_frames, latent_num, **kwargs)
        self.ca_no_pcd = True

# class HybridEncoderPCDStructuredLatentSNoPCDPoolingx2(HybridEncoderPCDStructuredLatentSNoPCD):

#     def __init__(self, num_frames, latent_num=768, **kwargs):
#         super().__init__(num_frames, latent_num, **kwargs)

#     # ! just add x2 pooling to reduce token length
#     def forward(self, x, pcd, **kwargs):

#         # def forward(self, x, num_frames=None):
#         assert x.shape[1] == 15  # rgb(3),normal(3),plucker_ray(6),xyz(3)
#         xyz = x[:, -3:, ...]  # for fps downsampling

#         # 0. retrieve VAE tokens
#         h = super().forward(
#             x, num_frames=self.num_frames
#         )  # ! support data augmentation, different FPS different latent corresponding to the same instance?

#         # st()
#         # pcu.save_mesh_v(f'{Path(logger.get_dir())}/anchor_all.ply',pcd[0].float().detach().cpu().numpy())

#         # ! add 3D PE.
#         # 1. unproj 2D tokens to 3D
#         token_xyz = xyz[..., 4::8, 4::8]

#         if self.pixel_aligned_query:

#             # h = self.spatial_token_reshape(h) # V frames merge to a single latent here.
#             # h = h + self.xyz_pos_embed(token_xyz) # directly add PE to h here.

#             # # ! PE over surface fps-pcd
#             # pcd_h = self.xyz_pos_embed(pcd) # directly add PE to h here.

#             # 2. fps sampling surface as pcd-structured latent.
#             h, query_pcd_xyz = self._process_token_xyz(
#                 pcd, token_xyz, h, c=kwargs.get('c'),
#                 x=x)  # aggregate with pixel-aligned operation.

#         else:
#             token_xyz = self.spatial_token_reshape(token_xyz)
#             h = self.spatial_token_reshape(
#                 h)  # V frames merge to a single latent here.
#             h = h + self.xyz_pos_embed(token_xyz)  # directly add PE to h here.

#             # ! PE over surface fps-pcd
#             pcd_h = self.xyz_pos_embed(pcd)  # directly add PE to h here.

#             # 2. fps sampling surface as pcd-structured latent.
#             query_pcd_h, query_pcd_xyz = self._process_token_xyz(pcd, pcd_h)

#             # 2.5 Cross attention to aggregate from all tokens.
#             if self.ca_no_pcd:
#                 h = self.agg_ca(query_pcd_h, h)
#             else:
#                 h = self.agg_ca(
#                     query_pcd_h, torch.cat([h, pcd_h], dim=1)
#                 )  # cross attend to aggregate info from both vae-h and pcd-h

#         # 3. add vit TX (5 layers, concat xyz-PE)
#         # h = h + self.xyz_pos_embed(fps_xyz) # TODO, add PE of query pts. directly add to h here.
#         h = self.srt.transformer(h)  # B L C

#         h = self.Mlp_out(h)  # equivalent to conv_out, 256 -> 8 in sd-VAE
#         # h = einops.rearrange(h, 'B L C -> B C L') # for VAE compat

#         return {
#             'h': h,
#             'query_pcd_xyz': query_pcd_xyz
#         }  # h_0, point cloud-structured latent space. For VAE later.






class HybridEncoderPCDStructuredLatentSNoPCD_PixelAlignedQuery(
        HybridEncoderPCDStructuredLatent):

    def __init__(self, num_frames, latent_num=768, **kwargs):
        super().__init__(num_frames, latent_num, **kwargs)
        self.ca_no_pcd = True
        self.pixel_aligned_query = True
        self.F = 4  # pixel-aligned query from nearest F views

        del self.agg_ca  # for average pooling now.

    def _pcd_to_homo(self, pcd):
        return torch.cat([pcd, torch.ones_like(pcd[..., 0:1])], -1)

    # ! FPS sampling
    def _process_token_xyz(self, pcd, token_xyz, h, c, x=None):
        V = c['cam_pos'].shape[1]

        # (Pdb) p c.keys()
        # dict_keys(['source_cv2wT_quat', 'cam_view', 'cam_view_proj', 'cam_pos', 'tanfov', 'orig_pose', 'orig_c2w', 'orig_w2c'])
        # (Pdb) p c['cam_view'].shape
        # torch.Size([8, 9, 4, 4])
        # (Pdb) p c['cam_pos'].shape
        # torch.Size([8, 9, 3])

        # ! 16x uniform downsample before FPS.
        # rand_start_pt = random.randint(0,16)
        # query_pcd_xyz, fps_idx = pytorch3d.ops.sample_farthest_points(
        #     pcd[:, rand_start_pt::16], K=self.latent_num, random_start_point=True) # B self.latent_num
        # query_pcd_h = masked_gather(pcd_h[:, rand_start_pt::16], fps_idx) # torch.gather with dim expansion

        # ! fps very slow on high-res pcd, but better.
        # '''
        query_pcd_xyz, fps_idx = pytorch3d.ops.sample_farthest_points(
            pcd, K=self.latent_num, random_start_point=True) # B self.latent_num
        # query_pcd_h = masked_gather(pcd_h, fps_idx) # torch.gather with dim expansion

        # '''
        # ! use unprojected xyz for pixel-aligned projection check

        # query_pcd_xyz = self.spatial_token_reshape(token_xyz)
        B, N = query_pcd_xyz.shape[:2]

        normal = x[:, 3:6, ...]
        normal_map = (normal * 127.5 + 127.5).float().to(
            torch.uint8)  # BV 3 H W

        normal_map = einops.rearrange(normal_map,
                                           '(B V) C H W -> B V C H W',
                                           B=B,
                                           V=V).detach().cpu()  # V C H W
        img_size = normal_map.shape[-1]

        # ! ====== single-view debug here
        for b in range(c['orig_w2c'].shape[0]):
            for V in range(c['orig_w2c'].shape[1]):
                selected_normal = normal_map[b, V]
                proj_point = c['orig_w2c'][b, V] @ self._pcd_to_homo(query_pcd_xyz[b]).permute(1, 0)
                proj_point[:2, ...] /= proj_point[2, ...]
                proj_point[2, ...] = 1 # homo


                intrin = c['orig_intrin'][b, V]
                proj_point = intrin @ proj_point[:3]
                proj_point = proj_point.permute(1,0)[..., :2] # 768 4

                # st()

                # proj_point = c['cam_view_proj'][b, V] @ self._pcd_to_homo(query_pcd_xyz[b]).permute(1, 0)

                # plot proj_point and save
                for uv_idx in range(proj_point.shape[0]):
                    # uv = proj_point[uv_idx] * 127.5 + 127.5
                    # uv = proj_point[uv_idx] * 127.5 + 127.5
                    uv = proj_point[uv_idx] * img_size
                    x, y = int(uv[0].clip(0, img_size)), int(uv[1].clip(0, img_size))
                    selected_normal[:, max(y - 1, 0):min(y + 1, img_size),
                                    max(x - 1, 0):min(x + 1, img_size)] = torch.Tensor([
                                        255, 0, 0
                                    ]).reshape(3, 1, 1).to(selected_normal)  # set to red

                torchvision.utils.save_image(selected_normal.float(),
                                            f'tmp/pifu_normal_{b}_{V}.jpg',
                                            normalize=True,
                                            value_range=(0, 255))
            

            st()
            pass

        st()
        # ! ====== single-view debug done


        # ! project pcd to each views
        batched_query_pcd = einops.repeat(self._pcd_to_homo(query_pcd_xyz),
                                          'B N C -> (B V N) C 1',
                                          V=V)
        batched_cam_view_proj = einops.repeat(c['cam_view_proj'],
                                              'B V H W -> (B V N) H W',
                                              N=N)

        batched_proj_uv = einops.rearrange(
            (batched_cam_view_proj @ batched_query_pcd),
            '(B V N) L 1 -> (B V) L N',
            B=B,
            V=V,
            N=N)  # BV 4 N
        batched_proj_uv = batched_proj_uv[..., :2, :]  # BV N 2

        # draw projected UV coordinate on 2d normal map
        # idx_to_vis = 15 * 32 + 16 # middle of the img
        # idx_to_vis = 16 * 6 + 15 * 32 + 16  # middle of the img
        idx_to_vis = 0 # use fps points here
        # st()

        selected_proj_uv = einops.rearrange(batched_proj_uv,
                                            '(B V) C N -> B V C N',
                                            B=B,
                                            V=V,
                                            N=N)[0, ...,
                                                 idx_to_vis]  # V 2 N -> V 2
        # selected_normal = einops.rearrange(normal_map,
        #                                    '(B V) C H W -> B V C H W',
        #                                    B=B,
        #                                    V=V)[0].detach().cpu()  # V C H W

        for uv_idx in range(selected_proj_uv.shape[0]):
            uv = selected_proj_uv[uv_idx] * 127.5 + 127.5
            x, y = int(uv[0].clip(0, 255)), int(uv[1].clip(0, 255))
            selected_normal[uv_idx, :,
                            max(y - 5, 0):min(y + 5, 255),
                            max(x - 5, 0):min(x + 5, 255)] = torch.Tensor([
                                255, 0, 0
                            ]).reshape(3, 1,
                                       1).to(selected_normal)  # set to red
            # selected_normal[uv_idx, :, max(y-5, 0):min(y+5, 255), max(x-5,0):min(x+5,255)] = torch.Tensor([255,0,0]).to(selected_normal) # set to red
        # st()
        torchvision.utils.save_image(selected_normal.float(),
                                     'pifu_normal.jpg',
                                     normalize=True,
                                     value_range=(0, 255))
        st()
        pass

        # ! grid sample
        query_pcd_h = index(
            h, batched_proj_uv)  # h: (B V) C H W, uv: (B V) N 2  -> BV 256 768

        query_pcd_h_to_gather = einops.rearrange(query_pcd_h,
                                                 '(B V) C N -> B N V C',
                                                 B=B,
                                                 V=V,
                                                 N=N)

        # ! find nearest F views
        _, knn_idx, _ = pytorch3d.ops.knn_points(
            query_pcd_xyz, c['cam_pos'], K=self.F,
            return_nn=False)  # knn_idx: B N F
        knn_idx_expanded = knn_idx[..., None].expand(
            -1, -1, -1, query_pcd_h_to_gather.shape[-1])  # B N F -> B N F C
        knn_pcd_h = torch.gather(
            query_pcd_h_to_gather, dim=2,
            index=knn_idx_expanded)  # torch.Size([8, 768, 4, 256])

        # average pooling knn feature.
        query_pcd_h = knn_pcd_h.mean(dim=2)

        # add PE
        pcd_h = self.xyz_pos_embed(query_pcd_xyz)  # pcd_h as PE feature.
        query_pcd_h = query_pcd_h + pcd_h

        # TODO: QKV aggregation with pcd_h as q, query_pcd_h as kv. Requires gather?
        '''not used; binary mask for aggregation.

        # * mask idx not used anymore. torch.gather() instead, more flexible.
        # knn_idx_mask = torch.zeros((B,N,V), device=knn_idx.device)
        # knn_idx_mask.scatter_(dim=2, index=knn_idx, src=torch.ones_like(knn_idx_mask)) # ! B N V

        # try gather
        # gather_idx = einops.rearrange(knn_idx_mask, 'B N V -> B N V 1').bool()

        # query_pcd_h = einops.rearrange(query_pcd_h, "(B V) C N -> B N V C", B=pcd_h.shape[0], N=self.latent_num, V=V) # torch.Size([8, 768, 4, 256])
        # ! apply KNN mask and average the feature.
        # query_pcd_h = einops.reduce(query_pcd_h * knn_idx_mask.unsqueeze(-1), 'B N V C -> B N C', 'sum') / self.F # B 768 256. average pooling aggregated feature, like in pifu.
        '''
        '''
        # pixel-aligned projection, not efficient enough.
        knn_cam_view_proj = pytorch3d.ops.knn_gather(einops.rearrange(c['cam_view_proj'], 'B V H W-> B V (H W)'), knn_idx) # get corresponding cam_view_projection matrix (P matrix)
        knn_cam_view_proj = einops.rearrange(knn_cam_view_proj, 'B N F (H W) -> (B N F) H W', H=4, W=4) # for matmul. H=W=4 here, P matrix.

        batched_query_pcd = einops.repeat(self._pcd_to_homo(query_pcd_xyz), 'B N C -> (B N F) C 1', F=self.F)
        xyz = knn_cam_view_proj @ batched_query_pcd # BNF 4 1

        # st()
        knn_spatial_feat = pytorch3d.ops.knn_gather(einops.rearrange(h, '(B V) C H W -> B V (C H W)', V=self.num_frames), knn_idx) # get corresponding feat for grid_sample
        knn_spatial_feat = einops.rearrange(knn_spatial_feat, 'B N F (C H W) -> (B N F) C H W', C=h.shape[-3], H=h.shape[-2], W=h.shape[-1])
        '''

        # grid_sample
        # https://github.com/shunsukesaito/PIFu/blob/f0a9c99ef887e1eb360e865a87aa5f166231980e/lib/geometry.py#L15

        # average pooling multi-view extracted information

        # return query_pcd_h, query_pcd_xyz
        return query_pcd_h, query_pcd_xyz
