# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from threading import local
import torch
import torch.nn as nn
from torch_utils import persistence
from .networks_stylegan2 import Generator as StyleGAN2Backbone
from .networks_stylegan2 import ToRGBLayer, SynthesisNetwork, MappingNetwork
from .volumetric_rendering.renderer import ImportanceRenderer
from .volumetric_rendering.ray_sampler import RaySampler, PatchRaySampler
import dnnlib
from pdb import set_trace as st
import math

import torch.nn.functional as F
import itertools
from ldm.modules.diffusionmodules.model import SimpleDecoder, Decoder


@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):

    def __init__(
            self,
            z_dim,  # Input latent (Z) dimensionality.
            c_dim,  # Conditioning label (C) dimensionality.
            w_dim,  # Intermediate latent (W) dimensionality.
            img_resolution,  # Output resolution.
            img_channels,  # Number of output color channels.
            sr_num_fp16_res=0,
            mapping_kwargs={},  # Arguments for MappingNetwork.
            rendering_kwargs={},
            sr_kwargs={},
            bcg_synthesis_kwargs={},
            # pifu_kwargs={},
            # ada_kwargs={},  # not used, place holder
            **synthesis_kwargs,  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.renderer = ImportanceRenderer()
        # if 'PatchRaySampler' in rendering_kwargs:
        #     self.ray_sampler = PatchRaySampler()
        # else:
        #     self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim,
                                          c_dim,
                                          w_dim,
                                          img_resolution=256,
                                          img_channels=32 * 3,
                                          mapping_kwargs=mapping_kwargs,
                                          **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(
            class_name=rendering_kwargs['superresolution_module'],
            channels=32,
            img_resolution=img_resolution,
            sr_num_fp16_res=sr_num_fp16_res,
            sr_antialias=rendering_kwargs['sr_antialias'],
            **sr_kwargs)

        # self.bcg_synthesis = None
        if rendering_kwargs.get('use_background', False):
            self.bcg_synthesis = SynthesisNetwork(
                w_dim,
                img_resolution=self.superresolution.input_resolution,
                img_channels=32,
                **bcg_synthesis_kwargs)
            self.bcg_mapping = MappingNetwork(z_dim=z_dim,
                                              c_dim=c_dim,
                                              w_dim=w_dim,
                                              num_ws=self.num_ws,
                                              **mapping_kwargs)
        # New mapping network for self-adaptive camera pose, dim = 3

        self.decoder = OSGDecoder(
            32, {
                'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                'decoder_output_dim': 32
            })
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs

        self._last_planes = None
        self.pool_256 = torch.nn.AdaptiveAvgPool2d((256, 256))

    def mapping(self,
                z,
                c,
                truncation_psi=1,
                truncation_cutoff=None,
                update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z,
                                     c *
                                     self.rendering_kwargs.get('c_scale', 0),
                                     truncation_psi=truncation_psi,
                                     truncation_cutoff=truncation_cutoff,
                                     update_emas=update_emas)

    def synthesis(self,
                  ws,
                  c,
                  neural_rendering_resolution=None,
                  update_emas=False,
                  cache_backbone=False,
                  use_cached_backbone=False,
                  return_meta=False,
                  return_raw_only=False,
                  **synthesis_kwargs):

        return_sampling_details_flag = self.rendering_kwargs.get(
            'return_sampling_details_flag', False)

        if return_sampling_details_flag:
            return_meta = True

        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        # cam2world_matrix = torch.eye(4, device=c.device).unsqueeze(0).repeat_interleave(c.shape[0], dim=0)
        # c[:, :16] = cam2world_matrix.view(-1, 16)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        H = W = self.neural_rendering_resolution
        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(
            cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(
                ws[:, :self.backbone.num_ws, :],  # ws, BS 14 512
                update_emas=update_emas,
                **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2],
                             planes.shape[-1])  # BS 96 256 256

        # Perform volume rendering
        # st()
        rendering_details = self.renderer(
            planes,
            self.decoder,
            ray_origins,
            ray_directions,
            self.rendering_kwargs,
            #   return_meta=True)
            return_meta=return_meta)

        # calibs = create_calib_matrix(c)
        # all_coords = rendering_details['all_coords']
        # B, num_rays, S, _ = all_coords.shape
        # all_coords_B3N = all_coords.reshape(B, -1, 3).permute(0,2,1)
        # homo_coords = torch.cat([all_coords, torch.zeros_like(all_coords[..., :1])], -1)
        # homo_coords[..., -1] = 1
        # homo_coords = homo_coords.reshape(homo_coords.shape[0], -1, 4)
        # homo_coords = homo_coords.permute(0,2,1)
        # xyz = calibs @ homo_coords
        # xyz = xyz.permute(0,2,1).reshape(B, H, W, S, 4)
        # st()

        # xyz_proj = perspective(all_coords_B3N, calibs)
        # xyz_proj = xyz_proj.permute(0,2,1).reshape(B, H, W, S, 3) # [0,0] - [1,1]
        # st()

        feature_samples, depth_samples, weights_samples = (
            rendering_details[k]
            for k in ['feature_samples', 'depth_samples', 'weights_samples'])

        if return_sampling_details_flag:
            shape_synthesized = rendering_details['shape_synthesized']
        else:
            shape_synthesized = None

        # Reshape into 'raw' neural-rendered image
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            N, feature_samples.shape[-1], H, W).contiguous()  # B 32 H W
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]  # B 3 H W
        if not return_raw_only:
            sr_image = self.superresolution(
                rgb_image,
                feature_image,
                ws[:, -1:, :],  # only use the last layer
                noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
                **{
                    k: synthesis_kwargs[k]
                    for k in synthesis_kwargs.keys() if k != 'noise_mode'
                })
        else:
            sr_image = rgb_image

        ret_dict = {
            'image': sr_image,
            'image_raw': rgb_image,
            'image_depth': depth_image,
            'weights_samples': weights_samples,
            'shape_synthesized': shape_synthesized
        }
        if return_meta:
            ret_dict.update({
                # 'feature_image': feature_image,
                'feature_volume':
                rendering_details['feature_volume'],
                'all_coords':
                rendering_details['all_coords'],
                'weights':
                rendering_details['weights'],
            })

        return ret_dict

    def sample(self,
               coordinates,
               directions,
               z,
               c,
               truncation_psi=1,
               truncation_cutoff=None,
               update_emas=False,
               **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes.
        ws = self.mapping(z,
                          c,
                          truncation_psi=truncation_psi,
                          truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        planes = self.backbone.synthesis(ws,
                                         update_emas=update_emas,
                                         **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2],
                             planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates,
                                       directions, self.rendering_kwargs)

    def sample_mixed(self,
                     coordinates,
                     directions,
                     ws,
                     truncation_psi=1,
                     truncation_cutoff=None,
                     update_emas=False,
                     **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws,
                                         update_emas=update_emas,
                                         **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2],
                             planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates,
                                       directions, self.rendering_kwargs)

    def forward(self,
                z,
                c,
                truncation_psi=1,
                truncation_cutoff=None,
                neural_rendering_resolution=None,
                update_emas=False,
                cache_backbone=False,
                use_cached_backbone=False,
                **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z,
                          c,
                          truncation_psi=truncation_psi,
                          truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        return self.synthesis(
            ws,
            c,
            update_emas=update_emas,
            neural_rendering_resolution=neural_rendering_resolution,
            cache_backbone=cache_backbone,
            use_cached_backbone=use_cached_backbone,
            **synthesis_kwargs)


from .networks_stylegan2 import FullyConnectedLayer

# class OSGDecoder(torch.nn.Module):

#     def __init__(self, n_features, options):
#         super().__init__()
#         self.hidden_dim = 64
#         self.output_dim = options['decoder_output_dim']
#         self.n_features = n_features

#         self.net = torch.nn.Sequential(
#             FullyConnectedLayer(n_features,
#                                 self.hidden_dim,
#                                 lr_multiplier=options['decoder_lr_mul']),
#             torch.nn.Softplus(),
#             FullyConnectedLayer(self.hidden_dim,
#                                 1 + options['decoder_output_dim'],
#                                 lr_multiplier=options['decoder_lr_mul']))

#     def forward(self, sampled_features, ray_directions):
#         # Aggregate features
#         sampled_features = sampled_features.mean(1)
#         x = sampled_features

#         N, M, C = x.shape
#         x = x.view(N * M, C)

#         x = self.net(x)
#         x = x.view(N, M, -1)
#         rgb = torch.sigmoid(x[..., 1:]) * (
#             1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
#         sigma = x[..., 0:1]
#         return {'rgb': rgb, 'sigma': sigma}


@persistence.persistent_class
class OSGDecoder(torch.nn.Module):

    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64
        self.decoder_output_dim = options['decoder_output_dim']

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features,
                                self.hidden_dim,
                                lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim,
                                1 + options['decoder_output_dim'],
                                lr_multiplier=options['decoder_lr_mul']))
        self.activation = options.get('decoder_activation', 'sigmoid')

    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N * M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = x[..., 1:]
        sigma = x[..., 0:1]
        if self.activation == "sigmoid":
            # Original EG3D
            rgb = torch.sigmoid(rgb) * (1 + 2 * 0.001) - 0.001
        elif self.activation == "lrelu":
            # StyleGAN2-style, use with toRGB
            rgb = torch.nn.functional.leaky_relu(rgb, 0.2,
                                                 inplace=True) * math.sqrt(2)
        return {'rgb': rgb, 'sigma': sigma}


class OSGDecoderFlexicube(OSGDecoder):
    # https://github.com/TencentARC/InstantMesh/blob/0a64425c6d390afa40128132cec42cd5c6408bbf/src/models/renderer/synthesizer_mesh.py#L15
    def __init__(self, n_features, options, hidden_dim: int = 64, num_layers: int = 4, activation: nn.Module = nn.ReLU):
        super().__init__(n_features, options)

        # self.net_sdf = nn.Sequential(
        #     nn.Linear(3 * n_features, hidden_dim),
        #     activation(),
        #     *itertools.chain(*[[
        #         nn.Linear(hidden_dim, hidden_dim),
        #         activation(),
        #     ] for _ in range(num_layers - 2)]),
        #     nn.Linear(hidden_dim, 1),
        # )
        # self.net_rgb = nn.Sequential(
        #     nn.Linear(3 * n_features, hidden_dim),
        #     activation(),
        #     *itertools.chain(*[[
        #         nn.Linear(hidden_dim, hidden_dim),
        #         activation(),
        #     ] for _ in range(num_layers - 2)]),
        #     nn.Linear(hidden_dim, 3),
        # )

        # ! sdf and rgb prediction adopts old tradition, softplus here
        # TODO, load pre-trained model weights
        self.net_sdf = torch.nn.Sequential(
            FullyConnectedLayer(n_features,
                                self.hidden_dim,
                                lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim,
                                1,
                                lr_multiplier=options['decoder_lr_mul']))

        self.net_rgb = torch.nn.Sequential(
            FullyConnectedLayer(n_features,
                                self.hidden_dim,
                                lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim,
                                options['decoder_output_dim'],
                                lr_multiplier=options['decoder_lr_mul']))


        # ! for following MLP, use new behaviour
        self.net_deformation = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 3),
        )
        self.net_weight = nn.Sequential(
            nn.Linear(8 * 3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 21),
        )

        # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    #  def forward(self, sampled_features, ray_directions):
    #     # Aggregate features
    #     sampled_features = sampled_features.mean(1)
    #     x = sampled_features

    #     N, M, C = x.shape
    #     x = x.view(N * M, C)

    #     x = self.net(x)
    #     x = x.view(N, M, -1)
    #     rgb = x[..., 1:]
    #     sigma = x[..., 0:1]
    #     if self.activation == "sigmoid":
    #         # Original EG3D
    #         rgb = torch.sigmoid(rgb) * (1 + 2 * 0.001) - 0.001
    #     elif self.activation == "lrelu":
    #         # StyleGAN2-style, use with toRGB
    #         rgb = torch.nn.functional.leaky_relu(rgb, 0.2,
    #                                              inplace=True) * math.sqrt(2)
    #     return {'rgb': rgb, 'sigma': sigma}


    def get_geometry_prediction(self, sampled_features, flexicubes_indices):
        _N, n_planes, _M, _C = sampled_features.shape

        sdf = self.net_sdf(sampled_features.mean(1)) # for compat issue
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes*_C)
        deformation = self.net_deformation(sampled_features)

        grid_features = torch.index_select(input=sampled_features, index=flexicubes_indices.reshape(-1), dim=1)
        grid_features = grid_features.reshape(
            sampled_features.shape[0], flexicubes_indices.shape[0], flexicubes_indices.shape[1] * sampled_features.shape[-1])
        weight = self.net_weight(grid_features) * 0.1

        return sdf, deformation, weight
    
    def get_texture_prediction(self, sampled_features):
        _N, n_planes, _M, _C = sampled_features.shape
        # sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes*_C)
        sampled_features = sampled_features.mean(1)

        rgb = self.net_rgb(sampled_features) # sigmoid feat by default
        rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001  # Uses sigmoid clamping from MipNeRF

        return rgb


class LRMOSGDecoder(nn.Module):
    """
    Triplane decoder that gives RGB and sigma values from sampled features.
    Using ReLU here instead of Softplus in the original implementation.
    
    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L112
    """
    def __init__(self, n_features: int,
                 hidden_dim: int = 64, num_layers: int = 4, activation: nn.Module = nn.ReLU):
        super().__init__()
        self.decoder_output_dim = 3
        self.net = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 1 + self.decoder_output_dim),
        )
        # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, sampled_features, ray_directions):
        # Aggregate features by mean
        # sampled_features = sampled_features.mean(1)
        # Aggregate features by concatenation
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes*_C)
        x = sampled_features

        N, M, C = x.shape
        x = x.contiguous().view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]

        return {'rgb': rgb, 'sigma': sigma}


class Triplane(torch.nn.Module):

    def __init__(
        self,
        c_dim=25,  # Conditioning label (C) dimensionality.
        img_resolution=128,  # Output resolution.
        img_channels=3,  # Number of output color channels.
        out_chans=96,
        triplane_size=224,
        rendering_kwargs={},
        decoder_in_chans=32,
        decoder_output_dim=32,
        sr_num_fp16_res=0,
        sr_kwargs={},
        create_triplane=False, # for overfitting single instance study
        bcg_synthesis_kwargs={},
        lrm_decoder=False,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution  # TODO
        self.img_channels = img_channels
        self.triplane_size = triplane_size

        self.decoder_in_chans = decoder_in_chans
        self.out_chans = out_chans

        self.renderer = ImportanceRenderer()

        if 'PatchRaySampler' in rendering_kwargs:
            self.ray_sampler = PatchRaySampler()
        else:
            self.ray_sampler = RaySampler()

        if lrm_decoder:
            self.decoder = LRMOSGDecoder(
                decoder_in_chans,)
        else:
            self.decoder = OSGDecoder(
                decoder_in_chans,
                {
                    'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                    # 'decoder_output_dim': 32
                    'decoder_output_dim': decoder_output_dim
                })

        self.neural_rendering_resolution = img_resolution  # TODO
        # self.neural_rendering_resolution = 128  # TODO
        self.rendering_kwargs = rendering_kwargs
        self.create_triplane = create_triplane
        if create_triplane:
            self.planes = nn.Parameter(torch.randn(1, out_chans, 256, 256))

        if bool(sr_kwargs):  # check whether empty
            assert decoder_in_chans == decoder_output_dim, 'tradition'
            if rendering_kwargs['superresolution_module'] in [
                    'torch_utils.components.PixelUnshuffleUpsample',
                    'torch_utils.components.NearestConvSR',
                    'torch_utils.components.NearestConvSR_Residual'
            ]:
                self.superresolution = dnnlib.util.construct_class_by_name(
                    class_name=rendering_kwargs['superresolution_module'],
                    # * for PixelUnshuffleUpsample
                    sr_ratio=2,  # 2x SR, 128 -> 256
                    output_dim=decoder_output_dim,
                    num_out_ch=3,
                )
            else:
                self.superresolution = dnnlib.util.construct_class_by_name(
                    class_name=rendering_kwargs['superresolution_module'],
                    # * for stylegan upsample
                    channels=decoder_output_dim,
                    img_resolution=img_resolution,
                    sr_num_fp16_res=sr_num_fp16_res,
                    sr_antialias=rendering_kwargs['sr_antialias'],
                    **sr_kwargs)
        else:
            self.superresolution = None

        self.bcg_synthesis = None

    # * pure reconstruction
    def forward(
            self,
            planes=None,
            # img,
            c=None,
            ws=None,
            ray_origins=None,
            ray_directions=None,
            z_bcg=None,
            neural_rendering_resolution=None,
            update_emas=False,
            cache_backbone=False,
            use_cached_backbone=False,
            return_meta=False,
            return_raw_only=False,
            sample_ray_only=False,
            fg_bbox=None,
            **synthesis_kwargs):

        cam2world_matrix = c[:, :16].reshape(-1, 4, 4)
        # cam2world_matrix = torch.eye(4, device=c.device).unsqueeze(0).repeat_interleave(c.shape[0], dim=0)
        # c[:, :16] = cam2world_matrix.view(-1, 16)
        intrinsics = c[:, 16:25].reshape(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        if ray_directions is None:  # when output video
            H = W = self.neural_rendering_resolution
            # Create a batch of rays for volume rendering
            # ray_origins, ray_directions, ray_bboxes = self.ray_sampler(
            #     cam2world_matrix, intrinsics, neural_rendering_resolution)

            if sample_ray_only: # ! for sampling
                ray_origins, ray_directions, ray_bboxes = self.ray_sampler(
                    cam2world_matrix, intrinsics, 
                    self.rendering_kwargs.get( 'patch_rendering_resolution' ),
                    self.neural_rendering_resolution, fg_bbox)

                # for patch supervision
                ret_dict = {
                    'ray_origins': ray_origins,
                    'ray_directions': ray_directions,
                    'ray_bboxes': ray_bboxes,
                }

                return ret_dict

            else: # ! for rendering
                ray_origins, ray_directions, _ = self.ray_sampler(
                    cam2world_matrix, intrinsics, self.neural_rendering_resolution,
                    self.neural_rendering_resolution)

        else:
            assert ray_origins is not None
            H = W = int(ray_directions.shape[1]**
                        0.5)  # dynamically set patch resolution

        # ! match the batch size, if not returned
        if planes is None:
            assert self.planes is not None
            planes = self.planes.repeat_interleave(c.shape[0], dim=0)
        return_sampling_details_flag = self.rendering_kwargs.get(
            'return_sampling_details_flag', False)

        if return_sampling_details_flag:
            return_meta = True

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape

        # Reshape output into three 32-channel planes
        if planes.shape[1] == 3 * 2 * self.decoder_in_chans:
            # if isinstance(planes, tuple):
            #     N *= 2
            triplane_bg = True
            # planes = torch.cat(planes, 0) # inference in parallel
            # ray_origins = ray_origins.repeat(2,1,1)
            # ray_directions = ray_directions.repeat(2,1,1)

        else:
            triplane_bg = False

        # assert not triplane_bg

        # ! hard coded, will fix later
        # if planes.shape[1] == 3 * self.decoder_in_chans:
        # else:

        # planes = planes.view(len(planes), 3, self.decoder_in_chans,
        planes = planes.reshape(
            len(planes),
            3,
            -1,  # ! support background plane
            planes.shape[-2],
            planes.shape[-1])  # BS 96 256 256

        # Perform volume rendering
        rendering_details = self.renderer(planes,
                                          self.decoder,
                                          ray_origins,
                                          ray_directions,
                                          self.rendering_kwargs,
                                          return_meta=return_meta)

        feature_samples, depth_samples, weights_samples = (
            rendering_details[k]
            for k in ['feature_samples', 'depth_samples', 'weights_samples'])

        if return_sampling_details_flag:
            shape_synthesized = rendering_details['shape_synthesized']
        else:
            shape_synthesized = None

        # Reshape into 'raw' neural-rendered image
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            N, feature_samples.shape[-1], H,
            W).contiguous()  # B 32 H W, in [-1,1]
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        weights_samples = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Generate Background
        # if self.bcg_synthesis:

        #     # bg composition
        #     # if self.decoder.activation == "sigmoid":
        #     #     feature_image = feature_image * 2 - 1 # Scale to (-1, 1), taken from ray marcher

        #     assert isinstance(
        #         z_bcg, torch.Tensor
        #     )  # 512 latents after reparmaterization, reuse the name
        #     # ws_bcg = ws[:,:self.bcg_synthesis.num_ws] if ws_bcg is None else ws_bcg[:,:self.bcg_synthesis.num_ws]

        #     with torch.autocast(device_type='cuda',
        #                         dtype=torch.float16,
        #                         enabled=False):

        #         ws_bcg = self.bcg_mapping(z_bcg, c=None)  # reuse the name
        #         if ws_bcg.size(1) < self.bcg_synthesis.num_ws:
        #             ws_bcg = torch.cat([
        #                 ws_bcg, ws_bcg[:, -1:].repeat(
        #                     1, self.bcg_synthesis.num_ws - ws_bcg.size(1), 1)
        #             ], 1)

        #         bcg_image = self.bcg_synthesis(ws_bcg,
        #                                        update_emas=update_emas,
        #                                        **synthesis_kwargs)
        #     bcg_image = torch.nn.functional.interpolate(
        #         bcg_image,
        #         size=feature_image.shape[2:],
        #         mode='bilinear',
        #         align_corners=False,
        #         antialias=self.rendering_kwargs['sr_antialias'])
        #     feature_image = feature_image + (1 - weights_samples) * bcg_image

        #     # Generate Raw image
        #     assert self.torgb
        #     rgb_image = self.torgb(feature_image,
        #                            ws_bcg[:, -1],
        #                            fused_modconv=False)
        #     rgb_image = rgb_image.to(dtype=torch.float32,
        #                              memory_format=torch.contiguous_format)
        #     # st()
        # else:

        mask_image = weights_samples * (1 + 2 * 0.001) - 0.001
        if triplane_bg:
            # true_bs = N // 2
            # weights_samples = weights_samples[:true_bs]
            # mask_image = mask_image[:true_bs]
            # feature_image = feature_image[:true_bs] * mask_image + feature_image[true_bs:] * (1-mask_image) # the first is foreground
            # depth_image = depth_image[:true_bs]

            # ! composited colors
            # rgb_final = (
            #     1 - fg_ret_dict['weights']
            # ) * bg_ret_dict['rgb_final'] + fg_ret_dict[
            #     'feature_samples']  # https://github.com/SizheAn/PanoHead/blob/17ad915941c7e2703d5aa3eb5ff12eac47c90e53/training/triplane.py#L127C45-L127C64

            # ret_dict.update({
            #     'feature_samples': rgb_final,
            # })
            # st()
            feature_image = (1 - mask_image) * rendering_details[
                'bg_ret_dict']['rgb_final'] + feature_image

        rgb_image = feature_image[:, :3]

        # # Run superresolution to get final image
        if self.superresolution is not None and not return_raw_only:
            # assert ws is not None, 'feed in [cls] token here for SR module'

            if ws is not None and ws.ndim == 2:
                ws = ws.unsqueeze(
                    1)[:, -1:, :]  # follow stylegan tradition, B, N, C

            sr_image = self.superresolution(
                rgb=rgb_image,
                x=feature_image,
                base_x=rgb_image,
                ws=ws,  # only use the last layer
                noise_mode=self.
                rendering_kwargs['superresolution_noise_mode'],  # none
                **{
                    k: synthesis_kwargs[k]
                    for k in synthesis_kwargs.keys() if k != 'noise_mode'
                })
        else:
            # sr_image = rgb_image
            sr_image = None

        if shape_synthesized is not None:
            shape_synthesized.update({
                'image_depth': depth_image,
            })  # for 3D loss easy computation, wrap all 3D in a single dict

        ret_dict = {
            'feature_image': feature_image,
            # 'image_raw': feature_image[:, :3],
            'image_raw': rgb_image,
            'image_depth': depth_image,
            'weights_samples': weights_samples,
            # 'silhouette': mask_image,
            # 'silhouette_normalized_3channel': (mask_image*2-1).repeat_interleave(3,1), # N 3 H W
            'shape_synthesized': shape_synthesized,
            "image_mask": mask_image,
        }

        if sr_image is not None:
            ret_dict.update({
                'image_sr': sr_image,
            })

        if return_meta:
            ret_dict.update({
                'feature_volume':
                rendering_details['feature_volume'],
                'all_coords':
                rendering_details['all_coords'],
                'weights':
                rendering_details['weights'],
            })

        return ret_dict


class Triplane_fg_bg_plane(Triplane):
    # a separate background plane

    def __init__(self,
                 c_dim=25,
                 img_resolution=128,
                 img_channels=3,
                 out_chans=96,
                 triplane_size=224,
                 rendering_kwargs={},
                 decoder_in_chans=32,
                 decoder_output_dim=32,
                 sr_num_fp16_res=0,
                 sr_kwargs={},
                 bcg_synthesis_kwargs={}):
        super().__init__(c_dim, img_resolution, img_channels, out_chans,
                         triplane_size, rendering_kwargs, decoder_in_chans,
                         decoder_output_dim, sr_num_fp16_res, sr_kwargs,
                         bcg_synthesis_kwargs)

        self.bcg_decoder = Decoder(
            ch=64,  # half channel size
            out_ch=32,
            # ch_mult=(1, 2, 4),
            ch_mult=(1, 2),  # use res=64 for now
            num_res_blocks=2,
            dropout=0.0,
            attn_resolutions=(),
            z_channels=4,
            resolution=64,
            in_channels=3,
        )

    # * pure reconstruction
    def forward(
            self,
            planes,
            bg_plane,
            # img,
            c,
            ws=None,
            z_bcg=None,
            neural_rendering_resolution=None,
            update_emas=False,
            cache_backbone=False,
            use_cached_backbone=False,
            return_meta=False,
            return_raw_only=False,
            **synthesis_kwargs):

        # ! match the batch size
        if planes is None:
            assert self.planes is not None
            planes = self.planes.repeat_interleave(c.shape[0], dim=0)
        return_sampling_details_flag = self.rendering_kwargs.get(
            'return_sampling_details_flag', False)

        if return_sampling_details_flag:
            return_meta = True

        cam2world_matrix = c[:, :16].reshape(-1, 4, 4)
        # cam2world_matrix = torch.eye(4, device=c.device).unsqueeze(0).repeat_interleave(c.shape[0], dim=0)
        # c[:, :16] = cam2world_matrix.view(-1, 16)
        intrinsics = c[:, 16:25].reshape(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        H = W = self.neural_rendering_resolution
        # Create a batch of rays for volume rendering
        ray_origins, ray_directions, _ = self.ray_sampler(
            cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape

        # # Reshape output into three 32-channel planes
        # if planes.shape[1] == 3 * 2 * self.decoder_in_chans:
        #     # if isinstance(planes, tuple):
        #     #     N *= 2
        #     triplane_bg = True
        #     # planes = torch.cat(planes, 0) # inference in parallel
        #     # ray_origins = ray_origins.repeat(2,1,1)
        #     # ray_directions = ray_directions.repeat(2,1,1)

        # else:
        #     triplane_bg = False

        # assert not triplane_bg

        planes = planes.view(
            len(planes),
            3,
            -1,  # ! support background plane
            planes.shape[-2],
            planes.shape[-1])  # BS 96 256 256

        # Perform volume rendering
        rendering_details = self.renderer(planes,
                                          self.decoder,
                                          ray_origins,
                                          ray_directions,
                                          self.rendering_kwargs,
                                          return_meta=return_meta)

        feature_samples, depth_samples, weights_samples = (
            rendering_details[k]
            for k in ['feature_samples', 'depth_samples', 'weights_samples'])

        if return_sampling_details_flag:
            shape_synthesized = rendering_details['shape_synthesized']
        else:
            shape_synthesized = None

        # Reshape into 'raw' neural-rendered image
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            N, feature_samples.shape[-1], H,
            W).contiguous()  # B 32 H W, in [-1,1]
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        weights_samples = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        bcg_image = self.bcg_decoder(bg_plane)
        bcg_image = torch.nn.functional.interpolate(
            bcg_image,
            size=feature_image.shape[2:],
            mode='bilinear',
            align_corners=False,
            antialias=self.rendering_kwargs['sr_antialias'])

        mask_image = weights_samples * (1 + 2 * 0.001) - 0.001

        # ! fuse fg/bg model output
        feature_image = feature_image + (1 - weights_samples) * bcg_image

        rgb_image = feature_image[:, :3]

        # # Run superresolution to get final image
        if self.superresolution is not None and not return_raw_only:
            # assert ws is not None, 'feed in [cls] token here for SR module'

            if ws is not None and ws.ndim == 2:
                ws = ws.unsqueeze(
                    1)[:, -1:, :]  # follow stylegan tradition, B, N, C

            sr_image = self.superresolution(
                rgb=rgb_image,
                x=feature_image,
                base_x=rgb_image,
                ws=ws,  # only use the last layer
                noise_mode=self.
                rendering_kwargs['superresolution_noise_mode'],  # none
                **{
                    k: synthesis_kwargs[k]
                    for k in synthesis_kwargs.keys() if k != 'noise_mode'
                })
        else:
            # sr_image = rgb_image
            sr_image = None

        if shape_synthesized is not None:
            shape_synthesized.update({
                'image_depth': depth_image,
            })  # for 3D loss easy computation, wrap all 3D in a single dict

        ret_dict = {
            'feature_image': feature_image,
            # 'image_raw': feature_image[:, :3],
            'image_raw': rgb_image,
            'image_depth': depth_image,
            'weights_samples': weights_samples,
            # 'silhouette': mask_image,
            # 'silhouette_normalized_3channel': (mask_image*2-1).repeat_interleave(3,1), # N 3 H W
            'shape_synthesized': shape_synthesized,
            "image_mask": mask_image,
        }

        if sr_image is not None:
            ret_dict.update({
                'image_sr': sr_image,
            })

        if return_meta:
            ret_dict.update({
                'feature_volume':
                rendering_details['feature_volume'],
                'all_coords':
                rendering_details['all_coords'],
                'weights':
                rendering_details['weights'],
            })

        return ret_dict

class TriplaneFlexiCube(Triplane):
    def __init__(self, c_dim=25, img_resolution=128, img_channels=3, out_chans=96, triplane_size=224, rendering_kwargs={}, decoder_in_chans=32, decoder_output_dim=32, sr_num_fp16_res=0, sr_kwargs={}, 
        create_triplane=False, # for overfitting single instance study
        bcg_synthesis_kwargs={},
        lrm_decoder=False,
    ):
        super().__init__(c_dim, img_resolution, img_channels, out_chans, triplane_size, rendering_kwargs, decoder_in_chans, decoder_output_dim, sr_num_fp16_res, sr_kwargs, bcg_synthesis_kwargs, lrm_decoder=lrm_decoder, create_triplane=create_triplane)
        # https://github.com/TencentARC/InstantMesh/blob/0a64425c6d390afa40128132cec42cd5c6408bbf/src/models/renderer/synthesizer_mesh.py#L93

    def get_geometry_prediction(self, planes, sample_coordinates, flexicubes_indices):
        plane_axes = self.plane_axes.to(planes.device)
        sampled_features = sample_from_planes(
            plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=self.rendering_kwargs['box_warp'])

        sdf, deformation, weight = self.decoder.get_geometry_prediction(sampled_features, flexicubes_indices)
        return sdf, deformation, weight
    
    def get_texture_prediction(self, planes, sample_coordinates):
        plane_axes = self.plane_axes.to(planes.device)
        sampled_features = sample_from_planes(
            plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=self.rendering_kwargs['box_warp'])

        rgb = self.decoder.get_texture_prediction(sampled_features)
        return rgb