import torch
from torch import nn
from nsr.triplane import Triplane_fg_bg_plane
# import timm
# from vit.vit_triplane import ViTTriplane, Triplane, ViTTriplaneDecomposed
from vit.vit_triplane import Triplane, ViTTriplaneDecomposed
import argparse
import inspect
import dnnlib
from guided_diffusion import dist_util

from pdb import set_trace as st

import vit.vision_transformer as vits
from guided_diffusion import logger
from .confnet import ConfNet

from ldm.modules.diffusionmodules.model import Encoder, Decoder, MVEncoder, MVEncoderGS, MVEncoderGSDynamicInp, MVEncoderGSDynamicInp_CA
from ldm.modules.diffusionmodules.mv_unet import MVUNet, LGM_MVEncoder
from torch.profiler import profile, record_function, ProfilerActivity

from nsr.gs import GaussianRenderer
from nsr.gs_surfel import GaussianRenderer2DGS
# from nsr.srt.encoder import ImprovedSRTEncoderVAE, ImprovedSRTEncoderVAE_L5_vitl, ImprovedSRTEncoderVAE_mlp_ratio4, ImprovedSRTEncoderVAE_L6, ImprovedSRTEncoderVAE_mlp_ratio4_f8, ImprovedSRTEncoderVAE_mlp_ratio4_heavyPatchify, ImprovedSRTEncoderVAE_mlp_ratio4_f8_L6, ImprovedSRTEncoderVAE_mlp_ratio4_L6, HybridEncoder, ImprovedSRTEncoderVAE_mlp_ratio4_decomposed, HybridEncoderPCDStructuredLatent
from nsr.srt.encoder import *

# from ldm.modules.diffusionmodules.openaimodel import MultiViewUNetModel_Encoder

# * create pre-trained encoder & triplane / other nsr decoder


class AE(torch.nn.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 img_size,
                 encoder_cls_token,
                 decoder_cls_token,
                 preprocess,
                 use_clip,
                 dino_version='v1',
                 clip_dtype=None,
                 no_dim_up_mlp=False,
                 dim_up_mlp_as_func=False,
                 uvit_skip_encoder=False,
                 confnet=None) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.img_size = img_size
        self.encoder_cls_token = encoder_cls_token
        self.decoder_cls_token = decoder_cls_token
        self.use_clip = use_clip
        self.dino_version = dino_version
        self.confnet = confnet

        if self.dino_version == 'v2':
            self.encoder.mask_token = None
            self.decoder.vit_decoder.mask_token = None

        if 'sd' not in self.dino_version:

            self.uvit_skip_encoder = uvit_skip_encoder
            if uvit_skip_encoder:
                logger.log(
                    f'enables uvit: length of vit_encoder.blocks: {len(self.encoder.blocks)}'
                )
                for blk in self.encoder.blocks[len(self.encoder.blocks) // 2:]:
                    blk.skip_linear = nn.Linear(2 * self.encoder.embed_dim,
                                                self.encoder.embed_dim)

                    # trunc_normal_(blk.skip_linear.weight, std=.02)
                    nn.init.constant_(blk.skip_linear.weight, 0)
                    if isinstance(
                            blk.skip_linear,
                            nn.Linear) and blk.skip_linear.bias is not None:
                        nn.init.constant_(blk.skip_linear.bias, 0)
            else:
                logger.log(f'disable uvit')
        else:
            if 'dit' not in self.dino_version:  # dino vit, not dit
                self.decoder.vit_decoder.cls_token = None
                self.decoder.vit_decoder.patch_embed.proj = nn.Identity()
                self.decoder.triplane_decoder.planes = None
                self.decoder.vit_decoder.mask_token = None

            if self.use_clip:
                self.clip_dtype = clip_dtype  # torch.float16

            else:

                if not no_dim_up_mlp and self.encoder.embed_dim != self.decoder.vit_decoder.embed_dim:
                    self.dim_up_mlp = nn.Linear(
                        self.encoder.embed_dim,
                        self.decoder.vit_decoder.embed_dim)
                    logger.log(
                        f"dim_up_mlp: {self.encoder.embed_dim} -> {self.decoder.vit_decoder.embed_dim}, as_func: {self.dim_up_mlp_as_func}"
                    )
                else:
                    logger.log('ignore dim_up_mlp: ', no_dim_up_mlp)

        self.preprocess = preprocess

        self.dim_up_mlp = None  # CLIP/B-16
        self.dim_up_mlp_as_func = dim_up_mlp_as_func

        # * remove certain components to make sure no unused parameters during DDP
        # self.decoder.vit_decoder.cls_token = nn.Identity()
        torch.cuda.empty_cache()
        # self.decoder.vit_decoder.patch_embed.proj.bias = nn.Identity()
        # self.decoder.vit_decoder.patch_embed.proj.weight = nn.Identity()
        # self.decoder.vit_decoder.patch_embed.proj.bias = nn.Identity()

    def encode(self, *args, **kwargs):
        if not self.use_clip:
            if self.dino_version == 'v1':
                latent = self.encode_dinov1(*args, **kwargs)
            elif self.dino_version == 'v2':
                if self.uvit_skip_encoder:
                    latent = self.encode_dinov2_uvit(*args, **kwargs)
                else:
                    latent = self.encode_dinov2(*args, **kwargs)
            else:
                latent = self.encoder(*args, **kwargs)

        else:
            latent = self.encode_clip(*args, **kwargs)

        return latent

    def encode_dinov1(self, x):
        # return self.encoder(img)
        x = self.encoder.prepare_tokens(x)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        if not self.encoder_cls_token:
            return x[:, 1:]

        return x

    def encode_dinov2(self, x):
        # return self.encoder(img)
        x = self.encoder.prepare_tokens_with_masks(x, masks=None)
        for blk in self.encoder.blocks:
            x = blk(x)
        x_norm = self.encoder.norm(x)

        if not self.encoder_cls_token:
            return x_norm[:, 1:]
        # else:
        # return x_norm[:, :1]

        # return {
        #     "x_norm_clstoken": x_norm[:, 0],
        #     "x_norm_patchtokens": x_norm[:, 1:],
        # }

        return x_norm

    def encode_dinov2_uvit(self, x):
        # return self.encoder(img)
        x = self.encoder.prepare_tokens_with_masks(x, masks=None)

        # for blk in self.encoder.blocks:
        #     x = blk(x)

        skips = [x]

        # in blks
        for blk in self.encoder.blocks[0:len(self.encoder.blocks) // 2 - 1]:
            x = blk(x)  # B 3 N C
            skips.append(x)

        # mid blks
        for blk in self.encoder.blocks[len(self.encoder.blocks) // 2 -
                                       1:len(self.encoder.blocks) // 2]:
            x = blk(x)  # B 3 N C

        # out blks
        for blk in self.encoder.blocks[len(self.encoder.blocks) // 2:]:
            x = x + blk.skip_linear(torch.cat(
                [x, skips.pop()], dim=-1))  # long skip connections in uvit
            x = blk(x)  # B 3 N C

        x_norm = self.encoder.norm(x)

        if not self.decoder_cls_token:
            return x_norm[:, 1:]

        return x_norm

    def encode_clip(self, x):
        # * replace with CLIP encoding pipeline
        # return self.encoder(img)
        # x = x.dtype(self.clip_dtype)
        x = self.encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.encoder.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.encoder.ln_post(x[:, 1:, :])  # * return the spatial tokens

        return x

        # x = self.ln_post(x[:, 0, :]) # * return the spatial tokens

        # if self.proj is not None:
        #     x = x @ self.proj

        # return x

    def decode_wo_triplane(self, latent, c=None, img_size=None):
        if img_size is None:
            img_size = self.img_size

        if self.dim_up_mlp is not None:
            if not self.dim_up_mlp_as_func:
                latent = self.dim_up_mlp(latent)
                # return self.decoder.vit_decode(latent, img_size)
            else:
                return self.decoder.vit_decode(
                    latent, img_size,
                    dim_up_mlp=self.dim_up_mlp)  # used in vae-ldm

        return self.decoder.vit_decode(latent, img_size, c=c)

    def decode(self, latent, c, img_size=None, return_raw_only=False):
        # if img_size is None:
        #     img_size = self.img_size

        # if self.dim_up_mlp is not None:
        #     latent = self.dim_up_mlp(latent)

        latent = self.decode_wo_triplane(latent, img_size=img_size, c=c)
        # return self.decoder.triplane_decode(latent, c, return_raw_only=return_raw_only)
        return self.decoder.triplane_decode(latent, c)

    def decode_after_vae_no_render(
        self,
        ret_dict,
        img_size=None,
    ):

        if img_size is None:
            img_size = self.img_size

        assert self.dim_up_mlp is None
        # if not self.dim_up_mlp_as_func:
        #     latent = self.dim_up_mlp(latent)
        # return self.decoder.vit_decode(latent, img_size)

        latent = self.decoder.vit_decode_backbone(ret_dict, img_size)
        ret_dict = self.decoder.vit_decode_postprocess(latent, ret_dict)
        return ret_dict

    def decode_after_vae_no_render_gs(
        self,
        ret_dict,
        img_size=None,
    ):

        ret_after_decoder = self.decode_after_vae_no_render(ret_dict, img_size)
        return self.decoder.forward_gaussians(ret_after_decoder, c=None)

    def decode_after_vae(
            self,
            #  latent,
            ret_dict,  # vae_dict
            c,
            img_size=None,
            return_raw_only=False):
        ret_dict = self.decode_after_vae_no_render(ret_dict, img_size)
        return self.decoder.triplane_decode(ret_dict, c)

    def decode_confmap(self, img):
        assert self.confnet is not None
        # https://github.com/elliottwu/unsup3d/blob/dc961410d61684561f19525c2f7e9ee6f4dacb91/unsup3d/model.py#L152
        # conf_sigma_l1 = self.confnet(img)  # Bx2xHxW
        return self.confnet(img)  # Bx1xHxW

    def encode_decode(self, img, c, return_raw_only=False):
        latent = self.encode(img)
        pred = self.decode(latent, c, return_raw_only=return_raw_only)
        if self.confnet is not None:
            pred.update({
                'conf_sigma': self.decode_confmap(img)  # 224x224
            })

        return pred

    def forward(self,
                img=None,
                c=None,
                latent=None,
                behaviour='enc_dec',
                coordinates=None,
                directions=None,
                return_raw_only=False,
                *args,
                **kwargs):
        """wrap all operations inside forward() for DDP use.
        """

        if behaviour == 'enc_dec':
            pred = self.encode_decode(img, c, return_raw_only=return_raw_only)
            return pred

        elif behaviour == 'enc':
            latent = self.encode(img)
            return latent

        elif behaviour == 'dec':
            assert latent is not None
            pred: dict = self.decode(latent,
                                     c,
                                     self.img_size,
                                     return_raw_only=return_raw_only)
            return pred

        elif behaviour == 'dec_wo_triplane':
            assert latent is not None
            pred: dict = self.decode_wo_triplane(latent, self.img_size)
            return pred

        elif behaviour == 'enc_dec_wo_triplane':
            # with profile(activities=[
            #     ProfilerActivity.CUDA], record_shapes=True) as prof:
                # with record_function("encoding"):
            latent = self.encode(img, c=c, **kwargs)

            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            # with profile(activities=[
            #     ProfilerActivity.CUDA], record_shapes=True) as prof:
            #     with record_function("decoding"):
            pred: dict = self.decode_wo_triplane(latent,
                                                img_size=self.img_size,
                                                c=c)
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            # st()

            return pred

        elif behaviour == 'encoder_vae':
            latent = self.encode(img)
            ret_dict = self.decoder.vae_reparameterization(latent, True)
            return ret_dict

        elif behaviour == 'decode_after_vae_no_render':
            pred: dict = self.decode_after_vae_no_render(latent, self.img_size)
            return pred

        elif behaviour == 'decode_gs_after_vae_no_render':
            pred: dict = self.decode_after_vae_no_render_gs(latent, self.img_size)
            return pred

        elif behaviour == 'decode_after_vae':
            pred: dict = self.decode_after_vae(latent, c, self.img_size)
            return pred

        # elif behaviour == 'gaussian_dec':
        #     assert latent is not None
        #     pred: dict = self.decoder.triplane_decode(
        #         latent, c, return_raw_only=return_raw_only, **kwargs)
        #     # pred: dict = self.decoder.triplane_decode(latent, c)

        elif behaviour == 'triplane_dec':
            assert latent is not None
            pred: dict = self.decoder.triplane_decode(
                latent, c, return_raw_only=return_raw_only, **kwargs)
            # pred: dict = self.decoder.triplane_decode(latent, c)

        elif behaviour == 'triplane_decode_grid':
            assert latent is not None
            pred: dict = self.decoder.triplane_decode_grid(latent, **kwargs)
            # pred: dict = self.decoder.triplane_decode(latent, c)

        elif behaviour == 'vit_postprocess_triplane_dec':
            assert latent is not None
            latent = self.decoder.vit_decode_postprocess(
                latent)  # translate spatial token from vit-decoder into 2D
            pred: dict = self.decoder.triplane_decode(
                latent, c)  # render with triplane

        elif behaviour == 'triplane_renderer':
            assert latent is not None
            pred: dict = self.decoder.triplane_renderer(
                latent, coordinates, directions)

        # elif behaviour == 'triplane_SR':
        #     assert latent is not None
        #     pred: dict = self.decoder.triplane_renderer(
        #         latent, coordinates, directions)

        elif behaviour == 'get_rendering_kwargs':
            pred = self.decoder.triplane_decoder.rendering_kwargs

        return pred


class AE_CLIPEncoder(AE):

    def __init__(self, encoder, decoder, img_size, cls_token) -> None:
        super().__init__(encoder, decoder, img_size, cls_token)


class AE_with_Diffusion(torch.nn.Module):

    def __init__(self, auto_encoder, denoise_model) -> None:
        super().__init__()
        self.auto_encoder = auto_encoder
        self.denoise_model = denoise_model  # simply for easy MPTrainer manipulation

    def forward(self,
                img,
                c,
                behaviour='enc_dec',
                latent=None,
                *args,
                **kwargs):
        # wrap auto_encoder and denoising model inside a single forward function to use DDP (only forward supported) and MPTrainer (single model) easier
        if behaviour == 'enc_dec':
            pred = self.auto_encoder(img, c)
            return pred
        elif behaviour == 'enc':
            latent = self.auto_encoder.encode(img)
            if self.auto_encoder.dim_up_mlp is not None:
                latent = self.auto_encoder.dim_up_mlp(latent)
            return latent
        elif behaviour == 'dec':
            assert latent is not None
            pred: dict = self.auto_encoder.decode(latent, c, self.img_size)
            return pred
        elif behaviour == 'denoise':
            assert latent is not None
            pred: dict = self.denoise_model(*args, **kwargs)
            return pred


def eg3d_options_default():

    opts = dnnlib.EasyDict(
        dict(
            cbase=32768,
            cmax=512,
            map_depth=2,
            g_class_name='nsr.triplane.TriPlaneGenerator',  # TODO
            g_num_fp16_res=0,
        ))

    return opts


def rendering_options_defaults(opts):

    rendering_options = {
        # 'image_resolution': c.training_set_kwargs.resolution,
        'image_resolution': 256,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'c_gen_conditioning_zero':
        True,  # if true, fill generator pose conditioning label with dummy zero vector
        # 'gpc_reg_prob': opts.gpc_reg_prob if opts.gen_pose_cond else None,
        'c_scale':
        opts.c_scale,  # mutliplier for generator pose conditioning label
        'superresolution_noise_mode': 'none',
        'density_reg': opts.density_reg,  # strength of density regularization
        'density_reg_p_dist': opts.
        density_reg_p_dist,  # distance at which to sample perturbed points for density regularization
        'reg_type': opts.
        reg_type,  # for experimenting with variations on density regularization
        'decoder_lr_mul': 1,
        # opts.decoder_lr_mul,  # learning rate multiplier for decoder
        'decoder_activation': 'sigmoid',
        'sr_antialias': True,
        'return_triplane_features': False,  # for DDF supervision
        'return_sampling_details_flag': False,

        # * shape default sr

        # 'superresolution_module': 'nsr.superresolution.SuperresolutionHybrid4X',
        # 'superresolution_module':
        # 'torch_utils.components.PixelUnshuffleUpsample',
        'superresolution_module': 'torch_utils.components.NearestConvSR',
    }

    if opts.cfg == 'ffhq':
        rendering_options.update({
            'superresolution_module':
            'nsr.superresolution.SuperresolutionHybrid8XDC',
            'focal': 2985.29 / 700,
            'depth_resolution':
            48 - 0,  # number of uniform samples to take per ray.
            'depth_resolution_importance':
            48 - 0,  # number of importance samples to take per ray.
            'bg_depth_resolution':
            16,  # 4/14 in stylenerf, https://github.com/facebookresearch/StyleNeRF/blob/7f5610a058f27fcc360c6b972181983d7df794cb/conf/model/stylenerf_ffhq.yaml#L48
            'ray_start':
            2.25,  # near point along each ray to start taking samples.
            'ray_end':
            3.3,  # far point along each ray to stop taking samples. 
            'box_warp':
            1,  # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'avg_camera_radius':
            2.7,  # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [
                0, 0, 0.2
            ],  # used only in the visualizer to control center of camera rotation.
            'superresolution_noise_mode': 'random',
        })
    elif opts.cfg == 'afhq':
        rendering_options.update({
            'superresolution_module':
            'nsr.superresolution.SuperresolutionHybrid8X',
            'superresolution_noise_mode': 'random',
            'focal': 4.2647,
            'depth_resolution': 48,
            'depth_resolution_importance': 48,
            'ray_start': 2.25,
            'ray_end': 3.3,
            'box_warp': 1,
            'avg_camera_radius': 2.7,
            'avg_camera_pivot': [0, 0, -0.06],
        })
    elif opts.cfg == 'shapenet':  # TODO, lies in a sphere
        rendering_options.update({
            'depth_resolution': 64,
            'depth_resolution_importance': 64,
            # * radius 1.2 setting, newly rendered images
            'ray_start': 0.2,
            'ray_end': 2.2,
            # 'ray_start': opts.ray_start,
            # 'ray_end': opts.ray_end,
            'box_warp': 2,  # TODO, how to set this value?
            'white_back': True,
            'avg_camera_radius': 1.2,
            'avg_camera_pivot': [0, 0, 0],
        })

    elif opts.cfg == 'eg3d_shapenet_aug_resolution':
        rendering_options.update({
            'depth_resolution': 80,
            'depth_resolution_importance': 80,
            'ray_start': 0.1,
            'ray_end': 1.9,  # 2.6/1.7*1.2
            'box_warp': 1.1,
            'white_back': True,
            'avg_camera_radius': 1.2,
            'avg_camera_pivot': [0, 0, 0],
        })

    elif opts.cfg == 'eg3d_shapenet_aug_resolution_chair':
        rendering_options.update({
            'depth_resolution': 96,
            'depth_resolution_importance': 96,
            'ray_start': 0.1,
            'ray_end': 1.9,  # 2.6/1.7*1.2
            'box_warp': 1.1,
            'white_back': True,
            'avg_camera_radius': 1.2,
            'avg_camera_pivot': [0, 0, 0],
        })

    elif opts.cfg == 'eg3d_shapenet_aug_resolution_chair_128':
        rendering_options.update({
            'depth_resolution': 128,
            'depth_resolution_importance': 128,
            'ray_start': 0.1,
            'ray_end': 1.9,  # 2.6/1.7*1.2
            'box_warp': 1.1,
            'white_back': True,
            'avg_camera_radius': 1.2,
            'avg_camera_pivot': [0, 0, 0],
        })

    elif opts.cfg == 'eg3d_shapenet_aug_resolution_chair_64':
        rendering_options.update({
            'depth_resolution': 64,
            'depth_resolution_importance': 64,
            'ray_start': 0.1,
            'ray_end': 1.9,  # 2.6/1.7*1.2
            'box_warp': 1.1,
            'white_back': True,
            'avg_camera_radius': 1.2,
            'avg_camera_pivot': [0, 0, 0],
        })

    elif opts.cfg == 'srn_shapenet_aug_resolution_chair_128':
        rendering_options.update({
            'depth_resolution': 128,
            'depth_resolution_importance': 128,
            'ray_start': 1.25,
            'ray_end': 2.75,
            'box_warp': 1.5,
            'white_back': True,
            'avg_camera_radius': 2,
            'avg_camera_pivot': [0, 0, 0],
        })

    elif opts.cfg == 'eg3d_shapenet_aug_resolution_chair_128_residualSR':
        rendering_options.update({
            'depth_resolution':
            128,
            'depth_resolution_importance':
            128,
            'ray_start':
            0.1,
            'ray_end':
            1.9,  # 2.6/1.7*1.2
            'box_warp':
            1.1,
            'white_back':
            True,
            'avg_camera_radius':
            1.2,
            'avg_camera_pivot': [0, 0, 0],
            'superresolution_module':
            'torch_utils.components.NearestConvSR_Residual',
        })

    elif opts.cfg == 'shapenet_tuneray':  # TODO, lies in a sphere
        rendering_options.update({
            'depth_resolution': 64,
            'depth_resolution_importance': 64,
            # * radius 1.2 setting, newly rendered images
            'ray_start': opts.ray_start,
            'ray_end': opts.ray_end,
            'box_warp':
            opts.ray_end - opts.ray_start,  # TODO, how to set this value?
            'white_back': True,
            'avg_camera_radius': 1.2,
            'avg_camera_pivot': [0, 0, 0],
        })

    elif opts.cfg == 'shapenet_tuneray_aug_resolution':  # to differentiate hwc
        rendering_options.update({
            'depth_resolution': 80,
            'depth_resolution_importance': 80,
            # * radius 1.2 setting, newly rendered images
            'ray_start': opts.ray_start,
            'ray_end': opts.ray_end,
            'box_warp':
            opts.ray_end - opts.ray_start,  # TODO, how to set this value?
            'white_back': True,
            'avg_camera_radius': 1.2,
            'avg_camera_pivot': [0, 0, 0],
        })

    elif opts.cfg == 'shapenet_tuneray_aug_resolution_64':  # to differentiate hwc
        rendering_options.update({
            'depth_resolution': 128,
            'depth_resolution_importance': 128,
            # * radius 1.2 setting, newly rendered images
            'ray_start': opts.ray_start,
            'ray_end': opts.ray_end,
            'box_warp':
            opts.ray_end - opts.ray_start,  # TODO, how to set this value?
            'white_back': True,
            'avg_camera_radius': 1.2,
            'avg_camera_pivot': [0, 0, 0],
        })

    elif opts.cfg == 'shapenet_tuneray_aug_resolution_64_96':  # to differentiate hwc
        rendering_options.update({
            'depth_resolution': 96,
            'depth_resolution_importance': 96,
            # * radius 1.2 setting, newly rendered images
            'ray_start': opts.ray_start,
            'ray_end': opts.ray_end,
            'box_warp':
            opts.ray_end - opts.ray_start,  # TODO, how to set this value?
            'white_back': True,
            'avg_camera_radius': 1.2,
            'avg_camera_pivot': [0, 0, 0],
        })
    # ! default version
    elif opts.cfg == 'shapenet_tuneray_aug_resolution_64_96_nearestSR':  # to differentiate hwc
        rendering_options.update({
            'depth_resolution':
            96,
            'depth_resolution_importance':
            96,
            # * radius 1.2 setting, newly rendered images
            'ray_start':
            opts.ray_start,
            'ray_end':
            opts.ray_end,
            'box_warp':
            opts.ray_end - opts.ray_start,  # TODO, how to set this value?
            'white_back':
            True,
            'avg_camera_radius':
            1.2,
            'avg_camera_pivot': [0, 0, 0],
            'superresolution_module':
            'torch_utils.components.NearestConvSR',
        })

    # ! 64+64, since ssdnerf adopts this setting
    elif opts.cfg == 'shapenet_tuneray_aug_resolution_64_64_nearestSR':  # to differentiate hwc
        rendering_options.update({
            'depth_resolution':
            64,
            'depth_resolution_importance':
            64,
            # * radius 1.2 setting, newly rendered images
            'ray_start':
            opts.ray_start,
            'ray_end':
            opts.ray_end,
            'box_warp':
            opts.ray_end - opts.ray_start,  # TODO, how to set this value?
            'white_back':
            True,
            'avg_camera_radius':
            1.2,
            'avg_camera_pivot': [0, 0, 0],
            'superresolution_module':
            'torch_utils.components.NearestConvSR',
        })

    # ! 64+64+patch, since ssdnerf adopts this setting
    elif opts.cfg == 'shapenet_tuneray_aug_resolution_64_64_nearestSR_patch':  # to differentiate hwc
        rendering_options.update({
            'depth_resolution':
            64,
            'depth_resolution_importance':
            64,
            # * radius 1.2 setting, newly rendered images
            'ray_start':
            opts.ray_start,
            'ray_end':
            opts.ray_end,
            'box_warp':
            opts.ray_end - opts.ray_start,  # TODO, how to set this value?
            'white_back':
            True,
            'avg_camera_radius':
            1.2,
            'avg_camera_pivot': [0, 0, 0],
            'superresolution_module':
            'torch_utils.components.NearestConvSR',
            # patch configs
            'PatchRaySampler':
            True,
            # 'patch_rendering_resolution': 32,
            # 'patch_rendering_resolution': 48,
            'patch_rendering_resolution':
            opts.patch_rendering_resolution,
        })

    elif opts.cfg == 'objverse_tuneray_aug_resolution_64_64_nearestSR':  # to differentiate hwc
        rendering_options.update({
            'depth_resolution':
            64,
            'depth_resolution_importance':
            64,
            # * radius 1.2 setting, newly rendered images
            'ray_start':
            opts.ray_start,
            # 'auto',
            'ray_end':
            opts.ray_end,
            # 'auto',
            'box_warp':
            opts.ray_end - opts.ray_start,  # TODO, how to set this value?
            # 2,
            'white_back':
            True,
            'avg_camera_radius':
            1.946,  # ?
            'avg_camera_pivot': [0, 0, 0],
            'superresolution_module':
            'torch_utils.components.NearestConvSR',
            # patch configs
            # 'PatchRaySampler': False,
            # 'patch_rendering_resolution': 32,
            # 'patch_rendering_resolution': 48,
            # 'patch_rendering_resolution': opts.patch_rendering_resolution,
        })

    elif opts.cfg == 'objverse_tuneray_aug_resolution_64_64_auto':  # to differentiate hwc
        rendering_options.update({
            'depth_resolution':
            64,
            'depth_resolution_importance':
            64,
            # * radius 1.2 setting, newly rendered images
            'ray_start':
            'auto',
            'ray_end':
            'auto',
            'box_warp':
            0.9,
            'white_back':
            True,
            'radius_range': [1.5, 2],
            # 'z_near': 1.5-0.45, # radius in [1.5, 2], https://github.com/modelscope/richdreamer/issues/12#issuecomment-1897734616
            # 'z_far': 2.0+0.45,
            'sampler_bbox_min':
            -0.45,
            'sampler_bbox_max':
            0.45,
            # 'avg_camera_pivot': [0, 0, 0], # not used
            'filter_out_of_bbox':
            True,
            # 'superresolution_module':
            # 'torch_utils.components.NearestConvSR',
            # patch configs
            'PatchRaySampler':
            True,
            # 'patch_rendering_resolution': 32,
            # 'patch_rendering_resolution': 48,
            'patch_rendering_resolution':
            opts.patch_rendering_resolution,
        })
        rendering_options['z_near'] = rendering_options['radius_range'][
            0] + rendering_options['sampler_bbox_min']
        rendering_options['z_far'] = rendering_options['radius_range'][
            1] + rendering_options['sampler_bbox_max']

    elif opts.cfg == 'objverse_tuneray_aug_resolution_56_56_auto':  # to differentiate hwc
        rendering_options.update({
            'depth_resolution':
            56,
            'depth_resolution_importance':
            56,
            # * radius 1.2 setting, newly rendered images
            'ray_start':
            'auto',
            'ray_end':
            'auto',
            'box_warp':
            0.9,
            'white_back':
            True,
            'radius_range': [1.5, 2],
            # 'z_near': 1.5-0.45, # radius in [1.5, 2], https://github.com/modelscope/richdreamer/issues/12#issuecomment-1897734616
            # 'z_far': 2.0+0.45,
            'sampler_bbox_min':
            -0.45,
            'sampler_bbox_max':
            0.45,
            # 'avg_camera_pivot': [0, 0, 0], # not used
            'filter_out_of_bbox':
            True,
            # 'superresolution_module':
            # 'torch_utils.components.NearestConvSR',
            # patch configs
            'PatchRaySampler':
            True,
            # 'patch_rendering_resolution': 32,
            # 'patch_rendering_resolution': 48,
            'patch_rendering_resolution':
            opts.patch_rendering_resolution,
        })
        rendering_options['z_near'] = rendering_options['radius_range'][
            0] + rendering_options['sampler_bbox_min']
        rendering_options['z_far'] = rendering_options['radius_range'][
            1] + rendering_options['sampler_bbox_max']

    elif opts.cfg == 'shapenet_tuneray_aug_resolution_64_96_nearestResidualSR':  # to differentiate hwc
        rendering_options.update({
            'depth_resolution':
            96,
            'depth_resolution_importance':
            96,
            # * radius 1.2 setting, newly rendered images
            'ray_start':
            opts.ray_start,
            'ray_end':
            opts.ray_end,
            'box_warp':
            opts.ray_end - opts.ray_start,  # TODO, how to set this value?
            'white_back':
            True,
            'avg_camera_radius':
            1.2,
            'avg_camera_pivot': [0, 0, 0],
            'superresolution_module':
            'torch_utils.components.NearestConvSR_Residual',
        })

    elif opts.cfg == 'shapenet_tuneray_aug_resolution_64_64_nearestResidualSR':  # to differentiate hwc
        rendering_options.update({
            'depth_resolution':
            64,
            'depth_resolution_importance':
            64,
            # * radius 1.2 setting, newly rendered images
            'ray_start':
            opts.ray_start,
            'ray_end':
            opts.ray_end,
            'box_warp':
            opts.ray_end - opts.ray_start,  # TODO, how to set this value?
            'white_back':
            True,
            'avg_camera_radius':
            1.2,
            'avg_camera_pivot': [0, 0, 0],
            'superresolution_module':
            'torch_utils.components.NearestConvSR_Residual',
        })

    elif opts.cfg == 'shapenet_tuneray_aug_resolution_64_104':  # to differentiate hwc
        rendering_options.update({
            'depth_resolution': 104,
            'depth_resolution_importance': 104,
            # * radius 1.2 setting, newly rendered images
            'ray_start': opts.ray_start,
            'ray_end': opts.ray_end,
            'box_warp':
            opts.ray_end - opts.ray_start,  # TODO, how to set this value?
            'white_back': True,
            'avg_camera_radius': 1.2,
            'avg_camera_pivot': [0, 0, 0],
        })

    rendering_options.update({'return_sampling_details_flag': True})
    rendering_options.update({'return_sampling_details_flag': True})

    return rendering_options


def model_encoder_defaults():

    return dict(
        use_clip=False,
        arch_encoder="vits",
        arch_decoder="vits",
        load_pretrain_encoder=False,
        encoder_lr=1e-5,
        encoder_weight_decay=
        0.001,  # https://github.com/google-research/vision_transformer
        no_dim_up_mlp=False,
        dim_up_mlp_as_func=False,
        decoder_load_pretrained=True,
        uvit_skip_encoder=False,
        # vae ldm
        vae_p=1,
        ldm_z_channels=4,
        ldm_embed_dim=4,
        use_conf_map=False,
        # sd E, lite version by default
        sd_E_ch=64,
        z_channels=3 * 4,
        latent_num=768,
        sd_E_num_res_blocks=1,
        num_frames=4,
        # vit_decoder
        arch_dit_decoder='DiT2-B/2',
        return_all_dit_layers=False,
        # sd D
        # sd_D_ch=32,
        # sd_D_res_blocks=1,
        # sd_D_res_blocks=1,
        lrm_decoder=False,
        gs_rendering=False,
        surfel_rendering=False,
        plane_n=3,
        in_plane_attention=True,
        vae_dit_token_size=16,
    )


def triplane_decoder_defaults():
    opts = dict(
        triplane_fg_bg=False,flexicube_decoder=False,
        cfg='shapenet',
        density_reg=0.25,
        density_reg_p_dist=0.004,
        reg_type='l1',
        triplane_decoder_lr=0.0025,  # follow eg3d G lr
        super_resolution_lr=0.0025,
        # triplane_decoder_wd=0.1,
        c_scale=1,
        nsr_lr=0.02,
        triplane_size=224,
        decoder_in_chans=32,
        triplane_in_chans=-1,
        decoder_output_dim=3,
        out_chans=96,
        c_dim=25,  # Conditioning label (C) dimensionality.
        # ray_start=0.2,
        # ray_end=2.2,
        ray_start=0.6,  # shapenet default
        ray_end=1.8,
        rendering_kwargs={},
        sr_training=False,
        bcg_synthesis=False,  # from panohead
        bcg_synthesis_kwargs={},  # G_kwargs.copy()
        #
        image_size=128,  # raw 3D rendering output resolution.
        patch_rendering_resolution=45,
    )

    # else:
    #     assert False, "Need to specify config"

    # opts = dict(opts)
    # opts.pop('cfg')

    return opts


def vit_decoder_defaults():
    res = dict(
        vit_decoder_lr=1e-5,  # follow eg3d G lr
        vit_decoder_wd=0.001,
    )
    return res


def nsr_decoder_defaults():
    res = {
        'decomposed': False,
    }  # TODO, add defaults for all nsr
    res.update(triplane_decoder_defaults())  # triplane by default now
    res.update(vit_decoder_defaults())  # type: ignore
    return res


def loss_defaults():
    opt = dict(
        color_criterion='mse',
        l2_lambda=1.0,
        lpips_lambda=0.,
        lpips_delay_iter=0,
        sr_delay_iter=0,
        # kl_anneal=0,
        kl_anneal=False,
        latent_lambda=0.,
        latent_criterion='mse',
        kl_lambda=0.0,
        pt_ft_kl=False,
        ft_kl=False,
        # pt_kl_lambda=1e-8,
        # kl_anneal=False,
        ssim_lambda=0.,
        l1_lambda=0.,
        id_lambda=0.0,
        depth_lambda=0.0,  # TODO
        alpha_lambda=0.0,  # TODO
        fg_mse=False,
        bg_lamdba=0.0,
        density_reg=0.0,  # tvloss in eg3d
        density_reg_p_dist=0.004,  # 'density regularization strength.'
        density_reg_every=4,  # lazy density reg

        # 3D supervision, ffhq/afhq eg3d warm up
        shape_uniform_lambda=0.005,
        shape_importance_lambda=0.01,
        shape_depth_lambda=0.,
        xyz_lambda=0.0,
        emd_lambda=0.0,
        cd_lambda=0.0,
        pruning_ot_lambda=0.0,
        # 2dgs
        lambda_normal=0.0,
        lambda_dist=0.0,

        #gghead reg
        lambda_scale_reg=0.0,
        lambda_opa_reg=0.0,

        # gan loss
        rec_cvD_lambda=0.01,
        nvs_cvD_lambda=0.025,
        patchgan_disc_factor=0.01,
        patchgan_disc_g_weight=0.2,  # 
        r1_gamma=1.0,  # ffhq default value for eg3d
        sds_lamdba=1.0,
        nvs_D_lr_mul=1,  # compared with 1e-4
        cano_D_lr_mul=1,  # compared with 1e-4

        # lsgm loss
        ce_balanced_kl=1.,
        p_eps_lambda=1,
        # symmetric loss
        symmetry_loss=False,
        depth_smoothness_lambda=0.0,
        ce_lambda=1.0,
        negative_entropy_lambda=1.0,
        grad_clip=False,
        online_mask=False,  # in unsup3d
        fps_sampling=False,  # for emd loss
        subset_fps_sampling=False,  # for emd loss
        subset_half_fps_sampling=False,
        # gaussian loss
        commitment_loss_lambda=0.0,
        rand_aug_bg=False,
    )
    return opt


def dataset_defaults():
    res = dict(
        use_lmdb=False,
        dataset_size=-1,
        use_wds=False,
        use_lmdb_compressed=True,
        compile=False,
        interval=1,
        objv_dataset=False,
        decode_encode_img_only=False,
        load_wds_diff=False,
        load_wds_latent=False,
        eval_load_wds_instance=True,
        shards_lst="",
        eval_shards_lst="",
        mv_input=False,
        duplicate_sample=True,
        orthog_duplicate=False,
        split_chunk_input=False,  # split=8 per chunk
        load_real=False,
        load_mv_real=False,
        load_gso=False,
        four_view_for_latent=False,
        single_view_for_i23d=False,
        shuffle_across_cls=False,
        load_extra_36_view=False,
        mv_latent_dir='',
        append_depth=False,
        append_xyz=False,
        read_normal=False,
        plucker_embedding=False,
        perturb_pcd_scale=0.0,
        gs_cam_format=False,
        frame_0_as_canonical=
        False,  # transform the first pose to a fixed position
        pcd_path=None,
        stage_1_output_dir='',
        load_pcd=False,
        use_chunk=False, # jpeg chunk
        split_chunk_size=8,
        load_caption_dataset=False,
        load_mv_dataset=False,
        export_mesh=False,
    )
    return res


def encoder_and_nsr_defaults():
    """
    Defaults for image training.
    """
    # ViT configs
    res = dict(
        dino_version='v1',
        encoder_in_channels=3,
        img_size=[224],
        patch_size=16,  # ViT-S/16
        in_chans=384,
        num_classes=0,
        embed_dim=384,  # Check ViT encoder dim
        depth=6,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer='nn.LayerNorm',
        # img_resolution=128,  # Output resolution.
        cls_token=False,
        # image_size=128,  # rendered output resolution.
        # img_channels=3,  # Number of output color channels.
        encoder_cls_token=False,
        decoder_cls_token=False,
        sr_kwargs={},
        sr_ratio=2,
        # sd configs
    )
    # Triplane configs
    res.update(model_encoder_defaults())
    res.update(nsr_decoder_defaults())
    res.update(
        ae_classname='vit.vit_triplane.ViTTriplaneDecomposed')  # if add SR
    return res


def create_3DAE_model(
        arch_encoder,
        arch_decoder,
        dino_version='v1',
        img_size=[224],
        patch_size=16,
        in_chans=384,
        num_classes=0,
        embed_dim=1024,  # Check ViT encoder dim
        depth=6,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        # norm_layer=nn.LayerNorm,
        norm_layer='nn.LayerNorm',
        out_chans=96,
        decoder_in_chans=32,
        triplane_in_chans=-1,
        decoder_output_dim=32,
        encoder_cls_token=False,
        decoder_cls_token=False,
        c_dim=25,  # Conditioning label (C) dimensionality.
        image_size=128,  # Output resolution.
        img_channels=3,  # Number of output color channels.
        rendering_kwargs={},
        load_pretrain_encoder=False,
        decomposed=True,
        triplane_size=224,
        ae_classname='ViTTriplaneDecomposed',
        use_clip=False,
        sr_kwargs={},
        sr_ratio=2,
        no_dim_up_mlp=False,
        dim_up_mlp_as_func=False,
        decoder_load_pretrained=True,
        uvit_skip_encoder=False,
        bcg_synthesis_kwargs={},
        # decoder params
        vae_p=1,
        ldm_z_channels=4,
        vae_dit_token_size=16,
        ldm_embed_dim=4,
        use_conf_map=False,
        triplane_fg_bg=False,
        flexicube_decoder=False,
        encoder_in_channels=3,
        sd_E_ch=64,
        z_channels=3 * 4,
        latent_num=768,
        sd_E_num_res_blocks=1,
        num_frames=4,
        arch_dit_decoder='DiT2-B/2',
        in_plane_attention=True,
        lrm_decoder=False,
        gs_rendering=False,
        surfel_rendering=False,
        return_all_dit_layers=False,
        plane_n=3,
        *args,
        **kwargs):

    # TODO, check pre-trained ViT encoder cfgs

    preprocess = None
    clip_dtype = None
    if load_pretrain_encoder:
        if not use_clip:
            if dino_version == 'v1':
                encoder = torch.hub.load(
                    'facebookresearch/dino:main',
                    'dino_{}{}'.format(arch_encoder, patch_size))
                logger.log(
                    f'loaded pre-trained dino v1 ViT-S{patch_size} encoder ckpt'
                )
            elif dino_version == 'v2':
                encoder = torch.hub.load(
                    'facebookresearch/dinov2',
                    'dinov2_{}{}'.format(arch_encoder, patch_size))
                logger.log(
                    f'loaded pre-trained dino v2 {arch_encoder}{patch_size} encoder ckpt'
                )
            elif 'sd' in dino_version:  # just for compat

                if 'mv' in dino_version:
                    if 'lgm' in dino_version:
                        encoder_cls = MVUNet(
                            input_size=256,
                            up_channels=(1024, 1024, 512, 256,
                                         128),  # one more decoder
                            up_attention=(True, True, True, False, False),
                            splat_size=128,
                            output_size=
                            512,  # render & supervise Gaussians at a higher resolution.
                            batch_size=8,
                            num_views=8,
                            gradient_accumulation_steps=1,
                            # mixed_precision='bf16',
                        )
                    elif 'gs' in dino_version:
                        encoder_cls = MVEncoder
                    else:
                        encoder_cls = MVEncoder

                else:
                    encoder_cls = Encoder

                encoder = encoder_cls(  # mono input
                    double_z=True,
                    resolution=256,
                    in_channels=encoder_in_channels,
                    # ch=128,
                    ch=64,  # ! fit in the memory
                    # ch_mult=[1,2,4,4],
                    # num_res_blocks=2,
                    ch_mult=[1, 2, 4, 4],
                    num_res_blocks=1,
                    dropout=0.0,
                    attn_resolutions=[],
                    out_ch=3,  # unused
                    z_channels=z_channels,
                )  # stable diffusion encoder
            else:
                raise NotImplementedError()

        else:
            import clip
            model, preprocess = clip.load("ViT-B/16", device=dist_util.dev())
            model.float()  # convert weight to float32
            clip_dtype = model.dtype
            encoder = getattr(
                model, 'visual')  # only use the CLIP visual encoder here
            encoder.requires_grad_(False)
            logger.log(
                f'loaded pre-trained CLIP ViT-B{patch_size} encoder, fixed.')

    elif 'sd' in dino_version:
        attn_kwargs = {}
        attn_type = "mv-vanilla"
        if 'mv' in dino_version:
            if 'lgm' in dino_version:
                encoder = LGM_MVEncoder(
                    in_channels=9,
                    # input_size=256,
                    up_channels=(1024, 1024, 512, 256,
                                 128),  # one more decoder
                    up_attention=(True, True, True, False, False),
                    # splat_size=128,
                    # output_size=
                    # 512,  # render & supervise Gaussians at a higher resolution.
                    # batch_size=8,
                    # num_views=8,
                    # gradient_accumulation_steps=1,
                    # mixed_precision='bf16',
                )

            elif 'srt' in dino_version:
                if 'hybrid' in dino_version:
                    encoder_cls = HybridEncoder # best overall performance
                # if 'vitl' in dino_version:
                #     encoder_cls = ImprovedSRTEncoderVAE_L5_vitl
                # elif 'l6' in dino_version:
                #     encoder_cls = ImprovedSRTEncoderVAE_L6
                # elif 'mlp4' in dino_version:
                #     if 'f8'in dino_version:
                #         # encoder_cls = ImprovedSRTEncoderVAE_mlp_ratio4_f8
                #         encoder_cls = ImprovedSRTEncoderVAE_mlp_ratio4_f8_L6
                elif 'l6' in dino_version:
                    encoder_cls = ImprovedSRTEncoderVAE_mlp_ratio4_L6
                elif 'heavy' in dino_version:
                    encoder_cls = ImprovedSRTEncoderVAE_mlp_ratio4_heavyPatchify

                elif 'decomposed' in dino_version:
                    encoder_cls = ImprovedSRTEncoderVAE_mlp_ratio4_decomposed
                elif 'pcd-structured' in dino_version:
                    attn_kwargs = {
                        'n_heads': 8,
                        'd_head': 64,
                    }
                    if 'nopcd' in dino_version:
                        encoder_cls = HybridEncoderPCDStructuredLatentSNoPCD
                    elif 'uniformfps' in dino_version:
                        encoder_cls = HybridEncoderPCDStructuredLatentUniformFPS
                    elif 'pixelaligned' in dino_version:
                        encoder_cls = HybridEncoderPCDStructuredLatentSNoPCD_PixelAlignedQuery
                    else:
                        encoder_cls = HybridEncoderPCDStructuredLatent
                else:
                    encoder_cls = ImprovedSRTEncoderVAE_mlp_ratio4 # best overall performance
                # else:  # default version
                #     encoder_cls = ImprovedSRTEncoderVAE

            elif 'gs' in dino_version:

                if 'dynaInp' in dino_version:
                    if 'ca' in dino_version:
                        encoder_cls = MVEncoderGSDynamicInp_CA
                    else:
                        encoder_cls = MVEncoderGSDynamicInp
                else:
                    encoder_cls = MVEncoderGS

                attn_kwargs = {
                    'n_heads': 8,
                    'd_head': 64,
                }

            else:
                if 'dynaInp' in dino_version:
                    if 'ca' in dino_version:
                        encoder_cls = MVEncoderGSDynamicInp_CA
                    else:
                        encoder_cls = MVEncoderGSDynamicInp
                else:
                    encoder_cls = MVEncoder
                attn_kwargs = {
                    'n_heads': 8,
                    'd_head': 64,
                }

        else:
            encoder_cls = Encoder

        if 'lgm' not in dino_version:  # TODO, for compat now
            # st()
            encoder = encoder_cls(
                double_z=True,
                resolution=256,
                in_channels=encoder_in_channels,
                # ch=128,
                # ch=64, # ! fit in the memory
                ch=sd_E_ch,
                # ch_mult=[1,2,4,4],
                # num_res_blocks=2,
                ch_mult=[1, 2, 4, 4],
                # num_res_blocks=1,
                num_res_blocks=sd_E_num_res_blocks,
                num_frames=num_frames,
                dropout=0.0,
                attn_resolutions=[],
                out_ch=3,  # unused
                z_channels=z_channels,  # 4 * 3
                attn_kwargs=attn_kwargs,
                attn_type=attn_type,
                latent_num=latent_num,
            )  # stable diffusion encoder

    else:
        encoder = vits.__dict__[arch_encoder](
            patch_size=patch_size,
            drop_path_rate=drop_path_rate,  # stochastic depth
            img_size=img_size)

    assert decomposed
    if decomposed:

        if not gs_rendering:
            if triplane_in_chans == -1:
                triplane_in_chans = decoder_in_chans

            if triplane_fg_bg:
                triplane_renderer_cls = Triplane_fg_bg_plane
            elif flexicube_decoder:
                triplane_renderer_cls = Triplane
            else:
                triplane_renderer_cls = Triplane

            # triplane_decoder = Triplane(
            triplane_decoder = triplane_renderer_cls(
                c_dim,  # Conditioning label (C) dimensionality.
                image_size,  # Output resolution.
                img_channels,  # Number of output color channels.
                rendering_kwargs=rendering_kwargs,
                out_chans=out_chans,
                # create_triplane=True,  # compatability, remove later
                triplane_size=triplane_size,
                decoder_in_chans=triplane_in_chans,
                decoder_output_dim=decoder_output_dim,
                sr_kwargs=sr_kwargs,
                bcg_synthesis_kwargs=bcg_synthesis_kwargs,
                lrm_decoder=lrm_decoder)
        elif surfel_rendering:
            triplane_decoder = GaussianRenderer2DGS(
                image_size, out_chans, rendering_kwargs=rendering_kwargs)
        else:
            triplane_decoder = GaussianRenderer(
                image_size, out_chans, rendering_kwargs=rendering_kwargs)

        if load_pretrain_encoder:

            if dino_version == 'v1':
                vit_decoder = torch.hub.load(
                    'facebookresearch/dino:main',
                    'dino_{}{}'.format(arch_decoder, patch_size))
                logger.log(
                    'loaded pre-trained decoder',
                    "facebookresearch/dino:main', 'dino_{}{}".format(
                        arch_decoder, patch_size))
            else:

                vit_decoder = torch.hub.load(
                    'facebookresearch/dinov2',
                    # 'dinov2_{}{}'.format(arch_decoder, patch_size))
                    'dinov2_{}{}'.format(arch_decoder, patch_size),
                    pretrained=decoder_load_pretrained)
                logger.log(
                    'loaded pre-trained decoder',
                    "facebookresearch/dinov2', 'dinov2_{}{}".format(
                        arch_decoder,
                        patch_size), 'pretrianed=', decoder_load_pretrained)

        elif 'dit' in dino_version:
            from dit.dit_decoder import DiT2_models, DiTBlock

            # st()
            vit_decoder = DiT2_models[arch_dit_decoder](
                input_size=16,
                num_classes=0,
                learn_sigma=False,
                in_channels=embed_dim,
                mixed_prediction=False,
                context_dim=None,  # add CLIP text embedding
                roll_out=True,
                plane_n=4 if ('gs' in dino_version
                              and 'trilatent' not in dino_version) else 3,
                return_all_layers=return_all_dit_layers,
                in_plane_attention=in_plane_attention,
                vit_blk=DiTBlock,
            )

        else:  # has bug on global token, to fix
            vit_decoder = vits.__dict__[arch_decoder](
                patch_size=patch_size,
                drop_path_rate=drop_path_rate,  # stochastic depth
                img_size=img_size)

        # decoder = ViTTriplaneDecomposed(vit_decoder, triplane_decoder)
        # if True:
        decoder_kwargs = dict(
            class_name=ae_classname,
            vit_decoder=vit_decoder,
            triplane_decoder=triplane_decoder,
            # encoder_cls_token=encoder_cls_token,
            cls_token=decoder_cls_token,
            sr_ratio=sr_ratio,
            vae_p=vae_p,
            ldm_z_channels=ldm_z_channels,
            ldm_embed_dim=ldm_embed_dim,
            vae_dit_token_size=vae_dit_token_size,
            plane_n=plane_n,
        )
        decoder = dnnlib.util.construct_class_by_name(**decoder_kwargs)
    else:
        # deprecated
        decoder = ViTTriplane(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            out_chans,
            cls_token,
            c_dim,  # Conditioning label (C) dimensionality.
            image_size,  # Output resolution.
            img_channels,  # Number of output color channels.
            # TODO, replace with c
            rendering_kwargs=rendering_kwargs,
        )
    # if return_encoder_decoder:
    #     return encoder, decoder, img_size[0], cls_token
    # else:

    if use_conf_map:
        confnet = ConfNet(cin=3, cout=1, nf=64, zdim=128)
    else:
        confnet = None

    auto_encoder = AE(
        encoder,
        decoder,
        img_size[0],
        encoder_cls_token,
        decoder_cls_token,
        preprocess,
        use_clip,
        dino_version,
        clip_dtype,
        no_dim_up_mlp=no_dim_up_mlp,
        dim_up_mlp_as_func=dim_up_mlp_as_func,
        uvit_skip_encoder=uvit_skip_encoder,
        confnet=confnet,
    )

    logger.log(auto_encoder)
    torch.cuda.empty_cache()

    return auto_encoder


# def create_3DAE_Diffusion_model(
#         arch_encoder,
#         arch_decoder,
#         img_size=[224],
#         patch_size=16,
#         in_chans=384,
#         num_classes=0,
#         embed_dim=1024,  # Check ViT encoder dim
#         depth=6,
#         num_heads=16,
#         mlp_ratio=4.,
#         qkv_bias=False,
#         qk_scale=None,
#         drop_rate=0.1,
#         attn_drop_rate=0.,
#         drop_path_rate=0.,
#         # norm_layer=nn.LayerNorm,
#         norm_layer='nn.LayerNorm',
#         out_chans=96,
#         decoder_in_chans=32,
#         decoder_output_dim=32,
#         cls_token=False,
#         c_dim=25,  # Conditioning label (C) dimensionality.
#         img_resolution=128,  # Output resolution.
#         img_channels=3,  # Number of output color channels.
#         rendering_kwargs={},
#         load_pretrain_encoder=False,
#         decomposed=True,
#         triplane_size=224,
#         ae_classname='ViTTriplaneDecomposed',
#         # return_encoder_decoder=False,
#         *args,
#         **kwargs
#         ):

#     # TODO, check pre-trained ViT encoder cfgs

#     encoder, decoder, img_size, cls_token = create_3DAE_model(
#         arch_encoder,
#         arch_decoder,
#         img_size,
#         patch_size,
#         in_chans,
#         num_classes,
#         embed_dim,  # Check ViT encoder dim
#         depth,
#         num_heads,
#         mlp_ratio,
#         qkv_bias,
#         qk_scale,
#         drop_rate,
#         attn_drop_rate,
#         drop_path_rate,
#         # norm_layer=nn.LayerNorm,
#         norm_layer,
#         out_chans=96,
#         decoder_in_chans=32,
#         decoder_output_dim=32,
#         cls_token=False,
#         c_dim=25,  # Conditioning label (C) dimensionality.
#         img_resolution=128,  # Output resolution.
#         img_channels=3,  # Number of output color channels.
#         rendering_kwargs={},
#         load_pretrain_encoder=False,
#         decomposed=True,
#         triplane_size=224,
#         ae_classname='ViTTriplaneDecomposed',
#         return_encoder_decoder=False,
#         *args,
#         **kwargs
#     ) # type: ignore


def create_Triplane(
        c_dim=25,  # Conditioning label (C) dimensionality.
        img_resolution=128,  # Output resolution.
        img_channels=3,  # Number of output color channels.
        rendering_kwargs={},
        decoder_output_dim=32,
        *args,
        **kwargs):

    decoder = Triplane(
        c_dim,  # Conditioning label (C) dimensionality.
        img_resolution,  # Output resolution.
        img_channels,  # Number of output color channels.
        # TODO, replace with c
        rendering_kwargs=rendering_kwargs,
        create_triplane=True,
        decoder_output_dim=decoder_output_dim)
    return decoder


def DiT_defaults():
    return {
        'dit_model': "DiT-B/16",
        'vae': "ema"
        # dit_model="DiT-XL/2",
        # dit_patch_size=8,
    }
