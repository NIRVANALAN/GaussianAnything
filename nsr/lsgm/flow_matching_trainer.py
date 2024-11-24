"""
https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/diffusion/ddpm.py#L30
"""
import random
import pytorch3d
import copy
import point_cloud_utils as pcu
import cv2
import matplotlib.pyplot as plt
import torch
import gc
import functools
import json
import os
from pathlib import Path
from pdb import set_trace as st
from typing import Any
from click import prompt
import einops
import blobfile as bf
import imageio
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision
from PIL import Image
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import trimesh
from nsr.camera_utils import generate_input_camera

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import LossAwareSampler, UniformSampler
# from .train_util import TrainLoop3DRec
from guided_diffusion.train_util import (TrainLoop, calc_average_loss,
                                         find_ema_checkpoint,
                                         find_resume_checkpoint,
                                         get_blob_logdir, log_loss_dict,
                                         log_rec3d_loss_dict,
                                         parse_resume_step_from_filename)
from guided_diffusion.gaussian_diffusion import ModelMeanType

from ldm.modules.encoders.modules import FrozenClipImageEmbedder, TextEmbedder, FrozenCLIPTextEmbedder, FrozenOpenCLIPImagePredictionEmbedder, FrozenOpenCLIPImageEmbedder

import dnnlib
from dnnlib.util import requires_grad
from dnnlib.util import calculate_adaptive_weight

from ..train_util_diffusion import TrainLoop3DDiffusion
from ..cvD.nvsD_canoD import TrainLoop3DcvD_nvsD_canoD

from guided_diffusion.continuous_diffusion_utils import get_mixed_prediction, different_p_q_objectives, kl_per_group_vada, kl_balancer
# from .train_util_diffusion_lsgm_noD_joint import TrainLoop3DDiffusionLSGMJointnoD  # joint diffusion and rec class
# from .controlLDM import TrainLoop3DDiffusionLSGM_Control  # joint diffusion and rec class
from .train_util_diffusion_lsgm_noD_joint import TrainLoop3DDiffusionLSGMJointnoD  # joint diffusion and rec class

# ! add new schedulers from https://github.com/Stability-AI/generative-models

from .crossattn_cldm import TrainLoop3DDiffusionLSGM_crossattn

# import SD stuffs
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
from omegaconf import ListConfig, OmegaConf
from sgm.modules import UNCONDITIONAL_CONFIG

from sgm.util import (default, disabled_train, get_obj_from_str,
                      instantiate_from_config, log_txt_as_img)

from transport import create_transport, Sampler
import math

# for gs rendering
from utils.gs_utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World
from utils.general_utils import matrix_to_quaternion
from utils.mesh_util import post_process_mesh, to_cam_open3d_compat
from datasets.g_buffer_objaverse import focal2fov, fov2focal

import open3d as o3d

# from sgm.sampling_utils.demo.streamlit_helpers import init_sampling

def sample_uniform_cameras_on_sphere(num_samples=1):
    # Step 1: Sample azimuth angles uniformly from [0, 2*pi)
    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    
    # Step 2: Sample cos(phi) uniformly from [-1, 1]
    cos_phi = np.random.uniform(-1, 1, num_samples)
    
    # Step 3: Calculate the elevation angle (phi) from cos(phi)
    phi = np.arccos(cos_phi)  # phi will be in [0, pi]
    
    # Step 4: Convert spherical coordinates to Cartesian coordinates (x, y, z)
    # x = np.sin(phi) * np.cos(theta)
    # y = np.sin(phi) * np.sin(theta)
    # z = np.cos(phi)
    
    # Combine the x, y, z coordinates into a single array
    # cameras = np.vstack((x, y, z)).T  # Shape: (num_samples, 3)
    
    # return cameras
    return theta, phi



class FlowMatchingEngine(TrainLoop3DDiffusionLSGM_crossattn):

    def __init__(
        self,
        *,
        rec_model,
        denoise_model,
        diffusion,
        sde_diffusion,
        control_model,
        control_key,
        only_mid_control,
        loss_class,
        data,
        eval_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        eval_interval,
        save_interval,
        resume_checkpoint,
        resume_cldm_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=0.001,
        schedule_sampler=None,
        weight_decay=0,
        lr_anneal_steps=0,
        iterations=10001,
        ignore_resume_opt=False,
        freeze_ae=False,
        denoised_ae=True,
        triplane_scaling_divider=10,
        use_amp=False,
        diffusion_input_size=224,
        normalize_clip_encoding=False,
        scale_clip_encoding=1,
        cfg_dropout_prob=0,
        cond_key='img_sr',
        use_eos_feature=False,
        compile=False,
        snr_type='lognorm',
        # denoiser_config,
        # conditioner_config: Union[None, Dict, ListConfig,
        #                           OmegaConf] = None,
        # sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        # loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        **kwargs):
        super().__init__(rec_model=rec_model,
                         denoise_model=denoise_model,
                         diffusion=diffusion,
                         sde_diffusion=sde_diffusion,
                         control_model=control_model,
                         control_key=control_key,
                         only_mid_control=only_mid_control,
                         loss_class=loss_class,
                         data=data,
                         eval_data=eval_data,
                         batch_size=batch_size,
                         microbatch=microbatch,
                         lr=lr,
                         ema_rate=ema_rate,
                         log_interval=log_interval,
                         eval_interval=eval_interval,
                         save_interval=save_interval,
                         resume_checkpoint=resume_checkpoint,
                         resume_cldm_checkpoint=resume_cldm_checkpoint,
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         schedule_sampler=schedule_sampler,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         ignore_resume_opt=ignore_resume_opt,
                         freeze_ae=freeze_ae,
                         denoised_ae=denoised_ae,
                         triplane_scaling_divider=triplane_scaling_divider,
                         use_amp=use_amp,
                         diffusion_input_size=diffusion_input_size,
                         normalize_clip_encoding=normalize_clip_encoding,
                         scale_clip_encoding=scale_clip_encoding,
                         cfg_dropout_prob=cfg_dropout_prob,
                         cond_key=cond_key,
                         use_eos_feature=use_eos_feature,
                         compile=compile,
                         **kwargs)

        #  ! sgm diffusion pipeline
        # ! reuse the conditioner
        self.snr_type = snr_type
        self.latent_key = 'latent'

        if self.cond_key == 'caption': # ! text pretrain
            if snr_type == 'stage2-t23d': 
                ldm_configs = OmegaConf.load(
                    'sgm/configs/stage2-t23d.yaml')['ldm_configs']
            elif snr_type == 'stage1-t23d': 
                ldm_configs = OmegaConf.load(
                    'sgm/configs/stage1-t23d.yaml')['ldm_configs']
                self.latent_key = 'normalized-fps-xyz' # learn xyz diff
            else: # just simple t23d, no xyz condition
                ldm_configs = OmegaConf.load(
                    'sgm/configs/t23d-clipl-compat-fm.yaml')['ldm_configs']
        else: # 

            # assert 'lognorm' in snr_type
            if snr_type == 'lognorm': # by default
                ldm_configs = OmegaConf.load(
                    'sgm/configs/img23d-clipl-compat-fm-lognorm.yaml')['ldm_configs']
            # st()
            # if snr_type == 'lognorm-highres': # by default
            elif snr_type == 'img-uniform-gvp': # by default
                ldm_configs = OmegaConf.load(
                    'sgm/configs/img23d-clipl-compat-fm-lognorm-336-uniform.yaml')['ldm_configs']
                # self.latent_key = 'fps-xyz' # xyz diffusion
                self.latent_key = 'normalized-fps-xyz' # to std

            elif snr_type == 'img-uniform-gvp-dino': # by default
                ldm_configs = OmegaConf.load(
                    'sgm/configs/img23d-clipl-compat-fm-lognorm-480-uniform-clay-dinoonly.yaml')['ldm_configs']
                self.latent_key = 'normalized-fps-xyz' # to std

            # elif snr_type == 'img-uniform-gvp-dino-xl': # by default
            #     ldm_configs = OmegaConf.load(
            #         'sgm/configs/img23d-clipl-compat-fm-lognorm-480-uniform-clay-dinoonly.yaml')['ldm_configs']
            #     self.latent_key = 'normalized-fps-xyz' # to std

            elif snr_type == 'img-uniform-gvp-dino-stage2': # by default
                ldm_configs = OmegaConf.load(
                    'sgm/configs/stage2-i23d.yaml')['ldm_configs']
                # self.latent_key = 'normalized-fps-xyz' # to std

            elif snr_type == 'img-uniform-gvp-clay': # contains both text and image condition
                ldm_configs = OmegaConf.load(
                    'sgm/configs/img23d-clipl-compat-fm-lognorm-480-uniform-clay.yaml')['ldm_configs']
                # self.latent_key = 'fps-xyz' # xyz diffusion
                self.latent_key = 'normalized-fps-xyz' # to std

            elif snr_type == 'pcd-cond-tex':
                ldm_configs = OmegaConf.load(
                    'sgm/configs/img23d-clipl-compat-fm-lognorm-336-uniform-pcdcond.yaml')['ldm_configs']
                    # 'sgm/configs/img23d-clipl-compat-fm-lognorm-336.yaml')['ldm_configs']

            # ! stage-2 text-xyz conditioned
            elif snr_type == 'stage2-t23d':
                ldm_configs = OmegaConf.load(
                    'sgm/configs/stage2-t23d.yaml')['ldm_configs']

            elif snr_type == 'lognorm-mv':
                ldm_configs = OmegaConf.load(
                    'sgm/configs/mv23d-clipl-compat-fm-lognorm.yaml')['ldm_configs']
            
            # ! mv version
            elif snr_type == 'lognorm-mv-plucker':
                ldm_configs = OmegaConf.load(
                    'sgm/configs/mv23d-plucker-clipl-compat-fm-lognorm-noclip.yaml')['ldm_configs']
                    # 'sgm/configs/mv23d-plucker-clipl-compat-fm-lognorm.yaml')['ldm_configs']

            elif snr_type == 'stage1-mv-t23dpt':
                self.latent_key = 'normalized-fps-xyz' # learn xyz diff
                ldm_configs = OmegaConf.load(
                    'sgm/configs/stage1-mv23d-t23dpt.yaml')['ldm_configs']

            elif snr_type == 'stage1-mv-i23dpt':
                self.latent_key = 'normalized-fps-xyz' # learn xyz diff
                ldm_configs = OmegaConf.load(
                    'sgm/configs/stage1-mv23d-i23dpt.yaml')['ldm_configs']

            elif snr_type == 'stage1-mv-i23dpt-noi23d':
                self.latent_key = 'normalized-fps-xyz' # learn xyz diff
                ldm_configs = OmegaConf.load(
                    'sgm/configs/stage1-mv23d-i23dpt-noi23d.yaml')['ldm_configs']

            elif snr_type == 'stage2-mv-i23dpt':
                # self.latent_key = 'normalized-fps-xyz' # learn xyz diff
                ldm_configs = OmegaConf.load(
                    'sgm/configs/stage2-mv23d-i23dpt.yaml')['ldm_configs']


            else:
                ldm_configs = OmegaConf.load(
                    'sgm/configs/img23d-clipl-compat-fm.yaml')['ldm_configs']

        self.loss_fn = (
            instantiate_from_config(ldm_configs.loss_fn_config)
            # if loss_fn_config is not None
            # else None
        )

        # self.denoiser = instantiate_from_config(
        #     ldm_configs.denoiser_config).to(dist_util.dev())

        self.transport_sampler = Sampler(self.loss_fn.transport, guider_config=ldm_configs.guider_config)

        self.conditioner = instantiate_from_config(
            default(ldm_configs.conditioner_config,
                    UNCONDITIONAL_CONFIG)).to(dist_util.dev())

        # ! setup optimizer (with cond embedder params here)
        self._set_grad_flag()
        self._setup_opt2()
        self._load_model2()


    def _set_grad_flag(self):    
        requires_grad(self.ddpm_model, True) # do not change this flag during training.

    def _setup_opt(self):
        pass # see below

    def _setup_opt2(self):
        # ! add trainable conditioner parameters
        # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/diffusion.py#L219

        # params = list(self.ddpm_model.parameters())

        # https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/7
        self.opt = AdamW([{
            'name': 'ddpm',
            # 'params': self.ddpm_model.parameters(),
            'params': filter(lambda p: p.requires_grad, self.ddpm_model.parameters()), # if you want to freeze some layers
        },
        ],
                         lr=self.lr,
                         weight_decay=self.weight_decay)
        
        
        embedder_params = []
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                embedder_params = embedder_params + list(embedder.parameters())


        if len(embedder_params) != 0:
            self.opt.add_param_group(
                {
                    'name': 'embedder',
                    'params': embedder_params,
                    'lr': self.lr*0.5, # smaller lr to finetune dino/clip
                }
            )
        
        print(self.opt)

    def save(self, mp_trainer=None, model_name='ddpm'):
        # save embedder params also
        super().save(mp_trainer, model_name)

        # save embedder ckpt
        if dist_util.get_rank() == 0:
            for embedder in self.conditioner.embedders:
                if embedder.is_trainable:
                    # embedder_params = embedder_params + list(embedder.parameters())
                    model_name = embedder.__class__.__name__
                    filename = f"embedder_{model_name}{(self.step+self.resume_step):07d}.pt"
                    with bf.BlobFile(bf.join(get_blob_logdir(), filename),
                                        "wb") as f:
                        th.save(embedder.state_dict(), f)

        dist_util.synchronize()

    def _load_model2(self):

        # ! load embedder
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                # embedder_params = embedder_params + list(embedder.parameters())
                model_name = embedder.__class__.__name__
                filename = f"embedder_{model_name}{(self.step+self.resume_step):07d}.pt"
                # embedder_FrozenDinov2ImageEmbedderMV2115000.pt

                # with bf.BlobFile(bf.join(get_blob_logdir(), filename),
                #                     "wb") as f:
                #     th.save(embedder.state_dict(), f)

                split = self.resume_checkpoint.split("model")
                resume_checkpoint = str(
                    Path(split[0]) / filename)
                if os.path.exists(resume_checkpoint):
                    if dist.get_rank() == 0:
                        logger.log(
                            f"loading cond embedder from checkpoint: {resume_checkpoint}...")
                        # if model is None:
                        #     model = self.model
                        embedder.load_state_dict(
                            dist_util.load_state_dict(
                                resume_checkpoint,
                                map_location=dist_util.dev(),
                            ))
                else:
                    logger.log(f'{resume_checkpoint} not found.')

                if dist_util.get_world_size() > 1:
                    dist_util.sync_params(embedder.parameters())


    def instantiate_cond_stage(self, normalize_clip_encoding,
                               scale_clip_encoding, cfg_dropout_prob,
                               use_eos_feature):
        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/diffusion/ddpm.py#L509C1-L509C46
        # self.cond_stage_model.train = disabled_train  # type: ignore
        # if self.cond_key == 'caption':
        #     self.cond_txt_model = TextEmbedder(dropout_prob=cfg_dropout_prob,
        #                                        use_eos_feature=use_eos_feature)
        # elif self.cond_key == 'img':
        #     self.cond_img_model = FrozenOpenCLIPImagePredictionEmbedder(
        #         1, 1,
        #         FrozenOpenCLIPImageEmbedder(freeze=True,
        #                                     device=dist_util.dev(),
        #                                     init_device=dist_util.dev()))

        # else:  # zero-shot Text to 3D using normalized clip latent
        #     self.cond_stage_model = FrozenClipImageEmbedder(
        #         'ViT-L/14',
        #         dropout_prob=cfg_dropout_prob,
        #         normalize_encoding=normalize_clip_encoding,
        #         scale_clip_encoding=scale_clip_encoding)
        #     self.cond_stage_model.freeze()

        #     self.cond_txt_model = FrozenCLIPTextEmbedder(
        #         dropout_prob=cfg_dropout_prob,
        #         scale_clip_encoding=scale_clip_encoding)
        #     self.cond_txt_model.freeze()
        pass # initialized in the self.__init__() using SD api



    # ! already merged
    def prepare_ddpm(self, eps, mode='p'):
        raise NotImplementedError('already implemented in self.denoiser')

    # merged from noD.py

    # use sota denoiser, loss_fn etc.
    def ldm_train_step(self, batch, behaviour='cano', *args, **kwargs):

        # ! enable the gradient of both models
        # requires_grad(self.ddpm_model, True)
        self._set_grad_flag() # more flexible

        self.mp_trainer.zero_grad()  # !!!!

        if 'img' in batch:
            batch_size = batch['img'].shape[0]
        else:
            batch_size = len(batch['caption'])

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v[i:i+self.microbatch]
                for k, v in batch.items()
            }

            # move condition to self.dtype
            # =================================== ae part ===================================
            # with th.cuda.amp.autocast(dtype=th.bfloat16,
            with th.cuda.amp.autocast(dtype=self.dtype,
                                      enabled=self.mp_trainer.use_amp):

                loss = th.tensor(0.).to(dist_util.dev())

                assert 'latent' in micro
                # st() # torchvision.utils.save_image(micro['img'], 'tmp/img.png', normalize=True, value_range=(-1,1))
                # vae_out = {self.latent_name: micro['latent']}
                # else:
                #     vae_out = self.ddp_rec_model(
                #         img=micro['img_to_encoder'],
                #         c=micro['c'],
                #         behaviour='encoder_vae',
                #     )  # pred: (B, 3, 64, 64)

                # eps = vae_out[self.latent_name] / self.triplane_scaling_divider
                # ! if training xyz only
                # eps = vae_out[self.latent_name][..., -3:] / self.triplane_scaling_divider

                # ! if training texture only
                eps = micro[self.latent_key] / self.triplane_scaling_divider

                if self.cond_key == 'img-c':
                    micro['img-c'] = {
                        # 'img': micro['img'].to(self.dtype),
                        'img': micro['mv_img'].to(self.dtype), # for compat issue
                        'c': micro['c'].to(self.dtype),
                    }

                    # log_rec3d_loss_dict({
                    #     f"mv-alpha/{i}": self.ddpm_model.blocks[i].mv_alpha[0] for i in range(len(self.ddpm_model.blocks))
                    # })


                loss, loss_other_info = self.loss_fn(self.ddp_ddpm_model,
                                                    #  self.denoiser,
                                                     self.conditioner, 
                                                     eps.to(self.dtype),
                                                     micro)  # type: ignore
                loss = loss.mean()
                # log_rec3d_loss_dict({})

                log_rec3d_loss_dict({
                    # 'eps_mean':
                    # eps.mean(),
                    # 'eps_std':
                    # eps.std([1, 2, 3]).mean(0),
                    # 'pred_x0_std':
                    # loss_other_info['model_output'].std([1, 2, 3]).mean(0),
                    "p_loss":
                    loss,
                })

            self.mp_trainer.backward(loss)  # joint gradient descent

        # update ddpm accordingly
        self.mp_trainer.optimize(self.opt)

        # ! directly eval_cldm() for sampling.
        # if dist_util.get_rank() == 0 and self.step % 500 == 0:
        #     self.log_control_images(vae_out, micro, loss_other_info)

    @th.inference_mode()
    def log_control_images(self, vae_out, micro, ddpm_ret):

        if 'posterior' in vae_out:
            vae_out.pop('posterior')  # for calculating kl loss
        vae_out_for_pred = {self.latent_name: vae_out[self.latent_name][0:1].to(self.dtype)}

        with th.cuda.amp.autocast(dtype=self.dtype,
                                    enabled=self.mp_trainer.use_amp):
            pred = self.ddp_rec_model(latent=vae_out_for_pred,
                                    c=micro['c'][0:1],
                                    behaviour=self.render_latent_behaviour)

        assert isinstance(pred, dict)

        pred_img = pred['image_raw']
        if 'img' in micro:
            gt_img = micro['img']
        else:
            gt_img = th.zeros_like(pred['image_raw'])

        if 'depth' in micro:
            gt_depth = micro['depth']
            if gt_depth.ndim == 3:
                gt_depth = gt_depth.unsqueeze(1)
            gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                      gt_depth.min())
        else:
            gt_depth = th.zeros_like(gt_img[:, 0:1, ...])

        if 'image_depth' in pred:
            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())
        else:
            pred_depth = th.zeros_like(gt_depth)

        gt_img = self.pool_128(gt_img)
        gt_depth = self.pool_128(gt_depth)
        # cond = self.get_c_input(micro)
        # hint = th.cat(cond['c_concat'], 1)

        gt_vis = th.cat(
            [
                gt_img,
                gt_img,
                gt_img,
                # self.pool_128(hint),
                # gt_img,
                gt_depth.repeat_interleave(3, dim=1)
            ],
            dim=-1)[0:1]  # TODO, fail to load depth. range [0, 1]

        # eps_t_p_3D = eps_t_p.reshape(batch_size, eps_t_p.shape[1]//3, 3, -1) # B C 3 L

        # self.sampler

        noised_latent, sigmas, x_start = [
            ddpm_ret[k] for k in ['noised_input', 'sigmas', 'model_output']
        ]

        noised_latent = {
            'latent_normalized_2Ddiffusion':
            noised_latent[0:1].to(self.dtype) * self.triplane_scaling_divider,
        }

        denoised_latent = {
            'latent_normalized_2Ddiffusion':
            x_start[0:1].to(self.dtype) * self.triplane_scaling_divider,
        }
         
        with th.cuda.amp.autocast(dtype=self.dtype,
                                    enabled=self.mp_trainer.use_amp):
            noised_ae_pred = self.ddp_rec_model(
                img=None,
                c=micro['c'][0:1],
                latent=noised_latent,
                behaviour=self.render_latent_behaviour)

            # pred_x0 = self.sde_diffusion._predict_x0_from_eps(
            # eps_t_p, pred_eps_p, logsnr_p)  # for VAE loss, denosied latent

            # pred_xstart_3D
            denoised_ae_pred = self.ddp_rec_model(
                img=None,
                c=micro['c'][0:1],
                latent=denoised_latent,
                # latent=pred_x0[0:1] * self.
                # triplane_scaling_divider,  # TODO, how to define the scale automatically?
                behaviour=self.render_latent_behaviour)

        pred_vis = th.cat(
            [
                self.pool_128(img) for img in (
                    pred_img[0:1],
                    noised_ae_pred['image_raw'][0:1],
                    denoised_ae_pred['image_raw'][0:1],  # controlnet result
                    pred_depth[0:1].repeat_interleave(3, dim=1))
            ],
            dim=-1)  # B, 3, H, W

        if 'img' in micro:
            vis = th.cat([gt_vis, pred_vis],
                         dim=-2)[0].permute(1, 2,
                                            0).cpu()  # ! pred in range[-1, 1]
        else:
            vis = pred_vis[0].permute(1, 2, 0).cpu()

        # vis_grid = torchvision.utils.make_grid(vis) # HWC
        vis = vis.numpy() * 127.5 + 127.5
        vis = vis.clip(0, 255).astype(np.uint8)
        img_save_path = f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{sigmas[0].item():3}.jpg'
        Image.fromarray(vis).save(img_save_path)

        # if self.cond_key == 'caption':
        #     with open(f'{logger.get_dir()}/{self.step+self.resume_step}caption_{t_p[0].item():3}.txt', 'w') as f:
        #         f.write(micro['caption'][0])

        print('log denoised vis to: ', img_save_path)

        th.cuda.empty_cache()

    @th.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        use_cfg=True,
        # cfg_scale=4, # default value in SiT
        # cfg_scale=1.5, # default value in SiT
        cfg_scale=4.0, # default value in SiT
        seed=42,
        **kwargs,
    ):
        # self.sampler
        sample_fn = self.transport_sampler.sample_ode(num_steps=250, cfg=True) # default ode sampling setting.


        th.manual_seed(seed) # to reproduce result
        zs = th.randn(batch_size, *shape).to(dist_util.dev()).to(self.dtype)
        # st()
        assert use_cfg
        # sample_model_kwargs = {'uc': uc, 'cond': cond}       
        model_fn = self.ddpm_model.forward_with_cfg # default

        # ! prepare_inputs in VanillaCFG, for compat issue
        c_out = {}
        for k in cond:
            # if k in ["vector", "crossattn", "concat", 'fps-xyz']:
            c_out[k] = th.cat((cond[k], uc[k]), 0)
            # else:
            #     assert cond[k] == uc[k]
            #     c_out[k] = cond[k]
        sample_model_kwargs = {'context': c_out, 'cfg_scale': cfg_scale}
        zs = th.cat([zs, zs], 0)

        with th.cuda.amp.autocast(dtype=self.dtype,
                                    enabled=self.mp_trainer.use_amp):

            samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        # return samples
        return samples * self.triplane_scaling_divider

    @th.inference_mode()
    def eval_cldm(
        self,
        prompt="",
        # use_ddim=False,
        # unconditional_guidance_scale=1.0,
        unconditional_guidance_scale=4.0,
        seed=42,
        save_img=False,
        use_train_trajectory=False,
        camera=None,
        num_samples=1,
        num_instances=1,
        export_mesh=False,
    ):
        # ! slightly modified for new API. combined with
        # /cpfs01/shared/V2V/V2V_hdd/yslan/Repo/generative-models/sgm/models/diffusion.py:249 log_images()
        # TODO, support batch_size > 1

        self.ddpm_model.eval()

        args = dnnlib.EasyDict(
            dict(
                batch_size=1,
                image_size=self.diffusion_input_size,
                denoise_in_channels=self.rec_model.decoder.triplane_decoder.
                out_chans,  # type: ignore
                clip_denoised=False,
                class_cond=False))

        model_kwargs = {}

        uc = None
        log = dict()

        ucg_keys = [self.cond_key] # i23d

        # if self.cond_key == 'caption':
        if self.cond_key in ['caption', 'img-xyz']:
            # batch_c = {self.cond_key: prompt}
            # batch_c = {self.cond_key: prompt}
            batch_c = next(self.data) # ! use training set to evaluate t23d for now.
        elif self.cond_key == 'img-caption':
            batch_c = {'caption': prompt, 'img': batch['img'].to(dist_util.dev()).to(self.dtype)}
        else: 
            batch = next(self.data) # random cond here
            if self.cond_key == 'img-c':
                batch_c = {
                    self.cond_key: {
                        # 'img': batch['img'].to(self.dtype).to(dist_util.dev()),
                        'img': batch['mv_img'].to(self.dtype).to(dist_util.dev()),
                        'c': batch['c'].to(self.dtype).to(dist_util.dev()),
                    },
                    'img': batch['img'].to(self.dtype).to(dist_util.dev()) # required by clip
                }

            else:
                batch_c = {self.cond_key: batch[self.cond_key].to(dist_util.dev()).to(self.dtype)}

        with th.cuda.amp.autocast(dtype=self.dtype,
                                    enabled=self.mp_trainer.use_amp):

            c, uc = self.conditioner.get_unconditional_conditioning(
                batch_c,
                force_uc_zero_embeddings=ucg_keys
                if len(self.conditioner.embedders) > 0 else [],
            )

        sampling_kwargs = {'seed': seed, 'cfg_scale': unconditional_guidance_scale}

        N = 3  # hard coded, to update
        z_shape = (
            N,
            self.ddpm_model.in_channels if not self.ddpm_model.roll_out else
            3 * self.ddpm_model.in_channels,  # type: ignore
            self.diffusion_input_size,
            self.diffusion_input_size)

        for k in c:
            if isinstance(c[k], th.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(dist_util.dev()),
                                  (c, uc))
    
        samples = self.sample(c,
                              shape=z_shape[1:],
                              uc=uc,
                              batch_size=N,
                              **sampling_kwargs)
        # st() # do rendering first


        # ! get c
        if 'img' in self.cond_key:
            img_save_path = f'{logger.get_dir()}/{self.step+self.resume_step}_imgcond.jpg'
            if 'c' in self.cond_key:
                torchvision.utils.save_image(batch_c['img'][0], img_save_path, value_range=(-1,1), normalize=True, padding=0) # torch.Size([24, 6, 3, 256, 256])
                th.save(batch_c['img-c']['c'][0], f'{logger.get_dir()}/{self.step+self.resume_step}_c.pt')
            else:
                torchvision.utils.save_image(batch_c['img'][0:1], img_save_path, value_range=(-1,1), normalize=True, padding=0)

        assert camera is not None
        batch = {'c': camera.clone()}

        # else:
        #     if use_train_trajectory:
        #         batch = next(iter(self.data))
        #     else:
        #         try:
        #             batch = next(self.eval_data)
        #         except Exception as e:
        #             self.eval_data = iter(self.eval_data)
        #             batch = next(self.eval_data)

        #     if camera is not None:
        #         batch['c'] = camera.clone()


        # rendering
        for i in range(samples.shape[0]):
            th.cuda.empty_cache()

            # ! render sampled latent
            name_prefix = f'{self.step + self.resume_step}_{i}'

            if self.cond_key == 'caption':
                name_prefix = f'{name_prefix}_{prompt}'

            with th.cuda.amp.autocast(dtype=self.dtype,
                                        enabled=self.mp_trainer.use_amp):

                self.render_video_given_triplane(
                    samples[i:i+1].to(self.dtype), # default version
                    self.rec_model,  # compatible with join_model
                    name_prefix=name_prefix,
                    save_img=save_img,
                    render_reference=batch,
                   export_mesh=False)

        self.ddpm_model.train()


class FlowMatchingEngine_gs(FlowMatchingEngine):

    def __init__(
        self,
        *,
        rec_model,
        denoise_model,
        diffusion,
        sde_diffusion,
        control_model,
        control_key,
        only_mid_control,
        loss_class,
        data,
        eval_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        eval_interval,
        save_interval,
        resume_checkpoint,
        resume_cldm_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=0.001,
        schedule_sampler=None,
        weight_decay=0,
        lr_anneal_steps=0,
        iterations=10001,
        ignore_resume_opt=False,
        freeze_ae=False,
        denoised_ae=True,
        triplane_scaling_divider=10,
        use_amp=False,
        diffusion_input_size=224,
        normalize_clip_encoding=False,
        scale_clip_encoding=1,
        cfg_dropout_prob=0,
        cond_key='img_sr',
        use_eos_feature=False,
        compile=False,
        snr_type='lognorm',
        **kwargs):
        super().__init__(rec_model=rec_model,
                         denoise_model=denoise_model,
                         diffusion=diffusion,
                         sde_diffusion=sde_diffusion,
                         control_model=control_model,
                         control_key=control_key,
                         only_mid_control=only_mid_control,
                         loss_class=loss_class,
                         data=data,
                         eval_data=eval_data,
                         batch_size=batch_size,
                         microbatch=microbatch,
                         lr=lr,
                         ema_rate=ema_rate,
                         log_interval=log_interval,
                         eval_interval=eval_interval,
                         save_interval=save_interval,
                         resume_checkpoint=resume_checkpoint,
                         resume_cldm_checkpoint=resume_cldm_checkpoint,
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         schedule_sampler=schedule_sampler,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         ignore_resume_opt=ignore_resume_opt,
                         freeze_ae=freeze_ae,
                         denoised_ae=denoised_ae,
                         triplane_scaling_divider=triplane_scaling_divider,
                         use_amp=use_amp,
                         diffusion_input_size=diffusion_input_size,
                         normalize_clip_encoding=normalize_clip_encoding,
                         scale_clip_encoding=scale_clip_encoding,
                         cfg_dropout_prob=cfg_dropout_prob,
                         cond_key=cond_key,
                         use_eos_feature=use_eos_feature,
                         compile=compile,
                         snr_type=snr_type,
                         **kwargs)

        self.gs_bg_color=th.tensor([1,1,1], dtype=th.float32, device=dist_util.dev())
        self.latent_name = 'latent_normalized'  # normalized triplane latent
        # self.pcd_unnormalize_fn = lambda x: x.clip(-1,1) * 0.45 # [-1,1] -> [-0.45, 0.45] as in g-buffer dataset.
        # self.pcd_unnormalize_fn = lambda x: (x * 0.1862).clip(-0.45, 0.45) # [-1,1] -> [-0.45, 0.45] as in g-buffer dataset.
        
        # /cpfs01/user/lanyushi.p/logs/nips24/LSGM/t23d/FM/9cls/gs/i23d/dit-b/gpu4-batch32-lr1e-4-gs_surf_latent_224-drop0.33-same
        # self.pcd_unnormalize_fn = lambda x: (x * 0.158).clip(-0.45, 0.45) # [-1,1] -> [-0.45, 0.45] as in g-buffer dataset.

        # self.feat_scale_factor = th.Tensor([0.99227685, 1.014337  , 0.20842505, 0.98727155, 0.3305389 ,
        #     0.38729668, 1.0155401 , 0.9728264 , 1.0009694 , 0.97328585,
        #     0.2881106 , 0.1652732 , 0.3482468 , 0.9971449 , 0.99895126,
        #     0.18491288]).float().reshape(1,1,-1)

        # stat for normalization
        # self.xyz_mean = torch.Tensor([-0.00053714, 0.08095618, -0.01914407] ).reshape(1, 3).float()
        # self.xyz_std = th.Tensor([0.14593576, 0.15753542, 0.18873914] ).reshape(1,3).float().to(dist_util.dev())
        self.xyz_std = 0.164

        # ! for debug
        self.kl_mean = th.Tensor([ 0.0184,  0.0024,  0.0926,  0.0517,  0.1781,  0.7137, -0.0355,  0.0267,
         0.0183,  0.0164, -0.5090,  0.2406,  0.2733, -0.0256, -0.0285,  0.0761]).reshape(1,16).float().to(dist_util.dev())
        self.kl_std = th.Tensor([1.0018, 1.0309, 1.3001, 1.0160, 0.8182, 0.8023, 1.0591, 0.9789, 0.9966,
        0.9448, 0.8908, 1.4595, 0.7957, 0.9871, 1.0236, 1.2923]).reshape(1,16).float().to(dist_util.dev())

        # ! for surfel-gs rendering
        self.zfar = 100.0
        self.znear = 0.01

    def unnormalize_pcd_act(self, x):
        return x * self.xyz_std

    def unnormalize_kl_feat(self, latent):
        # return latent / self.feat_scale_factor
        # return (latent-self.kl_mean) / self.kl_std
        return (latent * self.kl_std) + self.kl_mean
    
    # def unnormalize_kl_feat(self, latent):
    #     return latent * self.feat_scale_factor

    @th.inference_mode()
    def eval_cldm(
        self,
        prompt="Yellow rubber duck",
        # use_ddim=False,
        # unconditional_guidance_scale=1.0,
        save_img=False,
        use_train_trajectory=False,
        camera=None,
        num_samples=1,
        num_instances=1,
        export_mesh=False,
    ):
        self.ddpm_model.eval()

        args = dnnlib.EasyDict(
            dict(
                batch_size=1,
                image_size=self.diffusion_input_size,
                denoise_in_channels=self.rec_model.decoder.triplane_decoder.
                out_chans,  # type: ignore
                clip_denoised=False,
                class_cond=False))

        model_kwargs = {}

        uc = None
        log = dict()

        ucg_keys = [self.cond_key] # i23d

        if self.cond_key == 'caption':
            if prompt == '':
                batch = next(self.data) # random cond here
                batch_c = {self.cond_key: prompt,
                'fps-xyz': batch['fps-xyz'].to(self.dtype).to(dist_util.dev()), 
                }
            else:
                # ! TODO, update the cascaded generation fps-xyz loading. Manual load for now.
                batch_c = {
                    self.cond_key: prompt
                }
                if self.latent_key == 'latent': # stage 2
                    # hard-coded path for now
                    # fps_xyz_output_prefix = '/nas/shared/public/yslan/logs/nips24/LSGM/t23d/FM/9cls/gs-disentangle/cascade_check/clay/stage1/eval/clip_text/50w-iter/'
                    # stage1_num_steps = '500000'

                    batch = next(self.data) # random cond here
                    batch_c[self.cond_key] = batch[self.cond_key][0:1] # colorizing GT xyz
                    gt_xyz = batch['fps-xyz'][0:1]
                    gt_kl_latent = batch['latent'][0:1]

                    # cascaded = False
                    # st()
                    # if self.step % 1e4 == 0:
                    #     cascaded = True
                    # else:

                    cascaded = False
                    prompt = batch[self.cond_key][0:1] # replace with on-the-fly GT point clouds
                    
                    batch_c[self.cond_key] = prompt

                    # ! for logging two-stage cascaded result. change the path to your stage-1 output pcd logdir.
                    if cascaded: # ! use stage-1 as output
                        # fps_xyz_output_prefix = '/nas/shared/public/yslan/logs/nips24/LSGM/t23d/FM/9cls/gs-disentangle/cascade_check/clay/stage1/eval/clip_text/60w-iter/'
                        fps_xyz_output_prefix = ''
                        stage1_num_steps = '600000'

                        fps_xyz = torch.from_numpy(pcu.load_mesh_v(f'{fps_xyz_output_prefix}/{stage1_num_steps}_0_{prompt}.ply') ).clip(-0.45,0.45).unsqueeze(0)

                        batch_c.update({
                            'fps-xyz': fps_xyz.to(self.dtype).to(dist_util.dev())
                        })
                    else:
                        # use gt as condition
                        # batch_c = {k: v[0:1].to(self.dtype).to(dist_util.dev()) for k, v in batch_c.items() if k in [self.cond_key, 'fps-xyz']}
                        for k in ['fps-xyz']:
                            batch_c[k] = batch[k][0:1].to(self.dtype).to(dist_util.dev())
                        batch_c[self.cond_key] = prompt
        else: 
            batch = next(self.data) # random cond here

            #! debugging, get GT xyz and KL latent for disentangled debugging

            if self.cond_key == 'img-c':
                prompt = batch['caption'][0:1]
                batch_c = {
                    self.cond_key: {
                        'img': batch['mv_img'][0:1].to(self.dtype).to(dist_util.dev()),
                        'c': batch['c'][0:1].to(self.dtype).to(dist_util.dev()),
                    },
                    'img': batch['img'][0:1].to(self.dtype).to(dist_util.dev()),
                    'caption': prompt,
                    'fps-xyz': batch['fps-xyz'][0:1].to(self.dtype).to(dist_util.dev())
                }
            
            elif self.cond_key == 'img-caption':
                batch_c = {'caption': prompt, 'img': batch['img'].to(dist_util.dev()).to(self.dtype)}

            elif self.cond_key == 'img-xyz':
                # load local xyz here
                # fps_xyz = torch.from_numpy(pcu.load_mesh_v('/cpfs01/user/lanyushi.p/Repo/diffusion-3d/tmp/sampled-0.ply') ).clip(-0.45,0.45).unsqueeze(0)
                # fps_xyz = torch.from_numpy(pcu.load_mesh_v('/cpfs01/user/lanyushi.p/Repo/diffusion-3d/tmp/sampled-2.ply') ).clip(-0.45,0.45).unsqueeze(0)
                # fps_xyz = torch.from_numpy(pcu.load_mesh_v('/cpfs01/user/lanyushi.p/Repo/diffusion-3d/tmp/sampled-1.ply') ).clip(-0.45,0.45).unsqueeze(0)
                # fps_xyz = torch.from_numpy(pcu.load_mesh_v('/cpfs01/user/lanyushi.p/Repo/diffusion-3d/tmp/sampled-3.ply') ).clip(-0.45,0.45).unsqueeze(0)
                # fps_xyz = torch.from_numpy(pcu.load_mesh_v('/nas/shared/V2V/yslan/logs/nips24/LSGM/t23d/FM/9cls/gs-disentangle/cascade_check/xyz_output_fullset_stillclip_but448_eval/1725000_0_0.ply') ).clip(-0.45,0.45).unsqueeze(0)
                # fps_xyz = torch.from_numpy(pcu.load_mesh_v('/nas/shared/V2V/yslan/logs/nips24/LSGM/t23d/FM/9cls/gs-disentangle/cascade_check/xyz_output_fullset_stillclip_but448_eval/1725000_0_0.ply') ).clip(-0.45,0.45).unsqueeze(0)

                # fps_xyz = torch.from_numpy(pcu.load_mesh_v('/nas/shared/public/yslan/logs/nips24/LSGM/t23d/FM/9cls/gs-disentangle/cascade_check/clay/stage1/eval/dino_img/debug/1875000_0.ply') ).clip(-0.45,0.45).unsqueeze(0)
                # fps_xyz = torch.from_numpy(pcu.load_mesh_v('/nas/shared/public/yslan/logs/nips24/LSGM/t23d/FM/9cls/gs-disentangle/cascade_check/clay/stage1/eval/dino_img/debug/1875000_0_0.ply') ).clip(-0.45,0.45).unsqueeze(0)

                # ! edit
                # st()
                # fps_xyz[..., 2:3] *= 4
                # fps_xyz[..., 2:3] *= 3

                # fps_xyz = torch.from_numpy(pcu.load_mesh_v('/nas/shared/V2V/yslan/logs/nips24/LSGM/t23d/FM/9cls/gs-disentangle/cascade_check/xyz_output_fullset_stillclip_but448_eval/1725000_0_1.ply') ).clip(-0.45,0.45).unsqueeze(0)
                batch_c = {
                    # 'img': batch['img'][[1,0]].to(self.dtype).to(dist_util.dev()),
                    'img': batch['img'][0:1].to(self.dtype).to(dist_util.dev()),
                    'fps-xyz': batch['fps-xyz'][0:1].to(self.dtype).to(dist_util.dev()),
                    # 'caption': batch['caption']
                    # 'fps-xyz': fps_xyz.repeat(batch['img'].shape[0],1,1).to(self.dtype).to(dist_util.dev()),
                }

            else:

                # gt_xyz = batch['fps-xyz'][0:1]
                # gt_kl_latent = batch['latent'][0:1]

                batch_c = {self.cond_key: batch[self.cond_key][0:1].to(dist_util.dev()).to(self.dtype), }

        # swap for more results, hard-coded here.
        # if 'img' in batch_c:
        #     batch_c['img'] = batch_c['img'][[1,0]]
            # batch_c['fps-xyz'] = batch_c['fps-xyz'][[1,0]]

        with th.cuda.amp.autocast(dtype=self.dtype,
                                    enabled=self.mp_trainer.use_amp):

            c, uc = self.conditioner.get_unconditional_conditioning(
                batch_c,
                force_uc_zero_embeddings=ucg_keys
                if len(self.conditioner.embedders) > 0 else [],
            )

        sampling_kwargs = {}

        N = num_samples  # hard coded, to update
        z_shape = (N, 768, self.ddpm_model.in_channels)

        for k in c:
            if isinstance(c[k], th.Tensor):
                # c[k], uc[k] = map(lambda y: y[k][:N].to(dist_util.dev()),
                #                   (c, uc))
                assert c[k].shape[0] == 1 # ! support batch inference
                c[k], uc[k] = map(lambda y: y[k].repeat_interleave(N, 0).to(dist_util.dev()),
                                (c, uc)) # support bs>1 sampling given a condition
    
        samples = self.sample(c,
                              shape=z_shape[1:],
                              uc=uc,
                              batch_size=N,
                              **sampling_kwargs)

        # ! get c
        if 'img' in self.cond_key:
            img_save_path = f'{logger.get_dir()}/{self.step+self.resume_step}_imgcond.jpg'
            if 'c' in self.cond_key:
                mv_img_save_path = f'{logger.get_dir()}/{self.step+self.resume_step}_mv-imgcond.jpg'
                torchvision.utils.save_image(batch_c['img-c']['img'][0], mv_img_save_path, value_range=(-1,1), normalize=True, padding=0) # torch.Size([24, 6, 3, 256, 256])
                torchvision.utils.save_image(batch_c['img'][0], img_save_path, value_range=(-1,1), normalize=True, padding=0) # torch.Size([24, 6, 3, 256, 256])
            else:
                torchvision.utils.save_image(batch_c['img'], img_save_path, value_range=(-1,1), normalize=True, padding=0)

        assert camera is not None
        batch = {'c': camera.clone()}

        # rendering
        for i in range(samples.shape[0]):
            th.cuda.empty_cache()

            # ! render sampled latent
            name_prefix = f'{self.step + self.resume_step}_{i}'

            # if self.cond_key in ['caption', 'img-c']:
            if self.cond_key in ['caption']:
                if isinstance(prompt, list):
                    name_prefix = f'{name_prefix}_{"-".join(prompt[0].split())}'
                else:
                    name_prefix = f'{name_prefix}_{"-".join(prompt.split())}'

            with th.cuda.amp.autocast(dtype=self.dtype,
                                        enabled=self.mp_trainer.use_amp):

            #     # ! todo, transform to gs camera
                if self.latent_key != 'latent': # normalized-xyz
                    pcu.save_mesh_v( f'{logger.get_dir()}/{name_prefix}.ply', self.unnormalize_pcd_act(samples[i]).detach().cpu().float().numpy())
                    logger.log(f'point cloud saved to {logger.get_dir()}/{name_prefix}.ply')
                else:
                    # ! editing debug
                    self.render_gs_video_given_latent(
                        # samples[i:i+1].to(self.dtype), # default version
                        # th.cat([gt_kl_latent.to(samples), gt_xyz.to(samples)], dim=-1), 

                        # ! xyz-cond kl feature gen:
                        th.cat([samples[i:i+1], batch_c['fps-xyz'][i:i+1]], dim=-1), # ! debugging xyz diffusion

                        # ! xyz debugging
                        # th.cat([gt_kl_latent.to(samples), samples[i:i+1]], dim=-1), # ! debugging xyz diffusion
                        # th.cat([samples[i:i+1], gt_xyz.to(samples), ], dim=-1) # ! debugging kl feature diffusion
                        self.rec_model,  # compatible with join_model
                        name_prefix=name_prefix,
                        save_img=save_img,
                        render_reference=batch,
                        export_mesh=False)

                # st()
                pass

                # for noise_scale in np.linspace(0,0.1, 10):
                #     per_scale_name_prefix = f'{name_prefix}_{noise_scale}'
                #     self.render_gs_video_given_latent( th.cat([gt_kl_latent.to(samples), (gt_xyz+noise_scale*th.randn_like(gt_xyz)).to(samples)], dim=-1), self.rec_model,  name_prefix=per_scale_name_prefix, save_img=save_img, render_reference=batch, export_mesh=False)
                # self.render_gs_video_given_latent( th.cat([gt_kl_latent.to(samples), (batch_c['fps-xyz'][0:1]).to(samples)], dim=-1), self.rec_model,  name_prefix=per_scale_name_prefix, save_img=save_img, render_reference=batch, export_mesh=False)

                # pcu.save_mesh_v( f'{logger.get_dir()}/sampled-4.ply', self.unnormalize_pcd_act(samples[0]).detach().cpu().float().numpy())
                # st()
                # pcu.save_mesh_v( f'tmp/sampled-3.ply', self.unnormalize_pcd_act(samples[0]).detach().cpu().float().numpy())

        gc.collect()
        self.ddpm_model.train()

    @torch.no_grad()
    def export_mesh_from_2dgs(self, all_rgbs, all_depths, all_alphas, cam_pathes, idx, i):
        # https://github.com/autonomousvision/LaRa/blob/main/evaluation.py
        n_thread = 1 # avoid TSDF cpu hanging bug.
        os.environ["MKL_NUM_THREADS"] = f"{n_thread}" 
        os.environ["NUMEXPR_NUM_THREADS"] = f"{n_thread}" 
        os.environ["OMP_NUM_THREADS"] = f"4" 
        os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_thread}" 
        os.environ["OPENBLAS_NUM_THREADS"] = f"{n_thread}" 

        # copied from: https://github.com/hbb1/2d-gaussian-splatting/blob/19eb5f1e091a582e911b4282fe2832bac4c89f0f/render.py#L23
        logger.log("exporting mesh ...")
        # os.makedirs(train_dir, exist_ok=True)
        train_dir = logger.get_dir()

        # for g-objv
        # aabb = [-0.5,-0.5,-0.5,0.5,0.5,0.5]
        # aabb = None
        aabb = [-0.45,-0.45,-0.45,0.45,0.45,0.45]
        self.aabb = np.array(aabb).reshape(2,3)*1.1

        # center = self.aabb.mean(0)
        # radius = np.linalg.norm(self.aabb[1] - self.aabb[0]) * 0.5
        # voxel_size = radius / 256
        # sdf_trunc = voxel_size * 2
        # print("using aabb")

        # set the active_sh to 0 to export only diffuse texture

        # gaussExtractor.gaussians.active_sh_degree = 0
        # gaussExtractor.reconstruction(scene.getTrainCameras())

        # extract the mesh and save
        # if args.unbounded:
        #     name = 'fuse_unbounded.ply'
        #     mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        # else:

        # name = f'{idx}-{i}-fuse.ply'
        # name = f'mesh.obj'
        name = f'{idx}/mesh_raw.obj'
        # st()
        # depth_trunc = (radius * 2.0) if depth_trunc < 0  else depth_trunc
        # voxel_size = (depth_trunc / mesh_res) if voxel_size < 0 else voxel_size
        # sdf_trunc = 5.0 * voxel_size if sdf_trunc < 0 else sdf_trunc
        # mesh = self.extract_mesh_bounded(all_rgbs, all_depths, all_alphas, cam_pathes, voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc, mask_backgrond=False)
        mesh = self.extract_mesh_bounded(all_rgbs, all_depths, all_alphas, cam_pathes)
        
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        logger.log("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        # mesh_post = post_process_mesh(mesh, cluster_to_keep=num_cluster)
        mesh_post = post_process_mesh(mesh)
        # o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.obj', '_post.obj')), mesh_post)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('_raw.obj', '.obj')), mesh_post)
        logger.log("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.obj', '_post.obj'))))


    @torch.no_grad()
    def extract_mesh_bounded(self, rgbmaps, depthmaps, alpha_maps, cam_pathes, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, alpha_thres=0.08, mask_backgrond=False):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """

        if self.aabb is not None: # as in lara.
            center = self.aabb.mean(0)
            radius = np.linalg.norm(self.aabb[1] - self.aabb[0]) * 0.5
            voxel_size = radius / 256
            sdf_trunc = voxel_size * 2
            print("using aabb")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        # render_reference = th.load('eval_pose.pt', map_location='cpu').numpy()

        # ! use uni_mesh_path, from Lara, Chen et al, ECCV 24'

        # '''


        # for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
        for i, cam in tqdm(enumerate(cam_pathes), desc="TSDF integration progress"):
            # rgb = self.rgbmaps[i]
            # depth = self.depthmaps[i]
            cam = self.c_to_3dgs_format(cam)
            cam_o3d = to_cam_open3d_compat(cam)

            rgb = rgbmaps[i][0]
            depth = depthmaps[i][0]
            alpha = alpha_maps[i][0]

            # if we have mask provided, use it
            # if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
            #     depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0
            
            depth[(alpha < alpha_thres)] = 0
            if self.aabb is not None:
                campos = cam['cam_pos'].cpu().numpy()
                depth_trunc = np.linalg.norm(campos - center, axis=-1) + radius

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, 
                convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh


    @th.inference_mode()
    def render_gs_video_given_latent(self,
                                    planes,
                                    rec_model,
                                    name_prefix='0',
                                    save_img=False, 
                                    render_reference=None, 
                                    export_mesh=False, 
                                    output_dir=None, 
                                    for_fid=False,):

        batch_size, L, C = planes.shape

        # ddpm_latent = { self.latent_name: planes[..., :-3] * self.feat_scale_factor.to(planes),  # kl-reg latent
        #                 'query_pcd_xyz': self.pcd_unnormalize_fn(planes[..., -3:]) }

        # ddpm_latent = { self.latent_name: self.unnormalize_kl_feat(planes[..., :-3]),  # kl-reg latent
        # ddpm_latent = { self.latent_name: planes[..., :-3],  # kl-reg latent
        #                     'query_pcd_xyz': self.unnormalize_pcd_act(planes[..., -3:]) }

        ddpm_latent = { self.latent_name: planes[..., :-3],  # kl-reg latent
                            'query_pcd_xyz': planes[..., -3:]}
        
        ddpm_latent.update(rec_model(latent=ddpm_latent, behaviour='decode_gs_after_vae_no_render')) 

        # ! editing debug, raw scaling
        
        # for beacon
        # edited_fps_xyz[..., 2] *= 1.5
        # edited_fps_xyz[..., :2] *= 0.75

        # z_mask = edited_fps_xyz[..., 2] > 0
        # edited_fps_xyz[..., 2] *= 1.25 # only apply to upper points

        # z_dim_coord = edited_fps_xyz[..., 2]
        # edited_fps_xyz[..., 2] = th.where(z_dim_coord>0, z_dim_coord*1.25, z_dim_coord)
        # edited_fps_xyz[..., :2] *= 0.6

        fine_scale = 'gaussians_upsampled_3'
        # ddpm_latent[fine_scale][..., :2] *= 1.5
        # ddpm_latent[fine_scale][..., 2:3] *= 0.75

        # ddpm_latent[fine_scale][..., :2] *= 3
        # ddpm_latent[fine_scale][..., 2:3] *= 0.75

        # z_dim_coord = ddpm_latent[fine_scale][..., 2]
        # ddpm_latent[fine_scale][..., 2] = th.where(z_dim_coord>0.24, z_dim_coord+0.1, z_dim_coord)

        # pcu.save_mesh_v(f'{output_dir}/gaussian.ply', ddpm_latent['gaussians_upsampled'][0, ..., :3].cpu().numpy())
        # fps-downsampling?
        pred_gaussians_xyz = ddpm_latent['gaussians_upsampled_3'][..., :3]
        fine_gs = ddpm_latent[fine_scale]

        # K=4096
        # query_pcd_xyz, fps_idx = pytorch3d.ops.sample_farthest_points(
        #     pred_gaussians_xyz, K=K,
        #     # random_start_point=False)  # B self.latent_num
        #     random_start_point=True)  # B self.latent_num

        if output_dir is None:
            output_dir = logger.get_dir()

        # pcu.save_mesh_v(f'{output_dir}/{name_prefix}-gaussian-{K}.ply', query_pcd_xyz[0].cpu().numpy())

        np.save(f'{output_dir}/{name_prefix}-gaussian.npy', fine_gs.cpu().numpy())

        # return None, None

        try:
            # video_out = imageio.get_writer(
            #     f'{output_dir}/gs_{name_prefix}.mp4',
            #     mode='I',
            #     fps=15,
            #     codec='libx264')

            video_out = imageio.get_writer(
                f'{output_dir}/{name_prefix}-gs.mp4',
                mode='I',
                fps=15,
                codec='libx264')

        except Exception as e:
            logger.log(e)

            # return # some caption are too tired and cannot be parsed as file name

        # !for FID

        ''' # if for uniform FID rendering. Will not adopt this later.
        azimuths = []
        elevations = []
        frame_number = 10

        for i in range(frame_number): # 1030 * 5 * 10, for FID 50K

            azi, elevation = sample_uniform_cameras_on_sphere()
            # azi, elevation = azi[0] / np.pi * 180, elevation[0] / np.pi * 180
            # azi, elevation = azi[0] / np.pi * 180, 0
            azi, elevation = azi[0] / np.pi * 180, (elevation[0]-np.pi*0.5) / np.pi * 180 # [-0.5 pi, 0.5 pi]
            azimuths.append(azi)
            elevations.append(elevation)

        azimuths = np.array(azimuths)
        elevations = np.array(elevations)

        # azimuths = np.array(list(range(0,360,30))).astype(float)
        # frame_number = azimuths.shape[0]
        # elevations = np.array([10]*azimuths.shape[0]).astype(float)

        zero123pp_pose, _ = generate_input_camera(1.8, [[elevations[i], azimuths[i]] for i in range(frame_number)], fov=30)
        K = th.Tensor([1.3889, 0.0000, 0.5000, 0.0000, 1.3889, 0.5000, 0.0000, 0.0000, 0.0039]).to(zero123pp_pose) # keeps the same
        render_reference = th.cat([zero123pp_pose.reshape(frame_number,-1), K.unsqueeze(0).repeat(frame_number,1)], dim=-1).cpu().numpy()
        '''

        render_reference = th.load('eval_pose.pt', map_location='cpu').numpy()[:24]
        # rand_start_idx = random.randint(0,2)
        # render_reference = render_reference[rand_start_idx::3] # randomly render 8 views, maintain fixed azimuths
        # assert len(render_reference)==8

        # assert render_reference is None
            # render_reference = self.eval_data # compat
        # else: # use train_traj

        # for key in ['ins', 'bbox', 'caption']:
        #     if key in render_reference:
        #         render_reference.pop(key)

        # render_reference = [ { k:v[idx:idx+1] for k, v in render_reference.items() } for idx in range(40) ]
        all_rgbs, all_depths, all_alphas = [], [], []

        # for i, batch in enumerate(tqdm(self.eval_data)):
        for i, micro_c in enumerate(tqdm(render_reference)):
            # micro = {
            #     k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v
            #     for k, v in batch.items() 
            # }

            # c = self.eval_data.post_process.c_to_3dgs_format(micro_c)
            c = self.c_to_3dgs_format(micro_c)
            for k in c.keys(): # to cuda
                if isinstance(c[k], th.Tensor) and k != 'tanfov':
                    c[k] = c[k].unsqueeze(0).unsqueeze(0).to(dist_util.dev()) # actually, could render 40 views together.
            c['tanfov'] = th.tensor(c['tanfov']).to(dist_util.dev())

            pred = rec_model(
                img=None,
                c=c, # TODO, to dict
                latent=ddpm_latent, # render gs
                behaviour='triplane_dec',
                bg_color=self.gs_bg_color,
                render_all_scale=True, # for better visualization
                )
            
            # ! if visualizing a single scale
            fine_scale_key = list(pred.keys())[-1]
            # pred = pred[fine_scale_key]

            # for k in pred.keys():
            #     pred[k] = einops.rearrange(pred[k], 'B V ... -> (B V) ...') # merge 
            
            # pred_vis = self._make_vis_img(pred)

            # vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            # vis = vis * 127.5 + 127.5
            # vis = vis.clip(0, 255).astype(np.uint8)

            # # if not save_img:
            # for j in range(vis.shape[0]
            #             ):  # ! currently only export one plane at a time
            #     video_out.append_data(vis[j])

            # save multi-scale rendering

            all_rgbs.append(einops.rearrange(pred[fine_scale_key]['image'], 'B V ... -> (B V) ...'))
            all_depths.append(einops.rearrange(pred[fine_scale_key]['depth'], 'B V ... -> (B V) ...'))
            all_alphas.append(einops.rearrange(pred[fine_scale_key]['alpha'], 'B V ... -> (B V) ...'))

            all_pred_vis = {}
            for key in pred.keys():
                pred_scale = pred[key] # only show finest result here
                for k in pred_scale.keys():
                    pred_scale[k] = einops.rearrange(pred_scale[k], 'B V ... -> (B V) ...') # merge 
                
                pred_vis = self._make_vis_img(pred_scale)

                vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
                vis = vis * 127.5 + 127.5
                vis = vis.clip(0, 255).astype(np.uint8)

                all_pred_vis[key] = vis

            # all_pred_vis_concat = np.concatenate([cv2.resize(all_pred_vis[k][0], (384*3, 384)) for k in ['gaussians_base', 'gaussians_upsampled', 'gaussians_upsampled_2']], axis=0)
            # all_pred_vis_concat = np.concatenate([cv2.resize(all_pred_vis[k][0], (256*3, 256)) for k in ['gaussians_base', 'gaussians_upsampled', 'gaussians_upsampled_2']], axis=0)
            # all_pred_vis_concat = np.concatenate([cv2.resize(all_pred_vis[k][0], (384*len(all_pred_vis.keys()), 384)) for k in all_pred_vis.keys()], axis=0)
            # all_pred_vis_concat = np.concatenate([cv2.resize(all_pred_vis[k][0], (384*3, 384)) for k in all_pred_vis.keys()], axis=0)

            all_pred_vis_concat = np.concatenate([cv2.resize(all_pred_vis[k][0], (512*3, 512)) for k in all_pred_vis.keys()], axis=0)

            video_out.append_data(all_pred_vis_concat)

        if save_img: # for fid 
            for idx in range(len(all_rgbs)):
                sampled_img = Image.fromarray(
                    (all_rgbs[idx][0].permute(1, 2, 0).cpu().numpy() *
                        255).clip(0, 255).astype(np.uint8))
                sampled_img.save(os.path.join(output_dir,f'{name_prefix}-{idx}.jpg'))


        # if not save_img:
        video_out.close()
        print('logged video to: ',
            f'{output_dir}/{name_prefix}.mp4')

        del video_out, pred, pred_vis, vis
        return all_rgbs, all_depths, all_alphas

    @th.no_grad()
    def _make_vis_img(self, pred):

        # if True:
        pred_depth = pred['image_depth']
        pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                        pred_depth.min())

        pred_depth = pred_depth.cpu()[0].permute(1, 2, 0).numpy()
        pred_depth = (plt.cm.viridis(pred_depth[..., 0])[..., :3]) * 2 - 1
        pred_depth = th.from_numpy(pred_depth).to(
            pred['image_raw'].device).permute(2, 0, 1).unsqueeze(0)

        gen_img = pred['image_raw']
        rend_normal = pred['rend_normal']

        pred_vis = th.cat(
            [
                gen_img,
                rend_normal,
                pred_depth,
            ],
            dim=-1)  # B, 3, H, W
        
        return pred_vis


    def _set_grad_flag(self):    
        requires_grad(self.ddpm_model, True) # 

    @th.inference_mode()
    def sample_and_save(self, batch_c, ucg_keys, num_samples, camera, save_img, idx=0, save_dir='', export_mesh=False, stage1_idx=0):

        with th.cuda.amp.autocast(dtype=self.dtype,
                                    enabled=self.mp_trainer.use_amp):

            c, uc = self.conditioner.get_unconditional_conditioning(
                batch_c,
                force_uc_zero_embeddings=ucg_keys
                if len(self.conditioner.embedders) > 0 else [],
            )

        sampling_kwargs = {}

        N = num_samples  # hard coded, to update
        z_shape = (N, 768, self.ddpm_model.in_channels)

        for k in c:
            if isinstance(c[k], th.Tensor):
                # c[k], uc[k] = map(lambda y: y[k][:N].to(dist_util.dev()),
                #                   (c, uc))
                assert c[k].shape[0] == 1 # ! support batch inference
                c[k], uc[k] = map(lambda y: y[k].repeat_interleave(N, 0).to(dist_util.dev()),
                                (c, uc)) # support bs>1 sampling given a condition
    
        samples = self.sample(c,
                            shape=z_shape[1:],
                            uc=uc,
                            batch_size=N,
                            **sampling_kwargs)

        # ! get c
        if save_dir == '':
            save_dir = logger.get_dir()

        if 'img' in self.cond_key:
            # img_save_path = f'{save_dir}/{idx}_imgcond.jpg'
            img_save_path = f'{save_dir}/{idx}/imgcond.jpg'
            os.makedirs(f'{save_dir}/{idx}', exist_ok=True)
            if 'c' in self.cond_key:
                torchvision.utils.save_image(batch_c['img-c']['img'][0], img_save_path, value_range=(-1,1), normalize=True, padding=0) # torch.Size([24, 6, 3, 256, 256])
            else:
                torchvision.utils.save_image(batch_c['img'], img_save_path, value_range=(-1,1), normalize=True, padding=0)

        assert camera is not None
        # batch = {'c': camera.clone()}

        # rendering
        for i in range(samples.shape[0]):
            th.cuda.empty_cache()

            if self.cond_key in ['caption']:
                name_prefix = f'{batch_c["caption"]}_sample-{stage1_idx}-{i}'
            else:
                # ! render sampled latent
                # name_prefix = f'{idx}_sample-{i}'
                name_prefix = f'{idx}/sample-{stage1_idx}-{i}'

            # if self.cond_key in ['caption', 'img-c']:

            with th.cuda.amp.autocast(dtype=self.dtype,
                                        enabled=self.mp_trainer.use_amp):

            #     # ! todo, transform to gs camera
                if self.latent_key != 'latent': # normalized-xyz
                    pcd_export_dir = f'{save_dir}/{name_prefix}.ply'
                    pcu.save_mesh_v(pcd_export_dir, self.unnormalize_pcd_act(samples[i]).detach().cpu().float().numpy())
                    logger.log(f'point cloud saved to {pcd_export_dir}')
                else:
                    # ! editing debug
                    all_rgbs, all_depths, all_alphas = self.render_gs_video_given_latent(
                        th.cat([samples[i:i+1], batch_c['fps-xyz'][0:1]], dim=-1), # ! debugging xyz diffusion
                        self.rec_model,  # compatible with join_model
                        name_prefix=name_prefix,
                        save_img=save_img,
                        render_reference=camera,
                        export_mesh=False,)
                        # for_fid=False)

                    if export_mesh:
                        self.export_mesh_from_2dgs(all_rgbs, all_depths, all_alphas, camera, idx, i)

        # mesh = self.extract_mesh_bounded(all_rgbs, all_depths, all_alphas, cam_pathes, voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc, mask_backgrond=False)


    @th.inference_mode()
    def eval_and_export(
        self,
        prompt="Yellow rubber duck",
        # use_ddim=False,
        # unconditional_guidance_scale=1.0,
        save_img=False,
        use_train_trajectory=False,
        camera=None,
        num_samples=1,
        stage_1_output_dir='',
        num_instances=1,
        export_mesh=False,
    ):
        self.ddpm_model.eval()

        args = dnnlib.EasyDict(
            dict(
                batch_size=1,
                image_size=self.diffusion_input_size,
                denoise_in_channels=self.rec_model.decoder.triplane_decoder.
                out_chans,  # type: ignore
                clip_denoised=False,
                class_cond=False))

        model_kwargs = {}

        uc = None
        log = dict()

        ucg_keys = [self.cond_key] # i23d

        def sample_and_save(batch_c, idx=0):

            with th.cuda.amp.autocast(dtype=self.dtype,
                                        enabled=self.mp_trainer.use_amp):

                c, uc = self.conditioner.get_unconditional_conditioning(
                    batch_c,
                    force_uc_zero_embeddings=ucg_keys
                    if len(self.conditioner.embedders) > 0 else [],
                )

            sampling_kwargs = {}

            N = num_samples  # hard coded, to update
            z_shape = (N, 768, self.ddpm_model.in_channels)

            for k in c:
                if isinstance(c[k], th.Tensor):
                    # c[k], uc[k] = map(lambda y: y[k][:N].to(dist_util.dev()),
                    #                   (c, uc))
                    assert c[k].shape[0] == 1 # ! support batch inference
                    c[k], uc[k] = map(lambda y: y[k].repeat_interleave(N, 0).to(dist_util.dev()),
                                    (c, uc)) # support bs>1 sampling given a condition
        
            samples = self.sample(c,
                                shape=z_shape[1:],
                                uc=uc,
                                batch_size=N,
                                **sampling_kwargs)

            # ! get c
            if 'img' in self.cond_key:
                img_save_path = f'{logger.get_dir()}/{idx}_imgcond.jpg'
                if 'c' in self.cond_key:
                    torchvision.utils.save_image(batch_c['img-c']['img'][0], img_save_path, value_range=(-1,1), normalize=True, padding=0) # torch.Size([24, 6, 3, 256, 256])
                else:
                    torchvision.utils.save_image(batch_c['img'], img_save_path, value_range=(-1,1), normalize=True, padding=0)

            assert camera is not None
            batch = {'c': camera.clone()}

            # rendering
            for i in range(samples.shape[0]):
                th.cuda.empty_cache()

                if self.cond_key in ['caption']:
                    name_prefix = f'{batch_c["caption"]}_sample-{idx}-{i}'
                else:
                    # ! render sampled latent
                    name_prefix = f'{idx}_sample-{i}'

                # if self.cond_key in ['caption', 'img-c']:

                with th.cuda.amp.autocast(dtype=self.dtype,
                                            enabled=self.mp_trainer.use_amp):

                #     # ! todo, transform to gs camera
                    if self.latent_key != 'latent': # normalized-xyz
                        pcu.save_mesh_v( f'{logger.get_dir()}/{name_prefix}.ply', self.unnormalize_pcd_act(samples[i]).detach().cpu().float().numpy())
                        logger.log(f'point cloud saved to {logger.get_dir()}/{name_prefix}.ply')
                    else:
                        # ! editing debug
                        all_rgbs, all_depths, all_alphas = self.render_gs_video_given_latent(
                            # samples[i:i+1].to(self.dtype), # default version
                            # th.cat([gt_kl_latent.to(samples), gt_xyz.to(samples)], dim=-1), 

                            # ! xyz-cond kl feature gen:
                            # th.cat([samples[i:i+1], batch_c['fps-xyz'][i:i+1]], dim=-1), # ! debugging xyz diffusion
                            th.cat([samples[i:i+1], batch_c['fps-xyz'][0:1]], dim=-1), # ! debugging xyz diffusion

                            # ! xyz debugging
                            # th.cat([gt_kl_latent.to(samples), samples[i:i+1]], dim=-1), # ! debugging xyz diffusion
                            # th.cat([samples[i:i+1], gt_xyz.to(samples), ], dim=-1) # ! debugging kl feature diffusion
                            self.rec_model,  # compatible with join_model
                            name_prefix=name_prefix,
                            save_img=save_img,
                            render_reference=batch,
                            export_mesh=False)

                        if export_mesh:
                            self.export_mesh_from_2dgs(all_rgbs, all_depths, idx, i)

        if self.cond_key == 'caption':
            assert prompt != ''
            batch_c = {self.cond_key: prompt}

            if self.latent_key == 'latent': # t23d, stage-2
                for i in range(2): # 8 * num_samples here
                    fps_xyz = torch.from_numpy(pcu.load_mesh_v(f'{stage_1_output_dir}/{prompt}_sample-0-{i}.ply') ).clip(-0.45,0.45).unsqueeze(0)

                    # ! if editing, change the latent points accordingly.
                    # edited_fps_xyz = fps_xyz.clone() # B N 3
                    # z_dim_coord = edited_fps_xyz[..., 2]
                    # edited_fps_xyz[..., 2] = th.where(z_dim_coord>0.24, z_dim_coord+0.075, z_dim_coord)

                    batch_c.update({
                        'fps-xyz': fps_xyz.to(self.dtype).to(dist_util.dev())
                        # 'fps-xyz': edited_fps_xyz.to(self.dtype).to(dist_util.dev())
                    })

                    sample_and_save(batch_c, idx=i)
            else:
                sample_and_save(batch_c)
            


    @th.inference_mode()
    def eval_t23d_and_export(
        self,
        prompt="Yellow rubber duck",
        # use_ddim=False,
        # unconditional_guidance_scale=1.0,
        save_img=False,
        use_train_trajectory=False,
        camera=None,
        num_samples=1,
        stage_1_output_dir='',
        num_instances=1,
        export_mesh=False,
    ):
        self.ddpm_model.eval()

        args = dnnlib.EasyDict(
            dict(
                batch_size=1,
                image_size=self.diffusion_input_size,
                denoise_in_channels=self.rec_model.decoder.triplane_decoder.
                out_chans,  # type: ignore
                clip_denoised=False,
                class_cond=False))

        model_kwargs = {}

        uc = None
        log = dict()

        ucg_keys = [self.cond_key] # i23d

        assert self.cond_key == 'caption' and prompt != ''
        batch_c = {self.cond_key: prompt}

        if self.latent_key == 'latent': # t23d, stage-2
            fps_xyz = torch.from_numpy(pcu.load_mesh_v(f'{stage_1_output_dir}/{prompt}_sample-0.ply') ).clip(-0.45,0.45).unsqueeze(0)
            batch_c.update({
                'fps-xyz': fps_xyz.to(self.dtype).to(dist_util.dev())
            })

        self.sample_and_save(batch_c, ucg_keys, num_samples, camera,)


    @th.inference_mode()
    def eval_i23d_and_export(
        self,
        prompt="Yellow rubber duck",
        # use_ddim=False,
        # unconditional_guidance_scale=1.0,
        save_img=False,
        use_train_trajectory=False,
        camera=None,
        num_samples=1,
        stage_1_output_dir='',
        num_instances=1,
        export_mesh=False,
    ):
        self.ddpm_model.eval()

        args = dnnlib.EasyDict(
            dict(
                batch_size=1,
                image_size=self.diffusion_input_size,
                denoise_in_channels=self.rec_model.decoder.triplane_decoder.
                out_chans,  # type: ignore
                clip_denoised=False,
                class_cond=False))

        model_kwargs = {}

        uc = None
        log = dict()

        ucg_keys = [self.cond_key] # i23d


        for idx, batch in enumerate(tqdm(self.data)):

            ins = batch['ins'][0]

            # obj_folder, _, frame = ins.split('/')
            ins = ins.split('/')
            # obj_folder, frame = ins[0], ins[-1] # for gso

            if len(ins) >2:
                if ins[1] == 'render_mvs_25': # gso
                    obj_folder, frame = ins[0], int(ins[-1].split('.')[0])
                    ins_name = f'{obj_folder}/{str(frame)}'
                else:
                    obj_folder, frame = os.path.join(ins[1], ins[2]), ins[-1] # for objv
                    frame = int(frame.split('.')[0])
                    ins_name = f'{obj_folder}/{str(frame)}'
            else: # folder of images, e.g., instantmesh
                ins_name = ins[0].split('.')[0]


            # pcd_export_dir = f'{logger.get_dir()}/{ins_name}/sample-0.ply'

            # if os.path.exists(pcd_export_dir):
            #     continue

            #! debugging, get GT xyz and KL latent for disentangled debugging

            if self.cond_key == 'img-c': # mv23d
                prompt = batch['caption'][0:1]
                batch_c = {
                    self.cond_key: {
                        'img': batch['mv_img'][0:1].to(self.dtype).to(dist_util.dev()),
                        'c': batch['c'][0:1].to(self.dtype).to(dist_util.dev()),
                    },
                    'img': batch['img'][0:1].to(self.dtype).to(dist_util.dev()),
                    'caption': prompt,
                }

                if self.latent_key == 'latent': # stage-2
                    fps_xyz = torch.from_numpy(pcu.load_mesh_v(f'{stage_1_output_dir}/{idx}_sample-0.ply') ).clip(-0.45,0.45).unsqueeze(0)
                    # fps_xyz = torch.from_numpy(pcu.load_mesh_v(f'{stage_1_output_dir}/sample-0.ply') ).clip(-0.45,0.45).unsqueeze(0)
                    batch_c.update({
                        'fps-xyz': fps_xyz[0:1].to(self.dtype).to(dist_util.dev()),
                    })
            
            # elif self.cond_key == 'img-caption':
            #     batch_c = {'caption': prompt, 'img': batch['img'].to(dist_util.dev()).to(self.dtype)}

            elif self.cond_key == 'img-xyz': # stage-2

                for i in range(2):

                    stage1_pcd_output_path = f'{stage_1_output_dir}/{ins_name}/sample-0-{i}.ply'

                    fps_xyz = trimesh.load(stage1_pcd_output_path).vertices # pcu may fail on py=3.9
                    fps_xyz = torch.from_numpy(fps_xyz).clip(-0.45,0.45).unsqueeze(0)

                    # fps_xyz = None # ! TODO, load from local directory
                    batch_c = {
                        'img': batch['img'][0:1].to(self.dtype).to(dist_util.dev()),
                        'fps-xyz': fps_xyz[0:1].to(self.dtype).to(dist_util.dev()),
                    }
                    
                    self.sample_and_save(batch_c, ucg_keys, num_samples, camera, save_img, idx=ins_name, export_mesh=export_mesh, stage1_idx=i) # type: ignore

            else: # stage-1 data
                batch_c = {self.cond_key: batch[self.cond_key][0:1].to(dist_util.dev()).to(self.dtype), }
                if self.cond_key == 'caption' and self.latent_key == 'latent': # t23d, stage-2
                    fps_xyz = torch.from_numpy(pcu.load_mesh_v(f'{stage_1_output_dir}/{idx}_sample-0.ply') ).clip(-0.45,0.45).unsqueeze(0)
                    batch_c.update({
                        'fps-xyz': fps_xyz.to(self.dtype).to(dist_util.dev())
                    })


            # save_dir = f'{logger.get_dir()}/{ins}'
            # os.mkdir(save_dir, exists_ok=True, parents=True)

            # self.sample_and_save(batch_c, ucg_keys, num_samples, camera, save_img, idx=f'{idx}-{ins}', export_mesh=export_mesh)
                self.sample_and_save(batch_c, ucg_keys, num_samples, camera, save_img, idx=ins_name, export_mesh=export_mesh) # type: ignore


        gc.collect()




    def get_source_cw2wT(self, source_cameras_view_to_world):
        return matrix_to_quaternion(
            source_cameras_view_to_world[:3, :3].transpose(0, 1))

    def c_to_3dgs_format(self, pose):
        # TODO, switch to torch version (batched later)

        c2w = pose[:16].reshape(4, 4)  # 3x4

        # ! load cam
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        fx = pose[16]
        FovX = focal2fov(fx, 1)
        FovY = focal2fov(fx, 1)

        tanfovx = math.tan(FovX * 0.5)
        tanfovy = math.tan(FovY * 0.5)

        assert tanfovx == tanfovy

        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0

        world_view_transform = torch.tensor(getWorld2View2(R, T, trans,
                                                           scale)).transpose(
                                                               0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear,
                                                zfar=self.zfar,
                                                fovX=FovX,
                                                fovY=FovY).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
            projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        view_world_transform = torch.tensor(getView2World(R, T, trans,
                                                          scale)).transpose(
                                                              0, 1)

        # item.update(viewpoint_cam=[viewpoint_cam])
        c = {}
        c["source_cv2wT_quat"] = self.get_source_cw2wT(view_world_transform)
        c.update(
            projection_matrix=projection_matrix, # K
            cam_view=world_view_transform,  # world_view_transform
            cam_view_proj=full_proj_transform,  # full_proj_transform
            cam_pos=camera_center,
            tanfov=tanfovx,  # TODO, fix in the renderer
            # orig_c2w=c2w,
            # orig_w2c=w2c,
            orig_pose=torch.from_numpy(pose),
            orig_c2w=torch.from_numpy(c2w),
            orig_w2c=torch.from_numpy(w2c),
            # tanfovy=tanfovy,
        )

        return c  # dict for gs rendering



class FlowMatchingEngine_gs_clay(FlowMatchingEngine_gs):

    def __init__(
        self,
        *,
        rec_model,
        denoise_model,
        diffusion,
        sde_diffusion,
        control_model,
        control_key,
        only_mid_control,
        loss_class,
        data,
        eval_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        eval_interval,
        save_interval,
        resume_checkpoint,
        resume_cldm_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=0.001,
        schedule_sampler=None,
        weight_decay=0,
        lr_anneal_steps=0,
        iterations=10001,
        ignore_resume_opt=False,
        freeze_ae=False,
        denoised_ae=True,
        triplane_scaling_divider=10,
        use_amp=False,
        diffusion_input_size=224,
        normalize_clip_encoding=False,
        scale_clip_encoding=1,
        cfg_dropout_prob=0,
        cond_key='img_sr',
        use_eos_feature=False,
        compile=False,
        snr_type='lognorm',
        **kwargs):
        super().__init__(rec_model=rec_model,
                         denoise_model=denoise_model,
                         diffusion=diffusion,
                         sde_diffusion=sde_diffusion,
                         control_model=control_model,
                         control_key=control_key,
                         only_mid_control=only_mid_control,
                         loss_class=loss_class,
                         data=data,
                         eval_data=eval_data,
                         batch_size=batch_size,
                         microbatch=microbatch,
                         lr=lr,
                         ema_rate=ema_rate,
                         log_interval=log_interval,
                         eval_interval=eval_interval,
                         save_interval=save_interval,
                         resume_checkpoint=resume_checkpoint,
                         resume_cldm_checkpoint=resume_cldm_checkpoint,
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         schedule_sampler=schedule_sampler,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         ignore_resume_opt=ignore_resume_opt,
                         freeze_ae=freeze_ae,
                         denoised_ae=denoised_ae,
                         triplane_scaling_divider=triplane_scaling_divider,
                         use_amp=use_amp,
                         diffusion_input_size=diffusion_input_size,
                         normalize_clip_encoding=normalize_clip_encoding,
                         scale_clip_encoding=scale_clip_encoding,
                         cfg_dropout_prob=cfg_dropout_prob,
                         cond_key=cond_key,
                         use_eos_feature=use_eos_feature,
                         compile=compile,
                         snr_type=snr_type,
                         **kwargs)

        # self._init_new_ca_weight() # after ckpt loading


    def _set_grad_flag(self):    
        # unfree CA only
        requires_grad(self.ddpm_model, True) # 
        # for k, v in self.ddpm_model.named_parameters():
        #     # if 'cross_attn_dino' in k:
        #     if 'mv' in k: # for mv dino
        #         v.requires_grad_(True)
        #         if self.step == 0:
        #             logger.log(k)
        #     else:
        #         v.requires_grad_(False)
                
    def _init_new_ca_weight(self):
        blks_to_copy = ['cross_attn_dino', 'prenorm_ca_dino']

        for blk in self.ddpm_model.blocks:
            for param_name in blks_to_copy:
                try:
                    getattr(blk, param_name.replace('dino', 'dino_mv')).load_state_dict(getattr(blk, param_name).state_dict())
                except Exception as e:
                    logger.log(e) # some key misalignment
