import copy
import cv2
import einops
from collections import defaultdict
import matplotlib.pyplot as plt
import random
# import emd
import pytorch3d.loss
# import imageio.v3
import functools
import json
import os
from pathlib import Path
from pdb import set_trace as st
from einops import rearrange
import webdataset as wds

from nsr.camera_utils import generate_input_camera, uni_mesh_path
import point_cloud_utils as pcu
import traceback
import blobfile as bf
from datasets.g_buffer_objaverse import focal2fov, fov2focal
import math
import imageio
import numpy as np
# from sympy import O
import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch as th
import open3d as o3d
import torch.distributed as dist
import torchvision
from PIL import Image
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pytorch3d.ops

from torch.profiler import profile, record_function, ProfilerActivity

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import LossAwareSampler, UniformSampler
from guided_diffusion.train_util import (calc_average_loss,
                                         find_ema_checkpoint,
                                         find_resume_checkpoint,
                                         get_blob_logdir, log_rec3d_loss_dict,
                                         parse_resume_step_from_filename)

from datasets.g_buffer_objaverse import unity2blender, unity2blender_th, PostProcess

from nsr.volumetric_rendering.ray_sampler import RaySampler
from utils.mesh_util import post_process_mesh, to_cam_open3d_compat
from .camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from nsr.camera_utils import generate_input_camera, uni_mesh_path, sample_uniform_cameras_on_sphere 

from utils.gs_utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World
from utils.general_utils import matrix_to_quaternion
from utils.mesh_util import post_process_mesh, to_cam_open3d_compat, smooth_mesh
from datasets.g_buffer_objaverse import focal2fov, fov2focal

from .train_util import TrainLoop3DRec
import kornia


@th.autocast(device_type='cuda', dtype=th.float16, enabled=False)
def psnr(input, target, max_val):
    return kornia.metrics.psnr(input, target, max_val)


def calc_emd(output, gt, eps=0.005, iterations=50):
    import utils.emd.emd_module as emd
    emd_loss = emd.emdModule()
    dist, _ = emd_loss(output, gt, eps, iterations)
    emd_out = torch.sqrt(dist).mean(1)
    return emd_out


class TrainLoop3DRecNV(TrainLoop3DRec):
    # supervise the training of novel view
    def __init__(self,
                 *,
                 rec_model,
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
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
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
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)
        self.rec_cano = True

    def forward_backward(self, batch, *args, **kwargs):
        # return super().forward_backward(batch, *args, **kwargs)

        self.mp_trainer_rec.zero_grad()
        batch_size = batch['img_to_encoder'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            # st()
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev())
                for k, v in batch.items()
            }

            # ! concat novel-view? next version. also add self reconstruction, patch-based loss in the next version. verify novel-view prediction first.

            # wrap forward within amp
            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                target_nvs = {}
                target_cano = {}

                latent = self.rec_model(img=micro['img_to_encoder'],
                                        behaviour='enc_dec_wo_triplane')

                pred = self.rec_model(
                    latent=latent,
                    c=micro['nv_c'],  # predict novel view here
                    behaviour='triplane_dec')

                for k, v in micro.items():
                    if k[:2] == 'nv':
                        orig_key = k.replace('nv_', '')
                        target_nvs[orig_key] = v
                        target_cano[orig_key] = micro[orig_key]

                with self.rec_model.no_sync():  # type: ignore
                    loss, loss_dict, fg_mask = self.loss_class(
                        pred,
                        target_nvs,
                        step=self.step + self.resume_step,
                        test_mode=False,
                        return_fg_mask=True,
                        conf_sigma_l1=None,
                        conf_sigma_percl=None)
                    log_rec3d_loss_dict(loss_dict)

                if self.rec_cano:

                    pred_cano = self.rec_model(latent=latent,
                                               c=micro['c'],
                                               behaviour='triplane_dec')

                    with self.rec_model.no_sync():  # type: ignore

                        fg_mask = target_cano['depth_mask'].unsqueeze(
                            1).repeat_interleave(3, 1).float()

                        loss_cano, loss_cano_dict = self.loss_class.calc_2d_rec_loss(
                            pred_cano['image_raw'],
                            target_cano['img'],
                            fg_mask,
                            step=self.step + self.resume_step,
                            test_mode=False,
                        )

                    loss = loss + loss_cano

                    # remove redundant log
                    log_rec3d_loss_dict({
                        f'cano_{k}': v
                        for k, v in loss_cano_dict.items()
                        #  if "loss" in k
                    })

            self.mp_trainer_rec.backward(loss)

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                if self.rec_cano:
                    self.log_img(micro, pred, pred_cano)
                else:
                    self.log_img(micro, pred, None)

    @th.inference_mode()
    def log_img(self, micro, pred, pred_cano):
        # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

        def norm_depth(pred_depth):  # to [-1,1]
            # pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())
            return -(pred_depth * 2 - 1)

        pred_img = pred['image_raw']
        gt_img = micro['img']

        # infer novel view also
        # if self.loss_class.opt.symmetry_loss:
        #     pred_nv_img = nvs_pred
        # else:
        # ! replace with novel view prediction

        # ! log another novel-view prediction
        # pred_nv_img = self.rec_model(
        #     img=micro['img_to_encoder'],
        #     c=self.novel_view_poses)  # pred: (B, 3, 64, 64)

        # if 'depth' in micro:
        gt_depth = micro['depth']
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        gt_depth = norm_depth(gt_depth)
        # gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
        #                                           gt_depth.min())
        # if True:
        fg_mask = pred['image_mask'] * 2 - 1  # 0-1
        input_fg_mask = pred_cano['image_mask'] * 2 - 1  # 0-1
        if 'image_depth' in pred:
            pred_depth = norm_depth(pred['image_depth'])
            pred_nv_depth = norm_depth(pred_cano['image_depth'])
        else:
            pred_depth = th.zeros_like(gt_depth)
            pred_nv_depth = th.zeros_like(gt_depth)

        if 'image_sr' in pred:
            if pred['image_sr'].shape[-1] == 512:
                pred_img = th.cat([self.pool_512(pred_img), pred['image_sr']],
                                  dim=-1)
                gt_img = th.cat([self.pool_512(micro['img']), micro['img_sr']],
                                dim=-1)
                pred_depth = self.pool_512(pred_depth)
                gt_depth = self.pool_512(gt_depth)

            elif pred['image_sr'].shape[-1] == 256:
                pred_img = th.cat([self.pool_256(pred_img), pred['image_sr']],
                                  dim=-1)
                gt_img = th.cat([self.pool_256(micro['img']), micro['img_sr']],
                                dim=-1)
                pred_depth = self.pool_256(pred_depth)
                gt_depth = self.pool_256(gt_depth)

            else:
                pred_img = th.cat([self.pool_128(pred_img), pred['image_sr']],
                                  dim=-1)
                gt_img = th.cat([self.pool_128(micro['img']), micro['img_sr']],
                                dim=-1)
                gt_depth = self.pool_128(gt_depth)
                pred_depth = self.pool_128(pred_depth)
        else:
            gt_img = self.pool_64(gt_img)
            gt_depth = self.pool_64(gt_depth)

        pred_vis = th.cat([
            pred_img,
            pred_depth.repeat_interleave(3, dim=1),
            fg_mask.repeat_interleave(3, dim=1),
        ],
                          dim=-1)  # B, 3, H, W

        pred_vis_nv = th.cat([
            pred_cano['image_raw'],
            pred_nv_depth.repeat_interleave(3, dim=1),
            input_fg_mask.repeat_interleave(3, dim=1),
        ],
                             dim=-1)  # B, 3, H, W

        pred_vis = th.cat([pred_vis, pred_vis_nv], dim=-2)  # cat in H dim

        gt_vis = th.cat([
            gt_img,
            gt_depth.repeat_interleave(3, dim=1),
            th.zeros_like(gt_img)
        ],
                        dim=-1)  # TODO, fail to load depth. range [0, 1]

        if 'conf_sigma' in pred:
            gt_vis = th.cat([gt_vis, fg_mask], dim=-1)  # placeholder

        # vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
        vis = th.cat([gt_vis, pred_vis], dim=-2)
        # .permute(
        #     0, 2, 3, 1).cpu()
        vis_tensor = torchvision.utils.make_grid(vis, nrow=vis.shape[-1] //
                                                 64)  # HWC
        torchvision.utils.save_image(
            vis_tensor,
            f'{logger.get_dir()}/{self.step+self.resume_step}.jpg',
            value_range=(-1, 1),
            normalize=True)
        # vis = vis.numpy() * 127.5 + 127.5
        # vis = vis.clip(0, 255).astype(np.uint8)

        # Image.fromarray(vis).save(
        #     f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

        logger.log('log vis to: ',
                   f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

        # self.writer.add_image(f'images',
        #                       vis,
        #                       self.step + self.resume_step,
        #                       dataformats='HWC')


# return pred


class TrainLoop3DRecNVPatch(TrainLoop3DRecNV):
    # add patch rendering
    def __init__(self,
                 *,
                 rec_model,
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
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
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
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)
        # the rendrer
        self.eg3d_model = self.rec_model.module.decoder.triplane_decoder  # type: ignore
        # self.rec_cano = False
        self.rec_cano = True

    def forward_backward(self, batch, *args, **kwargs):
        # add patch sampling

        self.mp_trainer_rec.zero_grad()
        batch_size = batch['img_to_encoder'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev())
                for k, v in batch.items()
            }

            # ! sample rendering patch
            target = {
                **self.eg3d_model(
                    c=micro['nv_c'],  # type: ignore
                    ws=None,
                    planes=None,
                    sample_ray_only=True,
                    fg_bbox=micro['nv_bbox']),  # rays o / dir
            }

            patch_rendering_resolution = self.eg3d_model.rendering_kwargs[
                'patch_rendering_resolution']  # type: ignore
            cropped_target = {
                k:
                th.empty_like(v)
                [..., :patch_rendering_resolution, :patch_rendering_resolution]
                if k not in [
                    'ins_idx', 'img_to_encoder', 'img_sr', 'nv_img_to_encoder',
                    'nv_img_sr', 'c'
                ] else v
                for k, v in micro.items()
            }

            # crop according to uv sampling
            for j in range(micro['img'].shape[0]):
                top, left, height, width = target['ray_bboxes'][
                    j]  # list of tuple
                # for key in ('img', 'depth_mask', 'depth', 'depth_mask_sr'): # type: ignore
                for key in ('img', 'depth_mask', 'depth'):  # type: ignore
                    # target[key][i:i+1] = torchvision.transforms.functional.crop(
                    # cropped_target[key][
                    #     j:j + 1] = torchvision.transforms.functional.crop(
                    #         micro[key][j:j + 1], top, left, height, width)

                    cropped_target[f'{key}'][  # ! no nv_ here
                        j:j + 1] = torchvision.transforms.functional.crop(
                            micro[f'nv_{key}'][j:j + 1], top, left, height,
                            width)

            # target.update(cropped_target)

            # wrap forward within amp
            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                # target_nvs = {}
                # target_cano = {}

                latent = self.rec_model(img=micro['img_to_encoder'],
                                        behaviour='enc_dec_wo_triplane')

                pred_nv = self.rec_model(
                    latent=latent,
                    c=micro['nv_c'],  # predict novel view here
                    behaviour='triplane_dec',
                    ray_origins=target['ray_origins'],
                    ray_directions=target['ray_directions'],
                )

                # ! directly retrieve from target
                # for k, v in target.items():
                #     if k[:2] == 'nv':
                #         orig_key = k.replace('nv_', '')
                #         target_nvs[orig_key] = v
                #         target_cano[orig_key] = target[orig_key]

                with self.rec_model.no_sync():  # type: ignore
                    loss, loss_dict, _ = self.loss_class(pred_nv,
                                                         cropped_target,
                                                         step=self.step +
                                                         self.resume_step,
                                                         test_mode=False,
                                                         return_fg_mask=True,
                                                         conf_sigma_l1=None,
                                                         conf_sigma_percl=None)
                    log_rec3d_loss_dict(loss_dict)

                if self.rec_cano:

                    cano_target = {
                        **self.eg3d_model(
                            c=micro['c'],  # type: ignore
                            ws=None,
                            planes=None,
                            sample_ray_only=True,
                            fg_bbox=micro['bbox']),  # rays o / dir
                    }

                    cano_cropped_target = {
                        k: th.empty_like(v)
                        for k, v in cropped_target.items()
                    }

                    for j in range(micro['img'].shape[0]):
                        top, left, height, width = cano_target['ray_bboxes'][
                            j]  # list of tuple
                        # for key in ('img', 'depth_mask', 'depth', 'depth_mask_sr'): # type: ignore
                        for key in ('img', 'depth_mask',
                                    'depth'):  # type: ignore
                            # target[key][i:i+1] = torchvision.transforms.functional.crop(
                            cano_cropped_target[key][
                                j:j +
                                1] = torchvision.transforms.functional.crop(
                                    micro[key][j:j + 1], top, left, height,
                                    width)

                    # cano_target.update(cano_cropped_target)

                    pred_cano = self.rec_model(
                        latent=latent,
                        c=micro['c'],
                        behaviour='triplane_dec',
                        ray_origins=cano_target['ray_origins'],
                        ray_directions=cano_target['ray_directions'],
                    )

                    with self.rec_model.no_sync():  # type: ignore

                        fg_mask = cano_cropped_target['depth_mask'].unsqueeze(
                            1).repeat_interleave(3, 1).float()

                        loss_cano, loss_cano_dict = self.loss_class.calc_2d_rec_loss(
                            pred_cano['image_raw'],
                            cano_cropped_target['img'],
                            fg_mask,
                            step=self.step + self.resume_step,
                            test_mode=False,
                        )

                    loss = loss + loss_cano

                    # remove redundant log
                    log_rec3d_loss_dict({
                        f'cano_{k}': v
                        for k, v in loss_cano_dict.items()
                        #  if "loss" in k
                    })

            self.mp_trainer_rec.backward(loss)

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                self.log_patch_img(cropped_target, pred_nv, pred_cano)

    @th.inference_mode()
    def log_patch_img(self, micro, pred, pred_cano):
        # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

        def norm_depth(pred_depth):  # to [-1,1]
            # pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())
            return -(pred_depth * 2 - 1)

        pred_img = pred['image_raw']
        gt_img = micro['img']

        # infer novel view also
        # if self.loss_class.opt.symmetry_loss:
        #     pred_nv_img = nvs_pred
        # else:
        # ! replace with novel view prediction

        # ! log another novel-view prediction
        # pred_nv_img = self.rec_model(
        #     img=micro['img_to_encoder'],
        #     c=self.novel_view_poses)  # pred: (B, 3, 64, 64)

        # if 'depth' in micro:
        gt_depth = micro['depth']
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        gt_depth = norm_depth(gt_depth)
        # gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
        #                                           gt_depth.min())
        # if True:
        fg_mask = pred['image_mask'] * 2 - 1  # 0-1
        input_fg_mask = pred_cano['image_mask'] * 2 - 1  # 0-1
        if 'image_depth' in pred:
            pred_depth = norm_depth(pred['image_depth'])
            pred_cano_depth = norm_depth(pred_cano['image_depth'])
        else:
            pred_depth = th.zeros_like(gt_depth)
            pred_cano_depth = th.zeros_like(gt_depth)

        # if 'image_sr' in pred:
        #     if pred['image_sr'].shape[-1] == 512:
        #         pred_img = th.cat([self.pool_512(pred_img), pred['image_sr']],
        #                           dim=-1)
        #         gt_img = th.cat([self.pool_512(micro['img']), micro['img_sr']],
        #                         dim=-1)
        #         pred_depth = self.pool_512(pred_depth)
        #         gt_depth = self.pool_512(gt_depth)

        #     elif pred['image_sr'].shape[-1] == 256:
        #         pred_img = th.cat([self.pool_256(pred_img), pred['image_sr']],
        #                           dim=-1)
        #         gt_img = th.cat([self.pool_256(micro['img']), micro['img_sr']],
        #                         dim=-1)
        #         pred_depth = self.pool_256(pred_depth)
        #         gt_depth = self.pool_256(gt_depth)

        #     else:
        #         pred_img = th.cat([self.pool_128(pred_img), pred['image_sr']],
        #                           dim=-1)
        #         gt_img = th.cat([self.pool_128(micro['img']), micro['img_sr']],
        #                         dim=-1)
        #         gt_depth = self.pool_128(gt_depth)
        #         pred_depth = self.pool_128(pred_depth)
        # else:
        #     gt_img = self.pool_64(gt_img)
        #     gt_depth = self.pool_64(gt_depth)

        pred_vis = th.cat([
            pred_img,
            pred_depth.repeat_interleave(3, dim=1),
            fg_mask.repeat_interleave(3, dim=1),
        ],
                          dim=-1)  # B, 3, H, W

        pred_vis_nv = th.cat([
            pred_cano['image_raw'],
            pred_cano_depth.repeat_interleave(3, dim=1),
            input_fg_mask.repeat_interleave(3, dim=1),
        ],
                             dim=-1)  # B, 3, H, W

        pred_vis = th.cat([pred_vis, pred_vis_nv], dim=-2)  # cat in H dim

        gt_vis = th.cat([
            gt_img,
            gt_depth.repeat_interleave(3, dim=1),
            th.zeros_like(gt_img)
        ],
                        dim=-1)  # TODO, fail to load depth. range [0, 1]

        # if 'conf_sigma' in pred:
        #     gt_vis = th.cat([gt_vis, fg_mask], dim=-1)  # placeholder

        # vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
        # st()
        vis = th.cat([gt_vis, pred_vis], dim=-2)
        # .permute(
        #     0, 2, 3, 1).cpu()
        vis_tensor = torchvision.utils.make_grid(vis, nrow=vis.shape[-1] //
                                                 64)  # HWC
        torchvision.utils.save_image(
            vis_tensor,
            f'{logger.get_dir()}/{self.step+self.resume_step}.jpg',
            value_range=(-1, 1),
            normalize=True)

        logger.log('log vis to: ',
                   f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

        # self.writer.add_image(f'images',
        #                       vis,
        #                       self.step + self.resume_step,
        #                       dataformats='HWC')


class TrainLoop3DRecNVPatchSingleForward(TrainLoop3DRecNVPatch):

    def __init__(self,
                 *,
                 rec_model,
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
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
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
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)

    def forward_backward(self, batch, *args, **kwargs):
        # add patch sampling

        self.mp_trainer_rec.zero_grad()
        batch_size = batch['img_to_encoder'].shape[0]

        batch.pop('caption')  # not required
        batch.pop('ins')  # not required
        batch.pop('nv_caption')  # not required
        batch.pop('nv_ins')  # not required

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v[i:i + self.microbatch]
                for k, v in batch.items()
            }

            # ! sample rendering patch
            target = {
                **self.eg3d_model(
                    c=micro['nv_c'],  # type: ignore
                    ws=None,
                    planes=None,
                    sample_ray_only=True,
                    fg_bbox=micro['nv_bbox']),  # rays o / dir
            }

            patch_rendering_resolution = self.eg3d_model.rendering_kwargs[
                'patch_rendering_resolution']  # type: ignore
            cropped_target = {
                k:
                th.empty_like(v)
                [..., :patch_rendering_resolution, :patch_rendering_resolution]
                if k not in [
                    'ins_idx', 'img_to_encoder', 'img_sr', 'nv_img_to_encoder',
                    'nv_img_sr', 'c', 'caption', 'nv_caption'
                ] else v
                for k, v in micro.items()
            }

            # crop according to uv sampling
            for j in range(micro['img'].shape[0]):
                top, left, height, width = target['ray_bboxes'][
                    j]  # list of tuple
                # for key in ('img', 'depth_mask', 'depth', 'depth_mask_sr'): # type: ignore
                for key in ('img', 'depth_mask', 'depth'):  # type: ignore
                    # target[key][i:i+1] = torchvision.transforms.functional.crop(
                    # cropped_target[key][
                    #     j:j + 1] = torchvision.transforms.functional.crop(
                    #         micro[key][j:j + 1], top, left, height, width)

                    cropped_target[f'{key}'][  # ! no nv_ here
                        j:j + 1] = torchvision.transforms.functional.crop(
                            micro[f'nv_{key}'][j:j + 1], top, left, height,
                            width)

            # ! cano view loss
            # cano_target = {
            #     **self.eg3d_model(
            #         c=micro['c'],  # type: ignore
            #         ws=None,
            #         planes=None,
            #         sample_ray_only=True,
            #         fg_bbox=micro['bbox']),  # rays o / dir
            # }

            # cano_cropped_target = {
            #     k: th.empty_like(v)
            #     for k, v in cropped_target.items()
            # }

            # for j in range(micro['img'].shape[0]):
            #     top, left, height, width = cano_target['ray_bboxes'][
            #         j]  # list of tuple
            #     # for key in ('img', 'depth_mask', 'depth', 'depth_mask_sr'): # type: ignore
            #     for key in ('img', 'depth_mask', 'depth'):  # type: ignore
            #         # target[key][i:i+1] = torchvision.transforms.functional.crop(
            #         cano_cropped_target[key][
            #             j:j + 1] = torchvision.transforms.functional.crop(
            #                 micro[key][j:j + 1], top, left, height, width)

            # ! vit no amp
            latent = self.rec_model(img=micro['img_to_encoder'].to(self.dtype),
                                    behaviour='enc_dec_wo_triplane')

            # wrap forward within amp
            with th.autocast(device_type='cuda',
                            #  dtype=th.float16,
                             dtype=th.bfloat16, # avoid NAN
                             enabled=self.mp_trainer_rec.use_amp):

                # c = th.cat([micro['nv_c'], micro['c']]),  # predict novel view here
                # c = th.cat([micro['nv_c'].repeat(3, 1), micro['c']]),  # predict novel view here
                instance_mv_num = batch_size // 4  # 4 pairs by default
                # instance_mv_num = 4
                # ! roll views for multi-view supervision
                c = th.cat([
                    micro['nv_c'].roll(instance_mv_num * i, dims=0)
                    for i in range(1, 4)
                ]
                           # + [micro['c']]
                           )  # predict novel view here

                ray_origins = th.cat(
                    [
                        target['ray_origins'].roll(instance_mv_num * i, dims=0)
                        for i in range(1, 4)
                    ]
                    # + [cano_target['ray_origins'] ]
                    ,
                    0)

                ray_directions = th.cat([
                    target['ray_directions'].roll(instance_mv_num * i, dims=0)
                    for i in range(1, 4)
                ]
                                        # + [cano_target['ray_directions'] ]
                                        )

                pred_nv_cano = self.rec_model(
                    # latent=latent.expand(2,),
                    latent={
                        'latent_after_vit': # ! triplane for rendering
                        # latent['latent_after_vit'].repeat(2, 1, 1, 1)
                        latent['latent_after_vit'].repeat(3, 1, 1, 1)
                    },
                    c=c,
                    behaviour='triplane_dec',
                    # ray_origins=target['ray_origins'],
                    # ray_directions=target['ray_directions'],
                    ray_origins=ray_origins,
                    ray_directions=ray_directions,
                )

                pred_nv_cano.update(
                    latent
                )  # torchvision.utils.save_image(pred_nv_cano['image_raw'], 'pred.png', normalize=True)
                # gt = {
                #     k: th.cat([v, cano_cropped_target[k]], 0)
                #     for k, v in cropped_target.items()
                # }
                gt = {
                    k:
                    th.cat(
                        [
                            v.roll(instance_mv_num * i, dims=0)
                            for i in range(1, 4)
                        ]
                        # + [cano_cropped_target[k] ]
                        ,
                        0)
                    for k, v in cropped_target.items()
                }  # torchvision.utils.save_image(gt['img'], 'gt.png', normalize=True)

                with self.rec_model.no_sync():  # type: ignore
                    loss, loss_dict, _ = self.loss_class(
                        pred_nv_cano,
                        gt,  # prepare merged data
                        step=self.step + self.resume_step,
                        test_mode=False,
                        return_fg_mask=True,
                        conf_sigma_l1=None,
                        conf_sigma_percl=None)
                    log_rec3d_loss_dict(loss_dict)

            self.mp_trainer_rec.backward(loss)

            # for name, p in self.rec_model.named_parameters():
            #     if p.grad is None:
            #         logger.log(f"found rec unused param: {name}")

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                micro_bs = micro['img_to_encoder'].shape[0]
                self.log_patch_img( # record one cano view and one novel view
                    cropped_target,
                    {
                        k: pred_nv_cano[k][-micro_bs:]
                        for k in ['image_raw', 'image_depth', 'image_mask']
                    },
                    {
                        k: pred_nv_cano[k][:micro_bs]
                        for k in ['image_raw', 'image_depth', 'image_mask']
                    },
                )

    def eval_loop(self):
        return super().eval_loop()

    @th.inference_mode()
    # def eval_loop(self, c_list:list):
    def eval_novelview_loop_old(self, camera=None):
        # novel view synthesis given evaluation camera trajectory

        all_loss_dict = []
        novel_view_micro = {}

        # ! randomly inference an instance

        export_mesh = True
        if export_mesh:
            Path(f'{logger.get_dir()}/FID_Cals/').mkdir(parents=True,
                                                        exist_ok=True)

        # for i in range(0, len(c_list), 1): # TODO, larger batch size for eval

        batch = {}
        # if camera is not None:
        #     # batch['c'] = camera.to(batch['c'].device())
        #     batch['c'] = camera.clone()
        # else:
        #     batch =

        for eval_idx, render_reference in enumerate(tqdm(self.eval_data)):

            if eval_idx > 500:
                break

            video_out = imageio.get_writer(
                f'{logger.get_dir()}/video_novelview_{self.step+self.resume_step}_{eval_idx}.mp4',
                mode='I',
                fps=25,
                codec='libx264')

            with open(
                    f'{logger.get_dir()}/triplane_{self.step+self.resume_step}_{eval_idx}_caption.txt',
                    'w') as f:
                f.write(render_reference['caption'])

            for key in ['ins', 'bbox', 'caption']:
                if key in render_reference:
                    render_reference.pop(key)

            real_flag = False
            mv_flag = False  # TODO, use full-instance for evaluation? Calculate the metrics.
            if render_reference['c'].shape[:2] == (1, 40):
                real_flag = True
                # real img monocular reconstruction
                # compat lst for enumerate
                render_reference = [{
                    k: v[0][idx:idx + 1]
                    for k, v in render_reference.items()
                } for idx in range(40)]

            elif render_reference['c'].shape[0] == 8:
                mv_flag = True

                render_reference = {
                    k: v[:4]
                    for k, v in render_reference.items()
                }

                # save gt
                torchvision.utils.save_image(
                    render_reference[0:4]['img'],
                    logger.get_dir() + '/FID_Cals/{}_inp.png'.format(eval_idx),
                    padding=0,
                    normalize=True,
                    value_range=(-1, 1),
                )
                # torchvision.utils.save_image(render_reference[4:8]['img'],
                #     logger.get_dir() + '/FID_Cals/{}_inp2.png'.format(eval_idx),
                #     padding=0,
                #     normalize=True,
                #     value_range=(-1,1),
                # )

            else:
                # compat lst for enumerate
                st()
                render_reference = [{
                    k: v[idx:idx + 1]
                    for k, v in render_reference.items()
                } for idx in range(40)]

                # ! single-view version
                render_reference[0]['img_to_encoder'] = render_reference[14][
                    'img_to_encoder']  # encode side view
                render_reference[0]['img'] = render_reference[14][
                    'img']  # encode side view

                # save gt
                torchvision.utils.save_image(
                    render_reference[0]['img'],
                    logger.get_dir() + '/FID_Cals/{}_gt.png'.format(eval_idx),
                    padding=0,
                    normalize=True,
                    value_range=(-1, 1))

            # ! TODO, merge with render_video_given_triplane later
            for i, batch in enumerate(render_reference):
                # for i in range(0, 8, self.microbatch):
                # c = c_list[i].to(dist_util.dev()).reshape(1, -1)
                micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

                st()
                if i == 0:
                    if mv_flag:
                        novel_view_micro = None
                    else:
                        novel_view_micro = {
                            k:
                            v[0:1].to(dist_util.dev()).repeat_interleave(
                                # v[14:15].to(dist_util.dev()).repeat_interleave(
                                micro['img'].shape[0],
                                0) if isinstance(v, th.Tensor) else v[0:1]
                            for k, v in batch.items()
                        }

                else:
                    if i == 1:

                        # ! output mesh
                        if export_mesh:

                            # ! get planes first
                            # self.latent_name = 'latent_normalized'  # normalized triplane latent

                            # ddpm_latent = {
                            #     self.latent_name: planes,
                            # }
                            # ddpm_latent.update(self.rec_model(latent=ddpm_latent, behaviour='decode_after_vae_no_render'))

                            # mesh_size = 512
                            # mesh_size = 256
                            mesh_size = 384
                            # mesh_size = 320
                            # mesh_thres = 3 # TODO, requires tuning
                            # mesh_thres = 5 # TODO, requires tuning
                            mesh_thres = 10  # TODO, requires tuning
                            import mcubes
                            import trimesh
                            dump_path = f'{logger.get_dir()}/mesh/'

                            os.makedirs(dump_path, exist_ok=True)

                            grid_out = self.rec_model(
                                latent=pred,
                                grid_size=mesh_size,
                                behaviour='triplane_decode_grid',
                            )

                            vtx, faces = mcubes.marching_cubes(
                                grid_out['sigma'].squeeze(0).squeeze(
                                    -1).cpu().numpy(), mesh_thres)
                            vtx = vtx / (mesh_size - 1) * 2 - 1

                            # vtx_tensor = th.tensor(vtx, dtype=th.float32, device=dist_util.dev()).unsqueeze(0)
                            # vtx_colors = self.model.synthesizer.forward_points(planes, vtx_tensor)['rgb'].squeeze(0).cpu().numpy()  # (0, 1)
                            # vtx_colors = (vtx_colors * 255).astype(np.uint8)

                            # mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)
                            mesh = trimesh.Trimesh(
                                vertices=vtx,
                                faces=faces,
                            )

                            mesh_dump_path = os.path.join(
                                dump_path, f'{eval_idx}.ply')
                            mesh.export(mesh_dump_path, 'ply')

                            print(f"Mesh dumped to {dump_path}")
                            del grid_out, mesh
                            th.cuda.empty_cache()
                            # return
                            # st()

                    # if novel_view_micro['c'].shape[0] < micro['img'].shape[0]:
                    novel_view_micro = {
                        k:
                        v[0:1].to(dist_util.dev()).repeat_interleave(
                            micro['img'].shape[0], 0)
                        for k, v in novel_view_micro.items()
                    }

                pred = self.rec_model(img=novel_view_micro['img_to_encoder'],
                                      c=micro['c'])  # pred: (B, 3, 64, 64)
                # target = {
                #     'img': micro['img'],
                #     'depth': micro['depth'],
                #     'depth_mask': micro['depth_mask']
                # }
                # targe

                # if not export_mesh:
                if not real_flag:
                    _, loss_dict = self.loss_class(pred, micro, test_mode=True)
                    all_loss_dict.append(loss_dict)

                # ! move to other places, add tensorboard

                # pred_vis = th.cat([
                #     pred['image_raw'],
                #     -pred['image_depth'].repeat_interleave(3, dim=1)
                # ],
                #                   dim=-1)

                # normalize depth
                # if True:
                pred_depth = pred['image_depth']
                pred_depth = (pred_depth - pred_depth.min()) / (
                    pred_depth.max() - pred_depth.min())
                if 'image_sr' in pred:

                    if pred['image_sr'].shape[-1] == 512:

                        pred_vis = th.cat([
                            micro['img_sr'],
                            self.pool_512(pred['image_raw']), pred['image_sr'],
                            self.pool_512(pred_depth).repeat_interleave(3,
                                                                        dim=1)
                        ],
                                          dim=-1)

                    elif pred['image_sr'].shape[-1] == 256:

                        pred_vis = th.cat([
                            micro['img_sr'],
                            self.pool_256(pred['image_raw']), pred['image_sr'],
                            self.pool_256(pred_depth).repeat_interleave(3,
                                                                        dim=1)
                        ],
                                          dim=-1)

                    else:
                        pred_vis = th.cat([
                            micro['img_sr'],
                            self.pool_128(pred['image_raw']),
                            self.pool_128(pred['image_sr']),
                            self.pool_128(pred_depth).repeat_interleave(3,
                                                                        dim=1)
                        ],
                                          dim=-1)

                else:
                    # pred_vis = th.cat([
                    #     self.pool_64(micro['img']), pred['image_raw'],
                    #     pred_depth.repeat_interleave(3, dim=1)
                    # ],
                    #                   dim=-1)  # B, 3, H, W

                    pooled_depth = self.pool_128(pred_depth).repeat_interleave(
                        3, dim=1)
                    pred_vis = th.cat(
                        [
                            # self.pool_128(micro['img']),
                            self.pool_128(novel_view_micro['img']
                                          ),  # use the input here
                            self.pool_128(pred['image_raw']),
                            pooled_depth,
                        ],
                        dim=-1)  # B, 3, H, W

                vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
                vis = vis * 127.5 + 127.5
                vis = vis.clip(0, 255).astype(np.uint8)

                if export_mesh:
                    # save image
                    torchvision.utils.save_image(
                        pred['image_raw'],
                        logger.get_dir() +
                        '/FID_Cals/{}_{}.png'.format(eval_idx, i),
                        padding=0,
                        normalize=True,
                        value_range=(-1, 1))

                    torchvision.utils.save_image(
                        pooled_depth,
                        logger.get_dir() +
                        '/FID_Cals/{}_{}_dpeth.png'.format(eval_idx, i),
                        padding=0,
                        normalize=True,
                        value_range=(0, 1))

                # st()

                for j in range(vis.shape[0]):
                    video_out.append_data(vis[j])

            video_out.close()

        # if not export_mesh:
        if not real_flag or mv_flag:
            val_scores_for_logging = calc_average_loss(all_loss_dict)
            with open(os.path.join(logger.get_dir(), 'scores_novelview.json'),
                      'a') as f:
                json.dump({'step': self.step, **val_scores_for_logging}, f)

            # * log to tensorboard
            for k, v in val_scores_for_logging.items():
                self.writer.add_scalar(f'Eval/NovelView/{k}', v,
                                       self.step + self.resume_step)

        del video_out
        # del pred_vis
        # del pred

        th.cuda.empty_cache()

    @th.inference_mode()
    # def eval_loop(self, c_list:list):
    def eval_novelview_loop(self, camera=None, save_latent=False):
        # novel view synthesis given evaluation camera trajectory
        if save_latent:  # for diffusion learning
            latent_dir = Path(f'{logger.get_dir()}/latent_dir')
            latent_dir.mkdir(exist_ok=True, parents=True)

            # wds_path = os.path.join(logger.get_dir(), 'latent_dir',
            #                         f'wds-%06d.tar')
            # sink = wds.ShardWriter(wds_path, start_shard=0)

        # eval_batch_size = 20
        # eval_batch_size = 1
        eval_batch_size = 40  # ! for i23d

        latent_rec_statistics = False

        for eval_idx, micro in enumerate(tqdm(self.eval_data)):

            # if eval_idx > 500:
            #     break

            latent = self.rec_model(
                img=micro['img_to_encoder'],
                behaviour='encoder_vae')  # pred: (B, 3, 64, 64)
            # torchvision.utils.save_image(micro['img'], 'inp.jpg')
            if micro['img'].shape[0] == 40:
                assert eval_batch_size == 40

            if save_latent:
                # np.save(f'{logger.get_dir()}/latent_dir/{eval_idx}.npy', latent[self.latent_name].cpu().numpy())

                latent_save_dir = f'{logger.get_dir()}/latent_dir/{micro["ins"][0]}'
                Path(latent_save_dir).mkdir(parents=True, exist_ok=True)

                np.save(f'{latent_save_dir}/latent.npy',
                        latent[self.latent_name][0].cpu().numpy())
                assert all([
                    micro['ins'][0] == micro['ins'][i]
                    for i in range(micro['c'].shape[0])
                ])  # ! assert same instance

                # for i in range(micro['img'].shape[0]):

                #     compressed_sample = {
                #         'latent':latent[self.latent_name][0].cpu().numpy(), # 12 32 32
                #         'caption': micro['caption'][0].encode('utf-8'),
                #         'ins': micro['ins'][0].encode('utf-8'),
                #         'c': micro['c'][i].cpu().numpy(),
                #         'img': micro['img'][i].cpu().numpy() # 128x128, for diffusion log
                #     }

                #     sink.write({
                #         "__key__": f"sample_{eval_idx*eval_batch_size+i:07d}",
                #         'sample.pyd': compressed_sample
                #     })

            if latent_rec_statistics:
                gen_imgs = self.render_video_given_triplane(
                    latent[self.latent_name],
                    self.rec_model,  # compatible with join_model
                    name_prefix=f'{self.step + self.resume_step}_{eval_idx}',
                    save_img=False,
                    render_reference={'c': micro['c']},
                    save_mesh=False,
                    render_reference_length=4,
                    return_gen_imgs=True)
                rec_psnr = psnr((micro['img'] / 2 + 0.5),
                                (gen_imgs.cpu() / 2 + 0.5), 1.0)
                with open(
                        os.path.join(logger.get_dir(),
                                     'four_view_rec_psnr.json'), 'a') as f:
                    json.dump(
                        {
                            f'{eval_idx}': {
                                'ins': micro["ins"][0],
                                'psnr': rec_psnr.item(),
                            }
                        }, f)
                #  save to json

            elif eval_idx < 30:
                # if False:
                self.render_video_given_triplane(
                    latent[self.latent_name],
                    self.rec_model,  # compatible with join_model
                    name_prefix=f'{self.step + self.resume_step}_{micro["ins"][0].split("/")[0]}_{eval_idx}',
                    save_img=False,
                    render_reference={'c': camera},
                    save_mesh=True)


class TrainLoop3DRecNVPatchSingleForwardMV(TrainLoop3DRecNVPatchSingleForward):

    def __init__(self,
                 *,
                 rec_model,
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
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
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
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)

    def forward_backward(self, batch, behaviour='g_step', *args, **kwargs):
        # add patch sampling

        if behaviour == 'g_step':
            self.mp_trainer_rec.zero_grad()
        else:
            self.mp_trainer_disc.zero_grad()

        batch_size = batch['img_to_encoder'].shape[0]

        batch.pop('caption')  # not required
        batch.pop('ins')  # not required
        batch.pop('nv_caption')  # not required
        batch.pop('nv_ins')  # not required

        if '__key__' in batch.keys():
            batch.pop('__key__')

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v[i:i + self.microbatch]
                for k, v in batch.items()
            }

            # ! sample rendering patch
            # nv_c = th.cat([micro['nv_c'], micro['c']])
            nv_c = th.cat([micro['nv_c'], micro['c']])
            # nv_c = micro['nv_c']
            target = {
                **self.eg3d_model(
                    c=nv_c,  # type: ignore
                    ws=None,
                    planes=None,
                    sample_ray_only=True,
                    fg_bbox=th.cat([micro['nv_bbox'], micro['bbox']])),  # rays o / dir
            }

            patch_rendering_resolution = self.eg3d_model.rendering_kwargs[
                'patch_rendering_resolution']  # type: ignore
            cropped_target = {
                k:
                th.empty_like(v).repeat_interleave(2, 0)
                # th.empty_like(v).repeat_interleave(1, 0)
                [..., :patch_rendering_resolution, :patch_rendering_resolution]
                if k not in [
                    'ins_idx', 'img_to_encoder', 'img_sr', 'nv_img_to_encoder',
                    'nv_img_sr', 'c', 'caption', 'nv_caption'
                ] else v
                for k, v in micro.items()
            }

            # crop according to uv sampling
            for j in range(2 * self.microbatch):
                top, left, height, width = target['ray_bboxes'][
                    j]  # list of tuple
                # for key in ('img', 'depth_mask', 'depth', 'depth_mask_sr'): # type: ignore
                for key in ('img', 'depth_mask', 'depth'):  # type: ignore

                    if j < self.microbatch:
                        cropped_target[f'{key}'][  # ! no nv_ here
                            j:j + 1] = torchvision.transforms.functional.crop(
                                micro[f'nv_{key}'][j:j + 1], top, left, height,
                                width)
                    else:
                        cropped_target[f'{key}'][  # ! no nv_ here
                            j:j + 1] = torchvision.transforms.functional.crop(
                                micro[f'{key}'][j - self.microbatch:j -
                                                self.microbatch + 1], top,
                                left, height, width)

            # for j in range(batch_size, 2*batch_size, 1):
            #     top, left, height, width = target['ray_bboxes'][
            #         j]  # list of tuple
            #     # for key in ('img', 'depth_mask', 'depth', 'depth_mask_sr'): # type: ignore
            #     for key in ('img', 'depth_mask', 'depth'):  # type: ignore

            #         cropped_target[f'{key}'][  # ! no nv_ here
            #             j:j + 1] = torchvision.transforms.functional.crop(
            #                 micro[f'{key}'][j-batch_size:j-batch_size + 1], top, left, height,
            #                 width)

            # wrap forward within amp
            with th.autocast(device_type='cuda',
                             dtype=self.dtype,
                             enabled=self.mp_trainer_rec.use_amp):

                # c = th.cat([micro['nv_c'], micro['c']]),  # predict novel view here
                # c = th.cat([micro['nv_c'].repeat(3, 1), micro['c']]),  # predict novel view here
                # instance_mv_num = batch_size // 4  # 4 pairs by default
                # instance_mv_num = 4
                # ! roll views for multi-view supervision
                # c = micro['nv_c']

                # ! vit no amp
                latent = self.rec_model(img=micro['img_to_encoder'].to(self.dtype),
                                        behaviour='enc_dec_wo_triplane')

                # # ! disable amp in rendering and loss
                # with th.autocast(device_type='cuda',
                #                 dtype=th.float16,
                #                 enabled=False):

                ray_origins = target['ray_origins']
                ray_directions = target['ray_directions']

                pred_nv_cano = self.rec_model(
                    # latent=latent.expand(2,),
                    latent={
                        'latent_after_vit':  # ! triplane for rendering
                        # latent['latent_after_vit'].repeat_interleave(4, dim=0).repeat(2,1,1,1)  # NV=4
                        latent['latent_after_vit'].repeat_interleave(6, dim=0).repeat(2,1,1,1)  # NV=6
                        # latent['latent_after_vit'].repeat_interleave(10, dim=0).repeat(2,1,1,1)  # NV=4
                        # latent['latent_after_vit'].repeat_interleave(8, dim=0)  # NV=4
                    },
                    c=nv_c,
                    behaviour='triplane_dec',
                    ray_origins=ray_origins,
                    ray_directions=ray_directions,
                )

                pred_nv_cano.update(
                    latent
                )  # torchvision.utils.save_image(pred_nv_cano['image_raw'], 'pred.png', normalize=True)
                gt = cropped_target

                with self.rec_model.no_sync():  # type: ignore
                    loss, loss_dict, _ = self.loss_class(
                        pred_nv_cano,
                        gt,  # prepare merged data
                        step=self.step + self.resume_step,
                        test_mode=False,
                        return_fg_mask=True,
                        behaviour=behaviour,
                        conf_sigma_l1=None,
                        conf_sigma_percl=None, 
                        # dtype=self.dtype
                        )
                    log_rec3d_loss_dict(loss_dict)

            if behaviour == 'g_step':
                self.mp_trainer_rec.backward(loss)
            else:
                self.mp_trainer_disc.backward(loss)

            # for name, p in self.rec_model.named_parameters():
            #     if p.grad is None:
            #         logger.log(f"found rec unused param: {name}")
            # torchvision.utils.save_image(cropped_target['img'], 'gt.png', normalize=True)
            # torchvision.utils.save_image( pred_nv_cano['image_raw'], 'pred.png', normalize=True)

            if dist_util.get_rank() == 0 and self.step % 500 == 0 and i == 0 and behaviour == 'g_step':
                try:
                    torchvision.utils.save_image(
                        th.cat(
                            [cropped_target['img'], pred_nv_cano['image_raw']
                             ], ),
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg',
                        normalize=True, nrow=6*2)

                    logger.log(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')
                except Exception as e:
                    logger.log(e)

                # micro_bs = micro['img_to_encoder'].shape[0]
                # self.log_patch_img( # record one cano view and one novel view
                #     cropped_target,
                #     {
                #         k: pred_nv_cano[k][0:1]
                #         for k in ['image_raw', 'image_depth', 'image_mask']
                #     },
                #     {
                #         k: pred_nv_cano[k][1:2]
                #         for k in ['image_raw', 'image_depth', 'image_mask']
                #     },
                # )

    # def save(self):
    #     return super().save()


class TrainLoop3DRecNVPatchSingleForwardMVAdvLoss(
        TrainLoop3DRecNVPatchSingleForwardMV):

    def __init__(self,
                 *,
                 rec_model,
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
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
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
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)

        # create discriminator
        disc_params = self.loss_class.get_trainable_parameters()

        self.mp_trainer_disc = MixedPrecisionTrainer(
            model=self.loss_class.discriminator,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            model_name='disc',
            use_amp=use_amp,
            model_params=disc_params)

        # st() # check self.lr
        self.opt_disc = AdamW(
            self.mp_trainer_disc.master_params,
            lr=self.lr,  # follow sd code base
            betas=(0, 0.999),
            eps=1e-8)

        # TODO, is loss cls already in the DDP?
        if self.use_ddp:
            self.ddp_disc = DDP(
                self.loss_class.discriminator,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.ddp_disc = self.loss_class.discriminator

    # def run_st

    # def run_step(self, batch, *args):
    #     self.forward_backward(batch)
    #     took_step = self.mp_trainer_rec.optimize(self.opt)
    #     if took_step:
    #         self._update_ema()
    #     self._anneal_lr()
    #     self.log_step()

    def save(self, mp_trainer=None, model_name='rec'):
        if mp_trainer is None:
            mp_trainer = self.mp_trainer_rec

        def save_checkpoint(rate, params):
            state_dict = mp_trainer.master_params_to_state_dict(params)
            if dist_util.get_rank() == 0:
                logger.log(f"saving model {model_name} {rate}...")
                if not rate:
                    filename = f"model_{model_name}{(self.step+self.resume_step):07d}.pt"
                else:
                    filename = f"ema_{model_name}_{rate}_{(self.step+self.resume_step):07d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename),
                                 "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, mp_trainer.master_params)

        dist.barrier()

    # ! load disc
    def _load_and_sync_parameters(self, submodule_name=''):
        super()._load_and_sync_parameters(submodule_name)
        # load disc

        resume_checkpoint = self.resume_checkpoint.replace(
            'rec', 'disc')  # * default behaviour
        if os.path.exists(resume_checkpoint):
            if dist_util.get_rank() == 0:
                logger.log(
                    f"loading disc model from checkpoint: {resume_checkpoint}..."
                )
                map_location = {
                    'cuda:%d' % 0: 'cuda:%d' % dist_util.get_rank()
                }  # configure map_location properly

                resume_state_dict = dist_util.load_state_dict(
                    resume_checkpoint, map_location=map_location)
                model_state_dict = self.loss_class.discriminator.state_dict()

                for k, v in resume_state_dict.items():
                    if k in model_state_dict.keys():
                        if v.size() == model_state_dict[k].size():
                            model_state_dict[k] = v
                            # model_state_dict[k].copy_(v)
                        else:
                            logger.log('!!!! partially load: ', k, ": ",
                                       v.size(), "state_dict: ",
                                       model_state_dict[k].size())

            if dist_util.get_world_size() > 1:
                # dist_util.sync_params(self.model.named_parameters())
                dist_util.sync_params(
                    self.loss_class.get_trainable_parameters())
                logger.log('synced disc params')

    def run_step(self, batch, step='g_step'):
        # self.forward_backward(batch)

        if step == 'g_step':
            self.forward_backward(batch, behaviour='g_step')
            took_step_g_rec = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        elif step == 'd_step':
            self.forward_backward(batch, behaviour='d_step')
            _ = self.mp_trainer_disc.optimize(self.opt_disc)

        self._anneal_lr()
        self.log_step()

    def run_loop(self, batch=None):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            batch = next(self.data)
            self.run_step(batch, 'g_step')

            batch = next(self.data)
            self.run_step(batch, 'd_step')

            if self.step % 1000 == 0:
                dist_util.synchronize()
                if self.step % 5000 == 0:
                    th.cuda.empty_cache()  # avoid memory leak

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                # * log to tensorboard
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            if self.step % self.eval_interval == 0 and self.step != 0:
                if dist_util.get_rank() == 0:
                    try:
                        self.eval_loop()
                    except Exception as e:
                        logger.log(e)
                dist_util.synchronize()

            # if self.step % self.save_interval == 0 and self.step != 0:
            if self.step % self.save_interval == 0:
                self.save()
                self.save(self.mp_trainer_disc,
                          self.mp_trainer_disc.model_name)
                dist_util.synchronize()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                logger.log('reached maximum iterations, exiting')

                # Save the last checkpoint if it wasn't already saved.
                if (self.step -
                        1) % self.save_interval != 0 and self.step != 1:
                    self.save()

                exit()

        # Save the last checkpoint if it wasn't already saved.
        # if (self.step - 1) % self.save_interval != 0 and self.step != 1:
        if (self.step - 1) % self.save_interval != 0:
            self.save()  # save rec
            self.save(self.mp_trainer_disc, self.mp_trainer_disc.model_name)


class TrainLoop3DRecNVPatchSingleForwardMV_NoCrop(
        TrainLoop3DRecNVPatchSingleForwardMV):

    def __init__(self,
                 *,
                 rec_model,
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
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 num_frames=4,
                 **kwargs):
        super().__init__(rec_model=rec_model,
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
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)

        self.num_frames = num_frames
        self.ray_sampler = RaySampler()

        print(self.opt)

        # ! requires tuning
        N = 768 # hyp param, overfitting now
        # self.scale_expected_threshold = (1 / (N/2)) ** 0.5 * 0.45
        self.scale_expected_threshold = 0.0075
        self.latent_name = 'latent_normalized'  # normalized triplane latent


        # to transform to 3dgs
        self.gs_bg_color=th.tensor([1,1,1], dtype=th.float32, device=dist_util.dev())
        self.post_process = PostProcess(
            384,
            384,
            imgnet_normalize=True,
            plucker_embedding=True,
            decode_encode_img_only=False,
            mv_input=True,
            split_chunk_input=16,
            duplicate_sample=True,
            append_depth=False,
            append_xyz=False,
            gs_cam_format=True,
            orthog_duplicate=False,
            frame_0_as_canonical=False,
            pcd_path='pcd_path',
            load_pcd=True,
            split_chunk_size=16,
        )

        self.zfar = 100.0
        self.znear = 0.01


    # def _init_optim_groups(self, kwargs):
    #     return super()._init_optim_groups({**kwargs, 'ignore_encoder': True}) # freeze MVEncoder to accelerate training.

    def forward_backward(self, batch, behaviour='g_step', *args, **kwargs):
        # add patch sampling

        self.mp_trainer_rec.zero_grad()
        batch_size = batch['img_to_encoder'].shape[0]

        batch.pop('caption')  # not required
        ins = batch.pop('ins')  # not required

        if '__key__' in batch.keys():
            batch.pop('__key__')

        assert isinstance(batch['c'], dict)

        for i in range(0, batch_size, self.microbatch):

            micro = {}

            for k, v in batch.items():  # grad acc
                if isinstance(v, th.Tensor):
                    micro[k] = v[i:i + self.microbatch].to(dist_util.dev())
                elif isinstance(v, list):
                    micro[k] = v[i:i + self.microbatch]
                elif isinstance(v, dict):  #
                    assert k in ['c', 'nv_c']
                    micro[k] = {
                        key:
                        value[i:i + self.microbatch].to(dist_util.dev()) if
                        isinstance(value, th.Tensor) else value  # can be float
                        for key, value in v.items()
                    }

            assert micro['img_to_encoder'].shape[1] == 15
            micro['normal'] = micro['img_to_encoder'][:, 3:6]
            micro['nv_normal'] = micro['nv_img_to_encoder'][:, 3:6]

            # ! concat nv_c to render N+N views

            indices = np.random.permutation(self.num_frames)
            indices, indices_nv = indices[:4], indices[-4:] # make sure thorough pose converage.
            # indices, indices_nv = indices[:2], indices[-2:] # ! 2+2 views for supervision, as in gs-lrm. 
            # indices_nv = np.random.permutation(self.num_frames)[:6] # randomly pick 4+4 views for supervision.

            # indices = np.arange(self.num_frames)
            # indices_nv = np.arange(self.num_frames)

            nv_c = {}
            for key in micro['c'].keys():
                if isinstance(micro['c'][key], th.Tensor):
                    nv_c[key] = th.cat([micro['c'][key][:, indices], micro['nv_c'][key][:, indices_nv]],
                                       1)  # B 2V ...
                else:
                    nv_c[key] = micro['c'][key]  # float, will remove later

            target = {}

            for key in ('img', 'depth_mask', 'depth', 'normal',):  # type: ignore
                # st()
                target[key] = th.cat([
                    rearrange(micro[key], '(B V) ... -> B V ...', V=self.num_frames)[:, indices],
                    rearrange(micro[f'nv_{key}'], '(B V) ... -> B V ...', V=self.num_frames)[:, indices_nv]
                    # rearrange(micro[key][:, indices], '(B V) ... -> B V ...', V=4),
                    # rearrange(micro[f'nv_{key}'][:, indices], '(B V) ... -> B V ...', V=4)
                ], 1)  # B 2*V H W
                target[key] = rearrange(target[key],
                                        'B V ... -> (B V) ...')  # concat

            # st()

            # wrap forward within amp
            with th.autocast(device_type='cuda',
                             dtype=self.dtype,
                             enabled=self.mp_trainer_rec.use_amp):
                
                # ! vit no amp
                # with profile(activities=[
                #     ProfilerActivity.CUDA], record_shapes=True) as prof:
                #     with record_function("get_gs"):
                latent = self.rec_model(
                    img=micro['img_to_encoder'].to(self.dtype),
                    behaviour='enc_dec_wo_triplane',
                    c=micro['c'],
                    pcd=micro['fps_pcd'], # send in pcd for surface reference.
                )  # send in input-view C since pixel-aligned gaussians required

                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

                gaussians, query_pcd_xyz = latent['gaussians'], latent['query_pcd_xyz']
                # query_pcd_xyz = latent['query_pcd_xyz']

                # if self.loss_class.opt.rand_aug_bg and random.random()>0.9:
                if self.loss_class.opt.rand_aug_bg:
                    bg_color=torch.randint(0,255,(3,), device=dist_util.dev()) / 255.0
                else:
                    bg_color=torch.tensor([1,1,1], dtype=torch.float32, device=dist_util.dev())
                
                def visualize_latent_activations(latent, b_idx=0, write=False, ):

                    def normalize_latent_plane(latent_plane):
                        avg_p1 = latent_plane.detach().cpu().numpy().mean(0, keepdims=0)
                        avg_p1 = (avg_p1 - avg_p1.min()) / (avg_p1.max() - avg_p1.min())
                        # return avg_p1
                        return ((avg_p1).clip(0,1)*255.0).astype(np.uint8)

                    p1, p2, p3 = (normalize_latent_plane(latent_plane) for latent_plane in (latent[b_idx, 0:4], latent[b_idx,4:8], latent[b_idx,8:12]))

                    if write:
                        plt.imsave(os.path.join(logger.get_dir(), f'{self.step}_{b_idx}_1.jpg'), p1)
                        plt.imsave(os.path.join(logger.get_dir(), f'{self.step}_{b_idx}_2.jpg'), p2)
                        plt.imsave(os.path.join(logger.get_dir(), f'{self.step}_{b_idx}_3.jpg'), p3)
                        # imageio.imwrite(os.path.join(logger.get_dir(), f'{b_idx}_1.jpg'), p1)
                        # imageio.imwrite(os.path.join(logger.get_dir(), f'{b_idx}_2.jpg'), p2)
                        # imageio.imwrite(os.path.join(logger.get_dir(), f'{b_idx}_3.jpg'), p3)

                    return p1, p2, p3


                # with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU,], record_shapes=True) as prof:
                #     # ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #     with record_function("rendering"):

                pred_nv_cano = self.rec_model(
                    latent=latent,
                    # latent={
                    #     'gaussians': latent['gaussians'].repeat_interleave(2,0)
                    # },
                    c=nv_c,
                    behaviour='triplane_dec',
                    bg_color=bg_color,
                )

                fine_scale_key = list(pred_nv_cano.keys())[-1]
                
                # st()
                fine_gaussians = latent[fine_scale_key]
                fine_gaussians_opa = fine_gaussians[..., 3:4]

                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

                # st() # torchvision.utils.save_image(pred_nv_cano['image_raw'][0], 'pred.jpg', normalize=True, value_range=(-1,1))

                if self.loss_class.opt.rand_aug_bg:
                    # bg_color
                    alpha_mask = target['depth_mask'].float().unsqueeze(1) # B 1 H W
                    target['img'] = target['img'] * alpha_mask + (bg_color.reshape(1,3,1,1) * 2 - 1) * (1-alpha_mask)
                
                target['depth_mask'] = target['depth_mask'].unsqueeze(1)
                target['depth'] = target['depth'].unsqueeze(1)

                multiscale_target = defaultdict(dict)
                multiscale_pred = defaultdict(dict)


                for idx, (gaussian_wavelet_key, gaussian_wavelet) in enumerate(pred_nv_cano.items()):
                    gs_output_size = pred_nv_cano[gaussian_wavelet_key]['image_raw'].shape[-1]
                    for k in gaussian_wavelet.keys():
                        pred_nv_cano[gaussian_wavelet_key][k] = rearrange(
                            gaussian_wavelet[k], 'B V ... -> (B V) ...')  # match GT shape order

                    # if idx == 0: # only KL calculation in scale 0
                    if gaussian_wavelet_key == fine_scale_key:
                        pred_nv_cano[gaussian_wavelet_key].update(
                            {
                                k: latent[k] for k in ['posterior']
                            }
                        )  # ! for KL supervision

                    # ! prepare target according to the wavelet size
                    for k in target.keys():

                        if target[key].shape[-1] == gs_output_size:
                            multiscale_target[gaussian_wavelet_key][k] = target[k]
                        else:

                            if k in ('depth', 'normal'):
                                mode = 'nearest'
                            else:
                                mode='bilinear'

                            multiscale_target[gaussian_wavelet_key][k] = F.interpolate(target[k], size=(gs_output_size, gs_output_size), mode=mode)

                # st()

                # st()
                # torchvision.utils.save_image(target['img'], 'gt.jpg', normalize=True, value_range=(-1,1))
                # torchvision.utils.save_image(pred_nv_cano['image_raw'], 'pred.jpg', normalize=True, value_range=(-1,1))
                # torchvision.utils.save_image(micro['img'], 'inp_gt.jpg', normalize=True, value_range=(-1,1))
                # torchvision.utils.save_image(micro['nv_img'], 'nv_gt.jpg', normalize=True, value_range=(-1,1))

                # if self.loss_class.opt.rand_aug_bg:
                #     # bg_color
                #     alpha_mask = target['depth_mask'].float().unsqueeze(1) # B 1 H W
                #     target['img'] = target['img'] * alpha_mask + (bg_color.reshape(1,3,1,1) * 2 - 1) * (1-alpha_mask)

                lod_num = len(pred_nv_cano.keys())
                random_scale_for_lpips = random.choice(list(pred_nv_cano.keys()))

                with self.rec_model.no_sync():  # type: ignore
                #     with profile(activities=[
                #         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                    # with record_function("loss"):

                    loss = th.tensor(0., device=dist_util.dev())
                    loss_dict = {}

                    if behaviour == 'd_step':

                        loss_scale, loss_dict_scale, _ = self.loss_class(
                            pred_nv_cano[random_scale_for_lpips],
                            multiscale_target[random_scale_for_lpips],  # prepare merged data
                            step=self.step + self.resume_step,
                            test_mode=False,
                            return_fg_mask=True,
                            behaviour=behaviour,
                            conf_sigma_l1=None,
                            conf_sigma_percl=None, 
                            ignore_kl=True,  # only calculate once
                            ignore_lpips=True, # lpips on each lod
                            ignore_d_loss=False)
                        
                        loss = loss + loss_scale
                        loss_dict.update(
                            {
                                f"{gaussian_wavelet_key.replace('gaussians_', '')}/{loss_key}": loss_v for loss_key, loss_v in loss_dict_scale.items()
                            }
                        )

                    else:

                        for scale_idx, gaussian_wavelet_key in enumerate(pred_nv_cano.keys()): # ! multi-scale gs rendering supervision

                            loss_scale, loss_dict_scale, _ = self.loss_class(
                                pred_nv_cano[gaussian_wavelet_key],
                                multiscale_target[gaussian_wavelet_key],  # prepare merged data
                                step=self.step + self.resume_step,
                                test_mode=False,
                                return_fg_mask=True,
                                behaviour=behaviour,
                                conf_sigma_l1=None,
                                conf_sigma_percl=None, 
                                ignore_kl=gaussian_wavelet_key!=fine_scale_key,  # only calculate once
                                ignore_lpips=gaussian_wavelet_key!=random_scale_for_lpips, # lpips on each lod
                                ignore_d_loss=gaussian_wavelet_key!=fine_scale_key)
                            
                            loss = loss + loss_scale
                            loss_dict.update(
                                {
                                    f"{gaussian_wavelet_key.replace('gaussians_', '')}/{loss_key}": loss_v for loss_key, loss_v in loss_dict_scale.items()
                                }
                            )

                pos = latent['pos']
                opacity = gaussians[..., 3:4]
                scaling = gaussians[..., 4:6] # 2dgs here

                if self.step % self.log_interval == 0 and dist_util.get_rank(
                ) == 0:
                    with th.no_grad():  # save idx 0 here
                        try:
                            self.writer.add_histogram("scene/opacity_hist",
                                                    opacity[0][:],
                                                    self.step + self.resume_step)
                            self.writer.add_histogram("scene/scale_hist",
                                                    scaling[0][:],
                                                    self.step + self.resume_step)
                        except Exception as e:
                            logger.log(e)

                if behaviour == 'g_step':

                    # ! 2dgs loss
                    # debugging now, open it from the beginning
                    # if (self.step + self.resume_step) >= 2000 and self.loss_class.opt.lambda_normal > 0:
                    surf_normal = multiscale_target[fine_scale_key]['normal'] * multiscale_target[fine_scale_key]['depth_mask'] # foreground supervision only.

                    # ! hard-coded
                    # rend_normal = pred_nv_cano['rend_normal'] # ! supervise disk normal with GT normal here instead; 
                    # st()
                    rend_normal = pred_nv_cano[fine_scale_key]['rend_normal']
                    rend_dist = pred_nv_cano[fine_scale_key]['dist']

                    if self.loss_class.opt.lambda_scale_reg > 0:
                        scale_reg = (scaling-self.scale_expected_threshold).square().mean() * self.loss_class.opt.lambda_scale_reg 
                        loss = loss + scale_reg
                        loss_dict.update({'loss_scale_reg': scale_reg})
                    
                    if self.loss_class.opt.lambda_opa_reg > 0:
                        # small_base_opa = latent['gaussians_base_opa']
                        opa_reg = (-self.loss_class.beta_mvp_base_dist.log_prob(latent['gaussians_base_opa'].clamp(min=1/255, max=0.99)).mean()) * self.loss_class.opt.lambda_opa_reg
                        # ! also on the fine stage
                        opa_reg_fine = (-self.loss_class.beta_mvp_base_dist.log_prob(fine_gaussians_opa.clamp(min=1/255, max=0.99)).mean()) * self.loss_class.opt.lambda_opa_reg
                        # opa_reg = (1-latent['gaussians_base_opa'].mean() ) * self.loss_class.opt.lambda_opa_reg
                        loss = loss + opa_reg + opa_reg_fine
                        loss_dict.update({'loss_opa_reg': opa_reg, 'loss_opa_reg_fine': opa_reg_fine})


                    if (self.step + self.resume_step) >= 35000 and self.loss_class.opt.lambda_normal > 0:
                    # if (self.step + self.resume_step) >= 2000 and self.loss_class.opt.lambda_normal > 0:
                        # surf_normal = unity2blender_th(surf_normal) # ! g-buffer normal system is different

                        normal_error = (1 - (rend_normal * surf_normal).sum(dim=1)) # B H W
                        # normal_loss = self.loss_class.opt.lambda_normal * (normal_error.sum() / target['depth_mask'].sum()) # average with fg area ratio
                        normal_loss = self.loss_class.opt.lambda_normal * normal_error.mean()

                        loss = loss + normal_loss

                        loss_dict.update({'loss_normal': normal_loss})

                    # if (self.step + self.resume_step) >= 1500 and self.loss_class.opt.lambda_dist > 0:
                    if (self.step + self.resume_step) >= 15000 and self.loss_class.opt.lambda_dist > 0:
                    # if (self.step + self.resume_step) >= 300 and self.loss_class.opt.lambda_dist > 0:
                        dist_loss = self.loss_class.opt.lambda_dist * (rend_dist).mean()
                        loss = loss + dist_loss
                        loss_dict.update({'loss_dist': dist_loss})

                    if self.loss_class.opt.pruning_ot_lambda > 0:
                        # for now, save and analyze first
                        # selected_pts_mask_scaling = th.where(th.max(scaling, dim=-1).values < 0.01 * 0.9, True, False)
                        selected_pts_mask_scaling = th.where(
                            th.max(scaling, dim=-1).values > 0.05 * 0.9, True,
                            False)
                        # selected_pts_mask_opacity = th.where(opacity[..., 0] < 0.1, True, False) # B N
                        selected_pts_mask_opacity = th.where(
                            opacity[..., 0] < 0.01, True,
                            False)  # 0.005 in the original 3dgs setting

                        selected_scaling_pts = pos[0][selected_pts_mask_scaling[0]]
                        selected_opacity_pts = pos[0][selected_pts_mask_opacity[0]]

                        pcu.save_mesh_v(
                            f'tmp/voxel/cd/10/scaling_masked_pts_0.05.ply',
                            selected_scaling_pts.detach().cpu().numpy(),
                        )

                        pcu.save_mesh_v(
                            f'tmp/voxel/cd/10/opacity_masked_pts_0.01.ply',
                            selected_opacity_pts.detach().cpu().numpy(),
                        )

                        # st()
                        # pass

                    if self.loss_class.opt.cd_lambda > 0:
                        # fuse depth to 3D point cloud to supervise the gaussians
                        B = latent['pos'].shape[0]
                        # c = micro['c']
                        # H = micro['depth'].shape[-1]
                        # V = 4
                        # # ! prepare 3D xyz ground truth

                        # cam2world_matrix = c['orig_c2w'][:, :, :16].reshape(
                        #     B * V, 4, 4)
                        # intrinsics = c['orig_pose'][:, :,
                        #                             16:25].reshape(B * V, 3, 3)

                        # # ! already in the world space after ray_sampler()
                        # ray_origins, ray_directions = self.ray_sampler(  # shape:
                        #     cam2world_matrix, intrinsics, H // 2)[:2]

                        # # depth = th.nn.functional.interpolate(micro['depth'].unsqueeze(1), (128,128), mode='nearest')[:, 0] # since each view has 128x128 Gaussians
                        # # depth = th.nn.functional.interpolate(micro['depth'].unsqueeze(1), (128,128), mode='nearest')[:, 0] # since each view has 128x128 Gaussians
                        # depth_128 = th.nn.functional.interpolate(
                        #     micro['depth'].unsqueeze(1), (128, 128),
                        #     mode='nearest'
                        # )[:, 0]  # since each view has 128x128 Gaussians
                        # depth = depth_128.reshape(B * V, -1).unsqueeze(-1)
                        # # depth = micro['depth'].reshape(B*V, -1).unsqueeze(-1)

                        # gt_pos = ray_origins + depth * ray_directions  # BV HW 3, already in the world space
                        # gt_pos = rearrange(gt_pos,
                        #                    '(B V) N C -> B (V N) C',
                        #                    B=B,
                        #                    V=V)
                        # gt_pos = gt_pos.clip(-0.45, 0.45)

                        # TODO
                        gt_pos = micro[
                            'fps_pcd']  # all the same, will update later.

                        # ! use online here
                        # gt_pos = query_pcd_xyz

                        cd_loss = pytorch3d.loss.chamfer_distance(
                            gt_pos, latent['pos']
                        )[0] * self.loss_class.opt.cd_lambda  # V=4 GT for now. Test with V=8 GT later.
                        # st()

                        # for vis
                        if False:
                            torchvision.utils.save_image(micro['img'],
                                                        'gt.jpg',
                                                        value_range=(-1, 1),
                                                        normalize=True)
                            with th.no_grad():
                                for b in range(B):
                                    pcu.save_mesh_v(
                                        f'tmp/voxel/cd/10/again_pred-{b}.ply',
                                        latent['pos'][b].detach().cpu().numpy(),
                                    )
                                    # pcu.save_mesh_v(
                                    #     f'tmp/voxel/cd/10/again-gt-{b}.ply',
                                    #     gt_pos[b].detach().cpu().numpy(),
                                    # )
                        # st()

                        loss = loss + cd_loss
                        loss_dict.update({'loss_cd': cd_loss})

                    elif self.loss_class.opt.xyz_lambda > 0:
                        '''
                        B = latent['per_view_pos'].shape[0] // 4
                        V = 4
                        c = micro['c']
                        H = micro['depth'].shape[-1]
                        # ! prepare 3D xyz ground truth

                        cam2world_matrix = c['orig_c2w'][:, :, :16].reshape(
                            B * V, 4, 4)
                        intrinsics = c['orig_pose'][:, :,
                                                    16:25].reshape(B * V, 3, 3)

                        # ! already in the world space after ray_sampler()
                        ray_origins, ray_directions = self.ray_sampler(  # shape: 
                            cam2world_matrix, intrinsics, H // 2)[:2]
                        # self.gs.output_size,)[:2]
                        # depth = rearrange(micro['depth'], '(B V) H W -> ')
                        depth_128 = th.nn.functional.interpolate(
                            micro['depth'].unsqueeze(1), (128, 128),
                            mode='nearest'
                        )[:, 0]  # since each view has 128x128 Gaussians
                        depth = depth_128.reshape(B * V, -1).unsqueeze(-1)
                        fg_mask = th.nn.functional.interpolate(
                            micro['depth_mask'].unsqueeze(1).to(th.uint8),
                            (128, 128),
                            mode='nearest').squeeze(1)  # B*V H W
                        fg_mask = fg_mask.reshape(B * V, -1).unsqueeze(-1)
                        gt_pos = ray_origins + depth * ray_directions  # BV HW 3, already in the world space
                        # st()
                        gt_pos = fg_mask * gt_pos.clip(
                            -0.45, 0.45)  # g-buffer objaverse range
                        pred = fg_mask * latent['per_view_pos']

                        # for vis
                        if True:
                            torchvision.utils.save_image(micro['img'],
                                                        'gt.jpg',
                                                        value_range=(-1, 1),
                                                        normalize=True)
                            with th.no_grad():
                                gt_pos_vis = rearrange(gt_pos,
                                                    '(B V) N C -> B V N C',
                                                    B=B,
                                                    V=V)
                                pred_pos_vis = rearrange(pred,
                                                        '(B V) N C -> B V N C',
                                                        B=B,
                                                        V=V)
                                # save

                                for b in range(B):
                                    for v in range(V):
                                        # pcu.save_mesh_v(f'tmp/dust3r/add3dsupp-pred-{b}-{v}.ply',
                                        #                 pred_pos_vis[b][v].detach().cpu().numpy(),)
                                        # pcu.save_mesh_v(f'tmp/dust3r/add3dsupp-gt-{b}-{v}.ply',
                                        #                 gt_pos_vis[b][v].detach().cpu().numpy(),)
                                        pcu.save_mesh_v(
                                            f'tmp/lambda50/no3dsupp-pred-{b}-{v}.ply',
                                            pred_pos_vis[b]
                                            [v].detach().cpu().numpy(),
                                        )
                                        pcu.save_mesh_v(
                                            f'tmp/lambda50/no3dsupp-gt-{b}-{v}.ply',
                                            gt_pos_vis[b]
                                            [v].detach().cpu().numpy(),
                                        )
                        st()

                        xyz_loss = th.nn.functional.mse_loss(
                            gt_pos, pred
                        ) * self.loss_class.opt.xyz_lambda  # ! 15% nonzero points
                        loss = loss + xyz_loss
                        '''

                        # ! directly gs center supervision with l1 loss, follow LION VAE

                        # xyz_loss = th.nn.functional.l1_loss(
                        #     query_pcd_xyz, pred
                        # ) * self.loss_class.opt.xyz_lambda  # ! 15% nonzero points
                        xyz_loss = self.loss_class.criterion_xyz(query_pcd_xyz, latent['pos']) * self.loss_class.opt.xyz_lambda
                        loss = loss + xyz_loss

                        # only calculate foreground gt_pos here?
                        loss_dict.update({'loss_xyz': xyz_loss})

                    elif self.loss_class.opt.emd_lambda > 0:
                        # rand_pt_size = 4096  # K value. Input Error! The size of the point clouds should be a multiple of 1024.
                        pred = latent['pos']
                        rand_pt_size = min(2048, max(pred.shape[1], 1024))  # K value. Input Error! The size of the point clouds should be a multiple of 1024.

                        if micro['fps_pcd'].shape[0] == pred.shape[0]:
                            gt_point = micro['fps_pcd']
                        else:  # overfit memory dataset
                            gt_point = micro[
                                'fps_pcd'][::
                                        4]  # consecutive 4 views are from the same ID

                        B, gt_point_N = gt_point.shape[:2]
                        # random sample pred points
                        # sampled_pred =
                        # rand_pt_idx = torch.randint(high=pred.shape[1]-gt_point_N, size=(B,))


                        # pcu.save_mesh_v( f'tmp/voxel/emd/gt-half.ply', gt_point[0, ::4].detach().cpu().numpy(),)

                        # for b in range(gt_point.shape[0]):
                        #     pcu.save_mesh_v( f'{logger.get_dir()}/gt-{b}.ply', gt_point[b].detach().cpu().numpy(),)

                        #     pcu.save_mesh_v( f'{logger.get_dir()}/pred-{b}.ply', pred[b].detach().cpu().numpy(),)

                        # pcu.save_mesh_v( f'0.ply', latent['pos'][0].detach().cpu().numpy())
                        # st()  

                        if self.loss_class.opt.fps_sampling:  # O(N*K). reduce K later.

                            if self.loss_class.opt.subset_fps_sampling:
                                rand_pt_size = 1024  # for faster calculation
                                # ! uniform sampling with randomness
                                # sampled_gt_pts_for_emd_loss = gt_point[:, random.randint(0,9)::9][:, :1024] # direct uniform downsample to the K size
                                # sampled_gt_pts_for_emd_loss = gt_point[:, random.randint(0,9)::4][:, :1024] # direct uniform downsample to the K size

                                rand_perm = torch.randperm(
                                    gt_point.shape[1]
                                )[:rand_pt_size]  # shuffle the xyz before downsample - fps sampling
                                sampled_gt_pts_for_emd_loss = gt_point[:, rand_perm]

                                # sampled_gt_pts_for_emd_loss = gt_point[:, ::4]
                                # sampled_pred_pts_for_emd_loss = pred[:, ::32]
                                # sampled_gt_pts_for_emd_loss = pytorch3d.ops.sample_farthest_points(
                                #     gt_point[:, ::4], K=rand_pt_size)[0] # V4

                                if self.loss_class.opt.subset_half_fps_sampling:
                                    # sampled_pred_pts_for_emd_loss = pytorch3d.ops.sample_farthest_points(
                                    #     pred, K=rand_pt_size)[0] # V5
                                    rand_perm = torch.randperm(
                                        pred.shape[1]
                                    )[:4096]  # shuffle the xyz before downsample - fps sampling
                                    sampled_pred_pts_for_emd_loss = pytorch3d.ops.sample_farthest_points(
                                        pred[:, rand_perm],
                                        K=rand_pt_size)[0]  # improve randomness
                                else:
                                    sampled_pred_pts_for_emd_loss = pytorch3d.ops.sample_farthest_points(
                                        pred[:, ::4], K=rand_pt_size)[0]  # V5

                                # rand_perm = torch.randperm(pred.shape[1]) # shuffle the xyz before downsample - fps sampling
                                # sampled_pred_pts_for_emd_loss = pytorch3d.ops.sample_farthest_points(
                                #     pred[:, rand_perm][:, ::4], K=rand_pt_size)[0] # rand perm before downsampling, V6

                                # sampled_pred_pts_for_emd_loss = pytorch3d.ops.sample_farthest_points(
                                #     pred[:, rand_perm][:, ::8], K=rand_pt_size)[0] # rand perm before downsampling, V7

                                # sampled_pred_pts_for_emd_loss = pytorch3d.ops.sample_farthest_points(
                                #     pred[:, self.step%2::4], K=rand_pt_size)[0] # rand perm before downsampling, V8, based on V50

                            else:
                                sampled_gt_pts_for_emd_loss = pytorch3d.ops.sample_farthest_points(
                                    gt_point, K=rand_pt_size)[0]

                                # if self.loss_class.opt.subset_half_fps_sampling:
                                # rand_pt_size = 4096 # K value. Input Error! The size of the point clouds should be a multiple of 1024.

                                sampled_pred_pts_for_emd_loss = pytorch3d.ops.sample_farthest_points(
                                    pred, K=rand_pt_size)[0]

                                # else:
                                #     sampled_pred_pts_for_emd_loss = pytorch3d.ops.sample_farthest_points(
                                #         pred, K=rand_pt_size)[0]

                        else:  # random sampling
                            rand_pt_idx_pred = torch.randint(high=pred.shape[1] -
                                                            rand_pt_size,
                                                            size=(1, ))[0]
                            rand_pt_idx_gt = torch.randint(high=gt_point.shape[1] -
                                                        rand_pt_size,
                                                        size=(1, ))[0]

                            sampled_pred_pts_for_emd_loss = pred[:,
                                                                rand_pt_idx_pred:
                                                                rand_pt_idx_pred +
                                                                rand_pt_size, ...]
                            sampled_gt_pts_for_emd_loss = gt_point[:,
                                                                rand_pt_idx_gt:
                                                                rand_pt_idx_gt +
                                                                rand_pt_size,
                                                                ...]

                        # only calculate foreground gt_pos here?

                        emd_loss = calc_emd(sampled_gt_pts_for_emd_loss,
                                            sampled_pred_pts_for_emd_loss).mean(
                                            ) * self.loss_class.opt.emd_lambda
                        loss = loss + emd_loss
                        loss_dict.update({'loss_emd': emd_loss})

                    if self.loss_class.opt.commitment_loss_lambda > 0:
                        ellipsoid_vol = torch.prod(scaling, dim=-1, keepdim=True) / ((0.01 * 0.9)**3) # * (4/3*torch.pi). normalized vol
                        commitment = ellipsoid_vol * opacity
                        to_be_pruned_ellipsoid_idx = commitment < (3/4)**3 * 0.9 # those points shall have larger vol*opacity contribution
                        commitment_loss = -commitment[to_be_pruned_ellipsoid_idx].mean() * self.loss_class.opt.commitment_loss_lambda
                        
                        loss = loss + commitment_loss
                        loss_dict.update({'loss_commitment': commitment_loss})
                        loss_dict.update({'loss_commitment_opacity': opacity.mean()})
                        loss_dict.update({'loss_commitment_vol': ellipsoid_vol.mean()})


                log_rec3d_loss_dict(loss_dict)

            # self.mp_trainer_rec.backward(loss)
            if behaviour == 'g_step':
                self.mp_trainer_rec.backward(loss)
            else:
                self.mp_trainer_disc.backward(loss)

            # for name, p in self.rec_model.named_parameters():
            #     if p.grad is None:
            #         logger.log(f"found rec unused param: {name}")

            #     print(name, p.grad.mean(), p.grad.abs().max())

            if dist_util.get_rank() == 0 and self.step % 500 == 0 and i == 0 and behaviour=='g_step':
            # if dist_util.get_rank() == 0 and self.step % 1 == 0 and i == 0:
                try:
                    torchvision.utils.save_image(
                        th.cat([target['img'][::1], pred_nv_cano[fine_scale_key]['image_raw'][::1]], ),
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg',
                        normalize=True, value_range=(-1,1),nrow=len(indices)*2)

                    # save depth and normal and alpha
                    torchvision.utils.save_image(
                        th.cat([surf_normal[::1], rend_normal[::1]], ),
                        f'{logger.get_dir()}/{self.step+self.resume_step}_normal_new.jpg',
                        normalize=True, value_range=(-1,1), nrow=len(indices)*2)

                    torchvision.utils.save_image(
                        th.cat([target['depth'][::1], pred_nv_cano[fine_scale_key]['image_depth'][::1]], ),
                        f'{logger.get_dir()}/{self.step+self.resume_step}_depth.jpg',
                        normalize=True, nrow=len(indices)*2)

                    # torchvision.utils.save_image( pred_nv_cano['image_depth'][::1], f'{logger.get_dir()}/{self.step+self.resume_step}_depth.jpg', normalize=True, nrow=len(indices)*2)

                    torchvision.utils.save_image(
                        th.cat([target['depth_mask'][::1], pred_nv_cano[fine_scale_key]['image_mask'][::1]], ),
                        f'{logger.get_dir()}/{self.step+self.resume_step}_alpha.jpg',
                        normalize=True, value_range=(0,1), nrow=len(indices)*2)


                    logger.log(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

                except Exception as e:
                    logger.log('Exception when saving log: ', e)

            # if self.step % 2500 == 0:
            #     th.cuda.empty_cache() # free vram

    @torch.no_grad()
    def export_mesh_from_2dgs(self, all_rgbs, all_depths, all_alphas, cam_pathes, latent_save_dir):
        # https://github.com/autonomousvision/LaRa/blob/main/evaluation.py
        n_thread = 1 # avoid TSDF cpu hanging bug.
        os.environ["MKL_NUM_THREADS"] = f"{n_thread}" 
        os.environ["NUMEXPR_NUM_THREADS"] = f"{n_thread}" 
        os.environ["OMP_NUM_THREADS"] = f"4" 
        os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_thread}" 
        os.environ["OPENBLAS_NUM_THREADS"] = f"{n_thread}" 

        # copied from: https://github.com/hbb1/2d-gaussian-splatting/blob/19eb5f1e091a582e911b4282fe2832bac4c89f0f/render.py#L23
        logger.log("exporting mesh ...")

        # for g-objv
        aabb = [-0.45,-0.45,-0.45,0.45,0.45,0.45]
        self.aabb = np.array(aabb).reshape(2,3)*1.1

        name = f'{latent_save_dir}/mesh_raw.obj'
        mesh = self.extract_mesh_bounded(all_rgbs, all_depths, all_alphas, cam_pathes)
        
        o3d.io.write_triangle_mesh(name, mesh)
        logger.log("mesh saved at {}".format(name))
        mesh_post = smooth_mesh(mesh)
        o3d.io.write_triangle_mesh(name.replace('_raw.obj', '.obj'), mesh_post)
        logger.log("mesh post processed saved at {}".format(name.replace('.obj', '_post.obj')))

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
            # radius = np.linalg.norm(self.aabb[1] - self.aabb[0]) * 0.5
            radius = np.linalg.norm(self.aabb[1] - self.aabb[0]) * 0.5
            # voxel_size = radius / 256
            voxel_size = radius / 192 # less holes
            # sdf_trunc = voxel_size * 16 # less holes, slower integration
            sdf_trunc = voxel_size * 12 # 
            print("using aabb")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        # print(f'depth_truc: {depth_trunc}')

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
    def eval_novelview_loop(self, camera=None, save_latent=False):
        # novel view synthesis given evaluation camera trajectory
        if save_latent:  # for diffusion learning
            latent_dir = Path(f'{logger.get_dir()}/latent_dir')
            latent_dir.mkdir(exist_ok=True, parents=True)

        render_reference=uni_mesh_path(10)

        for eval_idx, micro in enumerate(tqdm(self.eval_data)):

            latent_save_dir = f'{logger.get_dir()}/latent_dir/{micro["ins"][0]}'

            all_latent_file = sorted(Path(latent_save_dir).glob('*.npz') )
            if len(all_latent_file) == 0:
                save_prefix = 0
            else:
                save_prefix = int(all_latent_file[-1].stem[-1] ) + 1

            Path(latent_save_dir).mkdir(parents=True, exist_ok=True)

            with th.autocast(device_type='cuda',
                             dtype=self.dtype,
                             enabled=self.mp_trainer_rec.use_amp):

                latent = self.rec_model(
                    img=micro['img_to_encoder'].to(self.dtype),
                    behaviour='enc_dec_wo_triplane',
                    c=micro['c'],
                    pcd=micro['fps_pcd'], # send in pcd for surface reference.
                )  # send in input-view C since pixel-aligned gaussians required

            # fine_scale_key = list(pred.keys())[-1]
            # fine_scale_key = 'gaussians_upsampled_2'
            fine_scale_key = 'gaussians_upsampled_3'
            export_mesh = True # for debug

            if True:

                # if eval_idx < 1500 and eval_idx % 3 == 0:
                if eval_idx < 1500:
                    all_rgbs, all_depths, all_alphas=self.render_gs_video_given_latent(
                        latent,
                        self.rec_model,  # compatible with join_model
                        name_prefix=f'{self.step + self.resume_step}_{micro["ins"][0].split("/")[0]}_{eval_idx}',
                        save_img=False,
                        render_reference=render_reference,
                        export_mesh=False)

                    if export_mesh:
                        self.export_mesh_from_2dgs(all_rgbs, all_depths, all_alphas, render_reference, latent_save_dir)

                # ! B=2 here
                np.savez_compressed(f'{latent_save_dir}/latent-{save_prefix}.npz', 
                    latent_normalized=latent['latent_normalized'].cpu().numpy(),
                    query_pcd_xyz=latent['query_pcd_xyz'].cpu().numpy()
                )

                # st()
                for scale in ['gaussians_upsampled', 'gaussians_base', 'gaussians_upsampled_2', 'gaussians_upsampled_3']:
                    np.save(f'{latent_save_dir}/{scale}.npy', latent[scale].cpu().numpy())
            

    @th.inference_mode()
    def render_gs_video_given_latent(self,
                                    ddpm_latent,
                                    rec_model,
                                    name_prefix='0',
                                    save_img=False, 
                                    render_reference=None, 
                                    export_mesh=False):

        all_rgbs, all_depths, all_alphas = [], [], []

        # batch_size, L, C = planes.shape

        # ddpm_latent = { self.latent_name: planes[..., :-3] * self.triplane_scaling_divider,  # kl-reg latent
        #                 'query_pcd_xyz': self.pcd_unnormalize_fn(planes[..., -3:]) }
        
        # ddpm_latent.update(rec_model(latent=ddpm_latent, behaviour='decode_gs_after_vae_no_render')) 
        

        # assert render_reference is None
            # render_reference = self.eval_data # compat
        # else: # use train_traj

        # for key in ['ins', 'bbox', 'caption']:
        #     if key in render_reference:
        #         render_reference.pop(key)

        # render_reference = [ { k:v[idx:idx+1] for k, v in render_reference.items() } for idx in range(40) ]

        video_out = imageio.get_writer(
            f'{logger.get_dir()}/gs_{name_prefix}.mp4',
            mode='I',
            fps=15,
            codec='libx264')

        # for i, batch in enumerate(tqdm(self.eval_data)):
        for i, micro_c in enumerate(tqdm(render_reference)):
            # micro = {
            #     k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v
            #     for k, v in batch.items() 
            # }

            c = self.post_process.c_to_3dgs_format(micro_c)
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
                render_all_scale=True,
                )

            fine_scale_key = list(pred.keys())[-1]

            all_rgbs.append(einops.rearrange(pred[fine_scale_key]['image'], 'B V ... -> (B V) ...'))
            all_depths.append(einops.rearrange(pred[fine_scale_key]['depth'], 'B V ... -> (B V) ...'))
            all_alphas.append(einops.rearrange(pred[fine_scale_key]['alpha'], 'B V ... -> (B V) ...'))

            # st()
            # fine_scale_key = list(pred.keys())[-1]
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
            # all_pred_vis_concat = np.concatenate([cv2.resize(all_pred_vis[k][0], (256*3, 256)) for k in ['gaussians_base', 'gaussians_upsampled',]], axis=0)
            # all_pred_vis_concat = np.concatenate([cv2.resize(all_pred_vis[k][0], (384*3, 384)) for k in all_pred_vis.keys()], axis=0)
            all_pred_vis_concat = np.concatenate([cv2.resize(all_pred_vis[k][0], (512*3, 512)) for k in all_pred_vis.keys()], axis=0)

            # for j in range(vis.shape[0]):  
            video_out.append_data(all_pred_vis_concat)

        video_out.close()

        print('logged video to: ',
            f'{logger.get_dir()}/triplane_{name_prefix}.mp4')

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


class TrainLoop3DRecNVPatchSingleForwardMV_NoCrop_adv(TrainLoop3DRecNVPatchSingleForwardMV_NoCrop):
    def __init__(self, *, rec_model, loss_class, data, eval_data, batch_size, microbatch, lr, ema_rate, log_interval, eval_interval, save_interval, resume_checkpoint, use_fp16=False, fp16_scale_growth=0.001, weight_decay=0, lr_anneal_steps=0, iterations=10001, load_submodule_name='', ignore_resume_opt=False, model_name='rec', use_amp=False, num_frames=4, **kwargs):
        super().__init__(rec_model=rec_model, loss_class=loss_class, data=data, eval_data=eval_data, batch_size=batch_size, microbatch=microbatch, lr=lr, ema_rate=ema_rate, log_interval=log_interval, eval_interval=eval_interval, save_interval=save_interval, resume_checkpoint=resume_checkpoint, use_fp16=use_fp16, fp16_scale_growth=fp16_scale_growth, weight_decay=weight_decay, lr_anneal_steps=lr_anneal_steps, iterations=iterations, load_submodule_name=load_submodule_name, ignore_resume_opt=ignore_resume_opt, model_name=model_name, use_amp=use_amp, num_frames=num_frames, **kwargs)

        # create discriminator
        # ! copied from ln3diff tri-plane version
        disc_params = self.loss_class.get_trainable_parameters()

        self.mp_trainer_disc = MixedPrecisionTrainer(
            model=self.loss_class.discriminator,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            model_name='disc',
            use_amp=use_amp,
            model_params=disc_params)

        # st() # check self.lr
        self.opt_disc = AdamW(
            self.mp_trainer_disc.master_params,
            lr=self.lr,  # follow sd code base
            betas=(0, 0.999),
            eps=1e-8)

        # TODO, is loss cls already in the DDP?
        if self.use_ddp:
            self.ddp_disc = DDP(
                self.loss_class.discriminator,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.ddp_disc = self.loss_class.discriminator

    def save(self, mp_trainer=None, model_name='rec'):
        if mp_trainer is None:
            mp_trainer = self.mp_trainer_rec

        def save_checkpoint(rate, params):
            state_dict = mp_trainer.master_params_to_state_dict(params)
            if dist_util.get_rank() == 0:
                logger.log(f"saving model {model_name} {rate}...")
                if not rate:
                    filename = f"model_{model_name}{(self.step+self.resume_step):07d}.pt"
                else:
                    filename = f"ema_{model_name}_{rate}_{(self.step+self.resume_step):07d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename),
                                 "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, mp_trainer.master_params)

        dist.barrier()

    def run_step(self, batch, step='g_step'):
        # self.forward_backward(batch)

        if step == 'g_step':
            self.forward_backward(batch, behaviour='g_step')
            took_step_g_rec = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        elif step == 'd_step':
            self.forward_backward(batch, behaviour='d_step')
            _ = self.mp_trainer_disc.optimize(self.opt_disc)

        self._anneal_lr()
        self.log_step()

    def run_loop(self, batch=None):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            batch = next(self.data)
            self.run_step(batch, 'g_step')

            batch = next(self.data)
            self.run_step(batch, 'd_step')

            if self.step % 1000 == 0:
                dist_util.synchronize()
                if self.step % 5000 == 0:
                    th.cuda.empty_cache()  # avoid memory leak

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                # * log to tensorboard
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            if self.step % self.eval_interval == 0 and self.step != 0:
                if dist_util.get_rank() == 0:
                    try:
                        self.eval_loop()
                    except Exception as e:
                        logger.log(e)
                dist_util.synchronize()

            # if self.step % self.save_interval == 0 and self.step != 0:
            if self.step % self.save_interval == 0:
                self.save()
                self.save(self.mp_trainer_disc,
                          self.mp_trainer_disc.model_name)
                dist_util.synchronize()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                logger.log('reached maximum iterations, exiting')

                # Save the last checkpoint if it wasn't already saved.
                if (self.step -
                        1) % self.save_interval != 0 and self.step != 1:
                    self.save()

                exit()

        # Save the last checkpoint if it wasn't already saved.
        # if (self.step - 1) % self.save_interval != 0 and self.step != 1:
        if (self.step - 1) % self.save_interval != 0:
            try:
                self.save()  # save rec
                self.save(self.mp_trainer_disc, self.mp_trainer_disc.model_name)
            except Exception as e:
                logger.log(e)

    # ! load disc
    def _load_and_sync_parameters(self, submodule_name=''):
        super()._load_and_sync_parameters(submodule_name)
        # load disc

        resume_checkpoint = self.resume_checkpoint.replace(
            'rec', 'disc')  # * default behaviour
        if os.path.exists(resume_checkpoint):
            if dist_util.get_rank() == 0:
                logger.log(
                    f"loading disc model from checkpoint: {resume_checkpoint}..."
                )
                map_location = {
                    'cuda:%d' % 0: 'cuda:%d' % dist_util.get_rank()
                }  # configure map_location properly

                resume_state_dict = dist_util.load_state_dict(
                    resume_checkpoint, map_location=map_location)
                model_state_dict = self.loss_class.discriminator.state_dict()

                for k, v in resume_state_dict.items():
                    if k in model_state_dict.keys():
                        if v.size() == model_state_dict[k].size():
                            model_state_dict[k] = v
                            # model_state_dict[k].copy_(v)
                        else:
                            logger.log('!!!! partially load: ', k, ": ",
                                       v.size(), "state_dict: ",
                                       model_state_dict[k].size())

            if dist_util.get_world_size() > 1:
                # dist_util.sync_params(self.model.named_parameters())
                dist_util.sync_params(
                    self.loss_class.get_trainable_parameters())
                logger.log('synced disc params')

