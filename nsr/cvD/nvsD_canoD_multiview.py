import functools
import json
import os
from pathlib import Path
from pdb import set_trace as st
import torchvision
import blobfile as bf
import imageio
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision
from PIL import Image
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from tqdm import tqdm

from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion import dist_util, logger
from guided_diffusion.train_util import (calc_average_loss,
                                         log_rec3d_loss_dict,
                                         find_resume_checkpoint)

from torch.optim import AdamW

from .nvsD_canoD import TrainLoop3DcvD_nvsD_canoD


class TrainLoop3DcvD_nvsD_canoD_multiview(TrainLoop3DcvD_nvsD_canoD):

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
                         use_amp=use_amp,
                         **kwargs)
        assert not self.mp_trainer_rec.use_amp, 'amp may lead to grad nan?'

    def forward_G_rec(self, batch):  # update G

        self.mp_trainer_rec.zero_grad()
        self.rec_model.requires_grad_(True)

        self.ddp_cano_cvD.requires_grad_(False)
        self.ddp_nvs_cvD.requires_grad_(False)

        batch_size = batch['img'].shape[0]

        target_cano = {}

        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }

            for k, v in micro.items():
                if k[:2] == 'nv':
                    orig_key = k.replace('nv_', '')
                    # target_nvs[orig_key] = v
                    target_cano[orig_key] = micro[orig_key]

            # last_batch = (i + self.microbatch) >= batch_size

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                pred = self.rec_model(
                    img=micro['img_to_encoder'], c=micro['c']
                )  # render novel view for first half of the batch for D loss

                target_for_rec = micro
                cano_pred = pred

                with self.rec_model.no_sync():  # type: ignore
                    loss, loss_dict, fg_mask = self.loss_class(
                        cano_pred,
                        target_for_rec,
                        test_mode=False,
                        step=self.step + self.resume_step,
                        return_fg_mask=True)

                if 'image_sr' in cano_pred:
                    raise NotImplementedError()
                    # concat both resolution
                    vision_aided_loss = self.ddp_cano_cvD(
                        th.cat([
                            th.nn.functional.interpolate(
                                cano_pred['image_raw'],
                                size=cano_pred['image_sr'].shape[2:],
                                mode='bilinear',
                                align_corners=False,
                                antialias=True),
                            cano_pred['image_sr'],
                        ],
                               dim=1),  # 6 channel input
                        for_G=True).mean()  # [B, 1] shape

                else:
                    vision_aided_loss = self.ddp_cano_cvD(
                        cano_pred['image_raw'],
                        for_G=True).mean()  # [B, 1] shape

                # last_layer = self.rec_model.module.decoder.triplane_decoder.decoder.net[  # type: ignore
                #     -1].weight  # type: ignore

                d_weight = th.tensor(self.loss_class.opt.rec_cvD_lambda).to(
                    dist_util.dev())
                # d_weight = calculate_adaptive_weight(
                #     loss,
                #     vision_aided_loss,
                #     last_layer,
                #     disc_weight_max=0.1) * self.loss_class.opt.rec_cvD_lambda
                loss += vision_aided_loss * d_weight

                loss_dict.update({
                    'vision_aided_loss/G_rec':
                    (vision_aided_loss * d_weight).detach(),
                    'd_weight':
                    d_weight
                })

                log_rec3d_loss_dict(loss_dict)

            self.mp_trainer_rec.backward(
                loss)  # no nvs cvD loss, following VQ3D

            # DDP some parameters no grad:
            # for name, p in self.ddp_model.named_parameters():
            #     if p.grad is None:
            #         print(f"(in rec)found rec unused param: {name}")

            # ! move to other places, add tensorboard

            # if dist_util.get_rank() == 0 and self.step % 500 == 0:
            #     with th.no_grad():
            #         # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

            #         gt_depth = micro['depth']
            #         if gt_depth.ndim == 3:
            #             gt_depth = gt_depth.unsqueeze(1)
            #         gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
            #                                                   gt_depth.min())
            #         # if True:
            #         pred_depth = pred['image_depth']
            #         pred_depth = (pred_depth - pred_depth.min()) / (
            #             pred_depth.max() - pred_depth.min())
            #         pred_img = pred['image_raw']
            #         gt_img = micro['img']

            #         if 'image_sr' in pred:
            #             if pred['image_sr'].shape[-1] == 512:
            #                 pred_img = th.cat(
            #                     [self.pool_512(pred_img), pred['image_sr']],
            #                     dim=-1)
            #                 gt_img = th.cat(
            #                     [self.pool_512(micro['img']), micro['img_sr']],
            #                     dim=-1)
            #                 pred_depth = self.pool_512(pred_depth)
            #                 gt_depth = self.pool_512(gt_depth)

            #             elif pred['image_sr'].shape[-1] == 256:
            #                 pred_img = th.cat(
            #                     [self.pool_256(pred_img), pred['image_sr']],
            #                     dim=-1)
            #                 gt_img = th.cat(
            #                     [self.pool_256(micro['img']), micro['img_sr']],
            #                     dim=-1)
            #                 pred_depth = self.pool_256(pred_depth)
            #                 gt_depth = self.pool_256(gt_depth)

            #             else:
            #                 pred_img = th.cat(
            #                     [self.pool_128(pred_img), pred['image_sr']],
            #                     dim=-1)
            #                 gt_img = th.cat(
            #                     [self.pool_128(micro['img']), micro['img_sr']],
            #                     dim=-1)
            #                 gt_depth = self.pool_128(gt_depth)
            #                 pred_depth = self.pool_128(pred_depth)
            #         else:
            #             gt_img = self.pool_64(gt_img)
            #             gt_depth = self.pool_64(gt_depth)

            #         gt_vis = th.cat(
            #             [gt_img, gt_depth.repeat_interleave(3, dim=1)],
            #             dim=-1)  # TODO, fail to load depth. range [0, 1]

            #         pred_vis = th.cat(
            #             [pred_img,
            #              pred_depth.repeat_interleave(3, dim=1)],
            #             dim=-1)  # B, 3, H, W

            #         vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
            #             1, 2, 0).cpu()  # ! pred in range[-1, 1]
            #         # vis_grid = torchvision.utils.make_grid(vis) # HWC
            #         vis = vis.numpy() * 127.5 + 127.5
            #         vis = vis.clip(0, 255).astype(np.uint8)
            #         Image.fromarray(vis).save(
            #             f'{logger.get_dir()}/{self.step+self.resume_step}_rec.jpg'
            #         )
            #         print(
            #             'log vis to: ',
            #             f'{logger.get_dir()}/{self.step+self.resume_step}_rec.jpg'
            #         )

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                with th.no_grad():
                    # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

                    def norm_depth(pred_depth):  # to [-1,1]
                        # pred_depth = pred['image_depth']
                        pred_depth = (pred_depth - pred_depth.min()) / (
                            pred_depth.max() - pred_depth.min())
                        return -(pred_depth * 2 - 1)

                    pred_img = pred['image_raw'].clip(-1, 1)
                    gt_img = micro['img']

                    # infer novel view also
                    pred_nv_img = self.rec_model(
                        img=micro['img_to_encoder'],
                        c=self.novel_view_poses)  # pred: (B, 3, 64, 64)

                    # if 'depth' in micro:
                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = norm_depth(gt_depth)
                    # gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                    #                                           gt_depth.min())
                    # if True:
                    if 'image_depth' in pred:
                        # pred_depth = pred['image_depth']
                        # pred_depth = (pred_depth - pred_depth.min()) / (
                        #     pred_depth.max() - pred_depth.min())
                        pred_depth = norm_depth(pred['image_depth'])
                        pred_nv_depth = norm_depth(pred_nv_img['image_depth'])
                    else:
                        pred_depth = th.zeros_like(gt_depth)
                        pred_nv_depth = th.zeros_like(gt_depth)

                    if 'image_sr' in pred:
                        if pred['image_sr'].shape[-1] == 512:
                            pred_img = th.cat(
                                [self.pool_512(pred_img), pred['image_sr']],
                                dim=-1)
                            gt_img = th.cat(
                                [self.pool_512(micro['img']), micro['img_sr']],
                                dim=-1)
                            pred_depth = self.pool_512(pred_depth)
                            gt_depth = self.pool_512(gt_depth)

                        elif pred['image_sr'].shape[-1] == 256:
                            pred_img = th.cat(
                                [self.pool_256(pred_img), pred['image_sr']],
                                dim=-1)
                            gt_img = th.cat(
                                [self.pool_256(micro['img']), micro['img_sr']],
                                dim=-1)
                            pred_depth = self.pool_256(pred_depth)
                            gt_depth = self.pool_256(gt_depth)

                        else:
                            pred_img = th.cat(
                                [self.pool_128(pred_img), pred['image_sr']],
                                dim=-1)
                            gt_img = th.cat(
                                [self.pool_128(micro['img']), micro['img_sr']],
                                dim=-1)
                            gt_depth = self.pool_128(gt_depth)
                            pred_depth = self.pool_128(pred_depth)

                    if gt_img.shape[-1] == 64:
                        gt_depth = self.pool_64(gt_depth)
                    elif gt_img.shape[-1] == 128:
                        gt_depth = self.pool_128(gt_depth)
                    # else:
                    # gt_depth = self.pool_64(gt_depth)

                    # st()
                    pred_vis = th.cat(
                        [pred_img,
                         pred_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # B, 3, H, W

                    pred_vis_nv = th.cat([
                        pred_nv_img['image_raw'].clip(-1, 1),
                        pred_nv_depth.repeat_interleave(3, dim=1)
                    ],
                                         dim=-1)  # B, 3, H, W
                    pred_vis = th.cat([pred_vis, pred_vis_nv],
                                      dim=-2)  # cat in H dim

                    gt_vis = th.cat(
                        [gt_img, gt_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # TODO, fail to load depth. range [0, 1]

                    # vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
                    vis = th.cat([gt_vis, pred_vis], dim=-2)
                    # .permute(
                    #     0, 2, 3, 1).cpu()
                    vis_tensor = torchvision.utils.make_grid(
                        vis, nrow=vis.shape[-1] // 64)  # HWC
                    torchvision.utils.save_image(
                        vis_tensor,
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg',
                        normalize=True,
                        value_range=(-1, 1))
                    # vis = vis.numpy() * 127.5 + 127.5
                    # vis = vis.clip(0, 255).astype(np.uint8)

                    # Image.fromarray(vis).save(
                    #     f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

                    logger.log(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

    def forward_G_nvs(self, batch):  # update G

        self.mp_trainer_rec.zero_grad()
        self.rec_model.requires_grad_(True)

        self.ddp_cano_cvD.requires_grad_(False)
        self.ddp_nvs_cvD.requires_grad_(False)  # only use novel view D

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }
            target_nvs = {}

            for k, v in micro.items():
                if k[:2] == 'nv':
                    orig_key = k.replace('nv_', '')
                    target_nvs[orig_key] = v
                    # target_cano[orig_key] = micro[orig_key]

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                nvs_pred = self.rec_model(
                    img=micro['img_to_encoder'],
                    c=micro['nv_c'],
                )  # predict novel view here
                # c=th.cat([
                #     micro['c'][1:],
                #     micro['c'][:1],
                # ]))  # ! render novel views only for D loss

                # add cvD supervision

                if 'image_sr' in nvs_pred:
                    raise NotImplementedError()
                    # concat sr and raw results
                    vision_aided_loss = self.ddp_nvs_cvD(
                        # pred_nv['image_sr'],
                        # 0.5 * pred_nv['image_sr'] + 0.5 * th.nn.functional.interpolate(pred_nv['image_raw'], size=pred_nv['image_sr'].shape[2:], mode='bilinear'),
                        th.cat([
                            th.nn.functional.interpolate(
                                nvs_pred['image_raw'],
                                size=nvs_pred['image_sr'].shape[2:],
                                mode='bilinear',
                                align_corners=False,
                                antialias=True),
                            nvs_pred['image_sr'],
                        ],
                               dim=1),
                        for_G=True).mean()  # ! for debugging

                    # supervise sr only
                    # vision_aided_loss = self.ddp_nvs_cvD(
                    #     # pred_nv['image_sr'],
                    #     # 0.5 * pred_nv['image_sr'] + 0.5 * th.nn.functional.interpolate(pred_nv['image_raw'], size=pred_nv['image_sr'].shape[2:], mode='bilinear'),
                    #     th.cat([nvs_pred['image_sr'],
                    #     th.nn.functional.interpolate(nvs_pred['image_raw'], size=nvs_pred['image_sr'].shape[2:], mode='bilinear',
                    #                         align_corners=False,
                    #                         antialias=True),]),
                    #     for_G=True).mean()  # ! for debugging

                    # pred_nv['image_raw'], for_G=True).mean()  # [B, 1] shape
                else:
                    vision_aided_loss = self.ddp_nvs_cvD(
                        nvs_pred['image_raw'],
                        for_G=True).mean()  # [B, 1] shape

                # ! add nv reconstruction loss
                with self.rec_model.no_sync():  # type: ignore
                    loss, loss_dict, fg_mask = self.loss_class(
                        nvs_pred,
                        target_nvs,
                        step=self.step + self.resume_step,
                        test_mode=False,
                        return_fg_mask=True,
                        conf_sigma_l1=None,
                        conf_sigma_percl=None)
                    log_rec3d_loss_dict(loss_dict)

                loss += vision_aided_loss * self.loss_class.opt.nvs_cvD_lambda

                log_rec3d_loss_dict({
                    'vision_aided_loss/G_nvs':
                    vision_aided_loss * self.loss_class.opt.nvs_cvD_lambda,
                    **{f'{k}_nv': v for k, v in loss_dict.items()}
                })

            self.mp_trainer_rec.backward(loss)

            # ! move to other places, add tensorboard

            # if dist_util.get_rank() == 0 and self.step % 500 == 0:
            if dist_util.get_rank() == 0 and self.step % 500 == 1:
                with th.no_grad():
                    # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

                    def norm_depth(pred_depth):  # to [-1,1]
                        # pred_depth = pred['image_depth']
                        pred_depth = (pred_depth - pred_depth.min()) / (
                            pred_depth.max() - pred_depth.min())
                        return -(pred_depth * 2 - 1)

                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = norm_depth(gt_depth)

                    # if True:
                    # pred_depth = nvs_pred['image_depth']
                    # pred_depth = (pred_depth - pred_depth.min()) / (
                    #     pred_depth.max() - pred_depth.min())
                    pred_depth = norm_depth(nvs_pred['image_depth'])
                    pred_img = nvs_pred['image_raw']
                    gt_img = micro['img']

                    if 'image_sr' in nvs_pred:

                        if nvs_pred['image_sr'].shape[-1] == 512:
                            pred_img = th.cat([
                                self.pool_512(pred_img), nvs_pred['image_sr']
                            ],
                                              dim=-1)
                            gt_img = th.cat(
                                [self.pool_512(micro['img']), micro['img_sr']],
                                dim=-1)
                            pred_depth = self.pool_512(pred_depth)
                            gt_depth = self.pool_512(gt_depth)

                        elif nvs_pred['image_sr'].shape[-1] == 256:
                            pred_img = th.cat([
                                self.pool_256(pred_img), nvs_pred['image_sr']
                            ],
                                              dim=-1)
                            gt_img = th.cat(
                                [self.pool_256(micro['img']), micro['img_sr']],
                                dim=-1)
                            pred_depth = self.pool_256(pred_depth)
                            gt_depth = self.pool_256(gt_depth)

                        else:
                            pred_img = th.cat([
                                self.pool_128(pred_img), nvs_pred['image_sr']
                            ],
                                              dim=-1)
                            gt_img = th.cat(
                                [self.pool_128(micro['img']), micro['img_sr']],
                                dim=-1)
                            gt_depth = self.pool_128(gt_depth)
                            pred_depth = self.pool_128(pred_depth)

                    if gt_img.shape[-1] == 64:
                        gt_depth = self.pool_64(gt_depth)
                    elif gt_img.shape[-1] == 128:
                        gt_depth = self.pool_128(gt_depth)

                    # else:
                    #     gt_img = self.pool_64(gt_img)
                    #     gt_depth = self.pool_64(gt_depth)

                    gt_vis = th.cat(
                        [gt_img, gt_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # TODO, fail to load depth. range [0, 1]

                    pred_vis = th.cat(
                        [pred_img,
                         pred_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # B, 3, H, W

                    # vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
                    #     1, 2, 0).cpu()  # ! pred in range[-1, 1]
                    vis = th.cat([gt_vis, pred_vis], dim=-2)

                    vis = torchvision.utils.make_grid(
                        vis,
                        normalize=True,
                        scale_each=True,
                        value_range=(-1, 1)).cpu().permute(1, 2, 0)  # H W 3
                    vis = vis.numpy() * 255
                    vis = vis.clip(0, 255).astype(np.uint8)

                    # print(vis.shape)

                    Image.fromarray(vis).save(
                        f'{logger.get_dir()}/{self.step+self.resume_step}_nvs.jpg'
                    )
                    print(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}_nvs.jpg'
                    )
