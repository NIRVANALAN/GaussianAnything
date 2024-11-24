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

from ..train_util import TrainLoopBasic, TrainLoop3DRec
import vision_aided_loss
from dnnlib.util import calculate_adaptive_weight


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


from ..train_util_cvD import TrainLoop3DcvD
# from .nvD import


class TrainLoop3DcvD_canoD(TrainLoop3DcvD):

    def __init__(self,
                 *,
                 model,
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
        super().__init__(model=model,
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
                         use_amp=use_amp, cvD_name='cano_cvD',
                         **kwargs)

        device = dist_util.dev()

        # self.canonical_cvD = vision_aided_loss.Discriminator(
        #     cv_type='clip', loss_type='multilevel_sigmoid_s',
        #     device=device).to(device)
        # self.canonical_cvD.cv_ensemble.requires_grad_(
        #     False)  # Freeze feature extractor

        # self._load_and_sync_parameters(model=self.canonical_cvD,
        #                                model_name='cvD')

        # self.mp_trainer_canonical_cvD = MixedPrecisionTrainer(
        #     model=self.canonical_cvD,
        #     use_fp16=self.use_fp16,
        #     fp16_scale_growth=fp16_scale_growth,
        #     model_name='canonical_cvD',
        #     use_amp=use_amp)

        # self.opt_cano_cvD = AdamW(
        #     self.mp_trainer_canonical_cvD.master_params,
        #     lr=1e-5,  # same as the G
        #     betas=(0, 0.99),
        #     eps=1e-8)  # dlr in biggan cfg

        # if self.use_ddp:
        #     self.ddp_canonical_cvD = DDP(
        #         self.canonical_cvD,
        #         device_ids=[dist_util.dev()],
        #         output_device=dist_util.dev(),
        #         broadcast_buffers=False,
        #         bucket_cap_mb=128,
        #         find_unused_parameters=False,
        #     )
        # else:
        #     self.ddp_canonical_cvD = self.canonical_cvD

        th.cuda.empty_cache()

    def run_step(self, batch, step='g_step'):
        # self.forward_backward(batch)

        if step == 'g_step_rec':
            self.forward_G_rec(batch)
            took_step_g_rec = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        # elif step == 'g_step_nvs':
        #     self.forward_G_nvs(batch)
        #     took_step_g_nvs = self.mp_trainer.optimize(self.opt)

        #     if took_step_g_nvs:
        #         self._update_ema()  # g_ema

        elif step == 'd_step':
            self.forward_D(batch)
            _ = self.mp_trainer_cvD.optimize(self.opt_cvD)
            # _ = self.mp_trainer_canonical_cvD.optimize(self.opt_cano_cvD)

        else:
            return

        self._anneal_lr()
        self.log_step()

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            dist_util.synchronize()

            # batch, cond = next(self.data)
            # if batch is None:
            batch = next(self.data)
            self.run_step(batch, 'g_step_rec')

            # batch = next(self.data)
            # self.run_step(batch, 'g_step_nvs')

            batch = next(self.data)
            self.run_step(batch, 'd_step')

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                # * log to tensorboard
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            if self.step % self.eval_interval == 0 and self.step != 0:
                if dist_util.get_rank() == 0:
                    self.eval_loop()
                    # self.eval_novelview_loop()
                    # let all processes sync up before starting with a new epoch of training
                dist_util.synchronize()

            if self.step % self.save_interval == 0:
                self.save()
                self.save(self.mp_trainer_cvD, 'cano_cvD')
                # self.save(self.mp_trainer_canonical_cvD, 'cano_cvD')

                dist_util.synchronize()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                print('reached maximum iterations, exiting')

                # Save the last checkpoint if it wasn't already saved.
                if (self.step - 1) % self.save_interval != 0:

                    self.save()
                    self.save(self.mp_trainer_cvD, 'cano_cvD')
                    # self.save(self.mp_trainer_canonical_cvD, 'cano_cvD')

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            # self.save(self.mp_trainer_canonical_cvD, 'cvD')

    def forward_D(self, batch):  # update D
        # self.mp_trainer_canonical_cvD.zero_grad()
        self.mp_trainer_cvD.zero_grad()

        self.rec_model.requires_grad_(False)

        # update two D
        self.ddp_nvs_cvD.requires_grad_(True)
        # self.ddp_canonical_cvD.requires_grad_(True)

        batch_size = batch['img'].shape[0]

        # * sample a new batch for D training
        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_cvD.use_amp):

                novel_view_c = th.cat([
                    micro['c'][batch_size // 2:], micro['c'][batch_size // 2:]
                ])

                latent = self.rec_model(img=micro['img_to_encoder'],
                                        behaviour='enc_dec_wo_triplane')

                # TODO, optimize with one encoder, and two triplane decoder
                cano_pred = self.rec_model(latent=latent,
                                           c=micro['c'],
                                           behaviour='triplane_dec')

                # nvs_pred = self.rec_model(latent=latent,
                #                           c=novel_view_c,
                #                           behaviour='triplane_dec')

                # d_loss_nvs = self.run_D_Diter(
                #     real=cano_pred['image_raw'],
                #     fake=nvs_pred['image_raw'],
                #     D=self.ddp_cvD)  # TODO, add SR for FFHQ

                d_loss_cano = self.run_D_Diter(
                    real=micro['img_to_encoder'],
                    fake=cano_pred['image_raw'],
                    D=self.ddp_nvs_cvD)  # TODO, add SR for FFHQ

            # log_rec3d_loss_dict({'vision_aided_loss/D_nvs': d_loss_nvs})
            log_rec3d_loss_dict({'vision_aided_loss/D_cano': d_loss_cano})

            self.mp_trainer_cvD.backward(d_loss_cano)
            # self.mp_trainer_cvD.backward(d_loss_nvs)

    def forward_G_rec(self, batch):  # update G

        self.mp_trainer_rec.zero_grad()
        self.rec_model.requires_grad_(True)

        # self.ddp_canonical_cvD.requires_grad_(False)
        self.ddp_nvs_cvD.requires_grad_(False)

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }

            last_batch = (i + self.microbatch) >= batch_size

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                pred = self.rec_model(
                    img=micro['img_to_encoder'], c=micro['c']
                )  # render novel view for first half of the batch for D loss

                target_for_rec = micro
                pred_for_rec = pred

                if last_batch or not self.use_ddp:
                    loss, loss_dict = self.loss_class(pred_for_rec,
                                                      target_for_rec,
                                                      test_mode=False)
                else:
                    with self.rec_model.no_sync():  # type: ignore
                        loss, loss_dict = self.loss_class(pred_for_rec,
                                                          target_for_rec,
                                                          test_mode=False)

                # add cvD supervision
                vision_aided_loss = self.ddp_nvs_cvD(
                    pred_for_rec['image_raw'],
                    for_G=True).mean()  # [B, 1] shape

                last_layer = self.rec_model.module.decoder.triplane_decoder.decoder.net[  # type: ignore
                    -1].weight  # type: ignore

                d_weight = calculate_adaptive_weight(
                    loss, vision_aided_loss, last_layer,
                    # disc_weight_max=1) * 1
                    disc_weight_max=0.1) * 0.1
                loss += vision_aided_loss * d_weight

                loss_dict.update({
                    'vision_aided_loss/G_rec': vision_aided_loss,
                    'd_weight': d_weight
                })

                log_rec3d_loss_dict(loss_dict)

            self.mp_trainer_rec.backward(loss)  # no nvs cvD loss, following VQ3D

            # ! move to other places, add tensorboard

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                with th.no_grad():
                    # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                              gt_depth.min())
                    # if True:
                    pred_depth = pred['image_depth']
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min())
                    pred_img = pred['image_raw']
                    gt_img = micro['img']

                    if 'image_sr' in pred:
                        pred_img = th.cat(
                            [self.pool_512(pred_img), pred['image_sr']],
                            dim=-1)
                        gt_img = th.cat(
                            [self.pool_512(micro['img']), micro['img_sr']],
                            dim=-1)
                        pred_depth = self.pool_512(pred_depth)
                        gt_depth = self.pool_512(gt_depth)

                    gt_vis = th.cat(
                        [gt_img, gt_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # TODO, fail to load depth. range [0, 1]

                    pred_vis = th.cat(
                        [pred_img,
                         pred_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # B, 3, H, W

                    vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
                        1, 2, 0).cpu()  # ! pred in range[-1, 1]
                    # vis_grid = torchvision.utils.make_grid(vis) # HWC
                    vis = vis.numpy() * 127.5 + 127.5
                    vis = vis.clip(0, 255).astype(np.uint8)
                    Image.fromarray(vis).save(
                        f'{logger.get_dir()}/{self.step+self.resume_step}_rec.jpg'
                    )
                    print(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}_rec.jpg'
                    )

    def forward_G_nvs(self, batch):  # update G

        self.mp_trainer_rec.zero_grad()
        self.rec_model.requires_grad_(True)

        # self.ddp_canonical_cvD.requires_grad_(False)
        self.ddp_nvs_cvD.requires_grad_(False)  # only use novel view D

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_cvD.use_amp):

                pred_nv = self.rec_model(
                    img=micro['img_to_encoder'],
                    c=th.cat([
                        micro['c'][batch_size // 2:],
                        micro['c'][:batch_size // 2],
                    ]))  # ! render novel views only for D loss

                # add cvD supervision
                vision_aided_loss = self.ddp_nvs_cvD(
                    pred_nv['image_raw'], for_G=True).mean()  # [B, 1] shape

                loss = vision_aided_loss * 0.1

                log_rec3d_loss_dict({
                    'vision_aided_loss/G_nvs':
                    vision_aided_loss,
                })

            self.mp_trainer_rec.backward(loss)

            # ! move to other places, add tensorboard

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                with th.no_grad():
                    # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                              gt_depth.min())
                    # if True:
                    pred_depth = pred_nv['image_depth']
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min())
                    pred_img = pred_nv['image_raw']
                    gt_img = micro['img']

                    if 'image_sr' in pred_nv:
                        pred_img = th.cat(
                            [self.pool_512(pred_img), pred_nv['image_sr']],
                            dim=-1)
                        gt_img = th.cat(
                            [self.pool_512(micro['img']), micro['img_sr']],
                            dim=-1)
                        pred_depth = self.pool_512(pred_depth)
                        gt_depth = self.pool_512(gt_depth)

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
