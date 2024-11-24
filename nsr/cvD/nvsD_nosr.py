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


class TrainLoop3DcvD_nvsD_noSR(TrainLoop3DcvD):

    def __init__(self,
                 *,
                #  model,
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
                         SR_TRAINING=False,
                         **kwargs)

    def run_step(self, batch, step='g_step'):
        # self.forward_backward(batch)

        if step == 'g_step_rec':
            self.forward_G_rec(batch)
            took_step_g_rec = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        elif step == 'g_step_nvs':
            self.forward_G_nvs(batch)
            took_step_g_nvs = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_nvs:
                self._update_ema()  # g_ema

        elif step == 'd_step':
            self.forward_D(batch)
            _ = self.mp_trainer_cvD.optimize(self.opt_cvD)

        self._anneal_lr()
        self.log_step()

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            dist_util.synchronize()

            batch = next(self.data)
            self.run_step(batch, 'g_step_rec') # pure VAE reconstruction

            batch = next(self.data)
            self.run_step(batch, 'g_step_nvs')

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
                self.save(self.mp_trainer_cvD, 'cvD')
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
                    self.save(self.mp_trainer_cvD, 'cvD')

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.save(self.mp_trainer_cvD, 'cvD')

    def forward_D(self, batch):  # update D
        self.rec_model.requires_grad_(False)

        self.mp_trainer_cvD.zero_grad()
        self.ddp_nvs_cvD.requires_grad_(True)

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
                    micro['c'][1:], micro['c'][:1]
                ])
                latent = self.rec_model(img=micro['img_to_encoder'],
                                        behaviour='enc_dec_wo_triplane')

                cano_pred = self.rec_model(latent=latent,
                                           c=micro['c'],
                                           behaviour='triplane_dec', 
                                           return_raw_only=True)

                nvs_pred = self.rec_model(latent=latent,
                                            c=novel_view_c,
                                            behaviour='triplane_dec',
                                            return_raw_only=True)

                # if 'image_sr' in nvs_pred:
                #     d_loss_nvs = self.run_D_Diter(
                #         real=th.cat([
                #             th.nn.functional.interpolate(
                #                 cano_pred['image_raw'],
                #                 size=cano_pred['image_sr'].shape[2:],
                #                 mode='bilinear',
                #                 align_corners=False,
                #                 antialias=True),
                #             cano_pred['image_sr'],
                #         ], dim=1),
                #         fake=th.cat([
                #             th.nn.functional.interpolate(
                #                 nvs_pred['image_raw'],
                #                 size=nvs_pred['image_sr'].shape[2:],
                #                 mode='bilinear',
                #                 align_corners=False,
                #                 antialias=True),
                #             nvs_pred['image_sr'],
                #         ], dim=1),
                #         D=self.ddp_nvs_cvD)  # TODO, add SR for FFHQ

                # else:
                d_loss_nvs = self.run_D_Diter(
                    real=cano_pred['image_raw'],
                    fake=nvs_pred['image_raw'],
                    D=self.ddp_nvs_cvD)  # TODO, add SR for FFHQ

                log_rec3d_loss_dict(
                    {'vision_aided_loss/D_nvs': d_loss_nvs})
                self.mp_trainer_cvD.backward(d_loss_nvs)

    def forward_G_rec(self, batch):  # update G

        self.mp_trainer_rec.zero_grad()
        self.rec_model.requires_grad_(True)
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

                log_rec3d_loss_dict(loss_dict)

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
                    pred_depth = pred['image_depth']
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min())
                    pred_img = pred['image_raw']
                    gt_img = micro['img']

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
                    else:
                        gt_depth = self.pool_64(gt_depth)

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
        self.ddp_nvs_cvD.requires_grad_(False)  # only use novel view D

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                nvs_pred = self.rec_model(
                    img=micro['img_to_encoder'],
                    c=th.cat([
                        micro['c'][1:],
                        micro['c'][:1],
                    ]))  # ! render novel views only for D loss

                # add cvD supervision

                # if 'image_sr' in nvs_pred:
                #     # concat sr and raw results
                #     vision_aided_loss = self.ddp_nvs_cvD(
                #         nvs_pred['image_raw'],
                #         # th.cat([
                #         #     th.nn.functional.interpolate(
                #         #         size=nvs_pred['image_sr'].shape[2:],
                #         #         mode='bilinear',
                #         #         align_corners=False,
                #         #         antialias=True),
                #         #     # nvs_pred['image_sr'],
                #         # ], dim=1),
                #         for_G=True).mean()  
                # else:
                vision_aided_loss = self.ddp_nvs_cvD(
                    nvs_pred['image_raw'],
                    for_G=True).mean()  # [B, 1] shape

                loss = vision_aided_loss * self.loss_class.opt.nvs_cvD_lambda

                log_rec3d_loss_dict({
                    'vision_aided_loss/G_nvs': loss
                })

            self.mp_trainer_rec.backward(loss)

            # ! move to other places, add tensorboard

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                with th.no_grad():
                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                              gt_depth.min())
                    # if True:
                    pred_depth = nvs_pred['image_depth']
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min())
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
                    else:
                        gt_depth = self.pool_64(gt_depth)

                    gt_vis = th.cat(
                        [gt_img, gt_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # TODO, fail to load depth. range [0, 1]

                    pred_vis = th.cat(
                        [pred_img,
                         pred_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # B, 3, H, W

                    vis = th.cat([gt_vis, pred_vis], dim=-2)

                    vis = torchvision.utils.make_grid(
                        vis,
                        normalize=True,
                        scale_each=True,
                        value_range=(-1, 1)).cpu().permute(1, 2, 0)  # H W 3
                    vis = vis.numpy() * 255
                    vis = vis.clip(0, 255).astype(np.uint8)

                    Image.fromarray(vis).save(
                        f'{logger.get_dir()}/{self.step+self.resume_step}_nvs.jpg'
                    )
                    print(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}_nvs.jpg'
                    )

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
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        dist.barrier()

    def _load_and_sync_parameters(self, model=None, model_name='rec'):
        resume_checkpoint, self.resume_step = find_resume_checkpoint(
            self.resume_checkpoint, model_name) or self.resume_checkpoint

        if model is None:
            model = self.rec_model  # default model in the parent class

        print(resume_checkpoint)

        if resume_checkpoint and Path(resume_checkpoint).exists():
            if dist_util.get_rank() == 0:

                logger.log(
                    f"loading model from checkpoint: {resume_checkpoint}...")
                map_location = {
                    'cuda:%d' % 0: 'cuda:%d' % dist_util.get_rank()
                }  # configure map_location properly

                print(f'mark {model_name} loading ', flush=True)
                resume_state_dict = dist_util.load_state_dict(
                    resume_checkpoint, map_location=map_location)
                print(f'mark {model_name} loading finished', flush=True)

                model_state_dict = model.state_dict()

                for k, v in resume_state_dict.items():
                    if k in model_state_dict.keys() and v.size(
                    ) == model_state_dict[k].size():
                        model_state_dict[k] = v
                    elif 'IN' in k:
                        print('ignore ', k)
                    else:
                        print('!!!! ignore key: ', k, ": ", v.size(),)
                            #   'shape in model: ', model_state_dict[k].size())

                model.load_state_dict(model_state_dict, strict=True)
                del model_state_dict

        if dist_util.get_world_size() > 1:
            dist_util.sync_params(model.parameters())
            print(f'synced {model_name} params')
