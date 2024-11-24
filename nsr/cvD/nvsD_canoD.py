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

def flip_yaw(pose_matrix):
    flipped = pose_matrix.clone()
    flipped[:, 0, 1] *= -1
    flipped[:, 0, 2] *= -1
    flipped[:, 1, 0] *= -1
    flipped[:, 2, 0] *= -1
    flipped[:, 0, 3] *= -1
    # st()
    return flipped


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


from ..train_util_cvD import TrainLoop3DcvD
# from .nvD import


class TrainLoop3DcvD_nvsD_canoD(TrainLoop3DcvD):
    # class TrainLoop3DcvD_nvsD_canoD():

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

        device = dist_util.dev()

        self.cano_cvD = vision_aided_loss.Discriminator(
            cv_type='clip', loss_type='multilevel_sigmoid_s',
            device=device).to(device)
        self.cano_cvD.cv_ensemble.requires_grad_(
            False)  # Freeze feature extractor
        # self.cano_cvD.train()

        cvD_model_params = list(self.cano_cvD.parameters())
        SR_TRAINING = False
        if SR_TRAINING:  # replace the conv1 with 6 channel input
            # width, patch_size = self.nvs_cvD.cv_ensemble
            vision_width, vision_patch_size = [
                self.cano_cvD.cv_ensemble.models[0].model.conv1.weight.shape[k]
                for k in [0, -1]
            ]
            self.cano_cvD.cv_ensemble.models[0].model.conv1 = th.nn.Conv2d(
                in_channels=6,
                out_channels=vision_width,
                kernel_size=vision_patch_size,
                stride=vision_patch_size,
                bias=False).to(dist_util.dev())
            cvD_model_params += list(
                self.cano_cvD.cv_ensemble.models[0].model.conv1.parameters())

            self.cano_cvD.cv_ensemble.models[
                0].image_mean = self.cano_cvD.cv_ensemble.models[
                    0].image_mean.repeat(2)
            self.cano_cvD.cv_ensemble.models[
                0].image_std = self.cano_cvD.cv_ensemble.models[
                    0].image_std.repeat(2)

        # logger.log(f'cano_cvD_model_params: {cvD_model_params}')

        self._load_and_sync_parameters(model=self.cano_cvD,
                                       model_name='cano_cvD')

        self.mp_trainer_canonical_cvD = MixedPrecisionTrainer(
            model=self.cano_cvD,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            model_name='canonical_cvD',
            use_amp=use_amp,
            model_params=cvD_model_params)

        # cano_lr = 2e-5 * (lr / 1e-5) # D_lr=2e-4 in cvD by default
        # cano_lr = 5e-5 * (lr / 1e-5) # D_lr=2e-4 in cvD by default
        cano_lr = 2e-4 * (
            lr / 1e-5)  # D_lr=2e-4 in cvD by default. 1e-4 still overfitting
        self.opt_cano_cvD = AdamW(
            self.mp_trainer_canonical_cvD.master_params,
            lr=cano_lr,  # same as the G
            betas=(0, 0.999),
            eps=1e-8)  # dlr in biggan cfg

        logger.log(f'cpt_cano_cvD lr: {cano_lr}')

        if self.use_ddp:
            self.ddp_cano_cvD = DDP(
                self.cano_cvD,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.ddp_cano_cvD = self.cano_cvD

        th.cuda.empty_cache()

    def run_step(self, batch, step='g_step'):
        # self.forward_backward(batch)

        if step == 'g_step_rec':
            self.forward_G_rec(batch)
            took_step_g_rec = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        elif step == 'd_step_rec':
            self.forward_D(batch, behaviour='rec')
            # _ = self.mp_trainer_cvD.optimize(self.opt_cvD)
            _ = self.mp_trainer_canonical_cvD.optimize(self.opt_cano_cvD)

        elif step == 'g_step_nvs':
            self.forward_G_nvs(batch)
            took_step_g_nvs = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_nvs:
                self._update_ema()  # g_ema

        elif step == 'd_step_nvs':
            self.forward_D(batch, behaviour='nvs')
            _ = self.mp_trainer_cvD.optimize(self.opt_cvD)
            # _ = self.mp_trainer_canonical_cvD.optimize(self.opt_cano_cvD)

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

            if self.novel_view_poses is None:
                self.novel_view_poses = th.roll(batch['c'], 1, 0).to(
                    dist_util.dev())  # save for eval visualization use

            self.run_step(batch, 'g_step_rec')

            # if self.step % 2 == 0:
            batch = next(self.data)
            self.run_step(batch, 'd_step_rec')

            # if self.step % 2 == 1:
            batch = next(self.data)
            self.run_step(batch, 'g_step_nvs')

            batch = next(self.data)
            self.run_step(batch, 'd_step_nvs')

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                # * log to tensorboard
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            # if self.step % self.eval_interval == 0 and self.step != 0:
            if self.step % self.eval_interval == 0:
                if dist_util.get_rank() == 0:
                    self.eval_loop()
                    # self.eval_novelview_loop()
                    # let all processes sync up before starting with a new epoch of training
                th.cuda.empty_cache()
                dist_util.synchronize()

            if self.step % self.save_interval == 0:
                self.save()
                self.save(self.mp_trainer_cvD, self.mp_trainer_cvD.model_name)
                self.save(self.mp_trainer_canonical_cvD,
                          self.mp_trainer_canonical_cvD.model_name)

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
                    self.save(self.mp_trainer_cvD,
                              self.mp_trainer_cvD.model_name)
                    self.save(self.mp_trainer_canonical_cvD,
                              self.mp_trainer_canonical_cvD.model_name)

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.save(self.mp_trainer_canonical_cvD, 'cvD')

    def forward_D(self, batch, behaviour):  # update D
        self.mp_trainer_canonical_cvD.zero_grad()
        self.mp_trainer_cvD.zero_grad()

        self.rec_model.requires_grad_(False)
        # self.ddp_model.requires_grad_(False)

        # update two D
        if behaviour == 'nvs':
            self.ddp_nvs_cvD.requires_grad_(True)
            self.ddp_cano_cvD.requires_grad_(False)
        else:  # update rec canonical D
            self.ddp_nvs_cvD.requires_grad_(False)
            self.ddp_cano_cvD.requires_grad_(True)

        batch_size = batch['img'].shape[0]

        # * sample a new batch for D training
        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_canonical_cvD.use_amp):

                novel_view_c = th.cat([micro['c'][1:], micro['c'][:1]])

                latent = self.rec_model(img=micro['img_to_encoder'],
                                        behaviour='enc_dec_wo_triplane')

                cano_pred = self.rec_model(latent=latent,
                                           c=micro['c'],
                                           behaviour='triplane_dec')

                # TODO, optimize with one encoder, and two triplane decoder
                # FIXME: quit autocast to runbackward
                if behaviour == 'rec':

                    if 'image_sr' in cano_pred:
                        # try concat them in batch
                        # d_loss = self.run_D_Diter(
                        #     real=th.cat([
                        #         th.nn.functional.interpolate(
                        #             micro['img'],
                        #             size=micro['img_sr'].shape[2:],
                        #             mode='bilinear',
                        #             align_corners=False,
                        #             antialias=True),
                        #         micro['img_sr'],
                        #     ],
                        #                 dim=1),
                        #     fake=th.cat([
                        #         th.nn.functional.interpolate(
                        #             cano_pred['image_raw'],
                        #             size=cano_pred['image_sr'].shape[2:],
                        #             mode='bilinear',
                        #             align_corners=False,
                        #             antialias=True),
                        #         cano_pred['image_sr'],
                        #     ],
                        #                 dim=1),
                        #     D=self.ddp_cano_cvD)  # TODO, add SR for FFHQ

                        d_loss = self.run_D_Diter(
                            real= micro['img_sr'],
                            fake=cano_pred['image_sr'],
                            D=self.ddp_cano_cvD)  # TODO, add SR for FFHQ

                    else:
                        d_loss = self.run_D_Diter(
                            real=micro['img'],
                            fake=cano_pred['image_raw'],
                            D=self.ddp_cano_cvD)  # TODO, add SR for FFHQ

                    log_rec3d_loss_dict(
                        {'vision_aided_loss/D_cano': d_loss})
                    # self.mp_trainer_canonical_cvD.backward(d_loss)
                else:
                    assert behaviour == 'nvs'

                    nvs_pred = self.rec_model(latent=latent,
                                                c=novel_view_c,
                                                behaviour='triplane_dec')

                    if 'image_sr' in nvs_pred:

                        d_loss = self.run_D_Diter(
                            real=cano_pred['image_sr'],
                            # th.cat([
                                # th.nn.functional.interpolate(
                                #     cano_pred['image_raw'],
                                #     size=cano_pred['image_sr'].shape[2:],
                                #     mode='bilinear',
                                #     align_corners=False,
                                #     antialias=True),
                            # ],
                                        # dim=1),
                            fake= nvs_pred['image_sr'],
                            # th.cat([
                                # th.nn.functional.interpolate(
                                #     nvs_pred['image_raw'],
                                #     size=nvs_pred['image_sr'].shape[2:],
                                #     mode='bilinear',
                                #     align_corners=False,
                                #     antialias=True),
                            # ],
                                        # dim=1),
                            D=self.ddp_nvs_cvD)  # TODO, add SR for FFHQ

                    else:
                        d_loss = self.run_D_Diter(
                            real=cano_pred['image_raw'],
                            fake=nvs_pred['image_raw'],
                            D=self.ddp_nvs_cvD)  # TODO, add SR for FFHQ

                    log_rec3d_loss_dict(
                        {'vision_aided_loss/D_nvs': d_loss})
                    # self.mp_trainer_cvD.backward(d_loss)

            if behaviour == 'rec':
                self.mp_trainer_canonical_cvD.backward(d_loss)
            else:
                assert behaviour == 'nvs'
                self.mp_trainer_cvD.backward(d_loss)

    def forward_G_rec(self, batch):  # update G

        self.mp_trainer_rec.zero_grad()
        self.rec_model.requires_grad_(True)

        self.ddp_cano_cvD.requires_grad_(False)
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
                cano_pred = pred

                # if last_batch or not self.use_ddp:
                #     loss, loss_dict = self.loss_class(cano_pred,
                #                                       target_for_rec,
                #                                       test_mode=False,
                #                                       step=self.step +
                #                                       self.resume_step)
                # else:
                with self.rec_model.no_sync():  # type: ignore
                    loss, loss_dict, fg_mask = self.loss_class(cano_pred,
                                                        target_for_rec,
                                                        test_mode=False,
                                                        step=self.step +
                                                        self.resume_step, 
                                                        return_fg_mask=True)

                # cano_pred_img = cano_pred['image_raw']
                
                if self.loss_class.opt.symmetry_loss:
                    pose, intrinsics = micro['c'][:, :16].reshape(
                        -1, 4, 4), micro['c'][:, 16:]
                    flipped_pose = flip_yaw(pose)
                    mirror_c = th.cat(
                        [flipped_pose.reshape(-1, 16), intrinsics], -1)

                    nvs_pred = self.rec_model(latent={
                        k: v
                        for k, v in pred.items() if 'latent' in k
                    },
                                              c=mirror_c,
                                              behaviour='triplane_dec',
                                              return_raw_only=True)
                    # cano_pred_img = th.cat([cano_pred_img, nvs_pred['image_raw']], 0)

                    # concat data for supervision
                    nvs_gt = {
                        k: th.flip(target_for_rec[k], [-1])
                        for k in
                        ['img']  # fliplr leads to wrong color; B 3 H W shape
                    }
                    flipped_fg_mask = th.flip(fg_mask, [-1])
                    if 'conf_sigma' in pred:
                        conf_sigma = th.flip(pred['conf_sigma'], [-1])
                        conf_sigma = th.nn.AdaptiveAvgPool2d(fg_mask.shape[-2:])(conf_sigma) # dynamically resize to target img size
                    else:
                        conf_sigma=None
                    
                    with self.rec_model.no_sync():  # type: ignore
                        loss_symm, loss_dict_symm = self.loss_class.calc_2d_rec_loss(
                            nvs_pred['image_raw'],
                            nvs_gt['img'],
                            flipped_fg_mask,
                            # test_mode=True,
                            test_mode=False,
                            step=self.step + self.resume_step,
                            conf_sigma=conf_sigma,
                        )
                        loss += (loss_symm * 1.0) # as in unsup3d
                        # if conf_sigma is not None:
                        #     conf_loss = th.nn.functional.mse_loss(conf_sigma, flipped_fg_mask) * 0.2
                        #     loss += conf_loss # a log that regularizes all confidence to 1
                        #     loss_dict[f'conf_loss'] = conf_loss
                        for k, v in loss_dict_symm.items():
                            loss_dict[f'{k}_symm'] = v


                # add cvD supervision
                # ! TODO

                if 'image_sr' in cano_pred:
                    # concat both resolution
                    # vision_aided_loss = self.ddp_cano_cvD(
                    #     th.cat([
                    #         th.nn.functional.interpolate(
                    #             cano_pred['image_raw'],
                    #             size=cano_pred['image_sr'].shape[2:],
                    #             mode='bilinear',
                    #             align_corners=False,
                    #             antialias=True),
                    #         cano_pred['image_sr'],
                    #     ],
                    #            dim=1),  # 6 channel input
                    #     for_G=True).mean()  # [B, 1] shape

                    vision_aided_loss = self.ddp_cano_cvD(
                            cano_pred['image_sr'],
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

                    def norm_depth(pred_depth): # to [-1,1]
                        # pred_depth = pred['image_depth']
                        pred_depth = (pred_depth - pred_depth.min()) / (
                            pred_depth.max() - pred_depth.min())
                        return -(pred_depth * 2 - 1)

                    pred_img = pred['image_raw'].clip(-1,1)
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
                        pred_nv_depth = norm_depth(
                            pred_nv_img['image_depth'])
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
                            pred_nv_depth = self.pool_512(pred_nv_depth)
                            gt_depth = self.pool_512(gt_depth)

                            pred_vis_nv = th.cat(
                                [self.pool_512(pred_nv_img['image_raw']), pred_nv_img['image_sr']],
                                dim=-1)

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
                        # pred_nv_img['image_raw'].clip(-1,1),
                        pred_vis_nv,
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
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg', normalize=True, value_range=(-1,1))
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

                if 'image_sr' in nvs_pred:
                    # concat sr and raw results
                    # vision_aided_loss = self.ddp_nvs_cvD(
                    #     # pred_nv['image_sr'],
                    #     # 0.5 * pred_nv['image_sr'] + 0.5 * th.nn.functional.interpolate(pred_nv['image_raw'], size=pred_nv['image_sr'].shape[2:], mode='bilinear'),
                    #     th.cat([
                    #         th.nn.functional.interpolate(
                    #             nvs_pred['image_raw'],
                    #             size=nvs_pred['image_sr'].shape[2:],
                    #             mode='bilinear',
                    #             align_corners=False,
                    #             antialias=True),
                    #         nvs_pred['image_sr'],
                    #     ],
                    #            dim=1),
                    #     for_G=True).mean()  # ! for debugging

                    vision_aided_loss = self.ddp_nvs_cvD(
                        # pred_nv['image_sr'],
                        # 0.5 * pred_nv['image_sr'] + 0.5 * th.nn.functional.interpolate(pred_nv['image_raw'], size=pred_nv['image_sr'].shape[2:], mode='bilinear'),
                        nvs_pred['image_sr'],
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

                loss = vision_aided_loss * self.loss_class.opt.nvs_cvD_lambda

                log_rec3d_loss_dict({
                    'vision_aided_loss/G_nvs': loss
                    # vision_aided_loss * self.loss_class.opt.nvs_cvD_lambda,
                })

            self.mp_trainer_rec.backward(loss)

            # ! move to other places, add tensorboard

            # if dist_util.get_rank() == 0 and self.step % 500 == 0:
            if dist_util.get_rank() == 0 and self.step % 500 == 1:
                with th.no_grad():
                    # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

                    def norm_depth(pred_depth): # to [-1,1]
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

class TrainLoop3DcvD_nvsD_canoD_eg3d(TrainLoop3DcvD_nvsD_canoD):
    def __init__(self, *, rec_model, loss_class, data, eval_data, batch_size, microbatch, lr, ema_rate, log_interval, eval_interval, save_interval, resume_checkpoint, use_fp16=False, fp16_scale_growth=0.001, weight_decay=0, lr_anneal_steps=0, iterations=10001, load_submodule_name='', ignore_resume_opt=False, use_amp=False, **kwargs):
        super().__init__(rec_model=rec_model, loss_class=loss_class, data=data, eval_data=eval_data, batch_size=batch_size, microbatch=microbatch, lr=lr, ema_rate=ema_rate, log_interval=log_interval, eval_interval=eval_interval, save_interval=save_interval, resume_checkpoint=resume_checkpoint, use_fp16=use_fp16, fp16_scale_growth=fp16_scale_growth, weight_decay=weight_decay, lr_anneal_steps=lr_anneal_steps, iterations=iterations, load_submodule_name=load_submodule_name, ignore_resume_opt=ignore_resume_opt, use_amp=use_amp, **kwargs)
        self.rendering_kwargs = self.rec_model.module.decoder.triplane_decoder.rendering_kwargs # type: ignore
        self._prepare_nvs_pose() # for eval novelview visualization

    @th.inference_mode()
    def eval_novelview_loop(self):
        # novel view synthesis given evaluation camera trajectory
        # for i in range(0, len(c_list), 1): # TODO, larger batch size for eval
        for i, batch in enumerate(tqdm(self.eval_data)):
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            video_out = imageio.get_writer(
                f'{logger.get_dir()}/video_novelview_{self.step+self.resume_step}_batch_{i}.mp4',
                mode='I',
                fps=60,
                codec='libx264')

            for idx, c in enumerate(self.all_nvs_params):
                pred = self.rec_model(img=micro['img_to_encoder'],
                                    c=c.unsqueeze(0).repeat_interleave(micro['img'].shape[0], 0))  # pred: (B, 3, 64, 64)
                                    #   c=micro['c'])  # pred: (B, 3, 64, 64)

                # normalize depth
                # if True:
                pred_depth = pred['image_depth']
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                                pred_depth.min())
                if 'image_sr' in pred:

                    if pred['image_sr'].shape[-1] == 512:

                        pred_vis = th.cat([
                            micro['img_sr'],
                            self.pool_512(pred['image_raw']), pred['image_sr'],
                            self.pool_512(pred_depth).repeat_interleave(3, dim=1)
                        ],
                                        dim=-1)

                    elif pred['image_sr'].shape[-1] == 256:

                        pred_vis = th.cat([
                            micro['img_sr'],
                            self.pool_256(pred['image_raw']), pred['image_sr'],
                            self.pool_256(pred_depth).repeat_interleave(3, dim=1)
                        ],
                                        dim=-1)

                    else:
                        pred_vis = th.cat([
                            micro['img_sr'],
                            self.pool_128(pred['image_raw']),
                            self.pool_128(pred['image_sr']),
                            self.pool_128(pred_depth).repeat_interleave(3, dim=1)
                        ],
                                        dim=-1)

                else:

                    # st()
                    pred_vis = th.cat([
                        self.pool_128(micro['img']), 
                        self.pool_128(pred['image_raw']),
                        self.pool_128(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                    dim=-1)  # B, 3, H, W

                    # ! cooncat h dim
                    pred_vis = pred_vis.permute(0,2,3,1).flatten(0,1) # H W 3

                # vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
                # vis = pred_vis.permute(1,2,0).cpu().numpy()
                vis = pred_vis.cpu().numpy()
                vis = vis * 127.5 + 127.5
                vis = vis.clip(0, 255).astype(np.uint8)

                # for j in range(vis.shape[0]):
                    # video_out.append_data(vis[j])
                video_out.append_data(vis)

            video_out.close()

        th.cuda.empty_cache()

    
    def _prepare_nvs_pose(self):
        from nsr.camera_utils import LookAtPoseSampler, FOV_to_intrinsics

        device = dist_util.dev()
        
        fov_deg = 18.837 # for ffhq/afhq
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)

        all_nvs_params = []

        pitch_range = 0.25
        yaw_range = 0.35
        num_keyframes = 10 # how many nv poses to sample from
        w_frames = 1

        cam_pivot = th.Tensor(self.rendering_kwargs.get('avg_camera_pivot')).to(device)
        cam_radius = self.rendering_kwargs.get('avg_camera_radius')

        for frame_idx in range(num_keyframes):

            cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                    3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                    cam_pivot, radius=cam_radius, device=device)

            camera_params = th.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            all_nvs_params.append(camera_params)
        
        self.all_nvs_params = th.cat(all_nvs_params, 0)