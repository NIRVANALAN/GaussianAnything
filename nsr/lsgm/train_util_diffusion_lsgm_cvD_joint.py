import copy
import functools
import json
import os
from pathlib import Path
from pdb import set_trace as st
from typing import Any

import vision_aided_loss
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
from dnnlib.util import requires_grad
from guided_diffusion.nn import update_ema

from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion import dist_util, logger
from guided_diffusion.train_util import (calc_average_loss,
                                         log_rec3d_loss_dict,
                                         find_resume_checkpoint)
from guided_diffusion.continuous_diffusion_utils import get_mixed_prediction, different_p_q_objectives, kl_per_group_vada, kl_balancer

from .train_util_diffusion_lsgm_noD_joint import TrainLoop3DDiffusionLSGMJointnoD

from nsr.losses.builder import kl_coeff


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


class TrainLoop3DDiffusionLSGM_cvD(TrainLoop3DDiffusionLSGMJointnoD):

    def __init__(self,
                 *,
                 rec_model,
                 denoise_model,
                 diffusion,
                 sde_diffusion,
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
                 triplane_scaling_divider=1,
                 use_amp=False,
                 diffusion_input_size=224,
                 init_cvD=True,
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         denoise_model=denoise_model,
                         diffusion=diffusion,
                         sde_diffusion=sde_diffusion,
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
                         triplane_scaling_divider=triplane_scaling_divider,
                         use_amp=use_amp,
                         diffusion_input_size=diffusion_input_size,
                         **kwargs)

        #     self.setup_cvD()
        # def setup_cvD(self):
        device = dist_util.dev()

        # TODO copied from nvs_canoD, could be merged
        # * create vision aided model
        # TODO, load model api

        # nvs D
        if init_cvD:
            self.nvs_cvD = vision_aided_loss.Discriminator(
                cv_type='clip', loss_type='multilevel_sigmoid_s',
                device=device).to(device)
            self.nvs_cvD.cv_ensemble.requires_grad_(
                False)  # Freeze feature extractor
            self._load_and_sync_parameters(model=self.nvs_cvD, model_name='cvD')

            self.mp_trainer_nvs_cvD = MixedPrecisionTrainer(
                model=self.nvs_cvD,
                use_fp16=self.use_fp16,
                fp16_scale_growth=fp16_scale_growth,
                model_name='cvD',
                use_amp=use_amp,
                # use_amp=
                # False,  # assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer."
                model_params=list(self.nvs_cvD.decoder.parameters()))
            cvD_lr = 2e-4 * (lr / 1e-5) * self.loss_class.opt.nvs_D_lr_mul
            # cvD_lr = 1e-5*(lr/1e-5)
            self.opt_cvD = AdamW(self.mp_trainer_nvs_cvD.master_params,
                                lr=cvD_lr,
                                betas=(0, 0.999),
                                eps=1e-8)  # dlr in biggan cfg

            logger.log(f'cpt_cvD lr: {cvD_lr}')

            if self.use_ddp:
                self.ddp_nvs_cvD = DDP(
                    self.nvs_cvD,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
            else:
                self.ddp_nvs_cvD = self.nvs_cvD

            # cano d
            self.cano_cvD = vision_aided_loss.Discriminator(
                cv_type='clip', loss_type='multilevel_sigmoid_s',
                device=device).to(device)
            self.cano_cvD.cv_ensemble.requires_grad_(
                False)  # Freeze feature extractor
            # self.cano_cvD.train()

            self._load_and_sync_parameters(model=self.cano_cvD,
                                        model_name='cano_cvD')

            self.mp_trainer_cano_cvD = MixedPrecisionTrainer(
                model=self.cano_cvD,
                use_fp16=self.use_fp16,
                fp16_scale_growth=fp16_scale_growth,
                model_name='canonical_cvD',
                use_amp=use_amp,
                model_params=list(self.cano_cvD.decoder.parameters()))

            cano_lr = 2e-4 * (
                lr / 1e-5)  # D_lr=2e-4 in cvD by default. 1e-4 still overfitting
            self.opt_cano_cvD = AdamW(
                self.mp_trainer_cano_cvD.master_params,
                lr=cano_lr,  # same as the G
                betas=(0, 0.999),
                eps=1e-8)  # dlr in biggan cfg

            logger.log(f'cpt_cano_cvD lr: {cano_lr}')

            self.ddp_cano_cvD = DDP(
                self.cano_cvD,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )

        # Fix decoder
        requires_grad(self.rec_model.decoder, False)

    def _post_run_step(self):
        if self.step % self.log_interval == 0 and dist_util.get_rank() == 0 and self.step != 0:
            out = logger.dumpkvs()
            # * log to tensorboard
            for k, v in out.items():
                self.writer.add_scalar(f'Loss/{k}', v,
                                       self.step + self.resume_step)

        if self.step % self.eval_interval == 0 and self.step != 0:
        # if self.step % self.eval_interval == 0:
            if dist_util.get_rank() == 0:
                self.eval_ddpm_sample(self.rec_model)
                if self.sde_diffusion.args.train_vae:
                    self.eval_loop(self.rec_model)

        if self.step % self.save_interval == 0 and self.step != 0:
            self.save(self.mp_trainer, self.mp_trainer.model_name)

        self.step += 1

        if self.step > self.iterations:
            print('reached maximum iterations, exiting')

            # Save the last checkpoint if it wasn't already saved.
            if (self.step - 1) % self.save_interval != 0:
                self.save(self.mp_trainer, self.mp_trainer.model_name)
            exit()

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            # dist_util.synchronize()

            batch = next(self.data)
            self.run_step(batch, 'cano_ddpm_only')

            # batch = next(self.data)
            # self.run_step(batch, 'cano_ddpm_step')

            # batch = next(self.data)
            # self.run_step(batch, 'd_step_rec')

            # batch = next(self.data)
            # self.run_step(batch, 'nvs_ddpm_step')

            # batch = next(self.data)
            # self.run_step(batch, 'd_step_nvs')

            self._post_run_step()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            # self.save(self.mp_trainer_canonical_cvD, 'cvD')

    def run_step(self, batch, step='g_step'):
        # self.forward_backward(batch)
        if step == 'ce_ddpm_step':
            self.ce_ddpm_step(batch)

        elif step in ['ce', 'ddpm', 'cano_ddpm_only']:
            self.joint_rec_ddpm(batch, step)

        elif step == 'cano_ddpm_step':
            self.joint_rec_ddpm(batch, 'cano')

        elif step == 'd_step_rec':
            self.forward_D(batch, behaviour='rec')

        elif step == 'nvs_ddpm_step':
            self.joint_rec_ddpm(batch, 'nvs')

        elif step == 'd_step_nvs':
            self.forward_D(batch, behaviour='nvs')

        self._anneal_lr()
        self.log_step()

    def flip_encoder_grad(self, mode=True):
        requires_grad(self.rec_model.encoder, mode)

    def forward_D(self, batch, behaviour):  # update D

        self.flip_encoder_grad(False)
        self.rec_model.eval()
        # self.ddp_model.requires_grad_(False)

        # update two D
        if behaviour == 'nvs':
            self.mp_trainer_nvs_cvD.zero_grad()
            self.ddp_nvs_cvD.requires_grad_(True)
            self.ddp_nvs_cvD.train()
            self.ddp_cano_cvD.requires_grad_(False)
            self.ddp_cano_cvD.eval()
        else:  # update rec canonical D
            self.mp_trainer_cano_cvD.zero_grad()
            self.ddp_nvs_cvD.requires_grad_(False)
            self.ddp_nvs_cvD.eval()
            self.ddp_cano_cvD.requires_grad_(True)
            self.ddp_cano_cvD.train()

        batch_size = batch['img'].shape[0]

        # * sample a new batch for D training
        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_cano_cvD.use_amp):

                latent = self.ddp_rec_model(img=micro['img_to_encoder'],
                                            behaviour='enc_dec_wo_triplane')

                cano_pred = self.ddp_rec_model(latent=latent,
                                               c=micro['c'],
                                               behaviour='triplane_dec')

                # TODO, optimize with one encoder, and two triplane decoder
                # FIXME quit autocast to runbackward
                if behaviour == 'rec':
                    if 'image_sr' in cano_pred:
                        # d_loss_cano = self.run_D_Diter(
                        #     # real=micro['img_sr'],
                        #     # fake=cano_pred['image_sr'],
                        #     real=0.5 * micro['img_sr'] + 0.5 * th.nn.functional.interpolate(micro['img'], size=micro['img_sr'].shape[2:], mode='bilinear'),
                        #     fake=0.5 * cano_pred['image_sr'] + 0.5 * th.nn.functional.interpolate(cano_pred['image_raw'], size=cano_pred['image_sr'].shape[2:], mode='bilinear'),
                        #     D=self.ddp_canonical_cvD)  # ! failed, color bias

                        # try concat them in batch
                        d_loss = self.run_D_Diter(
                            real=th.cat([
                                th.nn.functional.interpolate(
                                    micro['img'],
                                    size=micro['img_sr'].shape[2:],
                                    mode='bilinear',
                                    align_corners=False,
                                    antialias=True),
                                micro['img_sr'],
                            ],
                                        dim=1),
                            fake=th.cat([
                                th.nn.functional.interpolate(
                                    cano_pred['image_raw'],
                                    size=cano_pred['image_sr'].shape[2:],
                                    mode='bilinear',
                                    align_corners=False,
                                    antialias=True),
                                cano_pred['image_sr'],
                            ],
                                        dim=1),
                            D=self.ddp_cano_cvD)  # TODO, add SR for FFHQ
                    else:
                        d_loss = self.run_D_Diter(real=micro['img'],
                                                  fake=cano_pred['image_raw'],
                                                  D=self.ddp_cano_cvD)

                    log_rec3d_loss_dict({'vision_aided_loss/D_cano': d_loss})
                    # self.mp_trainer_canonical_cvD.backward(d_loss_cano)
                else:
                    assert behaviour == 'nvs'
                    novel_view_c = th.roll(micro['c'], 1, 0)

                    nvs_pred = self.ddp_rec_model(latent=latent,
                                                  c=novel_view_c,
                                                  behaviour='triplane_dec')

                    if 'image_sr' in nvs_pred:

                        d_loss = self.run_D_Diter(
                            real=th.cat([
                                th.nn.functional.interpolate(
                                    cano_pred['image_raw'],
                                    size=cano_pred['image_sr'].shape[2:],
                                    mode='bilinear',
                                    align_corners=False,
                                    antialias=True),
                                cano_pred['image_sr'],
                            ],
                                        dim=1),
                            fake=th.cat([
                                th.nn.functional.interpolate(
                                    nvs_pred['image_raw'],
                                    size=nvs_pred['image_sr'].shape[2:],
                                    mode='bilinear',
                                    align_corners=False,
                                    antialias=True),
                                nvs_pred['image_sr'],
                            ],
                                        dim=1),
                            D=self.ddp_nvs_cvD)  # TODO, add SR for FFHQ

                    else:
                        d_loss = self.run_D_Diter(
                            real=cano_pred['image_raw'],
                            fake=nvs_pred['image_raw'],
                            D=self.ddp_nvs_cvD)  # TODO, add SR for FFHQ

                    log_rec3d_loss_dict({'vision_aided_loss/D_nvs': d_loss})
                    # self.mp_trainer_cvD.backward(d_loss_nvs)
            # quit autocast to run backward()
            if behaviour == 'rec':
                self.mp_trainer_cano_cvD.backward(d_loss)
                # assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer."
                _ = self.mp_trainer_cano_cvD.optimize(self.opt_cano_cvD)
            else:
                assert behaviour == 'nvs'
                self.mp_trainer_nvs_cvD.backward(d_loss)
                _ = self.mp_trainer_nvs_cvD.optimize(self.opt_cvD)

        self.flip_encoder_grad(True)
        self.rec_model.train()

    # def forward_ddpm(self, eps):
    #     args = self.sde_diffusion.args

    #     # sample noise
    #     noise = th.randn(size=eps.size(), device=eps.device
    #                      )  # note that this noise value is currently shared!

    #     # get diffusion quantities for p (sgm prior) sampling scheme and reweighting for q (vae)
    #     t_p, var_t_p, m_t_p, obj_weight_t_p, obj_weight_t_q, g2_t_p = \
    #         self.sde_diffusion.iw_quantities(args.iw_sample_p)
    #     eps_t_p = self.sde_diffusion.sample_q(eps, noise, var_t_p, m_t_p)
    #     # logsnr_p = self.sde_diffusion.log_snr(m_t_p,
    #     #                                         var_t_p)  # for p only

    #     pred_eps_p, pred_x0_p, logsnr_p = self.ddpm_step(
    #         eps_t_p, t_p, m_t_p, var_t_p)

    #     # ! batchify for mixing_component
    #     # mixing normal trick
    #     mixing_component = self.sde_diffusion.mixing_component(
    #         eps_t_p, var_t_p, t_p, enabled=True)  # TODO, which should I use?
    #     pred_eps_p = get_mixed_prediction(
    #         True, pred_eps_p,
    #         self.ddp_ddpm_model(x=None,
    #                             timesteps=None,
    #                             get_attr='mixing_logit'), mixing_component)

    #     # ! eps loss equivalent to snr weighting of x0 loss, see "progressive distillation"
    #     with self.ddp_ddpm_model.no_sync():  # type: ignore
    #         l2_term_p = th.square(pred_eps_p - noise)  # ? weights

    #     p_eps_objective = th.mean(obj_weight_t_p * l2_term_p)

    #     log_rec3d_loss_dict(
    #         dict(mixing_logit=self.ddp_ddpm_model(
    #             x=None, timesteps=None, get_attr='mixing_logit').detach(), ))

    #     return {
    #         'pred_eps_p': pred_eps_p,
    #         'eps_t_p': eps_t_p,
    #         'p_eps_objective': p_eps_objective,
    #         'pred_x0_p': pred_x0_p,
    #         'logsnr_p': logsnr_p
    #     }

    # ddpm + rec loss
    def joint_rec_ddpm(self, batch, behaviour='cano', *args, **kwargs):
        """
        add sds grad to all ae predicted x_0 
        """
        args = self.sde_diffusion.args

        # ! enable the gradient of both models
        # requires_grad(self.rec_model, True)
        self.flip_encoder_grad(True)
        self.rec_model.train()

        requires_grad(self.ddpm_model, True)
        self.ddpm_model.train()

        requires_grad(self.ddp_cano_cvD, False)
        requires_grad(self.ddp_nvs_cvD, False)
        self.ddp_cano_cvD.eval()
        self.ddp_nvs_cvD.eval()

        self.mp_trainer.zero_grad()

        # if args.train_vae:
        #     for param in self.rec_model.decoder.triplane_decoder.parameters(  # type: ignore
        #     ):  # type: ignore
        #         param.requires_grad_(
        #             False
        #         )  # ! disable triplane_decoder grad in each iteration indepenently;

        assert args.train_vae

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v
                for k, v in batch.items()
            }

            # =================================== ae part ===================================
            with th.cuda.amp.autocast(dtype=th.float16,
                                      enabled=self.mp_trainer.use_amp):
                #   and args.train_vae):

                loss = th.tensor(0.).to(dist_util.dev())
                vision_aided_loss = th.tensor(0.).to(dist_util.dev())

                vae_out = self.ddp_rec_model(
                    img=micro['img_to_encoder'],
                    c=micro['c'],
                    behaviour='encoder_vae',
                )  # pred: (B, 3, 64, 64)
                eps = vae_out[self.latent_name]

                if 'bg_plane' in vae_out:
                    eps = th.cat((eps, vae_out['bg_plane']), dim=1) # include background, B 12+4 32 32

                # eps = pred[self.latent_name]
                # eps = vae_out.pop(self.latent_name)

                # ! running diffusion forward
                p_sample_batch = self.prepare_ddpm(eps)
                # ddpm_ret = self.forward_ddpm(eps)
                ddpm_ret = self.apply_model(p_sample_batch)
                # p_loss = ddpm_ret['p_eps_objective']
                loss += ddpm_ret['p_eps_objective'].mean()

                # =====================================================================
                # ! reconstruction loss + gan loss
                if behaviour != 'cano_ddpm_only':
                    if behaviour == 'cano':
                        cano_pred = self.ddp_rec_model(
                            latent=vae_out,
                            c=micro['c'],
                            behaviour=self.render_latent_behaviour)

                        with self.ddp_model.no_sync():  # type: ignore
                            q_vae_recon_loss, loss_dict = self.loss_class(
                                cano_pred, micro, test_mode=False)
                        loss += q_vae_recon_loss

                        # add gan loss
                        vision_aided_loss = self.ddp_cano_cvD(
                            cano_pred['image_raw'], for_G=True
                        ).mean(
                        ) * self.loss_class.opt.rec_cvD_lambda  # [B, 1] shape

                        loss_dict.update({
                            'vision_aided_loss/G_rec':
                            vision_aided_loss.detach(),
                        })
                        log_rec3d_loss_dict(loss_dict)

                        if dist_util.get_rank() == 0 and self.step % 500 == 0:
                            self.cano_ddpm_log(cano_pred, micro, ddpm_ret)

                    else:
                        assert behaviour == 'nvs'

                        nvs_pred = self.ddp_rec_model(
                            img=micro['img_to_encoder'],
                            c=th.roll(micro['c'], 1, 0),
                        )  # ! render novel views only for D loss

                        vision_aided_loss = self.ddp_nvs_cvD(
                            nvs_pred['image_raw'], for_G=True
                        ).mean(
                        ) * self.loss_class.opt.nvs_cvD_lambda  # [B, 1] shape

                        log_rec3d_loss_dict(
                            {'vision_aided_loss/G_nvs': vision_aided_loss})

                        if dist_util.get_rank() == 0 and self.step % 500 == 1:
                            self.nvs_log(nvs_pred, micro)

                else:
                    cano_pred = self.ddp_rec_model(
                        latent=vae_out,
                        c=micro['c'],
                        behaviour=self.render_latent_behaviour)

                    with self.ddp_model.no_sync():  # type: ignore
                        q_vae_recon_loss, loss_dict = self.loss_class(
                            {
                                **vae_out,  # include latent here.
                                **cano_pred,
                            },
                            micro,
                            test_mode=False)
                        # pred,
                        # micro,
                        # test_mode=False)
                    log_rec3d_loss_dict(loss_dict)
                    loss += q_vae_recon_loss

                loss += vision_aided_loss

            self.mp_trainer.backward(loss)

        # quit for loop
        _ = self.mp_trainer.optimize(self.opt, clip_grad=self.loss_class.opt.grad_clip)

    @th.inference_mode()
    def cano_ddpm_log(self, cano_pred, micro, ddpm_ret):
        assert isinstance(cano_pred, dict)
        behaviour = 'cano'

        gt_depth = micro['depth']
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                  gt_depth.min())

        if 'image_depth' in cano_pred:
            pred_depth = cano_pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())
        else:
            pred_depth = th.zeros_like(gt_depth)

        pred_img = cano_pred['image_raw']
        gt_img = micro['img']

        if 'image_sr' in cano_pred:
            if cano_pred['image_sr'].shape[-1] == 512:
                pred_img = th.cat(
                    [self.pool_512(pred_img), cano_pred['image_sr']], dim=-1)
                gt_img = th.cat([self.pool_512(micro['img']), micro['img_sr']],
                                dim=-1)
                pred_depth = self.pool_512(pred_depth)
                gt_depth = self.pool_512(gt_depth)

            elif cano_pred['image_sr'].shape[-1] == 256:
                pred_img = th.cat(
                    [self.pool_256(pred_img), cano_pred['image_sr']], dim=-1)
                gt_img = th.cat([self.pool_256(micro['img']), micro['img_sr']],
                                dim=-1)
                pred_depth = self.pool_256(pred_depth)
                gt_depth = self.pool_256(gt_depth)

            else:
                pred_img = th.cat(
                    [self.pool_128(pred_img), cano_pred['image_sr']], dim=-1)
                gt_img = th.cat([self.pool_128(micro['img']), micro['img_sr']],
                                dim=-1)
                gt_depth = self.pool_128(gt_depth)
                pred_depth = self.pool_128(pred_depth)
        else:
            gt_img = self.pool_64(gt_img)
            gt_depth = self.pool_64(gt_depth)

        gt_vis = th.cat([
            gt_img, micro['img'], micro['img'],
            gt_depth.repeat_interleave(3, dim=1)
        ],
                        dim=-1)[0:1]  # TODO, fail to load depth. range [0, 1]

        # eps_t_p_3D = eps_t_p.reshape(batch_size, eps_t_p.shape[1]//3, 3, -1) # B C 3 L
        eps_t_p, pred_eps_p, logsnr_p = (ddpm_ret[k]
                                         for k in ('eps_t_p', 'pred_eps_p',
                                                   'logsnr_p'))

        if 'bg_plane' in cano_pred:
            noised_latent = {
                'latent_normalized_2Ddiffusion': eps_t_p[0:1, :12] * self.triplane_scaling_divider,  
                'bg_plane': eps_t_p[0:1, 12:16] * self.triplane_scaling_divider,  
            }
        else:
            noised_latent = {
                'latent_normalized_2Ddiffusion': eps_t_p[0:1] * self.triplane_scaling_divider,
            }

        # st() # split bg_plane here
        noised_ae_pred = self.ddp_rec_model(
            img=None,
            c=micro['c'][0:1],
            latent=noised_latent,
            behaviour=self.render_latent_behaviour)

        pred_x0 = self.sde_diffusion._predict_x0_from_eps(
            eps_t_p, pred_eps_p, logsnr_p)  # for VAE loss, denosied latent

        if 'bg_plane' in cano_pred:
            denoised_latent = {
                'latent_normalized_2Ddiffusion': pred_x0[0:1, :12] * self.triplane_scaling_divider,  
                'bg_plane': pred_x0[0:1, 12:16] * self.triplane_scaling_divider,  
            }
        else:
            denoised_latent = {
                'latent_normalized_2Ddiffusion': pred_x0[0:1] * self.triplane_scaling_divider,
            }

        # pred_xstart_3D
        denoised_ae_pred = self.ddp_rec_model(
            img=None,
            c=micro['c'][0:1],
            latent=denoised_latent,
            behaviour=self.render_latent_behaviour)

        pred_vis = th.cat([
            pred_img[0:1], noised_ae_pred['image_raw'][0:1],
            denoised_ae_pred['image_raw'][0:1],
            pred_depth[0:1].repeat_interleave(3, dim=1)
        ],
                          dim=-1)  # B, 3, H, W

        vis = th.cat([gt_vis, pred_vis],
                     dim=-2)[0].permute(1, 2,
                                        0).cpu()  # ! pred in range[-1, 1]

        # vis_grid = torchvision.utils.make_grid(vis) # HWC
        vis = vis.numpy() * 127.5 + 127.5
        vis = vis.clip(0, 255).astype(np.uint8)
        Image.fromarray(vis).save(
            # f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t[0].item()}_{behaviour}.jpg'
            f'{logger.get_dir()}/{self.step+self.resume_step}_{behaviour}.jpg')
        print(
            'log denoised vis to: ',
            f'{logger.get_dir()}/{self.step+self.resume_step}_{behaviour}.jpg')
        del vis, pred_vis, pred_x0, pred_eps_p, micro

        th.cuda.empty_cache()

    @th.inference_mode()
    def nvs_log(self, nvs_pred, micro):
        behaviour = 'nvs'

        if dist_util.get_rank() == 0 and self.step % 500 == 1:
            # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

            gt_depth = micro['depth']
            if gt_depth.ndim == 3:
                gt_depth = gt_depth.unsqueeze(1)
            gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                      gt_depth.min())
            # if True:
            pred_depth = nvs_pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())
            pred_img = nvs_pred['image_raw']
            gt_img = micro['img']

            if 'image_sr' in nvs_pred:

                if nvs_pred['image_sr'].shape[-1] == 512:
                    pred_img = th.cat(
                        [self.pool_512(pred_img), nvs_pred['image_sr']],
                        dim=-1)
                    gt_img = th.cat(
                        [self.pool_512(micro['img']), micro['img_sr']], dim=-1)
                    pred_depth = self.pool_512(pred_depth)
                    gt_depth = self.pool_512(gt_depth)

                elif nvs_pred['image_sr'].shape[-1] == 256:
                    pred_img = th.cat(
                        [self.pool_256(pred_img), nvs_pred['image_sr']],
                        dim=-1)
                    gt_img = th.cat(
                        [self.pool_256(micro['img']), micro['img_sr']], dim=-1)
                    pred_depth = self.pool_256(pred_depth)
                    gt_depth = self.pool_256(gt_depth)

                else:
                    pred_img = th.cat(
                        [self.pool_128(pred_img), nvs_pred['image_sr']],
                        dim=-1)
                    gt_img = th.cat(
                        [self.pool_128(micro['img']), micro['img_sr']], dim=-1)
                    gt_depth = self.pool_128(gt_depth)
                    pred_depth = self.pool_128(pred_depth)

            else:
                gt_img = self.pool_64(gt_img)
                gt_depth = self.pool_64(gt_depth)

            gt_vis = th.cat(
                [gt_img, gt_depth.repeat_interleave(3, dim=1)],
                dim=-1)  # TODO, fail to load depth. range [0, 1]

            pred_vis = th.cat(
                [pred_img, pred_depth.repeat_interleave(3, dim=1)],
                dim=-1)  # B, 3, H, W

            # vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
            #     1, 2, 0).cpu()  # ! pred in range[-1, 1]
            vis = th.cat([gt_vis, pred_vis], dim=-2)

            vis = torchvision.utils.make_grid(
                vis, normalize=True, scale_each=True,
                value_range=(-1, 1)).cpu().permute(1, 2, 0)  # H W 3
            vis = vis.numpy() * 255
            vis = vis.clip(0, 255).astype(np.uint8)

            Image.fromarray(vis).save(
                f'{logger.get_dir()}/{self.step+self.resume_step}_nvs.jpg')
            print('log vis to: ',
                  f'{logger.get_dir()}/{self.step+self.resume_step}_nvs.jpg')

    # ! all copied from train_util_cvD.py; should merge later.
    def run_D_Diter(self, real, fake, D=None):
        # Dmain: Minimize logits for generated images and maximize logits for real images.
        if D is None:
            D = self.ddp_nvs_cvD

        lossD = D(real, for_real=True).mean() + D(fake, for_real=False).mean()
        return lossD

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

        if model_name == 'ddpm':
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint(rate, params)

        dist.barrier()

    def _load_and_sync_parameters(self, model=None, model_name='rec'):
        resume_checkpoint, self.resume_step = find_resume_checkpoint(
            self.resume_checkpoint, model_name) or self.resume_checkpoint

        if model is None:
            model = self.ddp_rec_model  # default model in the parent class

        logger.log(resume_checkpoint)

        if resume_checkpoint and Path(resume_checkpoint).exists():
            if dist_util.get_rank() == 0:

                logger.log(
                    f"loading model from checkpoint: {resume_checkpoint}...")
                map_location = {
                    'cuda:%d' % 0: 'cuda:%d' % dist_util.get_rank()
                }  # configure map_location properly

                logger.log(f'mark {model_name} loading ', )
                resume_state_dict = dist_util.load_state_dict(
                    resume_checkpoint, map_location=map_location)
                logger.log(f'mark {model_name} loading finished', )

                model_state_dict = model.state_dict()

                for k, v in resume_state_dict.items():

                    if k in model_state_dict.keys() and v.size(
                    ) == model_state_dict[k].size():
                        model_state_dict[k] = v

                    # elif 'IN' in k and model_name == 'rec' and getattr(model.decoder, 'decomposed_IN', False):
                    #     model_state_dict[k.replace('IN', 'superresolution.norm.norm_layer')] = v # decomposed IN
                    elif 'attn.wk' in k or 'attn.wv' in k:  # old qkv
                        logger.log('ignore ', k)

                    elif 'decoder.vit_decoder.blocks' in k:
                        # st()
                        # load from 2D ViT pre-trained into 3D ViT blocks.
                        assert len(model.decoder.vit_decoder.blocks[0].vit_blks
                                   ) == 2  # assert depth=2 here.
                        fusion_ca_depth = len(
                            model.decoder.vit_decoder.blocks[0].vit_blks)
                        vit_subblk_index = int(k.split('.')[3])
                        vit_blk_keyname = ('.').join(k.split('.')[4:])
                        fusion_blk_index = vit_subblk_index // fusion_ca_depth
                        fusion_blk_subindex = vit_subblk_index % fusion_ca_depth
                        model_state_dict[
                            f'decoder.vit_decoder.blocks.{fusion_blk_index}.vit_blks.{fusion_blk_subindex}.{vit_blk_keyname}'] = v
                        # logger.log('load 2D ViT weight: {}'.format(f'decoder.vit_decoder.blocks.{fusion_blk_index}.vit_blks.{fusion_blk_subindex}.{vit_blk_keyname}'))

                    elif 'IN' in k:
                        logger.log('ignore ', k)

                    elif 'quant_conv' in k:
                        logger.log('ignore ', k)

                    else:
                        logger.log(
                            '!!!! ignore key: ',
                            k,
                            ": ",
                            v.size(),
                        )
                        if k in model_state_dict:
                            logger.log('shape in model: ',
                                       model_state_dict[k].size())
                        else:
                            logger.log(k, 'not in model_state_dict')

                model.load_state_dict(model_state_dict, strict=True)
                del model_state_dict

        if dist_util.get_world_size() > 1:
            dist_util.sync_params(model.parameters())
            logger.log(f'synced {model_name} params')


class TrainLoop3DDiffusionLSGM_cvD_scaling(TrainLoop3DDiffusionLSGM_cvD):

    def __init__(self,
                 *,
                 rec_model,
                 denoise_model,
                 diffusion,
                 sde_diffusion,
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
                 triplane_scaling_divider=1,
                 use_amp=False,
                 diffusion_input_size=224,
                 init_cvD=True,
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         denoise_model=denoise_model,
                         diffusion=diffusion,
                         sde_diffusion=sde_diffusion,
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
                         triplane_scaling_divider=triplane_scaling_divider,
                         use_amp=use_amp,
                         diffusion_input_size=diffusion_input_size,
                         init_cvD=init_cvD,
                         **kwargs)

    def _update_latent_stat_ema(self, latent: th.Tensor):
        # update the miu/var of ema_latent
        for rate, params in zip(self.ema_rate,
                                [self.ddpm_model.ema_latent_mean]):
            update_ema(params, latent.mean(0, keepdim=True), rate=rate)
        for rate, params in zip(self.ema_rate,
                                [self.ddpm_model.ema_latent_std]):
            update_ema(params, latent.std([1,2,3]).mean(0, keepdim=True), rate=rate)

        log_rec3d_loss_dict({'ema_latent_std': self.ddpm_model.ema_latent_std.mean()})
        log_rec3d_loss_dict({'ema_latent_mean': self.ddpm_model.ema_latent_mean.mean()})

    # def _init_optim_groups(self, rec_model, freeze_decoder=True):
    #     # unfreeze decoder when scaling is enabled
    #     return super()._init_optim_groups(rec_model, freeze_decoder=False)

    def _standarize(self, eps):
        # scaled_eps = (eps - self.ddpm_model.ema_latent_mean
        #         ) / self.ddpm_model.ema_latent_std
        # scaled_eps = eps - self.ddpm_model.ema_latent_mean
        # scaled_eps = eps.div(self.ddpm_model.ema_latent_std)
        # scaled_eps = eps + self.ddpm_model.ema_latent_std
        scaled_eps = eps.add(-self.ddpm_model.ema_latent_mean).mul(1/self.ddpm_model.ema_latent_std)
        return scaled_eps

    def _unstandarize(self, scaled_eps):
        return scaled_eps.mul(self.ddpm_model.ema_latent_std).add(self.ddpm_model.ema_latent_mean)


class TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm(TrainLoop3DDiffusionLSGM_cvD_scaling):
    def __init__(self, *, rec_model, denoise_model, diffusion, sde_diffusion, loss_class, data, eval_data, batch_size, microbatch, lr, ema_rate, log_interval, eval_interval, save_interval, resume_checkpoint, use_fp16=False, fp16_scale_growth=0.001, weight_decay=0, lr_anneal_steps=0, iterations=10001, triplane_scaling_divider=1, use_amp=False, diffusion_input_size=224,init_cvD=False, **kwargs):
        super().__init__(rec_model=rec_model, denoise_model=denoise_model, diffusion=diffusion, sde_diffusion=sde_diffusion, loss_class=loss_class, data=data, eval_data=eval_data, batch_size=batch_size, microbatch=microbatch, lr=lr, ema_rate=ema_rate, log_interval=log_interval, eval_interval=eval_interval, save_interval=save_interval, resume_checkpoint=resume_checkpoint, use_fp16=use_fp16, fp16_scale_growth=fp16_scale_growth, weight_decay=weight_decay, lr_anneal_steps=lr_anneal_steps, iterations=iterations, triplane_scaling_divider=triplane_scaling_divider, use_amp=use_amp, diffusion_input_size=diffusion_input_size,
                         init_cvD=init_cvD, **kwargs)

    def _setup_opt(self):
        # TODO, two optims groups.
        self.opt = AdamW([{
            'name': 'ddpm',
            'params': self.ddpm_model.parameters(),
        }],
                         lr=self.lr,
                         weight_decay=self.weight_decay)

        for rec_param_group in self._init_optim_groups(self.rec_model, True): # freeze D
            self.opt.add_param_group(rec_param_group)
        logger.log(self.opt)


    def next_n_batch(self, n=1):
        '''sample n batch at the same time.
        '''
        all_batch_list = [next(self.data) for _ in range(n)]
        return {
            k: th.cat([batch[k] for batch in all_batch_list], 0)
            for k in all_batch_list[0].keys()
        }
        # pass

    def subset_batch(self, batch=None, micro_batchsize=4, big_endian=False):
        '''sample a batch subset
        '''
        if batch is None:
            batch = next(self.data)
        if big_endian:
            return {
                k: v[-micro_batchsize:]
                for k, v in batch.items()
            }
        else:
            return {
                k: v[:micro_batchsize]
                for k, v in batch.items()
            }
        # pass


    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            # dist_util.synchronize()

            # batch = self.next_n_batch(n=4)
            batch = self.next_n_batch(n=6) # effective BS=72
            self.run_step(batch, 'ddpm') # ddpm fixed

            batch = next(self.data)
            self.run_step(batch, 'ce')

            # batch = next(self.data)
            # self.run_step(batch, 'cano_ddpm_step')

            # batch = next(self.data)
            # self.run_step(batch, 'd_step_rec')

            # batch = next(self.data)
            # self.run_step(batch, 'nvs_ddpm_step')

            # batch = next(self.data)
            # self.run_step(batch, 'd_step_nvs')

            self._post_run_step()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            # self.save(self.mp_trainer_canonical_cvD, 'cvD')

    # def _init_optim_groups(self, rec_model, freeze_decoder=True):
    #     # unfreeze decoder when scaling is enabled
    #     # return super()._init_optim_groups(rec_model, freeze_decoder=False)
    #     return super()._init_optim_groups(rec_model, freeze_decoder=True)

    def entropy_weight(self, normal_entropy=None):
        return self.loss_class.opt.negative_entropy_lambda 

    # ddpm + rec loss
    def joint_rec_ddpm(self, batch, behaviour='ddpm', *args, **kwargs):
        """
        add sds grad to all ae predicted x_0 
        """
        args = self.sde_diffusion.args

        # ! enable the gradient of both models
        # requires_grad(self.rec_model, True)
        # if behaviour == 'ce': # ll sampling? later. train encoder.
        if 'ce' in behaviour: # ll sampling? later. train encoder.
            ##############################################
            ###### Update the VAE encoder/decoder ########
            ##############################################
            requires_grad(self.ddpm_model, False)
            self.ddpm_model.eval()
            ce_flag = True

            if behaviour == 'ce_E': # unfreeze E and freeze D
                requires_grad(self.rec_model.encoder, True)
                self.rec_model.encoder.train()
                requires_grad(self.rec_model.decoder, False)
                self.rec_model.decoder.eval()

            else: # train all
                requires_grad(self.rec_model, True)
                self.rec_model.train()

        else: # train ddpm.
            ce_flag = False
            # self.flip_encoder_grad(False)
            requires_grad(self.rec_model, False)
            self.rec_model.eval()
            requires_grad(self.ddpm_model, True)
            self.ddpm_model.train()

        self.mp_trainer.zero_grad()

        # assert args.train_vae

        batch_size = batch['img'].shape[0]

        # for i in range(0, batch_size, self.microbatch):
        for i in range(0, batch_size, batch_size):

            micro = {
                k:
                v[i:i + batch_size].to(dist_util.dev()) if isinstance(
                # v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v
                for k, v in batch.items()
            }

            # =================================== ae part ===================================
            with th.cuda.amp.autocast(dtype=th.float16,
                                    #   enabled=self.mp_trainer.use_amp):
                                      enabled=False):
                #   and args.train_vae):

                loss = th.tensor(0.).to(dist_util.dev())

                # with th.cuda.amp.autocast(dtype=th.float16,
                #                         enabled=False):
                # quit amp in encoder, avoid nan.
                vae_out = self.ddp_rec_model(
                    img=micro['img_to_encoder'],
                    c=micro['c'],
                    behaviour='encoder_vae',
                )  # pred: (B, 3, 64, 64)
                eps = vae_out[self.latent_name]

                # ! prepare for diffusion
                if 'bg_plane' in vae_out:
                    eps = th.cat((eps, vae_out['bg_plane']), dim=1) # include background, B 12+4 32 32

                if ce_flag:
                    p_sample_batch = self.prepare_ddpm(eps, 'q')
                else: # sgm prior
                    eps.requires_grad_(True)
                    p_sample_batch = self.prepare_ddpm(eps, 'p')

                # ! running diffusion forward
                ddpm_ret = self.apply_model(p_sample_batch)
                # p_loss = ddpm_ret['p_eps_objective']
                p_loss = ddpm_ret['p_eps_objective'].mean()
                if ce_flag:
                    cross_entropy = p_loss  # why collapse?
                    normal_entropy = vae_out['posterior'].normal_entropy()
                    negative_entropy = -normal_entropy * self.entropy_weight(normal_entropy)
                    ce_loss = (cross_entropy + negative_entropy.mean())

                    if self.diffusion_ce_anneal: # gradually add ce lambda 
                        raise NotImplementedError()
                        diffusion_ce_lambda = kl_coeff(
                            step=self.step + self.resume_step,
                            constant_step=5e3,
                            total_step=20e3,  
                            min_kl_coeff=1e-2,
                            max_kl_coeff=self.loss_class.opt.negative_entropy_lambda)
                        ce_loss *= diffusion_ce_lambda

                        log_rec3d_loss_dict({
                            'diffusion_ce_lambda': diffusion_ce_lambda,
                        })

                    loss += ce_loss
                else:
                    loss += p_loss  # p loss

                if ce_flag and 'D' in behaviour: # ce only on E
                    # =====================================================================
                    # ! reconstruction loss + gan loss

                    with th.cuda.amp.autocast(dtype=th.float16,
                                            enabled=False):

                        # 24GB memory use till now.
                        cano_pred = self.ddp_rec_model( 
                            latent=vae_out,
                            c=micro['c'],
                            behaviour=self.render_latent_behaviour)

                        with self.ddp_model.no_sync():  # type: ignore
                            q_vae_recon_loss, loss_dict = self.loss_class(
                                {
                                    **vae_out,  # include latent here.
                                    **cano_pred,
                                },
                                micro,
                                test_mode=False)
                        
                        log_rec3d_loss_dict({
                            **loss_dict,
                            'negative_entropy': negative_entropy.mean(),
                        })
                        loss += q_vae_recon_loss

                        # save image log
                        if dist_util.get_rank() == 0 and self.step % 500 == 0:
                            self.cano_ddpm_log(cano_pred, micro, ddpm_ret)

            self.mp_trainer.backward(loss) # grad accumulation

        # quit micro
        _ = self.mp_trainer.optimize(self.opt, clip_grad=self.loss_class.opt.grad_clip)

class TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD(TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm):
    def __init__(self, *, rec_model, denoise_model, diffusion, sde_diffusion, loss_class, data, eval_data, batch_size, microbatch, lr, ema_rate, log_interval, eval_interval, save_interval, resume_checkpoint, use_fp16=False, fp16_scale_growth=0.001, weight_decay=0, lr_anneal_steps=0, iterations=10001, triplane_scaling_divider=1, use_amp=False, diffusion_input_size=224, init_cvD=False, **kwargs):
        super().__init__(rec_model=rec_model, denoise_model=denoise_model, diffusion=diffusion, sde_diffusion=sde_diffusion, loss_class=loss_class, data=data, eval_data=eval_data, batch_size=batch_size, microbatch=microbatch, lr=lr, ema_rate=ema_rate, log_interval=log_interval, eval_interval=eval_interval, save_interval=save_interval, resume_checkpoint=resume_checkpoint, use_fp16=use_fp16, fp16_scale_growth=fp16_scale_growth, weight_decay=weight_decay, lr_anneal_steps=lr_anneal_steps, iterations=iterations, triplane_scaling_divider=triplane_scaling_divider, use_amp=use_amp, diffusion_input_size=diffusion_input_size, init_cvD=init_cvD, **kwargs)

    def _setup_opt(self):
        # TODO, two optims groups.
        self.opt = AdamW([{
            'name': 'ddpm',
            'params': self.ddpm_model.parameters(),
        }],
                         lr=self.lr,
                         weight_decay=self.weight_decay)

        for rec_param_group in self._init_optim_groups(self.rec_model, freeze_decoder=False):
            self.opt.add_param_group(rec_param_group)
        logger.log(self.opt)

class TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_weightingv0(TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD):
    '''
    1. weight CE with ema(var(eps)), since ce decreases, sigma decreases.
    2. clip entorpy (log sigma) with 0; avoid it form increasing too much
    3. add eps scaling back with ema_rate=0.9999, make sure the std=1.
    4. add grad clipping by default
    '''
    def __init__(self, *, rec_model, denoise_model, diffusion, sde_diffusion, loss_class, data, eval_data, batch_size, microbatch, lr, ema_rate, log_interval, eval_interval, save_interval, resume_checkpoint, use_fp16=False, fp16_scale_growth=0.001, weight_decay=0, lr_anneal_steps=0, iterations=10001, triplane_scaling_divider=1, use_amp=False, diffusion_input_size=224, init_cvD=False, **kwargs):
        super().__init__(rec_model=rec_model, denoise_model=denoise_model, diffusion=diffusion, sde_diffusion=sde_diffusion, loss_class=loss_class, data=data, eval_data=eval_data, batch_size=batch_size, microbatch=microbatch, lr=lr, ema_rate=ema_rate, log_interval=log_interval, eval_interval=eval_interval, save_interval=save_interval, resume_checkpoint=resume_checkpoint, use_fp16=use_fp16, fp16_scale_growth=fp16_scale_growth, weight_decay=weight_decay, lr_anneal_steps=lr_anneal_steps, iterations=iterations, triplane_scaling_divider=triplane_scaling_divider, use_amp=use_amp, diffusion_input_size=diffusion_input_size, init_cvD=init_cvD, **kwargs)
        
        # for dynamic entropy penalize
        self.entropy_const = 0.5 * (np.log(2 * np.pi) + 1)
        # self._load_and_sync_parameters
    

    # def _load_model(self):
    #     # TODO, for currently compatability
    #     self._load_and_sync_parameters(model=self.model) # load to joint class
    
    # def save(self):
    #     return super().save()

    def prepare_ddpm(self, eps, mode='p'):

        log_rec3d_loss_dict(
            {
                f'unscaled_eps_mean': eps.mean(),
                f'unscaled_eps_std': eps.std([1,2,3]).mean(0),
            }
        )

        scaled_eps = self._standarize(eps)
        p_sample_batch = super().prepare_ddpm(scaled_eps, mode)

        # update ema; this will not affect the diffusion computation of this batch.
        self._update_latent_stat_ema(eps)

        return p_sample_batch

    def ce_weight(self):
        return self.loss_class.opt.ce_lambda * (self.ddpm_model.ema_latent_std.mean().detach())

    # def ce_weight(self):
    #     return self.loss_class.opt.ce_lambda

    def entropy_weight(self, normal_entropy=None):
        '''if log(sigma) > 0; stop penalty.
        '''
        # basically L1
        negative_entroy_lambda = self.loss_class.opt.negative_entropy_lambda
        # return th.where(normal_entropy>self.entropy_const, -negative_entroy_lambda, negative_entroy_lambda) # if log(sigma) > 0, weight = 0.
        # return negative_entroy_lambda * (1/self.ddpm_model.ema_latent_std.mean().detach()**2) # if log(sigma) > 0, weight = 0.
        return negative_entroy_lambda * (1/self.ddpm_model.ema_latent_std.mean().detach()) # if log(sigma) > 0, weight = 0.

class TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_iterativeED(TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_weightingv0):
    def __init__(self, *, rec_model, denoise_model, diffusion, sde_diffusion, loss_class, data, eval_data, batch_size, microbatch, lr, ema_rate, log_interval, eval_interval, save_interval, resume_checkpoint, use_fp16=False, fp16_scale_growth=0.001, weight_decay=0, lr_anneal_steps=0, iterations=10001, triplane_scaling_divider=1, use_amp=False, diffusion_input_size=224, init_cvD=False, diffusion_ce_anneal=False, **kwargs):
        super().__init__(rec_model=rec_model, denoise_model=denoise_model, diffusion=diffusion, sde_diffusion=sde_diffusion, loss_class=loss_class, data=data, eval_data=eval_data, batch_size=batch_size, microbatch=microbatch, lr=lr, ema_rate=ema_rate, log_interval=log_interval, eval_interval=eval_interval, save_interval=save_interval, resume_checkpoint=resume_checkpoint, use_fp16=use_fp16, fp16_scale_growth=fp16_scale_growth, weight_decay=weight_decay, lr_anneal_steps=lr_anneal_steps, iterations=iterations, triplane_scaling_divider=triplane_scaling_divider, use_amp=use_amp, diffusion_input_size=diffusion_input_size, init_cvD=init_cvD, **kwargs)
        self.diffusion_ce_anneal = diffusion_ce_anneal

    def run_step(self, batch, step='g_step'):

        assert step in ['ce', 'ddpm', 'cano_ddpm_only', 'ce_ED', 'ce_E', 'ce_D', 'D', 'ED']
        self.joint_rec_ddpm(batch, step)

        self._anneal_lr()
        self.log_step()


    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            batch = self.next_n_batch(n=12) # effective BS=48
            self.run_step(batch, 'ddpm') # ddpm fixed AE

            batch = self.next_n_batch(n=3) # effective BS=12
            self.run_step(batch, 'ce_ED')

            self._post_run_step()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()


    @th.inference_mode()
    def log_diffusion_images(self, vae_out, p_sample_batch, micro, ddpm_ret):

        eps_t_p, t_p, logsnr_p = (p_sample_batch[k] for k in (
            'eps_t_p',
            't_p',
            'logsnr_p',
        ))
        pred_eps_p = ddpm_ret['pred_eps_p']

        vae_out.pop('posterior')  # for calculating kl loss
        vae_out_for_pred = {
            k: v[0:1].to(dist_util.dev()) if isinstance(v, th.Tensor) else v
            for k, v in vae_out.items()
        }

        pred = self.ddp_rec_model(latent=vae_out_for_pred,
                                  c=micro['c'][0:1],
                                  behaviour=self.render_latent_behaviour)
        assert isinstance(pred, dict)

        pred_img = pred['image_raw']
        gt_img = micro['img']

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
                # self.pool_128(hint),
                gt_img,
                gt_depth.repeat_interleave(3, dim=1)
            ],
            dim=-1)[0:1]  # TODO, fail to load depth. range [0, 1]

        # eps_t_p_3D = eps_t_p.reshape(batch_size, eps_t_p.shape[1]//3, 3, -1) # B C 3 L

        if 'bg_plane' in vae_out:
            noised_latent = {
                'latent_normalized_2Ddiffusion':
                eps_t_p[0:1, :12] * self.triplane_scaling_divider,
                'bg_plane':
                eps_t_p[0:1, 12:16] * self.triplane_scaling_divider,
            }
        else:
            noised_latent = {
                'latent_normalized_2Ddiffusion':
                eps_t_p[0:1] * self.triplane_scaling_divider,
            }

        noised_ae_pred = self.ddp_rec_model(
            img=None,
            c=micro['c'][0:1],
            latent=noised_latent,
            # latent=eps_t_p[0:1] * self.
            # triplane_scaling_divider,  # TODO, how to define the scale automatically
            behaviour=self.render_latent_behaviour)

        pred_x0 = self.sde_diffusion._predict_x0_from_eps(
            eps_t_p, pred_eps_p, logsnr_p)  # for VAE loss, denosied latent

        if 'bg_plane' in vae_out:
            denoised_latent = {
                'latent_normalized_2Ddiffusion':
                pred_x0[0:1, :12] * self.triplane_scaling_divider,
                'bg_plane':
                pred_x0[0:1, 12:16] * self.triplane_scaling_divider,
            }
        else:
            denoised_latent = {
                'latent_normalized_2Ddiffusion':
                pred_x0[0:1] * self.triplane_scaling_divider,
            }

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

        vis = th.cat([gt_vis, pred_vis],
                     dim=-2)[0].permute(1, 2,
                                        0).cpu()  # ! pred in range[-1, 1]

        # vis_grid = torchvision.utils.make_grid(vis) # HWC
        vis = vis.numpy() * 127.5 + 127.5
        vis = vis.clip(0, 255).astype(np.uint8)
        Image.fromarray(vis).save(
            f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t_p[0].item():3}.jpg'
        )
        print(
            'log denoised vis to: ',
            f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t_p[0].item():3}.jpg'
        )

        th.cuda.empty_cache()


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
            pred_nv_depth = norm_depth(pred_cano['image_depth'])
        else:
            pred_depth = th.zeros_like(gt_depth)
            pred_nv_depth = th.zeros_like(gt_depth)

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



class TrainLoop3D_LDM(TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_iterativeED):
    def __init__(self, *, rec_model, denoise_model, diffusion, sde_diffusion, loss_class, data, eval_data, batch_size, microbatch, lr, ema_rate, log_interval, eval_interval, save_interval, resume_checkpoint, use_fp16=False, fp16_scale_growth=0.001, weight_decay=0, lr_anneal_steps=0, iterations=10001, triplane_scaling_divider=1, use_amp=False, diffusion_input_size=224, init_cvD=False, diffusion_ce_anneal=False, **kwargs):
        super().__init__(rec_model=rec_model, denoise_model=denoise_model, diffusion=diffusion, sde_diffusion=sde_diffusion, loss_class=loss_class, data=data, eval_data=eval_data, batch_size=batch_size, microbatch=microbatch, lr=lr, ema_rate=ema_rate, log_interval=log_interval, eval_interval=eval_interval, save_interval=save_interval, resume_checkpoint=resume_checkpoint, use_fp16=use_fp16, fp16_scale_growth=fp16_scale_growth, weight_decay=weight_decay, lr_anneal_steps=lr_anneal_steps, iterations=iterations, triplane_scaling_divider=triplane_scaling_divider, use_amp=use_amp, diffusion_input_size=diffusion_input_size, init_cvD=init_cvD, diffusion_ce_anneal=diffusion_ce_anneal, **kwargs)

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            batch = self.next_n_batch(n=2) # effective BS=64, micro=4, 30.7gib
            self.run_step(batch, 'ddpm') # ddpm fixed AE

            # batch = self.next_n_batch(n=1) # 
            # self.run_step(batch, 'ce_ED')

            self._post_run_step()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

class TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_iterativeED_nv(TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_iterativeED):
    # reconstruction function from train_nv_util.py
    def __init__(self, *, rec_model, denoise_model, diffusion, sde_diffusion, loss_class, data, eval_data, batch_size, microbatch, lr, ema_rate, log_interval, eval_interval, save_interval, resume_checkpoint, use_fp16=False, fp16_scale_growth=0.001, weight_decay=0, lr_anneal_steps=0, iterations=10001, triplane_scaling_divider=1, use_amp=False, diffusion_input_size=224, init_cvD=False, diffusion_ce_anneal=False, **kwargs):
        super().__init__(rec_model=rec_model, denoise_model=denoise_model, diffusion=diffusion, sde_diffusion=sde_diffusion, loss_class=loss_class, data=data, eval_data=eval_data, batch_size=batch_size, microbatch=microbatch, lr=lr, ema_rate=ema_rate, log_interval=log_interval, eval_interval=eval_interval, save_interval=save_interval, resume_checkpoint=resume_checkpoint, use_fp16=use_fp16, fp16_scale_growth=fp16_scale_growth, weight_decay=weight_decay, lr_anneal_steps=lr_anneal_steps, iterations=iterations, triplane_scaling_divider=triplane_scaling_divider, use_amp=use_amp, diffusion_input_size=diffusion_input_size, init_cvD=init_cvD, diffusion_ce_anneal=diffusion_ce_anneal, **kwargs)

        # ! for rendering
        self.eg3d_model = self.rec_model.decoder.triplane_decoder  # type: ignore
        self.renderdiff_loss = False # whether to render denoised latent for reconstruction loss

        # self.inner_loop_k = 2
        # self.ce_d_loop_k = 6

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            batch = self.next_n_batch(n=2) # effective BS=2*8
            self.run_step(batch, 'ddpm')

            # if self.step % self.inner_loop_k == 1: # train E per 2 steps
            batch = next(self.data) # sample a new batch for rec training
            # self.run_step(self.subset_batch(batch, micro_batchsize=6, big_endian=False), 'ce_ED') # freeze D, train E with diffusion prior
            # self.run_step(batch, 'ce_ED') # 
            self.run_step(batch, 'ce_E') # 

                # if self.step % self.ce_d_loop_k == 1: # train D per 4 steps
                #     batch = next(self.data) # sample a new batch for rec training
                #     self.run_step(self.subset_batch(batch, micro_batchsize=4, big_endian=True), 'ED') # freeze E, train D

            self._post_run_step()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    # ddpm + rec loss
    def joint_rec_ddpm(self, batch, behaviour='ddpm', *args, **kwargs):
        """
        add sds grad to all ae predicted x_0 
        """
        args = self.sde_diffusion.args

        # ! enable the gradient of both models
        # requires_grad(self.rec_model, True)
        # if behaviour == 'ce': # ll sampling? later. train encoder.
        ce_flag = False
        diffusion_flag = True
        if 'ce' in behaviour: # ll sampling? later. train encoder.
            ##############################################
            ###### Update the VAE encoder/decoder ########
            ##############################################
            requires_grad(self.ddpm_model, False)
            self.ddpm_model.eval()
            ce_flag = True

            if behaviour == 'ce_E': # unfreeze E and freeze D
                requires_grad(self.rec_model.encoder, True)
                self.rec_model.encoder.train()
                requires_grad(self.rec_model.decoder, False)
                self.rec_model.decoder.eval()

            elif behaviour == 'ce_D': # unfreeze E and freeze D
                requires_grad(self.rec_model.encoder, False)
                self.rec_model.encoder.eval()
                requires_grad(self.rec_model.decoder, True)
                self.rec_model.decoder.train()

            else: # train all, may oom
                requires_grad(self.rec_model, True)
                self.rec_model.train()

        elif behaviour == 'ED': # just train E and D
            diffusion_flag = False
            requires_grad(self.ddpm_model, False)
            self.ddpm_model.eval()
            requires_grad(self.rec_model, True)
            self.rec_model.train()

        elif behaviour == 'D':
            diffusion_flag = False
            requires_grad(self.rec_model.encoder, False)
            self.rec_model.encoder.eval()
            requires_grad(self.rec_model.decoder, True)
            self.rec_model.decoder.train()

        else: # train ddpm.
            # self.flip_encoder_grad(False)
            requires_grad(self.rec_model, False)
            self.rec_model.eval()
            requires_grad(self.ddpm_model, True)
            self.ddpm_model.train()

        self.mp_trainer.zero_grad()

        assert args.train_vae

        batch_size = batch['img'].shape[0]

        # for i in range(0, batch_size, self.microbatch):
        for i in range(0, batch_size, batch_size):

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
                k: th.empty_like(v)
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

            # ! cano view loss
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
                for key in ('img', 'depth_mask', 'depth'):  # type: ignore
                    # target[key][i:i+1] = torchvision.transforms.functional.crop(
                    cano_cropped_target[key][
                        j:j + 1] = torchvision.transforms.functional.crop(
                            micro[key][j:j + 1], top, left, height, width)

            # =================================== ae part ===================================
            with th.cuda.amp.autocast(dtype=th.float16,
                                    #   enabled=self.mp_trainer.use_amp):
                                      enabled=False):
                #   and args.train_vae):

                loss = th.tensor(0.).to(dist_util.dev())

                # with th.cuda.amp.autocast(dtype=th.float16,
                #                         enabled=False):
                # quit amp in encoder, avoid nan.
                vae_out = self.ddp_rec_model(
                    img=micro['img_to_encoder'],
                    c=micro['c'],
                    behaviour='encoder_vae',
                )  # pred: (B, 3, 64, 64)


                if diffusion_flag:

                    eps = vae_out[self.latent_name] # 12542mib, bs=4

                    # '''
                    # ! prepare for diffusion
                    if 'bg_plane' in vae_out:
                        eps = th.cat((eps, vae_out['bg_plane']), dim=1) # include background, B 12+4 32 32

                    if ce_flag:
                        p_sample_batch = self.prepare_ddpm(eps, 'q')
                    else:
                        eps.requires_grad_(True)
                        p_sample_batch = self.prepare_ddpm(eps, 'p')

                    # ! running diffusion forward
                    ddpm_ret = self.apply_model(p_sample_batch)
                    # p_loss = ddpm_ret['p_eps_objective']
                    p_loss = ddpm_ret['p_eps_objective'].mean()
                    # st() # 12890mib

                    if ce_flag:
                        cross_entropy = p_loss  # why collapse?
                        normal_entropy = vae_out['posterior'].normal_entropy()
                        entropy_weight = self.entropy_weight(normal_entropy)
                        negative_entropy = -normal_entropy * entropy_weight
                        ce_loss = (cross_entropy + negative_entropy.mean())

                        # if self.diffusion_ce_anneal: # gradually add ce lambda 
                        #     diffusion_ce_lambda = kl_coeff(
                        #         step=self.step + self.resume_step,
                        #         constant_step=5e3+self.resume_step,
                        #         total_step=25e3,  
                        #         min_kl_coeff=1e-5,
                        #         max_kl_coeff=self.loss_class.opt.negative_entropy_lambda)
                        #     # diffusion_ce_lambda = th.tensor(1e-5).to(dist_util.dev())
                        #     ce_loss *= diffusion_ce_lambda

                        log_rec3d_loss_dict({
                            # 'diffusion_ce_lambda': diffusion_ce_lambda,
                            'negative_entropy': negative_entropy.mean(),
                            'entropy_weight': entropy_weight,
                            'ce_loss': ce_loss
                        })

                        loss += ce_loss
                    else:
                        loss += p_loss  # p loss

                
                # ! do reconstruction supervision 

                # '''

                if ce_flag or not diffusion_flag:  # vae part
                    latent_to_decode = vae_out
                else:
                    latent_to_decode =  { # diffusion part
                        self.latent_name: ddpm_ret['pred_x0_p']
                    } # render denoised latent

                # with th.cuda.amp.autocast(dtype=th.float16,
                #                         enabled=False):
                # st()
                if ce_flag or self.renderdiff_loss or not diffusion_flag:
                    # ! do vae latent -> triplane decode
                    latent_to_decode.update(self.ddp_rec_model(latent=latent_to_decode, behaviour='decode_after_vae_no_render'))  # triplane, 19mib bs=4
                    
                    # ! do render
                    # st()
                    pred_nv_cano = self.ddp_rec_model( # 24gb, bs=4
                        # latent=latent.expand(2,),
                        latent={
                            'latent_after_vit': # ! triplane for rendering
                            latent_to_decode['latent_after_vit'].repeat(2, 1, 1, 1)
                        },
                        c=th.cat([micro['nv_c'],
                                micro['c']]),  # predict novel view here
                        behaviour='triplane_dec',
                        # ray_origins=target['ray_origins'],
                        # ray_directions=target['ray_directions'],
                        ray_origins=th.cat(
                            [target['ray_origins'], cano_target['ray_origins']],
                            0),
                        ray_directions=th.cat([
                            target['ray_directions'], cano_target['ray_directions']
                        ]),
                    )

                    pred_nv_cano.update({ # for kld
                        'posterior': vae_out['posterior'],
                        'latent_normalized_2Ddiffusion': vae_out['latent_normalized_2Ddiffusion']
                    })
                    
                    # ! 2D loss

                    with self.ddp_model.no_sync():  # type: ignore
                        loss_rec, loss_rec_dict, _ = self.loss_class(
                            pred_nv_cano,
                            {
                                k: th.cat([v, cano_cropped_target[k]], 0)
                                for k, v in cropped_target.items()
                            },  # prepare merged data
                            step=self.step + self.resume_step,
                            test_mode=False,
                            return_fg_mask=True,
                            conf_sigma_l1=None,
                            conf_sigma_percl=None)

                        if diffusion_flag and not ce_flag:
                            prefix = 'denoised_'
                        else:
                            prefix = ''

                        log_rec3d_loss_dict({
                            f'{prefix}{k}': v for k, v in loss_rec_dict.items()
                        })

                    loss += loss_rec # l2, LPIPS, Alpha loss

                    # save image log
                    # if dist_util.get_rank() == 0 and self.step % 500 == 0:
                    #     self.cano_ddpm_log(cano_pred, micro, ddpm_ret)

            self.mp_trainer.backward(loss) # grad accumulation, 27gib

            # st()

            # for name, p in self.model.named_parameters():
            #     if p.grad is None:
            #         logger.log(f"found rec unused param: {name}")

        # _ = self.mp_trainer.optimize(self.opt, clip_grad=self.loss_class.opt.grad_clip)
        _ = self.mp_trainer.optimize(self.opt, clip_grad=True)

        if dist_util.get_rank() == 0:
            if  self.step % 500 == 0: # log diffusion
                self.log_diffusion_images(vae_out, p_sample_batch, micro, ddpm_ret)
            elif self.step % 500 == 1 and ce_flag: # log reconstruction
                # st()
                micro_bs = micro['img_to_encoder'].shape[0]
                self.log_patch_img(
                    cropped_target,
                    {
                        k: pred_nv_cano[k][:micro_bs]
                        for k in ['image_raw', 'image_depth', 'image_mask']
                    },
                    {
                        k: pred_nv_cano[k][micro_bs:]
                        for k in ['image_raw', 'image_depth', 'image_mask']
                    },
                )

    def _init_optim_groups(self, rec_model, freeze_decoder=False):
        # unfreeze decoder when scaling is enabled
        return super()._init_optim_groups(rec_model, freeze_decoder=True)

# class TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_iterativeED_nv_noCE(TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_iterativeED_nv):
#     """no sepatate CE schedule, use single schedule for joint ddpm/nv-rec training with entropy regularization
#     """
#     def __init__(self, *, rec_model, denoise_model, diffusion, sde_diffusion, loss_class, data, eval_data, batch_size, microbatch, lr, ema_rate, log_interval, eval_interval, save_interval, resume_checkpoint, use_fp16=False, fp16_scale_growth=0.001, weight_decay=0, lr_anneal_steps=0, iterations=10001, triplane_scaling_divider=1, use_amp=False, diffusion_input_size=224, init_cvD=False, diffusion_ce_anneal=False, **kwargs):
#         super().__init__(rec_model=rec_model, denoise_model=denoise_model, diffusion=diffusion, sde_diffusion=sde_diffusion, loss_class=loss_class, data=data, eval_data=eval_data, batch_size=batch_size, microbatch=microbatch, lr=lr, ema_rate=ema_rate, log_interval=log_interval, eval_interval=eval_interval, save_interval=save_interval, resume_checkpoint=resume_checkpoint, use_fp16=use_fp16, fp16_scale_growth=fp16_scale_growth, weight_decay=weight_decay, lr_anneal_steps=lr_anneal_steps, iterations=iterations, triplane_scaling_divider=triplane_scaling_divider, use_amp=use_amp, diffusion_input_size=diffusion_input_size, init_cvD=init_cvD, diffusion_ce_anneal=diffusion_ce_anneal, **kwargs)

#     def run_loop(self):
#         while (not self.lr_anneal_steps
#                or self.step + self.resume_step < self.lr_anneal_steps):

#             batch = self.next_n_batch(n=2) # effective BS=2*8
#             self.run_step(batch, 'ddpm')

#             # if self.step % self.inner_loop_k == 1: # train E per 2 steps
#             batch = next(self.data) # sample a new batch for rec training
#             self.run_step(self.subset_batch(batch, micro_batchsize=6, big_endian=False), 'ce_ED') # freeze D, train E with diffusion prior

#             self._post_run_step()

#         # Save the last checkpoint if it wasn't already saved.
#         if (self.step - 1) % self.save_interval != 0:
#             self.save()