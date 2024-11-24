"""
Modified from:
https://github.com/NVlabs/LSGM/blob/main/training_obj_joint.py
"""
import copy
import functools
import json
import os
from pathlib import Path
from pdb import set_trace as st
from typing import Any

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

import dnnlib
from dnnlib.util import requires_grad
from dnnlib.util import calculate_adaptive_weight

from ..train_util_diffusion import TrainLoop3DDiffusion
from ..cvD.nvsD_canoD import TrainLoop3DcvD_nvsD_canoD

from guided_diffusion.continuous_diffusion_utils import get_mixed_prediction, different_p_q_objectives, kl_per_group_vada, kl_balancer
# import utils as lsgm_utils


class TrainLoop3DDiffusionLSGM_noD(TrainLoop3DDiffusion):
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
                 **kwargs):
        super().__init__(
            rec_model=rec_model,
            denoise_model=denoise_model,
            diffusion=diffusion,
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
            schedule_sampler=schedule_sampler,
            weight_decay=weight_decay,
            lr_anneal_steps=lr_anneal_steps,
            iterations=iterations,
            ignore_resume_opt=ignore_resume_opt,
            #  freeze_ae=freeze_ae,
            freeze_ae=not sde_diffusion.args.train_vae,
            denoised_ae=denoised_ae,
            triplane_scaling_divider=triplane_scaling_divider,
            use_amp=use_amp,
            diffusion_input_size=diffusion_input_size,
            **kwargs)

        assert sde_diffusion is not None
        sde_diffusion.args.batch_size = batch_size
        self.sde_diffusion = sde_diffusion
        self.latent_name = 'latent_normalized_2Ddiffusion'  # normalized triplane latent
        self.render_latent_behaviour = 'decode_after_vae'  # directly render using triplane operations

        self.pool_512 = th.nn.AdaptiveAvgPool2d((512, 512))
        self.pool_256 = th.nn.AdaptiveAvgPool2d((256, 256))
        self.pool_128 = th.nn.AdaptiveAvgPool2d((128, 128))
        self.pool_64 = th.nn.AdaptiveAvgPool2d((64, 64))

        self.ddp_ddpm_model = self.ddp_model

        # if sde_diffusion.args.joint_train:
        # assert sde_diffusion.args.train_vae

    def run_step(self, batch, step='diffusion_step_rec'):

        # if step == 'diffusion_step_rec':

        self.forward_diffusion(batch, behaviour='diffusion_step_rec')

        # if took_step_ddpm:
        self._update_ema()

        self._anneal_lr()
        self.log_step()

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            dist_util.synchronize()

            batch = next(self.data)
            self.run_step(batch, step='diffusion_step_rec')

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
                    self.eval_ddpm_sample()
                    if self.sde_diffusion.args.train_vae:
                        self.eval_loop()

                th.cuda.empty_cache()
                dist_util.synchronize()

            if self.step % self.save_interval == 0:
                self.save(self.mp_trainer, self.mp_trainer.model_name)
                if self.sde_diffusion.args.train_vae:
                    self.save(self.mp_trainer_rec,
                              self.mp_trainer_rec.model_name)

                # dist_util.synchronize()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                print('reached maximum iterations, exiting')

                # Save the last checkpoint if it wasn't already saved.
                if (self.step - 1) % self.save_interval != 0:

                    self.save(self.mp_trainer, self.mp_trainer.model_name)
                    if self.sde_diffusion.args.train_vae:
                        self.save(self.mp_trainer_rec,
                                  self.mp_trainer_rec.model_name)

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            # self.save(self.mp_trainer_canonical_cvD, 'cvD')

    # ! duplicated code, needs refactor later
    def ddpm_step(self, eps, t, logsnr, model_kwargs={}):
        """helper function for ddpm predictions; returns predicted eps, x0 and logsnr
        """
        args = self.sde_diffusion.args
        pred_params = self.ddp_ddpm_model(eps, t, **model_kwargs)
        # pred_params = self.ddp_model(eps, t, **model_kwargs)
        if args.pred_type == 'eps':
            pred_eps = pred_params
            pred_x0 = self.sde_diffusion._predict_x0_from_eps(
                eps, pred_params, logsnr)  # for VAE loss, denosied latent
        elif args.pred_type == 'x0':
            # ! transform to pred_eps format for mixing_component
            pred_x0 = pred_params
            pred_eps = self.sde_diffusion._predict_eps_from_x0(
                eps, pred_params, logsnr)
        else:
            raise NotImplementedError(f'{args.pred_type} not implemented.')

        return pred_eps, pred_x0, logsnr

    # def apply_model(self, p_sample_batch, model_kwargs={}):
    #     # args = self.sde_diffusion.args
    #     noise, eps_t_p, t_p, logsnr_p, obj_weight_t_p, var_t_p = (
    #         p_sample_batch[k] for k in ('noise', 'eps_t_p', 't_p', 'logsnr_p',
    #                                     'obj_weight_t_p', 'var_t_p'))

    #     pred_eps_p, pred_x0_p, logsnr_p = self.ddpm_step(
    #         eps_t_p, t_p, logsnr_p, model_kwargs)

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

    def forward_diffusion(self, batch, behaviour='rec', *args, **kwargs):
        """
        add sds grad to all ae predicted x_0 
        """
        args = self.sde_diffusion.args

        # self.ddp_ddpm_model.requires_grad_(True)
        requires_grad(self.ddp_rec_model.module, args.train_vae)
        # self.ddp_rec_model.requires_grad_(args.train_vae)

        if args.train_vae:
            for param in self.ddp_rec_model.module.decoder.triplane_decoder.parameters(  # type: ignore
            ):  # type: ignore
                param.requires_grad_(
                    False
                )  # ! disable triplane_decoder grad in each iteration indepenently;

        self.mp_trainer_rec.zero_grad()
        self.mp_trainer.zero_grad()

        batch_size = batch['img'].shape[0]

        # # update ddpm params
        # took_step_ddpm = self.mp_trainer_ddpm.optimize(
        #     self.opt_ddpm)  # TODO, update two groups of parameters

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v
                for k, v in batch.items()
            }

            last_batch = (i + self.microbatch) >= batch_size

            q_vae_recon_loss = th.tensor(0.0).to(dist_util.dev())
            # vision_aided_loss = th.tensor(0.0).to(dist_util.dev())
            # denoise_loss = th.tensor(0.0).to(dist_util.dev())

            # =================================== ae part ===================================
            with th.cuda.amp.autocast(dtype=th.float16,
                                      enabled=self.mp_trainer.use_amp):
                #   and args.train_vae):

                assert behaviour == 'diffusion_step_rec'

                # ! train vae with CE; ddpm fixed
                requires_grad(self.ddp_model.module, False)
                # if args.train_vae:
                #     assert args.add_rendering_loss
                with th.set_grad_enabled(args.train_vae):
                    vae_out = self.ddp_rec_model(
                        img=micro['img_to_encoder'],
                        c=micro['c'],
                        # behaviour='enc_dec_wo_triplane'
                        behaviour='encoder_vae',
                    )  # pred: (B, 3, 64, 64)
                    # TODO, no need to render if not SSD; no need to do ViT decoder if only the latent is needed. update later

                # TODO, train diff and sds together, available?
                all_log_q = [vae_out['log_q_2Ddiffusion']]
                eps = vae_out[self.latent_name]
                eps.requires_grad_(True)  # single stage diffusion

                # t, weights = self.schedule_sampler.sample(
                #     eps.shape[0], dist_util.dev())

                noise = th.randn(
                    size=eps.size(), device=eps.device
                )  # note that this noise value is currently shared!
                model_kwargs = {}

                # get diffusion quantities for p (sgm prior) sampling scheme and reweighting for q (vae)
                t_p, var_t_p, m_t_p, obj_weight_t_p, obj_weight_t_q, g2_t_p = \
                    self.sde_diffusion.iw_quantities(args.iw_sample_p)
                eps_t_p = self.sde_diffusion.sample_q(eps, noise, var_t_p,
                                                      m_t_p)
                logsnr_p = self.sde_diffusion.log_snr(m_t_p,
                                                      var_t_p)  # for p only

                # in case we want to train q (vae) with another batch using a different sampling scheme for times t
                if args.iw_sample_q in ['ll_uniform', 'll_iw']:
                    t_q, var_t_q, m_t_q, obj_weight_t_q, _, g2_t_q = \
                        self.sde_diffusion.iw_quantities(args.iw_sample_q)
                    eps_t_q = self.sde_diffusion.sample_q(
                        eps, noise, var_t_q, m_t_q)

                    eps_t_p = eps_t_p.detach().requires_grad_(
                        True)  # ! p just not updated here
                    eps_t = th.cat([eps_t_p, eps_t_q], dim=0)
                    var_t = th.cat([var_t_p, var_t_q], dim=0)
                    t = th.cat([t_p, t_q], dim=0)
                    noise = th.cat([noise, noise], dim=0)
                    # logsnr = self.sde_diffusion.log_snr(m_t_q, var_t_p)
                else:
                    eps_t, m_t, var_t, t, g2_t = eps_t_p, m_t_p, var_t_p, t_p, g2_t_p

                # run the diffusion model
                eps_t.requires_grad_(True)  # 2*BS, 12, 16, 16
                pred_params = self.ddp_model(eps_t, t, **model_kwargs)

                if args.pred_type == 'eps':
                    pred_eps = pred_params
                elif args.pred_type == 'x0':
                    # ! transform to pred_eps format for mixing_component
                    pred_eps = self.sde_diffusion._predict_eps_from_x0(
                        eps_t, pred_params, logsnr_p)
                else:
                    raise NotImplementedError(
                        f'{args.pred_type} not implemented.')

                # mixing normal trick
                mixing_component = self.sde_diffusion.mixing_component(
                    eps_t, var_t, t, enabled=True)  # TODO, which should I use?
                pred_eps = get_mixed_prediction(
                    # True, pred_params,
                    True,
                    pred_eps,
                    self.ddp_model(x=None,
                                   timesteps=None,
                                   get_attr='mixing_logit'),
                    mixing_component)

                # ! eps loss equivalent to snr weighting of x0 loss, see "progressive distillation"
                if last_batch or not self.use_ddp:
                    l2_term = th.square(pred_eps - noise)
                else:
                    with self.ddp_model.no_sync():  # type: ignore
                        l2_term = th.square(pred_eps - noise)  # ? weights

                # nelbo loss with kl balancing
                # ! remainign parts of cross entropy in likelihook training
                # unpack separate objectives, in case we want to train q (vae) using a different sampling scheme for times t
                if args.iw_sample_q in ['ll_uniform',
                                        'll_iw']:  # ll_iw by default
                    l2_term_p, l2_term_q = th.chunk(l2_term, chunks=2, dim=0)
                    p_objective = th.mean(obj_weight_t_p * l2_term_p,
                                          dim=[1, 2, 3])
                    cross_entropy_per_var = obj_weight_t_q * l2_term_q
                else:
                    p_objective = th.mean(obj_weight_t_p * l2_term,
                                          dim=[1, 2, 3])
                    cross_entropy_per_var = obj_weight_t_q * l2_term

                cross_entropy_per_var += self.sde_diffusion.cross_entropy_const(
                    args.sde_time_eps)
                all_neg_log_p = [cross_entropy_per_var
                                 ]  # since only one vae group

                kl_all_list, kl_vals_per_group, kl_diag_list = kl_per_group_vada(
                    all_log_q, all_neg_log_p)  # return the mean of two terms

                # nelbo loss with kl balancing
                balanced_kl, kl_coeffs, kl_vals = kl_balancer(kl_all_list,
                                                              kl_coeff=1.0,
                                                              kl_balance=False,
                                                              alpha_i=None)

                # ! update vae for CE
                # ! single stage diffusion for rec side 1: bind vae prior and diffusion prior
                if args.train_vae:
                    # if args.add_rendering_loss:
                    # if args.joint_train:
                    with th.set_grad_enabled(args.train_vae):
                        target = micro
                        pred = self.ddp_rec_model(
                            latent=vae_out,
                            # latent={
                            #     **vae_out, self.latent_name: pred_x0,
                            #     'latent_name': self.latent_name
                            # },
                            c=micro['c'],
                            behaviour=self.render_latent_behaviour)

                        # vae reconstruction loss
                        if last_batch or not self.use_ddp:
                            q_vae_recon_loss, loss_dict = self.loss_class(
                                pred, target, test_mode=False)
                        else:
                            with self.ddp_model.no_sync():  # type: ignore
                                q_vae_recon_loss, loss_dict = self.loss_class(
                                    pred, target, test_mode=False)

                        log_rec3d_loss_dict(loss_dict)

                # ! calculate p/q loss;
                nelbo_loss = balanced_kl + q_vae_recon_loss
                q_loss = th.mean(nelbo_loss)
                p_loss = th.mean(p_objective)

                log_rec3d_loss_dict(
                    dict(
                        q_vae_recon_loss=q_vae_recon_loss,
                        p_loss=p_loss,
                        balanced_kl=balanced_kl,
                        mixing_logit=self.ddp_model(
                            x=None, timesteps=None,
                            get_attr='mixing_logit').detach(),
                    ))

                # ! single stage diffusion for rec side 2: generative feature
                if args.p_rendering_loss:
                    with th.set_grad_enabled(args.train_vae):

                        # ! transform fro pred_eps format back to pred_x0, for p only.
                        pred_x0 = self.sde_diffusion._predict_x0_from_eps(
                            eps_t_p, pred_eps[:eps_t_p.shape[0]],
                            logsnr_p)  # for VAE loss, denosied latent

                        target = micro
                        pred = self.ddp_rec_model(
                            # latent=vae_out,
                            latent={
                                **vae_out, self.latent_name: pred_x0,
                                'latent_name': self.latent_name
                            },
                            c=micro['c'],
                            behaviour=self.render_latent_behaviour)

                        # vae reconstruction loss
                        if last_batch or not self.use_ddp:
                            p_vae_recon_loss, loss_dict = self.loss_class(
                                pred, target, test_mode=False)
                        else:
                            with self.ddp_model.no_sync():  # type: ignore
                                p_vae_recon_loss, loss_dict = self.loss_class(
                                    pred, target, test_mode=False)
                        log_rec3d_loss_dict(
                            dict(p_vae_recon_loss=p_vae_recon_loss, ))

            # ! backpropagate q_loss for vae and update vae params, if trained
            if args.train_vae:
                self.mp_trainer_rec.backward(
                    q_loss,
                    retain_graph=different_p_q_objectives(
                        args.iw_sample_p, args.iw_sample_q))

            # if we use different p and q objectives or are not training the vae, discard gradients and backpropagate p_loss
            if different_p_q_objectives(
                    args.iw_sample_p, args.iw_sample_q) or not args.train_vae:
                if args.train_vae:
                    # discard current gradients computed by weighted loss for VAE
                    self.mp_trainer_rec.zero_grad()

                self.mp_trainer.backward(p_loss)

            # TODO, merge visualization with original AE
            # =================================== denoised AE log part ===================================

            if dist_util.get_rank(
            ) == 0 and self.step % 500 == 0 and behaviour != 'diff':

                with th.no_grad():

                    if not args.train_vae:
                        vae_out.pop('posterior')  # for calculating kl loss
                        vae_out_for_pred = {
                            k: v[0:1].to(dist_util.dev()) if isinstance(
                                v, th.Tensor) else v
                            for k, v in vae_out.items()
                        }

                        pred = self.ddp_rec_model(
                            latent=vae_out_for_pred,
                            c=micro['c'][0:1],
                            behaviour=self.render_latent_behaviour)
                    assert isinstance(pred, dict)
                    assert pred is not None

                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                              gt_depth.min())

                    # pred_depth = pred['image_depth']
                    # pred_depth = (pred_depth - pred_depth.min()) / (
                    #     pred_depth.max() - pred_depth.min())

                    if 'image_depth' in pred:
                        pred_depth = pred['image_depth']
                        pred_depth = (pred_depth - pred_depth.min()) / (
                            pred_depth.max() - pred_depth.min())
                    else:
                        pred_depth = th.zeros_like(gt_depth)

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
                        gt_img = self.pool_64(gt_img)
                        gt_depth = self.pool_64(gt_depth)

                    gt_vis = th.cat(
                        [
                            gt_img, micro['img'], micro['img'],
                            gt_depth.repeat_interleave(3, dim=1)
                        ],
                        dim=-1)[0:1]  # TODO, fail to load depth. range [0, 1]

                    # eps_t_p_3D = eps_t_p.reshape(batch_size, eps_t_p.shape[1]//3, 3, -1) # B C 3 L

                    noised_ae_pred = self.ddp_rec_model(
                        img=None,
                        c=micro['c'][0:1],
                        latent=eps_t_p[0:1] * self.
                        triplane_scaling_divider,  # TODO, how to define the scale automatically
                        behaviour=self.render_latent_behaviour)

                    # ! test time, use discrete diffusion model
                    params_p, _ = th.chunk(pred_eps, chunks=2,
                                           dim=0)  # get predicted noise

                    # TODO, implement for SDE difusion?
                    # ! two values isclose(rtol=1e-03, atol=1e-04)
                    # pred_xstart = self.diffusion._predict_xstart_from_eps(
                    #     x_t=eps_t_p,
                    #     t=th.tensor(t_p.detach() *
                    #                 self.diffusion.num_timesteps).long(),
                    #     eps=params_p)

                    pred_x0 = self.sde_diffusion._predict_x0_from_eps(
                        eps_t_p, params_p,
                        logsnr_p)  # for VAE loss, denosied latent

                    # pred_xstart_3D
                    denoised_ae_pred = self.ddp_rec_model(
                        img=None,
                        c=micro['c'][0:1],
                        latent=pred_x0[0:1] * self.
                        triplane_scaling_divider,  # TODO, how to define the scale automatically?
                        behaviour=self.render_latent_behaviour)

                    pred_vis = th.cat([
                        pred_img[0:1], noised_ae_pred['image_raw'][0:1],
                        denoised_ae_pred['image_raw'][0:1],
                        pred_depth[0:1].repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)  # B, 3, H, W

                    vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
                        1, 2, 0).cpu()  # ! pred in range[-1, 1]

                    # vis_grid = torchvision.utils.make_grid(vis) # HWC
                    vis = vis.numpy() * 127.5 + 127.5
                    vis = vis.clip(0, 255).astype(np.uint8)
                    Image.fromarray(vis).save(
                        f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t[0].item()}_{behaviour}.jpg'
                    )
                    print(
                        'log denoised vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t[0].item()}_{behaviour}.jpg'
                    )
                    del vis, pred_vis, pred_x0, pred_eps, micro, vae_out

                    th.cuda.empty_cache()

    # ! copied from train_util.py
    # TODO, needs to lint the class inheritance chain later.
    @th.inference_mode()
    def eval_novelview_loop(self):
        # novel view synthesis given evaluation camera trajectory
        video_out = imageio.get_writer(
            f'{logger.get_dir()}/video_novelview_{self.step+self.resume_step}.mp4',
            mode='I',
            fps=60,
            codec='libx264')

        all_loss_dict = []
        novel_view_micro = {}

        # for i in range(0, len(c_list), 1): # TODO, larger batch size for eval
        for i, batch in enumerate(tqdm(self.eval_data)):
            # for i in range(0, 8, self.microbatch):
            # c = c_list[i].to(dist_util.dev()).reshape(1, -1)
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            if i == 0:
                novel_view_micro = {
                    k: v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in batch.items()
                }
            else:
                # if novel_view_micro['c'].shape[0] < micro['img'].shape[0]:
                novel_view_micro = {
                    k: v[0:1].to(dist_util.dev()).repeat_interleave(
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
                pred_vis = th.cat([
                    self.pool_64(micro['img']), pred['image_raw'],
                    pred_depth.repeat_interleave(3, dim=1)
                ],
                                  dim=-1)  # B, 3, H, W

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            for j in range(vis.shape[0]):
                video_out.append_data(vis[j])

        video_out.close()

        val_scores_for_logging = calc_average_loss(all_loss_dict)
        with open(os.path.join(logger.get_dir(), 'scores_novelview.json'),
                  'a') as f:
            json.dump({'step': self.step, **val_scores_for_logging}, f)

        # * log to tensorboard
        for k, v in val_scores_for_logging.items():
            self.writer.add_scalar(f'Eval/NovelView/{k}', v,
                                   self.step + self.resume_step)
        del video_out, vis, pred_vis, pred, micro
        th.cuda.empty_cache()

    # @th.no_grad()
    # def eval_loop(self, c_list:list):
    @th.inference_mode()
    def eval_loop(self):
        # novel view synthesis given evaluation camera trajectory
        video_out = imageio.get_writer(
            f'{logger.get_dir()}/video_{self.step+self.resume_step}.mp4',
            mode='I',
            fps=60,
            codec='libx264')
        all_loss_dict = []
        self.rec_model.eval()

        # for i in range(0, len(c_list), 1): # TODO, larger batch size for eval
        for i, batch in enumerate(tqdm(self.eval_data)):
            # for i in range(0, 8, self.microbatch):
            # c = c_list[i].to(dist_util.dev()).reshape(1, -1)
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            pred = self.rec_model(img=micro['img_to_encoder'],
                                  c=micro['c'])  # pred: (B, 3, 64, 64)
            # target = {
            #     'img': micro['img'],
            #     'depth': micro['depth'],
            #     'depth_mask': micro['depth_mask']
            # }

            # if last_batch or not self.use_ddp:
            #     loss, loss_dict = self.loss_class(pred, target)
            # else:
            #     with self.ddp_model.no_sync():  # type: ignore
            _, loss_dict = self.loss_class(pred, micro, test_mode=True)
            all_loss_dict.append(loss_dict)

            # ! move to other places, add tensorboard
            # gt_vis = th.cat([micro['img'], micro['img']], dim=-1) # TODO, fail to load depth. range [0, 1]
            # pred_vis = th.cat([
            #     pred['image_raw'],
            #     -pred['image_depth'].repeat_interleave(3, dim=1)
            # ],
            #                   dim=-1)
            # vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(1,2,0).cpu().numpy() # ! pred in range[-1, 1]

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
                pred_vis = th.cat([
                    self.pool_64(micro['img']), pred['image_raw'],
                    pred_depth.repeat_interleave(3, dim=1)
                ],
                                  dim=-1)  # B, 3, H, W

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            for j in range(vis.shape[0]):
                video_out.append_data(vis[j])

        video_out.close()

        val_scores_for_logging = calc_average_loss(all_loss_dict)
        with open(os.path.join(logger.get_dir(), 'scores.json'), 'a') as f:
            json.dump({'step': self.step, **val_scores_for_logging}, f)

        # * log to tensorboard
        for k, v in val_scores_for_logging.items():
            self.writer.add_scalar(f'Eval/Rec/{k}', v,
                                   self.step + self.resume_step)

        del video_out, vis, pred_vis, pred, micro
        th.cuda.empty_cache()
        self.eval_novelview_loop()
        self.rec_model.train()

    # for compatablity with p_sample, to lint
    def apply_model_inference(self, x_noisy, t, c=None, model_kwargs={}):
        # control = self.ddp_control_model(x=x_noisy,
        #                                  hint=th.cat(c['c_concat'], 1),
        #                                  timesteps=t,
        #                                  context=None)
        # control = [c * scale for c, scale in zip(control, self.control_scales)]
        pred_params = self.ddp_ddpm_model(x_noisy, t, 
                                    **model_kwargs
                                    )


        assert args.pred_type == 'eps'
        # mixing normal trick
        mixing_component = self.sde_diffusion.mixing_component(
            eps, var_t, t, enabled=True)  # TODO, which should I use?
        pred_eps = get_mixed_prediction(
            True, pred_eps,
            self.ddp_ddpm_model(x=None, timesteps=None, get_attr='mixing_logit'), mixing_component)

        return pred_params

    @th.inference_mode()
    def eval_ddpm_sample(self):

        args = dnnlib.EasyDict(
            dict(
                batch_size=1,
                image_size=self.diffusion_input_size,
                denoise_in_channels=self.ddp_rec_model.module.decoder.
                triplane_decoder.out_chans,  # type: ignore
                clip_denoised=False,
                class_cond=False,
                use_ddim=False))

        model_kwargs = {}

        if args.class_cond:
            classes = th.randint(low=0,
                                 high=NUM_CLASSES,
                                 size=(args.batch_size, ),
                                 device=dist_util.dev())
            model_kwargs["y"] = classes

        diffusion = self.diffusion
        sample_fn = (diffusion.p_sample_loop
                     if not args.use_ddim else diffusion.ddim_sample_loop)

        for i in range(1):
            triplane_sample = sample_fn(
                # self.ddp_model,
                self,
                (
                    args.batch_size,
                    self.ddp_rec_model.module.decoder.ldm_z_channels *
                    3,  # type: ignore
                    self.diffusion_input_size,
                    self.diffusion_input_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                mixing_normal=True,  # !
            )
            th.cuda.empty_cache()

            self.render_video_given_triplane(
                triplane_sample,
                name_prefix=f'{self.step + self.resume_step}_{i}')
            # st()
            del triplane_sample
            th.cuda.empty_cache()
