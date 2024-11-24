"""
from ControlNet/cldm/cldm.py
"""
import copy
import functools
import json
import os
from pathlib import Path
from pdb import set_trace as st
from typing import Any
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
from .train_util_diffusion_lsgm_noD_joint import TrainLoop3DDiffusionLSGMJointnoD  # joint diffusion and rec class


class TrainLoop3DDiffusionLSGM_Control(TrainLoop3DDiffusionLSGMJointnoD):

    def __init__(self,
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
                         resume_cldm_checkpoint=None,
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
                         **kwargs)
        self.resume_cldm_checkpoint = resume_cldm_checkpoint
        self.control_model = control_model
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.sd_locked = True
        self._setup_control_model()

    def _setup_control_model(self):

        requires_grad(self.rec_model, False)
        requires_grad(self.ddpm_model, self.sd_locked)

        self.mp_cldm_trainer = MixedPrecisionTrainer(
            model=self.control_model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
            use_amp=self.use_amp,
            model_name='cldm')

        self.ddp_control_model = DDP(
            self.control_model,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False,
        )

        # ! load trainable copy
        try:
            logger.log(f"load pretrained controlnet, not trainable copy.")
            self._load_and_sync_parameters(model=self.control_model,
                                           model_name='cldm',
                                           resume_checkpoint=self.resume_cldm_checkpoint,
                                           )  # if available
        except:
            logger.log(f"load trainable copy to controlnet")
            self._load_and_sync_parameters(
                model=self.control_model,
                model_name='ddpm')  # load pre-trained SD

        cldm_param = [{
            'name': 'cldm.parameters()',
            'params': self.control_model.parameters(),
        }]
        if self.sde_diffusion.args.unfix_logit:
            self.ddpm_model.mixing_logit.requires_grad_(True)
            cldm_param.append({
                'name': 'mixing_logit',
                'params': self.ddpm_model.mixing_logit,
            })

        self.opt_cldm = AdamW(cldm_param,
                              lr=self.lr,
                              weight_decay=self.weight_decay)
        if self.sd_locked:
            del self.opt

    # def _load_model(self):
    #     super()._load_model()
    #     # ! load pre-trained "SD" and controlNet also
    #     self._load_and_sync_parameters(model=self.contro,
    #                                    model_name='cldm') #

    # def _setup_opt(self):
    # TODO, two optims groups.

    # for rec_param_group in self._init_optim_groups(self.rec_model):
    #     self.opt.add_param_group(rec_param_group)

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            # dist_util.synchronize()

            batch = next(self.data)
            self.run_step(batch, step='cldm_step')

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                # * log to tensorboard
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            if self.step % self.eval_interval == 0 and self.step != 0:
                # if self.step % self.eval_interval == 0:
                if dist_util.get_rank() == 0:
                    # self.eval_ddpm_sample()
                    self.eval_cldm()
                    # if self.sde_diffusion.args.train_vae:
                    #     self.eval_loop()

                th.cuda.empty_cache()
                dist_util.synchronize()

            if self.step % self.save_interval == 0:
                self.save(self.mp_cldm_trainer,
                          self.mp_cldm_trainer.model_name)
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                print('reached maximum iterations, exiting')

                # Save the last checkpoint if it wasn't already saved.
                if (self.step - 1) % self.save_interval != 0:

                    self.save(self.mp_cldm_trainer,
                              self.mp_cldm_trainer.model_name)
                    # if self.sde_diffusion.args.train_vae:
                    #     self.save(self.mp_trainer_rec,
                    #               self.mp_trainer_rec.model_name)

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save(
                self.mp_cldm_trainer,
                self.mp_cldm_trainer.model_name)  # rec and ddpm all fixed.
            # st()
            # self.save(self.mp_trainer_canonical_cvD, 'cvD')

    def _update_cldm_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_cldm_params):
            update_ema(params, self.mp_cldm_trainer.master_params, rate=rate)

    def run_step(self, batch, step='cldm_step'):

        # if step == 'diffusion_step_rec':

        if step == 'cldm_step':
            self.cldm_train_step(batch)

        # if took_step_ddpm:
        # self._update_cldm_ema()

        self._anneal_lr()
        self.log_step()

    @th.no_grad()
    def get_c_input(self, batch, bs=None, *args, **kwargs):
        # x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        # control = control.to(self.device)
        # control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=th.contiguous_format).float()
        # return x, dict(c_crossattn=[c], c_concat=[control])
        return dict(c_concat=[control])

    # for compatablity with p_sample, to lint
    def apply_model_inference(self, x_noisy, t, c, model_kwargs={}):
        control = self.ddp_control_model(x=x_noisy,
                                         hint=th.cat(c['c_concat'], 1),
                                         timesteps=t,
                                         context=None)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        pred_params = self.ddp_ddpm_model(
            x_noisy, t, **{
                **model_kwargs, 'control': control
            })
        return pred_params

    def apply_control_model(self, p_sample_batch, cond):
        x_noisy, t, = (p_sample_batch[k] for k in ('eps_t_p', 't_p'))

        control = self.ddp_control_model(x=x_noisy,
                                         hint=th.cat(cond['c_concat'], 1),
                                         timesteps=t,
                                         context=None)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        return control

    def apply_model(self, p_sample_batch, cond, model_kwargs={}):
        control = self.apply_control_model(p_sample_batch,
                                           cond)  # len(control): 13
        return super().apply_model(p_sample_batch, **{
            **model_kwargs, 'control': control
        })

    # ddpm + rec loss
    def cldm_train_step(self, batch, behaviour='cano', *args, **kwargs):
        """
        add sds grad to all ae predicted x_0 
        """

        # ! enable the gradient of both models
        requires_grad(self.ddp_control_model, True)

        self.mp_cldm_trainer.zero_grad()  # !!!!

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
                                      enabled=self.mp_cldm_trainer.use_amp):

                loss = th.tensor(0.).to(dist_util.dev())

                vae_out = self.ddp_rec_model(
                    img=micro['img_to_encoder'],
                    c=micro['c'],
                    behaviour='encoder_vae',
                )  # pred: (B, 3, 64, 64)
                # eps = vae_out[self.latent_name]
                eps = vae_out.pop(self.latent_name)

                p_sample_batch = self.prepare_ddpm(eps)
                cond = self.get_c_input(micro)

                # ! running diffusion forward
                ddpm_ret = self.apply_model(p_sample_batch, cond)
                if self.sde_diffusion.args.p_rendering_loss:

                    target = micro
                    pred = self.ddp_rec_model(
                        # latent=vae_out,
                        latent={
                            # **vae_out, 
                            self.latent_name:
                            ddpm_ret['pred_x0_p'],
                            'latent_name': self.latent_name
                        },
                        c=micro['c'],
                        behaviour=self.render_latent_behaviour)

                    # vae reconstruction loss
                    with self.ddp_control_model.no_sync():  # type: ignore
                        p_vae_recon_loss, rec_loss_dict = self.loss_class(
                            pred, target, test_mode=False)
                    log_rec3d_loss_dict(rec_loss_dict)
                    # log_rec3d_loss_dict(
                    #     dict(p_vae_recon_loss=p_vae_recon_loss, ))
                    loss = p_vae_recon_loss + ddpm_ret['p_eps_objective']  # TODO, add obj_weight_t_p?
                else:
                    loss = ddpm_ret['p_eps_objective']

                # =====================================================================

            self.mp_cldm_trainer.backward(loss)  # joint gradient descent

        # update ddpm accordingly
        self.mp_cldm_trainer.optimize(self.opt_cldm)

        if dist_util.get_rank() == 0 and self.step % 500 == 0:
            self.log_control_images(vae_out, p_sample_batch, micro,
                                    ddpm_ret)

    @th.inference_mode()
    def log_control_images(self, vae_out, p_sample_batch, micro, ddpm_ret):

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
        cond = self.get_c_input(micro)
        hint = th.cat(cond['c_concat'], 1)

        gt_vis = th.cat([
            gt_img,
            self.pool_128(hint), gt_img,
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

        pred_x0 = self.sde_diffusion._predict_x0_from_eps(
            eps_t_p, pred_eps_p, logsnr_p)  # for VAE loss, denosied latent

        # pred_xstart_3D
        denoised_ae_pred = self.ddp_rec_model(
            img=None,
            c=micro['c'][0:1],
            latent=pred_x0[0:1] * self.
            triplane_scaling_divider,  # TODO, how to define the scale automatically?
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
    def eval_cldm(self):
        self.control_model.eval()

        args = dnnlib.EasyDict(
            dict(
                batch_size=1,
                image_size=self.diffusion_input_size,
                denoise_in_channels=self.rec_model.decoder.triplane_decoder.
                out_chans,  # type: ignore
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

        # for i, batch in enumerate(tqdm(self.eval_data)):
        batch = next(iter(self.eval_data))

        # use the first frame as the condition now
        novel_view_cond = {
            k: v[0:1].to(dist_util.dev())  # .repeat_interleave(
            # micro['img'].shape[0], 0)
            for k, v in batch.items()
        }
        cond = self.get_c_input(novel_view_cond)
        hint = th.cat(cond['c_concat'], 1)

        # record cond images
        torchvision.utils.save_image(
            hint,
            f'{logger.get_dir()}/{self.step + self.resume_step}_cond.jpg',
            normalize=True,
            value_range=(-1, 1))

        # broadcast to args.batch_size
        cond = {
            k:
            [cond.repeat_interleave(args.batch_size, 0) for cond in cond_list]
            for k, cond_list in cond.items()  # list of Tensors
        }

        for i in range(1):
            triplane_sample = sample_fn(
                self,
                (
                    args.batch_size,
                    self.rec_model.decoder.ldm_z_channels * 3,  # type: ignore
                    self.diffusion_input_size,
                    self.diffusion_input_size),
                cond=cond,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                mixing_normal=True,  # !
                device=dist_util.dev())
            th.cuda.empty_cache()

            self.render_video_given_triplane(
                triplane_sample,
                self.rec_model,  # compatible with join_model
                name_prefix=f'{self.step + self.resume_step}_{i}')

            del triplane_sample
            th.cuda.empty_cache()

        self.control_model.train()