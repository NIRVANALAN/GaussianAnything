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

from dnnlib.util import requires_grad
from dnnlib.util import calculate_adaptive_weight

from ..train_util_diffusion import TrainLoop3DDiffusion, TrainLoopDiffusionWithRec
from ..cvD.nvsD_canoD import TrainLoop3DcvD_nvsD_canoD

from guided_diffusion.continuous_diffusion_utils import get_mixed_prediction, different_p_q_objectives, kl_per_group_vada, kl_balancer
# import utils as lsgm_utils


class JointDenoiseRecModel(th.nn.Module):

    def __init__(self, ddpm_model, rec_model, diffusion_input_size) -> None:
        super().__init__()
        # del ddpm_model
        # th.cuda.empty_cache()
        # self.ddpm_model = th.nn.Identity()
        self.ddpm_model = ddpm_model
        self.rec_model = rec_model

        self._setup_latent_stat(diffusion_input_size)

    def _setup_latent_stat(self, diffusion_input_size): # for dynamic EMA tracking.
        latent_size = (
            1,
            self.ddpm_model.in_channels,  # type: ignore
            diffusion_input_size,
            diffusion_input_size),

        self.ddpm_model.register_buffer(
            'ema_latent_std',
            th.ones(*latent_size).to(dist_util.dev()), persistent=True)
        self.ddpm_model.register_buffer(
            'ema_latent_mean',
            th.zeros(*latent_size).to(dist_util.dev()), persistent=True)

    # TODO, lint api.
    def forward(
        self,
        *args,
        model_name='ddpm',
        **kwargs,
    ):
        if model_name == 'ddpm':
            return self.ddpm_model(*args, **kwargs)
        elif model_name == 'rec':
            return self.rec_model(*args, **kwargs)
        else:
            raise NotImplementedError(model_name)


# TODO, merge with train_util_diffusion.py later
class SDETrainLoopJoint(TrainLoopDiffusionWithRec):
    """A dataclass with some required attribtues; copied from guided_diffusion TrainLoop
    """

    def __init__(
        self,
        rec_model,
        denoise_model,
        diffusion,  # not used
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
        train_vae=False,
        **kwargs,
    ) -> None:

        joint_model = JointDenoiseRecModel(denoise_model, rec_model, diffusion_input_size)
        super().__init__(
            model=joint_model,
            diffusion=diffusion,  # just for sampling
            loss_class=loss_class,
            data=data,
            eval_data=eval_data,
            eval_interval=eval_interval,
            batch_size=batch_size,
            microbatch=microbatch,
            lr=lr,
            ema_rate=ema_rate,
            log_interval=log_interval,
            save_interval=save_interval,
            resume_checkpoint=resume_checkpoint,
            use_fp16=use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            weight_decay=weight_decay,
            lr_anneal_steps=lr_anneal_steps,
            use_amp=use_amp,
            model_name='joint_denoise_rec_model',
            iterations=iterations,
            triplane_scaling_divider=triplane_scaling_divider,
            diffusion_input_size=diffusion_input_size,
            train_vae=train_vae,
            **kwargs)
        self.sde_diffusion = sde_diffusion
        # setup latent scaling factor

    # ! integrate the init_params_group for rec model
    def _setup_model(self):

        super()._setup_model()
        self.ddp_rec_model = functools.partial(self.model, model_name='rec')
        self.ddp_ddpm_model = functools.partial(self.model, model_name='ddpm')

        self.rec_model = self.ddp_model.module.rec_model
        self.ddpm_model = self.ddp_model.module.ddpm_model  # compatability

        # TODO, required?
        # for param in self.ddp_rec_model.module.decoder.triplane_decoder.parameters(  # type: ignore
        # ):  # type: ignore
        #     param.requires_grad_(
        #         False
        #     )  # ! disable triplane_decoder grad in each iteration indepenently;

    def _load_model(self):
        # TODO, for currently compatability
        if 'joint' in self.resume_checkpoint: # load joint directly
            self._load_and_sync_parameters(model=self.model, model_name=self.model_name)
        else: # from scratch
            self._load_and_sync_parameters(model=self.rec_model, model_name='rec')
            self._load_and_sync_parameters(model=self.ddpm_model,
                                        model_name='ddpm')

    def _setup_opt(self):
        # TODO, two optims groups.
        self.opt = AdamW([{
            'name': 'ddpm',
            'params': self.ddpm_model.parameters(),
        }],
                         lr=self.lr,
                         weight_decay=self.weight_decay)

        if self.train_vae:
            for rec_param_group in self._init_optim_groups(self.rec_model):
                self.opt.add_param_group(rec_param_group)

        print(self.opt)


class TrainLoop3DDiffusionLSGMJointnoD(SDETrainLoopJoint):

    def __init__(self,
                 *,
                 rec_model,
                 denoise_model,
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
                 resume_cldm_checkpoint=None,
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 triplane_scaling_divider=1,
                 use_amp=False,
                 diffusion_input_size=224,
                 diffusion_ce_anneal=False,
                #  compile=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         denoise_model=denoise_model,
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

        if sde_diffusion is not None:
            sde_diffusion.args.batch_size = batch_size
        self.latent_name = 'latent_normalized_2Ddiffusion'  # normalized triplane latent
        self.render_latent_behaviour = 'decode_after_vae'  # directly render using triplane operations
        self.diffusion_ce_anneal = diffusion_ce_anneal
        # assert sde_diffusion.args.train_vae

    def prepare_ddpm(self, eps, mode='p'):

        log_rec3d_loss_dict({
            f'eps_mean': eps.mean(),
            f'eps_std': eps.std([1,2,3]).mean(0),
            f'eps_max': eps.max()
        })

        args = self.sde_diffusion.args
        # sample noise
        noise = th.randn(size=eps.size(), device=eps.device
                         )  # note that this noise value is currently shared!

        # get diffusion quantities for p (sgm prior) sampling scheme and reweighting for q (vae)
        if mode == 'p':
            t_p, var_t_p, m_t_p, obj_weight_t_p, obj_weight_t_q, g2_t_p = \
                self.sde_diffusion.iw_quantities(args.iw_sample_p, noise.shape[0]) # TODO, q not used, fall back to original ddpm implementation
        else:
            assert mode == 'q'
            # assert args.iw_sample_q in ['ll_uniform', 'll_iw']
            t_p, var_t_p, m_t_p, obj_weight_t_p, obj_weight_t_q, g2_t_p = \
                self.sde_diffusion.iw_quantities(args.iw_sample_q, noise.shape[0]) # TODO, q not used, fall back to original ddpm implementation
        eps_t_p = self.sde_diffusion.sample_q(eps, noise, var_t_p, m_t_p)
        # ! important
        # eps_t_p = eps_t_p.detach().requires_grad_(True)
        # logsnr_p = self.sde_diffusion.log_snr(m_t_p,
        #                                         var_t_p)  # for p only
        logsnr_p = self.sde_diffusion.log_snr(m_t_p, var_t_p)  # for p only

        return {
            'noise': noise,
            't_p': t_p,
            'eps_t_p': eps_t_p,
            'logsnr_p': logsnr_p,
            'obj_weight_t_p': obj_weight_t_p,
            'var_t_p': var_t_p,
            'm_t_p': m_t_p,
            'eps': eps,
            'mode': mode
        }

    # merged from noD.py

    def ce_weight(self):
        return self.loss_class.opt.ce_lambda

    def apply_model(self, p_sample_batch, **model_kwargs):
        args = self.sde_diffusion.args
        # args = self.sde_diffusion.args
        noise, eps_t_p, t_p, logsnr_p, obj_weight_t_p, var_t_p, m_t_p = (
            p_sample_batch[k] for k in ('noise', 'eps_t_p', 't_p', 'logsnr_p',
                                        'obj_weight_t_p', 'var_t_p', 'm_t_p'))

        pred_eps_p, pred_x0_p = self.ddpm_step(eps_t_p, t_p, logsnr_p, var_t_p, m_t_p,
                                               **model_kwargs)

        # ! eps loss equivalent to snr weighting of x0 loss, see "progressive distillation"
        with self.ddp_model.no_sync():  # type: ignore
            if args.loss_type == 'eps':
                l2_term_p = th.square(pred_eps_p - noise)  # ? weights
            elif args.loss_type == 'x0':
                # l2_term_p = th.square(pred_eps_p - p_sample_batch['eps'])  # ? weights
                l2_term_p = th.square(
                    pred_x0_p - p_sample_batch['eps'].detach())  # ? weights
                # if args.loss_weight == 'snr':
                # obj_weight_t_p = th.sigmoid(th.exp(logsnr_p))
            else:
                raise NotImplementedError(args.loss_type)

        # p_eps_objective = th.mean(obj_weight_t_p * l2_term_p)
        p_eps_objective = obj_weight_t_p * l2_term_p

        if p_sample_batch['mode'] == 'q':
            ce_weight = self.ce_weight()
            p_eps_objective = p_eps_objective * ce_weight

            log_rec3d_loss_dict({
                'ce_weight': ce_weight,
            })


        log_rec3d_loss_dict({
            f"{p_sample_batch['mode']}_loss":
            p_eps_objective.mean(),
        })
        if self.ddpm_model.mixed_prediction:
            log_rec3d_loss_dict({
                'mixing_logit':
                self.ddp_ddpm_model(x=None,
                                    timesteps=None,
                                    get_attr='mixing_logit').detach(),})

        return {
            'pred_eps_p': pred_eps_p,
            'eps_t_p': eps_t_p,
            'p_eps_objective': p_eps_objective,
            'pred_x0_p': pred_x0_p,
            'logsnr_p': logsnr_p
        }

    def ddpm_step(self, eps_t, t, logsnr, var_t, m_t, **model_kwargs):
        """helper function for ddpm predictions; returns predicted eps, x0 and logsnr. 

        args notes: 
            eps_t is x_noisy
        """
        args = self.sde_diffusion.args
        pred_params = self.ddp_ddpm_model(x=eps_t, timesteps=t, **model_kwargs)
        # logsnr = self.sde_diffusion.log_snr(m_t, var_t)  # for p only
        if args.pred_type in ['eps', 'v']:
            if args.pred_type == 'v':
                pred_eps = self.sde_diffusion._predict_eps_from_z_and_v(
                    pred_params, var_t, eps_t, m_t
                )
                # pred_x0 = self.sde_diffusion._predict_x0_from_z_and_v(
                #     pred_params, var_t, eps_t, m_t)  # ! verified
            else:
                pred_eps = pred_params

            # mixing normal trick

            if self.ddpm_model.mixed_prediction:
                mixing_component = self.sde_diffusion.mixing_component(
                    eps_t, var_t, t, enabled=True)  # z_t * sigma_t
                pred_eps = get_mixed_prediction(
                    True, pred_eps,
                    self.ddp_ddpm_model(x=None,
                                        timesteps=None,
                                        get_attr='mixing_logit'), mixing_component)

            pred_x0 = self.sde_diffusion._predict_x0_from_eps( eps_t, pred_eps, logsnr)  # for VAE loss, denosied latent
            # eps, pred_params, logsnr)  # for VAE loss, denosied latent
        elif args.pred_type == 'x0':
            # ! pred_x0_mixed = alpha * pred_x0 + (1-alpha) * z_t * alpha_t
            pred_x0 = pred_params  # how to mix?

            # mixing normal trick
            mixing_component = self.sde_diffusion.mixing_component_x0(
                eps_t, var_t, t, enabled=True)  # z_t * alpha_t
            pred_x0 = get_mixed_prediction(
                True, pred_x0,
                self.ddp_ddpm_model(x=None,
                                    timesteps=None,
                                    get_attr='mixing_logit'), mixing_component)

            pred_eps = self.sde_diffusion._predict_eps_from_x0(
                eps_t, pred_x0, logsnr)
        else:
            raise NotImplementedError(f'{args.pred_type} not implemented.')

        log_rec3d_loss_dict({
            f'pred_x0_mean': pred_x0.mean(),
            f'pred_x0_std': pred_x0.std([1,2,3]).mean(0),
            f'pred_x0_max': pred_x0.max(),
        })

        return pred_eps, pred_x0

    def ddpm_loss(self, noise, pred_eps, last_batch):

        # ! eps loss equivalent to snr weighting of x0 loss, see "progressive distillation"
        if last_batch or not self.use_ddp:
            l2_term = th.square(pred_eps - noise)
        else:
            with self.ddp_model.no_sync():  # type: ignore
                l2_term = th.square(pred_eps - noise)  # ? weights
        return l2_term

    def run_step(self, batch, step='diffusion_step_rec'):

        if step == 'ce_ddpm_step':
            self.ce_ddpm_step(batch)
        elif step == 'p_rendering_step':
            self.p_rendering_step(batch)

        elif step == 'eps_step':
            self.eps_step(batch)

        # ! both took ddpm step
        self._update_ema()

        self._anneal_lr()
        self.log_step()

    @th.inference_mode()
    def _post_run_loop(self):

        # if self.step % self.eval_interval == 0 and self.step != 0:
        # if self.step % self.eval_interval == 0:
        #     if dist_util.get_rank() == 0:
        #         self.eval_ddpm_sample(
        #             self.rec_model,
        #             # self.ddpm_model
        #         )  # ! only support single GPU inference now.
        #         if self.sde_diffusion.args.train_vae:
        #             self.eval_loop(self.ddp_rec_model)

        if self.step % self.log_interval == 0 and dist_util.get_rank() == 0:
            out = logger.dumpkvs()
            # * log to tensorboard
            for k, v in out.items():
                self.writer.add_scalar(f'Loss/{k}', v,
                                       self.step + self.resume_step)

        # if self.step % self.eval_interval == 0 and self.step != 0:
        if self.step % self.eval_interval == 0:
            if dist_util.get_rank() == 0:
                self.eval_ddpm_sample(self.ddp_rec_model)
                if self.sde_diffusion.args.train_vae:
                    self.eval_loop(self.ddp_rec_model)

        if self.step % self.save_interval == 0:
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
            dist_util.synchronize()

            batch = next(self.data)
            self.run_step(batch, step='ce_ddpm_step')

            self._post_run_loop()

            # batch = next(self.data)
            # self.run_step(batch, step='p_rendering_step')

    def ce_ddpm_step(self, batch, behaviour='rec', *args, **kwargs):
        """
        add sds grad to all ae predicted x_0 
        """
        args = self.sde_diffusion.args
        assert args.train_vae

        requires_grad(self.rec_model, args.train_vae)
        requires_grad(self.ddpm_model, True)

        # TODO merge?
        self.mp_trainer.zero_grad()

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
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

                # ! part 1: train vae with CE; ddpm fixed
                # ! TODO, add KL_all_list? vae.decompose
                with th.set_grad_enabled(args.train_vae):
                    # vae_out = self.ddp_rec_model(
                    #     img=micro['img_to_encoder'],
                    #     c=micro['c'],
                    #     behaviour='encoder_vae',
                    # )  # pred: (B, 3, 64, 64)
                    # TODO, no need to render if not SSD; no need to do ViT decoder if only the latent is needed. update later
                    # if args.train_vae:
                    # if args.add_rendering_loss:
                    # if args.joint_train:
                    # with th.set_grad_enabled(args.train_vae):
                    pred = self.ddp_rec_model(
                        # latent=vae_out,
                        img=micro['img_to_encoder'],
                        c=micro['c'],
                    )
                    # behaviour=self.render_latent_behaviour)

                    # vae reconstruction loss
                    if last_batch or not self.use_ddp:
                        q_vae_recon_loss, loss_dict = self.loss_class(
                            pred, micro, test_mode=False)
                    else:
                        with self.ddp_model.no_sync():  # type: ignore
                            q_vae_recon_loss, loss_dict = self.loss_class(
                                pred, micro, test_mode=False)

                    log_rec3d_loss_dict(loss_dict)
                # '''

                # ! calculate p/q loss;
                # nelbo_loss = balanced_kl * self.loss_class.opt.ce_balanced_kl + q_vae_recon_loss
                nelbo_loss = q_vae_recon_loss
                q_loss = th.mean(nelbo_loss)

                # st()

                # all_log_q = [vae_out['log_q_2Ddiffusion']]
                # eps = vae_out[self.latent_name]
                # all_log_q = [pred['log_q_2Ddiffusion']]
                eps = pred[self.latent_name]

                if not args.train_vae:
                    eps.requires_grad_(True)  # single stage diffusion

                # sample noise
                noise = th.randn(
                    size=eps.size(), device=eps.device
                )  # note that this noise value is currently shared!

                # in case we want to train q (vae) with another batch using a different sampling scheme for times t
                '''
                assert args.iw_sample_q in ['ll_uniform', 'll_iw']
                t_q, var_t_q, m_t_q, obj_weight_t_q, _, g2_t_q = \
                    self.sde_diffusion.iw_quantities(args.iw_sample_q)
                eps_t_q = self.sde_diffusion.sample_q(eps, noise, var_t_q,
                                                      m_t_q)

                # eps_t = th.cat([eps_t_p, eps_t_q], dim=0)
                # var_t = th.cat([var_t_p, var_t_q], dim=0)
                # t = th.cat([t_p, t_q], dim=0)
                # noise = th.cat([noise, noise], dim=0)

                # run the diffusion model
                if not args.train_vae:
                    eps_t_q.requires_grad_(True)  # 2*BS, 12, 16, 16

                # ! For CE guidance.
                requires_grad(self.ddpm_model_module, False)
                pred_eps_q, _, _ = self.ddpm_step(eps_t_q, t_q, m_t_q, var_t_q)

                l2_term_q = self.ddpm_loss(noise, pred_eps_q, last_batch)

                # pred_eps = th.cat([pred_eps_p, pred_eps_q], dim=0)  # p then q

                # ÇE: nelbo loss with kl balancing
                assert args.iw_sample_q in ['ll_uniform', 'll_iw']
                # l2_term_p, l2_term_q = th.chunk(l2_term, chunks=2, dim=0)
                cross_entropy_per_var = obj_weight_t_q * l2_term_q

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
                # st()

                log_rec3d_loss_dict(
                    dict(
                        balanced_kl=balanced_kl,
                        l2_term_q=l2_term_q,
                        cross_entropy_per_var=cross_entropy_per_var.mean(),
                        all_log_q=all_log_q[0].mean(),
                    ))


                '''
                # ! update vae for CE
                # ! single stage diffusion for rec side 1: bind vae prior and diffusion prior

            # ! BP for CE and VAE; quit the AMP context.
            # if args.train_vae:
            #     self.mp_trainer.backward(q_loss)
            #     _ = self.mp_trainer.optimize(self.opt)
            #  retain_graph=different_p_q_objectives(
            #      args.iw_sample_p,
            #      args.iw_sample_q))

            log_rec3d_loss_dict(
                dict(q_vae_recon_loss=q_vae_recon_loss,
                     # all_log_q=all_log_q[0].mean(),
                     ))

            # ! adding p loss; enable ddpm gradient
            # self.mp_trainer.zero_grad()
            # requires_grad(self.rec_model_module,
            #               False)  # could be removed since eps_t_p.detach()
            with th.cuda.amp.autocast(dtype=th.float16,
                                      enabled=self.mp_trainer.use_amp):

                # first get diffusion quantities for p (sgm prior) sampling scheme and reweighting for q (vae)
                t_p, var_t_p, m_t_p, obj_weight_t_p, obj_weight_t_q, g2_t_p = \
                    self.sde_diffusion.iw_quantities(args.iw_sample_p)
                eps_t_p = self.sde_diffusion.sample_q(eps, noise, var_t_p,
                                                      m_t_p)
                eps_t_p = eps_t_p.detach(
                )  # .requires_grad_(True)  # ! update ddpm not rec module

                pred_eps_p, _, = self.ddpm_step(eps_t_p, t_p, m_t_p, var_t_p)
                l2_term_p = self.ddpm_loss(noise, pred_eps_p, last_batch)
                p_loss = th.mean(obj_weight_t_p * l2_term_p)

            # ! update ddpm
            self.mp_trainer.backward(p_loss +
                                     q_loss)  # just backward for p_loss
            _ = self.mp_trainer.optimize(self.opt)
            # requires_grad(self.rec_model_module, True)

            log_rec3d_loss_dict(
                dict(
                    p_loss=p_loss,
                    mixing_logit=self.ddp_ddpm_model(
                        x=None, timesteps=None,
                        get_attr='mixing_logit').detach(),
                ))

            # TODO, merge visualization with original AE
            # =================================== denoised AE log part ===================================

            # ! todo, wrap in a single function
            if dist_util.get_rank() == 0 and self.step % 500 == 0:

                with th.no_grad():

                    if not args.train_vae:
                        vae_out.pop('posterior')  # for calculating kl loss
                        vae_out_for_pred = {
                            k:
                            v[0:1].to(dist_util.dev()) if isinstance(
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
                            gt_img,
                            # micro['img'],
                            gt_depth.repeat_interleave(3, dim=1)
                        ],
                        dim=-1)[0:1]  # TODO, fail to load depth. range [0, 1]

                    pred_vis = th.cat([
                        pred_img[0:1], pred_depth[0:1].repeat_interleave(3,
                                                                         dim=1)
                    ],
                                      dim=-1)  # B, 3, H, W

                    vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
                        1, 2, 0).cpu()  # ! pred in range[-1, 1]

                    # vis_grid = torchvision.utils.make_grid(vis) # HWC
                    vis = vis.numpy() * 127.5 + 127.5
                    vis = vis.clip(0, 255).astype(np.uint8)
                    Image.fromarray(vis).save(
                        # f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t[0].item()}_{behaviour}.jpg'
                        f'{logger.get_dir()}/{self.step+self.resume_step}_{behaviour}.jpg'
                    )
                    print(
                        'log denoised vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}_{behaviour}.jpg'
                    )

                    th.cuda.empty_cache()

    def eps_step(self, batch, behaviour='rec', *args, **kwargs):
        """
        add sds grad to all ae predicted x_0 
        """
        args = self.sde_diffusion.args

        requires_grad(self.ddpm_model_module, True)
        requires_grad(self.rec_model_module, False)

        # TODO?
        # if args.train_vae:
        #     for param in self.ddp_rec_model.module.decoder.triplane_decoder.parameters(  # type: ignore
        #     ):  # type: ignore
        #         param.requires_grad_(
        #             False
        #         )  # ! disable triplane_decoder grad in each iteration indepenently;

        self.mp_trainer.zero_grad()

        # assert args.train_vae

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v
                for k, v in batch.items()
            }

            last_batch = (i + self.microbatch) >= batch_size

            # =================================== ae part ===================================
            with th.cuda.amp.autocast(dtype=th.float16,
                                      enabled=self.mp_trainer.use_amp):
                #   and args.train_vae):

                # ! part 1: train vae with CE; ddpm fixed
                # ! TODO, add KL_all_list? vae.decompose

                with th.set_grad_enabled(args.train_vae):
                    vae_out = self.ddp_rec_model(
                        img=micro['img_to_encoder'],
                        c=micro['c'],
                        behaviour='encoder_vae',
                    )  # pred: (B, 3, 64, 64)
                eps = vae_out[self.latent_name]

                # sample noise
                noise = th.randn(
                    size=eps.size(), device=eps.device
                )  # note that this noise value is currently shared!

                # get diffusion quantities for p (sgm prior) sampling scheme and reweighting for q (vae)
                t_p, var_t_p, m_t_p, obj_weight_t_p, obj_weight_t_q, g2_t_p = \
                    self.sde_diffusion.iw_quantities(args.iw_sample_p)
                eps_t_p = self.sde_diffusion.sample_q(eps, noise, var_t_p,
                                                      m_t_p)
                logsnr_p = self.sde_diffusion.log_snr(m_t_p,
                                                      var_t_p)  # for p only

                pred_eps_p, pred_x0_p, logsnr_p = self.ddpm_step(
                    eps_t_p, t_p, m_t_p, var_t_p)

                # ! batchify for mixing_component
                # mixing normal trick
                mixing_component = self.sde_diffusion.mixing_component(
                    eps_t_p, var_t_p, t_p,
                    enabled=True)  # TODO, which should I use?
                pred_eps_p = get_mixed_prediction(
                    True, pred_eps_p,
                    self.ddp_ddpm_model(x=None,
                                        timesteps=None,
                                        get_attr='mixing_logit'),
                    mixing_component)

                # ! eps loss equivalent to snr weighting of x0 loss, see "progressive distillation"
                if last_batch or not self.use_ddp:
                    l2_term_p = th.square(pred_eps_p - noise)
                else:
                    with self.ddp_ddpm_model.no_sync():  # type: ignore
                        l2_term_p = th.square(pred_eps_p - noise)  # ? weights

                p_eps_objective = th.mean(
                    obj_weight_t_p *
                    l2_term_p) * self.loss_class.opt.p_eps_lambda

                log_rec3d_loss_dict(
                    dict(mixing_logit=self.ddp_ddpm_model(
                        x=None, timesteps=None,
                        get_attr='mixing_logit').detach(), ))

                # =====================================================================
                # ! single stage diffusion for rec side 2: generative feature
                # if args.p_rendering_loss:
                #     target = micro
                #     pred = self.ddp_rec_model(
                #         # latent=vae_out,
                #         latent={
                #             **vae_out, self.latent_name: pred_x0_p,
                #             'latent_name': self.latent_name
                #         },
                #         c=micro['c'],
                #         behaviour=self.render_latent_behaviour)

                #     # vae reconstruction loss
                #     if last_batch or not self.use_ddp:
                #         p_vae_recon_loss, _ = self.loss_class(pred,
                #                                               target,
                #                                               test_mode=False)
                #     else:
                #         with self.ddp_model.no_sync():  # type: ignore
                #             p_vae_recon_loss, _ = self.loss_class(
                #                 pred, target, test_mode=False)
                #     log_rec3d_loss_dict(
                #         dict(p_vae_recon_loss=p_vae_recon_loss, ))
                #     p_loss = p_eps_objective + p_vae_recon_loss
                # else:
                p_loss = p_eps_objective

                log_rec3d_loss_dict(
                    dict(p_loss=p_loss, p_eps_objective=p_eps_objective))

                # ! to arrange: update vae params

            self.mp_trainer.backward(p_loss)

            # update ddpm accordingly
            _ = self.mp_trainer.optimize(
                self.opt)  # TODO, update two groups of parameters

            # TODO, merge visualization with original AE
            # ! todo, merge required
            # =================================== denoised AE log part ===================================
            if dist_util.get_rank(
            ) == 0 and self.step % 500 == 0 and behaviour != 'diff':

                with th.no_grad():

                    vae_out.pop('posterior')  # for calculating kl loss
                    vae_out_for_pred = {
                        k:
                        v[0:1].to(dist_util.dev())
                        if isinstance(v, th.Tensor) else v
                        for k, v in vae_out.items()
                    }

                    pred = self.ddp_rec_model(
                        latent=vae_out_for_pred,
                        c=micro['c'][0:1],
                        behaviour=self.render_latent_behaviour)
                    assert isinstance(pred, dict)

                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                              gt_depth.min())

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

                    pred_x0 = self.sde_diffusion._predict_x0_from_eps(
                        eps_t_p, pred_eps_p,
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
                        f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t_p[0].item():3}_{behaviour}.jpg'
                    )
                    print(
                        'log denoised vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t_p[0].item():3}_{behaviour}.jpg'
                    )
                    del vis, pred_vis, pred_x0, pred_eps_p, micro, vae_out

                    th.cuda.empty_cache()

    def p_rendering_step(self, batch, behaviour='rec', *args, **kwargs):
        """
        add sds grad to all ae predicted x_0 
        """
        args = self.sde_diffusion.args

        requires_grad(self.ddpm_model, True)
        requires_grad(self.rec_model, args.train_vae)

        # TODO?
        # if args.train_vae:
        #     for param in self.ddp_rec_model.module.decoder.triplane_decoder.parameters(  # type: ignore
        #     ):  # type: ignore
        #         param.requires_grad_(
        #             False
        #         )  # ! disable triplane_decoder grad in each iteration indepenently;

        self.mp_trainer.zero_grad()

        assert args.train_vae

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v
                for k, v in batch.items()
            }

            last_batch = (i + self.microbatch) >= batch_size

            # =================================== ae part ===================================
            with th.cuda.amp.autocast(dtype=th.float16,
                                      enabled=self.mp_trainer.use_amp):
                #   and args.train_vae):

                # ! part 1: train vae with CE; ddpm fixed
                # ! TODO, add KL_all_list? vae.decompose

                with th.set_grad_enabled(args.train_vae):
                    vae_out = self.ddp_rec_model(
                        img=micro['img_to_encoder'],
                        c=micro['c'],
                        behaviour='encoder_vae',
                    )  # pred: (B, 3, 64, 64)
                eps = vae_out[self.latent_name]

                # sample noise
                noise = th.randn(
                    size=eps.size(), device=eps.device
                )  # note that this noise value is currently shared!

                # get diffusion quantities for p (sgm prior) sampling scheme and reweighting for q (vae)
                t_p, var_t_p, m_t_p, obj_weight_t_p, obj_weight_t_q, g2_t_p = \
                    self.sde_diffusion.iw_quantities(args.iw_sample_p)
                eps_t_p = self.sde_diffusion.sample_q(eps, noise, var_t_p,
                                                      m_t_p)
                logsnr_p = self.sde_diffusion.log_snr(m_t_p,
                                                      var_t_p)  # for p only

                # pred_eps_p, pred_x0_p, logsnr_p = self.ddpm_step(
                pred_eps_p, pred_x0_p = self.ddpm_step(eps_t_p, t_p, logsnr_p,
                                                       var_t_p)
                # eps_t_p, t_p, m_t_p, var_t_p)

                # ! batchify for mixing_component
                # mixing normal trick
                # mixing_component = self.sde_diffusion.mixing_component(
                #     eps_t_p, var_t_p, t_p,
                #     enabled=True)  # TODO, which should I use?
                # pred_eps_p = get_mixed_prediction(
                #     True, pred_eps_p,
                #     self.ddp_ddpm_model(x=None,
                #                         timesteps=None,
                #                         get_attr='mixing_logit'),
                #     mixing_component)

                # ! eps loss equivalent to snr weighting of x0 loss, see "progressive distillation"
                if last_batch or not self.use_ddp:
                    l2_term_p = th.square(pred_eps_p - noise)
                else:
                    with self.ddp_model.no_sync():  # type: ignore
                        l2_term_p = th.square(pred_eps_p - noise)  # ? weights

                p_eps_objective = th.mean(obj_weight_t_p * l2_term_p)
                # st()

                log_rec3d_loss_dict(
                    dict(mixing_logit=self.ddp_ddpm_model(
                        x=None, timesteps=None,
                        get_attr='mixing_logit').detach(), ))

                # =====================================================================
                # ! single stage diffusion for rec side 2: generative feature
                if args.p_rendering_loss:
                    target = micro
                    pred = self.ddp_rec_model( # re-render 
                        latent={
                            **vae_out, self.latent_name: pred_x0_p,
                            'latent_name': self.latent_name
                        },
                        c=micro['c'],
                        behaviour=self.render_latent_behaviour)

                    # vae reconstruction loss
                    if last_batch or not self.use_ddp:
                        pred[self.latent_name] = vae_out[self.latent_name]
                        pred[
                            'latent_name'] = self.latent_name  # just for stats
                        p_vae_recon_loss, rec_loss_dict = self.loss_class(
                            pred, target, test_mode=False)
                    else:
                        with self.ddp_model.no_sync():  # type: ignore
                            p_vae_recon_loss, rec_loss_dict = self.loss_class(
                                pred, target, test_mode=False)
                    log_rec3d_loss_dict(
                        dict(p_vae_recon_loss=p_vae_recon_loss, ))

                    for key in rec_loss_dict.keys():
                        if 'latent' in key:
                            log_rec3d_loss_dict({key: rec_loss_dict[key]})

                    p_loss = p_eps_objective + p_vae_recon_loss
                else:
                    p_loss = p_eps_objective

                log_rec3d_loss_dict(
                    dict(p_loss=p_loss, p_eps_objective=p_eps_objective))

                # ! to arrange: update vae params

            self.mp_trainer.backward(p_loss)

            # update ddpm accordingly
            _ = self.mp_trainer.optimize(
                self.opt)  # TODO, update two groups of parameters

            # TODO, merge visualization with original AE
            # ! todo, merge required
            # =================================== denoised AE log part ===================================
            if dist_util.get_rank(
            ) == 0 and self.step % 500 == 0 and behaviour != 'diff':

                with th.no_grad():

                    vae_out.pop('posterior')  # for calculating kl loss
                    vae_out_for_pred = {
                        k:
                        v[0:1].to(dist_util.dev())
                        if isinstance(v, th.Tensor) else v
                        for k, v in vae_out.items()
                    }

                    pred = self.ddp_rec_model(
                        latent=vae_out_for_pred,
                        c=micro['c'][0:1],
                        behaviour=self.render_latent_behaviour)
                    assert isinstance(pred, dict)

                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                              gt_depth.min())

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

                    pred_x0 = self.sde_diffusion._predict_x0_from_eps(
                        eps_t_p, pred_eps_p,
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
                        f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t_p[0].item():3}_{behaviour}.jpg'
                    )
                    print(
                        'log denoised vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t_p[0].item():3}_{behaviour}.jpg'
                    )
                    del vis, pred_vis, pred_x0, pred_eps_p, micro, vae_out

                    th.cuda.empty_cache()


class TrainLoop3DDiffusionLSGMJointnoD_ponly(TrainLoop3DDiffusionLSGMJointnoD):

    def __init__(self,
                 *,
                 rec_model,
                 denoise_model,
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
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         denoise_model=denoise_model,
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

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            self._post_run_loop()

            # let all processes sync up before starting with a new epoch of training
            dist_util.synchronize()

            # batch = next(self.data)
            # self.run_step(batch, step='ce_ddpm_step')

            batch = next(self.data)
            self.run_step(batch, step='p_rendering_step')
            # self.run_step(batch, step='eps_step')
