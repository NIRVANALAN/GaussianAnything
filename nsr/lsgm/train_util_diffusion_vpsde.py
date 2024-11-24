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
from dnnlib.util import calculate_adaptive_weight

from ..train_util_diffusion import TrainLoop3DDiffusion
from ..cvD.nvsD_canoD import TrainLoop3DcvD_nvsD_canoD


class TrainLoop3DDiffusion_vpsde(TrainLoop3DDiffusion,TrainLoop3DcvD_nvsD_canoD):
    def __init__(self, *, rec_model, denoise_model, diffusion, loss_class, data, eval_data, batch_size, microbatch, lr, ema_rate, log_interval, eval_interval, save_interval, resume_checkpoint, use_fp16=False, fp16_scale_growth=0.001, schedule_sampler=None, weight_decay=0, lr_anneal_steps=0, iterations=10001, ignore_resume_opt=False, freeze_ae=False, denoised_ae=True, triplane_scaling_divider=10, use_amp=False, diffusion_input_size=224, **kwargs):
        super().__init__(rec_model=rec_model, denoise_model=denoise_model, diffusion=diffusion, loss_class=loss_class, data=data, eval_data=eval_data, batch_size=batch_size, microbatch=microbatch, lr=lr, ema_rate=ema_rate, log_interval=log_interval, eval_interval=eval_interval, save_interval=save_interval, resume_checkpoint=resume_checkpoint, use_fp16=use_fp16, fp16_scale_growth=fp16_scale_growth, schedule_sampler=schedule_sampler, weight_decay=weight_decay, lr_anneal_steps=lr_anneal_steps, iterations=iterations, ignore_resume_opt=ignore_resume_opt, freeze_ae=freeze_ae, denoised_ae=denoised_ae, triplane_scaling_divider=triplane_scaling_divider, use_amp=use_amp, diffusion_input_size=diffusion_input_size, **kwargs)

    def run_step(self, batch, step='g_step'):

        if step == 'diffusion_step_rec':
            self.forward_diffusion(batch, behaviour='diffusion_step_rec')
            _ = self.mp_trainer_rec.optimize(self.opt_rec) # TODO, update two groups of parameters
            took_step_ddpm = self.mp_trainer.optimize(self.opt) # TODO, update two groups of parameters

            if took_step_ddpm:
                self._update_ema()  # g_ema # TODO, ema only needs to track ddpm, remove ema tracking in rec

        elif step == 'd_step_rec':
            self.forward_D(batch, behaviour='rec')
            # _ = self.mp_trainer_cvD.optimize(self.opt_cvD)
            _ = self.mp_trainer_canonical_cvD.optimize(self.opt_cano_cvD)

        elif step == 'diffusion_step_nvs':
            self.forward_diffusion(batch, behaviour='diffusion_step_nvs')
            _ = self.mp_trainer_rec.optimize(self.opt_rec) # TODO, update two groups of parameters
            took_step_ddpm = self.mp_trainer.optimize(self.opt) # TODO, update two groups of parameters

            if took_step_ddpm:
                self._update_ema()  # g_ema

        elif step == 'd_step_nvs':
            self.forward_D(batch, behaviour='nvs')
            _ = self.mp_trainer_cvD.optimize(self.opt_cvD)

        self._anneal_lr()
        self.log_step()

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            dist_util.synchronize()

            # batch, cond = next(self.data)
            # if batch is None:
            # batch = next(self.data)
            # self.run_step(batch, 'g_step_rec')

            batch = next(self.data)
            self.run_step(batch, step='diffusion_step_rec')

            batch = next(self.data)
            self.run_step(batch, 'd_step_rec')

            # batch = next(self.data)
            # self.run_step(batch, 'g_step_nvs')

            batch = next(self.data)
            self.run_step(batch, step='diffusion_step_nvs')

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
                self.save(self.mp_trainer, self.mp_trainer.model_name)
                self.save(self.mp_trainer_rec, self.mp_trainer_rec.model_name)
                self.save(self.mp_trainer_cvD, 'cvD')
                self.save(self.mp_trainer_canonical_cvD, 'cano_cvD')

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

                    self.save(self.mp_trainer, self.mp_trainer.model_name)
                    self.save(self.mp_trainer_rec, self.mp_trainer_rec.model_name)
                    self.save(self.mp_trainer_cvD, 'cvD')
                    self.save(self.mp_trainer_canonical_cvD, 'cano_cvD')

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.save(self.mp_trainer_canonical_cvD, 'cvD')

    def forward_diffusion(self, batch, behaviour='rec', *args, **kwargs):
        """
        add sds grad to all ae predicted x_0 
        """

        self.ddp_cano_cvD.requires_grad_(False)
        self.ddp_nvs_cvD.requires_grad_(False)

        self.ddp_model.requires_grad_(True)
        self.ddp_rec_model.requires_grad_(True)

        # if behaviour != 'diff' and 'rec' in behaviour:
        # if behaviour != 'diff' and 'rec' in behaviour: # pure diffusion step
        #     self.ddp_rec_model.requires_grad_(True)
        for param in self.ddp_rec_model.module.decoder.triplane_decoder.parameters( # type: ignore
        ):  # type: ignore
            param.requires_grad_(False) # ! disable triplane_decoder grad in each iteration indepenently; 
        # else:

        self.mp_trainer_rec.zero_grad()
        self.mp_trainer.zero_grad()

        # ! no 'sds' step now, both add sds grad back to ViT

        # assert behaviour != 'sds'
        # if behaviour == 'sds':
        # else:
        #     self.ddp_ddpm_model.requires_grad_(True)

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev())
                for k, v in batch.items()
            }

            last_batch = (i + self.microbatch) >= batch_size

            vae_nelbo_loss = th.tensor(0.0).to(dist_util.dev())
            vision_aided_loss = th.tensor(0.0).to(dist_util.dev())
            denoise_loss = th.tensor(0.0).to(dist_util.dev())
            d_weight = th.tensor(0.0).to(dist_util.dev())

            # =================================== ae part ===================================
            with th.cuda.amp.autocast(dtype=th.float16,
                                      enabled=self.mp_trainer.use_amp
                                      and not self.freeze_ae):

                # apply vae
                vae_out = self.ddp_rec_model(
                    img=micro['img_to_encoder'],
                    c=micro['c'],
                    behaviour='enc_dec_wo_triplane')  # pred: (B, 3, 64, 64)
                

                if behaviour == 'diffusion_step_rec':
                    target = micro
                    pred = self.ddp_rec_model(latent=vae_out,
                                              c=micro['c'],
                                              behaviour='triplane_dec')

                    # vae reconstruction loss
                    if last_batch or not self.use_ddp:
                        vae_nelbo_loss, loss_dict = self.loss_class(pred,
                                                             target,
                                                             test_mode=False)
                    else:
                        with self.ddp_model.no_sync():  # type: ignore
                            vae_nelbo_loss, loss_dict = self.loss_class(
                                pred, target, test_mode=False)

                    last_layer = self.ddp_rec_model.module.decoder.triplane_decoder.decoder.net[  # type: ignore
                        -1].weight  # type: ignore

                    if 'image_sr' in pred:
                        vision_aided_loss = self.ddp_cano_cvD(
                            0.5 * pred['image_sr'] +
                            0.5 * th.nn.functional.interpolate(
                                pred['image_raw'],
                                size=pred['image_sr'].shape[2:],
                                mode='bilinear'),
                            for_G=True).mean()  # [B, 1] shape
                    else:
                        vision_aided_loss = self.ddp_cano_cvD(
                            pred['image_raw'], for_G=True
                        ).mean(
                        )   # [B, 1] shape

                    d_weight = calculate_adaptive_weight(
                        vae_nelbo_loss,
                        vision_aided_loss,
                        last_layer,
                        # disc_weight_max=1) * 1
                        disc_weight_max=1) * self.loss_class.opt.rec_cvD_lambda
                    # d_weight = self.loss_class.opt.rec_cvD_lambda # since decoder is fixed here. set to 0.001
                    
                    vision_aided_loss *= d_weight

                    # d_weight = self.loss_class.opt.rec_cvD_lambda
                    loss_dict.update({
                        'vision_aided_loss/G_rec':
                        vision_aided_loss,
                        'd_weight_G_rec':
                        d_weight,
                    })

                    log_rec3d_loss_dict(loss_dict)

                elif behaviour == 'diffusion_step_nvs':

                    novel_view_c = th.cat([micro['c'][1:], micro['c'][:1]])

                    pred = self.ddp_rec_model(latent=vae_out,
                                              c=novel_view_c,
                                              behaviour='triplane_dec')

                    if 'image_sr' in pred:
                        vision_aided_loss = self.ddp_nvs_cvD(
                            # pred_for_rec['image_sr'],
                            0.5 * pred['image_sr'] +
                            0.5 * th.nn.functional.interpolate(
                                pred['image_raw'],
                                size=pred['image_sr'].shape[2:],
                                mode='bilinear'),
                            for_G=True).mean()  # [B, 1] shape
                    else:
                        vision_aided_loss = self.ddp_nvs_cvD(
                            pred['image_raw'], for_G=True
                        ).mean(
                        )  # [B, 1] shape

                    d_weight = self.loss_class.opt.nvs_cvD_lambda
                    vision_aided_loss *= d_weight

                    log_rec3d_loss_dict({
                        'vision_aided_loss/G_nvs':
                        vision_aided_loss,
                    })

                    # ae_loss = th.tensor(0.0).to(dist_util.dev())

                # elif behaviour == 'diff':
                #     self.ddp_rec_model.requires_grad_(False)
                #     # assert self.ddp_rec_model.module.requires_grad == False, 'freeze ddpm_rec for pure diff step'
                else:
                    raise NotImplementedError(behaviour)
                #     assert behaviour == 'sds'

                # pred = None

                # if behaviour != 'sds': # also train diffusion
                # assert pred is not None

                # TODO, train diff and sds together, available?
                eps = vae_out[self.latent_name] 

                # if behaviour != 'sds':
                # micro_to_denoise.detach_()
                eps.requires_grad_(True) # single stage diffusion

                t, weights = self.schedule_sampler.sample(
                    eps.shape[0], dist_util.dev())
                noise = th.randn(size=vae_out.size(), device='cuda')  # note that this noise value is currently shared!

                model_kwargs = {}

                # ? 
                # or directly use SSD NeRF version?
                # get diffusion quantities for p (sgm prior) sampling scheme and reweighting for q (vae)

                # ! handle the sampling 

                # get diffusion quantities for p (sgm prior) sampling scheme and reweighting for q (vae)
                t_p, var_t_p, m_t_p, obj_weight_t_p, obj_weight_t_q, g2_t_p = \
                    diffusion.iw_quantities(args.batch_size, args.time_eps, args.iw_sample_p, args.iw_subvp_like_vp_sde)
                eps_t_p = diffusion.sample_q(vae_out, noise, var_t_p, m_t_p)

                # in case we want to train q (vae) with another batch using a different sampling scheme for times t
                if args.iw_sample_q in ['ll_uniform', 'll_iw']:
                    t_q, var_t_q, m_t_q, obj_weight_t_q, _, g2_t_q = \
                        diffusion.iw_quantities(args.batch_size, args.time_eps, args.iw_sample_q, args.iw_subvp_like_vp_sde)
                    eps_t_q = diffusion.sample_q(vae_out, noise, var_t_q, m_t_q)

                    eps_t_p = eps_t_p.detach().requires_grad_(True)
                    eps_t = th.cat([eps_t_p, eps_t_q], dim=0)
                    var_t = th.cat([var_t_p, var_t_q], dim=0)
                    t = th.cat([t_p, t_q], dim=0)
                    noise = th.cat([noise, noise], dim=0)
                else:
                    eps_t, m_t, var_t, t, g2_t = eps_t_p, m_t_p, var_t_p, t_p, g2_t_p
                
                # run the diffusion

                # mixing normal trick
                # TODO, create a new partial training_losses function 
                mixing_component = diffusion.mixing_component(eps_t, var_t, t, enabled=dae.mixed_prediction) # TODO, which should I use?
                params = utils.get_mixed_prediction(dae.mixed_prediction, pred_params, dae.mixing_logit, mixing_component)

                # nelbo loss with kl balancing




                # ! remainign parts of cross entropy in likelihook training

                cross_entropy_per_var += diffusion.cross_entropy_const(args.time_eps)
                cross_entropy = th.sum(cross_entropy_per_var, dim=[1, 2, 3])
                cross_entropy += remaining_neg_log_p_total  # for remaining scales if there is any
                all_neg_log_p = vae.decompose_eps(cross_entropy_per_var)
                all_neg_log_p.extend(remaining_neg_log_p_per_ver)  # add the remaining neg_log_p
                kl_all_list, kl_vals_per_group, kl_diag_list = utils.kl_per_group_vada(all_log_q, all_neg_log_p)


                kl_coeff = 1.0

                # ! calculate p/q loss; 
                # ? no spectral regularizer here
                # ? try adding grid_clip and sn later on.
                q_loss = th.mean(nelbo_loss) 
                p_loss = th.mean(p_objective) 

                # backpropagate q_loss for vae and update vae params, if trained
                if args.train_vae:
                    grad_scalar.scale(q_loss).backward(retain_graph=utils.different_p_q_objectives(args.iw_sample_p, args.iw_sample_q))
                    utils.average_gradients(vae.parameters(), args.distributed)
                    if args.grad_clip_max_norm > 0.:  # apply gradient clipping
                        grad_scalar.unscale_(vae_optimizer)
                        th.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=args.grad_clip_max_norm)
                    grad_scalar.step(vae_optimizer)

                # if we use different p and q objectives or are not training the vae, discard gradients and backpropagate p_loss
                if utils.different_p_q_objectives(args.iw_sample_p, args.iw_sample_q) or not args.train_vae:
                    if args.train_vae:
                        # discard current gradients computed by weighted loss for VAE
                        dae_optimizer.zero_grad()

                    # compute gradients with unweighted loss
                    grad_scalar.scale(p_loss).backward()

                # update dae parameters
                utils.average_gradients(dae.parameters(), args.distributed)
                if args.grad_clip_max_norm > 0.:         # apply gradient clipping
                    grad_scalar.unscale_(dae_optimizer)
                    th.nn.utils.clip_grad_norm_(dae.parameters(), max_norm=args.grad_clip_max_norm)
                grad_scalar.step(dae_optimizer)


                # unpack separate objectives, in case we want to train q (vae) using a different sampling scheme for times t
                if args.iw_sample_q in ['ll_uniform', 'll_iw']:
                    l2_term_p, l2_term_q = th.chunk(l2_term, chunks=2, dim=0)
                    p_objective = th.sum(obj_weight_t_p * l2_term_p, dim=[1, 2, 3])
                    # cross_entropy_per_var = obj_weight_t_q * l2_term_q
                else:
                    p_objective = th.sum(obj_weight_t_p * l2_term, dim=[1, 2, 3])
                    # cross_entropy_per_var = obj_weight_t_q * l2_term

                # print(micro_to_denoise.min(), micro_to_denoise.max())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    eps,  # x_start
                    t,
                    model_kwargs=model_kwargs,
                    return_detail=True)

                # ! DDPM step
                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                    # denoised_out = denoised_fn()
                else:
                    with self.ddp_model.no_sync():  # type: ignore
                        losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach())

                denoise_loss = (losses["loss"] * weights).mean()

                x_t = losses.pop('x_t')
                model_output = losses.pop('model_output')
                diffusion_target = losses.pop('diffusion_target')
                alpha_bar = losses.pop('alpha_bar')

                log_loss_dict(self.diffusion, t,
                              {k: v * weights
                               for k, v in losses.items()})

                # if behaviour == 'sds':
                # ! calculate sds grad, and add to the grad of

                # if 'rec' in behaviour and self.loss_class.opt.sds_lamdba > 0:  # only enable sds along with rec step
                #     w = (
                #         1 - alpha_bar**2
                #     ) / self.triplane_scaling_divider * self.loss_class.opt.sds_lamdba  # https://github.com/ashawkey/stable-dreamfusion/issues/106
                #     sds_grad = denoise_loss.clone().detach(
                #     ) * w  # * https://pytorch.org/docs/stable/generated/th.Tensor.detach.html. detach() returned Tensor share the same storage with previous one. add clone() here.

                #     # ae_loss = AddGradient.apply(latent[self.latent_name], sds_grad) # add sds_grad during backward

                #     def sds_hook(grad_to_add):

                #         def modify_grad(grad):
                #             return grad + grad_to_add  # add the sds grad to the original grad for BP

                #         return modify_grad

                #     eps[self.latent_name].register_hook(
                #         sds_hook(sds_grad))  # merge sds grad with rec/nvs ae step

                loss = vae_nelbo_loss + denoise_loss + vision_aided_loss  # caluclate loss within AMP

            # ! cvD loss

            # exit AMP before backward
            self.mp_trainer_rec.backward(loss)
            self.mp_trainer.backward(loss)

            # TODO, merge visualization with original AE
            # =================================== denoised AE log part ===================================

            if dist_util.get_rank() == 0 and self.step % 500 == 0 and behaviour != 'diff':
                with th.no_grad():
                    # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

                    # st()

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

                    # if 'image_sr' in pred:  # TODO
                    #     pred_img = th.cat(
                    #         [self.pool_512(pred_img), pred['image_sr']],
                    #         dim=-1)
                    #     gt_img = th.cat(
                    #         [self.pool_512(micro['img']), micro['img_sr']],
                    #         dim=-1)
                    #     pred_depth = self.pool_512(pred_depth)
                    #     gt_depth = self.pool_512(gt_depth)

                    gt_vis = th.cat(
                        [
                            gt_img, micro['img'], micro['img'],
                            gt_depth.repeat_interleave(3, dim=1)
                        ],
                        dim=-1)[0:1]  # TODO, fail to load depth. range [0, 1]

                    noised_ae_pred = self.ddp_rec_model(
                        img=None,
                        c=micro['c'][0:1],
                        latent=x_t[0:1] * self.
                        triplane_scaling_divider,  # TODO, how to define the scale automatically
                        behaviour=self.render_latent_behaviour)

                    # if denoised_out is None:
                    # if not self.denoised_ae:
                    # denoised_out = denoised_fn()

                    if self.diffusion.model_mean_type == ModelMeanType.START_X:
                        pred_xstart = model_output
                    else:  # * used here
                        pred_xstart = self.diffusion._predict_xstart_from_eps(
                            x_t=x_t, t=t, eps=model_output)

                    denoised_ae_pred = self.ddp_rec_model(
                        img=None,
                        c=micro['c'][0:1],
                        latent=pred_xstart[0:1] * self.
                        triplane_scaling_divider,  # TODO, how to define the scale automatically?
                        behaviour=self.render_latent_behaviour)

                    # denoised_out = denoised_ae_pred

                    # if not self.denoised_ae:
                    #     denoised_ae_pred = self.ddp_rec_model(
                    #         img=None,
                    #         c=micro['c'][0:1],
                    #         latent=denoised_out['pred_xstart'][0:1] * self.
                    #         triplane_scaling_divider,  # TODO, how to define the scale automatically
                    #         behaviour=self.render_latent_behaviour)
                    # else:
                    #     assert denoised_ae_pred is not None
                    #     denoised_ae_pred['image_raw'] = denoised_ae_pred[
                    #         'image_raw'][0:1]

                    # print(pred_img.shape)
                    # print('denoised_ae:', self.denoised_ae)

                    pred_vis = th.cat([
                        pred_img[0:1], noised_ae_pred['image_raw'][0:1],
                        denoised_ae_pred['image_raw'][0:1],
                        pred_depth[0:1].repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)  # B, 3, H, W
                    # s

                    vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
                        1, 2, 0).cpu()  # ! pred in range[-1, 1]

                    # vis = th.cat([
                    #     self.pool_128(micro['img']), x_t[:, :3, ...],
                    #     denoised_out['pred_xstart'][:, :3, ...]
                    # ],
                    #              dim=-1)[0].permute(
                    #                  1, 2, 0).cpu()  # ! pred in range[-1, 1]

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

                    th.cuda.empty_cache()
