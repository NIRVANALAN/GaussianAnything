"""
https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/diffusion/ddpm.py#L30
"""
import copy
import functools
import random
import json
import os
from pathlib import Path
from pdb import set_trace as st
from typing import Any
from click import prompt
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

from ldm.modules.encoders.modules import FrozenClipImageEmbedder, TextEmbedder, FrozenCLIPTextEmbedder, FrozenOpenCLIPImagePredictionEmbedder, FrozenOpenCLIPImageEmbedder

import dnnlib
from dnnlib.util import requires_grad
from dnnlib.util import calculate_adaptive_weight

from ..train_util_diffusion import TrainLoop3DDiffusion
from ..cvD.nvsD_canoD import TrainLoop3DcvD_nvsD_canoD

from guided_diffusion.continuous_diffusion_utils import get_mixed_prediction, different_p_q_objectives, kl_per_group_vada, kl_balancer
# from .train_util_diffusion_lsgm_noD_joint import TrainLoop3DDiffusionLSGMJointnoD  # joint diffusion and rec class
# from .controlLDM import TrainLoop3DDiffusionLSGM_Control  # joint diffusion and rec class
from .train_util_diffusion_lsgm_noD_joint import TrainLoop3DDiffusionLSGMJointnoD  # joint diffusion and rec class

__conditioning_keys__ = {
    'concat': 'c_concat',
    'crossattn': 'c_crossattn',
    'adm': 'y'
}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class TrainLoop3DDiffusionLSGM_crossattn(TrainLoop3DDiffusionLSGMJointnoD):

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
                 normalize_clip_encoding=False,
                 scale_clip_encoding=1.0,
                 cfg_dropout_prob=0.,
                 cond_key='img_sr',
                 use_eos_feature=False,
                 compile=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         denoise_model=denoise_model,
                         diffusion=diffusion,
                         sde_diffusion=sde_diffusion,
                         control_model=control_model,
                         control_key=control_key,
                         only_mid_control=only_mid_control,
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
                         resume_cldm_checkpoint=resume_cldm_checkpoint,
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
                         compile=compile,
                         **kwargs)
        self.conditioning_key = 'c_crossattn'
        self.cond_key = cond_key
        self.instantiate_cond_stage(normalize_clip_encoding,
                                    scale_clip_encoding, cfg_dropout_prob,
                                    use_eos_feature)
        requires_grad(self.rec_model, False)
        self.rec_model.eval()

        # self.normalize_clip_encoding = normalize_clip_encoding
        # self.cfg_dropout_prob = cfg_dropout_prob

    def instantiate_cond_stage(self, normalize_clip_encoding,
                               scale_clip_encoding, cfg_dropout_prob,
                               use_eos_feature):
        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/diffusion/ddpm.py#L509C1-L509C46
        # self.cond_stage_model.train = disabled_train  # type: ignore
        if self.cond_key == 'caption':
            self.cond_txt_model = TextEmbedder(dropout_prob=cfg_dropout_prob,
                                               use_eos_feature=use_eos_feature)
        elif self.cond_key == 'img':
            self.cond_img_model = FrozenOpenCLIPImagePredictionEmbedder(
                1, 1,
                FrozenOpenCLIPImageEmbedder(freeze=True,
                                            device=dist_util.dev(),
                                            init_device=dist_util.dev()))

        else:  # zero-shot Text to 3D using normalized clip latent
            self.cond_stage_model = FrozenClipImageEmbedder(
                'ViT-L/14',
                dropout_prob=cfg_dropout_prob,
                normalize_encoding=normalize_clip_encoding,
                scale_clip_encoding=scale_clip_encoding)
            self.cond_stage_model.freeze()

            self.cond_txt_model = FrozenCLIPTextEmbedder(
                dropout_prob=cfg_dropout_prob,
                scale_clip_encoding=scale_clip_encoding)
            self.cond_txt_model.freeze()

    @th.no_grad()
    def get_c_input(self,
                    batch,
                    bs=None,
                    use_text=False,
                    prompt="",
                    force_drop_ids=None,
                    *args,
                    **kwargs):
        if use_text:
            cond_inp = prompt
        else:
            if 'caption' in self.cond_key:  # support caption-img
                cond_inp = batch['caption']
            else:
                cond_inp = batch[self.cond_key]
        # if bs is not None:
        #     cond_inp = cond_inp[:bs]

        # using clip to transform control to tokens for crossattn
        control = None
        if 'caption' in self.cond_key:
            c = self.cond_txt_model(
                cond_inp,
                train=self.ddpm_model.training,
                force_drop_ids=force_drop_ids,
            )  # ! SD training text condition injection layer
            if bs is None:  # duplicated sample
                if c.shape[0] != batch['c'].shape[0]:
                    c = th.repeat_interleave(c,
                                             batch['c'].shape[0] // c.shape[0],
                                             dim=0)
            else:
                assert c.shape[0] == bs

            # st()
            # if 'img' in self.cond_key:

            # ! later
            # if 'img' in batch:
            #     control = batch['img'] + 0.02 * th.randn_like(
            #         batch['img'])  # follow SVD?

        elif self.cond_key == 'img':
            c = self.cond_img_model(cond_inp)
            # control = batch['img']
            control = batch['img'] + 0.02 * th.randn_like(
                batch['img'])  # follow SVD?

        else:  # zero shot
            if use_text:  # for test
                assert prompt != ""
                c = self.cond_txt_model.encode(prompt)  # ! for test
            else:
                cond_inp = cond_inp.to(
                    memory_format=th.contiguous_format).float()
                c = self.cond_stage_model(cond_inp)  # BS 768

        # if c.shape[0] < batch['img_to_encoder'].shape[0]:
        #     c = th.repeat_interleave(c, batch['img_to_encoder'].shape[0]//c.shape[0], dim=0)

        # return dict(c_concat=[control])
        # return  dict(c_crossattn=c, c_concat=batch['img'])
        # if self.cond_key == 'img':
        # return dict(c_crossattn=c, c_concat=control)
        return dict(c_crossattn=c)

        # else:
        #     return dict(c_crossattn=c)

        # return dict(__conditioning_keys__[self.cond_key]=)
        # return {self.conditioning_key: [c], 'c_concat': [cond_inp]}
        # return {self.conditioning_key: c, 'c_concat': [cond_inp]}

    # TODO, merge the APIs
    def apply_model_inference(self, x_noisy, t, c, model_kwargs={}):
        pred_params = self.ddp_ddpm_model(x_noisy,
                                          timesteps=t,
                                          **{
                                              **model_kwargs, 'context':
                                              c['c_crossattn'],
                                              'hint':
                                              c.get('c_concat', None)
                                          })
        return pred_params

    def apply_model(self, p_sample_batch, cond, model_kwargs={}):
        return super().apply_model(
            p_sample_batch,
            **{
                **model_kwargs, 'context': cond['c_crossattn'],
                'hint': cond.get('c_concat', None)
                # **cond,
            })

    def run_step(self, batch, step='ldm_step'):

        # if step == 'diffusion_step_rec':

        if step == 'ldm_step':
            self.ldm_train_step(batch)

        # if took_step_ddpm:
        # self._update_cldm_ema()

        self._anneal_lr()
        self.log_step()

    def run_loop(self):
        # eval camera
        camera = th.load('eval_pose.pt', map_location=dist_util.dev())

        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            # dist_util.synchronize()

            batch = next(self.data)
            self.run_step(batch, step='ldm_step')

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                try:
                    out = logger.dumpkvs()
                    # * log to tensorboard
                    for k, v in out.items():
                        self.writer.add_scalar(f'Loss/{k}', v,
                                            self.step + self.resume_step)
                except Exception as e:
                    continue # no storage

            # if self.step % self.eval_interval == 0 and self.step != 0:
            # st()
            if self.step % self.eval_interval == 0:
                # if self.step % self.eval_interval == 0:
                if dist_util.get_rank() == 0:
                    # self.eval_ddpm_sample()
                    # self.eval_cldm(use_ddim=True, unconditional_guidance_scale=7.5, prompt="") # during training, use image as condition
                    if self.cond_key == 'caption':
                        prompt_list = [
                            # "a voxelized dog",
                            # "a cute toy cat",
                            'Yellow rubber duck',
                            # "a blue plastic chair.",
                            # "An Eiffel Tower.",
                            # "A bowl of food.",
                            # "An engineer.",
                            # 'A bowl of vegetables',
                            # 'A birthday cupcake',
                            # 'A chair that looks like an avocado',
                        ]
                        self.eval_cldm(
                            # use_ddim=False,
                            # prompt="a voxelized dog",
                            # prompt="a blue plastic chair.",
                            # prompt="An Eiffel Tower.",
                            # prompt="A bowl of food.",
                            # prompt="An engineer.",
                            # prompt='A bowl of vegetables',
                            # prompt='A birthday cupcake',
                            prompt=prompt_list[random.randint(0,len(prompt_list)-1)],
                            use_train_trajectory=False,
                            camera=camera)  # fix condition bug first
                    else:
                        # i23d
                        self.eval_cldm(
                            use_train_trajectory=False,
                            camera=camera)  # fix condition bug first

                th.cuda.empty_cache()
                dist_util.synchronize()

            if self.step % self.save_interval == 0 and self.step != 0:
            # if self.step % self.save_interval == 0:
                self.save(self.mp_trainer, self.mp_trainer.model_name)
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                print('reached maximum iterations, exiting')

                # Save the last checkpoint if it wasn't already saved.
                if (self.step - 1) % self.save_interval != 0:

                    self.save(self.mp_trainer, self.mp_trainer.model_name)
                
                    # ! delete old ones
                    if (self.step - 1) % 1e5 == 0:
                        for idx in range(1,10):
                            os.remove(f"model_{self.mp_trainer.model_name}{(self.step+self.resume_step-idx*1e4):07d}.pt" )

                    # if self.sde_diffusion.args.train_vae:
                    #     self.save(self.mp_trainer_rec,
                    #               self.mp_trainer_rec.model_name)

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save(self.mp_trainer,
                      self.mp_trainer.model_name)  # rec and ddpm all fixed.
            # st()
            # self.save(self.mp_trainer_canonical_cvD, 'cvD')

    # ddpm + rec loss
    def ldm_train_step(self, batch, behaviour='cano', *args, **kwargs):
        """
        add sds grad to all ae predicted x_0 
        """

        # ! enable the gradient of both models
        requires_grad(self.ddpm_model, True)

        self.mp_trainer.zero_grad()  # !!!!

        if 'img' in batch:
            batch_size = batch['img'].shape[0]
        else:
            batch_size = len(batch['caption'])

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v
                for k, v in batch.items()
            }

            # =================================== ae part ===================================
            with th.cuda.amp.autocast(dtype=th.bfloat16,
                                      enabled=self.mp_trainer.use_amp):

                loss = th.tensor(0.).to(dist_util.dev())

                if 'latent' in micro:
                    vae_out = {self.latent_name: micro['latent']}
                else:
                    vae_out = self.ddp_rec_model(
                        img=micro['img_to_encoder'],
                        c=micro['c'],
                        behaviour='encoder_vae',
                    )  # pred: (B, 3, 64, 64)

                eps = vae_out[self.latent_name] / self.triplane_scaling_divider
                # eps = vae_out.pop(self.latent_name)

                if 'bg_plane' in vae_out:
                    eps = th.cat((eps, vae_out['bg_plane']),
                                 dim=1)  # include background, B 12+4 32 32

                p_sample_batch = self.prepare_ddpm(eps)
                cond = self.get_c_input(micro, bs=eps.shape[0])

                # ! running diffusion forward
                ddpm_ret = self.apply_model(p_sample_batch, cond)
                if self.sde_diffusion.args.p_rendering_loss:

                    target = micro
                    pred = self.ddp_rec_model(
                        # latent=vae_out,
                        latent={
                            # **vae_out,
                            self.latent_name: ddpm_ret['pred_x0_p'],
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
                    loss = p_vae_recon_loss + ddpm_ret[
                        'p_eps_objective']  # TODO, add obj_weight_t_p?
                else:
                    loss = ddpm_ret['p_eps_objective'].mean()

                # =====================================================================

            self.mp_trainer.backward(loss)  # joint gradient descent

        # update ddpm accordingly
        self.mp_trainer.optimize(self.opt)

        if dist_util.get_rank() == 0 and self.step % 500 == 0:
            self.log_control_images(vae_out, p_sample_batch, micro, ddpm_ret)

    @th.inference_mode()
    def log_control_images(self, vae_out, p_sample_batch, micro, ddpm_ret):

        eps_t_p, t_p, logsnr_p = (p_sample_batch[k] for k in (
            'eps_t_p',
            't_p',
            'logsnr_p',
        ))
        pred_eps_p = ddpm_ret['pred_eps_p']

        if 'posterior' in vae_out:
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
        if 'img' in micro:
            gt_img = micro['img']
        else:
            gt_img = th.zeros_like(pred['image_raw'])

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
                gt_img,
                # self.pool_128(hint),
                # gt_img,
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

        if 'img' in micro:
            vis = th.cat([gt_vis, pred_vis],
                         dim=-2)[0].permute(1, 2,
                                            0).cpu()  # ! pred in range[-1, 1]
        else:
            vis = pred_vis[0].permute(1, 2, 0).cpu()

        # vis_grid = torchvision.utils.make_grid(vis) # HWC
        vis = vis.numpy() * 127.5 + 127.5
        vis = vis.clip(0, 255).astype(np.uint8)
        Image.fromarray(vis).save(
            f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t_p[0].item():3}.jpg'
        )

        # if self.cond_key == 'caption':
        #     with open(f'{logger.get_dir()}/{self.step+self.resume_step}caption_{t_p[0].item():3}.txt', 'w') as f:
        #         f.write(micro['caption'][0])

        print(
            'log denoised vis to: ',
            f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t_p[0].item():3}.jpg'
        )

        th.cuda.empty_cache()

    @th.inference_mode()
    def eval_cldm(
        self,
        prompt="",
        use_ddim=False,
        unconditional_guidance_scale=1.0,
        save_img=False,
        use_train_trajectory=False,
        camera=None,
        num_samples=1,
        num_instances=1,
    ):
        self.ddpm_model.eval()

        args = dnnlib.EasyDict(
            dict(
                batch_size=1,
                image_size=self.diffusion_input_size,
                denoise_in_channels=self.rec_model.decoder.triplane_decoder.
                out_chans,  # type: ignore
                clip_denoised=False,
                class_cond=False,
                use_ddim=use_ddim))

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

        # use the first frame as the condition now
        extra_kwargs = {}

        uc = None
        if args.use_ddim:
            if unconditional_guidance_scale != 1.0:
                uc = self.get_c_input(
                    {self.cond_key: 'None'},
                    use_text=True,
                    prompt="None",
                    bs=1,  # TODO, support BS>1 later
                    force_drop_ids=np.array(
                        [  # ! make sure using dropped tokens
                            1
                        ]))  # use specific prompt for debug
            extra_kwargs.update(
                dict(
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc,  # TODO
                    # {
                    #     k : unconditional_guidance_scale
                    #     for k in cond.keys()
                    # }
                ))

        # hint = th.cat(cond['c_concat'], 1)

        # record cond images
        # broadcast to args.batch_size

        for instance in range(num_instances):

            if self.cond_key == 'caption':
                if camera is not None:
                    batch = {'c': camera.clone()}
            else:
                if use_train_trajectory:
                    batch = next(iter(self.data))
                else:
                    try:
                        batch = next(self.eval_data)
                    except Exception as e:
                        self.eval_data = iter(self.eval_data)
                        batch = next(self.eval_data)

                if camera is not None:
                    batch['c'] = camera.clone()

            # ! generate new samples

            novel_view_cond = {
                k:
                v[0:1].to(dist_util.dev())
                if isinstance(v, th.Tensor) else v[0:1]
                # micro['img'].shape[0], 0)
                for k, v in batch.items()
            }

            cond = self.get_c_input(
                novel_view_cond, use_text=prompt != "",
                prompt=prompt)  # use specific prompt for debug

            cond = {
                k: cond_v.repeat_interleave(args.batch_size, 0)
                for k, cond_v in cond.items()
                # if k == self.conditioning_key
            }

            if self.cond_key == 'caption':
                if prompt != '':
                    with open(
                            f'{logger.get_dir()}/triplane_{self.step+self.resume_step}_{instance}_caption.txt',
                            'w') as f:
                        f.write(prompt)
                else:
                    with open(
                            f'{logger.get_dir()}/triplane_{self.step+self.resume_step}_{instance}_caption.txt',
                            'w') as f:
                        try:
                            f.write(novel_view_cond['caption'][0])
                        except Exception as e:
                            pass

            elif self.cond_key == 'img':
                torchvision.utils.save_image(
                    cond['c_concat'],
                    f'{logger.get_dir()}/{self.step + self.resume_step}_{instance}_cond.jpg',
                    normalize=True,
                    value_range=(-1, 1))

            # continue

            for i in range(num_samples):
                triplane_sample = sample_fn(
                    self,
                    (
                        args.batch_size,
                        self.ddpm_model.in_channels
                        if not self.ddpm_model.roll_out else 3 *
                        self.ddpm_model.in_channels,  # type: ignore
                        self.diffusion_input_size,
                        self.diffusion_input_size),
                    cond=cond,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    # mixing_normal=True,  # !
                    mixing_normal=self.ddpm_model.mixed_prediction,  # !
                    device=dist_util.dev(),
                    **extra_kwargs)
                th.cuda.empty_cache()

                self.render_video_given_triplane(
                    triplane_sample,
                    self.rec_model,  # compatible with join_model
                    name_prefix=
                    f'{self.step + self.resume_step}_{instance}_{i}',
                    save_img=save_img,
                    render_reference=batch,
                    export_mesh=False)

                # save gt
                # video_out = imageio.get_writer(
                #     f'{logger.get_dir()}/triplane_{self.step + self.resume_step}_{i}_reference.mp4',
                #     mode='I',
                #     fps=15,
                #     codec='libx264')

                # for j in range(batch['img'].shape[0]
                #             ):  # ! currently only export one plane at a time
                #     cpu_gt = batch['img'][j].cpu().permute(1,2,0).numpy()
                #     cpu_gt = (cpu_gt*127.5)+127.5
                #     video_out.append_data(cpu_gt.astype(np.uint8))

                # video_out.close()
                # del video_out

            # del triplane_sample
            # th.cuda.empty_cache()

        self.ddpm_model.train()


class TrainLoop3DDiffusionLSGM_crossattn_controlNet(
        TrainLoop3DDiffusionLSGM_crossattn):

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
                 normalize_clip_encoding=False,
                 scale_clip_encoding=1,
                 cfg_dropout_prob=0,
                 cond_key='img_sr',
                 use_eos_feature=False,
                 compile=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         denoise_model=denoise_model,
                         diffusion=diffusion,
                         sde_diffusion=sde_diffusion,
                         control_model=control_model,
                         control_key=control_key,
                         only_mid_control=only_mid_control,
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
                         resume_cldm_checkpoint=resume_cldm_checkpoint,
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
                         normalize_clip_encoding=normalize_clip_encoding,
                         scale_clip_encoding=scale_clip_encoding,
                         cfg_dropout_prob=cfg_dropout_prob,
                         cond_key=cond_key,
                         use_eos_feature=use_eos_feature,
                         compile=compile,
                         **kwargs)

        # st()
        self.control_model = control_model
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.sd_locked = True
        self._setup_control_model()

    def _setup_control_model(self):

        requires_grad(self.rec_model, False)
        requires_grad(self.ddpm_model, False)

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

        requires_grad(self.ddp_control_model, True)

        # ! load trainable copy
        # TODO
        # st()
        try:
            logger.log(f"load pretrained controlnet, not trainable copy.")
            self._load_and_sync_parameters(model=self.control_model,
                                           model_name='cldm',
                                           resume_checkpoint=self.resume_cldm_checkpoint,
                                           )  # if available
        except:
            logger.log(f"load trainable copy to controlnet")
            model_state_dict = self.control_model.state_dict()
            for k, v in self.ddpm_model.state_dict().items():
                if k in model_state_dict.keys() and v.size(
                ) == model_state_dict[k].size():
                    model_state_dict[k] = v

            self.control_model.load_state_dict(model_state_dict)

            # self._load_and_sync_parameters(
            #     model=self.control_model,
            #     model_name='ddpm')  # load pre-trained SD

        cldm_param = [{
            'name': 'cldm.parameters()',
            'params': self.control_model.parameters(),
        }]
        # if self.sde_diffusion.args.unfix_logit:
        #     self.ddpm_model.mixing_logit.requires_grad_(True)
        #     cldm_param.append({
        #         'name': 'mixing_logit',
        #         'params': self.ddpm_model.mixing_logit,
        #     })

        self.opt_cldm = AdamW(cldm_param,
                              lr=self.lr,
                              weight_decay=self.weight_decay)
        if self.sd_locked:
            del self.opt
            del self.mp_trainer

    # add control during inference
    def apply_model_inference(self, x_noisy, t, c, model_kwargs={}):

        control = self.ddp_control_model(
            x=x_noisy,
            #  hint=th.cat(c['c_concat'], 1),
            hint=c['c_concat'],
            timesteps=t,
            context=None)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        model_kwargs.update({'control': control})

        return super().apply_model_inference(x_noisy, t, c, model_kwargs)

    def apply_control_model(self, p_sample_batch, cond):
        x_noisy, t, = (p_sample_batch[k] for k in ('eps_t_p', 't_p'))

        control = self.ddp_control_model(
            x=x_noisy,
            #  hint=th.cat(cond['c_concat'], 1),
            hint=cond['c_concat'],
            timesteps=t,
            context=None)

        control = [c * scale for c, scale in zip(control, self.control_scales)]
        return control

    def apply_model(self, p_sample_batch, cond, model_kwargs={}):

        control = self.apply_control_model(p_sample_batch,
                                           cond)  # len(control): 13
        model_kwargs.update({'control': control})

        return super().apply_model(p_sample_batch, cond, model_kwargs)

    # cldm loss
    def ldm_train_step(self, batch, behaviour='cano', *args, **kwargs):
        """
        add sds grad to all ae predicted x_0 
        """

        # ! enable the gradient of both models
        requires_grad(self.ddp_control_model, True)
        self.mp_cldm_trainer.zero_grad()  # !!!!

        if 'img' in batch:
            batch_size = batch['img'].shape[0]
        else:
            batch_size = len(batch['caption'])

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

                if 'latent' in micro:
                    vae_out = {self.latent_name: micro['latent']}
                else:
                    vae_out = self.ddp_rec_model(
                        img=micro['img_to_encoder'],
                        c=micro['c'],
                        behaviour='encoder_vae',
                    )  # pred: (B, 3, 64, 64)

                eps = vae_out[self.latent_name] / self.triplane_scaling_divider
                # eps = vae_out.pop(self.latent_name)

                if 'bg_plane' in vae_out:
                    eps = th.cat((eps, vae_out['bg_plane']),
                                 dim=1)  # include background, B 12+4 32 32

                p_sample_batch = self.prepare_ddpm(eps)
                cond = self.get_c_input(micro, bs=eps.shape[0])

                # ! running diffusion forward
                ddpm_ret = self.apply_model(p_sample_batch, cond)
                if self.sde_diffusion.args.p_rendering_loss:

                    target = micro
                    pred = self.ddp_rec_model(
                        # latent=vae_out,
                        latent={
                            # **vae_out,
                            self.latent_name: ddpm_ret['pred_x0_p'],
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
                    loss = p_vae_recon_loss + ddpm_ret[
                        'p_eps_objective']  # TODO, add obj_weight_t_p?
                else:
                    loss = ddpm_ret['p_eps_objective'].mean()

                # =====================================================================

            self.mp_cldm_trainer.backward(loss)  # joint gradient descent
            # p self.control_model.input_hint_block[0].bias

        # update ddpm accordingly
        self.mp_cldm_trainer.optimize(self.opt_cldm)

        if dist_util.get_rank() == 0 and self.step % 500 == 0:
            self.log_control_images(vae_out, p_sample_batch, micro, ddpm_ret)

    def run_loop(self):
        # eval camera
        camera = th.load('eval_pose.pt', map_location=dist_util.dev())

        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            # dist_util.synchronize()

            batch = next(self.data)
            self.run_step(batch, step='ldm_step')

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
                    # self.eval_ddpm_sample()
                    # self.eval_cldm(use_ddim=True, unconditional_guidance_scale=7.5, prompt="") # during training, use image as condition
                    if self.cond_key == 'caption':
                        self.eval_cldm(
                            use_ddim=False,
                            prompt="a voxelized dog",
                            # prompt="a blue plastic chair.",
                            use_train_trajectory=False,
                            camera=camera)  # fix condition bug first
                    else:
                        pass  # TODO
                        # self.eval_cldm(use_ddim=False,
                        #             prompt="",
                        #             use_train_trajectory=False,
                        #             camera=camera)  # fix condition bug first
                    # if self.sde_diffusion.args.train_vae:
                    #     self.eval_loop()

                th.cuda.empty_cache()
                dist_util.synchronize()

            if self.step % self.save_interval == 0 and self.step != 0:
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

                    self.save(self.mp_trainer, self.mp_trainer.model_name)
                    # if self.sde_diffusion.args.train_vae:
                    #     self.save(self.mp_trainer_rec,
                    #               self.mp_trainer_rec.model_name)

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save(self.mp_trainer, self.mp_trainer.model_name)
            # self.save(self.mp_trainer,
            #           self.mp_trainer.model_name)  # rec and ddpm all fixed.
            # st()
            # self.save(self.mp_trainer_canonical_cvD, 'cvD')
