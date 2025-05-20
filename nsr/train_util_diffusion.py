import copy
import functools
import json
import os
from pathlib import Path
from pdb import set_trace as st
# from PIL import Image
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

from guided_diffusion.gaussian_diffusion import _extract_into_tensor
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

import dnnlib
from huggingface_hub import hf_hub_download

# AMP
# from accelerate import Accelerator

# from ..guided_diffusion.train_util import TrainLoop

# use_amp = False
# use_amp = True


class TrainLoopDiffusionWithRec(TrainLoop):
    """an interface with rec_model required apis
    """

    def __init__(
        self,
        *,
        model,
        diffusion,
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
        schedule_sampler=None,
        model_name='ddpm',
        train_vae=True,
        **kwargs,
    ):
        super().__init__(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=batch_size,
            microbatch=microbatch,
            lr=lr,
            ema_rate=ema_rate,
            log_interval=log_interval,
            save_interval=save_interval,
            resume_checkpoint=resume_checkpoint,
            use_fp16=use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=weight_decay,
            lr_anneal_steps=lr_anneal_steps,
            use_amp=use_amp,
            model_name=model_name,
            train_vae=train_vae,
            **kwargs,
        )

        self.latent_name = 'latent_normalized'  # normalized triplane latent
        self.diffusion_input_size = diffusion_input_size
        self.render_latent_behaviour = 'triplane_dec'  # directly render using triplane operations

        self.loss_class = loss_class
        # self.rec_model = rec_model
        self.eval_interval = eval_interval
        self.eval_data = eval_data
        self.iterations = iterations
        # self.triplane_std = 10
        self.triplane_scaling_divider = triplane_scaling_divider

        if dist_util.get_rank() == 0:
            self.writer = SummaryWriter(log_dir=f'{logger.get_dir()}/runs')

    # def _init_optim_groups(self, rec_model):
    #     """for initializing the reconstruction model.
    #     """
    #     kwargs = self.kwargs
    #     optim_groups = [
    #         # vit encoder
    #         {
    #             'name': 'vit_encoder',
    #             'params': rec_model.encoder.parameters(),
    #             'lr': kwargs['encoder_lr'],
    #             'weight_decay': kwargs['encoder_weight_decay']
    #         },
    #         # vit decoder
    #         {
    #             'name': 'vit_decoder',
    #             'params': rec_model.decoder.vit_decoder.parameters(),
    #             'lr': kwargs['vit_decoder_lr'],
    #             'weight_decay': kwargs['vit_decoder_wd']
    #         },
    #         {
    #             'name': 'vit_decoder_pred',
    #             'params': rec_model.decoder.decoder_pred.parameters(),
    #             'lr': kwargs['vit_decoder_lr'],
    #             # 'weight_decay': 0
    #             'weight_decay': kwargs['vit_decoder_wd']
    #         },

    #         # triplane decoder
    #         {
    #             'name': 'triplane_decoder',
    #             'params': rec_model.decoder.triplane_decoder.parameters(),
    #             'lr': kwargs['triplane_decoder_lr'],
    #             # 'weight_decay': self.weight_decay
    #         },
    #     ]

    #     if rec_model.decoder.superresolution is not None:
    #         optim_groups.append({
    #             'name':
    #             'triplane_decoder_superresolution',
    #             'params':
    #             rec_model.decoder.superresolution.parameters(),
    #             'lr':
    #             kwargs['super_resolution_lr'],
    #         })

    #     return optim_groups

    def _init_optim_groups(self, rec_model, freeze_decoder=False):
        """for initializing the reconstruction model; fixing decoder part.
        """
        kwargs = self.kwargs
        optim_groups = [
            # vit encoder
            {
                'name': 'vit_encoder',
                'params': rec_model.encoder.parameters(),
                'lr': kwargs['encoder_lr'],
                'weight_decay': kwargs['encoder_weight_decay']
            },
        ]

        if not freeze_decoder:
            optim_groups += [
                # vit decoder
                {
                    'name': 'vit_decoder',
                    'params': rec_model.decoder.vit_decoder.parameters(),
                    'lr': kwargs['vit_decoder_lr'],
                    'weight_decay': kwargs['vit_decoder_wd']
                },
                {
                    'name': 'vit_decoder_pred',
                    'params': rec_model.decoder.decoder_pred.parameters(),
                    'lr': kwargs['vit_decoder_lr'],
                    # 'weight_decay': 0
                    'weight_decay': kwargs['vit_decoder_wd']
                },

                # triplane decoder
                {
                    'name': 'triplane_decoder',
                    'params': rec_model.decoder.triplane_decoder.parameters(),
                    'lr': kwargs['triplane_decoder_lr'],
                    # 'weight_decay': self.weight_decay
                },
            ]

            if rec_model.decoder.superresolution is not None:
                optim_groups.append({
                    'name':
                    'triplane_decoder_superresolution',
                    'params':
                    rec_model.decoder.superresolution.parameters(),
                    'lr':
                    kwargs['super_resolution_lr'],
                })

        return optim_groups

    # @th.no_grad()
    # # def eval_loop(self, c_list:list):
    # def eval_novelview_loop(self, rec_model):
    #     # novel view synthesis given evaluation camera trajectory
    #     video_out = imageio.get_writer(
    #         f'{logger.get_dir()}/video_novelview_{self.step+self.resume_step}.mp4',
    #         mode='I',
    #         fps=60,
    #         codec='libx264')

    #     all_loss_dict = []
    #     novel_view_micro = {}

    #     # for i in range(0, len(c_list), 1): # TODO, larger batch size for eval
    #     for i, batch in enumerate(tqdm(self.eval_data)):
    #         # for i in range(0, 8, self.microbatch):
    #         # c = c_list[i].to(dist_util.dev()).reshape(1, -1)
    #         micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

    #         # st()

    #         if i == 0:
    #             novel_view_micro = {
    #                 'img_to_encoder': micro['img_to_encoder'][0:1]
    #             }

    #             latent = rec_model(img=novel_view_micro['img_to_encoder'],
    #                                     behaviour='enc_dec_wo_triplane')

    #         # else:
    #         #     # if novel_view_micro['c'].shape[0] < micro['img'].shape[0]:
    #         #     novel_view_micro = {
    #         #         k:
    #         #         v[0:1].to(dist_util.dev()).repeat_interleave(
    #         #             micro['img'].shape[0], 0)
    #         #         for k, v in novel_view_micro.items()
    #         #     }

    #         # pred = rec_model(img=novel_view_micro['img_to_encoder'].repeat_interleave(micro['img'].shape[0], 0),
    #         #                  c=micro['c'])  # pred: (B, 3, 64, 64)

    #         # ! only render
    #         pred = rec_model(
    #             latent={
    #                 'latent_after_vit': latent['latent_after_vit'].repeat_interleave(micro['img'].shape[0], 0)
    #             },
    #             c=micro['c'],  # predict novel view here
    #             behaviour='triplane_dec',
    #         )

    #         # target = {
    #         #     'img': micro['img'],
    #         #     'depth': micro['depth'],
    #         #     'depth_mask': micro['depth_mask']
    #         # }
    #         # targe

    #         _, loss_dict = self.loss_class(pred, micro, test_mode=True)
    #         all_loss_dict.append(loss_dict)

    #         # ! move to other places, add tensorboard

    #         # pred_vis = th.cat([
    #         #     pred['image_raw'],
    #         #     -pred['image_depth'].repeat_interleave(3, dim=1)
    #         # ],
    #         #                   dim=-1)

    #         # normalize depth
    #         # if True:
    #         pred_depth = pred['image_depth']
    #         pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
    #                                                         pred_depth.min())
    #         if 'image_sr' in pred:
    #             if pred['image_sr'].shape[-1] == 512:
    #                 pred_vis = th.cat([
    #                     micro['img_sr'],
    #                     self.pool_512(pred['image_raw']), pred['image_sr'],
    #                     self.pool_512(pred_depth).repeat_interleave(3, dim=1)
    #                 ],
    #                                   dim=-1)
    #             else:
    #                 assert pred['image_sr'].shape[-1] == 128
    #                 pred_vis = th.cat([
    #                     micro['img_sr'],
    #                     self.pool_128(pred['image_raw']), pred['image_sr'],
    #                     self.pool_128(pred_depth).repeat_interleave(3, dim=1)
    #                 ],
    #                                   dim=-1)
    #         else:
    #             pred_vis = th.cat([
    #                 self.pool_128(micro['img']),
    #                 self.pool_128(pred['image_raw']),
    #                 self.pool_128(pred_depth).repeat_interleave(3, dim=1)
    #             ],
    #                               dim=-1)  # B, 3, H, W

    #         vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
    #         vis = vis * 127.5 + 127.5
    #         vis = vis.clip(0, 255).astype(np.uint8)

    #         for j in range(vis.shape[0]):
    #             video_out.append_data(vis[j])

    #     video_out.close()

    #     del video_out, vis, pred_vis, pred
    #     th.cuda.empty_cache()

    #     val_scores_for_logging = calc_average_loss(all_loss_dict)
    #     with open(os.path.join(logger.get_dir(), 'scores_novelview.json'),
    #               'a') as f:
    #         json.dump({'step': self.step, **val_scores_for_logging}, f)

    #     # * log to tensorboard
    #     for k, v in val_scores_for_logging.items():
    #         self.writer.add_scalar(f'Eval/NovelView/{k}', v,
    #                                self.step + self.resume_step)

    @th.no_grad()
    # def eval_loop(self, c_list:list):
    def eval_novelview_loop(self, rec_model):
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
                    k:
                    v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in batch.items()
                }
            else:
                # if novel_view_micro['c'].shape[0] < micro['img'].shape[0]:
                novel_view_micro = {
                    k:
                    v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in novel_view_micro.items()
                }

            pred = rec_model(img=novel_view_micro['img_to_encoder'],
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
                # pred_vis = th.cat([
                #     self.pool_64(micro['img']), pred['image_raw'],
                #     pred_depth.repeat_interleave(3, dim=1)
                # ],
                #                   dim=-1)  # B, 3, H, W

                pred_vis = th.cat([
                    self.pool_128(micro['img']),
                    self.pool_128(pred['image_raw']),
                    self.pool_128(pred_depth).repeat_interleave(3, dim=1)
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
        del video_out
        # del pred_vis
        # del pred

        th.cuda.empty_cache()

    @th.no_grad()
    def eval_loop(self, rec_model):
        # novel view synthesis given evaluation camera trajectory
        video_out = imageio.get_writer(
            f'{logger.get_dir()}/video_{self.step+self.resume_step}.mp4',
            mode='I',
            fps=60,
            codec='libx264')
        all_loss_dict = []

        # for i in range(0, len(c_list), 1): # TODO, larger batch size for eval
        for i, batch in enumerate(tqdm(self.eval_data)):
            # for i in range(0, 8, self.microbatch):
            # c = c_list[i].to(dist_util.dev()).reshape(1, -1)
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            # pred = self.model(img=micro['img_to_encoder'],
            #                   c=micro['c'])  # pred: (B, 3, 64, 64)

            # pred of rec model
            pred = rec_model(img=micro['img_to_encoder'],
                             c=micro['c'])  # pred: (B, 3, 64, 64)

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
                else:
                    assert pred['image_sr'].shape[-1] == 128
                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_128(pred['image_raw']), pred['image_sr'],
                        self.pool_128(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)
            else:
                pred_vis = th.cat([
                    self.pool_128(micro['img']),
                    self.pool_128(pred['image_raw']),
                    self.pool_128(pred_depth).repeat_interleave(3, dim=1)
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

        del video_out, vis, pred_vis, pred
        th.cuda.empty_cache()
        self.eval_novelview_loop(rec_model)

    def save(self, mp_trainer=None, model_name='ddpm'):
        if mp_trainer is None:
            mp_trainer = self.mp_trainer

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

        # save_checkpoint(0, self.mp_trainer_ddpm.master_params)
        try:
            save_checkpoint(0, mp_trainer.master_params)
            if model_name == 'ddpm':
                for rate, params in zip(self.ema_rate, self.ema_params):
                    save_checkpoint(rate, params)
        except Exception as e:
            logger.log(e)

        th.cuda.empty_cache()
        dist_util.synchronize()

    def _load_and_sync_parameters(self,
                                  model=None,
                                  model_name='ddpm',
                                  resume_checkpoint=None):
        hf_loading = 'yslan/GaussianAnything' in self.resume_checkpoint and 'home' not in self.resume_checkpoint 

        if not hf_loading and resume_checkpoint is None:
            resume_checkpoint, self.resume_step = find_resume_checkpoint(
                self.resume_checkpoint, model_name) or self.resume_checkpoint

        if model is None:
            model = self.model

        if hf_loading or resume_checkpoint and Path(resume_checkpoint).exists():
            if dist_util.get_rank() == 0:
                # ! rank 0 return will cause all other ranks to hang
                logger.log(
                    f"loading model from checkpoint: {resume_checkpoint}...")
                map_location = {
                    'cuda:%d' % 0: 'cuda:%d' % dist_util.get_rank()
                }  # configure map_location properly

                if hf_loading:
                    logger.log(f'mark {model_name} loading from hugging face')
                else:
                    logger.log(f'mark {model_name} loading ')

                if hf_loading:
                    hf_ckpt = self.resume_checkpoint.split('/')
                    repo_id = '/'.join(hf_ckpt[:2])
                    file_name = '/'.join(hf_ckpt[2:])
                    model_path = hf_hub_download(repo_id=repo_id, 
                            filename=file_name)

                    resume_state_dict = dist_util.load_state_dict(
                        model_path, map_location=map_location)

                else:
                    resume_state_dict = dist_util.load_state_dict(
                        resume_checkpoint, map_location=map_location)

                # logger.log(f'mark {model_name} loading finished')

                model_state_dict = model.state_dict()

                for k, v in resume_state_dict.items():
                    if k in model_state_dict.keys() and v.size(
                    ) == model_state_dict[k].size():
                        model_state_dict[k] = v

                    else:
                        print(
                            '!!!! ignore key: ',
                            k,
                            ": ",
                            v.size(),
                        )
                        if k in model_state_dict:
                            print('shape in model: ',
                                  model_state_dict[k].size())
                        else:
                            print(k, ' not in model')

                model.load_state_dict(model_state_dict, strict=True)
                del model_state_dict
        else:
            logger.log(f'{resume_checkpoint} not found.')
        # print(resume_checkpoint)

        if dist_util.get_world_size() > 1:
            dist_util.sync_params(model.parameters())
            # dist_util.sync_params(model.named_parameters())
            print(f'synced {model_name} params')

    @th.inference_mode()
    def apply_model_inference(self,
                              x_noisy,
                              t,
                              c=None,
                              model_kwargs={}):  # compatiable api
        # pred_params = self.ddp_model(x_noisy, t, c=c, model_kwargs=model_kwargs)
        pred_params = self.ddp_model(x_noisy, t,
                                     **model_kwargs)  # unconditional model
        return pred_params

    @th.inference_mode()
    def eval_ddpm_sample(self, rec_model, **kwargs):  # , ddpm_model=None):
        # rec_model.eval()
        # self.ddpm_model.eval()
        self.model.eval()

        # if ddpm_model is None:
        #     ddpm_model = self.ddp_model

        args = dnnlib.EasyDict(
            dict(
                batch_size=1,
                # image_size=224,
                image_size=self.diffusion_input_size,
                # ddpm_image_size=224,
                # denoise_in_channels=self.ddp_rec_model.module.decoder.triplane_decoder.out_chans,  # type: ignore
                denoise_in_channels=self.ddpm_model.
                in_channels,  # type: ignore
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

        # for i in range(2):
        for i in range(1):
            triplane_sample = sample_fn(
                # self.ddp_model,
                self,
                (args.batch_size, args.denoise_in_channels,
                 self.diffusion_input_size, self.diffusion_input_size),
                clip_denoised=args.clip_denoised,
                # model_kwargs=model_kwargs,
                mixing_normal=True,  # !
                device=dist_util.dev(),
                # model_kwargs=model_kwargs,
                **model_kwargs)
            th.cuda.empty_cache()
            self.render_video_given_triplane(
                triplane_sample,
                rec_model,
                name_prefix=f'{self.step + self.resume_step}_{i}')
            th.cuda.empty_cache()

        # rec_model.train()
        # self.ddpm_model.train()
        # ddpm_model.train()
        self.model.train()


    @th.inference_mode()
    def render_video_noise_schedule(self, name_prefix='0'):

        # planes *= self.triplane_std # denormalize for rendering

        video_out = imageio.get_writer(
            f'{logger.get_dir()}/triplane_visnoise_{name_prefix}.mp4',
            mode='I',
            fps=30,
            codec='libx264')

        for i, batch in enumerate(tqdm(self.eval_data)):
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            if i % 10 != 0:
                continue

            # ========= novel view plane settings ====
            if i == 0:
                novel_view_micro = {
                    k:
                    v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in batch.items()
                }
            else:
                # if novel_view_micro['c'].shape[0] < micro['img'].shape[0]:
                novel_view_micro = {
                    k:
                    v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in novel_view_micro.items()
                }

            latent = self.ddp_rec_model(
                img=novel_view_micro['img_to_encoder'],
                c=micro['c'])[self.latent_name]  # pred: (B, 3, 64, 64)

            x_start = latent / self.triplane_scaling_divider  # normalize std to 1
            # x_start = latent

            all_pred_vis = []
            # for t in th.range(0,
            #                   4001,
            #                   500,
            #                   dtype=th.long,
            #                   device=dist_util.dev()):  # cosine 4k steps
            for t in th.range(0,
                              1001,
                              125,
                              dtype=th.long,
                              device=dist_util.dev()):  # cosine 4k steps

                # ========= add noise according to t
                noise = th.randn_like(x_start)  # x_start is the x0 image
                x_t = self.diffusion.q_sample(
                    x_start, t, noise=noise
                )  # * add noise according to predefined schedule
                planes_x_t = (x_t * self.triplane_scaling_divider).clamp(
                    -50, 50)  # de-scaling noised x_t

                # planes_x_t = (x_t * 1).clamp(
                #     -50, 50)  # de-scaling noised x_t

                # ===== visualize
                pred = self.ddp_rec_model(
                    img=None,
                    c=micro['c'],
                    latent=planes_x_t,
                    behaviour=self.render_latent_behaviour
                )  # pred: (B, 3, 64, 64)

                # pred_depth = pred['image_depth']
                # pred_depth = (pred_depth - pred_depth.min()) / (
                #     pred_depth.max() - pred_depth.min())
                # pred_vis = th.cat([
                #     # self.pool_128(micro['img']),
                #     pred['image_raw'],
                # ],
                #                   dim=-1)  # B, 3, H, W
                pred_vis = pred['image_raw']

                all_pred_vis.append(pred_vis)
                # TODO, make grid

            all_pred_vis = torchvision.utils.make_grid(
                th.cat(all_pred_vis, 0),
                nrow=len(all_pred_vis),
                normalize=True,
                value_range=(-1, 1),
                scale_each=True)  # normalized to [-1,1]

            vis = all_pred_vis.permute(1, 2, 0).cpu().numpy()  # H W 3

            vis = (vis * 255).clip(0, 255).astype(np.uint8)

            video_out.append_data(vis)

        video_out.close()
        print('logged video to: ',
              f'{logger.get_dir()}/triplane_visnoise_{name_prefix}.mp4')

        th.cuda.empty_cache()

    @th.inference_mode()
    def plot_noise_nsr_curve(self, name_prefix='0'):
        # planes *= self.triplane_std # denormalize for rendering

        for i, batch in enumerate(tqdm(self.eval_data)):
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            if i % 10 != 0:
                continue

            # if i == 0:
            latent = self.ddp_rec_model(
                img=micro['img_to_encoder'],
                c=micro['c'],
                behaviour='enc_dec_wo_triplane')  # pred: (B, 3, 64, 64)

            x_start = latent[
                self.
                latent_name] / self.triplane_scaling_divider  # normalize std to 1

            snr_list = []
            snr_wo_data_list = []
            xt_mean = []
            xt_std = []

            for t in th.range(0,
                              1001,
                              5,
                              dtype=th.long,
                              device=dist_util.dev()):  # cosine 4k steps

                # ========= add noise according to t
                noise = th.randn_like(x_start)  # x_start is the x0 image

                beta_t = _extract_into_tensor(
                    self.diffusion.sqrt_alphas_cumprod, t, x_start.shape)
                one_minus_beta_t = _extract_into_tensor(
                    self.diffusion.sqrt_one_minus_alphas_cumprod, t,
                    x_start.shape)

                signal_t = beta_t * x_start
                noise_t = one_minus_beta_t * noise

                x_t = signal_t + noise_t

                snr = signal_t / (noise_t + 1e-6)
                snr_wo_data = beta_t / (one_minus_beta_t + 1e-6)

                snr_list.append(abs(snr).mean().cpu().numpy())
                snr_wo_data_list.append(abs(snr_wo_data).mean().cpu().numpy())
                xt_mean.append(x_t.mean().cpu().numpy())
                xt_std.append(x_t.std().cpu().numpy())

            print('xt_mean', xt_mean)
            print('xt_std', xt_std)
            print('snr', snr_list)

            th.save(
                {
                    'xt_mean': xt_mean,
                    'xt_std': xt_std,
                    'snr': snr_list,
                    'snr_wo_data': snr_wo_data_list,
                },
                Path(logger.get_dir()) / f'snr_{i}.pt')

        th.cuda.empty_cache()


# a legacy class for direct diffusion training, not joint.
class TrainLoop3DDiffusion(TrainLoopDiffusionWithRec):

    def __init__(
            self,
            *,
            # model,
            rec_model,
            denoise_model,
            diffusion,
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
            model=denoise_model,
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
            weight_decay=weight_decay,
            lr_anneal_steps=lr_anneal_steps,
            iterations=iterations,
            triplane_scaling_divider=triplane_scaling_divider,
            use_amp=use_amp,
            diffusion_input_size=diffusion_input_size,
            schedule_sampler=schedule_sampler,
        )

        # self.accelerator = Accelerator()

        self._load_and_sync_parameters(model=self.rec_model, model_name='rec')

        # * for loading EMA
        self.mp_trainer_rec = MixedPrecisionTrainer(
            model=self.rec_model,
            use_fp16=self.use_fp16,
            use_amp=use_amp,
            fp16_scale_growth=fp16_scale_growth,
            model_name='rec',
        )
        self.denoised_ae = denoised_ae

        if not freeze_ae:
            self.opt_rec = AdamW(
                self._init_optim_groups(self.mp_trainer_rec.model))
        else:
            print('!! freezing AE !!')

        # if not freeze_ae:
        if self.resume_step:
            if not ignore_resume_opt:
                self._load_optimizer_state()
            else:
                logger.warn("Ignoring optimizer state from checkpoint.")

            self.ema_params_rec = [
                self._load_ema_parameters(
                    rate,
                    self.rec_model,
                    self.mp_trainer_rec,
                    model_name=self.mp_trainer_rec.model_name)
                for rate in self.ema_rate
            ]  # for sync reconstruction model
        else:
            if not freeze_ae:
                self.ema_params_rec = [
                    copy.deepcopy(self.mp_trainer_rec.master_params)
                    for _ in range(len(self.ema_rate))
                ]

        if self.use_ddp is True:
            self.rec_model = th.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.rec_model)
            self.ddp_rec_model = DDP(
                self.rec_model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
                # find_unused_parameters=True,
            )
        else:
            self.ddp_rec_model = self.rec_model

        if freeze_ae:
            self.ddp_rec_model.eval()
            self.ddp_rec_model.requires_grad_(False)
        self.freeze_ae = freeze_ae

        # if use_amp:

    def _update_ema_rec(self):
        for rate, params in zip(self.ema_rate, self.ema_params_rec):
            update_ema(params, self.mp_trainer_rec.master_params, rate=rate)

    def run_loop(self, batch=None):
        th.cuda.empty_cache()
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            dist_util.synchronize()

            # if self.step % self.eval_interval == 0 and self.step != 0:
            if self.step % self.eval_interval == 0:
                if dist_util.get_rank() == 0:
                    self.eval_ddpm_sample(self.ddp_rec_model)
                #     continue # TODO, diffusion inference
                # self.eval_loop()
                # self.eval_novelview_loop()
                # let all processes sync up before starting with a new epoch of training
                dist_util.synchronize()
                th.cuda.empty_cache()

            batch = next(self.data)
            self.run_step(batch)
            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                # * log to tensorboard
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            if self.step % self.save_interval == 0 and self.step != 0:
                self.save()
                if not self.freeze_ae:
                    self.save(self.mp_trainer_rec, 'rec')
                dist_util.synchronize()

                th.cuda.empty_cache()
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
                    if not self.freeze_ae:
                        self.save(self.mp_trainer_rec, 'rec')

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            if not self.freeze_ae:
                self.save(self.mp_trainer_rec, 'rec')

    def run_step(self, batch, cond=None):
        self.forward_backward(batch,
                              cond)  # type: ignore # * 3D Reconstruction step
        took_step_ddpm = self.mp_trainer.optimize(self.opt)
        if took_step_ddpm:
            self._update_ema()

        if not self.freeze_ae:
            took_step_rec = self.mp_trainer_rec.optimize(self.opt_rec)
            if took_step_rec:
                self._update_ema_rec()

        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, *args, **kwargs):
        # return super().forward_backward(batch, *args, **kwargs)
        self.mp_trainer.zero_grad()
        # all_denoised_out = dict()
        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev())
                for k, v in batch.items()
            }

            last_batch = (i + self.microbatch) >= batch_size

            # if not freeze_ae:

            # =================================== ae part ===================================
            with th.cuda.amp.autocast(dtype=th.float16,
                                      enabled=self.mp_trainer_rec.use_amp
                                      and not self.freeze_ae):
                # with th.cuda.amp.autocast(dtype=th.float16,
                #                           enabled=False,): # ! debugging, no AMP on all the input

                latent = self.ddp_rec_model(
                    img=micro['img_to_encoder'],
                    c=micro['c'],
                    behaviour='enc_dec_wo_triplane')  # pred: (B, 3, 64, 64)

                if not self.freeze_ae:
                    target = micro
                    pred = self.rec_model(latent=latent,
                                          c=micro['c'],
                                          behaviour='triplane_dec')

                    if last_batch or not self.use_ddp:
                        ae_loss, loss_dict = self.loss_class(pred,
                                                             target,
                                                             test_mode=False)
                    else:
                        with self.ddp_model.no_sync():  # type: ignore
                            ae_loss, loss_dict = self.loss_class(
                                pred, target, test_mode=False)

                    log_rec3d_loss_dict(loss_dict)
                else:
                    ae_loss = th.tensor(0.0).to(dist_util.dev())

                # =================================== prepare for ddpm part ===================================

                micro_to_denoise = latent[
                    self.
                    latent_name] / self.triplane_scaling_divider  # normalize std to 1

                t, weights = self.schedule_sampler.sample(
                    micro_to_denoise.shape[0], dist_util.dev())

                model_kwargs = {}

                # print(micro_to_denoise.min(), micro_to_denoise.max())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro_to_denoise,  # x_start
                    t,
                    model_kwargs=model_kwargs,
                )

            with th.cuda.amp.autocast(dtype=th.float16,
                                      enabled=self.mp_trainer.use_amp):

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

                x_t = losses['x_t']
                model_output = losses['model_output']
                losses.pop('x_t')
                losses.pop('model_output')

                log_loss_dict(self.diffusion, t, {
                    k: v * weights
                    for k, v in losses.items()
                })

                # self.mp_trainer.backward(denoise_loss)
                # =================================== denosied ae part ===================================
                # if self.denoised_ae or self.step % 500 == 0:
                if self.denoised_ae:
                    with th.cuda.amp.autocast(
                            dtype=th.float16,
                            enabled=self.mp_trainer_rec.use_amp
                            and not self.freeze_ae):
                        # continue
                        denoised_out = denoised_fn()

                        denoised_ae_pred = self.ddp_rec_model(
                            img=None,
                            c=micro['c'],
                            latent=denoised_out['pred_xstart'] * self.
                            triplane_scaling_divider,  # TODO, how to define the scale automatically?
                            behaviour=self.render_latent_behaviour)

                        # if self.denoised_ae:

                        if last_batch or not self.use_ddp:
                            denoised_ae_loss, loss_dict = self.loss_class(
                                denoised_ae_pred, micro, test_mode=False)
                        else:
                            with self.ddp_model.no_sync():  # type: ignore
                                denoised_ae_loss, loss_dict = self.loss_class(
                                    denoised_ae_pred, micro, test_mode=False)

                        # * rename
                        loss_dict_denoise_ae = {}
                        for k, v in loss_dict.items():
                            loss_dict_denoise_ae[f'{k}_denoised'] = v.mean()
                        log_rec3d_loss_dict(loss_dict_denoise_ae)

                else:
                    denoised_ae_loss = th.tensor(0.0).to(dist_util.dev())

                loss = ae_loss + denoise_loss + denoised_ae_loss
                # self.mp_trainer.backward(denosied_ae_loss)
                # self.mp_trainer.backward(loss)

            # exit AMP before backward
            self.mp_trainer.backward(loss)
            # if self.freeze_ae:
            # else:
            # self.mp_trainer.backward(denoise_loss)

            # TODO, merge visualization with original AE
            # =================================== denoised AE log part ===================================

            # if dist_util.get_rank() == 0 and self.step % 500 == 0:
            if dist_util.get_rank() == 1 and self.step % 500 == 0:
                with th.no_grad():
                    # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                              gt_depth.min())
                    # if True:

                    if self.freeze_ae:
                        latent_micro = {
                            k:
                            v[0:1].to(dist_util.dev()) if v is not None else v
                            for k, v in latent.items()
                        }

                        pred = self.rec_model(latent=latent_micro,
                                              c=micro['c'][0:1],
                                              behaviour='triplane_dec')
                    else:
                        assert pred is not None

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

                    sr_w_code = latent_micro.get('sr_w_code', None)
                    if sr_w_code is not None:
                        sr_w_code = sr_w_code[0:1]

                    noised_ae_pred = self.ddp_rec_model(
                        img=None,
                        c=micro['c'][0:1],
                        latent={
                            'latent_normalized':
                            x_t[0:1] * self.triplane_scaling_divider,
                            # 'sr_w_code': getattr(self.ddp_rec_model.module.decoder,'w_avg').reshape(1,1,-1)
                            'sr_w_code': sr_w_code
                        },  # TODO, how to define the scale automatically
                        behaviour=self.render_latent_behaviour)

                    denoised_fn = functools.partial(
                        self.diffusion.p_mean_variance,
                        self.ddp_model,
                        x_t,  # x_start
                        t,
                        model_kwargs=model_kwargs)

                    denoised_out = denoised_fn()

                    denoised_ae_pred = self.ddp_rec_model(
                        img=None,
                        c=micro['c'][0:1],
                        # latent=denoised_out['pred_xstart'][0:1] * self.
                        # triplane_scaling_divider,  # TODO, how to define the scale automatically
                        latent={
                            'latent_normalized':
                            denoised_out['pred_xstart'][0:1] * self.
                            triplane_scaling_divider,  # TODO, how to define the scale automatically
                            #   'sr_w_code': getattr(self.ddp_rec_model.module.decoder,'w_avg').reshape(1,1,-1)
                            #   'sr_w_code': latent_micro['sr_w_code'][0:1]
                            'sr_w_code':
                            sr_w_code
                        },
                        behaviour=self.render_latent_behaviour)

                    assert denoised_ae_pred is not None

                    # print(pred_img.shape)
                    # print('denoised_ae:', self.denoised_ae)

                    pred_vis = th.cat([
                        pred_img[0:1], noised_ae_pred['image_raw'],
                        denoised_ae_pred['image_raw'],
                        pred_depth[0:1].repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)  # B, 3, H, W

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
                        f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t[0].item()}.jpg'
                    )
                    print(
                        'log denoised vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t[0].item()}.jpg'
                    )

                    th.cuda.empty_cache()
