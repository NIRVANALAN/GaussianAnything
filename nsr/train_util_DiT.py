import copy
import functools
import json
import os
from pathlib import Path
from pdb import set_trace as st

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

# from ..guided_diffusion.train_util import TrainLoop

# use_amp=True
use_amp = False
if use_amp:
    logger.log('DiT using AMP')

from .train_util_diffusion import TrainLoop3DDiffusion
import dnnlib


class TrainLoop3DDiffusionDiT(TrainLoop3DDiffusion):

    def __init__(self,
                 *,
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
                 **kwargs):
        super().__init__(rec_model=rec_model,
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
                         freeze_ae=freeze_ae,
                         denoised_ae=denoised_ae,
                         triplane_scaling_divider=triplane_scaling_divider,
                         use_amp=use_amp,
                         **kwargs)

    def eval_ddpm_sample(self):

        args = dnnlib.EasyDict(
            dict(batch_size=1,
                 image_size=224,
                 denoise_in_channels=24,
                 clip_denoised=True,
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
                self.ddp_model,
                (args.batch_size, args.denoise_in_channels, args.image_size,
                 args.image_size),  # 
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )  # B 8 H W*3

            # print(triplane_sample.shape)

            # B, C, H, W = triplane_sample.shape
            # triplane_sample = triplane_sample.reshape(B, C, H, W//3, 3).permute(0,1,4,2,3) # c*3 order
            # triplane_sample.reshape(B, -1, H, W//3) # B 24 H W

            self.render_video_given_triplane(
                triplane_sample,
                name_prefix=f'{self.step + self.resume_step}_{i}')


class TrainLoop3DDiffusionDiTOverfit(TrainLoop):

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
            **kwargs):

        super().__init__(model=denoise_model,
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
                         lr_anneal_steps=lr_anneal_steps,
                         weight_decay=weight_decay,
                         use_amp=use_amp)

        # self.accelerator = Accelerator()

        self.pool_512 = th.nn.AdaptiveAvgPool2d((512, 512))
        self.pool_128 = th.nn.AdaptiveAvgPool2d((128, 128))
        self.loss_class = loss_class
        self.rec_model = rec_model
        self.eval_interval = eval_interval
        self.eval_data = eval_data
        self.iterations = iterations
        # self.triplane_std = 10
        self.triplane_scaling_divider = triplane_scaling_divider

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
            self.opt_rec = AdamW(self._init_optim_groups(kwargs))
        else:
            print('!! freezing AE !!')

        if dist_util.get_rank() == 0:
            self.writer = SummaryWriter(log_dir=f'{logger.get_dir()}/runs')
            print(self.opt)
            if not freeze_ae:
                print(self.opt_rec)

        # if not freeze_ae:
        if self.resume_step:
            if not ignore_resume_opt:
                self._load_optimizer_state()
            else:
                logger.warn("Ignoring optimizer state from checkpoint.")
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            # if not freeze_ae:
            #     self.ema_params_rec = [
            #         self._load_ema_parameters(
            #             rate,
            #             self.rec_model,
            #             self.mp_trainer_rec,
            #             model_name=self.mp_trainer_rec.model_name)
            #         for rate in self.ema_rate
            #     ]
            # else:
            self.ema_params_rec = [
                self._load_ema_parameters(
                    rate,
                    self.rec_model,
                    self.mp_trainer_rec,
                    model_name=self.mp_trainer_rec.model_name)
                for rate in self.ema_rate
            ]
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

    def _init_optim_groups(self, kwargs):
        optim_groups = [
            # vit encoder
            {
                'name': 'vit_encoder',
                'params': self.mp_trainer_rec.model.encoder.parameters(),
                'lr': kwargs['encoder_lr'],
                'weight_decay': kwargs['encoder_weight_decay']
            },
            # vit decoder
            {
                'name':
                'vit_decoder',
                'params':
                self.mp_trainer_rec.model.decoder.vit_decoder.parameters(),
                'lr':
                kwargs['vit_decoder_lr'],
                'weight_decay':
                kwargs['vit_decoder_wd']
            },
            {
                'name':
                'vit_decoder_pred',
                'params':
                self.mp_trainer_rec.model.decoder.decoder_pred.parameters(),
                'lr':
                kwargs['vit_decoder_lr'],
                # 'weight_decay': 0
                'weight_decay':
                kwargs['vit_decoder_wd']
            },

            # triplane decoder
            {
                'name':
                'triplane_decoder',
                'params':
                self.mp_trainer_rec.model.decoder.triplane_decoder.parameters(
                ),
                'lr':
                kwargs['triplane_decoder_lr'],
                # 'weight_decay': self.weight_decay
            },
        ]

        if self.mp_trainer_rec.model.decoder.superresolution is not None:
            optim_groups.append({
                'name':
                'triplane_decoder_superresolution',
                'params':
                self.mp_trainer_rec.model.decoder.superresolution.parameters(),
                'lr':
                kwargs['super_resolution_lr'],
            })

        return optim_groups

    def run_loop(self, batch=None):
        th.cuda.empty_cache()
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            dist_util.synchronize()

            # batch, cond = next(self.data)
            # if batch is None:
            batch = next(self.data)
            self.run_step(batch)
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
                #     continue # TODO, diffusion inference
                # self.eval_loop()
                # self.eval_novelview_loop()
                # let all processes sync up before starting with a new epoch of training
                dist_util.synchronize()
                th.cuda.empty_cache()

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

                # pred = self.ddp_rec_model(img=micro['img_to_encoder'],
                #                           c=micro['c'])  # pred: (B, 3, 64, 64)
                # if not self.freeze_ae:
                #     target = micro

                #     if last_batch or not self.use_ddp:
                #         ae_loss, loss_dict = self.loss_class(pred,
                #                                              target,
                #                                              test_mode=False)
                #     else:
                #         with self.ddp_model.no_sync():  # type: ignore
                #             ae_loss, loss_dict = self.loss_class(
                #                 pred, target, test_mode=False)

                #     log_rec3d_loss_dict(loss_dict)
                # else:
                #     ae_loss = th.tensor(0.0).to(dist_util.dev())

                # micro_to_denoise = micro['img']
                # micro_to_denoise = micro['img'].repeat_interleave(
                    # 8, dim=1)  # B 3*8 H W
                micro_to_denoise = micro['img'].repeat_interleave(2, dim=1) # B 3*8 H W
                # micro_to_denoise = micro['img'].repeat_interleave(1, dim=1) # B 3*8 H W

                # micro_to_denoise = pred[
                #     'latent'] / self.triplane_scaling_divider  # normalize std to 1

                t, weights = self.schedule_sampler.sample(
                    micro_to_denoise.shape[0], dist_util.dev())

                # print('!!!', micro_to_denoise.dtype)
                # =================================== denoised part ===================================

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
                        # denoised_out = denoised_fn()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach())

                denoise_loss = (losses["loss"] * weights).mean()

                x_t = losses['x_t']
                losses.pop('x_t')

                log_loss_dict(self.diffusion, t,
                              {k: v * weights
                               for k, v in losses.items()})

                loss = denoise_loss  # ! leave only denoise_loss for debugging

            # exit AMP before backward
            self.mp_trainer.backward(loss)

            # TODO, merge visualization with original AE
            # =================================== denoised AE log part ===================================

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                with th.no_grad():
                    # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                              gt_depth.min())
                    # if True:
                    # pred_depth = pred['image_depth']
                    # pred_depth = (pred_depth - pred_depth.min()) / (
                    #     pred_depth.max() - pred_depth.min())
                    # pred_img = pred['image_raw']
                    # gt_img = micro['img']

                    # if 'image_sr' in pred:  # TODO
                    #     pred_img = th.cat(
                    #         [self.pool_512(pred_img), pred['image_sr']],
                    #         dim=-1)
                    #     gt_img = th.cat(
                    #         [self.pool_512(micro['img']), micro['img_sr']],
                    #         dim=-1)
                    #     pred_depth = self.pool_512(pred_depth)
                    #     gt_depth = self.pool_512(gt_depth)

                    # gt_vis = th.cat(
                    #     [
                    #         # gt_img,
                    #         micro['img'],
                    #         # gt_depth.repeat_interleave(3, dim=1)
                    #     ],
                    #     dim=-1)[0:1]  # TODO, fail to load depth. range [0, 1]

                    # if not self.denoised_ae:
                    #     # continue

                    #     denoised_ae_pred = self.ddp_rec_model(
                    #         img=None,
                    #         c=micro['c'][0:1],
                    #         latent=denoised_out['pred_xstart'][0:1] * self.
                    #         triplane_scaling_divider,  # TODO, how to define the scale automatically
                    #         behaviour='triplane_dec')

                    # assert denoised_ae_pred is not None

                    # print(pred_img.shape)
                    # print('denoised_ae:', self.denoised_ae)

                    # pred_vis = th.cat([
                    #     pred_img[0:1], denoised_ae_pred['image_raw'],
                    #     pred_depth[0:1].repeat_interleave(3, dim=1)
                    # ],
                    #                   dim=-1)  # B, 3, H, W

                    # vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
                    #     1, 2, 0).cpu()  # ! pred in range[-1, 1]

                    # x_t = self.diffusion.q_sample(
                    #     x_start, t, noise=noise
                    # )  # * add noise according to predefined schedule

                    denoised_fn = functools.partial(
                        self.diffusion.p_mean_variance,
                        self.ddp_model,
                        x_t,  # x_start
                        t,
                        model_kwargs=model_kwargs)

                    denoised_out = denoised_fn()

                    vis = th.cat([
                        micro['img'], x_t[:, :3, ...],
                        denoised_out['pred_xstart'][:, :3, ...]
                    ],
                                 dim=-1)[0].permute(
                                     1, 2, 0).cpu()  # ! pred in range[-1, 1]

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

    @th.no_grad()
    # def eval_loop(self, c_list:list):
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

            pred = self.model(img=novel_view_micro['img_to_encoder'],
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
                pred_vis = th.cat([
                    micro['img_sr'],
                    self.pool_512(pred['image_raw']), pred['image_sr'],
                    self.pool_512(pred_depth).repeat_interleave(3, dim=1)
                ],
                                  dim=-1)
            else:
                pred_vis = th.cat([
                    self.pool_128(micro['img']), pred['image_raw'],
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

    @th.no_grad()
    # def eval_loop(self, c_list:list):
    def eval_loop(self):
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
            pred = self.ddp_rec_model(img=micro['img_to_encoder'],
                                      c=micro['c'])  # pred: (B, 3, 64, 64)

            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())

            if 'image_sr' in pred:
                pred_vis = th.cat([
                    micro['img_sr'],
                    self.pool_512(pred['image_raw']), pred['image_sr'],
                    self.pool_512(pred_depth).repeat_interleave(3, dim=1)
                ],
                                  dim=-1)
            else:
                pred_vis = th.cat([
                    self.pool_128(micro['img']), pred['image_raw'],
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

        self.eval_novelview_loop()

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

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        dist.barrier()

    def _load_and_sync_parameters(self, model=None, model_name='ddpm'):
        resume_checkpoint, self.resume_step = find_resume_checkpoint(
            self.resume_checkpoint, model_name) or self.resume_checkpoint

        if model is None:
            model = self.model
        print(resume_checkpoint)

        if resume_checkpoint and Path(resume_checkpoint).exists():
            if dist_util.get_rank() == 0:

                # ! rank 0 return will cause all other ranks to hang
                # if not Path(resume_checkpoint).exists():
                #     logger.log(
                #         f"failed to load model from checkpoint: {resume_checkpoint}, not exist"
                #     )
                #     return

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

                    # elif 'IN' in k and model_name == 'rec' and getattr(model.decoder, 'decomposed_IN', False):
                    #     model_state_dict[k.replace('IN', 'superresolution.norm.norm_layer')] = v # decomposed IN

                    else:
                        print('!!!! ignore key: ', k, ": ", v.size(),
                              'shape in model: ', model_state_dict[k].size())

                model.load_state_dict(model_state_dict, strict=True)
                del model_state_dict

        if dist_util.get_world_size() > 1:
            dist_util.sync_params(model.parameters())
            print(f'synced {model_name} params')

    def _update_ema_rec(self):
        for rate, params in zip(self.ema_rate, self.ema_params_rec):
            update_ema(params, self.mp_trainer_rec.master_params, rate=rate)

    def eval_ddpm_sample(self):

        args = dnnlib.EasyDict(
            dict(
                batch_size=1,
                image_size=128,
                #  denoise_in_channels=3,
                # denoise_in_channels=24,
                 denoise_in_channels=6,
                #  denoise_in_channels=6,
                clip_denoised=True,
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
            img_sample = sample_fn(
                self.ddp_model,
                (args.batch_size, args.denoise_in_channels, args.image_size,
                 args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            pred_vis = img_sample

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()[0][..., :3]
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            Image.fromarray(vis).save(
                f'{logger.get_dir()}/{self.step + self.resume_step}_{i}.png')

            # th.cuda.empty_cache()
            # self.render_video_given_triplane(
            #     triplane_sample,
            #     name_prefix=f'{self.step + self.resume_step}_{i}')

            th.cuda.empty_cache()

    @th.inference_mode()
    def render_video_given_triplane(self, planes, name_prefix='0'):

        planes *= self.triplane_scaling_divider  # if setting clip_denoised=True, the sampled planes will lie in [-1,1]. Thus, values beyond [+- std] will be abandoned in this version. Move to IN for later experiments.

        # print(planes.min(), planes.max())

        # used during diffusion sampling inference
        video_out = imageio.get_writer(
            f'{logger.get_dir()}/triplane_{name_prefix}.mp4',
            mode='I',
            fps=60,
            codec='libx264')

        # for i in range(0, len(c_list), 1): # TODO, larger batch size for eval
        for i, batch in enumerate(tqdm(self.eval_data)):
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            pred = self.ddp_rec_model(img=None,
                                      c=micro['c'],
                                      latent=planes,
                                      behaviour='triplane_dec')

            # if True:
            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())

            if 'image_sr' in pred:
                pred_vis = th.cat([
                    micro['img_sr'],
                    self.pool_512(pred['image_raw']), pred['image_sr'],
                    self.pool_512(pred_depth).repeat_interleave(3, dim=1)
                ],
                                  dim=-1)
            else:
                pred_vis = th.cat([
                    self.pool_128(micro['img']), pred['image_raw'],
                    pred_depth.repeat_interleave(3, dim=1)
                ],
                                  dim=-1)  # B, 3, H, W

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            for j in range(vis.shape[0]):
                video_out.append_data(vis[j])

        video_out.close()
        print('logged video to: ',
              f'{logger.get_dir()}/triplane_{name_prefix}.mp4')

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

            latent = self.ddp_rec_model(
                img=novel_view_micro['img_to_encoder'],
                c=micro['c'])['latent']  # pred: (B, 3, 64, 64)

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
                    behaviour='triplane_dec')  # pred: (B, 3, 64, 64)

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
