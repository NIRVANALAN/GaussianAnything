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
from PIL import Image
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from tqdm import tqdm

from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion import dist_util, logger
from guided_diffusion.train_util import (calc_average_loss,
                                         log_rec3d_loss_dict,
                                         find_resume_checkpoint)

from torch.optim import AdamW

from .train_util import TrainLoopBasic, TrainLoop3DRec
# import vision_aided_loss
from dnnlib.util import calculate_adaptive_weight


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


class TrainLoop3DcvD(TrainLoop3DRec):

    def __init__(
            self,
            *,
            rec_model,
            loss_class,
            # diffusion,
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
            fp16_scale_growth=1e-3,
            # schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            iterations=10001,
            load_submodule_name='',
            ignore_resume_opt=False,
            use_amp=False,
            cvD_name='cvD',
            model_name='rec',
            # SR_TRAINING=True,
            SR_TRAINING=False,
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
                         model_name=model_name,
                         use_amp=use_amp, 
                         cvD_name=cvD_name,
                         **kwargs)

        # self.rec_model = self.ddp_model

        # device = loss_class.device
        device = dist_util.dev()
        # * create vision aided model
        # TODO, load model
        self.nvs_cvD = vision_aided_loss.Discriminator(
            cv_type='clip', loss_type='multilevel_sigmoid_s',
            device=device).to(device)
        self.nvs_cvD.cv_ensemble.requires_grad_(False)  # Freeze feature extractor
        # self.nvs_cvD.train()

        # 
        # SR_TRAINING = False
        cvD_model_params=list(self.nvs_cvD.decoder.parameters())
        self.SR_TRAINING = SR_TRAINING
        # SR_TRAINING = True
        if SR_TRAINING:
            # width, patch_size = self.nvs_cvD.cv_ensemble
            vision_width, vision_patch_size = [self.nvs_cvD.cv_ensemble.models[0].model.conv1.weight.shape[k] for k in [0, -1]]
            self.nvs_cvD.cv_ensemble.models[0].model.conv1 = th.nn.Conv2d(in_channels=6, out_channels=vision_width, kernel_size=vision_patch_size, stride=vision_patch_size, bias=False).to(dist_util.dev())
            self.nvs_cvD.cv_ensemble.models[0].model.conv1.requires_grad_(True)
            cvD_model_params += list(self.nvs_cvD.cv_ensemble.models[0].model.conv1.parameters())

            # change normalization metrics
            self.nvs_cvD.cv_ensemble.models[0].image_mean = self.nvs_cvD.cv_ensemble.models[0].image_mean.repeat(2)
            self.nvs_cvD.cv_ensemble.models[0].image_std = self.nvs_cvD.cv_ensemble.models[0].image_std.repeat(2)

        # logger.log(f'nvs_cvD_model_params: {cvD_model_params}')

        self._load_and_sync_parameters(model=self.nvs_cvD, model_name='cvD')

        self.mp_trainer_cvD = MixedPrecisionTrainer(
            model=self.nvs_cvD,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            model_name=cvD_name,
            use_amp=use_amp,
            model_params=cvD_model_params
            )

        # cvD_lr = 4e-5*(lr/1e-5)
        # cvD_lr = 4e-4*(lr/1e-5)
        cvD_lr = 1e-4*(lr/1e-5) * self.loss_class.opt.nvs_D_lr_mul
        # cvD_lr = 1e-5*(lr/1e-5)
        self.opt_cvD = AdamW(
            self.mp_trainer_cvD.master_params,
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

        th.cuda.empty_cache()

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

            # batch, cond = next(self.data)
            # if batch is None:
            batch = next(self.data)
            self.run_step(batch, 'g_step_rec')

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
                logger.log('reached maximum iterations, exiting')

                # Save the last checkpoint if it wasn't already saved.
                if (self.step - 1) % self.save_interval != 0:
                    self.save()
                    self.save(self.mp_trainer_cvD, 'cvD')

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.save(self.mp_trainer_cvD, 'cvD')

    # def forward_backward(self, batch, *args, **kwargs):
    # blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

    def run_D_Diter(self, real, fake, D=None):
        # Dmain: Minimize logits for generated images and maximize logits for real images.
        if D is None:
            D = self.ddp_nvs_cvD

        lossD = D(real, for_real=True).mean() + D(
            fake, for_real=False).mean()
        return lossD

    def forward_D(self, batch):  # update D
        self.mp_trainer_cvD.zero_grad()
        self.ddp_nvs_cvD.requires_grad_(True)
        self.rec_model.requires_grad_(False)

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

                # pred = self.rec_model(img=micro['img_to_encoder'],
                #                       c=micro['c'])  # pred: (B, 3, 64, 64)

                pred = self.rec_model(
                    img=micro['img_to_encoder'],
                    c=th.cat([
                        micro['c'][1:],
                        micro['c'][:1],  # half novel view, half orig view
                    ]))

                real_logits_cv = self.run_D_Diter(
                    real=micro['img_to_encoder'],
                    fake=pred['image_raw'])  # TODO, add SR for FFHQ

            log_rec3d_loss_dict({'vision_aided_loss/D': real_logits_cv})

            self.mp_trainer_cvD.backward(real_logits_cv)

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

            # VQ3D novel view d loss
            # duplicated_for_nvs = th.cat([
            #     micro['img_to_encoder'][:batch_size - 2],
            #     micro['img_to_encoder'][:2]
            # ], 0)

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                pred = self.rec_model(
                    img=micro['img_to_encoder'], c=micro['c']
                )  # render novel view for first half of the batch for D loss

                target_for_rec = micro
                pred_for_rec = pred

                # pred_for_rec = {
                #     k: v[:batch_size - 2] if v is not None else None
                #     for k, v in pred.items()
                # }
                # target_for_rec = {
                #     k: v[:batch_size - 2] if v is not None else None
                #     for k, v in target.items()
                # }

                if last_batch or not self.use_ddp:
                    loss, loss_dict = self.loss_class(pred_for_rec,
                                                      target_for_rec,
                                                      test_mode=False)
                else:
                    with self.rec_model.no_sync():  # type: ignore
                        loss, loss_dict = self.loss_class(pred_for_rec,
                                                          target_for_rec,
                                                          test_mode=False)

                # add cvD supervision
                vision_aided_loss = self.ddp_nvs_cvD(
                    pred_for_rec['image_raw'],
                    for_G=True).mean()  # [B, 1] shape

                last_layer = self.rec_model.module.decoder.triplane_decoder.decoder.net[  # type: ignore
                    -1].weight  # type: ignore

                d_weight = calculate_adaptive_weight(
                    loss, vision_aided_loss, last_layer,
                    # disc_weight_max=0.1) * 0.1
                    # disc_weight_max=0.1) * 0.05
                    disc_weight_max=1)
                loss += vision_aided_loss * d_weight

                loss_dict.update({
                    'vision_aided_loss/G_rec': vision_aided_loss,
                    'd_weight': d_weight
                })

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
                    logger.log(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}_rec.jpg'
                    )

    def forward_G_nvs(self, batch):  # update G

        self.mp_trainer_rec.zero_grad()
        self.rec_model.requires_grad_(True)
        self.ddp_nvs_cvD.requires_grad_(False)

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }

            # last_batch = (i + self.microbatch) >= batch_size

            # VQ3D novel view d loss
            # duplicated_for_nvs = th.cat([
            #     micro['img_to_encoder'][batch_size // 2:],
            #     micro['img_to_encoder'][:batch_size // 2]
            # ], 0)

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                pred = self.rec_model(
                    # img=duplicated_for_nvs, c=micro['c']
                    img=micro['img_to_encoder'],
                    c=th.cat([
                        micro['c'][1:],
                        micro['c'][:1],
                    ])
                )  # render novel view for first half of the batch for D loss

                # add cvD supervision
                vision_aided_loss = self.ddp_nvs_cvD(
                    pred['image_raw'], for_G=True).mean()  # [B, 1] shape

                # loss = vision_aided_loss * 0.01
                # loss = vision_aided_loss * 0.005
                # loss = vision_aided_loss * 0.1
                loss = vision_aided_loss * 0.01

                log_rec3d_loss_dict({
                    'vision_aided_loss/G_nvs':
                    vision_aided_loss,
                })

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
                        pred_img = th.cat(
                            [self.pool_512(pred_img), pred['image_sr']],
                            dim=-1)
                        gt_img = th.cat(
                            [self.pool_512(micro['img']), micro['img_sr']],
                            dim=-1)
                        pred_depth = self.pool_512(pred_depth)
                        gt_depth = self.pool_512(gt_depth)

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

                    # logger.log(vis.shape)

                    Image.fromarray(vis).save(
                        f'{logger.get_dir()}/{self.step+self.resume_step}_nvs.jpg'
                    )
                    logger.log(
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

        if model_name == 'ddpm':
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint(rate, params)

        dist.barrier()

    def _load_and_sync_parameters(self, model=None, model_name='rec'):
        resume_checkpoint, self.resume_step = find_resume_checkpoint(
            self.resume_checkpoint, model_name) or self.resume_checkpoint

        if model is None:
            model = self.rec_model  # default model in the parent class

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
                    elif 'attn.wk' in k or 'attn.wv' in k: # old qkv
                        logger.log('ignore ', k)

                    elif 'decoder.vit_decoder.blocks' in k:
                        # st()
                        # load from 2D ViT pre-trained into 3D ViT blocks. 
                        assert len(model.decoder.vit_decoder.blocks[0].vit_blks) == 2 # assert depth=2 here.
                        fusion_ca_depth = len(model.decoder.vit_decoder.blocks[0].vit_blks)
                        vit_subblk_index = int(k.split('.')[3])
                        vit_blk_keyname = ('.').join(k.split('.')[4:])
                        fusion_blk_index = vit_subblk_index // fusion_ca_depth
                        fusion_blk_subindex = vit_subblk_index % fusion_ca_depth
                        model_state_dict[f'decoder.vit_decoder.blocks.{fusion_blk_index}.vit_blks.{fusion_blk_subindex}.{vit_blk_keyname}'] = v
                        # logger.log('load 2D ViT weight: {}'.format(f'decoder.vit_decoder.blocks.{fusion_blk_index}.vit_blks.{fusion_blk_subindex}.{vit_blk_keyname}'))

                    elif 'IN' in k:
                        logger.log('ignore ', k)

                    elif 'quant_conv' in k:
                        logger.log('ignore ', k)

                    else:
                        logger.log('!!!! ignore key: ', k, ": ", v.size(),)
                        if k in model_state_dict:
                            logger.log('shape in model: ', model_state_dict[k].size())
                        else:
                            logger.log(k, 'not in model_state_dict')

                model.load_state_dict(model_state_dict, strict=True)
                del model_state_dict

        if dist_util.get_world_size() > 1:
            dist_util.sync_params(model.parameters())
            logger.log(f'synced {model_name} params')
