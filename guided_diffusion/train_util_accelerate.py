import copy
from pdb import set_trace as st
import functools
import os
import numpy as np

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

from pathlib import Path

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

# use_amp = True
# use_amp = False
# if use_amp:
# logger.log('ddpm use AMP to accelerate training')


class TrainLoop:

    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        use_amp=False,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = ([ema_rate] if isinstance(ema_rate, float) else
                         [float(x) for x in ema_rate.split(",")])
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            use_amp=use_amp,
        )

        self.opt = AdamW(self.mp_trainer.master_params,
                         lr=self.lr,
                         weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        # print('creating DDP')
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn("Distributed training requires CUDA. "
                            "Gradients will not be synchronized properly!")
            self.use_ddp = False
            self.ddp_model = self.model
        # print('creating DDP done')

    def _load_and_sync_parameters(self):
        resume_checkpoint, resume_step = find_resume_checkpoint(
        ) or self.resume_checkpoint

        if resume_checkpoint:
            if not Path(resume_checkpoint).exists():
                logger.log(
                    f"failed to load model from checkpoint: {resume_checkpoint}, not exist"
                )
                return

            # self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            self.resume_step = resume_step  # TODO, EMA part
            if dist.get_rank() == 0:
                logger.log(
                    f"loading model from checkpoint: {resume_checkpoint}...")
                # if model is None:
                #     model = self.model
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint,
                        map_location=dist_util.dev(),
                    ))

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self,
                             rate,
                             model=None,
                             mp_trainer=None,
                             model_name='ddpm'):

        if mp_trainer is None:
            mp_trainer = self.mp_trainer
        if model is None:
            model = self.model

        ema_params = copy.deepcopy(mp_trainer.master_params)

        main_checkpoint, _ = find_resume_checkpoint(
            self.resume_checkpoint, model_name) or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step,
                                             rate, model_name)
        if ema_checkpoint:

            if dist_util.get_rank() == 0:

                if not Path(ema_checkpoint).exists():
                    logger.log(
                        f"failed to load EMA from checkpoint: {ema_checkpoint}, not exist"
                    )
                    return

                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")

                map_location = {
                    'cuda:%d' % 0: 'cuda:%d' % dist_util.get_rank()
                }  # configure map_location properly

                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=map_location)

                model_ema_state_dict = model.state_dict()

                for k, v in state_dict.items():
                    if k in model_ema_state_dict.keys() and v.size(
                    ) == model_ema_state_dict[k].size():
                        model_ema_state_dict[k] = v
                    else:
                        logger.log('ignore key: ', k, ": ", v.size())

                ema_params = mp_trainer.state_dict_to_master_params(
                    model_ema_state_dict)

                del state_dict

        # print('ema mark 3, ', model_name, flush=True)
        if dist_util.get_world_size() > 1:
            dist_util.sync_params(ema_params)
        # print('ema mark 4, ', model_name, flush=True)
        # del ema_params
        return ema_params

    def _load_ema_parameters_freezeAE(
            self,
            rate,
            model,
            #  mp_trainer=None,
            model_name='rec'):

        # if mp_trainer is None:
        # mp_trainer = self.mp_trainer
        # if model is None:
        # model = self.model_rec

        # ema_params = copy.deepcopy(mp_trainer.master_params)

        main_checkpoint, _ = find_resume_checkpoint(
            self.resume_checkpoint, model_name) or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step,
                                             rate, model_name)
        if ema_checkpoint:

            if dist_util.get_rank() == 0:

                if not Path(ema_checkpoint).exists():
                    logger.log(
                        f"failed to load EMA from checkpoint: {ema_checkpoint}, not exist"
                    )
                    return

                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")

                map_location = {
                    'cuda:%d' % 0: 'cuda:%d' % dist_util.get_rank()
                }  # configure map_location properly

                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=map_location)

                model_ema_state_dict = model.state_dict()

                for k, v in state_dict.items():
                    if k in model_ema_state_dict.keys() and v.size(
                    ) == model_ema_state_dict[k].size():
                        model_ema_state_dict[k] = v
                    else:
                        logger.log('ignore key: ', k, ": ", v.size())

                ema_params = mp_trainer.state_dict_to_master_params(
                    model_ema_state_dict)

                del state_dict

        # print('ema mark 3, ', model_name, flush=True)
        if dist_util.get_world_size() > 1:
            dist_util.sync_params(ema_params)
        # print('ema mark 4, ', model_name, flush=True)
        # del ema_params
        return ema_params

    # def _load_ema_parameters(self, rate):
    #     ema_params = copy.deepcopy(self.mp_trainer.master_params)

    #     main_checkpoint, _ = find_resume_checkpoint() or self.resume_checkpoint
    #     ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
    #     if ema_checkpoint:
    #         if dist.get_rank() == 0:
    #             logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
    #             state_dict = dist_util.load_state_dict(
    #                 ema_checkpoint, map_location=dist_util.dev()
    #             )
    #             ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

    #     dist_util.sync_params(ema_params)
    #     return ema_params

    def _load_optimizer_state(self):
        main_checkpoint, _ = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint),
                                 f"opt{self.resume_step:06}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(
                f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):

            # st()
            with th.autocast(device_type=dist_util.dev(),
                             dtype=th.float16,
                             enabled=self.mp_trainer.use_amp):

                micro = batch[i:i + self.microbatch].to(dist_util.dev())
                micro_cond = {
                    k: v[i:i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                }
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(
                    micro.shape[0], dist_util.dev())

                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach())

                loss = (losses["loss"] * weights).mean()
                log_loss_dict(self.diffusion, t,
                              {k: v * weights
                               for k, v in losses.items()})

            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples",
                     (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):07d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):07d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename),
                                 "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                    bf.join(get_blob_logdir(),
                            f"opt{(self.step+self.resume_step):07d}.pt"),
                    "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split1 = Path(filename).stem[-6:]
    # split = filename.split("model")
    # if len(split) < 2:
    #     return 0
    # split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        print('fail to load model step', split1)
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint(resume_checkpoint='', model_name='ddpm'):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.

    if resume_checkpoint != '':
        step = parse_resume_step_from_filename(resume_checkpoint)
        split = resume_checkpoint.split("model")
        resume_ckpt_path = str(
            Path(split[0]) / f'model_{model_name}{step:07d}.pt')
    else:
        resume_ckpt_path = ''
        step = 0

    return resume_ckpt_path, step


def find_ema_checkpoint(main_checkpoint, step, rate, model_name=''):
    if main_checkpoint is None:
        return None
    if model_name == '':
        filename = f"ema_{rate}_{(step):07d}.pt"
    else:
        filename = f"ema_{model_name}_{rate}_{(step):07d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    # print(path)
    # st()
    if bf.exists(path):
        print('load ema model', path)
        return path
    else:
        print('fail to load ema model', path)
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(),
                                   values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def log_rec3d_loss_dict(loss_dict):
    for key, values in loss_dict.items():
        logger.logkv_mean(key, values.mean().item())


def calc_average_loss(all_loss_dicts):
    all_scores = {}  # todo, defaultdict
    mean_all_scores = {}

    for loss_dict in all_loss_dicts:
        for k, v in loss_dict.items():
            v = v.item()
            if k not in all_scores:
                # all_scores[f'{k}_val'] = [v]
                all_scores[k] = [v]
            else:
                all_scores[k].append(v)

    for k, v in all_scores.items():
        mean = np.mean(v)
        std = np.std(v)
        if k in ['loss_lpis', 'loss_ssim']:
            mean = 1 - mean
        result_str = '{} average loss is {:.4f} +- {:.4f}'.format(k, mean, std)
        mean_all_scores[k] = mean
        print(result_str)

    val_scores_for_logging = {
        f'{k}_val': v
        for k, v in mean_all_scores.items()
    }
    return val_scores_for_logging