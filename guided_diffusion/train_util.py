import copy
from tqdm import tqdm, trange
import imageio
from pdb import set_trace as st
import functools
import os
import numpy as np

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
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
        model_name='ddpm',
        train_vae=True,
        compile=False,
        clip_grad_throld=1.0,
        **kwargs
    ):

        self.kwargs = kwargs
        self.clip_grad_throld = clip_grad_throld
        self.pool_512 = th.nn.AdaptiveAvgPool2d((512, 512))
        self.pool_256 = th.nn.AdaptiveAvgPool2d((256, 256))
        self.pool_128 = th.nn.AdaptiveAvgPool2d((128, 128))
        self.pool_64 = th.nn.AdaptiveAvgPool2d((64, 64))

        self.use_amp = use_amp

        self.dtype = th.float32
        # if use_amp:
        #     if th.backends.cuda.matmul.allow_tf32: # a100
        #         self.dtype = th.bfloat16
        #     else:
        #         self.dtype = th.float16
        # else:

        if use_amp: 
            if th.cuda.get_device_capability(0)[0] < 8:
                self.dtype = th.float16 # e.g., v100
            else:
                self.dtype = th.bfloat16 # e.g., a100 / a6000

        self.model_name = model_name
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
        self.train_vae = train_vae

        self.sync_cuda = th.cuda.is_available()
        self.triplane_scaling_divider = 1.0
        self.latent_name = 'latent_normalized_2Ddiffusion'  # normalized triplane latent
        self.render_latent_behaviour = 'decode_after_vae'  # directly render using triplane operations
        self._setup_model()
        self._load_model()
        self._setup_opt()
    
    def _load_model(self):
        self._load_and_sync_parameters()
    
    def _setup_opt(self):
        self.opt = AdamW(self.mp_trainer.master_params,
                         lr=self.lr,
                         weight_decay=self.weight_decay)
    
    def _setup_model(self):

        # st()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
            use_amp=self.use_amp,
            model_name=self.model_name,
            clip_grad_throld=self.clip_grad_throld,
        )

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
            
        # for compatability

        # print('creating DDP')
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddpm_model = self.model
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

        # if compile:
        #     self.model = th.compile(self.model) # some op will break graph now
        #     logger.warn("compiling...")


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

        # ! debugging, remove to check which key fails.
        dist_util.sync_params(self.model.parameters())
        # dist_util.sync_params(self.model.named_parameters())

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

                    # elif 'IN' in k and model_name == 'rec' and getattr(model.decoder, 'decomposed_IN', False):
                    #     model_ema_state_dict[k.replace('IN', 'superresolution.norm.norm_layer')] = v # decomposed IN

                    else:
                        print('ignore key: ', k, ": ", v.size())

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
                        print('ignore key: ', k, ": ", v.size())

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

    @th.no_grad()
    def _make_vis_img(self, pred):

        # if True:
        pred_depth = pred['image_depth']
        pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                        pred_depth.min())


        pred_depth = pred_depth.cpu()[0].permute(1, 2, 0).numpy()
        pred_depth = (plt.cm.viridis(pred_depth[..., 0])[..., :3]) * 2 - 1
        pred_depth = th.from_numpy(pred_depth).to(
            pred['image_raw'].device).permute(2, 0, 1).unsqueeze(0)
        # rend_normal = pred['rend_normal']

        # if 'image_sr' in pred:

        #     gen_img = pred['image_sr']

        #     if pred['image_sr'].shape[-1] == 512:

        #         pred_vis = th.cat([
        #             micro['img_sr'],
        #             self.pool_512(pred['image_raw']), gen_img,
        #             self.pool_512(pred_depth).repeat_interleave(3, dim=1)
        #         ],
        #                             dim=-1)

        #     elif pred['image_sr'].shape[-1] == 128:

        #         pred_vis = th.cat([
        #             micro['img_sr'],
        #             self.pool_128(pred['image_raw']), pred['image_sr'],
        #             self.pool_128(pred_depth).repeat_interleave(3, dim=1)
        #         ],
        #                             dim=-1)

        # else:
        gen_img = pred['image_raw']

        pred_vis = th.cat(
            [
                gen_img,
                # rend_normal,
                pred_depth,
            ],
            dim=-1)  # B, 3, H, W
        
        return pred_vis

    @th.inference_mode()
    def render_video_given_triplane(self,
                                    planes,
                                    rec_model,
                                    name_prefix='0',
                                    save_img=False, 
                                    render_reference=None, 
                                    export_mesh=False):

        planes *= self.triplane_scaling_divider  # if setting clip_denoised=True, the sampled planes will lie in [-1,1]. Thus, values beyond [+- std] will be abandoned in this version. Move to IN for later experiments.

        # sr_w_code = getattr(self.ddp_rec_model.module.decoder, 'w_avg', None)
        # sr_w_code = None
        batch_size = planes.shape[0]

        # if sr_w_code is not None:
        #     sr_w_code = sr_w_code.reshape(1, 1,
        #                                   -1).repeat_interleave(batch_size, 0)

        # used during diffusion sampling inference
        # if not save_img:

        # ! mesh

        if planes.shape[1] == 16:  # ffhq/car
            ddpm_latent = {
                self.latent_name: planes[:, :12],
                'bg_plane': planes[:, 12:16],
            }
        else:
            ddpm_latent = {
                self.latent_name: planes,
            }
        
        ddpm_latent.update(rec_model(latent=ddpm_latent, behaviour='decode_after_vae_no_render')) 

        if export_mesh:
        # if True:
            # mesh_size = 512
            # mesh_size = 256
            mesh_size = 384
            # mesh_size = 320
            # mesh_thres = 3 # TODO, requires tuning
            # mesh_thres = 5 # TODO, requires tuning
            mesh_thres = 10 # TODO, requires tuning
            import mcubes
            import trimesh
            dump_path = f'{logger.get_dir()}/mesh/'

            os.makedirs(dump_path, exist_ok=True)

            grid_out = rec_model(
                latent=ddpm_latent,
                grid_size=mesh_size,
                behaviour='triplane_decode_grid',
            )
            
            vtx, faces = mcubes.marching_cubes(grid_out['sigma'].squeeze(0).squeeze(-1).cpu().numpy(), mesh_thres)
            vtx = vtx / (mesh_size - 1) * 2 - 1

            # vtx_tensor = th.tensor(vtx, dtype=th.float32, device=dist_util.dev()).unsqueeze(0)
            # vtx_colors = self.model.synthesizer.forward_points(planes, vtx_tensor)['rgb'].squeeze(0).cpu().numpy()  # (0, 1)
            # vtx_colors = (vtx_colors * 255).astype(np.uint8)
            
            # mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)
            mesh = trimesh.Trimesh(vertices=vtx, faces=faces,)

            mesh_dump_path = os.path.join(dump_path, f'{name_prefix}.ply')
            mesh.export(mesh_dump_path, 'ply')

            print(f"Mesh dumped to {dump_path}")
            del grid_out, mesh
            th.cuda.empty_cache()
            # return


        video_out = imageio.get_writer(
            f'{logger.get_dir()}/triplane_{name_prefix}.mp4',
            mode='I',
            fps=15,
            codec='libx264')

        if planes.shape[1] == 16:  # ffhq/car
            ddpm_latent = {
                self.latent_name: planes[:, :12],
                'bg_plane': planes[:, 12:16],
            }
        else:
            ddpm_latent = {
                self.latent_name: planes,
            }
        
        # TODO, duplicated?
        ddpm_latent.update(rec_model(latent=ddpm_latent, behaviour='decode_after_vae_no_render')) 

        # planes = planes.repeat_interleave(micro['c'].shape[0], 0)

        # for i in range(0, len(c_list), 1): # TODO, larger batch size for eval
        # micro_batchsize = 2
        # micro_batchsize = batch_size

        if render_reference is None:
            render_reference = self.eval_data # compat
        else: # use train_traj
            for key in ['ins', 'bbox', 'caption']:
                if key in render_reference:
                    render_reference.pop(key)
            # render_reference.pop('bbox')
            # render_reference.pop('caption')

            # compat lst for enumerate
            render_reference = [ { k:v[idx:idx+1] for k, v in render_reference.items() } for idx in range(40) ]

        # for i, batch in enumerate(tqdm(self.eval_data)):
        for i, batch in enumerate(tqdm(render_reference)):
            micro = {
                k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v
                for k, v in batch.items() 
            }
            # micro = {'c': batch['c'].to(dist_util.dev()).repeat_interleave(batch_size, 0)}

            # all_pred = []
            pred = rec_model(
                img=None,
                c=micro['c'],
                latent=ddpm_latent,
                # latent={
                #     # k: v.repeat_interleave(micro['c'].shape[0], 0) if v is not None else None
                #     k: v.repeat_interleave(micro['c'].shape[0], 0) if v is not None else None
                #     for k, v in ddpm_latent.items()
                # },
                behaviour='triplane_dec')
            
            # if True:
            # pred_depth = pred['image_depth']
            # pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
            #                                                 pred_depth.min())

            # if 'image_sr' in pred:

            #     gen_img = pred['image_sr']

            #     if pred['image_sr'].shape[-1] == 512:

            #         pred_vis = th.cat([
            #             micro['img_sr'],
            #             self.pool_512(pred['image_raw']), gen_img,
            #             self.pool_512(pred_depth).repeat_interleave(3, dim=1)
            #         ],
            #                           dim=-1)

            #     elif pred['image_sr'].shape[-1] == 128:

            #         pred_vis = th.cat([
            #             micro['img_sr'],
            #             self.pool_128(pred['image_raw']), pred['image_sr'],
            #             self.pool_128(pred_depth).repeat_interleave(3, dim=1)
            #         ],
            #                           dim=-1)

            # else:
            #     gen_img = pred['image_raw']

            #     pred_vis = th.cat(
            #         [
            #             # self.pool_128(micro['img']),
            #             self.pool_128(gen_img),
            #             self.pool_128(pred_depth.repeat_interleave(3, dim=1))
            #         ],
            #         dim=-1)  # B, 3, H, W
            pred_vis = self._make_vis_img(pred)

            if save_img:
                for batch_idx in range(gen_img.shape[0]):
                    sampled_img = Image.fromarray(
                        (gen_img[batch_idx].permute(1, 2, 0).cpu().numpy() *
                         127.5 + 127.5).clip(0, 255).astype(np.uint8))
                    if sampled_img.size != (512, 512):
                        sampled_img = sampled_img.resize(
                            (128, 128), Image.HAMMING)  # for shapenet
                    sampled_img.save(logger.get_dir() +
                                     '/FID_Cals/{}_{}.png'.format(
                                         int(name_prefix) * batch_size +
                                         batch_idx, i))
                    # print('FID_Cals/{}_{}.png'.format(int(name_prefix)*batch_size+batch_idx, i))

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            # if vis.shape[0] > 1:
            #     vis = np.concatenate(np.split(vis, vis.shape[0], axis=0),
            #                          axis=-3)

            # if not save_img:
            for j in range(vis.shape[0]
                        ):  # ! currently only export one plane at a time
                video_out.append_data(vis[j])

        # if not save_img:
        video_out.close()
        del video_out
        print('logged video to: ',
            f'{logger.get_dir()}/triplane_{name_prefix}.mp4')

        del vis, pred_vis, micro, pred, 

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
    # split1 = Path(filename).stem[-6:]
    split1 = Path(filename).stem[-7:]
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
        print('fine ema model', path)
        return path
    else:
        print('fail to find ema model', path)
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
        try:
            logger.logkv_mean(key, values.mean().item())
        except:
            print('type error:', key)
    


def calc_average_loss(all_loss_dicts, verbose=True):
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
        if verbose:
            print(result_str)

    val_scores_for_logging = {
        f'{k}_val': v
        for k, v in mean_all_scores.items()
    }
    return val_scores_for_logging