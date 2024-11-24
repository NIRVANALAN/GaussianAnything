"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import json
import sys
import os

sys.path.append('.')

from pdb import set_trace as st
import imageio
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    continuous_diffusion_defaults,
    control_net_defaults,
)

th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True
th.backends.cudnn.enabled = True

from pathlib import Path

from tqdm import tqdm, trange
import dnnlib
from nsr.train_util_diffusion import TrainLoop3DDiffusion as TrainLoop
from guided_diffusion.continuous_diffusion import make_diffusion as make_sde_diffusion
import nsr
import nsr.lsgm
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults, AE_with_Diffusion, rendering_options_defaults, eg3d_options_default, dataset_defaults

from datasets.shapenet import load_eval_data
from torch.utils.data import Subset
from datasets.eg3d_dataset import init_dataset_kwargs

SEED = 0


def main(args):

    # args.rendering_kwargs = rendering_options_defaults(args)

    dist_util.setup_dist(args)
    logger.configure(dir=args.logdir)

    th.cuda.empty_cache()

    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # * set denoise model args
    logger.log("creating model and diffusion...")
    args.img_size = [args.image_size_encoder]
    # ! no longer required for LDM
    # args.denoise_in_channels = args.out_chans
    # args.denoise_out_channels = args.out_chans
    args.image_size = args.image_size_encoder  # 224, follow the triplane size

    denoise_model, diffusion = create_model_and_diffusion(
        **args_to_dict(args,
                       model_and_diffusion_defaults().keys()))

    if 'cldm' in args.trainer_name:
        assert isinstance(denoise_model, tuple)
        denoise_model, controlNet = denoise_model

        controlNet.to(dist_util.dev())
        controlNet.train()
    else:
        controlNet = None

    opts = eg3d_options_default()
    if args.sr_training:
        args.sr_kwargs = dnnlib.EasyDict(
            channel_base=opts.cbase,
            channel_max=opts.cmax,
            fused_modconv_default='inference_only',
            use_noise=True
        )  # ! close noise injection? since noise_mode='none' in eg3d

    # denoise_model.load_state_dict(
    #     dist_util.load_state_dict(args.ddpm_model_path, map_location="cpu"))
    denoise_model.to(dist_util.dev())
    if args.use_fp16:
        denoise_model.convert_to_fp16()
    denoise_model.eval()

    # * auto-encoder reconstruction model
    logger.log("creating 3DAE...")
    auto_encoder = create_3DAE_model(
        **args_to_dict(args,
                       encoder_and_nsr_defaults().keys()))

    # logger.log("AE triplane decoder reuses G_ema decoder...")
    # auto_encoder.decoder.register_buffer('w_avg', G_ema.backbone.mapping.w_avg)

    # print(auto_encoder.decoder.w_avg.shape) # [512]

    # auto_encoder.load_state_dict(
    #     dist_util.load_state_dict(args.rec_model_path, map_location="cpu"))

    auto_encoder.to(dist_util.dev())
    auto_encoder.eval()

    # TODO, how to set the scale?
    logger.log("create dataset")

    if args.objv_dataset:
        from datasets.g_buffer_objaverse import load_data, load_eval_data, load_memory_data, load_wds_data
    else:  # shapenet
        from datasets.shapenet import load_data, load_eval_data, load_memory_data

    if args.cfg in ('afhq', 'ffhq'):
        # ! load data
        logger.log("creating eg3d data loader...")
        training_set_kwargs, dataset_name = init_dataset_kwargs(
            data=args.data_dir,
            class_name='datasets.eg3d_dataset.ImageFolderDataset'
        )  # only load pose here
        # if args.cond and not training_set_kwargs.use_labels:
        # raise Exception('check here')

        # training_set_kwargs.use_labels = args.cond
        training_set_kwargs.use_labels = True
        training_set_kwargs.xflip = True
        training_set_kwargs.random_seed = SEED
        # desc = f'{args.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'

        # * construct ffhq/afhq dataset
        training_set = dnnlib.util.construct_class_by_name(
            **training_set_kwargs)  # subclass of training.dataset.Dataset

        training_set = dnnlib.util.construct_class_by_name(
            **training_set_kwargs)  # subclass of training.dataset.Dataset

        # training_set_sampler = InfiniteSampler(
        #     dataset=training_set,
        #     rank=dist_util.get_rank(),
        #     num_replicas=dist_util.get_world_size(),
        #     seed=SEED)

        # data = iter(
        #     th.utils.data.DataLoader(dataset=training_set,
        #                             sampler=training_set_sampler,
        #                             batch_size=args.batch_size,
        #                             pin_memory=True,
        #                             num_workers=args.num_workers,))
        #                             #  prefetch_factor=2))

        eval_data = th.utils.data.DataLoader(dataset=Subset(
            training_set, np.arange(25)),
                                             batch_size=args.eval_batch_size,
                                             num_workers=1)

    else:

        logger.log("creating data loader...")

        if args.use_wds:
            if args.eval_data_dir == 'NONE':
                with open(args.eval_shards_lst) as f:
                    eval_shards_lst = [url.strip() for url in f.readlines()]
            else:
                eval_shards_lst = args.eval_data_dir  # auto expanded

            eval_data = load_wds_data(
                eval_shards_lst, args.image_size, args.image_size_encoder,
                args.eval_batch_size, args.num_workers,
                **args_to_dict(args,
                               dataset_defaults().keys()))

        else:
            eval_data = load_eval_data(
                file_path=args.eval_data_dir,
                batch_size=args.eval_batch_size,
                reso=args.image_size,
                reso_encoder=args.image_size_encoder,  # 224 -> 128
                num_workers=args.num_workers,
                # load_depth=True,  # for evaluation
                **args_to_dict(args,
                               dataset_defaults().keys()))

    TrainLoop = {
        'adm': nsr.TrainLoop3DDiffusion,
        'dit': nsr.TrainLoop3DDiffusionDiT,
        # lsgm
        'ssd': nsr.TrainLoop3DDiffusionSingleStage,
        # 'ssd_cvD': nsr.TrainLoop3DDiffusionSingleStagecvD,
        'ssd_cvD_sds': nsr.TrainLoop3DDiffusionSingleStagecvDSDS,
        'ssd_cvd_sds_no_separate_sds_step':
        nsr.TrainLoop3DDiffusionSingleStagecvDSDS_sdswithrec,
        # 'ssd_cvD': nsr.TrainLoop3DDiffusionSingleStagecvD,
        'ssd_cvD_sds': nsr.TrainLoop3DDiffusionSingleStagecvDSDS,
        'vpsde_lsgm_noD': nsr.lsgm.TrainLoop3DDiffusionLSGM_noD,  # use vpsde
        'vpsde_cldm': nsr.lsgm.TrainLoop3DDiffusionLSGM_Control,
        'vpsde_TrainLoop3DDiffusionLSGM_cvD':
        nsr.lsgm.TrainLoop3DDiffusionLSGM_cvD,
        'vpsde_lsgm_joint_noD':
        nsr.lsgm.TrainLoop3DDiffusionLSGMJointnoD,  # use vpsde
        'vpsde_lsgm_joint_noD_ponly':
        nsr.lsgm.TrainLoop3DDiffusionLSGMJointnoD_ponly,  # use vpsde
        'vpsde_crossattn': nsr.lsgm.TrainLoop3DDiffusionLSGM_crossattn,
        'vpsde_ldm': nsr.lsgm.TrainLoop3D_LDM,
        'sgm_legacy':
        nsr.lsgm.sgm_DiffusionEngine.DiffusionEngineLSGM,
    }[args.trainer_name]

    # continuous
    if 'vpsde' in args.trainer_name:
        sde_diffusion = make_sde_diffusion(
            dnnlib.EasyDict(
                args_to_dict(args,
                             continuous_diffusion_defaults().keys())))
        # assert args.mixed_prediction, 'enable mixed_prediction by default'
        logger.log('create VPSDE diffusion.')
    else:
        sde_diffusion = None

    # if 'cldm' in args.trainer_name:
    #     assert isinstance(denoise_model, tuple)
    #     denoise_model, controlNet = denoise_model

    #     controlNet.to(dist_util.dev())
    #     controlNet.train()
    # else:
    #     controlNet = None
    auto_encoder.decoder.rendering_kwargs = args.rendering_kwargs

    training_loop_class = TrainLoop(rec_model=auto_encoder,
                                    denoise_model=denoise_model,
                                    control_model=controlNet,
                                    diffusion=diffusion,
                                    sde_diffusion=sde_diffusion,
                                    loss_class=None,
                                    data=None,
                                    eval_data=eval_data,
                                    **vars(args))

    logger.log("sampling...")
    dist_util.synchronize()

    # all_images = []
    # all_labels = []
    # while len(all_images) * args.batch_size < args.num_samples:

    if dist_util.get_rank() == 0:

        (Path(logger.get_dir()) / 'FID_Cals').mkdir(exist_ok=True,
                                                    parents=True)

        with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

        # ! use pre-saved camera pose
        camera = th.load('eval_pose.pt', map_location=dist_util.dev())[:]

        # for sample_idx in trange(args.num_samples):
        model_kwargs = {}

        # if args.class_cond:
        #     classes = th.randint(low=0,
        #                          high=NUM_CLASSES,
        #                          size=(args.batch_size, ),
        #                          device=dist_util.dev())
        #     model_kwargs["y"] = classes
        # training_loop_class.step = sample_idx  # save to different position
        # if args.create_controlnet or 'crossattn' in args.trainer_name:

        training_loop_class.eval_cldm(
            prompt=args.prompt,
            unconditional_guidance_scale=args.
            unconditional_guidance_scale,
            use_ddim=args.use_ddim,
            save_img=args.save_img,
            use_train_trajectory=args.use_train_trajectory,
            camera=camera,
            num_instances=args.num_instances,
            num_samples=args.num_samples,
            # training_loop_class.rec_model,
            # training_loop_class.ddpm_model
        )


    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        image_size_encoder=224,
        triplane_scaling_divider=1.0,  # divide by this value
        diffusion_input_size=-1,
        trainer_name='adm',
        use_amp=False,
        # triplane_scaling_divider=1.0, # divide by this value

        # * sampling flags
        clip_denoised=False,
        num_samples=10,
        num_instances=10, # for i23d, loop different condition
        use_ddim=False,
        ddpm_model_path="",
        cldm_model_path="",
        rec_model_path="",

        # * eval logging flags
        logdir="/mnt/lustre/yslan/logs/nips23/",
        data_dir="",
        eval_data_dir="",
        eval_batch_size=1,
        num_workers=1,

        # * training flags for loading TrainingLoop class
        overfitting=False,
        image_size=128,
        iterations=150000,
        schedule_sampler="uniform",
        anneal_lr=False,
        lr=5e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        eval_interval=2500,
        save_interval=10000,
        resume_checkpoint="",
        resume_cldm_checkpoint="",
        resume_checkpoint_EG3D="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        load_submodule_name='',  # for loading pretrained auto_encoder model
        ignore_resume_opt=False,
        freeze_ae=False,
        denoised_ae=True,
        # inference prompt
        prompt="a red chair",
        interval=1,
        save_img=False,
        use_train_trajectory=
        False,  # use train trajectory to sample images for fid calculation
        unconditional_guidance_scale=1.0,
        use_eos_feature=False,
    )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(loss_defaults())
    defaults.update(continuous_diffusion_defaults())
    defaults.update(control_net_defaults())
    defaults.update(dataset_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":

    # os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    # os.environ["NCCL_DEBUG"] = "INFO"

    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.

    args = create_argparser().parse_args()

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus = th.cuda.device_count()

    args.rendering_kwargs = rendering_options_defaults(args)

    main(args)
