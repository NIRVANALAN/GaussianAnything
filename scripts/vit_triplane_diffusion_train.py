"""
Train a diffusion model on images.
"""
import json
import sys
import os

sys.path.append('.')

# from dnnlib import EasyDict
import traceback

import torch as th
from xformers.triton import FusedLayerNorm as LayerNorm
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

import argparse
import dnnlib
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
    continuous_diffusion_defaults,
    control_net_defaults,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.continuous_diffusion import make_diffusion as make_sde_diffusion
import nsr
import nsr.lsgm
# from nsr.train_util_diffusion import TrainLoop3DDiffusion as TrainLoop

from datasets.eg3d_dataset import LMDBDataset_MV_Compressed_eg3d
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults, rendering_options_defaults, eg3d_options_default, dataset_defaults
from datasets.shapenet import load_data, load_eval_data, load_memory_data
from nsr.losses.builder import E3DGELossClass

from torch_utils import legacy, misc
from torch.utils.data import Subset
from pdb import set_trace as st

from dnnlib.util import EasyDict, InfiniteSampler
# from .vit_triplane_train_FFHQ import init_dataset_kwargs
from datasets.eg3d_dataset import init_dataset_kwargs

th.backends.cudnn.enabled = True # https://zhuanlan.zhihu.com/p/635824460
th.backends.cudnn.benchmark = True

# from torch.utils.tensorboard import SummaryWriter

SEED = 0


def training_loop(args):
    # def training_loop(args):
    logger.log("dist setup...")
    # th.multiprocessing.set_start_method('spawn')
    th.autograd.set_detect_anomaly(False) # type: ignore
    # th.autograd.set_detect_anomaly(True)  # type: ignore
    # st()

    th.cuda.set_device(
        args.local_rank)  # set this line to avoid extra memory on rank 0
    th.cuda.empty_cache()

    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    dist_util.setup_dist(args)

    # st() # mark

    th.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    th.backends.cudnn.allow_tf32 = args.allow_tf32
    # st()

    # logger.configure(dir=args.logdir, format_strs=["tensorboard", "csv"])
    logger.configure(dir=args.logdir)

    logger.log("creating ViT encoder and NSR decoder...")
    # st() # mark
    device = dist_util.dev()

    args.img_size = [args.image_size_encoder]

    logger.log("creating model and diffusion...")
    # * set denoise model args

    if args.denoise_in_channels == -1:
        args.diffusion_input_size = args.image_size_encoder
        args.denoise_in_channels = args.out_chans
        args.denoise_out_channels = args.out_chans
    else:
        assert args.denoise_out_channels != -1

    # args.image_size = args.image_size_encoder  # 224, follow the triplane size

    # if args.diffusion_input_size == -1:
    # else:
    # args.image_size = args.diffusion_input_size

    if args.pred_type == 'v':  # for lsgm training
        assert args.predict_v == True  # for DDIM sampling
    
    # if not args.create_dit:

    denoise_model, diffusion = create_model_and_diffusion(
        **args_to_dict(args,
                       model_and_diffusion_defaults().keys()))

    opts = eg3d_options_default()
    if args.sr_training:
        args.sr_kwargs = dnnlib.EasyDict(
            channel_base=opts.cbase,
            channel_max=opts.cmax,
            fused_modconv_default='inference_only',
            use_noise=True
        )  # ! close noise injection? since noise_mode='none' in eg3d

    logger.log("creating encoder and NSR decoder...")
    auto_encoder = create_3DAE_model(
        **args_to_dict(args,
                       encoder_and_nsr_defaults().keys()))

    auto_encoder.to(device)
    auto_encoder.eval()

    # * load G_ema modules into autoencoder
    # * clone G_ema.decoder to auto_encoder triplane
    # logger.log("AE triplane decoder reuses G_ema decoder...")
    # auto_encoder.decoder.register_buffer('w_avg', G_ema.backbone.mapping.w_avg)

    # auto_encoder.decoder.triplane_decoder.decoder.load_state_dict(  # type: ignore
    #     G_ema.decoder.state_dict())  # type: ignore

    # set grad=False in this manner suppresses the DDP forward no grad error.

    # if args.sr_training:

    #     logger.log("AE triplane decoder reuses G_ema SR module...")
    #     # auto_encoder.decoder.triplane_decoder.superresolution.load_state_dict(  # type: ignore
    #     #     G_ema.superresolution.state_dict())  # type: ignore

    #     # set grad=False in this manner suppresses the DDP forward no grad error.
    # logger.log("freeze SR module...")
    # for param in auto_encoder.decoder.superresolution.parameters(): # type: ignore
    #     param.requires_grad_(False)

    #     # del G_ema
    #     th.cuda.empty_cache()

    if args.freeze_triplane_decoder:
        logger.log("freeze triplane decoder...")
        for param in auto_encoder.decoder.triplane_decoder.parameters(
        ):  # type: ignore
            # for param in auto_encoder.decoder.triplane_decoder.decoder.parameters(): # type: ignore
            param.requires_grad_(False)

    if args.cfg in ('afhq', 'ffhq'):

        if args.sr_training:

            logger.log("AE triplane decoder reuses G_ema SR module...")
            auto_encoder.decoder.triplane_decoder.superresolution.load_state_dict(  # type: ignore
                G_ema.superresolution.state_dict())  # type: ignore

            # set grad=False in this manner suppresses the DDP forward no grad error.
            for param in auto_encoder.decoder.triplane_decoder.superresolution.parameters(
            ):  # type: ignore
                param.requires_grad_(False)

        # ! load data
        if args.use_lmdb:
            logger.log("creating LMDB eg3d data loader...")
            training_set = LMDBDataset_MV_Compressed_eg3d(
                args.data_dir,
                args.image_size,
                args.image_size_encoder,
            )
        else:
            logger.log("creating eg3d data loader...")

            training_set_kwargs, dataset_name = init_dataset_kwargs(
                data=args.data_dir,
                class_name='datasets.eg3d_dataset.ImageFolderDataset',
                reso_gt=args.image_size)  # only load pose here
            # if args.cond and not training_set_kwargs.use_labels:
            # raise Exception('check here')

            # training_set_kwargs.use_labels = args.cond
            training_set_kwargs.use_labels = True
            training_set_kwargs.xflip = False
            training_set_kwargs.random_seed = SEED
            training_set_kwargs.max_size = args.dataset_size
            # desc = f'{args.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'

            # * construct ffhq/afhq dataset
            training_set = dnnlib.util.construct_class_by_name(
                **training_set_kwargs)  # subclass of training.dataset.Dataset

        training_set_sampler = InfiniteSampler(
            dataset=training_set,
            rank=dist_util.get_rank(),
            num_replicas=dist_util.get_world_size(),
            seed=SEED)

        data = iter(
            th.utils.data.DataLoader(
                dataset=training_set,
                sampler=training_set_sampler,
                batch_size=args.batch_size,
                pin_memory=True,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                prefetch_factor=max(8 // args.batch_size, 2),
            ))
        #  prefetch_factor=2))

        eval_data = th.utils.data.DataLoader(dataset=Subset(
            training_set, np.arange(8)),
                                             batch_size=args.eval_batch_size,
                                             num_workers=1)

    else:

        logger.log("creating data loader...")

        if args.objv_dataset:
            from datasets.g_buffer_objaverse import load_data, load_eval_data, load_memory_data, load_wds_data
        else:  # shapenet
            from datasets.shapenet import load_data, load_eval_data, load_memory_data

        # TODO, load shapenet data
        # data = load_data(
        # st() mark
        # if args.overfitting:
        #     logger.log("create overfitting memory dataset")
        #     data = load_memory_data(
        #         file_path=args.eval_data_dir,
        #         batch_size=args.batch_size,
        #         reso=args.image_size,
        #         reso_encoder=args.image_size_encoder,  # 224 -> 128
        #         num_workers=args.num_workers,
        #         load_depth=True  # for evaluation
        #     )
        # else:
        if args.use_wds:
            if args.data_dir == 'NONE':
                with open(args.shards_lst) as f:
                    shards_lst = [url.strip() for url in f.readlines()]
                data = load_wds_data(
                    shards_lst, args.image_size, args.image_size_encoder,
                    args.batch_size, args.num_workers,
                    **args_to_dict(args,
                                   dataset_defaults().keys()))

            else:
                data = load_wds_data(
                    args.data_dir, args.image_size, args.image_size_encoder,
                    args.batch_size, args.num_workers,
                    **args_to_dict(args,
                                   dataset_defaults().keys()))

                # eval_data = load_wds_data(
                #     args.data_dir,
                #     args.image_size,
                #     args.image_size_encoder,
                #     args.eval_batch_size,
                #     args.num_workers,
                #     decode_encode_img_only=args.decode_encode_img_only,
                #     load_wds_diff=args.load_wds_diff)

            if args.eval_data_dir == 'NONE':
                with open(args.eval_shards_lst) as f:
                    eval_shards_lst = [url.strip() for url in f.readlines()]
            else:
                eval_shards_lst = args.eval_data_dir  # auto expanded

            eval_data = load_wds_data(
                eval_shards_lst,
                args.image_size,
                args.image_size_encoder,
                args.eval_batch_size,
                args.num_workers,
                plucker_embedding=args.plucker_embedding,
                decode_encode_img_only=args.decode_encode_img_only,
                mv_input=args.mv_input,
                load_wds_diff=False,
                load_instance=True)

        else:
            logger.log("create all instances dataset")
            # st() mark
            data = load_data(
                file_path=args.data_dir,
                batch_size=args.batch_size,
                reso=args.image_size,
                reso_encoder=args.image_size_encoder,  # 224 -> 128
                num_workers=args.num_workers,
                **args_to_dict(args,
                                dataset_defaults().keys())
                # load_depth=args.load_depth,
                # preprocess=auto_encoder.preprocess,  # clip
                # dataset_size=args.dataset_size,
                # use_lmdb=args.use_lmdb,
                # trainer_name=args.trainer_name,
                # load_depth=True # for evaluation
            )
            eval_data = data
            # eval_data = load_eval_data(
            #     file_path=args.eval_data_dir,
            #     batch_size=args.eval_batch_size,
            #     reso=args.image_size,
            #     reso_encoder=args.image_size_encoder,  # 224 -> 128
            #     num_workers=args.num_workers,
            #     load_depth=True,  # for evaluation
            #     interval=args.interval,
            #     use_lmdb=args.use_lmdb,
            # )

    # let all processes sync up before starting with a new epoch of training

    if dist_util.get_rank() == 0:
        with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    args.schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion)

    opt = dnnlib.EasyDict(args_to_dict(args, loss_defaults().keys()))
    loss_class = E3DGELossClass(device, opt).to(device)

    logger.log("training...")

    TrainLoop = {
        'adm':
        nsr.TrainLoop3DDiffusion,
        'dit':
        nsr.TrainLoop3DDiffusionDiT,
        'ssd':
        nsr.TrainLoop3DDiffusionSingleStage,
        # 'ssd_cvD': nsr.TrainLoop3DDiffusionSingleStagecvD,
        'ssd_cvD_sds':
        nsr.TrainLoop3DDiffusionSingleStagecvDSDS,
        'ssd_cvd_sds_no_separate_sds_step':
        nsr.TrainLoop3DDiffusionSingleStagecvDSDS_sdswithrec,
        'vpsde_lsgm_noD':
        nsr.lsgm.TrainLoop3DDiffusionLSGM_noD,  # use vpsde
        'vpsde_TrainLoop3DDiffusionLSGM_cvD':
        nsr.lsgm.TrainLoop3DDiffusionLSGM_cvD,
        'vpsde_TrainLoop3DDiffusionLSGM_cvD_scaling':
        nsr.lsgm.TrainLoop3DDiffusionLSGM_cvD_scaling,
        'vpsde_TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm':
        nsr.lsgm.TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm,
        'vpsde_TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD':
        nsr.lsgm.TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD,
        'vpsde_TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_weightingv0':
        nsr.lsgm.
        TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_weightingv0,
        'vpsde_TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_iterativeED':
        nsr.lsgm.
        TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_iterativeED,
        'vpsde_TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_iterativeED_nv':
        nsr.lsgm.
        TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_iterativeED_nv,
        'vpsde_lsgm_joint_noD':
        nsr.lsgm.TrainLoop3DDiffusionLSGMJointnoD,  # use vpsde
        'vpsde_lsgm_joint_noD_ponly':
        nsr.lsgm.TrainLoop3DDiffusionLSGMJointnoD_ponly,  # use vpsde
        # control
        'vpsde_cldm':
        nsr.lsgm.TrainLoop3DDiffusionLSGM_Control,
        'vpsde_crossattn':
        nsr.lsgm.TrainLoop3DDiffusionLSGM_crossattn,
        'vpsde_crossattn_cldm':
        nsr.lsgm.crossattn_cldm.TrainLoop3DDiffusionLSGM_crossattn_controlNet,
        'vpsde_ldm':
        nsr.lsgm.TrainLoop3D_LDM,
        'sgm_legacy':
        nsr.lsgm.sgm_DiffusionEngine.DiffusionEngineLSGM,
    }[args.trainer_name]

    if 'vpsde' in args.trainer_name:
        sde_diffusion = make_sde_diffusion(
            dnnlib.EasyDict(
                args_to_dict(args,
                             continuous_diffusion_defaults().keys())))
        # assert args.mixed_prediction, 'enable mixed_prediction by default'
        logger.log('create VPSDE diffusion.')
    else:
        sde_diffusion = None

    if 'cldm' in args.trainer_name:
        assert isinstance(denoise_model, tuple)
        denoise_model, controlNet = denoise_model

        controlNet.to(dist_util.dev())
        controlNet.train()
    else:
        controlNet = None

    # st()
    denoise_model.to(dist_util.dev())
    denoise_model.train()

    auto_encoder.decoder.rendering_kwargs = args.rendering_kwargs
    TrainLoop(rec_model=auto_encoder,
              denoise_model=denoise_model,
              control_model=controlNet,
              diffusion=diffusion,
              sde_diffusion=sde_diffusion,
              loss_class=loss_class,
              data=data,
              eval_data=eval_data,
              **vars(args)).run_loop()

    dist_util.synchronize()


def create_argparser(**kwargs):
    # defaults.update(model_and_diffusion_defaults())

    defaults = dict(
        dataset_size=-1,
        diffusion_input_size=-1,
        trainer_name='adm',
        use_amp=False,
        train_vae=True,  # jldm?
        triplane_scaling_divider=1.0,  # divide by this value
        overfitting=False,
        num_workers=4,
        image_size=128,
        image_size_encoder=224,
        iterations=150000,
        schedule_sampler="uniform",
        anneal_lr=False,
        lr=5e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        eval_batch_size=12,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        eval_interval=2500,
        save_interval=10000,
        resume_checkpoint="",
        resume_checkpoint_EG3D="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        data_dir="",
        eval_data_dir="",
        load_depth=True,  # TODO
        logdir="/mnt/lustre/yslan/logs/nips23/",
        load_submodule_name='',  # for loading pretrained auto_encoder model
        ignore_resume_opt=False,
        # freeze_ae=False,
        denoised_ae=True,
        diffusion_ce_anneal=False,
        use_lmdb=False,
        interval=1,
        freeze_triplane_decoder=False,
        objv_dataset=False,
        use_eos_feature=False,
        clip_grad_throld=1.0,
        allow_tf32=True,
    )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(continuous_diffusion_defaults())
    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(dataset_defaults())  # type: ignore
    defaults.update(loss_defaults())
    defaults.update(control_net_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    # os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    # os.environ["NCCL_DEBUG"] = "INFO"
    th.multiprocessing.set_start_method('spawn')

    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.

    args = create_argparser().parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus = th.cuda.device_count()

    # opts = dnnlib.EasyDict(vars(args))  # compatiable with triplane original settings
    # opts = args
    args.rendering_kwargs = rendering_options_defaults(args)

    # Launch processes.
    logger.log('Launching processes...')

    logger.log('Available devices ', th.cuda.device_count())
    logger.log('Current cuda device ', th.cuda.current_device())
    # logger.log('GPU Device name:', th.cuda.get_device_name(th.cuda.current_device()))

    try:
        training_loop(args)
    # except KeyboardInterrupt as e:
    except Exception as e:
        # print(e)
        traceback.print_exc()
        dist_util.cleanup()  # clean port and socket when ctrl+c
