"""
Train a diffusion model on images.
"""
from pdb import set_trace as st
import random
import json
import sys
import os

sys.path.append('.')

import torch.distributed as dist

import traceback

import torch as th

from xformers.components.feedforward import fused_mlp

# if th.cuda.is_available(): # FIXME
#     from xformers.triton import FusedLayerNorm as LayerNorm

import torch.multiprocessing as mp
import numpy as np

import argparse
import dnnlib
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
from nsr.train_nv_util import TrainLoop3DRecNV, TrainLoop3DRec, TrainLoop3DRecNVPatch, TrainLoop3DRecNVPatchSingleForward, TrainLoop3DRecNVPatchSingleForwardMV, TrainLoop3DRecNVPatchSingleForwardMV_NoCrop, TrainLoop3DRecNVPatchSingleForwardMVAdvLoss, TrainLoop3DRecNVPatchSingleForwardMV_NoCrop_adv
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults, rendering_options_defaults, eg3d_options_default, dataset_defaults
# from datasets import g_buffer_objaverse
from nsr.losses.builder import E3DGELossClass, E3DGE_with_AdvLoss


# th.backends.cuda.matmul.allow_tf32 = True # https://huggingface.co/docs/diffusers/optimization/fp16
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True
th.backends.cudnn.enabled = True


def training_loop(args):
    # def training_loop(args):
    dist_util.setup_dist(args)
    # th.autograd.set_detect_anomaly(True) # type: ignore
    th.autograd.set_detect_anomaly(False)  # type: ignore
    # https://blog.csdn.net/qq_41682740/article/details/126304613

    SEED = args.seed

    # dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=th.cuda.device_count())
    logger.log(f"global_rank={args.global_rank}, local_rank={args.local_rank} init complete, seed={SEED}")
    th.cuda.set_device(args.local_rank)
    th.cuda.empty_cache()

    # * deterministic algorithms flags
    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # logger.configure(dir=args.logdir, format_strs=["tensorboard", "csv"])
    logger.configure(dir=args.logdir)

    logger.log("creating encoder and NSR decoder...")
    # device = dist_util.dev()
    device = th.device("cuda", args.local_rank)

    # shared eg3d opts
    opts = eg3d_options_default()

    if args.sr_training:
        args.sr_kwargs = dnnlib.EasyDict(
            channel_base=opts.cbase,
            channel_max=opts.cmax,
            fused_modconv_default='inference_only',
            use_noise=True
        )  # ! close noise injection? since noise_mode='none' in eg3d

    auto_encoder = create_3DAE_model(
        **args_to_dict(args,
                       encoder_and_nsr_defaults().keys()))
    auto_encoder.to(device)
    auto_encoder.train()

    logger.log("creating data loader...")
    # data = load_data(
    # st()
    if args.objv_dataset:
        from datasets.g_buffer_objaverse import load_data, load_eval_data, load_memory_data, load_wds_data
    else:  # shapenet
        from datasets.shapenet import load_data, load_eval_data, load_memory_data

    if args.overfitting:
        data = load_memory_data(
            file_path=args.data_dir,
            batch_size=args.batch_size,
            reso=args.image_size,
            reso_encoder=args.image_size_encoder,  # 224 -> 128
            num_workers=args.num_workers,
            # load_depth=args.depth_lambda > 0
            # load_depth=True,  # for evaluation
                    **args_to_dict(args,
                                   dataset_defaults().keys()))
        eval_data = None
    else:
        if args.use_wds:
            # st()
            if args.data_dir == 'NONE':
                with open(args.shards_lst) as f:
                    shards_lst = [url.strip() for url in f.readlines()]
                data = load_wds_data(
                    shards_lst,  # type: ignore
                    args.image_size,
                    args.image_size_encoder,
                    args.batch_size,
                    args.num_workers,
                    #  plucker_embedding=args.plucker_embedding,
                    #  mv_input=args.mv_input,
                    #  split_chunk_input=args.split_chunk_input,
                    **args_to_dict(args,
                                   dataset_defaults().keys()))

            elif not args.inference:
                data = load_wds_data(args.data_dir,
                                     args.image_size,
                                     args.image_size_encoder,
                                     args.batch_size,
                                     args.num_workers,
                                     plucker_embedding=args.plucker_embedding,
                                     mv_input=args.mv_input,
                                     split_chunk_input=args.split_chunk_input)
            else:
                data = None
            # ! load eval

            if args.eval_data_dir == 'NONE':
                with open(args.eval_shards_lst) as f:
                    eval_shards_lst = [url.strip() for url in f.readlines()]
            else:
                eval_shards_lst = args.eval_data_dir  # auto expanded

            eval_data = load_wds_data(
                eval_shards_lst,  # type: ignore
                args.image_size,
                args.image_size_encoder,
                args.eval_batch_size,
                args.num_workers,
                **args_to_dict(args,
                               dataset_defaults().keys()))
            # load_instance=True) # TODO

        else:

            if args.inference:
                data = None
            else:
                data = load_data(
                    file_path=args.data_dir,
                    batch_size=args.batch_size,
                    reso=args.image_size,
                    reso_encoder=args.image_size_encoder,  # 224 -> 128
                    num_workers=args.num_workers,
                    **args_to_dict(args,
                                   dataset_defaults().keys())
                                   )

                    # load_depth=True # for evaluation

            if args.pose_warm_up_iter > 0:
                overfitting_dataset = load_memory_data(
                    file_path=args.data_dir,
                    batch_size=args.batch_size,
                    reso=args.image_size,
                    reso_encoder=args.image_size_encoder,  # 224 -> 128
                    num_workers=args.num_workers,
                    # load_depth=args.depth_lambda > 0
                    # load_depth=True  # for evaluation
                    **args_to_dict(args,
                                   dataset_defaults().keys()))
                data = [data, overfitting_dataset, args.pose_warm_up_iter]

            eval_data = load_eval_data(
                file_path=args.eval_data_dir,
                batch_size=args.eval_batch_size,
                reso=args.image_size,
                reso_encoder=args.image_size_encoder,  # 224 -> 128
                num_workers=args.num_workers,
                load_depth=True,  # for evaluation
                preprocess=auto_encoder.preprocess,
                wds_split=args.wds_split,
                # interval=args.interval,
                # use_lmdb=args.use_lmdb,
                # plucker_embedding=args.plucker_embedding,
                # load_real=args.load_real,
                # four_view_for_latent=args.four_view_for_latent,
                # load_extra_36_view=args.load_extra_36_view,
                # shuffle_across_cls=args.shuffle_across_cls,
                **args_to_dict(args,
                               dataset_defaults().keys()))

    logger.log("creating data loader done...")

    args.img_size = [args.image_size_encoder]
    # try dry run
    # batch = next(data)
    # batch = None

    # logger.log("creating model and diffusion...")

    # let all processes sync up before starting with a new epoch of training
    dist_util.synchronize()

    # schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    opt = dnnlib.EasyDict(args_to_dict(args, loss_defaults().keys()))
    # opt.max_depth, opt.min_depth = args.rendering_kwargs.ray_end, args.rendering_kwargs.ray_start
    if 'disc' in args.trainer_name:
        loss_class = E3DGE_with_AdvLoss(
            device,
            opt,
            # disc_weight=args.patchgan_disc, # rec_cvD_lambda
            disc_factor=args.patchgan_disc_factor,  # reduce D update speed
            disc_weight=args.patchgan_disc_g_weight).to(device)
    else:
        loss_class = E3DGELossClass(device, opt).to(device)

    # writer = SummaryWriter() # TODO, add log dir

    logger.log("training...")

    TrainLoop = {
        'input_rec': TrainLoop3DRec,
        'nv_rec': TrainLoop3DRecNV,
        # 'nv_rec_patch': TrainLoop3DRecNVPatch,
        'nv_rec_patch': TrainLoop3DRecNVPatchSingleForward,
        'nv_rec_patch_mvE': TrainLoop3DRecNVPatchSingleForwardMV,
        'nv_rec_patch_mvE_disc': TrainLoop3DRecNVPatchSingleForwardMVAdvLoss,
        'nv_rec_patch_mvE_gs': TrainLoop3DRecNVPatchSingleForwardMV_NoCrop,
        'nv_rec_patch_mvE_gs_disc': TrainLoop3DRecNVPatchSingleForwardMV_NoCrop_adv,
    }[args.trainer_name]

    logger.log("creating TrainLoop done...")

    # th._dynamo.config.verbose=True # th212 required
    # th._dynamo.config.suppress_errors = True
    auto_encoder.decoder.rendering_kwargs = args.rendering_kwargs
    train_loop = TrainLoop(
        rec_model=auto_encoder,
        loss_class=loss_class,
        data=data,
        eval_data=eval_data,
        #   compile=args.compile,
        **vars(args))
    # train_loop.rendering_kwargs = args.rendering_kwargs

    if args.inference:
        camera = th.load('eval_pose.pt', map_location=dist_util.dev())
        train_loop.eval_novelview_loop(camera=camera,
                                       save_latent=args.save_latent)
    else:
        train_loop.run_loop()


def create_argparser(**kwargs):
    # defaults.update(model_and_diffusion_defaults())

    defaults = dict(
        seed=0,
        dataset_size=-1,
        trainer_name='input_rec',
        use_amp=False,
        overfitting=False,
        num_workers=4,
        image_size=128,
        image_size_encoder=224,
        iterations=150000,
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
        use_fp16=False,
        fp16_scale_growth=1e-3,
        data_dir="",
        eval_data_dir="",
        # load_depth=False, # TODO
        logdir="/mnt/lustre/yslan/logs/nips23/",
        # test warm up pose sampling training
        pose_warm_up_iter=-1,
        inference=False,
        export_latent=False,
        save_latent=False,
        wds_split=1,  # out of 4
    )

    defaults.update(dataset_defaults())  # type: ignore
    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(loss_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    th.multiprocessing.set_start_method('spawn')

    # os.environ[
    # "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    # os.environ["NCCL_DEBUG"]="INFO"

    args = create_argparser().parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    # if os.environ['WORLD_SIZE'] > 1:

    # for multi-node training
    if dist_util.get_world_size() > 1:
        args.global_rank = int(os.environ["RANK"])
    else:
        args.global_rank = 0

    args.gpus = th.cuda.device_count()

    opts = args

    args.rendering_kwargs = rendering_options_defaults(opts)

    # print(args)
    with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Launch processes.
    print('Launching processes...')

    try:
        training_loop(args)
    # except KeyboardInterrupt as e:
    except Exception as e:
        # print(e)
        traceback.print_exc()
        dist_util.cleanup()  # clean port and socket when ctrl+c
