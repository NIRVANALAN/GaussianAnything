"""
Train a diffusion model on images.
"""
import sys
import os
sys.path.append('.')

import torch as th
import torch.multiprocessing as mp

import argparse
import dnnlib
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
from nsr.train_util import TrainLoop3DTriplaneRec as TrainLoop
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, create_Triplane, loss_defaults
from datasets.g_buffer_objaverse import load_data, load_memory_data
from nsr.losses.builder import E3DGELossClass

from pdb import set_trace as st

from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults, rendering_options_defaults, eg3d_options_default

import random
import json
import sys
import os

sys.path.append('.')
import torch.distributed as dist

import traceback


# def training_loop(rank, master_addr, master_port, args):
def training_loop(args):
    # dist_util.setup_dist(rank, master_addr, master_port, args.gpus)
    dist_util.setup_dist(args)

    # logger.configure(dir=args.logdir, format_strs=["tensorboard", "csv"])
    logger.configure(dir=args.logdir)

    # print(args)
    with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    logger.log("creating data loader...")
    # TODO, load shapenet data
    # data = load_data(
    data = load_memory_data(
        file_path=args.data_dir,
        batch_size=args.batch_size,
        reso=args.image_size,
        reso_encoder=args.image_size_encoder, # 224 -> 128
        num_workers=args.num_workers,
        load_depth=args.depth_lambda > 0
        # load_depth=True # for evaluation
    )
    eval_data = data
    # eval_data = load_eval_data(
    #     file_path=args.data_dir,
    #     batch_size=args.eval_batch_size,
    #     reso=args.image_size,
    #     reso_encoder=args.image_size_encoder, # 224 -> 128
    #     num_workers=args.num_workers,
    #     load_depth=True # for evaluation
    # )
    # try dry run
    # batch = next(data)
    # batch = None

    # logger.log("creating model and diffusion...")
    logger.log("creating encoder and NSR decoder...")

    
    auto_encoder = create_Triplane( # basically overfitting tirplane
        **args_to_dict(args,
                       encoder_and_nsr_defaults().keys()))

    # auto_encoder = create_3DAE_model(
    #     **args_to_dict(args,
    #                    encoder_and_nsr_defaults().keys()))
    auto_encoder.to(dist_util.dev())

    # schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)


    opt = dnnlib.EasyDict(args_to_dict(args, loss_defaults().keys()) )
    loss_class = E3DGELossClass(dist_util.dev(), opt).to(dist_util.dev())

    logger.log("training...")
    TrainLoop(
        rec_model=auto_encoder,
        # encoder,
        # decoder
        loss_class=loss_class,
        # diffusion=diffusion,
        data=data,
        eval_data=eval_data,
        # data=batch,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        eval_interval=args.eval_interval,
    ).run_loop() # ! overfitting


def create_argparser(**kwargs):
    # defaults.update(model_and_diffusion_defaults())

    defaults = dict(
        num_workers=4,
        local_rank=0,
        gpus=1,
        image_size=128,
        image_size_encoder=224,
        iterations=150000,
        anneal_lr=False,
        lr=5e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        eval_batch_size=8,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        data_dir="",
        # load_depth=False, # TODO
        logdir="/mnt/lustre/yslan/logs/nips23/",
        eval_interval=2500,
    )

    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(loss_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    # os.environ[
    #     "TORCH_DISTRIBUTED_DEBUG"
    # ] = "DETAIL"  # set to DETAIL for runtime logging.
    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"

    args = create_argparser().parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus = th.cuda.device_count()

    opts = args

    args.rendering_kwargs = rendering_options_defaults(opts)

    # Launch processes.
    print('Launching processes...')

    try:
        training_loop(args)
    # except KeyboardInterrupt as e:
    except Exception as e:
        # print(e)
        traceback.print_exc()
        dist_util.cleanup() # clean port and socket when ctrl+c
