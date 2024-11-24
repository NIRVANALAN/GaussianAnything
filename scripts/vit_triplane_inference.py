"""
Train a diffusion model on images.
"""
import sys
import os

sys.path.append('.')
import torch.distributed as dist

import torch as th
import torch.multiprocessing as mp

import argparse
import dnnlib
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
from nsr.train_util import TrainLoop3DRec as TrainLoop
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults
from datasets.shapenet import load_data, load_eval_data, load_memory_data
from nsr.losses.builder import E3DGELossClass

from pdb import set_trace as st

th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True
th.backends.cudnn.enabled = True

SEED = 0


def training_loop(args):
    # def training_loop(args):
    dist_util.setup_dist(args)

    # dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=th.cuda.device_count())
    print(f"{args.local_rank=} init complete")
    th.cuda.set_device(args.local_rank)

    th.cuda.manual_seed_all(SEED)

    # logger.configure(dir=args.logdir, format_strs=["tensorboard", "csv"])
    logger.configure(dir=args.logdir)

    logger.log("creating data loader...")
    # TODO, load shapenet data
    # data = load_data(
    # if args.overfitting:
    #     data = load_memory_data(
    #         file_path=args.data_dir,
    #         batch_size=args.batch_size,
    #         reso=args.image_size,
    #         reso_encoder=args.image_size_encoder,  # 224 -> 128
    #         num_workers=args.num_workers,
    #         load_depth=args.depth_lambda > 0
    #         # load_depth=True # for evaluation
    #     )
    # else:
    #     data = load_data(
    #         file_path=args.data_dir,
    #         batch_size=args.batch_size,
    #         reso=args.image_size,
    #         reso_encoder=args.image_size_encoder,  # 224 -> 128
    #         num_workers=args.num_workers,
    #         load_depth=args.depth_lambda > 0
    #         # load_depth=True # for evaluation
    #     )
    eval_data = load_eval_data(
        file_path=args.data_dir,
        batch_size=args.eval_batch_size,
        reso=args.image_size,
        reso_encoder=args.image_size_encoder,  # 224 -> 128
        num_workers=args.num_workers,
        load_depth=True  # for evaluation
    )
    # try dry run
    # batch = next(data)
    # batch = None

    # logger.log("creating model and diffusion...")
    logger.log("creating encoder and NSR decoder...")
    # device = dist_util.dev()
    device = th.device("cuda", args.local_rank)

    auto_encoder = create_3DAE_model(
        **args_to_dict(args,
                       encoder_and_nsr_defaults().keys()))
    auto_encoder.to(device)
    auto_encoder.eval()

    # dist_util.sync_params(auto_encoder.named_parameters())

    # auto_encoder.train()

    # let all processes sync up before starting with a new epoch of training
    dist_util.synchronize()

    # noise = th.randn(1, 14 * 14, 384).to(device) # B, L, C
    # noise = th.randn(1, 3,224,224).to(device)
    # img = auto_encoder(noise, th.zeros(1, 25).to(device))
    # print(img['image'].shape)

    # if dist_util.get_rank()==0:
    #     print(auto_encoder)

    # schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    opt = dnnlib.EasyDict(args_to_dict(args, loss_defaults().keys()))
    loss_class = E3DGELossClass(device, opt).to(device)

    # logger.log("training...")
    TrainLoop(
        rec_model=auto_encoder,
        loss_class=loss_class,
        # diffusion=diffusion,
        data=None,
        eval_interval=-1,
        eval_data=eval_data,
        # data=batch,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_cldm_checkpoint=args.resume_cldm_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).eval_loop()  # ! overfitting


def create_argparser(**kwargs):
    # defaults.update(model_and_diffusion_defaults())

    defaults = dict(
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
    )

    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(loss_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"

    master_addr = '127.0.0.1'
    master_port = dist_util._find_free_port()

    args = create_argparser().parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus = th.cuda.device_count()
    args.master_addr = master_addr
    args.master_port = master_port

    # Launch processes.
    print('Launching processes...')
    training_loop(args)
