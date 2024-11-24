"""
Train a diffusion model on images.
"""
# import imageio
from pathlib import Path
import torchvision
import kornia
import lz4.frame
import gzip
import random
import json
import sys
import os
import lmdb
from tqdm import tqdm

sys.path.append('.')
import torch.distributed as dist
import pytorch3d.ops
import pickle
import traceback
from PIL import Image
import torch as th
if th.cuda.is_available():
    from xformers.triton import FusedLayerNorm as LayerNorm
import torch.multiprocessing as mp
import lzma
import webdataset as wds
import numpy as np

import point_cloud_utils as pcu
from torch.utils.data import DataLoader, Dataset
import imageio.v3 as iio

import argparse
import dnnlib
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
# from nsr.train_util import TrainLoop3DRec as TrainLoop
from nsr.train_nv_util import TrainLoop3DRecNV, TrainLoop3DRec, TrainLoop3DRecNVPatch
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults, rendering_options_defaults, eg3d_options_default
from datasets.shapenet import load_data, load_data_for_lmdb, load_eval_data, load_memory_data
from nsr.losses.builder import E3DGELossClass
from datasets.eg3d_dataset import init_dataset_kwargs
from nsr.volumetric_rendering.ray_sampler import RaySampler

# from .lmdb_create import encode_and_compress_image


def encode_and_compress_image(inp_array, is_image=False, compress=True):
    # Read the image using imageio
    # image = imageio.v3.imread(image_path)

    # Convert the image to bytes
    # with io.BytesIO() as byte_buffer:
    #     imageio.imsave(byte_buffer, image, format="png")
    #     image_bytes = byte_buffer.getvalue()
    if is_image:
        inp_bytes = iio.imwrite("<bytes>", inp_array, extension=".png")
    else:
        inp_bytes = inp_array.tobytes()

    # Compress the image data using gzip
    if compress:
        # compressed_data = gzip.compress(inp_bytes)
        compressed_data = lz4.frame.compress(inp_bytes)
        return compressed_data
    else:
        return inp_bytes


from pdb import set_trace as st
import bz2

# th.backends.cuda.matmul.allow_tf32 = True # https://huggingface.co/docs/diffusers/optimization/fp16


def training_loop(args):
    # def training_loop(args):
    # dist_util.setup_dist(args)
    # th.autograd.set_detect_anomaly(True) # type: ignore
    th.autograd.set_detect_anomaly(False)  # type: ignore
    # https://blog.csdn.net/qq_41682740/article/details/126304613

    SEED = args.seed

    # dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=th.cuda.device_count())
    # logger.log(f"{args.local_rank=} init complete, seed={SEED}")
    # th.cuda.set_device(args.local_rank)
    th.cuda.empty_cache()

    # * deterministic algorithms flags
    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    ray_sampler = RaySampler()

    # logger.configure(dir=args.logdir, format_strs=["tensorboard", "csv"])
    logger.configure(dir=args.logdir)

    logger.log("creating encoder and NSR decoder...")
    # device = dist_util.dev()
    # device = th.device("cuda", args.local_rank)

    # shared eg3d opts
    opts = eg3d_options_default()

    logger.log("creating data loader...")

    # let all processes sync up before starting with a new epoch of training
    dist_util.synchronize()

    # schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    opt = dnnlib.EasyDict(args_to_dict(args, loss_defaults().keys()))
    # opt.max_depth, opt.min_depth = args.rendering_kwargs.ray_end, args.rendering_kwargs.ray_start
    # loss_class = E3DGELossClass(device, opt).to(device)

    # writer = SummaryWriter() # TODO, add log dir

    logger.log("training...")

    def save_pcd_from_gs(lmdb_path, start_shard, wds_split):
        """
        Convert a PyTorch dataset to LMDB format.

        Parameters:
        - dataset: PyTorch dataset
        - lmdb_path: Path to store the LMDB database
        """

        # ! read dataset path

        # latent_dir = '/nas/shared/V2V/yslan/logs/nips23/Reconstruction/final/objav/vae/gs/infer-latents/768/8x8/animals/latent_dir/Animals'
        latent_dir = '/nas/shared/V2V/yslan/logs/nips23/Reconstruction/final/objav/vae/gs/infer-latents/768/8x8/animals-gs-latent-dim=10-fullset/latent_dir'

        ins_list = []

        for class_dir in os.listdir(latent_dir)[:]:
            for dict_dir in os.listdir(os.path.join(latent_dir, class_dir))[:]:
                for ins_dir in os.listdir(os.path.join(latent_dir, class_dir, dict_dir)):
                    ins_list.append(os.path.join(class_dir, dict_dir, ins_dir))

        K = 4096 # fps K

        for idx, ins in enumerate(tqdm(ins_list)):

            # sample_ins = sample.pop('ins')
            pcd_path = Path(f'{logger.get_dir()}/fps-pcd/{ins}')

            if (pcd_path / f'fps-{K}.ply').exists():
                continue

            # ! load gaussians
            gaussians = np.load(os.path.join(latent_dir,ins,'gaussians.npy'))

            points = gaussians[0,:, 0:3]

            # load opacity and scale
            opacity = gaussians[0,:, 3:4]
            # scale = gaussians[0,:, 4:6]
            # colors = gaussians[0, :, 10:13]

            opacity_mask = opacity < 0.005 # official threshold

            high_opacity_points = points[~opacity_mask[..., 0]]
            # high_opacity_colors = colors[~opacity_mask[..., 0]]
            high_opacity_points = th.from_numpy(high_opacity_points).to(dist_util.dev())


            pcd_path.mkdir(parents=True, exist_ok=True)

            try:
                
                fps_points = pytorch3d.ops.sample_farthest_points(
                    high_opacity_points.unsqueeze(0), K=K)[0]

                pcu.save_mesh_v(
                    str(pcd_path / f'fps-{K}.ply'),
                    fps_points[0].detach().cpu().numpy(),
                )

                assert (pcd_path / f'fps-{K}.ply').exists()
            
            except Exception as e:
                continue
                print(pcd_path, 'save failed: ', e)


    save_pcd_from_gs(os.path.join(logger.get_dir(), f'wds-%06d.tar'),
                        args.start_shard, args.wds_split)


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
        objv_dataset=False,
        pose_warm_up_iter=-1,
        start_shard=0,
        shuffle_across_cls=False,
        wds_split=1,  # out of 4
    )

    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(loss_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    # os.environ[
    # "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    # os.environ["NCCL_DEBUG"]="INFO"

    args = create_argparser().parse_args()
    # args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus = th.cuda.device_count()

    opts = args

    args.rendering_kwargs = rendering_options_defaults(opts)

    # print(args)
    with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Launch processes.
    print('Launching processes...')

    # try:
    training_loop(args)
    # except KeyboardInterrupt as e:
    # except Exception as e:
    #     # print(e)
    #     traceback.print_exc()
    #     dist_util.cleanup() # clean port and socket when ctrl+c
