#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import sys
import os

sys.path.append('..')
sys.path.append('.')

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from vit.vision_transformer import VisionTransformer as ViT
from vit.vit_triplane import ViTTriplane
from guided_diffusion import dist_util, logger

import click
import dnnlib

SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 1


class YourDataset(Dataset):
    def __init__(self):
        pass


@click.command()
@click.option('--cfg', help='Base configuration', type=str, default='ffhq')
@click.option('--sr-module',
              help='Superresolution module override',
              metavar='STR',
              required=False,
              default=None)
@click.option('--density_reg',
              help='Density regularization strength.',
              metavar='FLOAT',
              type=click.FloatRange(min=0),
              default=0.25,
              required=False,
              show_default=True)
@click.option('--density_reg_every',
              help='lazy density reg',
              metavar='int',
              type=click.FloatRange(min=1),
              default=4,
              required=False,
              show_default=True)
@click.option('--density_reg_p_dist',
              help='density regularization strength.',
              metavar='FLOAT',
              type=click.FloatRange(min=0),
              default=0.004,
              required=False,
              show_default=True)
@click.option('--reg_type',
              help='Type of regularization',
              metavar='STR',
              type=click.Choice([
                  'l1', 'l1-alt', 'monotonic-detach', 'monotonic-fixed',
                  'total-variation'
              ]),
              required=False,
              default='l1')
@click.option('--decoder_lr_mul',
              help='decoder learning rate multiplier.',
              metavar='FLOAT',
              type=click.FloatRange(min=0),
              default=1,
              required=False,
              show_default=True)
@click.option('--c_scale',
              help='Scale factor for generator pose conditioning.',
              metavar='FLOAT',
              type=click.FloatRange(min=0),
              required=False,
              default=1)
def main(**kwargs):
    # parser = ArgumentParser('DDP usage example')
    # parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work
    # args = parser.parse_args()

    opts = dnnlib.EasyDict(kwargs)  # Command line arguments.
    c = dnnlib.EasyDict()  # Main config dict.

    rendering_options = {
        # 'image_resolution': c.training_set_kwargs.resolution,
        'image_resolution': 256,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        # 'superresolution_module': sr_module,
        # 'c_gen_conditioning_zero': not opts.
        # gen_pose_cond,  # if true, fill generator pose conditioning label with dummy zero vector
        # 'gpc_reg_prob': opts.gpc_reg_prob if opts.gen_pose_cond else None,
        'c_scale':
        opts.c_scale,  # mutliplier for generator pose conditioning label
        # 'superresolution_noise_mode': opts.
        # sr_noise_mode,  # [random or none], whether to inject pixel noise into super-resolution layers
        'density_reg': opts.density_reg,  # strength of density regularization
        'density_reg_p_dist': opts.
        density_reg_p_dist,  # distance at which to sample perturbed points for density regularization
        'reg_type': opts.
        reg_type,  # for experimenting with variations on density regularization
        'decoder_lr_mul':
        opts.decoder_lr_mul,  # learning rate multiplier for decoder
        'sr_antialias': True,
        'return_triplane_features': True,  # for DDF supervision
        'return_sampling_details_flag': True,
    }

    if opts.cfg == 'ffhq':
        rendering_options.update({
            'focal': 2985.29 / 700,
            'depth_resolution':
            # 48,  # number of uniform samples to take per ray.
            36,  # number of uniform samples to take per ray.
            'depth_resolution_importance':
            # 48,  # number of importance samples to take per ray.
            36,  # number of importance samples to take per ray.
            'ray_start':
            2.25,  # near point along each ray to start taking samples.
            'ray_end':
            3.3,  # far point along each ray to stop taking samples. 
            'box_warp':
            1,  # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'avg_camera_radius':
            2.7,  # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [
                0, 0, 0.2
            ],  # used only in the visualizer to control center of camera rotation.
        })
    elif opts.cfg == 'afhq':
        rendering_options.update({
            'focal': 4.2647,
            'depth_resolution': 48,
            'depth_resolution_importance': 48,
            'ray_start': 2.25,
            'ray_end': 3.3,
            'box_warp': 1,
            'avg_camera_radius': 2.7,
            'avg_camera_pivot': [0, 0, -0.06],
        })
    elif opts.cfg == 'shapenet':
        rendering_options.update({
            'depth_resolution': 64,
            'depth_resolution_importance': 64,
            # 'ray_start': 0.1,
            # 'ray_end': 2.6,
            'ray_start': 0.1,
            'ray_end': 3.3,
            'box_warp': 1.6,
            'white_back': True,
            'avg_camera_radius': 1.7,
            'avg_camera_pivot': [0, 0, 0],
        })
    else:
        assert False, "Need to specify config"

    c.rendering_kwargs = rendering_options

    args = opts

    # keep track of whether the current process is the `master` process (totally optional, but I find it useful for data laoding, logging, etc.)
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.is_master = args.local_rank == 0

    # set the device
    # device = torch.cuda.device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")

    # initialize PyTorch distributed using environment variables (you could also do this more explicitly by specifying `rank` and `world_size`, but I find using environment variables makes it so that you can easily use the same script on different machines)
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            rank=args.local_rank,
                            world_size=torch.cuda.device_count())
    print(f"{args.local_rank=} init complete")
    torch.cuda.set_device(args.local_rank)

    # set the seed for all GPUs (also make sure to set the seed for random, numpy, etc.)
    torch.cuda.manual_seed_all(SEED)

    # initialize your model (BERT in this example)
    # model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # model = ViT(
    #     image_size = 256,
    #     patch_size = 32,
    #     num_classes = 1000,
    #     dim = 1024,
    #     depth = 6,
    #     heads = 16,
    #     mlp_dim = 2048,
    #     dropout = 0.1,
    #     emb_dropout = 0.1
    # )

    # TODO, check pre-trained ViT encoder cfgs
    model = ViTTriplane(
        img_size=[224],
        patch_size=16,
        in_chans=384,
        num_classes=0,
        embed_dim=384,  # Check ViT encoder dim
        depth=2,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        out_chans=96,
        c_dim=25,  # Conditioning label (C) dimensionality.
        img_resolution=128,  # Output resolution.
        img_channels=3,  # Number of output color channels.
        cls_token=False,
        # TODO, replace with c
        rendering_kwargs=c.rendering_kwargs,
    )
    # noise = torch.randn(1, 8, 8, 1024)

    # send your model to GPU
    model = model.to(device)

    # initialize distributed data parallel (DDP)
    model = DDP(model,
                device_ids=[args.local_rank],
                output_device=args.local_rank)

    dist_util.sync_params(model.named_parameters())

    # # initialize your dataset
    # dataset = YourDataset()

    # # initialize the DistributedSampler
    # sampler = DistributedSampler(dataset)

    # # initialize the dataloader
    # dataloader = DataLoader(
    #     dataset=dataset,
    #     sampler=sampler,
    #     batch_size=BATCH_SIZE
    # )

    # start your training!
    for epoch in range(NUM_EPOCHS):
        # put model in train mode
        model.train()

        # let all processes sync up before starting with a new epoch of training
        dist.barrier()

        noise = torch.randn(1, 14 * 14, 384).to(device)  # B, L, C
        img = model(noise, torch.zeros(1, 25).to(device))
        print(img['image'].shape)
    # st()

    # img = torch.randn(1, 3, 256, 256).to(device)

    # preds = model(img)
    # print(preds.shape)
    # assert preds.shape == (1, 1000), 'correct logits outputted'

    # for step, batch in enumerate(dataloader):
    #     # send batch to device
    #     batch = tuple(t.to(args.device) for t in batch)

    #     # forward pass
    #     outputs = model(*batch)

    #     # compute loss
    #     loss = outputs[0]

    #     # etc.


if __name__ == '__main__':
    main()
