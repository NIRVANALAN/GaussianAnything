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

    if args.sr_training:
        args.sr_kwargs = dnnlib.EasyDict(
            channel_base=opts.cbase,
            channel_max=opts.cmax,
            fused_modconv_default='inference_only',
            use_noise=True
        )  # ! close noise injection? since noise_mode='none' in eg3d

    if args.objv_dataset:
        from datasets.g_buffer_objaverse import load_data, load_eval_data, load_memory_data, load_data_for_lmdb
    else:  # shapenet
        from datasets.shapenet import load_data, load_eval_data, load_memory_data, load_data_for_lmdb

    # auto_encoder = create_3DAE_model(
    #     **args_to_dict(args,
    #                    encoder_and_nsr_defaults().keys()))
    # auto_encoder.to(device)
    # auto_encoder.train()

    logger.log("creating data loader...")
    # data = load_data(
    # st()
    # if args.overfitting:
    #     data = load_memory_data(
    #         file_path=args.data_dir,
    #         batch_size=args.batch_size,
    #         reso=args.image_size,
    #         reso_encoder=args.image_size_encoder,  # 224 -> 128
    #         num_workers=args.num_workers,
    #         # load_depth=args.depth_lambda > 0
    #         load_depth=True  # for evaluation
    #     )
    # else:
    if args.cfg in ('afhq', 'ffhq'):
        # ! load data
        logger.log("creating eg3d data loader...")
        training_set_kwargs, dataset_name = init_dataset_kwargs(
            data=args.data_dir,
            class_name='datasets.eg3d_dataset.ImageFolderDatasetLMDB',
            reso_gt=args.image_size)  # only load pose here
        # if args.cond and not training_set_kwargs.use_labels:
        # raise Exception('check here')

        # training_set_kwargs.use_labels = args.cond
        training_set_kwargs.use_labels = True
        training_set_kwargs.xflip = False
        training_set_kwargs.random_seed = SEED
        # training_set_kwargs.max_size = args.dataset_size
        # desc = f'{args.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'

        # * construct ffhq/afhq dataset
        training_set = dnnlib.util.construct_class_by_name(
            **training_set_kwargs)  # subclass of training.dataset.Dataset
        dataset_size = len(training_set)

        # training_set_sampler = InfiniteSampler(
        #     dataset=training_set,
        #     rank=dist_util.get_rank(),
        #     num_replicas=dist_util.get_world_size(),
        #     seed=SEED)

        data = DataLoader(
            training_set,
            shuffle=False,
            batch_size=1,
            num_workers=16,
            drop_last=False,
            # prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True,
        )

    else:
        # data, dataset_name, dataset_size, dataset = load_data_for_lmdb(
        data, dataset_name, dataset_size = load_data_for_lmdb(
            file_path=args.data_dir,
            batch_size=args.batch_size,
            reso=args.image_size,
            reso_encoder=args.image_size_encoder,  # 224 -> 128
            num_workers=args.num_workers,
            load_depth=True,
            preprocess=None,
            dataset_size=args.dataset_size,
            trainer_name=args.trainer_name,
            shuffle_across_cls=args.shuffle_across_cls,
            wds_split=args.wds_split,
            four_view_for_latent=True
            # wds_output_path=os.path.join(logger.get_dir(), f'wds-%06d.tar')
            # load_depth=True # for evaluation
        )
    #     if args.pose_warm_up_iter > 0:
    #         overfitting_dataset = load_memory_data(
    #             file_path=args.data_dir,
    #             batch_size=args.batch_size,
    #             reso=args.image_size,
    #             reso_encoder=args.image_size_encoder,  # 224 -> 128
    #             num_workers=args.num_workers,
    #             # load_depth=args.depth_lambda > 0
    #             load_depth=True  # for evaluation
    #         )
    #         data = [data, overfitting_dataset, args.pose_warm_up_iter]
    # eval_data = load_eval_data(
    #     file_path=args.eval_data_dir,
    #     batch_size=args.eval_batch_size,
    #     reso=args.image_size,
    #     reso_encoder=args.image_size_encoder,  # 224 -> 128
    #     num_workers=args.num_workers,
    #     load_depth=True,  # for evaluation
    #     preprocess=auto_encoder.preprocess)
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
    # loss_class = E3DGELossClass(device, opt).to(device)

    # writer = SummaryWriter() # TODO, add log dir

    logger.log("training...")

    # TrainLoop = {
    #     'input_rec': TrainLoop3DRec,
    #     'nv_rec': TrainLoop3DRecNV,
    #     'nv_rec_patch': TrainLoop3DRecNVPatch,
    # }[args.trainer_name]

    # TrainLoop(rec_model=auto_encoder,
    #           loss_class=loss_class,
    #           data=data,
    #           eval_data=eval_data,
    #           **vars(args)).run_loop()  # ! overfitting

    # Function to compress an image using gzip
    # def compress_image_gzip(image_path):
    # def encode_and_compress_image(inp_array, is_image=False):
    #     # Read the image using imageio
    #     # image = imageio.v3.imread(image_path)

    #     # Convert the image to bytes
    #     # with io.BytesIO() as byte_buffer:
    #     #     imageio.imsave(byte_buffer, image, format="png")
    #     #     image_bytes = byte_buffer.getvalue()
    #     if is_image:
    #         inp_bytes = iio.imwrite("<bytes>", inp_array, extension=".png")
    #     else:
    #         inp_bytes = inp_array.tobytes()

    #     # Compress the image data using gzip
    #     compressed_data = gzip.compress(inp_bytes)

    #     return compressed_data

    def save_pcd_from_depth(dataset_loader, dataset_size, lmdb_path,
                            start_shard, wds_split):
        """
        Convert a PyTorch dataset to LMDB format.

        Parameters:
        - dataset: PyTorch dataset
        - lmdb_path: Path to store the LMDB database
        """
        # env = lmdb.open(lmdb_path, map_size=1024 ** 4, readahead=False)  # Adjust map_size based on your dataset size
        # sink = wds.ShardWriter(lmdb_path, start_shard=start_shard)

        # with env.begin(write=True) as txn:

        # with env.begin(write=True) as txn:
        # txn.put("length".encode("utf-8"), str(dataset_size).encode("utf-8"))

        # K = 10000 # fps K
        K = 4096 # fps K
        # K = 128*128*2 # fps K, 32768
        # K = 1024*24 # 20480
        # K = 4096 # fps K

        # if True:

        # try:
        for idx, sample in enumerate(tqdm(dataset_loader)):

            # pass
            # remove the batch index of returned dict sample

            sample_ins = sample.pop('ins')
            # !!! add all()
            assert all([ sample_ins[i] == sample_ins[0] for i in range(0, len(sample_ins)) ]), sample_ins  # check the batch is the same instnace

            img_size = sample['raw_img'].shape[2]

            pcd_path = Path(f'{logger.get_dir()}/fps-pcd/{sample_ins[0]}')

            if (pcd_path / f'fps-{K}.ply').exists():
                continue

            pcd_path.mkdir(parents=True, exist_ok=True)

            # sample = {
            #     # k:v.squeeze(0).cpu().numpy() if isinstance(v, th.Tensor) else v[0] for k, v in sample.items()
            #     k:v.cpu().numpy() if isinstance(v, th.Tensor) else v for k, v in sample.items()
            #     # k:v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in sample.items()
            # }

            B = sample['c'].shape[0]

            cam2world_matrix = sample['c'][:, :16].reshape(B, 4, 4)
            intrinsics = sample['c'][:, 16:25].reshape(B, 3, 3)

            ray_origins, ray_directions = ray_sampler(  # shape: 
                cam2world_matrix, intrinsics, img_size)[:2]

            micro = sample

            # self.gs.output_size,)[:2]
            # depth = rearrange(micro['depth'], '(B V) H W -> ')
            # depth_128 = th.nn.functional.interpolate(
            #     micro['depth'].unsqueeze(1), (128, 128),
            #     mode='nearest'
            # )[:, 0]  # since each view has 128x128 Gaussians
            # depth = depth_128.reshape(B * V, -1).unsqueeze(-1)

            # fg_mask = (micro['depth'] > 0).unsqueeze(1).float()

            # fg_mask = micro['alpha_mask'].unsqueeze(1).float() # anti-alias? B 1 H W
            fg_mask = (micro['alpha_mask'] == 1).unsqueeze(1).float() # anti-alias? B 1 H W

            kernel = th.tensor([[0, 1, 0], [1, 1, 1], [0, 1,
                                                       0]]).to(fg_mask.device)

            # ! erode. but still some noise...
            '''
            erode_mask = kornia.morphology.erosion(fg_mask, kernel)  # B 1 H W
            # torchvision.utils.save_image(fg_mask.float()*2-1,'mask.jpg',  value_range=(-1,1), normalize=True)
            # torchvision.utils.save_image(erode_mask.float()*2-1,'erode_mask.jpg',  value_range=(-1,1), normalize=True)

            fg_mask = (erode_mask==1).float().reshape(B, -1).unsqueeze(-1) > 0 #
            # '''
            # fg_mask = fg_mask.reshape(B, -1).unsqueeze(-1) == 1  # ! for some failed data
            # ! no erode:
            fg_mask = fg_mask.reshape(B, -1).unsqueeze(-1) > 0  # ! for some failed data

            depth = micro['depth'].reshape(B, -1).unsqueeze(-1)
            depth = th.where(depth < 1.05, 0, depth)  # filter outlier
            depth[depth == 0] = 1e10  # so that rays_o will not appear in the final pcd.

            # fg_mask = depth>0

            # fg_mask = th.nn.functional.interpolate(
            #     micro['depth_mask'].unsqueeze(1).to(th.uint8),
            #     (128, 128),
            #     mode='nearest').squeeze(1)  # B*V H W
            # fg_mask = fg_mask.reshape(B * V, -1).unsqueeze(-1)


            # gt_pos = gt_pos[gt_pos.nonzero(as_tuple=True)].reshape(-1, 3) # return non-zero points for fps sampling

            # pcu.save_mesh_v(f'tmp/gt-512.ply', gt_pos.detach().cpu().numpy(),)

            # fps sampling
            try:
                
                gt_pos = ray_origins + depth * ray_directions  # BV HW 3, already in the world space
                gt_pos = fg_mask * gt_pos  # remove ray_origins when depth=0
                # gt_pos = gt_pos[[8,16,24,25,26, 27, 31, 35]]
                # gt_pos = gt_pos[[5,10,15,20,24,25,26]]
                # gt_pos = gt_pos[[4, 12, 20, 25]]
                gt_pos = gt_pos[:]
                # gt_pos = gt_pos[[25,26]]
                gt_pos = gt_pos.reshape(-1, 3).to(dist_util.dev())
                gt_pos = gt_pos.clip(-0.45, 0.45)
                gt_pos = th.where(gt_pos.abs()==0.45, 0, gt_pos) # no boundary here? Yes.

                # ! filter the zero points together here 

                nonzero_mask = (gt_pos != 0).all(dim=-1)  # Shape: (N, 3)
                nonzero_gt_pos = gt_pos[nonzero_mask] 

                fps_points = pytorch3d.ops.sample_farthest_points(
                    nonzero_gt_pos.unsqueeze(0), K=K)[0]

                pcu.save_mesh_v(
                    str(pcd_path / f'fps-{K}.ply'),
                    fps_points[0].detach().cpu().numpy(),
                )

                assert (pcd_path / f'fps-{K}.ply').exists()
            
            except Exception as e:

                st()
                pass

                print(pcd_path, 'save failed: ', e)
            
            # ! debug projection matrix

            # def pcd_to_homo(pcd):
            #     return th.cat([pcd, th.ones_like(pcd[..., 0:1])], -1)
            
            # st()

            # proj_point = th.inverse(cam2world_matrix[0]).to(fps_points) @ pcd_to_homo(fps_points[0]).permute(1, 0)
            # # proj_point = th.inverse(cam2world_matrix[0]).to(fps_points) @ pcd_to_homo((ray_origins + depth * ray_directions)[0].to(fps_points)).permute(1, 0)
            # proj_point[:2, ...] /= proj_point[2, ...]
            # proj_point[2, ...] = 1 # homo


            # proj_point = intrinsics[0].to(fps_points) @ proj_point[:3]
            # proj_point = proj_point.permute(1,0)[..., :2] # 768 4
            # st()

            # torchvision.utils.save_image(micro['raw_img'][::5].permute(0,3,1,2).float()/127.5-1,'raw.jpg',  value_range=(-1,1), normalize=True)

            # # encode batch images/depths/strings?  no need to encode ins/fname here; just save the caption

            # # sample = dataset_loader[idx]
            # compressed_sample = {}
            # sample['ins'] = sample_ins[0]
            # sample['caption'] = sample.pop('caption')[0]

            # for k, v in sample.items():

            #     # key = f'{idx}-{k}'.encode('utf-8')

            #     if 'img' in k: # only bytes required? laod the 512 depth bytes only.
            #         v = encode_and_compress_image(v, is_image=True, compress=True)
            #         # v = encode_and_compress_image(v, is_image=True, compress=False)
            #     # elif 'depth' in k:
            #     elif isinstance(v, str):
            #         v = v.encode('utf-8') # caption / instance name
            #     else: # regular bytes encoding
            #         v = encode_and_compress_image(v.astype(np.float32), is_image=False, compress=True)
            #         # v = encode_and_compress_image(v.astype(np.float32), is_image=False, compress=False)

            #     compressed_sample[k] = v

            # # st() # TODO, add .gz for compression after pipeline done
            # sink.write({
            #     "__key__": f"sample_{wds_split:03d}_{idx:07d}",
            #     # **{f'{k}.pyd': v for k, v in compressed_sample.items()}, # store as pickle, already compressed
            #     'sample.pyd': compressed_sample
            #     # 'sample.gz': compressed_sample
            # })

            # break
            # if idx > 25:
            #     break
        # except:
            # continue

        # sink.close()

    # convert_to_lmdb(data, os.path.join(logger.get_dir(), dataset_name)) convert_to_lmdb_compressed(data, os.ath.join(logger.get_dir(), dataset_name))
    # convert_to_lmdb_compressed(data, os.path.join(logger.get_dir()), dataset_size)
    save_pcd_from_depth(data, dataset_size,
                        os.path.join(logger.get_dir(), f'wds-%06d.tar'),
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
