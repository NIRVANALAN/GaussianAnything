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
from xformers.components.feedforward import fused_mlp
# from xformers.triton import FusedLayerNorm as LayerNorm
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

from transport import create_transport, Sampler
from transport.train_utils import parse_transport_args
from nsr.camera_utils import generate_input_camera, uni_mesh_path, sample_uniform_cameras_on_sphere

# from torch.utils.tensorboard import SummaryWriter

# SEED = 0



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

    th.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

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

    logger.log("creating data loader...")

    if args.objv_dataset:
        from datasets.g_buffer_objaverse import load_data, load_eval_data, load_memory_data, load_wds_data, load_data_cls
    else:  # shapenet
        from datasets.shapenet import load_data, load_eval_data, load_memory_data

    if args.i23d:
        data = load_eval_data(
            file_path=args.eval_data_dir,
            batch_size=args.eval_batch_size,
            reso=args.image_size,
            reso_encoder=args.image_size_encoder,  # 224 -> 128
            num_workers=args.num_workers,
            load_depth=True,  # for evaluation
            preprocess=auto_encoder.preprocess,
            **args_to_dict(args,
                            dataset_defaults().keys()))
    else:
        data = None # t23d sampling, only caption required

    # eval_dataset = load_data_cls(
    #     file_path=args.data_dir,
    #     batch_size=args.batch_size,
    #     reso=args.image_size,
    #     reso_encoder=args.image_size_encoder,  # 224 -> 128
    #     num_workers=args.num_workers,
    #     load_latent=True,
    #     return_dataset=True,
    #     **args_to_dict(args,
    #                     dataset_defaults().keys())
    # )

    eval_dataset = None 


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
        'flow_matching':
        nsr.lsgm.flow_matching_trainer.FlowMatchingEngine,
        'flow_matching_gs':  
        nsr.lsgm.flow_matching_trainer.FlowMatchingEngine_gs, # slightly modified sampling and rendering for gs
    }[args.trainer_name]


    # if 'vpsde' in args.trainer_name:
    #     sde_diffusion = make_sde_diffusion(
    #         dnnlib.EasyDict(
    #             args_to_dict(args,
    #                          continuous_diffusion_defaults().keys())))
    #     # assert args.mixed_prediction, 'enable mixed_prediction by default'
    #     logger.log('create VPSDE diffusion.')
    # else:
    sde_diffusion = None

    # if 'cldm' in args.trainer_name:
    #     assert isinstance(denoise_model, tuple)
    #     denoise_model, controlNet = denoise_model

    #     controlNet.to(dist_util.dev())
    #     controlNet.train()
    # else:
    controlNet = None

    # st()
    denoise_model.to(dist_util.dev())
    denoise_model.train()

    auto_encoder.decoder.rendering_kwargs = args.rendering_kwargs
    
    # camera = th.load('eval_pose.pt', map_location=dist_util.dev())[:]

    # if fid 

    # '''
    azimuths = []
    elevations = []
    frame_number = 10

    for i in range(frame_number): # 1030 * 5 * 10, for FID 50K

        azi, elevation = sample_uniform_cameras_on_sphere()
        # azi, elevation = azi[0] / np.pi * 180, elevation[0] / np.pi * 180
        azi, elevation = azi[0] / np.pi * 180, (elevation[0]-np.pi*0.5) / np.pi * 180 # [-0.5 pi, 0.5 pi]
        azimuths.append(azi)
        elevations.append(elevation)

    azimuths = np.array(azimuths)
    elevations = np.array(elevations)

    # azimuths = np.array(list(range(0,360,30))).astype(float)
    # frame_number = azimuths.shape[0]
    # elevations = np.array([10]*azimuths.shape[0]).astype(float)

    zero123pp_pose, _ = generate_input_camera(1.8, [[elevations[i], azimuths[i]] for i in range(frame_number)], fov=30)
    K = th.Tensor([1.3889, 0.0000, 0.5000, 0.0000, 1.3889, 0.5000, 0.0000, 0.0000, 0.0039]).to(zero123pp_pose) # keeps the same
    camera = th.cat([zero123pp_pose.reshape(frame_number,-1), K.unsqueeze(0).repeat(frame_number,1)], dim=-1)
    # '''

    # camera = uni_mesh_path(12, radius=2.0) # ! for exporting mesh

    training_loop_class=TrainLoop(rec_model=auto_encoder,
              denoise_model=denoise_model,
              control_model=controlNet,
              diffusion=diffusion,
              sde_diffusion=sde_diffusion,
              loss_class=loss_class,
              data=data,
            #   eval_data=None,
              eval_data=eval_dataset, # return dataset
              **vars(args))

    if args.i23d:
        # ! image-conditioned 3D generation
        training_loop_class.eval_i23d_and_export(
            prompt='',
            save_img=args.save_img,
            # use_train_trajectory=args.use_train_trajectory,
            camera=camera,
            # num_instances=args.num_instances,
            num_samples=args.num_samples,
            unconditional_guidance_scale=args.unconditional_guidance_scale,
            seed=args.seed,
            stage_1_output_dir=args.stage_1_output_dir,
            export_mesh=args.export_mesh,
        )


    else:
        # the script used in 3dtopia
        with open('datasets/caption-forpaper.txt', 'r') as f:
            all_prompts_available = [caption.strip() for caption in f.readlines()]

        for prompt in all_prompts_available:

            training_loop_class.eval_and_export(
                prompt=prompt,
                save_img=args.save_img,
                # use_train_trajectory=args.use_train_trajectory,
                camera=camera,
                unconditional_guidance_scale=args.unconditional_guidance_scale,
                # num_instances=args.num_instances,
                num_samples=args.num_samples,
                seed=args.seed,
                stage_1_output_dir=args.stage_1_output_dir,
                export_mesh=args.export_mesh,
            )


    dist_util.synchronize()
    logger.log('sampling complete')


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
        save_img=False,
        use_train_trajectory=
        False,  # use train trajectory to sample images for fid calculation
        unconditional_guidance_scale=1.0,
        seed=42,
        num_samples=10,
        num_instances=10, # for i23d, loop different condition
    )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(continuous_diffusion_defaults())
    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(dataset_defaults())  # type: ignore
    defaults.update(loss_defaults())
    defaults.update(control_net_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    # ! add transport args
    parse_transport_args(parser)

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
