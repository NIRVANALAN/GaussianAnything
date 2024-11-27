import argparse
import json
import sys
sys.path.append('.')
import torch
import torchvision
from torchvision import transforms
import numpy as np

import os
import gc
import dnnlib
from omegaconf import OmegaConf
from PIL import Image 
from dnnlib.util import EasyDict

import gradio as gr

import rembg

from huggingface_hub import hf_hub_download


"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os


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

from transport.train_utils import parse_transport_args

from utils.infer_utils import remove_background, resize_foreground

SEED = 0

def resize_to_224(img):
    img = transforms.functional.resize(img, 518, # required by dino.
        interpolation=transforms.InterpolationMode.LANCZOS)
    return img


def set_white_background(image):
    image = np.array(image).astype(np.float32) / 255.0
    mask = image[:, :, 3:4]
    image = image[:, :, :3] * mask + (1 - mask)
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    return image


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")



def main(args_1, args_2):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # args.rendering_kwargs = rendering_options_defaults(args)

    dist_util.setup_dist(args_1)
    logger.configure(dir=args_1.logdir)

    th.cuda.empty_cache()

    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # * set denoise model args
    logger.log("creating model and diffusion...")
    args_1.img_size = [args_1.image_size_encoder]
    args_1.image_size = args_1.image_size_encoder  # 224, follow the triplane size

    args_2.img_size = [args_2.image_size_encoder]
    args_2.image_size = args_2.image_size_encoder  # 224, follow the triplane size

    denoise_model_stage1, diffusion = create_model_and_diffusion(
        **args_to_dict(args_1,
                       model_and_diffusion_defaults().keys()))

    denoise_model_stage2, diffusion = create_model_and_diffusion(
        **args_to_dict(args_2,
                       model_and_diffusion_defaults().keys()))

    opts = eg3d_options_default()

    denoise_model_stage1.to(dist_util.dev())
    denoise_model_stage1.eval()
    denoise_model_stage2.to(dist_util.dev())
    denoise_model_stage2.eval()

    # * auto-encoder reconstruction model
    logger.log("creating 3DAE...")
    auto_encoder = create_3DAE_model(
        **args_to_dict(args_1,
                       encoder_and_nsr_defaults().keys()))

    auto_encoder.to(dist_util.dev())
    auto_encoder.eval()

    # faster inference
    # denoise_model = denoise_model.to(th.bfloat16)
    # auto_encoder = auto_encoder.to(th.bfloat16)

    # TODO, how to set the scale?
    logger.log("create dataset")

    if args_1.objv_dataset:
        from datasets.g_buffer_objaverse import load_data, load_eval_data, load_memory_data, load_wds_data
    else:  # shapenet
        from datasets.shapenet import load_data, load_eval_data, load_memory_data
    
    # load data if i23d
    # if args.i23d:
    #     data = load_eval_data(
    #         file_path=args.eval_data_dir,
    #         batch_size=args.eval_batch_size,
    #         reso=args.image_size,
    #         reso_encoder=args.image_size_encoder,  # 224 -> 128
    #         num_workers=args.num_workers,
    #         load_depth=True,  # for evaluation
    #         preprocess=auto_encoder.preprocess,
    #         **args_to_dict(args,
    #                         dataset_defaults().keys()))
    # else:
    data = None # t23d sampling, only caption required


    TrainLoop = {
        'flow_matching':
        nsr.lsgm.flow_matching_trainer.FlowMatchingEngine,
        'flow_matching_gs':  
        nsr.lsgm.flow_matching_trainer.FlowMatchingEngine_gs, # slightly modified sampling and rendering for gs
    }[args_1.trainer_name]

    # continuous
    sde_diffusion = None

    auto_encoder.decoder.rendering_kwargs = args_1.rendering_kwargs
    # stage_1_output_dir = args_2.stage_1_output_dir

    training_loop_class_stage1 = TrainLoop(rec_model=auto_encoder,
                                    denoise_model=denoise_model_stage1,
                                    control_model=None, # to remove
                                    diffusion=diffusion,
                                    sde_diffusion=sde_diffusion,
                                    loss_class=None,
                                    data=data,
                                    eval_data=None,
                                    **args_1)

    training_loop_class_stage2 = TrainLoop(rec_model=auto_encoder,
                                    denoise_model=denoise_model_stage2,
                                    control_model=None, # to remove
                                    diffusion=diffusion,
                                    sde_diffusion=sde_diffusion,
                                    loss_class=None,
                                    data=data,
                                    eval_data=None,
                                    **args_2)


    css = """
    h1 {
        text-align: center;
        display:block;
    }
    """


    def preprocess(input_image, preprocess_background=True, foreground_ratio=0.85):
        if preprocess_background:
            rembg_session = rembg.new_session()
            image = input_image.convert("RGB")
            image = remove_background(image, rembg_session)
            image = resize_foreground(image, foreground_ratio)
            image = set_white_background(image)
        else:
            image = input_image
            if image.mode == "RGBA":
                image = set_white_background(image)
        image = resize_to_224(image)
        return image


    def cascaded_generation(processed_image, seed, cfg_scale):
        # gc.collect()
        # stage-1, generate pcd
        stage_1_pcd = training_loop_class_stage1.eval_i23d_and_export_gradio(processed_image, seed, cfg_scale)
        # stage-2, generate surfel Gaussians, tsdf mesh etc.
        video_path, rgb_xyz_path, post_mesh_path = training_loop_class_stage2.eval_i23d_and_export_gradio(processed_image, seed, cfg_scale)
        return video_path, rgb_xyz_path, post_mesh_path, stage_1_pcd

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # GaussianAnything: Interactive Point Cloud Latent Diffusion for 3D Generation
            **GaussianAnything (arXiv 2024)** [[code](https://github.com/NIRVANALAN/GaussianAnything), [project page](https://nirvanalan.github.io/projects/GA/)] is a native 3D diffusion model that supports high-quality 2D Gaussians generation. 
            It first trains a 3D VAE on **Objaverse**, which compress each 3D asset into a compact point cloud-structured latent. 
            After that, a image/text-conditioned diffusion model is trained following LDM paradigm.
            The model used in the demo adopts 3D DiT architecture and flow-matching framework, and supports single-image condition.
            It is trained on 8 A100 GPUs for 1M iterations with batch size 256.
            Locally, on an NVIDIA A100/A10 GPU, each image-conditioned diffusion generation can be done within 20 seconds (time varies due to the adaptive-step ODE solver used in flow-mathcing.)
            Upload an image of an object or click on one of the provided examples to see how the GaussianAnything works.

            The 3D viewer will render a .glb point cloud exported from the centers of the surfel Gaussians, and an integrated TSDF mesh.
            For best results run the demo locally and render locally - to do so, clone the [main repository](https://github.com/NIRVANALAN/GaussianAnything).
            """
            )
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(
                        label="Input Image",
                        image_mode="RGBA",
                        sources="upload",
                        type="pil",
                        elem_id="content_image",
                    )
                    processed_image = gr.Image(label="Processed Image", interactive=False)

                # params
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            # with gr.Group():

                            cfg_scale = gr.Number(
                                label="CFG-scale", value=4.0, interactive=True,
                            )
                            seed = gr.Number(
                                label="Seed", value=42, interactive=True,
                            )

                            # num_steps = gr.Number(
                            #     label="ODE Sampling Steps", value=250, interactive=True,
                            # )

                        # with gr.Column():
                        # with gr.Row():
                        #         mesh_size = gr.Number(
                        #             label="Mesh Resolution", value=192, interactive=True,
                        #     )

                        #         mesh_thres = gr.Number(
                        #             label="Mesh Iso-surface", value=10, interactive=True,
                        #         )

                with gr.Row():
                    with gr.Group():
                        preprocess_background = gr.Checkbox(
                            label="Remove Background", value=False
                        )
                with gr.Row():
                    submit = gr.Button("Generate", elem_id="generate", variant="primary")

                with gr.Row(variant="panel"): 
                    gr.Examples(
                        examples=[
                            str(path) for path in sorted(Path('./assets/demo-image-for-i23d/instantmesh').glob('**/*.png'))
                        ] + [str(path) for path in sorted(Path('./assets/demo-image-for-i23d/gso').glob('**/*.png'))],
                        inputs=[input_image],
                        cache_examples=False,
                        label="Examples",
                        examples_per_page=20,
                    )

            with gr.Column():
                with gr.Row():
                    with gr.Tab("Stage-2 Output"):
                        with gr.Column():
                            output_video = gr.Video(value=None, width=512, label="Rendered Video (2 LoDs)", autoplay=True, loop=True)
                            # output_video = gr.Video(value=None, width=256, label="Rendered Video", autoplay=True)
                            output_gs = gr.Model3D(
                                height=256,
                                label="2DGS Center",
                                pan_speed=0.5,
                                clear_color=(1,1,1,1), # loading glb file only.
                            )
                            output_model = gr.Model3D(
                                height=256,
                                label="TSDF Mesh",
                                pan_speed=0.5,
                                clear_color=(1,1,1,1), # loading tsdf ply files.
                            )

                    with gr.Tab("Stage-1 Output"):
                        with gr.Column():
                            output_model_stage1 = gr.Model3D(
                                height=256,
                                label="Stage-1",
                                pan_speed=0.5,
                                clear_color=(1,1,1,1), # loading tsdf ply files.
                            )



        gr.Markdown(
            """
            ## Comments:
            1. The sampling time varies since ODE-based sampling method (dopri5 by default) has adaptive internal step, and reducing sampling steps may not reduce the overal sampling time. Sampling steps=250 is the emperical value that works well in most cases.
            2. The 3D viewer shows a colored .glb mesh extracted from volumetric tri-plane, and may differ slightly with the volume rendering result.
            3. If you find your result unsatisfying, tune the CFG scale and change the random seed. Usually slightly increase the CFG value can lead to better performance.
            # 3. Known limitations include:
            # - Texture details missing: since our VAE is trained on 192x192 resolution due the the resource constraints, the texture details generated by the final 3D-LDM may be blurry. We will keep improving the performance in the future.
            3. Regarding reconstruction performance, our model is slightly inferior to state-of-the-art multi-view LRM-based method (e.g. InstantMesh), but offers much better diversity, flexibility and editing potential due to the intrinsic nature of diffusion model.

            ## How does it work?

            GaussianAnything is a native 3D Latent Diffusion Model that supports direct 3D asset generation via diffusion sampling. 
            Compared to SDS-based ([DreamFusion](https://dreamfusion3d.github.io/)), mulit-view generation-based ([MVDream](https://arxiv.org/abs/2308.16512), [Zero123++](https://github.com/SUDO-AI-3D/zero123plus), [Instant3D](https://instant-3d.github.io/)) and feedforward 3D reconstruction-based ([LRM](https://yiconghong.me/LRM/), [InstantMesh](https://github.com/TencentARC/InstantMesh), [LGM](https://github.com/3DTopia/LGM)), 
            GaussianAnything supports feedforward 3D generation with a unified framework.
            Like 2D/Video AIGC pipeline, GaussianAnything first trains a 3D-VAE and then conduct LDM training (text/image conditioned) on the learned latent space. Some related methods from the industry ([Shape-E](https://github.com/openai/shap-e), [CLAY](https://github.com/CLAY-3D/OpenCLAY), [Meta 3D Gen](https://arxiv.org/abs/2303.05371)) also follow the same paradigm.
            Though currently the performance of the origin 3D LDM's works are overall inferior to reconstruction-based methods, we believe the proposed method has much potential and scales better with more data and compute resources, and may yield better 3D editing performance due to its compatability with diffusion model.
            For more results see the [project page](https://nirvanalan.github.io/projects/GA/).
            """
        )

        submit.click(fn=check_input_image, inputs=[input_image]).success(
            fn=preprocess,
            inputs=[input_image, preprocess_background],
            outputs=[processed_image],
        ).success(
            # fn=reconstruct_and_export,
            # inputs=[processed_image],
            # outputs=[output_model, output_video],
            fn=cascaded_generation,
            inputs=[processed_image, seed, cfg_scale],
            # inputs=[processed_image, num_steps, seed, mesh_size, mesh_thres, unconditional_guidance_scale, args.stage_1_output_dir],
            outputs=[output_video, output_gs, output_model, output_model_stage1],
        )

    demo.queue(max_size=1)
    demo.launch(share=True)

if __name__ == "__main__":

    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.

    with open('configs/gradio_i23d_stage2_args.json') as f:
        args_2 = json.load(f)
        args_2 = EasyDict(args_2)
        args_2.local_rank = 0
        args_2.gpus = 1

    with open('configs/gradio_i23d_stage1_args.json') as f:
        args_1 = json.load(f)
        args_1 = EasyDict(args_1)
        args_1.local_rank = 0
        args_1.gpus = 1

    main(args_1, args_2)
