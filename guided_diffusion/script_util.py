import argparse
import inspect

from pdb import set_trace as st

from cldm.cldm import ControlledUnetModel, ControlNet

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel, EncoderUNetModel, UNetModelWithHint
import torch as th
# from dit.dit_models_xformers import DiT_models, TextCondDiTBlock
from dit.dit_models_xformers import TextCondDiTBlock, ImageCondDiTBlock, FinalLayer
from dit.dit_trilatent import DiT_models as DiT_models_t23d
from dit.dit_i23d import DiT_models as DiT_models_i23d

# if th.cuda.is_available():
#     from xformers.triton import FusedLayerNorm as LayerNorm

NUM_CLASSES = 1000


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        standarization_xt=False,
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        predict_v=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        mixed_prediction=False,  # ! to assign later
    )


def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        image_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )


def control_net_defaults():
    res = dict(
        only_mid_control=False,  # TODO
        control_key='img',
        normalize_clip_encoding=False,  # zero-shot text inference
        scale_clip_encoding=1.0,
        cfg_dropout_prob=0.0,  # dropout condition for CFG training
        cond_key='caption',
    )
    return res


def continuous_diffusion_defaults():
    # NVlabs/LSGM/train_vada.py
    res = dict(
        sde_time_eps=1e-2,
        sde_beta_start=0.1,
        sde_beta_end=20.0,
        sde_sde_type='vpsde',
        sde_sigma2_0=0.0,  # ?
        iw_sample_p='drop_sigma2t_iw',
        iw_sample_q='ll_iw',
        iw_subvp_like_vp_sde=False,
        train_vae=True,
        pred_type='eps',  # [x0, eps]
        # joint_train=False,
        p_rendering_loss=False,
        unfix_logit=False,
        loss_type='eps',
        loss_weight='simple',  # snr snr_sqrt sigmoid_snr
        # train_vae_denoise_rendering=False,
        diffusion_ce_anneal=True,
        enable_mixing_normal=True,
    )

    return res


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        # image_size=64,
        diffusion_input_size=224,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        denoise_in_channels=3,
        denoise_out_channels=3,
        # ! controlnet args
        create_controlnet=False,
        create_dit=False,
        i23d=False,
        create_unet_with_hint=False,
        dit_model_arch='DiT-L/2',
        # ! ldm unet support
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=-1,  # custom transformer support
        pooling_ctx_dim=-1,
        roll_out=False,  # whether concat in batch, not channel
        n_embed=
        None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        mixing_logit_init=-6,
        hint_channels=3,
        # unconditional_guidance_scale=1.0,
        # normalize_clip_encoding=False, # for zero-shot conditioning
    )
    res.update(diffusion_defaults())
    # res.update(continuous_diffusion_defaults())
    return res


def classifier_and_diffusion_defaults():
    res = classifier_defaults()
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    # image_size,
    diffusion_input_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    predict_v,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    denoise_in_channels,
    denoise_out_channels,
    standarization_xt,
    mixed_prediction,
    # controlnet
    create_controlnet,
    # only_mid_control,
    # control_key,
    use_spatial_transformer,
    transformer_depth,
    context_dim,
    pooling_ctx_dim,
    n_embed,
    legacy,
    mixing_logit_init,
    create_dit,
    i23d,
    create_unet_with_hint,
    dit_model_arch,
    roll_out,
    hint_channels,
    # unconditional_guidance_scale,
    # normalize_clip_encoding,
):
    model = create_model(
        diffusion_input_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        denoise_in_channels=denoise_in_channels,
        denoise_out_channels=denoise_out_channels,
        mixed_prediction=mixed_prediction,
        create_controlnet=create_controlnet,
        # only_mid_control=only_mid_control,
        # control_key=control_key,
        use_spatial_transformer=use_spatial_transformer,
        transformer_depth=transformer_depth,
        context_dim=context_dim,
        pooling_ctx_dim=pooling_ctx_dim,
        n_embed=n_embed,
        legacy=legacy,
        mixing_logit_init=mixing_logit_init,
        create_dit=create_dit,
        i23d=i23d,
        create_unet_with_hint=create_unet_with_hint,
        dit_model_arch=dit_model_arch,
        roll_out=roll_out,
        hint_channels=hint_channels,
        # normalize_clip_encoding=normalize_clip_encoding,
    )
    diffusion = create_gaussian_diffusion(
        diffusion_steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        predict_v=predict_v,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        standarization_xt=standarization_xt,
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    # denoise_in_channels=3,
    denoise_in_channels=-1,
    denoise_out_channels=3,
    mixed_prediction=False,
    create_controlnet=False,
    create_dit=False,
    # t23d=True,
    i23d=False,
    create_unet_with_hint=False,
    dit_model_arch='DiT-L/2',
    hint_channels=3,
    use_spatial_transformer=False,  # custom transformer support
    transformer_depth=1,  # custom transformer support
    context_dim=None,  # custom transformer support
    pooling_ctx_dim=768,
    n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
    legacy=True,
    mixing_logit_init=-6,
    roll_out=False,
    # normalize_clip_encoding=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 448:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 320:  # ffhq
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 224 and denoise_in_channels == 144:  # ffhq
            channel_mult = (1, 1, 2, 3, 4, 4)
        elif image_size == 224:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)

        elif image_size == 32:  # https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml#L37
            channel_mult = (1, 2, 4, 4)

        elif image_size == 16:  # B,12,16,16. just for baseline check. not good performance.
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(
            int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if create_controlnet:

        controlledUnetModel = ControlledUnetModel(
            image_size=image_size,
            in_channels=denoise_in_channels,
            model_channels=num_channels,
            # out_channels=(3 if not learn_sigma else 6),
            out_channels=(denoise_out_channels
                          if not learn_sigma else denoise_out_channels * 2),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            mixed_prediction=mixed_prediction,
            # ldm support
            use_spatial_transformer=use_spatial_transformer,
            transformer_depth=transformer_depth,
            context_dim=context_dim,
            pooling_ctx_dim=pooling_ctx_dim,
            n_embed=n_embed,
            legacy=legacy,
            mixing_logit_init=mixing_logit_init,
            roll_out=roll_out
            )

        controlNet = ControlNet(
            image_size=image_size,
            in_channels=denoise_in_channels,
            model_channels=num_channels,
            # ! condition channels
            hint_channels=hint_channels,
            # out_channels=(3 if not learn_sigma else 6),
            # out_channels=(denoise_out_channels
            #             if not learn_sigma else denoise_out_channels * 2),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            # num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            roll_out=roll_out
        )
        # mixed_prediction=mixed_prediction)

        return controlledUnetModel, controlNet

    elif create_dit:
        # if i23d
        if i23d:
            return DiT_models_i23d[dit_model_arch](
                input_size=image_size,
                num_classes=0,
                learn_sigma=learn_sigma,
                in_channels=denoise_in_channels,
                context_dim=context_dim,  # add CLIP text embedding
                roll_out=roll_out, 
                # vit_blk=ImageCondDiTBlock,
                pooling_ctx_dim=pooling_ctx_dim,)
        else: # t23d
            return DiT_models_t23d[dit_model_arch](
                input_size=image_size,
                num_classes=0,
                learn_sigma=learn_sigma,
                in_channels=denoise_in_channels,
                context_dim=context_dim,  # add CLIP text embedding
                roll_out=roll_out, 
                # vit_blk=TextCondDiTBlock
                )
    else:

        if create_unet_with_hint:
            unet_cls = UNetModelWithHint
        else:
            unet_cls = UNetModel

        # st()
        return unet_cls(
            image_size=image_size,
            in_channels=denoise_in_channels,
            model_channels=num_channels,
            # out_channels=(3 if not learn_sigma else 6),
            out_channels=(denoise_out_channels
                          if not learn_sigma else denoise_out_channels * 2),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            mixed_prediction=mixed_prediction,
            # ldm support
            use_spatial_transformer=use_spatial_transformer,
            transformer_depth=transformer_depth,
            context_dim=context_dim,
            pooling_ctx_dim=pooling_ctx_dim,
            n_embed=n_embed,
            legacy=legacy,
            mixing_logit_init=mixing_logit_init,
            roll_out=roll_out,
            hint_channels=hint_channels,
            # normalize_clip_encoding=normalize_clip_encoding,
        )


def create_classifier_and_diffusion(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
):
    classifier = create_classifier(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion


def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=1000,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        image_size=large_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )


def create_gaussian_diffusion(
    *,
    diffusion_steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    predict_v=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    standarization_xt=False,
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE  # * used here.
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    if predict_xstart:
        model_mean_type = gd.ModelMeanType.START_X
    elif predict_v:
        model_mean_type = gd.ModelMeanType.V
    else:
        model_mean_type = gd.ModelMeanType.EPSILON

        # model_mean_type=(
        #     gd.ModelMeanType.EPSILON if not predict_xstart else
        #     gd.ModelMeanType.START_X  # * used gd.ModelMeanType.EPSILON
        # ),

    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        # (
        #     gd.ModelMeanType.EPSILON if not predict_xstart else
        #     gd.ModelMeanType.START_X  # * used gd.ModelMeanType.EPSILON
        # ),
        model_var_type=((
            gd.ModelVarType.FIXED_LARGE  # * used here
            if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                        if not learn_sigma else gd.ModelVarType.LEARNED_RANGE),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        standarization_xt=standarization_xt,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
