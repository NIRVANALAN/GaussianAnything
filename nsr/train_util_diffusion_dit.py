from .train_util_diffusion import TrainLoop3DDiffusion
import dnnlib
import torch as th


class TrainLoop3DDiffusionDiT(TrainLoop3DDiffusion):

    def __init__(self,
                 *,
                 rec_model,
                 denoise_model,
                 diffusion,
                 loss_class,
                 data,
                 eval_data,
                 batch_size,
                 microbatch,
                 lr,
                 ema_rate,
                 log_interval,
                 eval_interval,
                 save_interval,
                 resume_checkpoint,
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 schedule_sampler=None,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 ignore_resume_opt=False,
                 freeze_ae=False,
                 denoised_ae=True,
                 triplane_scaling_divider=10,
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         denoise_model=denoise_model,
                         diffusion=diffusion,
                         loss_class=loss_class,
                         data=data,
                         eval_data=eval_data,
                         batch_size=batch_size,
                         microbatch=microbatch,
                         lr=lr,
                         ema_rate=ema_rate,
                         log_interval=log_interval,
                         eval_interval=eval_interval,
                         save_interval=save_interval,
                         resume_checkpoint=resume_checkpoint,
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         schedule_sampler=schedule_sampler,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         ignore_resume_opt=ignore_resume_opt,
                         freeze_ae=freeze_ae,
                         denoised_ae=denoised_ae,
                         triplane_scaling_divider=triplane_scaling_divider,
                         use_amp=use_amp,
                         **kwargs)

        self.latent_name = 'latent_from_vit'
        self.render_latent_behaviour = 'vit_postprocess_triplane_dec'  # translate latent into 2D spatial tokens, then triplane render

    def eval_ddpm_sample(self):

        args = dnnlib.EasyDict(
            dict(batch_size=1,
                 image_size=224,
                 denoise_in_channels=self.ddp_rec_model.module.decoder.triplane_decoder.out_chans, # type: ignore
                 clip_denoised=False,
                 class_cond=False,
                 use_ddim=False))

        model_kwargs = {}

        if args.class_cond:
            classes = th.randint(low=0,
                                 high=NUM_CLASSES,
                                 size=(args.batch_size, ),
                                 device=dist_util.dev())
            model_kwargs["y"] = classes

        diffusion = self.diffusion
        sample_fn = (diffusion.p_sample_loop
                     if not args.use_ddim else diffusion.ddim_sample_loop)

        vit_L = (224//14)**2 # vit sequence length

        if self.ddp_rec_model.module.decoder.vit_decoder.cls_token: 
            vit_L += 1

        for i in range(1):
            triplane_sample = sample_fn(
                self.ddp_model,
                (args.batch_size, vit_L, self.ddp_rec_model.module.decoder.vit_decoder.embed_dim), # vit token size, N L C
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            th.cuda.empty_cache()
            self.render_video_given_triplane(
                triplane_sample,
                name_prefix=f'{self.step + self.resume_step}_{i}')