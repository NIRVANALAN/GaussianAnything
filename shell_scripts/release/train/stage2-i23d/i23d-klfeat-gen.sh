set -x 

lpips_lambda=0.8

image_size=512
image_size_encoder=${image_size}

patch_size=14

cfg_dropout_prob=0.1 # SD config
dataset_name="9cls"
num_frames=8

data_dir=/cpfs01/user/lanyushi.p/data/unzip4_img
pcd_path=/mnt/sfs-common/yslan/Dataset/Obajverse/FPS_PCD/pcd-V=10_4096_polish_fullset/fps-pcd
mv_latent_dir=/mnt/sfs-common/yslan/open-source/latent_dir/gs-latent-dim=10-fullset-cascade-fixpcd-adv_xyzaug_loweradv_768-fixinprange/latent_dir

num_workers=6
NUM_GPUS=8
n_cond_frames=5
batch_size=4

# microbatch=${batch_size}
microbatch=$(( n_cond_frames*batch_size*2 ))

DATASET_FLAGS="
 --data_dir ${data_dir} \
 --eval_data_dir ${data_dir} \
"

lr=1e-4

kl_lambda=0
vit_lr=1e-5 # for improved-diffusion unet
ce_lambda=0 # ?
conv_lr=5e-5
alpha_lambda=1
scale_clip_encoding=1

triplane_scaling_divider=1.0 # for xyz diffusion

LR_FLAGS="--encoder_lr $vit_lr \
 --vit_decoder_lr $vit_lr \
 --lpips_lambda $lpips_lambda \
 --triplane_decoder_lr $conv_lr \
 --super_resolution_lr $conv_lr \
 --lr $lr \
 --kl_lambda ${kl_lambda} \
 --bg_lamdba 0.01 \
 --alpha_lambda ${alpha_lambda} \
"

TRAIN_FLAGS="--iterations 10001 --anneal_lr False \
 --batch_size $batch_size --save_interval 10000 \
 --microbatch ${microbatch} \
 --image_size_encoder $image_size_encoder \
 --image_size $image_size \
 --dino_version mv-sd-dit-srt-pcd-structured-nopcd \
 --sr_training False \
 --encoder_cls_token False \
 --decoder_cls_token False \
 --cls_token False \
 --weight_decay 0.05 \
 --no_dim_up_mlp True \
 --uvit_skip_encoder True \
 --decoder_load_pretrained True \
 --fg_mse False \
 --vae_p 2 \
 --plucker_embedding True \
 --encoder_in_channels 15 \
 --arch_dit_decoder DiT2-B/2 \
 --sd_E_ch 64 \
 --sd_E_num_res_blocks 1 \
 --lrm_decoder False \
 "


DDPM_MODEL_FLAGS="
--learn_sigma False \
--num_heads 8 \
--num_res_blocks 2 \
--num_channels 320 \
--attention_resolutions "4,2,1" \
--use_spatial_transformer True \
--transformer_depth 1 \
--context_dim 1024 \
"


# ! diffusion steps and noise schedule not used, since the continuous diffusion is adopted.
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear \
--use_kl False \
--triplane_scaling_divider ${triplane_scaling_divider} \
--trainer_name flow_matching_gs \
--mixed_prediction False \
--train_vae False \
--denoise_in_channels 10 \
--denoise_out_channels 10 \
--diffusion_input_size 32 \
--diffusion_ce_anneal True \
--create_controlnet False \
--p_rendering_loss False \
--pred_type x_start \
--predict_v False \
--create_dit True \
--i23d True \
--dit_model_arch DiT-PixArt-PCD-CLAY-stage2-L \
--train_vae False \
--use_eos_feature False \
--roll_out True \
"

# --dit_model_arch DiT-PixArt-PCD-CLAY-L \

# logdir=/nas/shared/public/yslan/logs/nips24/LSGM/t23d/FM/${dataset_name}/gs-disentangle/cascade_check/clay/stage2/dino_img/fullset_${NUM_GPUS}-${batch_size}
# logdir=/nas/shared/public/yslan/logs/nips24/LSGM/t23d/FM/${dataset_name}/gs-disentangle/cascade_check/clay/stage2/dino_img/fullset_${NUM_GPUS}-${batch_size}-ctd
# logdir=/nas/shared/public/yslan/logs/nips24/LSGM/t23d/FM/${dataset_name}/gs-disentangle/cascade_check/clay/stage2/dino_img/fullset_${NUM_GPUS}-${batch_size}-ctd2
# logdir=/nas/shared/public/yslan/logs/nips24/LSGM/t23d/FM/${dataset_name}/gs-disentangle/cascade_check/clay/stage2/dino_img/noperturb/fullset_${NUM_GPUS}-${batch_size}
logdir=/nas/shared/public/yslan/logs/nips24/LSGM/t23d/FM/${dataset_name}/gs-disentangle/cascade_check/clay/stage2/dino_img/noperturb/196w_latent-fullset_${NUM_GPUS}-${batch_size}-ctd

SR_TRAIN_FLAGS_v1_2XC="
--decoder_in_chans 32 \
--out_chans 96 \
--ae_classname vit.vit_triplane.pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr_cascade_x8x4x4_512 \
--logdir $logdir \
--arch_encoder vits \
--arch_decoder vitb \
--vit_decoder_wd 0.001 \
--encoder_weight_decay 0.001 \
--color_criterion mse \
--triplane_in_chans 32 \
--decoder_output_dim 3 \
--resume_checkpoint /nas/shared/public/yslan/logs/nips24/LSGM/t23d/FM/9cls/gs-disentangle/cascade_check/clay/stage2/dino_img/noperturb/196w_latent-fullset_8-4/model_joint_denoise_rec_model2045000.pt \
"

# /nas/shared/public/yslan/logs/vae/f=8-cascade/latent=768-8x3x3-fullset-surfacePCD-adv/bs1-gpu8-0.025-0.025-advFinestOnly_512_perturb-largeradv/model_rec1965000.pt \

# --resume_checkpoint /nas/shared/public/yslan/logs/nips24/LSGM/t23d/FM/9cls/gs-disentangle/cascade_check/clay/stage2/dino_img/noperturb/fullset_8-40/model_joint_denoise_rec_model1925000.pt \

# /nas/shared/public/yslan/logs/nips24/LSGM/t23d/FM/9cls/gs-disentangle/cascade_check/clay/stage2/dino_img/fullset_4-32-ctd2/model_joint_denoise_rec_model1885000.pt \

# /nas/shared/public/yslan/logs/nips24/LSGM/t23d/FM/9cls/gs-disentangle/cascade_check/clay/stage2/dino_img/fullset_8-32-ctd/model_joint_denoise_rec_model1825000.pt \

# /nas/shared/public/yslan/logs/nips24/LSGM/t23d/FM/9cls/gs-disentangle/cascade_check/clay/stage2/dino_img/fullset_8-32/model_joint_denoise_rec_model1785000.pt \
# /nas/shared/public/yslan/logs/vae/f=8-cascade/latent=768-8x3x3-fullset-surfacePCD-adv/bs2-gpu8-0.025-0.01-advFinestOnly-perturbxyz/model_rec1635000.pt \

# /nas/shared/V2V/yslan/logs/nips24/LSGM/t23d/FM/9cls/gs-disentangle/cascade_check/clay/CA_dinoonly_ditb_overfit_CAFirst_dino518/model_joint_denoise_rec_model1750000.pt \


# --resume_checkpoint /nas/shared/V2V/yslan/logs/nips24/LSGM/t23d/FM/9cls/gs-disentangle/cascade_check/xyz_output/model_joint_denoise_rec_model1600000.pt \

SR_TRAIN_FLAGS=${SR_TRAIN_FLAGS_v1_2XC}

rm -rf "$logdir"/runs
mkdir -p "$logdir"/
cp "$0" "$logdir"/

export OMP_NUM_THREADS=12
# export LC_ALL=en_US.UTF-8 # save caption txt bug
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OPENCV_IO_ENABLE_OPENEXR=1
export NCCL_IB_GID_INDEX=3 # https://github.com/huggingface/accelerate/issues/314#issuecomment-1821973930
# export CUDA_VISIBLE_DEVICES=0,1,2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=$NUM_GPUS \
  --nnodes 1 \
  --rdzv-endpoint=localhost:21378 \
 scripts/vit_triplane_sit_train.py \
 --num_workers ${num_workers} \
 --depth_lambda 0 \
 ${TRAIN_FLAGS}  \
 ${SR_TRAIN_FLAGS} \
 ${DATASET_FLAGS} \
 ${DIFFUSION_FLAGS} \
 ${DDPM_MODEL_FLAGS} \
 --overfitting False \
 --load_pretrain_encoder False \
 --iterations 5000001 \
 --eval_interval 5000 \
 --decomposed True \
 --logdir $logdir \
 --cfg objverse_tuneray_aug_resolution_64_64_auto \
 --patch_size ${patch_size} \
 --eval_batch_size 1 \
 ${LR_FLAGS} \
 --ce_lambda ${ce_lambda} \
 --negative_entropy_lambda ${ce_lambda} \
 --triplane_fg_bg False \
 --grad_clip True \
 --interval 5 \
 --log_interval 100 \
 --normalize_clip_encoding True \
 --scale_clip_encoding ${scale_clip_encoding} \
 --mixing_logit_init 10000 \
 --objv_dataset True \
 --cfg_dropout_prob ${cfg_dropout_prob} \
 --cond_key img-xyz \
 --use_lmdb_compressed False \
 --use_lmdb False \
 --use_amp True \
 --append_xyz True \
 --allow_tf32 True \
 --gs_cam_format True \
 --gs_rendering True \
 --shuffle_across_cls True \
 --z_channels 10 \
 --ldm_z_channels 10 \
 --ldm_embed_dim 10 \
 --load_wds_diff False \
 --load_wds_latent False \
 --compile False \
 --split_chunk_input True \
 --append_depth False \
 --mv_input True \
 --duplicate_sample False \
 --read_normal True \
 --enable_mixing_normal False \
 --use_wds False \
 --use_chunk True \
 --pt_ft_kl False \
 --surfel_rendering True \
 --clip_grad_throld 1.0 \
 --snr-type img-uniform-gvp-dino-stage2 \
 --load_pcd True \
 --num_frames ${num_frames} \
 --split_chunk_size 16 \
 --load_caption_dataset False \
 --plane_n 1 \
 --pooling_ctx_dim 768 \
 --pcd_path ${pcd_path} \
 --mv_latent_dir ${mv_latent_dir}
