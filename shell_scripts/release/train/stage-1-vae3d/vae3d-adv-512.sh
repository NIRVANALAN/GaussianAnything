set -x 
# vit_decoder_lr=1.001

lpips_lambda=0.8
l1_lambda=1.0 # following gaussian splatting

l2_lambda=0.0
ssim_lambda=0.0
lambda_normal=0.025

lambda_dist=1000
overfitting=False


NUM_GPUS=8
num_workers=8 # 
batch_size=1


patchgan_disc_factor=0.025
patchgan_disc_g_weight=0.025
perturb_pcd_scale=0.01


image_size=512
image_size_encoder=512

patch_size=14
kl_lambda=2.5e-06
patch_rendering_resolution=${image_size}

num_frames=8

microbatch=$(( num_frames*batch_size*2 ))

dataset_name=75K
data_dir=/cpfs01/user/lanyushi.p/data/chunk-jpeg-normal/bs_16_fixsave3/170K/512/


DATASET_FLAGS="
 --data_dir ${data_dir} \
 --eval_data_dir ${data_dir} \
"

# half LR since BS halved, during high-res finetuning
conv_lr=1e-4
lr=5e-5 

vit_decoder_lr=$lr
encoder_lr=${conv_lr} # scaling version , could be larger when multi-nodes
triplane_decoder_lr=$conv_lr
super_resolution_lr=$conv_lr

# * above the best lr config

LR_FLAGS="--encoder_lr $encoder_lr \
--vit_decoder_lr $vit_decoder_lr \
--triplane_decoder_lr $triplane_decoder_lr \
--super_resolution_lr $super_resolution_lr \
--lr $lr"

TRAIN_FLAGS="--iterations 5000 --anneal_lr False \
 --batch_size $batch_size \
 --microbatch ${microbatch} \
 --image_size_encoder $image_size_encoder \
 --dino_version mv-sd-dit-srt-pcd-structured-nopcd \
 --sr_training False \
 --cls_token False \
 --weight_decay 0.05 \
 --image_size $image_size \
 --kl_lambda ${kl_lambda} \
 --no_dim_up_mlp True \
 --uvit_skip_encoder False \
 --fg_mse True \
 --bg_lamdba 1.0 \
 --lpips_delay_iter 25000 \
 --sr_delay_iter 25000 \
 --kl_anneal True \
 --symmetry_loss False \
 --vae_p 2 \
 --plucker_embedding True \
 --encoder_in_channels 15 \
 --arch_dit_decoder DiT2-B/2 \
 --sd_E_ch 64 \
 --sd_E_num_res_blocks 1 \
 --lrm_decoder False \
 --resume_checkpoint /nas/shared/public/yslan/logs/vae/f=8-cascade/latent=768-8x3x3-fullset-surfacePCD-adv/bs1-gpu8-0.025-0.01-advFinestOnly_512_perturb/model_rec1665000.pt \
 "

logdir=/nas/shared/public/yslan/logs/vae/f=8-cascade/latent=768-8x3x3-fullset-surfacePCD-adv/bs${batch_size}-gpu${NUM_GPUS}-${patchgan_disc_factor}-${patchgan_disc_g_weight}-advFinestOnly_512_perturb-largeradv


SR_TRAIN_FLAGS_v1_2XC="
--decoder_in_chans 32 \
--out_chans 96 \
--alpha_lambda 1.0 \
--logdir $logdir \
--arch_encoder vits \
--arch_decoder vitb \
--vit_decoder_wd 0.001 \
--encoder_weight_decay 0.001 \
--color_criterion mse \
--decoder_output_dim 3 \
--ae_classname vit.vit_triplane.pcd_structured_latent_space_vae_decoder_cascaded \
"

# pcd_structured_latent_space_lion_learnoffset_surfel_sr_noptVAE
# pcd_structured_latent_space_lion_learnoffset_surfel_sr_noptVAE_debugscale_f1

SR_TRAIN_FLAGS=${SR_TRAIN_FLAGS_v1_2XC}


rm -rf "$logdir"/runs
mkdir -p "$logdir"/
cp "$0" "$logdir"/

# localedef -c -f UTF-8 -i en_US en_US.UTF-8
# export LC_ALL=en_US.UTF-8

export OPENCV_IO_ENABLE_OPENEXR=1
export OMP_NUM_THREADS=12
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_GID_INDEX=3 # https://github.com/huggingface/accelerate/issues/314#issuecomment-1821973930
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=6,7
# export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=4,5
# export CUDA_VISIBLE_DEVICES=0,1,2,3


  # --rdzv-endpoint=localhost:19381 \

torchrun --rdzv-endpoint=localhost:19408 \
  --nproc_per_node=$NUM_GPUS \
  --nnodes=1 \
  --rdzv_backend=c10d \
 scripts/vit_triplane_train.py \
 --trainer_name nv_rec_patch_mvE_gs_disc \
 --num_workers ${num_workers} \
 ${TRAIN_FLAGS}  \
 ${SR_TRAIN_FLAGS} \
 ${DATASET_FLAGS} \
 --lpips_lambda $lpips_lambda \
 --overfitting ${overfitting} \
 --load_pretrain_encoder False \
 --iterations 300000 \
 --save_interval 10000 \
 --eval_interval 250000000 \
 --decomposed True \
 --logdir $logdir \
 --decoder_load_pretrained False \
 --cfg objverse_tuneray_aug_resolution_64_64_auto \
 --patch_size ${patch_size} \
 --use_amp True \
 --eval_batch_size 1 \
 ${LR_FLAGS} \
 --l1_lambda ${l1_lambda} \
 --l2_lambda ${l2_lambda} \
 --ssim_lambda ${ssim_lambda} \
 --lambda_normal ${lambda_normal} \
 --lambda_dist ${lambda_dist} \
 --depth_smoothness_lambda 0 \
 --use_conf_map False \
 --objv_dataset True \
 --depth_lambda 0.0 \
 --patch_rendering_resolution ${patch_rendering_resolution} \
 --use_lmdb_compressed False \
 --use_lmdb False \
 --mv_input True \
 --split_chunk_input True \
 --four_view_for_latent False \
 --append_depth False \
 --gs_cam_format True \
 --gs_rendering True \
 --shuffle_across_cls True \
 --z_channels 10 \
 --ldm_z_channels 10 \
 --return_all_dit_layers False \
 --ldm_embed_dim 10 \
 --xyz_lambda 0.0 \
 --emd_lambda 0.0 \
 --cd_lambda 0.0 \
 --fps_sampling True \
 --subset_fps_sampling False \
 --subset_half_fps_sampling False \
 --num_frames ${num_frames} \
 --frame_0_as_canonical False \
 --split_chunk_size $((num_frames + num_frames)) \
 --read_normal True \
 --in_plane_attention False \
 --load_pcd True \
 --rand_aug_bg True \
 --use_wds False \
 --append_xyz True \
 --use_chunk True \
 --pcd_path /cpfs01/user/lanyushi.p/data/FPS_PCD/pcd-V=10_4096_polish_fullset/fps-pcd/ \
 --pt_ft_kl False \
 --ft_kl True \
 --lambda_scale_reg 0.0 \
 --latent_num 768 \
 --lambda_opa_reg 0.01 \
 --surfel_rendering True \
 --patchgan_disc_factor ${patchgan_disc_factor} \
 --patchgan_disc_g_weight ${patchgan_disc_g_weight} \
 --perturb_pcd_scale ${perturb_pcd_scale} \
 --latent_num 768 \
 --plane_n 1 \
