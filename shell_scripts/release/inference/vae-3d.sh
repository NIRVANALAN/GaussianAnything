set -x 

lpips_lambda=2.0
ssim_lambda=0.
l1_lambda=0. # following gaussian splatting
l2_lambda=1 # ! use_conf_map

NUM_GPUS=1

image_size=512
image_size_encoder=512

num_workers=2 # for debug

patch_size=14
kl_lambda=1.0e-06

perturb_pcd_scale=0

num_frames=8
batch_size=1 # ! actuall BS will double

microbatch=$(( num_frames*batch_size*2 ))

data_dir=./assets/demo-image-for-i23d/for-vae-reconstruction/

DATASET_FLAGS="
 --data_dir ${data_dir} \
 --eval_data_dir ${data_dir} \
"

# raw inference
conv_lr=0
lr=0

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

TRAIN_FLAGS="--iterations 10001 --anneal_lr False \
 --batch_size $batch_size --save_interval 10000 \
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
 --lpips_delay_iter 100 \
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
 --resume_checkpoint ./checkpoint/model_rec1965000.pt \
 "


logdir=./logs/latent_dir/768-512-perturb${perturb_pcd_scale}

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

SR_TRAIN_FLAGS=${SR_TRAIN_FLAGS_v1_2XC}


rm -rf "$logdir"/runs
mkdir -p "$logdir"/
cp "$0" "$logdir"/

# localedef -c -f UTF-8 -i en_US en_US.UTF-8
# export LC_ALL=en_US.UTF-8

export OPENCV_IO_ENABLE_OPENEXR=1
export OMP_NUM_THREADS=12
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_IB_GID_INDEX=3 # https://github.com/huggingface/accelerate/issues/314#issuecomment-1821973930

for wds_split in 0

do

export CUDA_VISIBLE_DEVICES=$(( 0 + $wds_split ))
port=$(( 14000 + $wds_split ))

torchrun --nproc_per_node=$NUM_GPUS \
  --nnodes=1 \
  --rdzv-endpoint=localhost:${port} \
  --rdzv_backend=c10d \
 scripts/vit_triplane_train.py \
 --trainer_name nv_rec_patch_mvE_gs \
 --num_workers ${num_workers} \
 ${TRAIN_FLAGS}  \
 ${SR_TRAIN_FLAGS} \
 ${DATASET_FLAGS} \
 --lpips_lambda $lpips_lambda \
 --overfitting False \
 --load_pretrain_encoder False \
 --iterations 5000001 \
 --save_interval 10000 \
 --eval_interval 250000000 \
 --decomposed True \
 --logdir $logdir \
 --decoder_load_pretrained False \
 --cfg objverse_tuneray_aug_resolution_64_64_auto \
 --patch_size ${patch_size} \
 --use_amp True \
 --eval_batch_size ${batch_size} \
 ${LR_FLAGS} \
 --l1_lambda ${l1_lambda} \
 --l2_lambda ${l2_lambda} \
 --ssim_lambda ${ssim_lambda} \
 --depth_smoothness_lambda 0 \
 --use_conf_map False \
 --objv_dataset True \
 --depth_lambda 0.5 \
 --use_lmdb_compressed False \
 --use_lmdb False \
 --mv_input True \
 --inference True \
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
 --pcd_path /mnt/sfs-common/yslan/Dataset/Obajverse/FPS_PCD/pcd-V=10_4096_polish_fullset/fps-pcd \
 --pt_ft_kl False \
 --surfel_rendering True \
 --plane_n 1 \
 --latent_num 768 \
 --perturb_pcd_scale ${perturb_pcd_scale} \
 --wds_split ${wds_split} \

#  --pcd_path /nas/shared/V2V/yslan/logs/nips23/Reconstruction/pcd-V=10_4096_polish/fps-pcd \

done

