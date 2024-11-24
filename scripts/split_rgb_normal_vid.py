# import imageio.v3 as imageio
import imageio
from tqdm import tqdm
from pdb import set_trace as st
import glob
import numpy as np
import os
from pathlib import Path
# ! pip install opencv-python
import cv2
import matplotlib.pyplot as plt


def save_2dgs_rgb_normal_vid(vid_path, output_dir):

    frames = imageio.v3.imread(vid_path)
    
    vid_name = Path(vid_path).stem

    # output frames here
    # output_dir = f'{ga_output_dir}/{index}'
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    # all_rgb_frames, all_normal_frames = [], []

    rgb_video_out = imageio.get_writer(
        f'{output_dir}/rgb/{vid_name}-rgb.mp4',
        mode='I',
        fps=24,
        codec='libx264')

    normal_video_out = imageio.get_writer(
        f'{output_dir}/normal/{vid_name}-normal.mp4',
        mode='I',
        fps=24,
        codec='libx264')

    depth_video_out = imageio.get_writer(
        f'{output_dir}/normal/{vid_name}-depth.mp4',
        mode='I',
        fps=24,
        codec='libx264')


    for idx, frame in enumerate(frames[:24]):
        # frame_size = 512
        frame_size = frame.shape[1] // 3

        # rgb_video_out.append_data(frame[-384:, :384])
        # normal_video_out.append_data(frame[-384:, 384*2:384*3])

        rgb = frame[-frame_size:, :frame_size]
        rgb_video_out.append_data(cv2.resize(rgb, (384, 384)))

        depth = frame[-frame_size:, frame_size*2:frame_size*3]
        depth_video_out.append_data(cv2.resize(depth, (384, 384)))

        normal = frame[-frame_size:, frame_size*1:frame_size*2]
        normal_video_out.append_data(cv2.resize(normal, (384, 384)))

    rgb_video_out.close()
    normal_video_out.close()
    depth_video_out.close()


# output_dir = '/mnt/sfs-common/yslan/Repo/3dgen/GA-logs/demo-video-buffer'
# vid_input_dir = '/mnt/sfs-common/yslan/open-source/latent_dir/gs-latent-dim=10-fullset-cascade-fixpcd-adv_xyzaug_loweradv_768-fixinprange'

output_dir = '/mnt/sfs-common/yslan/Repo/3dgen/GA-logs/demo-video-buffer-192'
vid_input_dir = '/mnt/sfs-common/yslan/open-source/latent_dir/gs-latent-dim=10-fullset-cascade-fixpcd-adv_xyzaug_768-512-perturb0-debug'

os.makedirs(os.path.join(output_dir, 'rgb'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'normal'), exist_ok=True)



all_vids = glob.glob(os.path.join(vid_input_dir, '*.mp4'))

for vid_path in tqdm(all_vids[:]):
    # if 'daily-used' in vid_path: # only on fancy categories.
    #     continue
    try:
        save_2dgs_rgb_normal_vid(vid_path, output_dir)
    except:
        print(vid_path)