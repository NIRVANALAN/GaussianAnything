import os
from tqdm import tqdm
import kiui
from kiui.op import recenter
import kornia
import collections
import math
import time
import itertools
import pickle
from typing import Any
import lmdb
import cv2
import trimesh

cv2.setNumThreads(0)  # disable multiprocess
# import imageio
import imageio.v3 as imageio
import numpy as np
from PIL import Image
import Imath
import OpenEXR
from pdb import set_trace as st
from pathlib import Path
import torchvision
from torchvision.transforms import v2

from einops import rearrange, repeat
from functools import partial
import io
from scipy.stats import special_ortho_group
import gzip
import random
import torch
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import lz4.frame
from nsr.volumetric_rendering.ray_sampler import RaySampler
import point_cloud_utils as pcu

import torch.multiprocessing

# torch.multiprocessing.set_sharing_strategy('file_system')

from utils.general_utils import PILtoTorch, matrix_to_quaternion

from guided_diffusion import logger
import json

import webdataset as wds
from webdataset.shardlists import expand_source
# st()

from .shapenet import LMDBDataset, LMDBDataset_MV_Compressed, decompress_and_open_image_gzip, decompress_array
from kiui.op import safe_normalize

from utils.gs_utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World

from nsr.camera_utils import generate_input_camera


def random_rotation_matrix():
    # Generate a random rotation matrix in 3D
    random_rotation_3d = special_ortho_group.rvs(3)

    # Embed the 3x3 rotation matrix into a 4x4 matrix
    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[:3, :3] = random_rotation_3d

    return rotation_matrix_4x4


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def resize_depth_mask(depth_to_resize, resolution):
    depth_resized = cv2.resize(depth_to_resize, (resolution, resolution),
                               interpolation=cv2.INTER_LANCZOS4)
    #    interpolation=cv2.INTER_AREA)
    return depth_resized, depth_resized > 0  # type: ignore


def resize_depth_mask_Tensor(depth_to_resize, resolution):

    if depth_to_resize.shape[-1] != resolution:
        depth_resized = torch.nn.functional.interpolate(
            input=depth_to_resize.unsqueeze(1),
            size=(resolution, resolution),
            # mode='bilinear',
            mode='nearest',
            # align_corners=False,
        ).squeeze(1)
    else:
        depth_resized = depth_to_resize

    return depth_resized.float(), depth_resized > 0  # type: ignore


class PostProcess:

    def __init__(
        self,
        reso,
        reso_encoder,
        imgnet_normalize,
        plucker_embedding,
        decode_encode_img_only,
        mv_input,
        split_chunk_input,
        duplicate_sample,
        append_depth,
        gs_cam_format,
        orthog_duplicate,
        frame_0_as_canonical,
        pcd_path=None,
        load_pcd=False,
        split_chunk_size=8,
        append_xyz=False,
    ) -> None:

        self.load_pcd = load_pcd

        if pcd_path is None:  # hard-coded
            pcd_path = '/cpfs01/user/lanyushi.p/data/FPS_PCD/pcd-V=6_256_again/fps-pcd/'

        self.pcd_path = Path(pcd_path)

        self.append_xyz = append_xyz
        if append_xyz:
            assert append_depth is False
        self.frame_0_as_canonical = frame_0_as_canonical
        self.gs_cam_format = gs_cam_format
        self.append_depth = append_depth
        self.plucker_embedding = plucker_embedding
        self.decode_encode_img_only = decode_encode_img_only
        self.duplicate_sample = duplicate_sample
        self.orthog_duplicate = orthog_duplicate

        self.zfar = 100.0
        self.znear = 0.01

        transformations = []
        if not split_chunk_input:
            transformations.append(transforms.ToTensor())

        if imgnet_normalize:
            transformations.append(
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))  # type: ignore
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)))  # type: ignore

        self.normalize = transforms.Compose(transformations)

        self.reso_encoder = reso_encoder
        self.reso = reso
        self.instance_data_length = 40
        # self.pair_per_instance = 1 # compat
        self.mv_input = mv_input
        self.split_chunk_input = split_chunk_input  # 8
        self.chunk_size = split_chunk_size if split_chunk_input else 40
        # assert self.chunk_size in [8, 10]
        self.V = self.chunk_size // 2  # 4 views as input
        # else:
        #     assert self.chunk_size == 20
        #     self.V = 12  # 6 + 6 here

        # st()
        assert split_chunk_input
        self.pair_per_instance = 1
        # else:
        #     self.pair_per_instance = 4 if mv_input else 2  # check whether improves IO

        self.ray_sampler = RaySampler()  # load xyz

    def gen_rays(self, c):
        # Generate rays
        intrinsics, c2w = c[16:], c[:16].reshape(4, 4)
        self.h = self.reso_encoder
        self.w = self.reso_encoder
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
            indexing='ij')

        # normalize to 0-1 pixel range
        yy = yy / self.h
        xx = xx / self.w

        # K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
        cx, cy, fx, fy = intrinsics[2], intrinsics[5], intrinsics[
            0], intrinsics[4]
        # cx *= self.w
        # cy *= self.h

        # f_x = f_y = fx * h / res_raw
        c2w = torch.from_numpy(c2w).float()

        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        del xx, yy, zz
        # st()
        dirs = (c2w[None, :3, :3] @ dirs)[..., 0]

        origins = c2w[None, :3, 3].expand(self.h * self.w, -1).contiguous()
        origins = origins.view(self.h, self.w, 3)
        dirs = dirs.view(self.h, self.w, 3)

        return origins, dirs

    def _post_process_batch_sample(self,
                                   sample):  # sample is an instance batch here
        caption, ins = sample[-2:]
        instance_samples = []

        for instance_idx in range(sample[0].shape[0]):
            instance_samples.append(
                self._post_process_sample(item[instance_idx]
                                          for item in sample[:-2]))

        return (*instance_samples, caption, ins)

    def _post_process_sample(self, data_sample):
        # raw_img, depth, c, bbox, caption, ins = data_sample
        # st()
        raw_img, depth, c, bbox = data_sample

        bbox = (bbox * (self.reso / 256)).astype(
            np.uint8)  # normalize bbox to the reso range

        if raw_img.shape[-2] != self.reso_encoder:
            img_to_encoder = cv2.resize(raw_img,
                                        (self.reso_encoder, self.reso_encoder),
                                        interpolation=cv2.INTER_LANCZOS4)
        else:
            img_to_encoder = raw_img

        img_to_encoder = self.normalize(img_to_encoder)
        if self.plucker_embedding:
            rays_o, rays_d = self.gen_rays(c)
            rays_plucker = torch.cat(
                [torch.cross(rays_o, rays_d, dim=-1), rays_d],
                dim=-1).permute(2, 0, 1)  # [h, w, 6] -> 6,h,w
            img_to_encoder = torch.cat([img_to_encoder, rays_plucker], 0)

        img = cv2.resize(raw_img, (self.reso, self.reso),
                         interpolation=cv2.INTER_LANCZOS4)

        img = torch.from_numpy(img).permute(2, 0, 1) / 127.5 - 1

        if self.decode_encode_img_only:
            depth_reso, fg_mask_reso = depth, depth
        else:
            depth_reso, fg_mask_reso = resize_depth_mask(depth, self.reso)

        # return {
        #     # **sample,
        #     'img_to_encoder': img_to_encoder,
        #     'img': img,
        #     'depth_mask': fg_mask_reso,
        #     # 'img_sr': img_sr,
        #     'depth': depth_reso,
        #     'c': c,
        #     'bbox': bbox,
        #     'caption': caption,
        #     'ins': ins
        #     # ! no need to load img_sr for now
        # }
        # if len(data_sample) == 4:
        return (img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox)
        # else:
        #     return (img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox, data_sample[-2], data_sample[-1])

    def canonicalize_pts(self, c, pcd, for_encoder=True, canonical_idx=0):
        # pcd: sampled in world space

        assert c.shape[0] == self.chunk_size
        assert for_encoder

        # st()

        B = c.shape[0]
        camera_poses = c[:, :16].reshape(B, 4, 4)  # 3x4

        cam_radius = np.linalg.norm(
            c[[0, self.V]][:, :16].reshape(2, 4, 4)[:, :3, 3],
            axis=-1,
            keepdims=False)  # since g-buffer adopts dynamic radius here.
        frame1_fixed_pos = np.repeat(np.eye(4)[None], 2, axis=0)
        frame1_fixed_pos[:, 2, -1] = -cam_radius

        transform = frame1_fixed_pos @ np.linalg.inv(camera_poses[[0, self.V
                                                                   ]])  # B 4 4
        transform = np.expand_dims(transform, axis=1)  # B 1 4 4
        # from LGM, https://github.com/3DTopia/LGM/blob/fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c/core/provider_objaverse.py#L127
        # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(c[[0,4]])

        repeated_homo_pcd = np.repeat(np.concatenate(
            [pcd, np.ones_like(pcd[..., 0:1])], -1)[None],
                                      2,
                                      axis=0)[..., None]  # B N 4 1
        new_pcd = (transform @ repeated_homo_pcd)[..., :3, 0]  # 2 N 3

        return new_pcd

    def canonicalize_pts_v6(self, c, pcd, for_encoder=True, canonical_idx=0):
        exit()  # deprecated function
        # pcd: sampled in world space

        assert c.shape[0] == self.chunk_size
        assert for_encoder
        encoder_canonical_idx = [0, 6, 12, 18]

        B = c.shape[0]
        camera_poses = c[:, :16].reshape(B, 4, 4)  # 3x4

        cam_radius = np.linalg.norm(
            c[encoder_canonical_idx][:, :16].reshape(4, 4, 4)[:, :3, 3],
            axis=-1,
            keepdims=False)  # since g-buffer adopts dynamic radius here.
        frame1_fixed_pos = np.repeat(np.eye(4)[None], 4, axis=0)
        frame1_fixed_pos[:, 2, -1] = -cam_radius

        transform = frame1_fixed_pos @ np.linalg.inv(
            camera_poses[encoder_canonical_idx])  # B 4 4
        transform = np.expand_dims(transform, axis=1)  # B 1 4 4
        # from LGM, https://github.com/3DTopia/LGM/blob/fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c/core/provider_objaverse.py#L127
        # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(c[[0,4]])

        repeated_homo_pcd = np.repeat(np.concatenate(
            [pcd, np.ones_like(pcd[..., 0:1])], -1)[None],
                                      4,
                                      axis=0)[..., None]  # B N 4 1
        new_pcd = (transform @ repeated_homo_pcd)[..., :3, 0]  # 2 N 3

        return new_pcd

    def normalize_camera(self, c, for_encoder=True, canonical_idx=0):
        assert c.shape[0] == self.chunk_size  # 8 o r10

        B = c.shape[0]
        camera_poses = c[:, :16].reshape(B, 4, 4)  # 3x4

        if for_encoder:
            encoder_canonical_idx = [0, self.V]
            # st()
            cam_radius = np.linalg.norm(
                c[encoder_canonical_idx][:, :16].reshape(2, 4, 4)[:, :3, 3],
                axis=-1,
                keepdims=False)  # since g-buffer adopts dynamic radius here.
            frame1_fixed_pos = np.repeat(np.eye(4)[None], 2, axis=0)
            frame1_fixed_pos[:, 2, -1] = -cam_radius

            transform = frame1_fixed_pos @ np.linalg.inv(
                camera_poses[encoder_canonical_idx])
            # from LGM, https://github.com/3DTopia/LGM/blob/fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c/core/provider_objaverse.py#L127
            # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(c[[0,4]])

            new_camera_poses = np.repeat(
                transform, self.V, axis=0
            ) @ camera_poses  # [V, 4, 4]. np.repeat() is th.repeat_interleave()

        else:
            cam_radius = np.linalg.norm(
                c[canonical_idx][:16].reshape(4, 4)[:3, 3],
                axis=-1,
                keepdims=False)  # since g-buffer adopts dynamic radius here.
            frame1_fixed_pos = np.eye(4)
            frame1_fixed_pos[2, -1] = -cam_radius

            transform = frame1_fixed_pos @ np.linalg.inv(
                camera_poses[canonical_idx])  # 4,4
            # from LGM, https://github.com/3DTopia/LGM/blob/fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c/core/provider_objaverse.py#L127
            # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(c[[0,4]])

            new_camera_poses = np.repeat(transform[None],
                                         self.chunk_size,
                                         axis=0) @ camera_poses  # [V, 4, 4]

        c = np.concatenate([new_camera_poses.reshape(B, 16), c[:, 16:]],
                           axis=-1)

        return c

    def normalize_camera_v6(self, c, for_encoder=True, canonical_idx=0):

        B = c.shape[0]
        camera_poses = c[:, :16].reshape(B, 4, 4)  # 3x4

        if for_encoder:
            assert c.shape[0] == 24
            encoder_canonical_idx = [0, 6, 12, 18]
            cam_radius = np.linalg.norm(
                c[encoder_canonical_idx][:, :16].reshape(4, 4, 4)[:, :3, 3],
                axis=-1,
                keepdims=False)  # since g-buffer adopts dynamic radius here.
            frame1_fixed_pos = np.repeat(np.eye(4)[None], 4, axis=0)
            frame1_fixed_pos[:, 2, -1] = -cam_radius

            transform = frame1_fixed_pos @ np.linalg.inv(
                camera_poses[encoder_canonical_idx])
            # from LGM, https://github.com/3DTopia/LGM/blob/fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c/core/provider_objaverse.py#L127
            # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(c[[0,4]])

            new_camera_poses = np.repeat(transform, 6,
                                         axis=0) @ camera_poses  # [V, 4, 4]

        else:
            assert c.shape[0] == 12
            cam_radius = np.linalg.norm(
                c[canonical_idx][:16].reshape(4, 4)[:3, 3],
                axis=-1,
                keepdims=False)  # since g-buffer adopts dynamic radius here.
            frame1_fixed_pos = np.eye(4)
            frame1_fixed_pos[2, -1] = -cam_radius

            transform = frame1_fixed_pos @ np.linalg.inv(
                camera_poses[canonical_idx])  # 4,4
            # from LGM, https://github.com/3DTopia/LGM/blob/fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c/core/provider_objaverse.py#L127
            # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(c[[0,4]])

            new_camera_poses = np.repeat(transform[None], 12,
                                         axis=0) @ camera_poses  # [V, 4, 4]

        c = np.concatenate([new_camera_poses.reshape(B, 16), c[:, 16:]],
                           axis=-1)

        return c

    def get_plucker_ray(self, c):
        rays_plucker = []
        for idx in range(c.shape[0]):
            rays_o, rays_d = self.gen_rays(c[idx])
            rays_plucker.append(
                torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d],
                          dim=-1).permute(2, 0, 1))  # [h, w, 6] -> 6,h,w
        rays_plucker = torch.stack(rays_plucker, 0)
        return rays_plucker

    def _unproj_depth_given_c(self, c, depth):
        # get xyz hxw for each pixel, like MCC
        # img_size = self.reso
        img_size = depth.shape[-1]

        B = c.shape[0]

        cam2world_matrix = c[:, :16].reshape(B, 4, 4)
        intrinsics = c[:, 16:25].reshape(B, 3, 3)

        ray_origins, ray_directions = self.ray_sampler(  # shape: 
            cam2world_matrix, intrinsics, img_size)[:2]

        depth = depth.reshape(B, -1).unsqueeze(-1)

        xyz = ray_origins + depth * ray_directions  # BV HW 3, already in the world space
        xyz = xyz.reshape(B, img_size, img_size, 3).permute(0, 3, 1,
                                                            2)  # B 3 H W
        xyz = xyz.clip(
            -0.45, 0.45)  # g-buffer saves depth with anti-alias = True .....
        xyz = torch.where(xyz.abs() == 0.45, 0, xyz)  # no boundary here? Yes.

        return xyz

    def _post_process_sample_batch(self, data_sample):
        # raw_img, depth, c, bbox, caption, ins = data_sample

        alpha = None
        if len(data_sample) == 4:
            raw_img, depth, c, bbox = data_sample
        else:
            raw_img, depth, c, alpha, bbox = data_sample  # put c to position 2

        if isinstance(depth, tuple):
            self.append_normal = True
            depth, normal = depth
        else:
            self.append_normal = False
            normal = None

        # if raw_img.shape[-1] == 4:
        #     depth_reso, _ = resize_depth_mask_Tensor(
        #         torch.from_numpy(depth), self.reso)
        #     raw_img, fg_mask_reso = raw_img[..., :3], raw_img[..., -1]
        #     # st() # ! check has 1 dim in alpha?
        # else:
        if not isinstance(depth, torch.Tensor):
            depth = torch.from_numpy(depth).float()
        else:
            depth = depth.float()

        depth_reso, fg_mask_reso = resize_depth_mask_Tensor(depth, self.reso)

        if alpha is None:
            alpha = fg_mask_reso
        else:
            # ! resize first
            # st()
            alpha = torch.from_numpy(alpha / 255.0).float()
            if alpha.shape[-1] != self.reso:  # bilinear inteprolate reshape
                alpha = torch.nn.functional.interpolate(
                    input=alpha.unsqueeze(1),
                    size=(self.reso, self.reso),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1)

        if self.reso < 256:
            bbox = (bbox * (self.reso / 256)).astype(
                np.uint8)  # normalize bbox to the reso range
        else:  # 3dgs
            bbox = bbox.astype(np.uint8)

        # st() # ! shall compat with 320 input

        # assert raw_img.shape[-2] == self.reso_encoder

        # img_to_encoder = cv2.resize(
        #     raw_img, (self.reso_encoder, self.reso_encoder),
        #     interpolation=cv2.INTER_LANCZOS4)
        # else:
        # img_to_encoder = raw_img

        raw_img = torch.from_numpy(raw_img).permute(0, 3, 1,
                                                    2) / 255.0  # [0,1]

        if normal is not None:
            normal = torch.from_numpy(normal).permute(0,3,1,2)

        # if raw_img.shape[-1] != self.reso:


        if raw_img.shape[1] != self.reso_encoder:
            img_to_encoder = torch.nn.functional.interpolate(
                input=raw_img,
                size=(self.reso_encoder, self.reso_encoder),
                mode='bilinear',
                align_corners=False,)
            img_to_encoder = self.normalize(img_to_encoder)

            if normal is not None:
                normal_for_encoder = torch.nn.functional.interpolate(
                    input=normal,
                    size=(self.reso_encoder, self.reso_encoder),
                    # mode='bilinear',
                    mode='nearest',
                    # align_corners=False,
                )

        else:
            img_to_encoder = self.normalize(raw_img)
            normal_for_encoder = normal

        if raw_img.shape[-1] != self.reso:
            img = torch.nn.functional.interpolate(
                input=raw_img,
                size=(self.reso, self.reso),
                mode='bilinear',
                align_corners=False,
            )  # [-1,1] range
            img = img * 2 - 1  # as gt

            if normal is not None:
                normal = torch.nn.functional.interpolate(
                    input=normal,
                    size=(self.reso, self.reso),
                    # mode='bilinear',
                    mode='nearest',
                    # align_corners=False,
                )

        else:
            img = raw_img * 2 - 1
        

        # fg_mask_reso = depth[..., -1:] # ! use

        pad_v6_fn = lambda x: torch.concat([x, x[:4]], 0) if isinstance(
            x, torch.Tensor) else np.concatenate([x, x[:4]], 0)

        # ! processing encoder input image.

        # ! normalize camera feats
        if self.frame_0_as_canonical:  # 4 views as input per batch

            # if self.chunk_size in [8, 10]:
            if True:
                # encoder_canonical_idx = [0, 4]
                # encoder_canonical_idx = [0, self.chunk_size//2]
                encoder_canonical_idx = [0, self.V]

                c_for_encoder = self.normalize_camera(c, for_encoder=True)
                c_for_render = self.normalize_camera(
                    c,
                    for_encoder=False,
                    canonical_idx=encoder_canonical_idx[0]
                )  # allocated to nv_c, frame0 (in 8 views) as the canonical
                c_for_render_nv = self.normalize_camera(
                    c,
                    for_encoder=False,
                    canonical_idx=encoder_canonical_idx[1]
                )  # allocated to nv_c, frame0 (in 8 views) as the canonical
                c_for_render = np.concatenate([c_for_render, c_for_render_nv],
                                              axis=-1)  # for compat
                # st()

            else:
                assert self.chunk_size == 20
                c_for_encoder = self.normalize_camera_v6(c,
                                                         for_encoder=True)  #

                paired_c_0 = np.concatenate([c[0:6], c[12:18]])
                paired_c_1 = np.concatenate([c[6:12], c[18:24]])

                def process_paired_camera(paired_c):
                    c_for_render = self.normalize_camera_v6(
                        paired_c, for_encoder=False, canonical_idx=0
                    )  # allocated to nv_c, frame0 (in 8 views) as the canonical
                    c_for_render_nv = self.normalize_camera_v6(
                        paired_c, for_encoder=False, canonical_idx=6
                    )  # allocated to nv_c, frame0 (in 8 views) as the canonical

                    c_for_render = np.concatenate(
                        [c_for_render, c_for_render_nv], axis=-1)  # for compat

                    return c_for_render

                paired_c_for_render_0 = process_paired_camera(paired_c_0)
                paired_c_for_render_1 = process_paired_camera(paired_c_1)

                c_for_render = np.empty(shape=(24, 50))
                c_for_render[list(range(6)) +
                             list(range(12, 18))] = paired_c_for_render_0
                c_for_render[list(range(6, 12)) +
                             list(range(18, 24))] = paired_c_for_render_1

        else:  # use g-buffer canonical c
            c_for_encoder, c_for_render = c, c

        if self.append_normal and normal is not None:
            img_to_encoder = torch.cat([img_to_encoder, normal_for_encoder],
            # img_to_encoder = torch.cat([img_to_encoder, normal],
                                       1)  # concat in C dim

        if self.plucker_embedding:
            # rays_plucker = self.get_plucker_ray(c)
            rays_plucker = self.get_plucker_ray(c_for_encoder)
            img_to_encoder = torch.cat([img_to_encoder, rays_plucker],
                                       1)  # concat in C dim

        # torchvision.utils.save_image(raw_img, 'tmp/inp.png', normalize=True, value_range=(0,1), nrow=1, padding=0)
        # torchvision.utils.save_image(rays_plucker[:,:3], 'tmp/plucker.png', normalize=True, value_range=(-1,1), nrow=1, padding=0)
        # torchvision.utils.save_image(depth_reso.unsqueeze(1), 'tmp/depth.png', normalize=True, nrow=1, padding=0)

        c = torch.from_numpy(c_for_render).to(torch.float32)

        if self.append_depth:
            normalized_depth = torch.from_numpy(depth_reso).clone().unsqueeze(
                1)  # min=0
            # normalized_depth -= torch.min(normalized_depth) # always 0 here
            # normalized_depth /= torch.max(normalized_depth)
            # normalized_depth = normalized_depth.unsqueeze(1) * 2 - 1 # normalize to [-1,1]
            # st()
            img_to_encoder = torch.cat([img_to_encoder, normalized_depth],
                                       1)  # concat in C dim
        elif self.append_xyz:
            depth_for_unproj = depth.clone()
            depth_for_unproj[depth_for_unproj ==
                  0] = 1e10  # so that rays_o will not appear in the final pcd.
            xyz = self._unproj_depth_given_c(c.float(), depth)
            # pcu.save_mesh_v(f'unproj_xyz_before_Nearest.ply', xyz[0:9].float().detach().permute(0,2,3,1).reshape(-1,3).cpu().numpy(),)

            if xyz.shape[-1] != self.reso_encoder:
                xyz = torch.nn.functional.interpolate(
                    input=xyz,  # [-1,1]
                    # size=(self.reso, self.reso),
                    size=(self.reso_encoder, self.reso_encoder),
                    mode='nearest',
                )

            # pcu.save_mesh_v(f'unproj_xyz_afterNearest.ply', xyz[0:9].float().detach().permute(0,2,3,1).reshape(-1,3).cpu().numpy(),)
            # st()
            img_to_encoder = torch.cat([img_to_encoder, xyz], 1)

        return (img_to_encoder, img, alpha, depth_reso, c,
                torch.from_numpy(bbox))

    def rand_sample_idx(self):
        return random.randint(0, self.instance_data_length - 1)

    def rand_pair(self):
        return (self.rand_sample_idx() for _ in range(2))

    def paired_post_process(self, sample):
        # repeat n times?
        all_inp_list = []
        all_nv_list = []
        caption, ins = sample[-2:]
        # expanded_return = []
        for _ in range(self.pair_per_instance):
            cano_idx, nv_idx = self.rand_pair()
            cano_sample = self._post_process_sample(item[cano_idx]
                                                    for item in sample[:-2])
            nv_sample = self._post_process_sample(item[nv_idx]
                                                  for item in sample[:-2])
            all_inp_list.extend(cano_sample)
            all_nv_list.extend(nv_sample)
        return (*all_inp_list, *all_nv_list, caption, ins)
        # return [cano_sample, nv_sample, caption, ins]
        # return (*cano_sample, *nv_sample, caption, ins)

    def get_source_cw2wT(self, source_cameras_view_to_world):
        return matrix_to_quaternion(
            source_cameras_view_to_world[:3, :3].transpose(0, 1))

    def c_to_3dgs_format(self, pose):
        # TODO, switch to torch version (batched later)

        c2w = pose[:16].reshape(4, 4)  # 3x4

        # ! load cam
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        fx = pose[16]
        FovX = focal2fov(fx, 1)
        FovY = focal2fov(fx, 1)

        tanfovx = math.tan(FovX * 0.5)
        tanfovy = math.tan(FovY * 0.5)

        assert tanfovx == tanfovy

        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0

        view_world_transform = torch.tensor(getView2World(R, T, trans,
                                                          scale)).transpose(
                                                              0, 1)

        world_view_transform = torch.tensor(getWorld2View2(R, T, trans,
                                                           scale)).transpose(
                                                               0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear,
                                                zfar=self.zfar,
                                                fovX=FovX,
                                                fovY=FovY).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
            projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        # item.update(viewpoint_cam=[viewpoint_cam])
        c = {}
        #
        c["source_cv2wT_quat"] = self.get_source_cw2wT(view_world_transform)
        c.update(
            # projection_matrix=projection_matrix, # K
            cam_view=world_view_transform,  # world_view_transform
            cam_view_proj=full_proj_transform,  # full_proj_transform
            cam_pos=camera_center,
            tanfov=tanfovx,  # TODO, fix in the renderer
            orig_pose=torch.from_numpy(pose),
            orig_c2w=torch.from_numpy(c2w),
            orig_w2c=torch.from_numpy(w2c),
            orig_intrin=torch.from_numpy(pose[16:]).reshape(3,3),
            # tanfovy=tanfovy,
        )

        return c  # dict for gs rendering

    def paired_post_process_chunk(self, sample):
        # st()

        # sample_npz, ins, caption = sample_pyd # three items
        # sample = *(sample[0][k] for k in ['raw_img', 'depth', 'c', 'bbox']), sample[-1], sample[-2]

        # repeat n times?
        all_inp_list = []
        all_nv_list = []
        auxiliary_sample = list(sample[-2:])
        # caption, ins = sample[-2:]
        ins = sample[-1]

        assert sample[0].shape[0] == self.chunk_size  # random chunks
        # expanded_return = []

        if self.load_pcd:
            # fps_pcd = pcu.load_mesh_v(
            #     # str(self.pcd_path / ins / 'fps-24576.ply'))  # N, 3
            #     str(self.pcd_path / ins / 'fps-4096.ply'))  # N, 3
            # #   'fps-4096.ply'))  # N, 3
            fps_pcd = trimesh.load(str(self.pcd_path / ins / 'fps-4096.ply')).vertices

            auxiliary_sample += [fps_pcd]

        assert self.duplicate_sample
        # st()
        if self.duplicate_sample:
            # ! shuffle before process, since frame_0_as_canonical fixed c.

            if self.chunk_size in [20, 18, 16, 12]:
                shuffle_sample = sample[:-2]  # no order shuffle required
            else:
                shuffle_sample = []
                # indices = torch.randperm(self.chunk_size)
                indices = np.random.permutation(self.chunk_size)
                for _, item in enumerate(sample[:-2]):
                    shuffle_sample.append(item[indices])  # random shuffle

            processed_sample = self._post_process_sample_batch(shuffle_sample)

            # ! process pcd if frmae_0 alignment

            if self.load_pcd:
                if self.frame_0_as_canonical:
                    # ! normalize camera feats

                    # normalized camera feats as in paper (transform the first pose to a fixed position)
                    # if self.chunk_size == 20:
                    #     auxiliary_sample[-1] = self.canonicalize_pts_v6(
                    #         c=shuffle_sample[2],
                    #         pcd=auxiliary_sample[-1],
                    #         for_encoder=True)  # B N 3
                    # else:
                    auxiliary_sample[-1] = self.canonicalize_pts(
                        c=shuffle_sample[2],
                        pcd=auxiliary_sample[-1],
                        for_encoder=True)  # B N 3
                else:
                    auxiliary_sample[-1] = np.repeat(
                        auxiliary_sample[-1][None], 2,
                        axis=0)  # share the same camera syste, just repeat

            assert not self.orthog_duplicate

            # if self.chunk_size == 8:
            all_inp_list.extend(item[:self.V] for item in processed_sample)
            all_nv_list.extend(item[self.V:] for item in processed_sample)

            # elif self.chunk_size == 20:  # V=6
            #     # indices_v6 = [np.random.permutation(self.chunk_size)[:12] for _ in range(2)] # random sample 6 views from chunks
            #     all_inp_list.extend(item[:12] for item in processed_sample)
            #     # indices_v6 = np.concatenate([np.arange(12, 20), np.arange(0,4)])
            #     all_nv_list.extend(
            #         item[12:] for item in
            #         processed_sample)  # already repeated inside batch fn
            # else:
            #     raise NotImplementedError(self.chunk_size)

            # else:
            #     all_inp_list.extend(item[:8] for item in processed_sample)
            #     all_nv_list.extend(item[8:] for item in processed_sample)

            # st()

            return (*all_inp_list, *all_nv_list, *auxiliary_sample)

        else:
            processed_sample = self._post_process_sample_batch(  # avoid shuffle shorten processing time
                item[:4] for item in sample[:-2])

            all_inp_list.extend(item for item in processed_sample)
            all_nv_list.extend(item
                               for item in processed_sample)  # ! placeholder

        # return (*all_inp_list, *all_nv_list, caption, ins)
        return (*all_inp_list, *all_nv_list, *auxiliary_sample)

        # randomly shuffle 8 views, avoid overfitting

    def single_sample_create_dict_noBatch(self, sample, prefix=''):
        # if len(sample) == 1:
        #     sample = sample[0]
        # assert len(sample) == 6
        img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox = sample

        if self.gs_cam_format:
            # TODO, can optimize later after model converges
            B, V, _ = c.shape  # B 4 25
            c = rearrange(c, 'B V C -> (B V) C').cpu().numpy()
            # c = c.cpu().numpy()
            all_gs_c = [self.c_to_3dgs_format(pose) for pose in c]
            # st()
            # all_gs_c = self.c_to_3dgs_format(c.cpu().numpy())
            c = {
                k:
                rearrange(torch.stack([gs_c[k] for gs_c in all_gs_c]),
                          '(B V) ... -> B V ...',
                          B=B,
                          V=V)
                # torch.stack([gs_c[k] for gs_c in all_gs_c])
                if isinstance(all_gs_c[0][k], torch.Tensor) else all_gs_c[0][k]
                for k in all_gs_c[0].keys()
            }
            # c = collate_gs_c

        return {
            # **sample,
            f'{prefix}img_to_encoder': img_to_encoder,
            f'{prefix}img': img,
            f'{prefix}depth_mask': fg_mask_reso,
            f'{prefix}depth': depth_reso,
            f'{prefix}c': c,
            f'{prefix}bbox': bbox,
        }

    def single_sample_create_dict(self, sample, prefix=''):
        # if len(sample) == 1:
        #     sample = sample[0]
        # assert len(sample) == 6
        img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox = sample

        if self.gs_cam_format:
            # TODO, can optimize later after model converges
            B, V, _ = c.shape  # B 4 25
            c = rearrange(c, 'B V C -> (B V) C').cpu().numpy()
            all_gs_c = [self.c_to_3dgs_format(pose) for pose in c]
            c = {
                k:
                rearrange(torch.stack([gs_c[k] for gs_c in all_gs_c]),
                          '(B V) ... -> B V ...',
                          B=B,
                          V=V)
                if isinstance(all_gs_c[0][k], torch.Tensor) else all_gs_c[0][k]
                for k in all_gs_c[0].keys()
            }
            # c = collate_gs_c

        return {
            # **sample,
            f'{prefix}img_to_encoder': img_to_encoder,
            f'{prefix}img': img,
            f'{prefix}depth_mask': fg_mask_reso,
            f'{prefix}depth': depth_reso,
            f'{prefix}c': c,
            f'{prefix}bbox': bbox,
        }

    def single_instance_sample_create_dict(self, sample, prfix=''):
        assert len(sample) == 42

        inp_sample_list = [[] for _ in range(6)]

        for item in sample[:40]:
            for item_idx in range(6):
                inp_sample_list[item_idx].append(item[0][item_idx])

        inp_sample = self.single_sample_create_dict(
            (torch.stack(item_list) for item_list in inp_sample_list),
            prefix='')

        return {
            **inp_sample,  # 
            'caption': sample[-2],
            'ins': sample[-1]
        }

    def decode_gzip(self, sample_pyd, shape=(256, 256)):
        # sample_npz, ins, caption = sample_pyd # three items
        # c, bbox, depth, ins, caption, raw_img = sample_pyd[:5], sample_pyd[5:]

        # wds.to_tuple('raw_img.jpeg', 'depth.jpeg',
        # 'd_near.npy',
        # 'd_far.npy',
        # "c.npy", 'bbox.npy', 'ins.txt', 'caption.txt'),

        # raw_img, depth, alpha_mask, d_near, d_far, c, bbox, ins, caption = sample_pyd
        raw_img, depth_alpha, = sample_pyd
        # return raw_img, depth_alpha
        # raw_img, caption = sample_pyd
        # return raw_img, caption
        # st()
        raw_img = rearrange(raw_img, 'h (b w) c -> b h w c', b=self.chunk_size)

        depth = rearrange(depth, 'h (b w) c -> b h w c', b=self.chunk_size)

        alpha_mask = rearrange(
            alpha_mask, 'h (b w) c -> b h w c', b=self.chunk_size) / 255.0

        d_far = d_far.reshape(self.chunk_size, 1, 1, 1)
        d_near = d_near.reshape(self.chunk_size, 1, 1, 1)
        # d = 1 / ( (d_normalized / 255) * (far-near) + near)
        depth = 1 / ((depth / 255) * (d_far - d_near) + d_near)
        depth = depth[..., 0]  # decoded from jpeg

        # depth = decompress_array(depth['depth'], (self.chunk_size, *shape),
        #                          np.float32,
        #                          decompress=True,
        #                          decompress_fn=lz4.frame.decompress)

        # return raw_img, depth, d_near, d_far,  c, bbox, caption, ins

        raw_img = np.concatenate([raw_img, alpha_mask[..., 0:1]], -1)

        return raw_img, depth, c, bbox, caption, ins

    def decode_zip(
        self,
        sample_pyd,
    ):
        shape = (self.reso_encoder, self.reso_encoder)
        if isinstance(sample_pyd, tuple):
            sample_pyd = sample_pyd[0]
        assert isinstance(sample_pyd, dict)

        raw_img = decompress_and_open_image_gzip(
            sample_pyd['raw_img'],
            is_img=True,
            decompress=True,
            decompress_fn=lz4.frame.decompress)

        caption = sample_pyd['caption'].decode('utf-8')
        ins = sample_pyd['ins'].decode('utf-8')

        c = decompress_array(sample_pyd['c'], (
            self.chunk_size,
            25,
        ),
                             np.float32,
                             decompress=True,
                             decompress_fn=lz4.frame.decompress)

        bbox = decompress_array(
            sample_pyd['bbox'],
            (
                self.chunk_size,
                4,
            ),
            np.float32,
            # decompress=False)
            decompress=True,
            decompress_fn=lz4.frame.decompress)

        if self.decode_encode_img_only:
            depth = np.zeros(shape=(self.chunk_size,
                                    *shape))  # save loading time
        else:
            depth = decompress_array(sample_pyd['depth'],
                                     (self.chunk_size, *shape),
                                     np.float32,
                                     decompress=True,
                                     decompress_fn=lz4.frame.decompress)

        # return {'raw_img': raw_img, 'depth': depth, 'bbox': bbox, 'caption': caption, 'ins': ins, 'c': c}
        # return raw_img, depth, c, bbox, caption, ins
        # return raw_img, bbox, caption, ins
        # return bbox, caption, ins
        return raw_img, depth, c, bbox, caption, ins
        # ! run single-instance pipeline first
        # return raw_img[0], depth[0], c[0], bbox[0], caption, ins

    def create_dict_nobatch(self, sample):
        # sample = [item[0] for item in sample] # wds wrap items in []

        sample_length = 6
        # if self.load_pcd:
        #     sample_length += 1

        cano_sample_list = [[] for _ in range(sample_length)]
        nv_sample_list = [[] for _ in range(sample_length)]
        # st()
        # bs = (len(sample)-2) // 6
        for idx in range(0, self.pair_per_instance):

            cano_sample = sample[sample_length * idx:sample_length * (idx + 1)]
            nv_sample = sample[sample_length * self.pair_per_instance +
                               sample_length * idx:sample_length *
                               self.pair_per_instance + sample_length *
                               (idx + 1)]

            for item_idx in range(sample_length):
                if self.frame_0_as_canonical:
                    # ! cycle input/output view for more pairs
                    if item_idx == 4:
                        cano_sample_list[item_idx].append(
                            cano_sample[item_idx][..., :25])
                        nv_sample_list[item_idx].append(
                            nv_sample[item_idx][..., :25])

                        cano_sample_list[item_idx].append(
                            nv_sample[item_idx][..., 25:])
                        nv_sample_list[item_idx].append(
                            cano_sample[item_idx][..., 25:])

                    else:
                        cano_sample_list[item_idx].append(
                            cano_sample[item_idx])
                        nv_sample_list[item_idx].append(nv_sample[item_idx])

                        cano_sample_list[item_idx].append(nv_sample[item_idx])
                        nv_sample_list[item_idx].append(cano_sample[item_idx])

                else:
                    cano_sample_list[item_idx].append(cano_sample[item_idx])
                    nv_sample_list[item_idx].append(nv_sample[item_idx])

                    cano_sample_list[item_idx].append(nv_sample[item_idx])
                    nv_sample_list[item_idx].append(cano_sample[item_idx])

        cano_sample = self.single_sample_create_dict_noBatch(
            (torch.stack(item_list, 0) for item_list in cano_sample_list),
            prefix=''
        )  # torch.Size([5, 10, 256, 256]). Since no batch dim here for now.

        nv_sample = self.single_sample_create_dict_noBatch(
            (torch.stack(item_list, 0) for item_list in nv_sample_list),
            prefix='nv_')

        ret_dict = {
            **cano_sample,
            **nv_sample,
        }

        if not self.load_pcd:
            ret_dict.update({'caption': sample[-2], 'ins': sample[-1]})

        else:
            # if self.frame_0_as_canonical:
            #     # fps_pcd = rearrange( sample[-1], 'B V ... -> (B V) ...')  # ! wrong order.
            #     # if self.chunk_size == 8:
            #     fps_pcd = rearrange(
            #         sample[-1], 'B V ... -> (V B) ...')  # mimic torch.repeat
            #     # else:
            #     #     fps_pcd = rearrange( sample[-1], 'B V ... -> (B V) ...')  # ugly code to match the input format...
            # else:
            #     fps_pcd = sample[-1].repeat(
            #         2, 1,
            #         1)  # mimic torch.cat(), from torch.Size([3, 4096, 3])

            # ! TODO, check fps_pcd order

            ret_dict.update({
                'caption': sample[-3],
                'ins': sample[-2],
                'fps_pcd': sample[-1]
            })

        return ret_dict

    def create_dict(self, sample):
        # sample = [item[0] for item in sample] # wds wrap items in []
        # st()

        sample_length = 6
        # if self.load_pcd:
        #     sample_length += 1

        cano_sample_list = [[] for _ in range(sample_length)]
        nv_sample_list = [[] for _ in range(sample_length)]
        # st()
        # bs = (len(sample)-2) // 6
        for idx in range(0, self.pair_per_instance):

            cano_sample = sample[sample_length * idx:sample_length * (idx + 1)]
            nv_sample = sample[sample_length * self.pair_per_instance +
                               sample_length * idx:sample_length *
                               self.pair_per_instance + sample_length *
                               (idx + 1)]

            for item_idx in range(sample_length):
                if self.frame_0_as_canonical:
                    # ! cycle input/output view for more pairs
                    if item_idx == 4:
                        cano_sample_list[item_idx].append(
                            cano_sample[item_idx][..., :25])
                        nv_sample_list[item_idx].append(
                            nv_sample[item_idx][..., :25])

                        cano_sample_list[item_idx].append(
                            nv_sample[item_idx][..., 25:])
                        nv_sample_list[item_idx].append(
                            cano_sample[item_idx][..., 25:])

                    else:
                        cano_sample_list[item_idx].append(
                            cano_sample[item_idx])
                        nv_sample_list[item_idx].append(nv_sample[item_idx])

                        cano_sample_list[item_idx].append(nv_sample[item_idx])
                        nv_sample_list[item_idx].append(cano_sample[item_idx])

                else:
                    cano_sample_list[item_idx].append(cano_sample[item_idx])
                    nv_sample_list[item_idx].append(nv_sample[item_idx])

                    cano_sample_list[item_idx].append(nv_sample[item_idx])
                    nv_sample_list[item_idx].append(cano_sample[item_idx])

        # if self.split_chunk_input:
        #     cano_sample = self.single_sample_create_dict(
        #         (torch.cat(item_list, 0) for item_list in cano_sample_list),
        #         prefix='')
        #     nv_sample = self.single_sample_create_dict(
        #         (torch.cat(item_list, 0) for item_list in nv_sample_list),
        #         prefix='nv_')

    # else:

    # st()
        cano_sample = self.single_sample_create_dict(
            (torch.cat(item_list, 0) for item_list in cano_sample_list),
            prefix='')  # torch.Size([4, 4, 10, 256, 256])

        nv_sample = self.single_sample_create_dict(
            (torch.cat(item_list, 0) for item_list in nv_sample_list),
            prefix='nv_')

        ret_dict = {
            **cano_sample,
            **nv_sample,
        }

        if not self.load_pcd:
            ret_dict.update({'caption': sample[-2], 'ins': sample[-1]})

        else:
            if self.frame_0_as_canonical:
                # fps_pcd = rearrange( sample[-1], 'B V ... -> (B V) ...')  # ! wrong order.
                # if self.chunk_size == 8:
                fps_pcd = rearrange(
                    sample[-1], 'B V ... -> (V B) ...')  # mimic torch.repeat
                # else:
                #     fps_pcd = rearrange( sample[-1], 'B V ... -> (B V) ...')  # ugly code to match the input format...
            else:
                fps_pcd = sample[-1].repeat(
                    2, 1,
                    1)  # mimic torch.cat(), from torch.Size([3, 4096, 3])

            ret_dict.update({
                'caption': sample[-3],
                'ins': sample[-2],
                'fps_pcd': fps_pcd
            })

        return ret_dict

    def prepare_mv_input(self, sample):

        # sample = [item[0] for item in sample] # wds wrap items in []
        bs = len(sample['caption'])  # number of instances
        chunk_size = sample['img'].shape[0] // bs

        assert self.split_chunk_input

        for k, v in sample.items():
            if isinstance(v, torch.Tensor) and k != 'fps_pcd':
                sample[k] = rearrange(v, "b f c ... -> (b f) c ...",
                                      f=self.V).contiguous()

        # # ! shift nv
        # else:
        #     for k, v in sample.items():
        #         if k not in ['ins', 'caption']:

        #             rolled_idx = torch.LongTensor(
        #                 list(
        #                     itertools.chain.from_iterable(
        #                         list(range(i, sample['img'].shape[0], bs))
        #                         for i in range(bs))))

        #             v = torch.index_select(v, dim=0, index=rolled_idx)
        #         sample[k] = v

        #     # img = sample['img']
        #     # gt = sample['nv_img']
        #     # torchvision.utils.save_image(img[0], 'inp.jpg', normalize=True)
        #     # torchvision.utils.save_image(gt[0], 'nv.jpg', normalize=True)

        #     for k, v in sample.items():
        #         if 'nv' in k:
        #             rolled_idx = torch.LongTensor(
        #                 list(
        #                     itertools.chain.from_iterable(
        #                         list(
        #                             np.roll(
        #                                 np.arange(i * chunk_size, (i + 1) *
        #                                           chunk_size), 4)
        #                             for i in range(bs)))))

        #             v = torch.index_select(v, dim=0, index=rolled_idx)
        #             sample[k] = v

        # torchvision.utils.save_image(sample['nv_img'], 'nv.png', normalize=True)
        # torchvision.utils.save_image(sample['img'], 'inp.png', normalize=True)

        return sample


def load_dataset(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        #   shuffle=True,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True,
        dataset_size=-1,
        trainer_name='input_rec',
        use_lmdb=False,
        use_wds=False,
        use_chunk=False,
        use_lmdb_compressed=False,
        infi_sampler=True):
    # st()
    # dataset_cls = {
    #     'input_rec': MultiViewDataset,
    #     'nv': NovelViewDataset,
    # }[trainer_name]
    # st()
    if use_wds:
        return load_wds_data(file_path, reso, reso_encoder, batch_size,
                             num_workers)

    if use_lmdb:
        logger.log('using LMDB dataset')
        # dataset_cls = LMDBDataset_MV #  2.5-3iter/s, but unstable, drops to 1 later.

        if use_lmdb_compressed:
            if 'nv' in trainer_name:
                dataset_cls = Objv_LMDBDataset_NV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
            else:
                dataset_cls = Objv_LMDBDataset_MV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
        else:
            if 'nv' in trainer_name:
                dataset_cls = Objv_LMDBDataset_NV_NoCompressed  #  2.5-3iter/s, but unstable, drops to 1 later.
            else:
                dataset_cls = Objv_LMDBDataset_MV_NoCompressed  #  2.5-3iter/s, but unstable, drops to 1 later.

        # dataset = dataset_cls(file_path)
    elif use_chunk:
        dataset_cls = ChunkObjaverseDataset
    else:
        if 'nv' in trainer_name:
            dataset_cls = NovelViewObjverseDataset
        else:
            dataset_cls = MultiViewObjverseDataset  # 1.5-2iter/s

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          test=False,
                          preprocess=preprocess,
                          load_depth=load_depth,
                          imgnet_normalize=imgnet_normalize,
                          dataset_size=dataset_size)

    logger.log('dataset_cls: {}, dataset size: {}'.format(
        trainer_name, len(dataset)))

    if use_chunk:

        def chunk_collate_fn(sample):
            # st()
            default_collate_sample = torch.utils.data.default_collate(
                sample[0])
            st()
            return default_collate_sample

        collate_fn = chunk_collate_fn
    else:
        collate_fn = None

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        drop_last=False,
                        pin_memory=True,
                        persistent_workers=num_workers > 0,
                        shuffle=use_chunk,
                        collate_fn=collate_fn)
    return loader


def chunk_collate_fn(sample):
    sample = torch.utils.data.default_collate(sample)
    # ! change from stack to cat
    # sample = self.post_process.prepare_mv_input(sample)

    bs = len(sample['caption'])  # number of instances
    # chunk_size = sample['img'].shape[0] // bs

    def merge_internal_batch(sample, merge_b_only=False):
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                if v.ndim > 1:
                    if k == 'fps_pcd' or merge_b_only:
                        sample[k] = rearrange(
                            v,
                            "b1 b2 ... -> (b1 b2) ...").float().contiguous()

                    else:
                        sample[k] = rearrange(
                            v, "b1 b2 f c ... -> (b1 b2 f) c ...").float(
                            ).contiguous()
                elif k == 'tanfov':
                    sample[k] = v[0].float().item()  # tanfov.

    if isinstance(sample['c'], dict):  # 3dgs
        merge_internal_batch(sample['c'], merge_b_only=True)
        merge_internal_batch(sample['nv_c'], merge_b_only=True)

    merge_internal_batch(sample)

    return sample

def chunk_ddpm_collate_fn(sample):
    sample = torch.utils.data.default_collate(sample)
    # ! change from stack to cat
    # sample = self.post_process.prepare_mv_input(sample)

    # bs = len(sample['caption'])  # number of instances
    # chunk_size = sample['img'].shape[0] // bs

    def merge_internal_batch(sample, merge_b_only=False):
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                if v.ndim > 1:
                    # if k in ['c', 'latent']:
                    sample[k] = rearrange(
                        v,
                        "b1 b2 ... -> (b1 b2) ...").float().contiguous()

                    # else: # img
                    #     sample[k] = rearrange(
                    #         v, "b1 b2 f ... -> (b1 b2 f) ...").float(
                    #         ).contiguous()

            else: # caption & ins
                v = [v[i][0] for i in range(len(v))]

    merge_internal_batch(sample)

    # if 'caption' in sample:
    #     sample['caption'] = sample['caption'][0] + sample['caption'][1]

    return sample




def load_data_cls(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        #   shuffle=True,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True,
        dataset_size=-1,
        trainer_name='input_rec',
        use_lmdb=False,
        use_wds=False,
        use_chunk=False,
        use_lmdb_compressed=False,
        # plucker_embedding=False,
        # frame_0_as_canonical=False,
        infi_sampler=True,
        load_latent=False,
        return_dataset=False,
        load_caption_dataset=False,
        load_mv_dataset=False,
        **kwargs):
    # st()
    # dataset_cls = {
    #     'input_rec': MultiViewDataset,
    #     'nv': NovelViewDataset,
    # }[trainer_name]
    # st()
    # if use_lmdb:
    #     logger.log('using LMDB dataset')
    #     # dataset_cls = LMDBDataset_MV #  2.5-3iter/s, but unstable, drops to 1 later.
    #     if 'nv' in trainer_name:
    #         dataset_cls = Objv_LMDBDataset_NV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
    #     else:
    #         dataset_cls = Objv_LMDBDataset_MV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.

    #     # dataset = dataset_cls(file_path)

    collate_fn = None

    if use_lmdb:
        logger.log('using LMDB dataset')
        # dataset_cls = LMDBDataset_MV #  2.5-3iter/s, but unstable, drops to 1 later.

        if use_lmdb_compressed:
            if 'nv' in trainer_name:
                dataset_cls = Objv_LMDBDataset_NV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
            else:
                dataset_cls = Objv_LMDBDataset_MV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
        else:
            if 'nv' in trainer_name:
                dataset_cls = Objv_LMDBDataset_NV_NoCompressed  #  2.5-3iter/s, but unstable, drops to 1 later.
            else:
                dataset_cls = Objv_LMDBDataset_MV_NoCompressed  #  2.5-3iter/s, but unstable, drops to 1 later.

    elif use_chunk:
        if load_latent:

            # if 'gs_cam_format' in kwargs:
            if kwargs['gs_cam_format']:
                if load_caption_dataset:
                    dataset_cls = ChunkObjaverseDatasetDDPMgsT23D
                    collate_fn = chunk_ddpm_collate_fn
                else:
                    if load_mv_dataset:
                        # dataset_cls = ChunkObjaverseDatasetDDPMgsMV23D # ! if multi-view 
                        dataset_cls = ChunkObjaverseDatasetDDPMgsMV23DSynthetic # ! if multi-view 
                        # collate_fn = chunk_ddpm_collate_fn
                        collate_fn = None
                    else:
                        dataset_cls = ChunkObjaverseDatasetDDPMgsI23D
                        collate_fn = None
            else:
                dataset_cls = ChunkObjaverseDatasetDDPM
                collate_fn = chunk_ddpm_collate_fn
        else:
            dataset_cls = ChunkObjaverseDataset
            collate_fn = chunk_collate_fn

    else:
        if 'nv' in trainer_name:
            dataset_cls = NovelViewObjverseDataset  # 1.5-2iter/s
        else:
            dataset_cls = MultiViewObjverseDataset

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          test=False,
                          preprocess=preprocess,
                          load_depth=load_depth,
                          imgnet_normalize=imgnet_normalize,
                          dataset_size=dataset_size,
                          **kwargs
                          #   plucker_embedding=plucker_embedding
                          )

    logger.log('dataset_cls: {}, dataset size: {}'.format(
        trainer_name, len(dataset)))

    # st()
    return dataset



def load_data(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        #   shuffle=True,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True,
        dataset_size=-1,
        trainer_name='input_rec',
        use_lmdb=False,
        use_wds=False,
        use_chunk=False,
        use_lmdb_compressed=False,
        # plucker_embedding=False,
        # frame_0_as_canonical=False,
        infi_sampler=True,
        load_latent=False,
        return_dataset=False,
        load_caption_dataset=False,
        load_mv_dataset=False,
        **kwargs):
    # st()
    # dataset_cls = {
    #     'input_rec': MultiViewDataset,
    #     'nv': NovelViewDataset,
    # }[trainer_name]
    # st()
    # if use_lmdb:
    #     logger.log('using LMDB dataset')
    #     # dataset_cls = LMDBDataset_MV #  2.5-3iter/s, but unstable, drops to 1 later.
    #     if 'nv' in trainer_name:
    #         dataset_cls = Objv_LMDBDataset_NV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
    #     else:
    #         dataset_cls = Objv_LMDBDataset_MV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.

    #     # dataset = dataset_cls(file_path)

    collate_fn = None

    if use_lmdb:
        logger.log('using LMDB dataset')
        # dataset_cls = LMDBDataset_MV #  2.5-3iter/s, but unstable, drops to 1 later.

        if use_lmdb_compressed:
            if 'nv' in trainer_name:
                dataset_cls = Objv_LMDBDataset_NV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
            else:
                dataset_cls = Objv_LMDBDataset_MV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
        else:
            if 'nv' in trainer_name:
                dataset_cls = Objv_LMDBDataset_NV_NoCompressed  #  2.5-3iter/s, but unstable, drops to 1 later.
            else:
                dataset_cls = Objv_LMDBDataset_MV_NoCompressed  #  2.5-3iter/s, but unstable, drops to 1 later.

    elif use_chunk:
        # st()
        if load_latent:

            if kwargs['gs_cam_format']:
                if load_caption_dataset:
                    dataset_cls = ChunkObjaverseDatasetDDPMgsT23D
                    # collate_fn = chunk_ddpm_collate_fn
                    collate_fn = None
                else:
                    if load_mv_dataset:
                        # dataset_cls = ChunkObjaverseDatasetDDPMgsMV23D
                        dataset_cls = ChunkObjaverseDatasetDDPMgsMV23DSynthetic # ! if multi-view 
                        # collate_fn = chunk_ddpm_collate_fn
                        collate_fn = None
                    else:
                        # dataset_cls = ChunkObjaverseDatasetDDPMgsI23D # load i23d
                        # collate_fn = None
                        # load mv dataset for i23d
                        dataset_cls = ChunkObjaverseDatasetDDPMgsI23D_loadMV
                        collate_fn = chunk_ddpm_collate_fn
            else:
                dataset_cls = ChunkObjaverseDatasetDDPM
                collate_fn = chunk_ddpm_collate_fn
        else:
            dataset_cls = ChunkObjaverseDataset
            collate_fn = chunk_collate_fn

    else:
        if 'nv' in trainer_name:
            dataset_cls = NovelViewObjverseDataset  # 1.5-2iter/s
        else:
            dataset_cls = MultiViewObjverseDataset

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          test=False,
                          preprocess=preprocess,
                          load_depth=load_depth,
                          imgnet_normalize=imgnet_normalize,
                          dataset_size=dataset_size,
                          **kwargs
                          #   plucker_embedding=plucker_embedding
                          )

    logger.log('dataset_cls: {}, dataset size: {}'.format(
        trainer_name, len(dataset)))

    # st()
    if return_dataset:
        return dataset

    assert infi_sampler
    if infi_sampler:
        train_sampler = DistributedSampler(dataset=dataset,
                                           shuffle=True,
                                           drop_last=True)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            sampler=train_sampler,
            collate_fn=collate_fn,
            # prefetch_factor=3 if num_workers>0 else None,
        )

        while True:
            yield from loader

    # else:
    #     # loader = DataLoader(dataset,
    #     #                     batch_size=batch_size,
    #     #                     num_workers=num_workers,
    #     #                     drop_last=False,
    #     #                     pin_memory=True,
    #     #                     persistent_workers=num_workers > 0,
    #     #                     shuffle=False)
    #     st()
    #     return dataset


def load_eval_data(
    file_path="",
    reso=64,
    reso_encoder=224,
    batch_size=1,
    num_workers=1,
    load_depth=False,
    preprocess=None,
    imgnet_normalize=True,
    interval=1,
    use_lmdb=False,
    plucker_embedding=False,
    load_real=False,
    load_mv_real=False,
    load_gso=False,
    four_view_for_latent=False,
    shuffle_across_cls=False,
    load_extra_36_view=False,
    gs_cam_format=False,
    single_view_for_i23d=False,
    use_chunk=False,
    **kwargs,
):
    collate_fn = None

    if use_lmdb:
        logger.log('using LMDB dataset')
        dataset_cls = Objv_LMDBDataset_MV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
        dataset = dataset_cls(file_path,
                              reso,
                              reso_encoder,
                              test=True,
                              preprocess=preprocess,
                              load_depth=load_depth,
                              imgnet_normalize=imgnet_normalize,
                              interval=interval)
    elif use_chunk:
        dataset = ChunkObjaverseDataset(
            file_path,
            reso,
            reso_encoder,
            test=False,
            preprocess=preprocess,
            load_depth=load_depth,
            imgnet_normalize=imgnet_normalize,
            #   dataset_size=dataset_size,
            plucker_embedding=plucker_embedding,
            wds_split_all=2,
            #   frame_0_as_canonical=frame_0_as_canonical,
            **kwargs)
        collate_fn = chunk_collate_fn

    elif load_real:
        if load_mv_real:
            dataset_cls = RealMVDataset
        elif load_gso:
            # st()
            dataset_cls = RealDataset_GSO
        else: # single-view i23d
            dataset_cls = RealDataset
         
        dataset = dataset_cls(file_path,
                              reso,
                              reso_encoder,
                              preprocess=preprocess,
                              load_depth=load_depth,
                              test=True,
                              imgnet_normalize=imgnet_normalize,
                              interval=interval,
                              plucker_embedding=plucker_embedding)

    else:
        dataset = MultiViewObjverseDataset(
            file_path,
            reso,
            reso_encoder,
            preprocess=preprocess,
            load_depth=load_depth,
            test=True,
            imgnet_normalize=imgnet_normalize,
            interval=interval,
            plucker_embedding=plucker_embedding,
            four_view_for_latent=four_view_for_latent,
            load_extra_36_view=load_extra_36_view,
            shuffle_across_cls=shuffle_across_cls,
            gs_cam_format=gs_cam_format,
            single_view_for_i23d=single_view_for_i23d,
            **kwargs)

    print('eval dataset size: {}'.format(len(dataset)))
    # train_sampler = DistributedSampler(dataset=dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )
    # sampler=train_sampler)
    # return loader
    return iter(loader)


def load_data_for_lmdb(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        #   shuffle=True,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True,
        dataset_size=-1,
        trainer_name='input_rec',
        shuffle_across_cls=False,
        four_view_for_latent=False,
        wds_split=1):
    # st()
    # dataset_cls = {
    #     'input_rec': MultiViewDataset,
    #     'nv': NovelViewDataset,
    # }[trainer_name]
    # if 'nv' in trainer_name:
    #     dataset_cls = NovelViewDataset
    # else:
    # dataset_cls = MultiViewDataset
    # st()
    # dataset_cls = MultiViewObjverseDatasetforLMDB
    dataset_cls = MultiViewObjverseDatasetforLMDB_nocaption

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          test=False,
                          preprocess=preprocess,
                          load_depth=load_depth,
                          imgnet_normalize=imgnet_normalize,
                          dataset_size=dataset_size,
                          shuffle_across_cls=shuffle_across_cls,
                          wds_split=wds_split,
                          four_view_for_latent=four_view_for_latent)

    logger.log('dataset_cls: {}, dataset size: {}'.format(
        trainer_name, len(dataset)))
    # train_sampler = DistributedSampler(dataset=dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        # prefetch_factor=2,
        # prefetch_factor=3,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    # sampler=train_sampler)

    # while True:
    #     yield from loader
    return loader, dataset.dataset_name, len(dataset)


def load_lmdb_for_lmdb(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        #   shuffle=True,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True,
        dataset_size=-1,
        trainer_name='input_rec'):
    # st()
    # dataset_cls = {
    #     'input_rec': MultiViewDataset,
    #     'nv': NovelViewDataset,
    # }[trainer_name]
    # if 'nv' in trainer_name:
    #     dataset_cls = NovelViewDataset
    # else:
    # dataset_cls = MultiViewDataset
    # st()
    dataset_cls = Objv_LMDBDataset_MV_Compressed_for_lmdb

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          test=False,
                          preprocess=preprocess,
                          load_depth=load_depth,
                          imgnet_normalize=imgnet_normalize,
                          dataset_size=dataset_size)

    logger.log('dataset_cls: {}, dataset size: {}'.format(
        trainer_name, len(dataset)))
    # train_sampler = DistributedSampler(dataset=dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        # prefetch_factor=2,
        # prefetch_factor=3,
        pin_memory=True,
        persistent_workers=True,
    )
    # sampler=train_sampler)

    # while True:
    #     yield from loader
    return loader, len(dataset)


def load_memory_data(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        num_workers=1,
        #  load_depth=True,
        preprocess=None,
        imgnet_normalize=True,
        use_chunk=True,
        **kwargs):
    # load a single-instance into the memory to speed up training IO
    # dataset = MultiViewObjverseDataset(file_path,

    collate_fn = None

    if use_chunk:
        dataset_cls = ChunkObjaverseDataset
        collate_fn = chunk_collate_fn
    else:
        dataset_cls = NovelViewObjverseDataset

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          preprocess=preprocess,
                          load_depth=True,
                          test=False,
                          overfitting=True,
                          imgnet_normalize=imgnet_normalize,
                          overfitting_bs=batch_size,
                          **kwargs)
    logger.log('!!!!!!! memory dataset size: {} !!!!!!'.format(len(dataset)))
    # train_sampler = DistributedSampler(dataset=dataset)
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        collate_fn = collate_fn
    )

    all_data: dict = next(
        iter(loader)
    )  # torchvision.utils.save_image(all_data['img'], 'gt.jpg', normalize=True, value_range=(-1,1))

    # st()

    if kwargs.get('gs_cam_format', False):  # gs rendering pipeline
        # ! load V=4 images for training in a batch.
        while True:
            # st()

            # indices = torch.randperm(len(dataset))[:4]
            indices = torch.randperm(
                len(dataset) * 2)[:batch_size]  # all instances
            # indices2 = torch.randperm(len(dataset))[:] # all instances

            batch_c = collections.defaultdict(dict)
            V = all_data['c']['source_cv2wT_quat'].shape[1]
            for k in ['c', 'nv_c']:
                for k_c, v_c in all_data[k].items():
                    if k_c == 'tanfov':
                        continue
                    try:
                        batch_c[k][
                            k_c] = torch.index_select(  # ! chunk data reading pipeline
                                v_c,
                                dim=0,
                                index=indices
                            ).reshape(batch_size, V, *v_c.shape[2:]).float(
                            ) if isinstance(

                                v_c,
                                torch.Tensor) else v_c  # float
                    except Exception as e:
                        st()
                        print(e)

            # ! read chunk not required, already float
            batch_c['c']['tanfov'] = all_data['c']['tanfov']
            batch_c['nv_c']['tanfov'] = all_data['nv_c']['tanfov']

            indices_range = torch.arange(indices[0]*V, (indices[0]+1)*V)
            batch_data = {}
            for k, v in all_data.items():
                if k not in ['c', 'nv_c']:
                    try:
                        if k == 'fps_pcd':
                            batch_data[k] = torch.index_select(
                                v, dim=0, index=indices).float() if isinstance(
                                    v, torch.Tensor) else v  # float
                        else:
                            batch_data[k] = torch.index_select(
                                v, dim=0, index=indices_range).float() if isinstance(
                                    v, torch.Tensor) else v  # float
                    except:
                        st()
                        print(e)

            memory_batch_data = {
                **batch_data,
                **batch_c,
            }


            yield memory_batch_data

    else:
        while True:
            start_idx = np.random.randint(0, len(dataset) - batch_size + 1)
            yield {
                k: v[start_idx:start_idx + batch_size]
                for k, v in all_data.items()
            }


def read_dnormal(normald_path, cond_pos, h=None, w=None):
    cond_cam_dis = np.linalg.norm(cond_pos, 2)

    near = 0.867  #sqrt(3) * 0.5
    near_distance = cond_cam_dis - near

    normald = cv2.imread(normald_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    normal, depth = normald[..., :3], normald[..., 3:]

    depth[depth < near_distance] = 0

    if h is not None:
        assert w is not None
        if depth.shape[1] != h:
            depth = cv2.resize(depth, (h, w), interpolation=cv2.INTER_NEAREST
                               )  # 512,512, 1 -> self.reso, self.reso
            # depth = cv2.resize(depth, (h, w), interpolation=cv2.INTER_LANCZOS4
            #                    )  # ! may fail if nearest. dirty data.
            # st()
        else:
            depth = depth[..., 0]

        if normal.shape[1] != h:
            normal = cv2.resize(normal, (h, w),
                                interpolation=cv2.INTER_NEAREST
                                )  # 512,512, 1 -> self.reso, self.reso

    else:
        depth = depth[..., 0]

    return torch.from_numpy(depth).float(), torch.from_numpy(normal).float()


def get_intri(target_im=None, h=None, w=None, normalize=False):
    if target_im is None:
        assert (h is not None and w is not None)
    else:
        h, w = target_im.shape[:2]

    fx = fy = 1422.222
    res_raw = 1024
    f_x = f_y = fx * h / res_raw
    K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
    if normalize:  # center is [0.5, 0.5], eg3d renderer tradition
        K[:6] /= h
    # print("intr: ", K)
    return K


def convert_pose(C2W):
    # https://github.com/modelscope/richdreamer/blob/c3d9a77fa15fc42dbae12c2d41d64aaec14efd37/dataset/gobjaverse/depth_warp_example.py#L402
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return torch.from_numpy(C2W)


def read_camera_matrix_single(json_file):
    with open(json_file, 'r', encoding='utf8') as reader:
        json_content = json.load(reader)
    '''
    # NOTE that different from unity2blender experiments.
    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = -np.array(json_content['y'])
    camera_matrix[:3, 2] = -np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])


    '''
    camera_matrix = np.eye(4)  # blender-based
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = np.array(json_content['y'])
    camera_matrix[:3, 2] = np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])
    # print(camera_matrix)
    # '''

    # return convert_pose(camera_matrix)
    return camera_matrix


def unity2blender(normal):
    normal_clone = normal.copy()
    normal_clone[..., 0] = -normal[..., -1]
    normal_clone[..., 1] = -normal[..., 0]
    normal_clone[..., 2] = normal[..., 1]

    return normal_clone

def unity2blender_fix(normal): # up blue, left green, front (towards inside) red
    normal_clone = normal.copy()
    # normal_clone[..., 0] = -normal[..., 2]
    # normal_clone[..., 1] = -normal[..., 0]
    normal_clone[..., 0] = -normal[..., 0] # swap r and g
    normal_clone[..., 1] = -normal[..., 2]
    normal_clone[..., 2] = normal[..., 1]

    return normal_clone

def unity2blender_th(normal):
    assert normal.shape[1] == 3 # B 3 H W...
    normal_clone = normal.clone()
    normal_clone[:, 0, ...] = -normal[:, -1, ...]
    normal_clone[:, 1, ...] = -normal[:, 0, ...]
    normal_clone[:, 2, ...] =  normal[:, 1, ...]

    return normal_clone


def blender2midas(img):
    '''Blender: rub
    midas: lub
    '''
    img[..., 0] = -img[..., 0]
    img[..., 1] = -img[..., 1]
    img[..., -1] = -img[..., -1]
    return img


def current_milli_time():
    return round(time.time() * 1000)


# modified from ShapeNet class
class MultiViewObjverseDataset(Dataset):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
            four_view_for_latent=False,
            single_view_for_i23d=False,
            load_extra_36_view=False,
            gs_cam_format=False,
            frame_0_as_canonical=False,
            **kwargs):
        self.load_extra_36_view = load_extra_36_view
        # st()
        self.gs_cam_format = gs_cam_format
        self.frame_0_as_canonical = frame_0_as_canonical
        self.four_view_for_latent = four_view_for_latent  # export 0 12 30 36, 4 views for reconstruction
        self.single_view_for_i23d = single_view_for_i23d
        self.file_path = file_path
        self.overfitting = overfitting
        self.scene_scale = scene_scale
        self.reso = reso
        self.reso_encoder = reso_encoder
        self.classes = False
        self.load_depth = load_depth
        self.preprocess = preprocess
        self.plucker_embedding = plucker_embedding
        self.intrinsics = get_intri(h=self.reso, w=self.reso,
                                    normalize=True).reshape(9)

        assert not self.classes, "Not support class condition now."

        dataset_name = Path(self.file_path).stem.split('_')[0]
        self.dataset_name = dataset_name

        self.zfar = 100.0
        self.znear = 0.01

        # if test:
        #     self.ins_list = sorted(os.listdir(self.file_path))[0:1]  # the first 1 instance for evaluation reference.
        # else:
        # ! TODO, read from list?

        def load_single_cls_instances(file_path):
            ins_list = []  # the first 1 instance for evaluation reference.
            # '''
            # for dict_dir in os.listdir(file_path)[:]:
            # for dict_dir in os.listdir(file_path)[:]:
            for dict_dir in os.listdir(file_path):
                # for dict_dir in os.listdir(file_path)[:2]:
                for ins_dir in os.listdir(os.path.join(file_path, dict_dir)):

                    # self.ins_list.append(os.path.join(self.file_path, dict_dir, ins_dir,))
                    # /nas/shared/V2V/yslan/logs/nips23/Reconstruction/final/objav/vae/MV/170K/infer-latents/189w/v=6-rotate/latent_dir
                    # st() # check latent whether saved
                    # root = '/nas/shared/V2V/yslan/logs/nips23/Reconstruction/final/objav/vae/MV/170K/infer-latents/189w/v=6-rotate/latent_dir'
                    # if os.path.exists(os.path.join(root,file_path.split('/')[-1], dict_dir, ins_dir, 'latent.npy') ):
                    # continue
                    # pcd_root = '/nas/shared/V2V/yslan/logs/nips23/Reconstruction/pcd-V=8_24576_polish'
                    # pcd_root = '/nas/shared/V2V/yslan/logs/nips23/Reconstruction/pcd-V=10_4096_polish'
                    # if os.path.exists(
                    #         os.path.join(pcd_root, 'fps-pcd',
                    #                      file_path.split('/')[-1], dict_dir,
                    #                      ins_dir, 'fps-4096.ply')):
                    #     continue

                    # ! split=8 has some missing instances
                    # root = '/cpfs01/user/lanyushi.p/data/chunk-jpeg-normal/bs_16_fixsave3/170K/384/'
                    # if os.path.exists(os.path.join(root,file_path.split('/')[-1], dict_dir, ins_dir,) ):
                    #     continue
                    # else:
                    # ins_list.append(
                    #     os.path.join(file_path, dict_dir, ins_dir,
                    #                 'campos_512_v4'))

                    # filter out some data
                    if not os.path.exists(os.path.join(file_path, dict_dir, ins_dir, 'campos_512_v2')):
                        continue
                    if not os.path.exists(os.path.join(file_path, dict_dir, ins_dir, 'campos_512_v2', '00025', '00025.png')):
                        continue
                    if len(os.listdir(os.path.join(file_path, dict_dir, ins_dir, 'campos_512_v2'))) != 40:
                        continue

                    ins_list.append(
                        os.path.join(file_path, dict_dir, ins_dir,
                                    'campos_512_v2'))

            # '''
            # check pcd performnace
            # ins_list.append(
            #     os.path.join(file_path, '0', '10634',
            #                     'campos_512_v4'))
            return ins_list

        # st()
        self.ins_list = []
        # for subset in ['Animals', 'Transportations_tar', 'Furnitures']:
        # for subset in ['Furnitures']:
        # selected subset for training
        # if False:
        if True:
            for subset in [  # ! around 17W instances in total. 
                    # 'Animals',
                    # 'BuildingsOutdoor',
                    # 'daily-used',
                    # 'Furnitures',
                    # 'Food',
                    # 'Plants',
                    # 'Electronics',
                    # 'Transportations_tar',
                    # 'Human-Shape',
                    'gobjaverse_alignment_unzip',
            ]:  # selected subset for training

                # if os.path.exists(f'{self.file_path}/{subset}.txt'):
                # dataset_list = f'{self.file_path}/{subset}_filtered.txt'
                dataset_list = f'{self.file_path}/{subset}_filtered_more.txt'
                assert os.path.exists(dataset_list)
                if os.path.exists(dataset_list):
                    with open(dataset_list, 'r') as f:
                        self.ins_list += [os.path.join(self.file_path, item.strip()) for item in f.readlines()]
                else:
                    self.ins_list += load_single_cls_instances(
                        os.path.join(self.file_path, subset))

                # st()
                # current_time = int(current_milli_time()
                #                    )  # randomly shuffle given current time
                # random.seed(current_time)
                # random.shuffle(self.ins_list)

        else:  # preprocess single class
            self.ins_list = load_single_cls_instances(self.file_path)

        self.ins_list = sorted(self.ins_list)

        if overfitting:
            self.ins_list = self.ins_list[:1]

        self.rgb_list = []
        self.frame0_pose_list = []
        self.pose_list = []
        self.depth_list = []
        self.data_ins_list = []
        self.instance_data_length = -1

        # self.pcd_path = Path('/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/Reconstruction/pcd-V=6/fps-pcd')
        self.pcd_path = Path(
            '/nas/shared/V2V/yslan/logs/nips23/Reconstruction/pcd-V=6/fps-pcd')

        with open(
                '/nas/shared/public/yslan/data/text_captions_cap3d.json') as f:
                # '/nas/shared/V2V/yslan/aigc3d/text_captions_cap3d.json') as f:
            self.caption_data = json.load(f)

        self.shuffle_across_cls = shuffle_across_cls


        # for ins in self.ins_list[47000:]:
        if four_view_for_latent:  # also saving dense pcd
            # self.wds_split_all = 1  # ! when dumping latent
            # self.wds_split_all = 2  # ! when dumping latent
            # self.wds_split_all = 4
            # self.wds_split_all = 6
            # self.wds_split_all = 4
            # self.wds_split_all = 5
            # self.wds_split_all = 6
            # self.wds_split_all = 7
            # self.wds_split_all = 1
            self.wds_split_all = 8
            # self.wds_split_all = 2
            # ins_list_to_process = self.ins_list
            all_ins_size = len(self.ins_list)
            ratio_size = all_ins_size // self.wds_split_all + 1
            # ratio_size = int(all_ins_size / self.wds_split_all) + 1

            ins_list_to_process = self.ins_list[ratio_size *
                                                (wds_split):ratio_size *
                                                (wds_split + 1)]

        else:  # ! create shards dataset
            # self.wds_split_all = 4
            self.wds_split_all = 8
            # self.wds_split_all = 1
            all_ins_size = len(self.ins_list)

            random.seed(0)
            random.shuffle(self.ins_list) # avoid same category appears in the same shard

            ratio_size = all_ins_size // self.wds_split_all + 1

            ins_list_to_process = self.ins_list[ratio_size * # 1 - 8
                                                (wds_split - 1):ratio_size *
                                                wds_split]

        # uniform_sample = False
        uniform_sample = True
        # st()
        for ins in tqdm(ins_list_to_process):
            # ins = os.path.join(
            #     # self.file_path, ins , 'campos_512_v4'
            #     self.file_path, ins ,
            #     # 'compos_512_v4'
            # )
            # cur_rgb_path = os.path.join(self.file_path, ins, 'compos_512_v4')
            # cur_pose_path = os.path.join(self.file_path, ins, 'pose')

            # st()
            # ][:27])

            if self.four_view_for_latent:
                # cur_all_fname = [t.split('.')[0] for t in os.listdir(ins)
                #                  ]  # use full set for training
                # cur_all_fname = [f'{idx:05d}' for idx in [0, 12, 30, 36]
                # cur_all_fname = [f'{idx:05d}' for idx in [6,12,18,24]
                # cur_all_fname = [f'{idx:05d}' for idx in [7,16,24,25]
                # cur_all_fname = [f'{idx:05d}' for idx in [25,26,0,9,18,27,33,39]]
                cur_all_fname = [
                    f'{idx:05d}'
                    for idx in [25, 26, 6, 12, 18, 24, 27, 31, 35, 39] # ! for extracting PCD
                ]
                # cur_all_fname = [f'{idx:05d}' for idx in [25,26,0,9,18,27,30,33,36,39]] # more down side for better bottom coverage.
                # cur_all_fname = [f'{idx:05d}' for idx in [25,0, 7,15]]
                # cur_all_fname = [f'{idx:05d}' for idx in [4,12,20,25,26]
                # cur_all_fname = [f'{idx:05d}' for idx in [6,12,18,24,25,26]
                # cur_all_fname = [f'{idx:05d}' for idx in [6,12,18,24,25,26, 39, 33, 27]
                # cur_all_fname = [f'{idx:05d}' for idx in [6,12,18,24,25,26, 39, 33, 27]

                # cur_all_fname = [
                #     f'{idx:05d}' for idx in [25, 26, 27, 30, 33, 36]
                # ]  # for pcd unprojection

                # cur_all_fname = [
                #     f'{idx:05d}' for idx in [25, 26, 27, 30] # ! for infer latents
                # ]  #

                # cur_all_fname = [
                #     f'{idx:05d}' for idx in [25, 27, 29, 31, 33, 35, 37
                #                              ]  # ! for infer latents
                # ]  #

                # cur_all_fname = [
                #     f'{idx:05d}' for idx in [25, 27, 31, 35
                #                              ]  # ! for infer latents
                # ]  #

                # cur_all_fname += [f'{idx:05d}' for idx in range(40) if idx not in [0,12,30,36]] # ! four views for inference
            elif self.single_view_for_i23d:
                # cur_all_fname = [f'{idx:05d}'
                #                  for idx in [16]]  # 20 is also fine
                cur_all_fname = [f'{idx:05d}'
                                 for idx in [2]]  # ! furniture side view

            else:
                cur_all_fname = [t.split('.')[0] for t in os.listdir(ins)
                                 ]  # use full set for training

                if shuffle_across_cls:
                    if uniform_sample:
                        cur_all_fname = sorted(cur_all_fname)
                        # 0-24, 25 views
                        # 25,26, 2 views
                        # 27-39, 13 views
                        uniform_all_fname = []

                        # !!!! if bs=9 or 8
                        for idx in range(6):
                            if idx % 2 == 0:
                                chunk_all_fname = [25]
                            else:
                                chunk_all_fname = [26]
                            # chunk_all_fname = [25] # no bottom view required as input
                            # start_1 = np.random.randint(0,5) # for first 24 views
                            # chunk_all_fname += [start_1+uniform_idx for uniform_idx in range(0,25,5)]

                            start_1 = np.random.randint(0,4) # for first 24 views, v=8
                            chunk_all_fname += [start_1+uniform_idx for uniform_idx in range(0,25,7)] # [0-21]

                            start_2 = np.random.randint(0,5) + 27 # for first 24 views
                            chunk_all_fname += [start_2, start_2 + 4, start_2 + 8]
                            assert len(chunk_all_fname) == 8, len(chunk_all_fname)
                            uniform_all_fname += [cur_all_fname[fname] for fname in chunk_all_fname]

                        # ! if bs=6
                        # for idx in range(8):

                        #     if idx % 2 == 0:
                        #         chunk_all_fname = [
                        #             25
                        #         ]  # no bottom view required as input
                        #     else:
                        #         chunk_all_fname = [
                        #             26
                        #         ]  # no bottom view required as input

                        #     start_1 = np.random.randint(
                        #         0, 7)  # for first 24 views
                        #     # chunk_all_fname += [start_1+uniform_idx for uniform_idx in range(0,25,5)]
                        #     chunk_all_fname += [
                        #         start_1 + uniform_idx
                        #         for uniform_idx in range(0, 25, 9)
                        #     ]  # 0 9 18
                        #     start_2 = np.random.randint(
                        #         0, 7) + 27  # for first 24 views
                        #     # chunk_all_fname += [start_2, start_2 + 4, start_2 + 8]
                        #     chunk_all_fname += [start_2,
                        #                         start_2 + 6]  # 2 frames
                        #     assert len(chunk_all_fname) == 6
                        #     uniform_all_fname += [
                        #         cur_all_fname[fname]
                        #         for fname in chunk_all_fname
                        #     ]

                        cur_all_fname = uniform_all_fname

                    else:
                        current_time = int(current_milli_time(
                        ))  # randomly shuffle given current time
                        random.seed(current_time)
                        random.shuffle(cur_all_fname)
                else:
                    cur_all_fname = sorted(cur_all_fname)

                # ! skip the check
                # if self.instance_data_length == -1:
                #     self.instance_data_length = len(cur_all_fname)
                # else:
                #     try:  # data missing?
                #         assert len(cur_all_fname) == self.instance_data_length
                #     except:
                #         # with open('error_log.txt', 'a') as f:
                #         #     f.write(str(e) + '\n')
                #         with open('missing_ins_new2.txt', 'a') as f:
                #             f.write(str(Path(ins.parent)) +
                #                     '\n')  # remove the "campos_512_v4"
                #         continue

            # if test: # use middle image as the novel view model input
            #     mid_index = len(cur_all_fname) // 3 * 2
            #     cur_all_fname.insert(0, cur_all_fname[mid_index])

            self.frame0_pose_list += ([
                os.path.join(ins, fname, fname + '.json')
                for fname in [cur_all_fname[0]]
            ] * len(cur_all_fname))

            self.pose_list += ([
                os.path.join(ins, fname, fname + '.json')
                for fname in cur_all_fname
            ])
            self.rgb_list += ([
                os.path.join(ins, fname, fname + '.png')
                for fname in cur_all_fname
            ])

            self.depth_list += ([
                os.path.join(ins, fname, fname + '_nd.exr')
                for fname in cur_all_fname
            ])
            self.data_ins_list += ([ins] * len(cur_all_fname))

        # check

        # ! setup normalizataion
        transformations = [
            transforms.ToTensor(),  # [0,1] range
        ]
        if imgnet_normalize:
            transformations.append(
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))  # type: ignore
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)))  # type: ignore

        # st()
        self.normalize = transforms.Compose(transformations)

    def get_source_cw2wT(self, source_cameras_view_to_world):
        return matrix_to_quaternion(
            source_cameras_view_to_world[:3, :3].transpose(0, 1))

    def c_to_3dgs_format(self, pose):
        # TODO, switch to torch version (batched later)

        c2w = pose[:16].reshape(4, 4)  # 3x4

        # ! load cam
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        fx = pose[16]
        FovX = focal2fov(fx, 1)
        FovY = focal2fov(fx, 1)

        tanfovx = math.tan(FovX * 0.5)
        tanfovy = math.tan(FovY * 0.5)

        assert tanfovx == tanfovy

        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0

        world_view_transform = torch.tensor(getWorld2View2(R, T, trans,
                                                           scale)).transpose(
                                                               0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear,
                                                zfar=self.zfar,
                                                fovX=FovX,
                                                fovY=FovY).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
            projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        view_world_transform = torch.tensor(getView2World(R, T, trans,
                                                          scale)).transpose(
                                                              0, 1)

        # item.update(viewpoint_cam=[viewpoint_cam])
        c = {}
        c["source_cv2wT_quat"] = self.get_source_cw2wT(view_world_transform)
        c.update(
            # projection_matrix=projection_matrix, # K
            cam_view=world_view_transform,  # world_view_transform
            cam_view_proj=full_proj_transform,  # full_proj_transform
            cam_pos=camera_center,
            tanfov=tanfovx,  # TODO, fix in the renderer
            # orig_c2w=c2w,
            # orig_w2c=w2c,
            orig_pose=torch.from_numpy(pose),
            orig_c2w=torch.from_numpy(c2w),
            orig_w2c=torch.from_numpy(w2c),
            # tanfovy=tanfovy,
        )

        return c  # dict for gs rendering

    def __len__(self):
        return len(self.rgb_list)

    def load_bbox(self, mask):
        # st()
        nonzero_value = torch.nonzero(mask)
        height, width = nonzero_value.max(dim=0)[0]
        top, left = nonzero_value.min(dim=0)[0]
        bbox = torch.tensor([top, left, height, width], dtype=torch.float32)
        return bbox

    def __getitem__(self, idx):
        # try:

        data = self._read_data(idx)
        return data

        # except Exception as e:
        #     # with open('error_log_pcd.txt', 'a') as f:
        #     with open('error_log_pcd.txt', 'a') as f:
        #         f.write(str(e) + '\n')
        #     with open('error_idx_pcd.txt', 'a') as f:
        #         f.write(str(self.data_ins_list[idx]) + '\n')
        #     print(e, flush=True)
        #     return {}

    def gen_rays(self, c2w):
        # Generate rays
        self.h = self.reso_encoder
        self.w = self.reso_encoder
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
            indexing='ij')

        # normalize to 0-1 pixel range
        yy = yy / self.h
        xx = xx / self.w

        # K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
        cx, cy, fx, fy = self.intrinsics[2], self.intrinsics[
            5], self.intrinsics[0], self.intrinsics[4]
        # cx *= self.w
        # cy *= self.h

        # f_x = f_y = fx * h / res_raw
        c2w = torch.from_numpy(c2w).float()

        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        del xx, yy, zz
        # st()
        dirs = (c2w[None, :3, :3] @ dirs)[..., 0]

        origins = c2w[None, :3, 3].expand(self.h * self.w, -1).contiguous()
        origins = origins.view(self.h, self.w, 3)
        dirs = dirs.view(self.h, self.w, 3)

        return origins, dirs

    def normalize_camera(self, c, c_frame0):
        # assert c.shape[0] == self.chunk_size  # 8 o r10

        B = c.shape[0]
        camera_poses = c[:, :16].reshape(B, 4, 4)  # 3x4
        canonical_camera_poses = c_frame0[:, :16].reshape(B, 4, 4)

        # if for_encoder:

        # encoder_canonical_idx = [0, self.V]
        # st()
        cam_radius = np.linalg.norm(
            c_frame0[:, :16].reshape(1, 4, 4)[:, :3, 3],
            axis=-1,
            keepdims=False)  # since g-buffer adopts dynamic radius here.

        frame1_fixed_pos = np.repeat(np.eye(4)[None], 1, axis=0)
        frame1_fixed_pos[:, 2, -1] = -cam_radius

        transform = frame1_fixed_pos @ np.linalg.inv(canonical_camera_poses)
        # from LGM, https://github.com/3DTopia/LGM/blob/fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c/core/provider_objaverse.py#L127
        # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(c[[0,4]])

        new_camera_poses = np.repeat(
            transform, 1, axis=0
        ) @ camera_poses  # [V, 4, 4]. np.repeat() is th.repeat_interleave()

        # else:
        #     cam_radius = np.linalg.norm(
        #         c[canonical_idx][:16].reshape(4, 4)[:3, 3],
        #         axis=-1,
        #         keepdims=False
        #     )  # since g-buffer adopts dynamic radius here.
        #     frame1_fixed_pos = np.eye(4)
        #     frame1_fixed_pos[2, -1] = -cam_radius

        #     transform = frame1_fixed_pos @ np.linalg.inv(
        #         camera_poses[canonical_idx])  # 4,4
        #     # from LGM, https://github.com/3DTopia/LGM/blob/fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c/core/provider_objaverse.py#L127
        #     # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(c[[0,4]])

        #     new_camera_poses = np.repeat(
        #         transform[None], self.chunk_size,
        #         axis=0) @ camera_poses  # [V, 4, 4]

        # st()
        c = np.concatenate([new_camera_poses.reshape(B, 16), c[:, 16:]],
                           axis=-1)
        # st()

        return c

    def _read_data(
        self,
        idx,
    ):
        rgb_fname = self.rgb_list[idx]
        pose_fname = self.pose_list[idx]

        raw_img = imageio.imread(rgb_fname)

        # ! RGBD
        alpha_mask = raw_img[..., -1:] / 255
        raw_img = alpha_mask * raw_img[..., :3] + (
            1 - alpha_mask) * np.ones_like(raw_img[..., :3]) * 255

        raw_img = raw_img.astype(
            np.uint8)  # otherwise, float64 won't call ToTensor()

        # return raw_img
        # st()

        if self.preprocess is None:
            img_to_encoder = cv2.resize(raw_img,
                                        (self.reso_encoder, self.reso_encoder),
                                        interpolation=cv2.INTER_LANCZOS4)
            # interpolation=cv2.INTER_AREA)
            img_to_encoder = img_to_encoder[
                ..., :3]  #[3, reso_encoder, reso_encoder]
            img_to_encoder = self.normalize(img_to_encoder)
        else:
            img_to_encoder = self.preprocess(Image.open(rgb_fname))  # clip

        # return img_to_encoder

        img = cv2.resize(raw_img, (self.reso, self.reso),
                         interpolation=cv2.INTER_LANCZOS4)

        #  interpolation=cv2.INTER_AREA)

        # img_sr = cv2.resize(raw_img, (512, 512), interpolation=cv2.INTER_AREA)
        # img_sr = cv2.resize(raw_img, (256, 256), interpolation=cv2.INTER_AREA) # just as refinement, since eg3d uses 64->128 final resolution
        # img_sr = cv2.resize(raw_img, (128, 128), interpolation=cv2.INTER_AREA) # just as refinement, since eg3d uses 64->128 final resolution

        # img_sr = cv2.resize(
        #     raw_img, (128, 128), interpolation=cv2.INTER_LANCZOS4
        # )  # just as refinement, since eg3d uses 64->128 final resolution

        # img = torch.from_numpy(img)[..., :3].permute(
        #     2, 0, 1) / 255.0  #[3, reso, reso]

        img = torch.from_numpy(img)[..., :3].permute(
            2, 0, 1
        ) / 127.5 - 1  #[3, reso, reso], normalize to [-1,1], follow triplane range

        # img_sr = torch.from_numpy(img_sr)[..., :3].permute(
        #     2, 0, 1
        # ) / 127.5 - 1  #[3, reso, reso], normalize to [-1,1], follow triplane range

        c2w = read_camera_matrix_single(pose_fname)  #[1, 4, 4] -> [1, 16]
        # c = np.concatenate([c2w, self.intrinsics], axis=0).reshape(25)  # 25, no '1' dim needed.

        # return c2w

        # if self.load_depth:
        # depth, depth_mask, depth_mask_sr = read_dnormal(self.depth_list[idx],
        # try:
        depth, normal = read_dnormal(self.depth_list[idx], c2w[:3, 3:],
                                     self.reso, self.reso)

        # ! frame0 alignment
        # if self.frame_0_as_canonical:

        # return depth
        # except:
        #     # print(self.depth_list[idx])
        #     raise NotImplementedError(self.depth_list[idx])
        # if depth

        try:
            bbox = self.load_bbox(depth > 0)
        except:
            print(rgb_fname, flush=True)
            with open('error_log.txt', 'a') as f:
                f.write(str(rgb_fname + '\n'))
            bbox = self.load_bbox(torch.ones_like(depth))

        # plucker

        # ! normalize camera

        c = np.concatenate([c2w.reshape(16), self.intrinsics],
                           axis=0).reshape(25).astype(
                               np.float32)  # 25, no '1' dim needed.

        if self.frame_0_as_canonical:  # 4 views as input per batch
            frame0_pose_name = self.frame0_pose_list[idx]
            c2w_frame0 = read_camera_matrix_single(
                frame0_pose_name)  #[1, 4, 4] -> [1, 16]
            c = self.normalize_camera(c[None], c2w_frame0[None])[0]
            c2w = c[:16].reshape(4, 4)  # !
            # st()
            # pass

        rays_o, rays_d = self.gen_rays(c2w)
        rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d],
                                 dim=-1)  # [h, w, 6]

        img_to_encoder = torch.cat(
            [img_to_encoder, rays_plucker.permute(2, 0, 1)],
            0).float()  # concat in C dim

        # ! add depth as input

        depth, normal = read_dnormal(self.depth_list[idx], c2w[:3, 3:],
                                     self.reso_encoder, self.reso_encoder)
        normalized_depth = depth.unsqueeze(0)  # min=0
        img_to_encoder = torch.cat([img_to_encoder, normalized_depth],
                                   0)  # concat in C dim

        if self.gs_cam_format:
            c = self.c_to_3dgs_format(c)
        else:
            c = torch.from_numpy(c)

        ret_dict = {
            # 'rgb_fname': rgb_fname,
            'img_to_encoder': img_to_encoder,
            'img': img,
            'c': c,
            # 'img_sr': img_sr,
            # 'ins_name': self.data_ins_list[idx]
        }

        # ins = str(
        #     (Path(self.data_ins_list[idx]).relative_to(self.file_path)).parent)

        pcd_ins = Path(self.data_ins_list[idx]).relative_to(
            Path(self.file_path).parent).parent
        # load pcd
        # fps_pcd = pcu.load_mesh_v(
        #     str(self.pcd_path / pcd_ins / 'fps-10000.ply'))

        ins = str(  # for compat
            (Path(self.data_ins_list[idx]).relative_to(self.file_path)).parent)
        # if self.shuffle_across_cls:
        caption = self.caption_data['/'.join(ins.split('/')[1:])]
        # else:
        # caption = self.caption_data[ins]

        ret_dict.update({
            'depth': depth,
            'normal': normal,
            'alpha_mask': alpha_mask,
            'depth_mask': depth > 0,
            # 'depth_mask_sr': depth_mask_sr,
            'bbox': bbox,
            'caption': caption,
            'rays_plucker': rays_plucker,  # cam embedding used in lgm
            'ins': ins,  # placeholder
            # 'fps_pcd': fps_pcd,
        })

        return ret_dict


# class MultiViewObjverseDatasetChunk(MultiViewObjverseDataset):

#     def __init__(self,
#                  file_path,
#                  reso,
#                  reso_encoder,
#                  preprocess=None,
#                  classes=False,
#                  load_depth=False,
#                  test=False,
#                  scene_scale=1,
#                  overfitting=False,
#                  imgnet_normalize=True,
#                  dataset_size=-1,
#                  overfitting_bs=-1,
#                  interval=1,
#                  plucker_embedding=False,
#                  shuffle_across_cls=False,
#                  wds_split=1,
#                  four_view_for_latent=False,
#                  single_view_for_i23d=False,
#                  load_extra_36_view=False,
#                  gs_cam_format=False,
#                  **kwargs):
#         super().__init__(file_path, reso, reso_encoder, preprocess, classes,
#                          load_depth, test, scene_scale, overfitting,
#                          imgnet_normalize, dataset_size, overfitting_bs,
#                          interval, plucker_embedding, shuffle_across_cls,
#                          wds_split, four_view_for_latent, single_view_for_i23d,
#                          load_extra_36_view, gs_cam_format, **kwargs)
#         # load 40 views at a time, for inferring latents.


# TODO merge all the useful APIs together
class ChunkObjaverseDataset(Dataset):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
            four_view_for_latent=False,
            single_view_for_i23d=False,
            load_extra_36_view=False,
            gs_cam_format=False,
            frame_0_as_canonical=True,
            split_chunk_size=10,
            mv_input=True,
            append_depth=False,
            append_xyz=False,
            wds_split_all=1,
            pcd_path=None,
            load_pcd=False,
            read_normal=False,
            load_raw=False,
            load_instance_only=False,
            mv_latent_dir='',
            perturb_pcd_scale=0.0,
            # shards_folder_num=4,
            # eval=False,
            **kwargs):

        super().__init__()

        # st()
        self.mv_latent_dir = mv_latent_dir

        self.load_raw = load_raw
        self.load_instance_only = load_instance_only
        self.read_normal = read_normal
        self.file_path = file_path
        self.chunk_size = split_chunk_size
        self.gs_cam_format = gs_cam_format
        self.frame_0_as_canonical = frame_0_as_canonical
        self.four_view_for_latent = four_view_for_latent  # export 0 12 30 36, 4 views for reconstruction
        self.overfitting = overfitting
        self.scene_scale = scene_scale
        self.reso = reso
        self.reso_encoder = reso_encoder
        self.classes = False
        self.load_depth = load_depth
        self.preprocess = preprocess
        self.plucker_embedding = plucker_embedding
        self.intrinsics = get_intri(h=self.reso, w=self.reso,
                                    normalize=True).reshape(9)
        self.perturb_pcd_scale = perturb_pcd_scale

        assert not self.classes, "Not support class condition now."

        dataset_name = Path(self.file_path).stem.split('_')[0]
        self.dataset_name = dataset_name
        self.ray_sampler = RaySampler()

        self.zfar = 100.0
        self.znear = 0.01

        # ! load all chunk paths
        self.chunk_list = []

        # if dataset_size != -1: # predefined instance
        #     self.chunk_list = self.fetch_chunk_list(os.path.join(self.file_path, 'debug'))
        # else:
        #     # for shard_idx in range(1, 5): # shard_dir 1-4 by default
        #     for shard_idx in os.listdir(self.file_path):
        #         self.chunk_list += self.fetch_chunk_list(os.path.join(self.file_path, shard_idx))

        def load_single_cls_instances(file_path):
            ins_list = []  # the first 1 instance for evaluation reference.

            for dict_dir in os.listdir(file_path)[:]: # ! for debugging
                for ins_dir in os.listdir(os.path.join(file_path, dict_dir)):
                    ins_list.append(
                        os.path.join(file_path, dict_dir, ins_dir,
                                    'campos_512_v4'))
            return ins_list
        
        # st()

        if self.load_raw:

            with open(
                    # '/nas/shared/V2V/yslan/aigc3d/text_captions_cap3d.json') as f:
                    # '/nas/shared/public/yslan//data/text_captions_cap3d.json') as f:
                    './dataset/text_captions_3dtopia.json') as f:
                self.caption_data = json.load(f)

            # with open
            #         # '/nas/shared/V2V/yslan/aigc3d/text_captions_cap3d.json') as f:
            #         '/nas/shared/public/yslan//data/text_captions_cap3d.json') as f:
            #         # '/cpfs01/shared/public/yhluo/Projects/threed/3D-Enhancer/develop/text_captions_3dtopia.json') as f:
            #     self.old_caption_data = json.load(f)

            for subset in [  # ! around 17.6 W instances in total. 
                    'Animals',
                    # 'daily-used',
                    # 'BuildingsOutdoor',
                    # 'Furnitures',
                    # 'Food',
                    # 'Plants',
                    # 'Electronics',
                    # 'Transportations_tar',
                    # 'Human-Shape',
            ]:  # selected subset for training
                # self.chunk_list += load_single_cls_instances(
                #     os.path.join(self.file_path, subset))
                with open(f'shell_scripts/raw_img_list/{subset}.txt', 'r') as f:
                    self.chunk_list += [os.path.join(subset, item.strip()) for item in f.readlines()]

            # st() # save to local
            # with open('/cpfs01/user/lanyushi.p/Repo/diffusion-3d/shell_scripts/shards_list/chunk_list.txt', 'w') as f:
            #     f.writelines(self.chunk_list)
            # load raw g-objv dataset
            # self.img_ext = 'png'  # ln3diff
            # for k, v in dataset_json.items(): # directly load from folders instead
            #     self.chunk_list.extend(v)
        else:

            # ! direclty load from json
            with open(f'{self.file_path}/dataset.json', 'r') as f:
                dataset_json = json.load(f)
                # dataset_json = {'Animals': ['Animals/0/10017/1']} 

            if self.chunk_size == 12:
                self.img_ext = 'png'  # ln3diff
                for k, v in dataset_json.items():
                    self.chunk_list.extend(v)
            else:
                # extract latent
                assert self.chunk_size in [16,18, 20]
                self.img_ext = 'jpg'  # more views
                for k, v in dataset_json.items():
                    # if k != 'BuildingsOutdoor':  # cannot be handled by gs
                    self.chunk_list.extend(v)
        
        # filter 
        # st()
        # root = '/nas/shared/V2V/yslan/logs/nips23/Reconstruction/final/objav/vae/gs/infer-latents/768/8x8/animals-gs-latent/latent_dir'
        # root = '/nas/shared/V2V/yslan/logs/nips23/Reconstruction/final/objav/vae/gs/infer-latents/768/8x8/animals-gs-latent-dim=10-fullset/latent_dir'
        # filtered_chunk_list = []
        # for v in self.chunk_list:
        #     if os.path.exists(os.path.join(root, v[:-2], 'gaussians.npy') ):
        #         continue
        #     filtered_chunk_list.append(v)
        # self.chunk_list = filtered_chunk_list

        dataset_size = len(self.chunk_list)
        self.chunk_list = sorted(self.chunk_list)

        # self.chunk_list, self.eval_list = self.chunk_list[:int(dataset_size*0.95)], self.chunk_list[int(dataset_size*0.95):]
        # self.chunk_list = self.eval_list

        # self.wds_split_all = wds_split_all # for 
        # self.wds_split_all = 1 
        # self.wds_split_all = 7
        # self.wds_split_all = 4
        self.wds_split_all = 1


        # ! filter
        # st()

        if wds_split_all != 1:
            # ! retrieve the right wds split
            all_ins_size = len(self.chunk_list)
            ratio_size = all_ins_size // self.wds_split_all + 1
            # ratio_size = int(all_ins_size / self.wds_split_all) + 1
            print('ratio_size: ', ratio_size, 'all_ins_size: ', all_ins_size)

            self.chunk_list = self.chunk_list[ratio_size *
                                                (wds_split):ratio_size *
                                                (wds_split + 1)]
    
        # st()

        # load images from raw
        self.rgb_list = []

        if self.load_instance_only:
            for ins in tqdm(self.chunk_list):

                ins_name = str(Path(ins).parent)
                # cur_all_fname = [f'{t:05d}' for t in range(40)] # load all instances for now

                self.rgb_list += ([
                    os.path.join(self.file_path, ins, fname + '.png')
                    for fname in [f'{t}' for t in range(2)]
                    # for fname in [f'{t:05d}' for t in range(2)]
                ]) # synthetic mv data

            # index mapping of mvi data to objv single-view data
            self.mvi_objv_mapping = {
                '0': '00000',
                '1': '00012',
            }

            # load gt mv data

            self.gt_chunk_list = []
            self.gt_mv_file_path = '/cpfs01/user/lanyushi.p/data/chunk-jpeg-normal/bs_16_fixsave3/170K/512/'
            assert self.chunk_size in [16,18, 20]

            with open(f'{self.gt_mv_file_path}/dataset.json', 'r') as f:
                dataset_json = json.load(f)
                # dataset_json = {'Animals': dataset_json['Animals'] } #

            self.img_ext = 'jpg'  # more views
            for k, v in dataset_json.items():
                # if k != 'BuildingsOutdoor':  # cannot be handled by gs
                self.gt_chunk_list.extend(v)


        elif self.load_raw:
            for ins in tqdm(self.chunk_list):
                # 
                # st()
                # ins = ins[len('/nas/shared/V2V/yslan/aigc3d/unzip4/'):]
                # ins_name = str(Path(ins).relative_to(self.file_path).parent)
                ins_name = str(Path(ins).parent)
                # latent_path = os.path.join(self.mv_latent_dir, ins_name, 'latent.npz')
                # if not os.path.exists(latent_path):
                #     continue

                cur_all_fname = [f'{t:05d}' for t in range(40)] # load all instances for now

                self.rgb_list += ([
                    os.path.join(self.file_path, ins, fname, fname + '.png')
                    for fname in cur_all_fname
                ])
    
        self.post_process = PostProcess(
            reso,
            reso_encoder,
            imgnet_normalize=imgnet_normalize,
            plucker_embedding=plucker_embedding,
            decode_encode_img_only=False,
            mv_input=mv_input,
            split_chunk_input=split_chunk_size,
            duplicate_sample=True,
            append_depth=append_depth,
            append_xyz=append_xyz,
            gs_cam_format=gs_cam_format,
            orthog_duplicate=False,
            frame_0_as_canonical=frame_0_as_canonical,
            pcd_path=pcd_path,
            load_pcd=load_pcd,
            split_chunk_size=split_chunk_size,
        )
        self.kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

        # self.no_bottom = True # avoid loading bottom vew

    def fetch_chunk_list(self, file_path):
        if os.path.isdir(file_path):
            chunks = [
                os.path.join(file_path, fname)
                for fname in os.listdir(file_path) if fname.isdigit()
            ]
            return chunks
        else:
            return []

    def _pre_process_chunk(self):
        # e.g., remove bottom view
        pass

    def read_chunk(self, chunk_path):
        # equivalent to decode_zip() in wds

        # reshape chunk
        raw_img = imageio.imread(
            os.path.join(chunk_path, f'raw_img.{self.img_ext}'))
        h, bw, c = raw_img.shape
        raw_img = raw_img.reshape(h, self.chunk_size, -1, c).transpose(
            (1, 0, 2, 3))
        c = np.load(os.path.join(chunk_path, 'c.npy'))

        with open(os.path.join(chunk_path, 'caption.txt'),
                  'r',
                  encoding="utf-8") as f:
            caption = f.read()

        with open(os.path.join(chunk_path, 'ins.txt'), 'r',
                  encoding="utf-8") as f:
            ins = f.read()

        bbox = np.load(os.path.join(chunk_path, 'bbox.npy'))
        
        if self.chunk_size > 16:

            depth_alpha = imageio.imread(
                os.path.join(chunk_path, 'depth_alpha.jpg'))  # 2h 10w
            depth_alpha = depth_alpha.reshape(h * 2, self.chunk_size,
                                            -1).transpose((1, 0, 2))

            depth, alpha = np.split(depth_alpha, 2, axis=1)

            d_near_far = np.load(os.path.join(chunk_path, 'd_near_far.npy'))

            d_near = d_near_far[0].reshape(self.chunk_size, 1, 1)
            d_far = d_near_far[1].reshape(self.chunk_size, 1, 1)
            # d = 1 / ( (d_normalized / 255) * (far-near) + near)
            depth = 1 / ((depth / 255) * (d_far - d_near) + d_near)

            depth[depth > 2.9] = 0.0  # background as 0, follow old tradition

            # ! filter anti-alias artifacts

            erode_mask = kornia.morphology.erosion(
                torch.from_numpy(alpha == 255).float().unsqueeze(1),
                self.kernel)  # B 1 H W
            depth = (torch.from_numpy(depth).unsqueeze(1) * erode_mask).squeeze(
                1)  # shrink anti-alias bug

        else:
            # load separate alpha and depth map

            alpha = imageio.imread(
                os.path.join(chunk_path, f'alpha.{self.img_ext}'))
            alpha = alpha.reshape(h, self.chunk_size, h).transpose(
                (1, 0, 2))            
            depth = np.load(os.path.join(chunk_path, 'depth.npz'))['depth']
            # depth = depth * (alpha==255) # mask out background



        # depth = np.stack([depth, alpha], -1)  # rgba

        # if self.no_bottom:
        #     raw_img
        #     pass

        if self.read_normal:
            normal = imageio.imread(os.path.join(
                chunk_path, 'normal.png')).astype(np.float32) / 255.0

            normal = (normal * 2 - 1).reshape(h, self.chunk_size, -1,
                                              3).transpose((1, 0, 2, 3))
            # fix g-buffer normal rendering coordinate
            # normal = unity2blender(normal) # ! still wrong
            normal = unity2blender_fix(normal) # ! 
            depth = (depth, normal)  # ?

        return raw_img, depth, c, alpha, bbox, caption, ins

    def __len__(self):
        return len(self.chunk_list)

    def __getitem__(self, index) -> Any:
        sample = self.read_chunk(
            os.path.join(self.file_path, self.chunk_list[index]))
        sample = self.post_process.paired_post_process_chunk(sample)

        sample = self.post_process.create_dict_nobatch(sample)

        # aug pcd
        # st()
        if self.perturb_pcd_scale > 0:
            if random.random() > 0.5:
                t = np.random.rand(sample['fps_pcd'].shape[0], 1, 1) * self.perturb_pcd_scale
                sample['fps_pcd'] = sample['fps_pcd'] + t * np.random.randn(*sample['fps_pcd'].shape) # type: ignore
                sample['fps_pcd'] = np.clip(sample['fps_pcd'], -0.45, 0.45) # truncate noisy augmentation

        return sample


class ChunkObjaverseDatasetDDPM(ChunkObjaverseDataset):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
            four_view_for_latent=False,
            single_view_for_i23d=False,
            load_extra_36_view=False,
            gs_cam_format=False,
            frame_0_as_canonical=True,
            split_chunk_size=10,
            mv_input=True,
            append_depth=False,
            append_xyz=False,
            pcd_path=None,
            load_pcd=False,
            read_normal=False,
            mv_latent_dir='',
            load_raw=False,
            # shards_folder_num=4,
            # eval=False,
            **kwargs):

        super().__init__(
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
            four_view_for_latent=False,
            single_view_for_i23d=False,
            load_extra_36_view=False,
            gs_cam_format=False,
            frame_0_as_canonical=True,
            split_chunk_size=split_chunk_size,
            mv_input=True,
            append_depth=False,
            append_xyz=False,
            pcd_path=None,
            load_pcd=False,
            read_normal=False,
            load_raw=load_raw,
            mv_latent_dir=mv_latent_dir,
            # shards_folder_num=4,
            # eval=False,
            **kwargs)

        self.n_cond_frames = 6
        self.perspective_transformer = v2.RandomPerspective(distortion_scale=0.4, p=0.15, fill=1, 
            interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.mv_resize_cls = torchvision.transforms.Resize(320, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
                max_size=None, antialias=True)

        # ! read img c, caption.

    def get_plucker_ray(self, c):
        rays_plucker = []
        for idx in range(c.shape[0]):
            rays_o, rays_d = self.gen_rays(c[idx])
            rays_plucker.append(
                torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d],
                          dim=-1).permute(2, 0, 1))  # [h, w, 6] -> 6,h,w
        rays_plucker = torch.stack(rays_plucker, 0)
        return rays_plucker

    def read_chunk(self, chunk_path):
        # equivalent to decode_zip() in wds

        # reshape chunk
        raw_img = imageio.imread(
            os.path.join(chunk_path, f'raw_img.{self.img_ext}')).astype(np.float32)
        h, bw, c = raw_img.shape
        raw_img = raw_img.reshape(h, self.chunk_size, -1, c).transpose(
            (1, 0, 2, 3))

        c = np.load(os.path.join(chunk_path, 'c.npy')).astype(np.float32)

        with open(os.path.join(chunk_path, 'caption.txt'),
                  'r',
                  encoding="utf-8") as f:
            caption = f.read()

        with open(os.path.join(chunk_path, 'ins.txt'), 'r',
                  encoding="utf-8") as f:
            ins = f.read()

        return raw_img, c, caption, ins
    
    def _load_latent(self, ins):
        # if 'adv' in self.mv_latent_dir: # new latent codes saved have 3 augmentations
        #     idx = random.choice([0,1,2])
        #     latent = np.load(os.path.join(self.mv_latent_dir, ins, f'latent-{idx}.npy'))  # pre-calculated VAE latent
        # else:
        latent = np.load(os.path.join(self.mv_latent_dir, ins, 'latent.npy'))  # pre-calculated VAE latent
        latent = repeat(latent, 'C H W -> B C H W', B=2)
        # return {'latent': latent}
        return latent

    def normalize_camera(self, c, c_frame0):
        # assert c.shape[0] == self.chunk_size  # 8 o r10

        B = c.shape[0]
        camera_poses = c[:, :16].reshape(B, 4, 4)  # 3x4
        canonical_camera_poses = c_frame0[:, :16].reshape(1, 4, 4)
        inverse_canonical_pose = np.linalg.inv(canonical_camera_poses)
        inverse_canonical_pose = np.repeat(inverse_canonical_pose, B, 0)

        cam_radius = np.linalg.norm(
            c_frame0[:, :16].reshape(1, 4, 4)[:, :3, 3],
            axis=-1,
            keepdims=False)  # since g-buffer adopts dynamic radius here.

        frame1_fixed_pos = np.repeat(np.eye(4)[None], 1, axis=0)
        frame1_fixed_pos[:, 2, -1] = -cam_radius

        transform = frame1_fixed_pos @ inverse_canonical_pose

        new_camera_poses = np.repeat(
            transform, 1, axis=0
        ) @ camera_poses  # [V, 4, 4]. np.repeat() is th.repeat_interleave()

        c = np.concatenate([new_camera_poses.reshape(B, 16), c[:, 16:]],
                           axis=-1)

        return c

    # @autocast
    # def plucker_embedding(self, c):
    #     rays_o, rays_d = self.gen_rays(c)
    #     rays_plucker = torch.cat(
    #         [torch.cross(rays_o, rays_d, dim=-1), rays_d],
    #         dim=-1).permute(2, 0, 1)  # [h, w, 6] -> 6,h,w

    #     return rays_plucker

    def __getitem__(self, index) -> Any:
        raw_img, c, caption, ins = self.read_chunk(
            os.path.join(self.file_path, self.chunk_list[index]))
        # sample = self.post_process.paired_post_process_chunk(sample)

        # ! random zoom in (scale augmentation)
        # for i in range(img.shape[0]):
        #     for v in range(img.shape[1]):
        #         if random.random() > 0.8:
        #             rand_bg_scale = random.randint(60,99) / 100
        #             st()
        #             img[i,v] = recenter(img[i,v], np.ones_like(img[i,v]), border_ratio=rand_bg_scale)

        # ! process
        raw_img = torch.from_numpy(raw_img).permute(0, 3, 1, 2) / 255.0  # [0,1]

        if raw_img.shape[-1] != self.reso:
            raw_img = torch.nn.functional.interpolate(
                input=raw_img,
                size=(self.reso, self.reso),
                mode='bilinear',
                align_corners=False,
            )
        img = raw_img * 2 - 1  # as gt

        # ! load latent
        latent, _ = self._load_latent(ins)

        # ! shuffle
        indices = np.random.permutation(self.chunk_size)
        img = img[indices]
        c = c[indices]

        img = self.perspective_transformer(img) # create 3D inconsistency

        # ! split along V and repeat other stuffs accordingly
        img = rearrange(img, '(B V) ... -> B V ...', B=2)[:, :self.n_cond_frames]
        c = rearrange(c, '(B V) ... -> B V ...', B=2)[:, :self.n_cond_frames] # 2 6 25


        # rand perspective aug
        caption = [caption, caption]
        ins = [ins, ins]

        # load plucker coord
        # st()
        # plucker_c = self.get_plucker_ray(rearrange(c[:, 1:1+self.n_cond_frames], "b t ... -> (b t) ..."))
        # plucker_c = rearrange(c, '(B V) ... -> B V ...', B=2) # 2 6 25

        # use view-space camera tradition
        c[0] = self.normalize_camera(c[0], c[0,0:1])
        c[1] = self.normalize_camera(c[1], c[1,0:1])

        # https://github.com/TencentARC/InstantMesh/blob/7fe95627cf819748f7830b2b278f302a9d798d17/src/model.py#L70
        # c = np.concatenate([c[..., :12], c[..., 16:17], c[..., 20:21], c[..., 18:19], c[..., 21:22]], axis=-1)
        # c = c + np.random.randn(*c.shape) * 0.04 - 0.02


        # ! to dict
        # sample = self.post_process.create_dict_nobatch(sample)
        ret_dict = {
            'caption': caption,
            'ins': ins,
            'c': c,
            'img': img, # fix inp img range to [-1,1]
            'latent': latent,
            # **latent
        }

        # st()

        return ret_dict


class ChunkObjaverseDatasetDDPMgs(ChunkObjaverseDatasetDDPM):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
            four_view_for_latent=False,
            single_view_for_i23d=False,
            load_extra_36_view=False,
            gs_cam_format=False,
            frame_0_as_canonical=True,
            split_chunk_size=10,
            mv_input=True,
            append_depth=False,
            append_xyz=False,
            pcd_path=None,
            load_pcd=False,
            read_normal=False,
            mv_latent_dir='',
            load_raw=False,
            # shards_folder_num=4,
            # eval=False,
            **kwargs):

        super().__init__(
            file_path,
            reso,
            reso_encoder,
            preprocess=preprocess,
            classes=classes,
            load_depth=load_depth,
            test=test,
            scene_scale=scene_scale,
            overfitting=overfitting,
            imgnet_normalize=imgnet_normalize,
            dataset_size=dataset_size,
            overfitting_bs=overfitting_bs,
            interval=interval,
            plucker_embedding=plucker_embedding,
            shuffle_across_cls=shuffle_across_cls,
            wds_split=wds_split,  # 4 splits to accelerate preprocessing
            four_view_for_latent=four_view_for_latent,
            single_view_for_i23d=single_view_for_i23d,
            load_extra_36_view=load_extra_36_view,
            gs_cam_format=gs_cam_format,
            frame_0_as_canonical=frame_0_as_canonical,
            split_chunk_size=split_chunk_size,
            mv_input=mv_input,
            append_depth=append_depth,
            append_xyz=append_xyz,
            pcd_path=pcd_path,
            load_pcd=load_pcd,
            read_normal=read_normal,
            mv_latent_dir=mv_latent_dir,
            load_raw=load_raw,
            # shards_folder_num=4,
            # eval=False,
            **kwargs)
        
        self.avoid_loading_first = False

    #     self.feat_scale_factor = torch.Tensor([0.99227685, 1.014337  , 0.20842505, 0.98727155, 0.3305389 ,
    #    0.38729668, 1.0155401 , 0.9728264 , 1.0009694 , 0.97328585,
    #    0.2881106 , 0.1652732 , 0.3482468 , 0.9971449 , 0.99895126,
    #    0.18491288]).float().reshape(1,1,-1)

        # stat for normalization
        # self.xyz_mean = torch.Tensor([-0.00053714, 0.08095618, -0.01914407] ).reshape(1, 3).float()
        # self.xyz_std = np.array([0.14593576, 0.15753542, 0.18873914] ).reshape(1,3).astype(np.float32)

        # self.xyz_std = np.array([0.14593576, 0.15753542, 0.18873914] ).reshape(1,3).astype(np.float32)
        self.xyz_std = 0.164 # a global scaler

        self.kl_mean = np.array([ 0.0184,  0.0024,  0.0926,  0.0517,  0.1781,  0.7137, -0.0355,  0.0267,
         0.0183,  0.0164, -0.5090,  0.2406,  0.2733, -0.0256, -0.0285,  0.0761]).reshape(1,16).astype(np.float32)

        self.kl_std = np.array([1.0018, 1.0309, 1.3001, 1.0160, 0.8182, 0.8023, 1.0591, 0.9789, 0.9966,
        0.9448, 0.8908, 1.4595, 0.7957, 0.9871, 1.0236, 1.2923]).reshape(1,16).astype(np.float32)


    def normalize_pcd_act(self, x):
        return x / self.xyz_std
    
    def normalize_kl_feat(self, latent):
        # return latent / self.feat_scale_factor
        return (latent-self.kl_mean) / self.kl_std

    def _load_latent(self, ins, rand_pick_one=False, pick_both=False):

        if 'adv' in self.mv_latent_dir: # new latent codes saved have 3 augmentations
            idx = random.choice([0,1,2])
            # idx = random.choice([0])
            latent = np.load(os.path.join(self.mv_latent_dir, ins, f'latent-{idx}.npz'))  # pre-calculated VAE latent
        else:
            latent = np.load(os.path.join(self.mv_latent_dir, ins, 'latent.npz'))  # pre-calculated VAE latent

        latent, fps_xyz = latent['latent_normalized'], latent['query_pcd_xyz'] # 2,768,16; 2,768,3

        if not pick_both:
            if rand_pick_one:
                rand_idx = random.randint(0,1)
            else:
                rand_idx = 0

            latent, fps_xyz = latent[rand_idx:rand_idx+1], fps_xyz[rand_idx:rand_idx+1]
        
        # per-channel normalize to std=1 & concat
        # latent_pcd = np.concatenate([self.normalize_kl_feat(latent), self.normalize_pcd_act(fps_xyz)], -1)
        # latent_pcd = np.concatenate([latent, self.normalize_pcd_act(fps_xyz)], -1)

        # return latent_pcd, fps_xyz
        return latent, fps_xyz


    def __getitem__(self, index) -> Any:
        raw_img, c, caption, ins = self.read_chunk(
            os.path.join(self.file_path, self.chunk_list[index]))
        # sample = self.post_process.paired_post_process_chunk(sample)

        # ! random zoom in (scale augmentation)
        # for i in range(img.shape[0]):
        #     for v in range(img.shape[1]):
        #         if random.random() > 0.8:
        #             rand_bg_scale = random.randint(60,99) / 100
        #             st()
        #             img[i,v] = recenter(img[i,v], np.ones_like(img[i,v]), border_ratio=rand_bg_scale)

        # ! process
        raw_img = torch.from_numpy(raw_img).permute(0, 3, 1, 2) / 255.0  # [0,1]

        if raw_img.shape[-1] != self.reso:
            raw_img = torch.nn.functional.interpolate(
                input=raw_img,
                size=(self.reso, self.reso),
                mode='bilinear',
                align_corners=False,
            )
        img = raw_img * 2 - 1  # as gt

        # ! load latent
        # latent, _ = self._load_latent(ins)

        latent, fps_xyz = self._load_latent(ins, pick_both=True) # analyzing xyz/latent disentangled diffusion
        # latent, fps_xyz = latent[0], fps_xyz[0] # remove batch dim here

        # fps_xyz = fps_xyz / self.scaling_factor # for xyz training
        normalized_fps_xyz = self.normalize_pcd_act(fps_xyz)

        if self.avoid_loading_first: # for training mv model
            index = list(range(1,6)) + list(range(7,12))
            img = img[index]
            c = c[index]
        
        # ! shuffle
        indices = np.random.permutation(img.shape[0])
        img = img[indices]
        c = c[indices]

        img = self.perspective_transformer(img) # create 3D inconsistency


        # ! split along V and repeat other stuffs accordingly
        img = rearrange(img, '(B V) ... -> B V ...', B=2)[:, :self.n_cond_frames]
        c = rearrange(c, '(B V) ... -> B V ...', B=2)[:, :self.n_cond_frames] # 2 6 25

        # rand perspective aug
        caption = [caption, caption]
        ins = [ins, ins]

        # load plucker coord
        # st()
        # plucker_c = self.get_plucker_ray(rearrange(c[:, 1:1+self.n_cond_frames], "b t ... -> (b t) ..."))
        # plucker_c = rearrange(c, '(B V) ... -> B V ...', B=2) # 2 6 25

        # use view-space camera tradition
        c[0] = self.normalize_camera(c[0], c[0,0:1])
        c[1] = self.normalize_camera(c[1], c[1,0:1])

        # ! to dict
        # sample = self.post_process.create_dict_nobatch(sample)
        ret_dict = {
            'caption': caption,
            'ins': ins,
            'c': c,
            'img': img, # fix inp img range to [-1,1]
            'latent': latent,
            'normalized-fps-xyz': normalized_fps_xyz
            # **latent
        }

        # st()

        return ret_dict




class ChunkObjaverseDatasetDDPMgsT23D(ChunkObjaverseDatasetDDPMgs):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
            four_view_for_latent=False,
            single_view_for_i23d=False,
            load_extra_36_view=False,
            gs_cam_format=False,
            frame_0_as_canonical=True,
            split_chunk_size=10,
            mv_input=True,
            append_depth=False,
            append_xyz=False,
            pcd_path=None,
            load_pcd=False,
            read_normal=False,
            mv_latent_dir='',
            # shards_folder_num=4,
            # eval=False,
            **kwargs):

        super().__init__(
            file_path,
            reso,
            reso_encoder,
            preprocess=preprocess,
            classes=classes,
            load_depth=load_depth,
            test=test,
            scene_scale=scene_scale,
            overfitting=overfitting,
            imgnet_normalize=imgnet_normalize,
            dataset_size=dataset_size,
            overfitting_bs=overfitting_bs,
            interval=interval,
            plucker_embedding=plucker_embedding,
            shuffle_across_cls=shuffle_across_cls,
            wds_split=wds_split,  # 4 splits to accelerate preprocessing
            four_view_for_latent=four_view_for_latent,
            single_view_for_i23d=single_view_for_i23d,
            load_extra_36_view=load_extra_36_view,
            gs_cam_format=gs_cam_format,
            frame_0_as_canonical=frame_0_as_canonical,
            split_chunk_size=split_chunk_size,
            mv_input=mv_input,
            append_depth=append_depth,
            append_xyz=append_xyz,
            pcd_path=pcd_path,
            load_pcd=load_pcd,
            read_normal=read_normal,
            mv_latent_dir=mv_latent_dir,
            load_raw=True,
            # shards_folder_num=4,
            # eval=False,
            **kwargs)

    # def __len__(self):
    #     return 40

    def __len__(self):
        return len(self.rgb_list) 

    def __getitem__(self, index) -> Any:

        
        rgb_path = self.rgb_list[index]
        ins = str(Path(rgb_path).relative_to(self.file_path).parent.parent.parent)

        # load caption
        caption = self.caption_data['/'.join(ins.split('/')[1:])]

        # chunk_path = os.path.join(self.file_path, self.chunk_list[index])

        # # load caption
        # with open(os.path.join(chunk_path, 'caption.txt'),
        #           'r',
        #           encoding="utf-8") as f:
        #     caption = f.read()

        # # load latent
        # with open(os.path.join(chunk_path, 'ins.txt'), 'r',
        #           encoding="utf-8") as f:
        #     ins = f.read()

        latent, fps_xyz = self._load_latent(ins, True) # analyzing xyz/latent disentangled diffusion
        latent, fps_xyz = latent[0], fps_xyz[0] # remove batch dim here

        # fps_xyz = fps_xyz / self.scaling_factor # for xyz training
        normalized_fps_xyz = self.normalize_pcd_act(fps_xyz)


        # ! to dict
        ret_dict = {
            # 'caption': caption,
            'latent': latent,
            # 'img': img,
            'fps-xyz': fps_xyz,
            'normalized-fps-xyz': normalized_fps_xyz,
            'caption': caption
        }

        return ret_dict


class ChunkObjaverseDatasetDDPMgsI23D(ChunkObjaverseDatasetDDPMgs):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
            four_view_for_latent=False,
            single_view_for_i23d=False,
            load_extra_36_view=False,
            gs_cam_format=False,
            frame_0_as_canonical=True,
            split_chunk_size=10,
            mv_input=True,
            append_depth=False,
            append_xyz=False,
            pcd_path=None,
            load_pcd=False,
            read_normal=False,
            mv_latent_dir='',
            # shards_folder_num=4,
            # eval=False,
            **kwargs):

        super().__init__(
            file_path,
            reso,
            reso_encoder,
            preprocess=preprocess,
            classes=classes,
            load_depth=load_depth,
            test=test,
            scene_scale=scene_scale,
            overfitting=overfitting,
            imgnet_normalize=imgnet_normalize,
            dataset_size=dataset_size,
            overfitting_bs=overfitting_bs,
            interval=interval,
            plucker_embedding=plucker_embedding,
            shuffle_across_cls=shuffle_across_cls,
            wds_split=wds_split,  # 4 splits to accelerate preprocessing
            four_view_for_latent=four_view_for_latent,
            single_view_for_i23d=single_view_for_i23d,
            load_extra_36_view=load_extra_36_view,
            gs_cam_format=gs_cam_format,
            frame_0_as_canonical=frame_0_as_canonical,
            split_chunk_size=split_chunk_size,
            mv_input=mv_input,
            append_depth=append_depth,
            append_xyz=append_xyz,
            pcd_path=pcd_path,
            load_pcd=load_pcd,
            read_normal=read_normal,
            mv_latent_dir=mv_latent_dir,
            load_raw=True,
            # shards_folder_num=4,
            # eval=False,
            **kwargs)

        assert self.load_raw
        self.scaling_factor = np.array([0.14593576, 0.15753542, 0.18873914])

    def __len__(self):
        return len(self.rgb_list)

    # def __len__(self):
    #     return 40

    def __getitem__(self, index) -> Any:

        rgb_path = self.rgb_list[index]
        ins = str(Path(rgb_path).relative_to(self.file_path).parent.parent.parent)

        raw_img = imageio.imread(rgb_path).astype(np.float32)
        alpha_mask = raw_img[..., -1:] / 255
        raw_img = alpha_mask * raw_img[..., :3] + (
            1 - alpha_mask) * np.ones_like(raw_img[..., :3]) * 255

        raw_img = cv2.resize(raw_img, (self.reso, self.reso), interpolation=cv2.INTER_CUBIC)
        raw_img = torch.from_numpy(raw_img).permute(2,0,1).clip(0,255)  # [0,1]
        img = raw_img / 127.5 - 1

        # with open(os.path.join(chunk_path, 'caption.txt'),
        #           'r',
        #           encoding="utf-8") as f:
        #     caption = f.read()

        # latent = self._load_latent(ins, True)[0]
        latent, fps_xyz = self._load_latent(ins, True) # analyzing xyz/latent disentangled diffusion
        latent, fps_xyz = latent[0], fps_xyz[0]

        # fps_xyz = fps_xyz / self.scaling_factor # for xyz training
        normalized_fps_xyz = self.normalize_pcd_act(fps_xyz)

        # load caption
        caption = self.caption_data['/'.join(ins.split('/')[1:])]

        # ! to dict
        ret_dict = {
            # 'caption': caption,
            'latent': latent,
            'img': img.numpy(), # no idea whether loading Tensor leads to 'too many files opened'
            'fps-xyz': fps_xyz,
            'normalized-fps-xyz': normalized_fps_xyz,
            'caption': caption
        }

        return ret_dict

class ChunkObjaverseDatasetDDPMgsMV23D(ChunkObjaverseDatasetDDPMgs):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
            four_view_for_latent=False,
            single_view_for_i23d=False,
            load_extra_36_view=False,
            gs_cam_format=False,
            frame_0_as_canonical=True,
            split_chunk_size=10,
            mv_input=True,
            append_depth=False,
            append_xyz=False,
            pcd_path=None,
            load_pcd=False,
            read_normal=False,
            mv_latent_dir='',
            # shards_folder_num=4,
            # eval=False,
            **kwargs):

        super().__init__(
            file_path,
            reso,
            reso_encoder,
            preprocess=preprocess,
            classes=classes,
            load_depth=load_depth,
            test=test,
            scene_scale=scene_scale,
            overfitting=overfitting,
            imgnet_normalize=imgnet_normalize,
            dataset_size=dataset_size,
            overfitting_bs=overfitting_bs,
            interval=interval,
            plucker_embedding=plucker_embedding,
            shuffle_across_cls=shuffle_across_cls,
            wds_split=wds_split,  # 4 splits to accelerate preprocessing
            four_view_for_latent=four_view_for_latent,
            single_view_for_i23d=single_view_for_i23d,
            load_extra_36_view=load_extra_36_view,
            gs_cam_format=gs_cam_format,
            frame_0_as_canonical=frame_0_as_canonical,
            split_chunk_size=split_chunk_size,
            mv_input=mv_input,
            append_depth=append_depth,
            append_xyz=append_xyz,
            pcd_path=pcd_path,
            load_pcd=load_pcd,
            read_normal=read_normal,
            mv_latent_dir=mv_latent_dir,
            load_raw=False,
            # shards_folder_num=4,
            # eval=False,
            **kwargs)

        assert not self.load_raw
        # self.scaling_factor = np.array([0.14593576, 0.15753542, 0.18873914])

        self.n_cond_frames = 4 # a easy version for now.
        self.avoid_loading_first = True

    def __getitem__(self, index) -> Any:
        raw_img, c, caption, ins = self.read_chunk(
            os.path.join(self.file_path, self.chunk_list[index]))

        # ! process
        raw_img = torch.from_numpy(raw_img).permute(0, 3, 1, 2) / 255.0  # [0,1]

        if raw_img.shape[-1] != self.reso:
            raw_img = torch.nn.functional.interpolate(
                input=raw_img,
                size=(self.reso, self.reso),
                mode='bilinear',
                align_corners=False,
            )
        img = raw_img * 2 - 1  # as gt

        # ! load latent
        # latent, _ = self._load_latent(ins)

        latent, fps_xyz = self._load_latent(ins, pick_both=True) # analyzing xyz/latent disentangled diffusion
        # latent, fps_xyz = latent[0], fps_xyz[0] # remove batch dim here

        # fps_xyz = fps_xyz / self.scaling_factor # for xyz training
        normalized_fps_xyz = self.normalize_pcd_act(fps_xyz)

        if self.avoid_loading_first: # for training mv model
            index = list(range(1,self.chunk_size//2)) + list(range(self.chunk_size//2+1, self.chunk_size))
            img = img[index]
            c = c[index]
        
        # ! shuffle
        indices = np.random.permutation(img.shape[0])
        img = img[indices]
        c = c[indices]

        aug_img = self.perspective_transformer(img) # create 3D inconsistency

        # ! split along V and repeat other stuffs accordingly
        img = rearrange(img, '(B V) ... -> B V ...', B=2)[:, 0:1] # only return first view (randomly sampled)

        aug_img = rearrange(aug_img, '(B V) ... -> B V ...', B=2)[:, 1:self.n_cond_frames+1]
        c = rearrange(c, '(B V) ... -> B V ...', B=2)[:, 1:self.n_cond_frames+1] # 2 6 25


        # use view-space camera tradition
        c[0] = self.normalize_camera(c[0], c[0,0:1])
        c[1] = self.normalize_camera(c[1], c[1,0:1])

        caption = [caption, caption]
        ins = [ins, ins]

        # ! to dict
        # sample = self.post_process.create_dict_nobatch(sample)
        ret_dict = {
            'caption': caption,
            'ins': ins,
            'c': c,
            'img': img, # fix inp img range to [-1,1]
            'mv_img': aug_img,
            'latent': latent,
            'normalized-fps-xyz': normalized_fps_xyz
            # **latent
        }

        # st()

        return ret_dict


class ChunkObjaverseDatasetDDPMgsMV23DSynthetic(ChunkObjaverseDatasetDDPMgs):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
            four_view_for_latent=False,
            single_view_for_i23d=False,
            load_extra_36_view=False,
            gs_cam_format=False,
            frame_0_as_canonical=True,
            split_chunk_size=10,
            mv_input=True,
            append_depth=False,
            append_xyz=False,
            pcd_path=None,
            load_pcd=False,
            read_normal=False,
            mv_latent_dir='',
            # shards_folder_num=4,
            # eval=False,
            **kwargs):

        super().__init__(
            file_path,
            reso,
            reso_encoder,
            preprocess=preprocess,
            classes=classes,
            load_depth=load_depth,
            test=test,
            scene_scale=scene_scale,
            overfitting=overfitting,
            imgnet_normalize=imgnet_normalize,
            dataset_size=dataset_size,
            overfitting_bs=overfitting_bs,
            interval=interval,
            plucker_embedding=plucker_embedding,
            shuffle_across_cls=shuffle_across_cls,
            wds_split=wds_split,  # 4 splits to accelerate preprocessing
            four_view_for_latent=four_view_for_latent,
            single_view_for_i23d=single_view_for_i23d,
            load_extra_36_view=load_extra_36_view,
            gs_cam_format=gs_cam_format,
            frame_0_as_canonical=frame_0_as_canonical,
            split_chunk_size=split_chunk_size,
            mv_input=mv_input,
            append_depth=append_depth,
            append_xyz=append_xyz,
            pcd_path=pcd_path,
            load_pcd=load_pcd,
            read_normal=read_normal,
            mv_latent_dir=mv_latent_dir,
            load_raw=True,
            load_instance_only=True,
            # shards_folder_num=4,
            # eval=False,
            **kwargs)

        # assert not self.load_raw
        # self.scaling_factor = np.array([0.14593576, 0.15753542, 0.18873914])

        self.n_cond_frames = 6 # a easy version for now.
        self.avoid_loading_first = True
        self.indices = np.array([0,1,2,3,4,5])
        self.img_root_dir = '/cpfs01/user/lanyushi.p/data/unzip4_img'


        azimuths = np.array([30, 90, 150, 210, 270, 330]).astype(float)
        elevations = np.array([20, -10, 20, -10, 20, -10]).astype(float)

        zero123pp_pose, _ = generate_input_camera(1.8, [[elevations[i], azimuths[i]] for i in range(6)], fov=30)
        K = torch.Tensor([1.3889, 0.0000, 0.5000, 0.0000, 1.3889, 0.5000, 0.0000, 0.0000, 0.0039]).to(zero123pp_pose) # keeps the same
        zero123pp_pose = torch.cat([zero123pp_pose.reshape(6,-1), K.unsqueeze(0).repeat(6,1)], dim=-1)


        eval_camera = zero123pp_pose[self.indices].float().cpu().numpy() # for normalization
        self.eval_camera = self.normalize_camera(eval_camera, eval_camera[0:1]) # the first img is not used.

        # self.load_synthetic_only = False
        self.load_synthetic_only = True

    def __len__(self):
        return len(self.rgb_list)
    
    def _getitem_synthetic(self, index) -> Any:

        rgb_fname = Path(self.rgb_list[index])
        # ins = self.mvi_objv_mapping(rgb_fname.parent.parent.stem)

        # ins = str(Path(rgb_fname).parent.parent.stem)

        ins = str((Path(rgb_fname).relative_to(self.file_path)).parent.parent)

        mv_img = imageio.imread(rgb_fname)
        # st()
        mv_img = rearrange(mv_img, '(n h) (m w) c -> (n m) h w c', n=3, m=2)[self.indices]        # (6, 3, 320, 320)
        mv_img = np.stack([recenter(img, np.ones_like(img), border_ratio=0.1) for img in mv_img], axis=0)
        mv_img = rearrange(mv_img, 'b h w c -> b c h w') # to torch tradition
        mv_img = torch.from_numpy(mv_img) / 127.5 - 1

        # ! load single-view image here
        img_idx = self.mvi_objv_mapping[rgb_fname.stem]
        img_path = os.path.join(self.img_root_dir, rgb_fname.parent.relative_to(self.file_path), img_idx, f'{img_idx}.png')

        raw_img = imageio.imread(img_path).astype(np.float32)
        alpha_mask = raw_img[..., -1:] / 255
        raw_img = alpha_mask * raw_img[..., :3] + (
            1 - alpha_mask) * np.ones_like(raw_img[..., :3]) * 255

        raw_img = cv2.resize(raw_img, (self.reso, self.reso), interpolation=cv2.INTER_CUBIC)
        raw_img = torch.from_numpy(raw_img).permute(2,0,1).clip(0,255)  # [0,1]
        img = raw_img / 127.5 - 1

        latent, fps_xyz = self._load_latent(ins, pick_both=False) # analyzing xyz/latent disentangled diffusion
        latent, fps_xyz = latent[0], fps_xyz[0]

        normalized_fps_xyz = self.normalize_pcd_act(fps_xyz) # for stage-1

        # use view-space camera tradition
        # ins = [ins, ins]
        # st()
        caption = self.caption_data['/'.join(ins.split('/')[1:])]

        # ! to dict
        # sample = self.post_process.create_dict_nobatch(sample)
        ret_dict = {
            'caption': caption,
            # 'ins': ins,
            'c': self.eval_camera,
            'img': img, # fix inp img range to [-1,1]
            'mv_img': mv_img,
            'latent': latent,
            'normalized-fps-xyz': normalized_fps_xyz,
            'fps-xyz': fps_xyz,
        }

        return ret_dict


    def _getitem_gt(self, index) -> Any:
        raw_img, c, caption, ins = self.read_chunk(
            os.path.join(self.gt_mv_file_path, self.gt_chunk_list[index]))

        # ! process
        raw_img = torch.from_numpy(raw_img).permute(0, 3, 1, 2) / 255.0  # [0,1]

        if raw_img.shape[-1] != self.reso:
            raw_img = torch.nn.functional.interpolate(
                input=raw_img,
                size=(self.reso, self.reso),
                mode='bilinear',
                align_corners=False,
            )
        img = raw_img * 2 - 1  # as gt

        # ! load latent
        # latent, _ = self._load_latent(ins)

        latent, fps_xyz = self._load_latent(ins, pick_both=True) # analyzing xyz/latent disentangled diffusion
        # latent, fps_xyz = latent[0], fps_xyz[0] # remove batch dim here

        # fps_xyz = fps_xyz / self.scaling_factor # for xyz training
        normalized_fps_xyz = self.normalize_pcd_act(fps_xyz)

        if self.avoid_loading_first: # for training mv model
            index = list(range(1,self.chunk_size//2)) + list(range(self.chunk_size//2+1, self.chunk_size))
            img = img[index]
            c = c[index]
        
        # ! shuffle
        indices = np.random.permutation(img.shape[0])
        img = img[indices]
        c = c[indices]

        # st()
        aug_img = self.mv_resize_cls(img)
        aug_img = self.perspective_transformer(aug_img) # create 3D inconsistency

        # ! split along V and repeat other stuffs accordingly
        img = rearrange(img, '(B V) ... -> B V ...', B=2)[:, 0:1] # only return first view (randomly sampled)

        aug_img = rearrange(aug_img, '(B V) ... -> B V ...', B=2)[:, 1:self.n_cond_frames+1]
        c = rearrange(c, '(B V) ... -> B V ...', B=2)[:, 1:self.n_cond_frames+1] # 2 6 25


        # use view-space camera tradition
        c[0] = self.normalize_camera(c[0], c[0,0:1])
        c[1] = self.normalize_camera(c[1], c[1,0:1])

        caption = [caption, caption]
        ins = [ins, ins]

        # ! to dict
        # sample = self.post_process.create_dict_nobatch(sample)
        ret_dict = {
            'caption': caption,
            'ins': ins,
            'c': c,
            'img': img, # fix inp img range to [-1,1]
            'mv_img': aug_img,
            'latent': latent,
            'normalized-fps-xyz': normalized_fps_xyz,
            'fps-xyz': fps_xyz,
        }

        return ret_dict


    def __getitem__(self, index) -> Any:
        # load synthetic version

        try:
            synthetic_mv = self._getitem_synthetic(index)
        except Exception as e:
            # logger.log(Path(self.rgb_list[index]), 'missing')
            synthetic_mv = self._getitem_synthetic(random.randint(0, len(self.rgb_list)//2))

        if self.load_synthetic_only:
            return synthetic_mv

        else:
            # load gt mv chunk
            gt_chunk_index = random.randint(0, len(self.gt_chunk_list)-1)
            gt_mv = self._getitem_gt(gt_chunk_index)

            # merge them together along batch dim
            merged_mv = {}
            for k, v in synthetic_mv.items(): # merge, synthetic - gt order
                if k not in ['caption', 'ins']:
                    if k == 'img':
                        merged_mv[k] = np.concatenate([v[None], gt_mv[k][:, 0]], axis=0).astype(np.float32)
                    else:
                        merged_mv[k] = np.concatenate([v[None], gt_mv[k]], axis=0).astype(np.float32)
                else:
                    merged_mv[k] = [v] + gt_mv[k] # list

            return merged_mv





class ChunkObjaverseDatasetDDPMgsI23D_loadMV(ChunkObjaverseDatasetDDPMgs):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
            four_view_for_latent=False,
            single_view_for_i23d=False,
            load_extra_36_view=False,
            gs_cam_format=False,
            frame_0_as_canonical=True,
            split_chunk_size=10,
            mv_input=True,
            append_depth=False,
            append_xyz=False,
            pcd_path=None,
            load_pcd=False,
            read_normal=False,
            mv_latent_dir='',
            canonicalize_pcd=False,
            # shards_folder_num=4,
            # eval=False,
            **kwargs):

        super().__init__(
            file_path,
            reso,
            reso_encoder,
            preprocess=preprocess,
            classes=classes,
            load_depth=load_depth,
            test=test,
            scene_scale=scene_scale,
            overfitting=overfitting,
            imgnet_normalize=imgnet_normalize,
            dataset_size=dataset_size,
            overfitting_bs=overfitting_bs,
            interval=interval,
            plucker_embedding=plucker_embedding,
            shuffle_across_cls=shuffle_across_cls,
            wds_split=wds_split,  # 4 splits to accelerate preprocessing
            four_view_for_latent=four_view_for_latent,
            single_view_for_i23d=single_view_for_i23d,
            load_extra_36_view=load_extra_36_view,
            gs_cam_format=gs_cam_format,
            frame_0_as_canonical=frame_0_as_canonical,
            split_chunk_size=split_chunk_size,
            mv_input=mv_input,
            append_depth=append_depth,
            append_xyz=append_xyz,
            pcd_path=pcd_path,
            load_pcd=load_pcd,
            read_normal=read_normal,
            mv_latent_dir=mv_latent_dir,
            load_raw=False,
            # shards_folder_num=4,
            # eval=False,
            **kwargs)

        assert not self.load_raw
        # self.scaling_factor = np.array([0.14593576, 0.15753542, 0.18873914])

        self.n_cond_frames = 5 # a easy version for now.
        self.avoid_loading_first = True

        # self.canonicalize_pcd = canonicalize_pcd
        # self.canonicalize_pcd = True
        self.canonicalize_pcd = False

    def canonicalize_xyz(self, c, pcd):

        B = c.shape[0]
        camera_poses_rot = c[:, :16].reshape(B, 4, 4)[:, :3, :3]

        R_inv = np.transpose(camera_poses_rot, (0,2,1)) # w2c rotation

        new_pcd = (R_inv @ np.transpose(pcd, (0,2,1))) # B 3 3 @ B 3 N
        new_pcd = np.transpose(new_pcd, (0,2,1))

        return new_pcd


    def __getitem__(self, index) -> Any:
        raw_img, c, caption, ins = self.read_chunk(
            os.path.join(self.file_path, self.chunk_list[index]))

        # ! process
        raw_img = torch.from_numpy(raw_img).permute(0, 3, 1, 2) / 255.0  # [0,1]

        if raw_img.shape[-1] != self.reso:
            raw_img = torch.nn.functional.interpolate(
                input=raw_img,
                size=(self.reso, self.reso),
                mode='bilinear',
                align_corners=False,
            )
        img = raw_img * 2 - 1  # as gt

        # ! load latent
        # latent, _ = self._load_latent(ins)

        if self.avoid_loading_first: # for training mv model
            index = list(range(1,self.chunk_size//2)) + list(range(self.chunk_size//2+1, self.chunk_size))
            img = img[index]
            c = c[index]
        
        # ! shuffle
        indices = np.random.permutation(img.shape[0])[:self.n_cond_frames*2]
        img = img[indices]
        c = c[indices]

        latent, fps_xyz = self._load_latent(ins, pick_both=True) # analyzing xyz/latent disentangled diffusion
        # latent, fps_xyz = latent[0], fps_xyz[0] # remove batch dim here

        fps_xyz = np.repeat(fps_xyz, self.n_cond_frames, 0)
        latent = np.repeat(latent, self.n_cond_frames, 0)
        normalized_fps_xyz = self.normalize_pcd_act(fps_xyz)

        if self.canonicalize_pcd:
            normalized_fps_xyz = self.canonicalize_xyz(c, normalized_fps_xyz)

        # repeat
        caption = [caption] * self.n_cond_frames * 2
        ins = [ins] * self.n_cond_frames * 2

        ret_dict = {
            'caption': caption,
            'ins': ins,
            'c': c,
            'img': img, # fix inp img range to [-1,1]
            'latent': latent,
            'normalized-fps-xyz': normalized_fps_xyz,
            'fps-xyz': fps_xyz,
            # **latent
        }

        return ret_dict


class RealDataset(Dataset):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
    ) -> None:
        super().__init__()

        self.file_path = file_path
        self.overfitting = overfitting
        self.scene_scale = scene_scale
        self.reso = reso
        self.reso_encoder = reso_encoder
        self.classes = False
        self.load_depth = load_depth
        self.preprocess = preprocess
        self.plucker_embedding = plucker_embedding

        self.rgb_list = []

        all_fname = [
            t for t in os.listdir(self.file_path)
            if t.split('.')[1] in ['png', 'jpg']
        ]

        all_fname = [name for name in all_fname if '-input' in name ]

        self.rgb_list += ([
            os.path.join(self.file_path, fname) for fname in all_fname
        ])

        # st()

        # if len(self.rgb_list) == 1:
        #     # placeholder
        #     self.rgb_list = self.rgb_list * 40

        # ! setup normalizataion
        transformations = [
            transforms.ToTensor(),  # [0,1] range
        ]

        assert imgnet_normalize
        if imgnet_normalize:
            transformations.append(
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))  # type: ignore
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)))  # type: ignore

        self.normalize = transforms.Compose(transformations)
        # camera = torch.load('eval_pose.pt', map_location='cpu')
        # self.eval_camera = camera

        # pre-cache
        # self.calc_rays_plucker()

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, index) -> Any:
        # return super().__getitem__(index)

        rgb_fname = self.rgb_list[index]
        # ! preprocess, normalize

        raw_img = imageio.imread(rgb_fname)

        # interpolation=cv2.INTER_AREA)
        if raw_img.shape[-1] == 4:
            alpha_mask = raw_img[..., 3:4] / 255.0
            bg_white = np.ones_like(alpha_mask) * 255.0
            raw_img = raw_img[..., :3] * alpha_mask + (
                1 - alpha_mask) * bg_white  #[3, reso_encoder, reso_encoder]
            raw_img = raw_img.astype(np.uint8)

        # raw_img = recenter(raw_img, np.ones_like(raw_img), border_ratio=0.2)

        # log gt
        img = cv2.resize(raw_img, (self.reso, self.reso),
                         interpolation=cv2.INTER_LANCZOS4)

        img = torch.from_numpy(img)[..., :3].permute(
            2, 0, 1
        ) / 127.5 - 1  #[3, reso, reso], normalize to [-1,1], follow triplane range

        ret_dict = {
            # 'rgb_fname': rgb_fname,
            # 'img_to_encoder':
            # img_to_encoder.unsqueeze(0).repeat_interleave(40, 0),
            'img': img,
            # 'c': self.eval_camera,  # TODO, get pre-calculated samples
            # 'ins': 'placeholder',
            # 'bbox': 'placeholder',
            # 'caption': 'placeholder',
        }

        # ! repeat as a intance

        return ret_dict




class RealDataset_GSO(Dataset):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
    ) -> None:
        super().__init__()

        self.file_path = file_path
        self.overfitting = overfitting
        self.scene_scale = scene_scale
        self.reso = reso
        self.reso_encoder = reso_encoder
        self.classes = False
        self.load_depth = load_depth
        self.preprocess = preprocess
        self.plucker_embedding = plucker_embedding

        self.rgb_list = []



        # ! for gso-rendering
        all_objs = os.listdir(self.file_path)
        all_objs.sort()

        if True: # instant-mesh picked images
        # if False:
            all_instances = os.listdir(self.file_path) 
            # all_fname = [
            #     t for t in all_instances
            #     if t.split('.')[1] in ['png', 'jpg']
            # ]

            # all_fname = [name for name in all_fname if '-input' in name ]

            # all_fname = ['house2-input.png', 'plant-input.png']
            all_fname = ['house2-input.png']

            self.rgb_list = [os.path.join(self.file_path, name) for name in all_fname]

        if False:
            for obj_folder in tqdm(all_objs[515:]):
            # for obj_folder in tqdm(all_objs[:515]):
            # for obj_folder in tqdm(all_objs[:]):
            # for obj_folder in tqdm(sorted(os.listdir(self.file_path))[515:]):
                # for idx in range(0,25,5):
                for idx in [0]: # only query frontal view is enough
                    self.rgb_list.append(os.path.join(self.file_path, obj_folder, 'rgba', f'{idx:03d}.png')) 


        # for free-3d rendering
        if False:
        # if True:
            # all_instances = sorted(os.listdir(self.file_path))

            all_instances = ['BAGEL_WITH_CHEESE',
                'BALANCING_CACTUS',
                'Baby_Elements_Stacking_Cups',
                'Breyer_Horse_Of_The_Year_2015',
                'COAST_GUARD_BOAT',
                'CONE_SORTING',
                'CREATIVE_BLOCKS_35_MM',
                'Cole_Hardware_Mini_Honey_Dipper',
                'FAIRY_TALE_BLOCKS',
                'FIRE_ENGINE',
                'FOOD_BEVERAGE_SET',
                'GEOMETRIC_PEG_BOARD',
                'Great_Dinos_Triceratops_Toy',
                'JUICER_SET',
                'STACKING_BEAR',
                'STACKING_RING',
                'Schleich_African_Black_Rhino']

            for instance in all_instances:
                self.rgb_list += ([
                    # os.path.join(self.file_path, instance, 'rgb', f'{fname:06d}.png') for fname in range(0,250,50)
                    # os.path.join(self.file_path, instance, 'rgb', f'{fname:06d}.png') for fname in range(0,250,100)
                    # os.path.join(self.file_path, instance, f'{fname:03d}.png') for fname in range(0,25,5)
                    os.path.join(self.file_path, instance, 'render_mvs_25', 'model', f'{fname:03d}.png') for fname in range(0,25,4)
                ])

        # if True: # g-objv animals images for i23d eval
        if False:
        # if True:
            objv_dataset = '/mnt/sfs-common/yslan/Dataset/Obajverse/chunk-jpeg-normal/bs_16_fixsave3/170K/512/'
            dataset_json = os.path.join(objv_dataset, 'dataset.json')
            with open(dataset_json, 'r') as f:
                dataset_json = json.load(f)

            # all_objs = dataset_json['Animals'][::3][:6250]
            all_objs = dataset_json['Animals'][::3][1100:2200][:600]

            for obj_folder in tqdm(all_objs[:]):
                for idx in [0]: # only query frontal view is enough
                    self.rgb_list.append(os.path.join(self.file_path, obj_folder, f'{idx}.jpg')) 
                
        
        # ! setup normalizataion
        transformations = [
            transforms.ToTensor(),  # [0,1] range
        ]

        assert imgnet_normalize
        if imgnet_normalize:
            transformations.append(
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))  # type: ignore
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)))  # type: ignore

        self.normalize = transforms.Compose(transformations)
        # camera = torch.load('eval_pose.pt', map_location='cpu')
        # self.eval_camera = camera

        # pre-cache
        # self.calc_rays_plucker()

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, index) -> Any:
        # return super().__getitem__(index)

        rgb_fname = self.rgb_list[index]
        # ! preprocess, normalize

        raw_img = imageio.imread(rgb_fname)

        # interpolation=cv2.INTER_AREA)
        if raw_img.shape[-1] == 4:
            alpha_mask = raw_img[..., 3:4] / 255.0
            bg_white = np.ones_like(alpha_mask) * 255.0
            raw_img = raw_img[..., :3] * alpha_mask + (
                1 - alpha_mask) * bg_white  #[3, reso_encoder, reso_encoder]
            raw_img = raw_img.astype(np.uint8)

        # raw_img = recenter(raw_img, np.ones_like(raw_img), border_ratio=0.2)

        # log gt
        img = cv2.resize(raw_img, (self.reso, self.reso),
                         interpolation=cv2.INTER_LANCZOS4)

        img = torch.from_numpy(img)[..., :3].permute(
            2, 0, 1
        ) / 127.5 - 1  #[3, reso, reso], normalize to [-1,1], follow triplane range

        ret_dict = {
            'img': img,
            # 'ins': str(Path(rgb_fname).parent.parent.stem), # for gso-rendering
            'ins': str(Path(rgb_fname).relative_to(self.file_path)), # for gso-rendering
            # 'ins': rgb_fname,
        }

        return ret_dict



class RealMVDataset(Dataset):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
    ) -> None:
        super().__init__()

        self.file_path = file_path
        self.overfitting = overfitting
        self.scene_scale = scene_scale
        self.reso = reso
        self.reso_encoder = reso_encoder
        self.classes = False
        self.load_depth = load_depth
        self.preprocess = preprocess
        self.plucker_embedding = plucker_embedding

        self.rgb_list = []

        all_fname = [
            t for t in os.listdir(self.file_path)
            if t.split('.')[1] in ['png', 'jpg']
        ]
        all_fname = [name for name in all_fname if '-input' in name ]
        # all_fname = [name for name in all_fname if 'sorting_board-input' in name ]
        # all_fname = [name for name in all_fname if 'teasure_chest-input' in name ]
        # all_fname = [name for name in all_fname if 'bubble_mart_blue-input' in name ]
        # all_fname = [name for name in all_fname if 'chair_comfort-input' in name ]
        self.rgb_list += ([
            os.path.join(self.file_path, fname) for fname in all_fname
        ])
        # if len(self.rgb_list) == 1:
        #     # placeholder
        #     self.rgb_list = self.rgb_list * 40

        # ! setup normalizataion
        transformations = [
            transforms.ToTensor(),  # [0,1] range
        ]

        azimuths = np.array([30, 90, 150, 210, 270, 330]).astype(float)
        elevations = np.array([20, -10, 20, -10, 20, -10]).astype(float)

        # zero123pp_pose, _ = generate_input_camera(1.6, [[elevations[i], azimuths[i]] for i in range(6)], fov=30)
        zero123pp_pose, _ = generate_input_camera(1.8, [[elevations[i], azimuths[i]] for i in range(6)], fov=30)
        K = torch.Tensor([1.3889, 0.0000, 0.5000, 0.0000, 1.3889, 0.5000, 0.0000, 0.0000, 0.0039]).to(zero123pp_pose) # keeps the same
        # st()
        zero123pp_pose = torch.cat([zero123pp_pose.reshape(6,-1), K.unsqueeze(0).repeat(6,1)], dim=-1)

        # ! directly adopt gt input 
        # self.indices = np.array([0,2,4,5])
        # eval_camera = zero123pp_pose[self.indices]
        # self.eval_camera = torch.cat([torch.zeros_like(eval_camera[0:1]),eval_camera], 0) # first c not used as condition here, just placeholder

        # ! adopt mv-diffusion output as input.
        # self.indices = np.array([1,0,2,4,5])
        self.indices = np.array([0,1,2,3,4,5])
        eval_camera = zero123pp_pose[self.indices].float().cpu().numpy() # for normalization

        # eval_camera = zero123pp_pose[self.indices]
        # self.eval_camera = eval_camera
        # self.eval_camera = torch.cat([torch.zeros_like(eval_camera[0:1]),eval_camera], 0) # first c not used as condition here, just placeholder

        # # * normalize here
        self.eval_camera = self.normalize_camera(eval_camera, eval_camera[0:1]) # the first img is not used. 

        # self.mv_resize_cls = torchvision.transforms.Resize(320, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
        #         max_size=None, antialias=True)

    def normalize_camera(self, c, c_frame0):
        # assert c.shape[0] == self.chunk_size  # 8 o r10

        B = c.shape[0]
        camera_poses = c[:, :16].reshape(B, 4, 4)  # 3x4
        canonical_camera_poses = c_frame0[:, :16].reshape(1, 4, 4)
        inverse_canonical_pose = np.linalg.inv(canonical_camera_poses)
        inverse_canonical_pose = np.repeat(inverse_canonical_pose, B, 0)

        cam_radius = np.linalg.norm(
            c_frame0[:, :16].reshape(1, 4, 4)[:, :3, 3],
            axis=-1,
            keepdims=False)  # since g-buffer adopts dynamic radius here.

        frame1_fixed_pos = np.repeat(np.eye(4)[None], 1, axis=0)
        frame1_fixed_pos[:, 2, -1] = -cam_radius

        transform = frame1_fixed_pos @ inverse_canonical_pose

        new_camera_poses = np.repeat(
            transform, 1, axis=0
        ) @ camera_poses  # [V, 4, 4]. np.repeat() is th.repeat_interleave()

        c = np.concatenate([new_camera_poses.reshape(B, 16), c[:, 16:]],
                           axis=-1)

        return c

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, index) -> Any:
        # return super().__getitem__(index)

        rgb_fname = self.rgb_list[index]

        raw_img = imageio.imread(rgb_fname)[..., :3]
        raw_img = cv2.resize(raw_img, (self.reso, self.reso), interpolation=cv2.INTER_CUBIC)
        raw_img = torch.from_numpy(raw_img).permute(2,0,1).clip(0,255)  # [0,1]
        img = raw_img / 127.5 - 1


        # ! if loading mv-diff output views
        mv_img = imageio.imread(rgb_fname.replace('-input', ''))
        mv_img = rearrange(mv_img, '(n h) (m w) c -> (n m) h w c', n=3, m=2)[self.indices]        # (6, 3, 320, 320)
        mv_img = np.stack([recenter(img, np.ones_like(img), border_ratio=0.1) for img in mv_img], axis=0)
        mv_img = rearrange(mv_img, 'b h w c -> b c h w') # to torch tradition
        mv_img = torch.from_numpy(mv_img) / 127.5 - 1

        ret_dict = {
            'img': img,
            'mv_img': mv_img,
            'c': self.eval_camera,
            'caption': 'null',
        }

        return ret_dict





class NovelViewObjverseDataset(MultiViewObjverseDataset):
    """novel view prediction version.
    """

    def __init__(self,
                 file_path,
                 reso,
                 reso_encoder,
                 preprocess=None,
                 classes=False,
                 load_depth=False,
                 test=False,
                 scene_scale=1,
                 overfitting=False,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 overfitting_bs=-1,
                 **kwargs):
        super().__init__(file_path, reso, reso_encoder, preprocess, classes,
                         load_depth, test, scene_scale, overfitting,
                         imgnet_normalize, dataset_size, overfitting_bs,
                         **kwargs)

    def __getitem__(self, idx):
        input_view = super().__getitem__(
            idx)  # get previous input view results

        # get novel view of the same instance
        novel_view = super().__getitem__(
            (idx // self.instance_data_length) * self.instance_data_length +
            random.randint(0, self.instance_data_length - 1))

        # assert input_view['ins_name'] == novel_view['ins_name'], 'should sample novel view from the same instance'

        input_view.update({f'nv_{k}': v for k, v in novel_view.items()})
        return input_view


class MultiViewObjverseDatasetforLMDB(MultiViewObjverseDataset):

    def __init__(
        self,
        file_path,
        reso,
        reso_encoder,
        preprocess=None,
        classes=False,
        load_depth=False,
        test=False,
        scene_scale=1,
        overfitting=False,
        imgnet_normalize=True,
        dataset_size=-1,
        overfitting_bs=-1,
        shuffle_across_cls=False,
        wds_split=1,
        four_view_for_latent=False,
    ):
        super().__init__(file_path,
                         reso,
                         reso_encoder,
                         preprocess,
                         classes,
                         load_depth,
                         test,
                         scene_scale,
                         overfitting,
                         imgnet_normalize,
                         dataset_size,
                         overfitting_bs,
                         shuffle_across_cls=shuffle_across_cls,
                         wds_split=wds_split,
                         four_view_for_latent=four_view_for_latent)

        # assert self.reso == 256
        self.load_caption = True

        with open(
                # '/cpfs01/shared/V2V/V2V_hdd/yslan/aigc3d/text_captions_cap3d.json'
                '/nas/shared/public/yslan/data/text_captions_cap3d.json') as f:
                # '/nas/shared/V2V/yslan/aigc3d/text_captions_cap3d.json') as f:
            self.caption_data = json.load(f)
        # lmdb_path = '/cpfs01/user/yangpeiqing.p/yslan/data/Furnitures_uncompressed/'

        # with open(os.path.join(lmdb_path, 'idx_to_ins_mapping.json')) as f:
        #     self.idx_to_ins_mapping = json.load(f)

    def __len__(self):
        return super().__len__()
        # return 100 # for speed debug

    def quantize_depth(self, depth):
        # https://developers.google.com/depthmap-metadata/encoding
        # RangeInverse encoding
        bg = depth == 0
        depth[bg] = 3  # no need to allocate capacity to it
        disparity = 1 / depth

        far = disparity.max().item()  # np array here
        near = disparity.min().item()

        # d_normalized = (far * (depth-near) / (depth * far - near)) # [0,1] range
        d_normalized = (disparity - near) / (far - near)  # [0,1] range
        # imageio.imwrite('depth_negative.jpeg', (((depth - near) / (far - near) * 255)<0).numpy().astype(np.uint8))
        # imageio.imwrite('depth_negative.jpeg', ((depth <0)*255).numpy().astype(np.uint8))
        d_normalized = np.nan_to_num(d_normalized.cpu().numpy())
        d_normalized = (np.clip(d_normalized, 0, 1) * 255).astype(np.uint8)
        # imageio.imwrite('depth.png', d_normalized)

        # d = 1 / ( (d_normalized / 255) * (far-near) + near)
        # diff = (d[~bg.numpy()] - depth[~bg].numpy()).sum()

        return d_normalized, near, far  # return disp

    def __getitem__(self, idx):
        # ret_dict = super().__getitem__(idx)
        rgb_fname = self.rgb_list[idx]
        pose_fname = self.pose_list[idx]
        raw_img = imageio.imread(rgb_fname)  # [..., :3]

        assert raw_img.shape[-1] == 4

        # st() # cv2.imwrite('img_CV2_90.jpg', a, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        # if raw_img.shape[-1] == 4:  # ! set bg to white

        alpha_mask = raw_img[..., -1:] / 255  # [0,1]

        raw_img = alpha_mask * raw_img[..., :3] + (
            1 - alpha_mask) * np.ones_like(raw_img[..., :3]) * 255

        raw_img = np.concatenate([raw_img, alpha_mask * 255], -1)
        raw_img = raw_img.astype(np.uint8)

        raw_img = cv2.resize(raw_img, (self.reso, self.reso),
                             interpolation=cv2.INTER_LANCZOS4)
        alpha_mask = raw_img[..., -1] / 255
        raw_img = raw_img[..., :3]

        # alpha_mask = cv2.resize(alpha_mask, (self.reso, self.reso),
        #                         interpolation=cv2.INTER_LANCZOS4)

        c2w = read_camera_matrix_single(pose_fname)  #[1, 4, 4] -> [1, 16]
        c = np.concatenate([c2w.reshape(16), self.intrinsics],
                           axis=0).reshape(25).astype(
                               np.float32)  # 25, no '1' dim needed.
        c = torch.from_numpy(c)
        # c = np.concatenate([c2w, self.intrinsics], axis=0).reshape(25)  # 25, no '1' dim needed.

        # if self.load_depth:
        # depth, depth_mask, depth_mask_sr = read_dnormal(self.depth_list[idx],
        # try:
        depth, normal = read_dnormal(self.depth_list[idx], c2w[:3, 3:],
                                     self.reso, self.reso)

        # ! quantize depth for fast decoding
        # d_normalized, d_near, d_far = self.quantize_depth(depth)

        # ! add frame_0 alignment

        # try:

        ins = str(
            (Path(self.data_ins_list[idx]).relative_to(self.file_path)).parent)
        # if self.shuffle_across_cls:
        if self.load_caption:
            caption = self.caption_data['/'.join(ins.split('/')[1:])]
            bbox = self.load_bbox(torch.from_numpy(alpha_mask) > 0)
        else:
            caption = '' # since in g-alignment-xl, some instances will fail.
            bbox = self.load_bbox(torch.from_numpy(np.ones_like(alpha_mask)) > 0)

        # else:
        #     caption = self.caption_data[ins]

        ret_dict = {
            'normal': normal,
            'raw_img': raw_img,
            'c': c,
            # 'depth_mask': depth_mask, # 64x64 here?
            'bbox': bbox,
            'ins': ins,
            'caption': caption,
            'alpha_mask': alpha_mask,
            'depth': depth,  # return for pcd creation
            # 'd_normalized': d_normalized,
            # 'd_near': d_near,
            # 'd_far': d_far,
            # 'fname': rgb_fname,
        }
        return ret_dict


class MultiViewObjverseDatasetforLMDB_nocaption(MultiViewObjverseDatasetforLMDB):

    def __init__(
        self,
        file_path,
        reso,
        reso_encoder,
        preprocess=None,
        classes=False,
        load_depth=False,
        test=False,
        scene_scale=1,
        overfitting=False,
        imgnet_normalize=True,
        dataset_size=-1,
        overfitting_bs=-1,
        shuffle_across_cls=False,
        wds_split=1,
        four_view_for_latent=False,
    ):
        super().__init__(file_path,
                         reso,
                         reso_encoder,
                         preprocess,
                         classes,
                         load_depth,
                         test,
                         scene_scale,
                         overfitting,
                         imgnet_normalize,
                         dataset_size,
                         overfitting_bs,
                         shuffle_across_cls=shuffle_across_cls,
                         wds_split=wds_split,
                         four_view_for_latent=four_view_for_latent)

        self.load_caption = False


class Objv_LMDBDataset_MV_Compressed(LMDBDataset_MV_Compressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 test=False,
                 **kwargs):
        super().__init__(lmdb_path,
                         reso,
                         reso_encoder,
                         imgnet_normalize,
                         dataset_size=dataset_size,
                         **kwargs)
        self.instance_data_length = 40  # ! could save some key attributes in LMDB
        if test:
            self.length = self.instance_data_length
        elif dataset_size > 0:
            self.length = dataset_size * self.instance_data_length

        # load caption data, and idx-to-ins mapping
        with open(
                '/cpfs01/shared/V2V/V2V_hdd/yslan/aigc3d/text_captions_cap3d.json'
        ) as f:
            self.caption_data = json.load(f)
        with open(os.path.join(lmdb_path, 'idx_to_ins_mapping.json')) as f:
            self.idx_to_ins_mapping = json.load(f)

    def _load_data(self, idx):
        # '''
        raw_img, depth, c, bbox = self._load_lmdb_data(idx)
        # raw_img, depth, c, bbox  = self._load_lmdb_data_no_decompress(idx)

        # resize depth and bbox
        caption = self.caption_data[self.idx_to_ins_mapping[str(idx)]]

        return {
            **self._post_process_sample(raw_img, depth),
            'c': c,
            'bbox': (bbox * (self.reso / 512.0)).astype(np.uint8),
            # 'bbox': (bbox*(self.reso/256.0)).astype(np.uint8), # TODO, double check 512 in wds?
            'caption': caption
        }
        # '''
        # raw_img, depth, c, bbox  = self._load_lmdb_data_no_decompress(idx)
        # st()
        # return {}

    def __getitem__(self, idx):
        return self._load_data(idx)


class Objv_LMDBDataset_MV_NoCompressed(Objv_LMDBDataset_MV_Compressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 test=False,
                 **kwargs):
        super().__init__(lmdb_path, reso, reso_encoder, imgnet_normalize,
                         dataset_size, test, **kwargs)

    def _load_data(self, idx):
        # '''
        raw_img, depth, c, bbox = self._load_lmdb_data_no_decompress(idx)

        # resize depth and bbox
        caption = self.caption_data[self.idx_to_ins_mapping[str(idx)]]

        return {
            **self._post_process_sample(raw_img, depth), 'c': c,
            'bbox': (bbox * (self.reso / 512.0)).astype(np.uint8),
            'caption': caption
        }
        return {}


class Objv_LMDBDataset_NV_NoCompressed(Objv_LMDBDataset_MV_NoCompressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 test=False,
                 **kwargs):
        super().__init__(lmdb_path, reso, reso_encoder, imgnet_normalize,
                         dataset_size, test, **kwargs)

    def __getitem__(self, idx):
        input_view = self._load_data(idx)  # get previous input view results

        # get novel view of the same instance
        try:
            novel_view = self._load_data(
                (idx // self.instance_data_length) *
                self.instance_data_length +
                random.randint(0, self.instance_data_length - 1))
        except Exception as e:
            raise NotImplementedError(idx)

        # assert input_view['ins_name'] == novel_view['ins_name'], 'should sample novel view from the same instance'

        input_view.update({f'nv_{k}': v for k, v in novel_view.items()})
        return input_view


class Objv_LMDBDataset_MV_Compressed_for_lmdb(LMDBDataset_MV_Compressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 test=False,
                 **kwargs):
        super().__init__(lmdb_path,
                         reso,
                         reso_encoder,
                         imgnet_normalize,
                         dataset_size=dataset_size,
                         **kwargs)
        self.instance_data_length = 40  # ! could save some key attributes in LMDB
        if test:
            self.length = self.instance_data_length
        elif dataset_size > 0:
            self.length = dataset_size * self.instance_data_length

        # load caption data, and idx-to-ins mapping
        with open(
                '/cpfs01/shared/V2V/V2V_hdd/yslan/aigc3d/text_captions_cap3d.json'
        ) as f:
            self.caption_data = json.load(f)
        with open(os.path.join(lmdb_path, 'idx_to_ins_mapping.json')) as f:
            self.idx_to_ins_mapping = json.load(f)

    # def _load_data(self, idx):
    #     # '''
    #     raw_img, depth, c, bbox  = self._load_lmdb_data(idx)

    #     # resize depth and bbox
    #     caption = self.caption_data[self.idx_to_ins_mapping[str(idx)]]

    #     # st()

    #     return {
    #         **self._post_process_sample(raw_img, depth), 'c': c,
    #         'bbox': (bbox*(self.reso/512.0)).astype(np.uint8),
    #         'caption': caption
    #     }
    #     # '''
    #     # raw_img, depth, c, bbox  = self._load_lmdb_data_no_decompress(idx)
    #     # st()
    #     # return {}

    def load_bbox(self, mask):
        nonzero_value = torch.nonzero(mask)
        height, width = nonzero_value.max(dim=0)[0]
        top, left = nonzero_value.min(dim=0)[0]
        bbox = torch.tensor([top, left, height, width], dtype=torch.float32)
        return bbox

    def __getitem__(self, idx):
        raw_img, depth, c, bbox = self._load_lmdb_data(idx)
        return {'raw_img': raw_img, 'depth': depth, 'c': c, 'bbox': bbox}


class Objv_LMDBDataset_NV_Compressed(Objv_LMDBDataset_MV_Compressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 **kwargs):
        super().__init__(lmdb_path, reso, reso_encoder, imgnet_normalize,
                         dataset_size, **kwargs)

    def __getitem__(self, idx):
        input_view = self._load_data(idx)  # get previous input view results

        # get novel view of the same instance
        try:
            novel_view = self._load_data(
                (idx // self.instance_data_length) *
                self.instance_data_length +
                random.randint(0, self.instance_data_length - 1))
        except Exception as e:
            raise NotImplementedError(idx)

        # assert input_view['ins_name'] == novel_view['ins_name'], 'should sample novel view from the same instance'

        input_view.update({f'nv_{k}': v for k, v in novel_view.items()})
        return input_view


#


# test tar loading
def load_wds_ResampledShard(file_path,
                            batch_size,
                            num_workers,
                            reso,
                            reso_encoder,
                            test=False,
                            preprocess=None,
                            imgnet_normalize=True,
                            plucker_embedding=False,
                            decode_encode_img_only=False,
                            load_instance=False,
                            mv_input=False,
                            split_chunk_input=False,
                            duplicate_sample=True,
                            append_depth=False,
                            append_normal=False,
                            gs_cam_format=False,
                            orthog_duplicate=False,
                            **kwargs):

    #     return raw_img, depth, c, bbox, sample_pyd['ins.pyd'], sample_pyd['fname.pyd']

    post_process_cls = PostProcess(
        reso,
        reso_encoder,
        imgnet_normalize=imgnet_normalize,
        plucker_embedding=plucker_embedding,
        decode_encode_img_only=decode_encode_img_only,
        mv_input=mv_input,
        split_chunk_input=split_chunk_input,
        duplicate_sample=duplicate_sample,
        append_depth=append_depth,
        gs_cam_format=gs_cam_format,
        orthog_duplicate=orthog_duplicate,
        append_normal=append_normal,
    )

    # ! add shuffling

    if isinstance(file_path, list):  # lst of shard urls
        all_shards = []
        for url_path in file_path:
            all_shards.extend(wds.shardlists.expand_source(url_path))
        logger.log('all_shards', all_shards)
    else:
        all_shards = file_path  # to be expanded

    if not load_instance:  # during reconstruction training, load pair
        if not split_chunk_input:
            dataset = wds.DataPipeline(
                wds.ResampledShards(all_shards),  # url_shard
                # at this point we have an iterator over all the shards
                wds.shuffle(50),
                wds.split_by_worker,  # if multi-node
                wds.tarfile_to_samples(),
                # add wds.split_by_node here if you are using multiple nodes
                wds.shuffle(
                    1000
                ),  # shuffles in the memory, leverage large RAM for more efficient loading
                wds.decode(wds.autodecode.basichandlers),  # TODO
                wds.to_tuple(
                    "sample.pyd"),  # extract the pyd from top level dict
                wds.map(post_process_cls.decode_zip),
                wds.map(post_process_cls.paired_post_process
                        ),  # create input-novelview paired samples
                # wds.map(post_process_cls._post_process_sample),
                # wds.detshuffle(1000),  # shuffles in the memory, leverage large RAM for more efficient loading
                wds.batched(
                    16,
                    partial=True,
                    # collation_fn=collate
                )  # streaming more data at once, and rebatch later
            )

        elif load_gzip:  # deprecated, no performance improve

            dataset = wds.DataPipeline(
                wds.ResampledShards(all_shards),  # url_shard
                # at this point we have an iterator over all the shards
                wds.shuffle(10),
                wds.split_by_worker,  # if multi-node
                wds.tarfile_to_samples(),
                # add wds.split_by_node here if you are using multiple nodes
                # wds.shuffle(
                #     100
                # ),  # shuffles in the memory, leverage large RAM for more efficient loading
                wds.decode('rgb8'),  # TODO
                # wds.decode(wds.autodecode.basichandlers),  # TODO
                # wds.to_tuple('raw_img.jpeg', 'depth.jpeg', 'alpha_mask.jpeg',
                #              'd_near.npy', 'd_far.npy', "c.npy", 'bbox.npy',
                #              'ins.txt', 'caption.txt'),
                wds.to_tuple('raw_img.png', 'depth_alpha.png'),
                # wds.to_tuple('raw_img.jpg', "c.npy", 'bbox.npy', 'depth.pyd', 'ins.txt', 'caption.txt'),
                # wds.to_tuple('raw_img.jpg', "c.npy", 'bbox.npy', 'ins.txt', 'caption.txt'),
                wds.map(post_process_cls.decode_gzip),
                # wds.map(post_process_cls.paired_post_process_chunk
                #         ),  # create input-novelview paired samples
                wds.batched(
                    20,
                    partial=True,
                    # collation_fn=collate
                )  # streaming more data at once, and rebatch later
            )

        else:
            dataset = wds.DataPipeline(
                wds.ResampledShards(all_shards),  # url_shard
                # at this point we have an iterator over all the shards
                wds.shuffle(100),
                wds.split_by_worker,  # if multi-node
                wds.tarfile_to_samples(),
                # add wds.split_by_node here if you are using multiple nodes
                wds.shuffle(
                    4000 // split_chunk_size
                ),  # shuffles in the memory, leverage large RAM for more efficient loading
                wds.decode(wds.autodecode.basichandlers),  # TODO
                wds.to_tuple(
                    "sample.pyd"),  # extract the pyd from top level dict
                wds.map(post_process_cls.decode_zip),
                wds.map(post_process_cls.paired_post_process_chunk
                        ),  # create input-novelview paired samples
                # wds.map(post_process_cls._post_process_sample),
                # wds.detshuffle(1000),  # shuffles in the memory, leverage large RAM for more efficient loading
                wds.batched(
                    120 // split_chunk_size,
                    partial=True,
                    # collation_fn=collate
                )  # streaming more data at once, and rebatch later
            )

        loader_shard = wds.WebLoader(
            dataset,
            num_workers=num_workers,
            drop_last=False,
            batch_size=None,
            shuffle=False,
            persistent_workers=num_workers > 0).unbatched().shuffle(
                1000 // split_chunk_size).batched(batch_size).map(
                    post_process_cls.create_dict)

        if mv_input:
            loader_shard = loader_shard.map(post_process_cls.prepare_mv_input)

    else:  # load single instance during test/eval
        assert batch_size == 1

        dataset = wds.DataPipeline(
            wds.ResampledShards(all_shards),  # url_shard
            # at this point we have an iterator over all the shards
            wds.shuffle(50),
            wds.split_by_worker,  # if multi-node
            wds.tarfile_to_samples(),
            # add wds.split_by_node here if you are using multiple nodes
            wds.detshuffle(
                100
            ),  # shuffles in the memory, leverage large RAM for more efficient loading
            wds.decode(wds.autodecode.basichandlers),  # TODO
            wds.to_tuple("sample.pyd"),  # extract the pyd from top level dict
            wds.map(post_process_cls.decode_zip),
            # wds.map(post_process_cls.paired_post_process), # create input-novelview paired samples
            wds.map(post_process_cls._post_process_batch_sample),
            # wds.detshuffle(1000),  # shuffles in the memory, leverage large RAM for more efficient loading
            wds.batched(
                2,
                partial=True,
                # collation_fn=collate
            )  # streaming more data at once, and rebatch later
        )

        loader_shard = wds.WebLoader(
            dataset,
            num_workers=num_workers,
            drop_last=False,
            batch_size=None,
            shuffle=False,
            persistent_workers=num_workers
            > 0).unbatched().shuffle(200).batched(batch_size).map(
                post_process_cls.single_instance_sample_create_dict)

        # persistent_workers=num_workers > 0).unbatched().batched(batch_size).map(post_process_cls.create_dict)
        # 1000).batched(batch_size).map(post_process_cls.create_dict)
    # .map(collate)
    # .map(collate)

    # .batched(batch_size)
    #

    # .unbatched().shuffle(1000).batched(batch_size).map(post_process)
    #     # https://github.com/webdataset/webdataset/issues/187

    # return next(iter(loader_shard))
    #return dataset
    return loader_shard


class PostProcessForDiff:

    def __init__(
        self,
        reso,
        reso_encoder,
        imgnet_normalize,
        plucker_embedding,
        decode_encode_img_only,
        mv_latent_dir,
    ) -> None:
        self.plucker_embedding = plucker_embedding

        self.mv_latent_dir = mv_latent_dir
        self.decode_encode_img_only = decode_encode_img_only

        transformations = [
            transforms.ToTensor(),  # [0,1] range
        ]
        if imgnet_normalize:
            transformations.append(
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))  # type: ignore
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)))  # type: ignore

        self.normalize = transforms.Compose(transformations)

        self.reso_encoder = reso_encoder
        self.reso = reso
        self.instance_data_length = 40
        # self.pair_per_instance = 1 # compat
        self.pair_per_instance = 2  # check whether improves IO
        # self.pair_per_instance = 3 # check whether improves IO
        # self.pair_per_instance = 4 # check whether improves IO
        self.camera = torch.load('eval_pose.pt', map_location='cpu').numpy()
        self.canonical_frame = self.camera[25:26]  # 1, 25 # inverse this
        self.canonical_frame_pos = self.canonical_frame[:, :16].reshape(4, 4)

    def get_rays_kiui(self, c, opengl=True):
        h, w = self.reso_encoder, self.reso_encoder
        intrinsics, pose = c[16:], c[:16].reshape(4, 4)
        # cx, cy, fx, fy = intrinsics[2], intrinsics[5]
        fx = fy = 525  # pixel space
        cx = cy = 256  # rendering default K
        factor = self.reso / (cx * 2)  # 128 / 512
        fx = fx * factor
        fy = fy * factor

        x, y = torch.meshgrid(
            torch.arange(w, device=pose.device),
            torch.arange(h, device=pose.device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()

        cx = w * 0.5
        cy = h * 0.5

        # focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - cx + 0.5) / fx,
                    (y - cy + 0.5) / fy * (-1.0 if opengl else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if opengl else 1.0),
        )  # [hw, 3]

        rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
        rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d)  # [hw, 3]

        rays_o = rays_o.view(h, w, 3)
        rays_d = safe_normalize(rays_d).view(h, w, 3)

        return rays_o, rays_d

    def gen_rays(self, c):
        # Generate rays
        intrinsics, c2w = c[16:], c[:16].reshape(4, 4)
        self.h = self.reso_encoder
        self.w = self.reso_encoder
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
            indexing='ij')

        # normalize to 0-1 pixel range
        yy = yy / self.h
        xx = xx / self.w

        # K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
        cx, cy, fx, fy = intrinsics[2], intrinsics[5], intrinsics[
            0], intrinsics[4]
        # cx *= self.w
        # cy *= self.h

        # f_x = f_y = fx * h / res_raw
        c2w = torch.from_numpy(c2w).float()

        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        del xx, yy, zz
        # st()
        dirs = (c2w[None, :3, :3] @ dirs)[..., 0]

        origins = c2w[None, :3, 3].expand(self.h * self.w, -1).contiguous()
        origins = origins.view(self.h, self.w, 3)
        dirs = dirs.view(self.h, self.w, 3)

        return origins, dirs

    def normalize_camera(self, c):
        # assert c.shape[0] == self.chunk_size  # 8 o r10

        c = c[None]  # api compat
        B = c.shape[0]

        camera_poses = c[:, :16].reshape(B, 4, 4)  # 3x4

        cam_radius = np.linalg.norm(
            self.canonical_frame_pos.reshape(4, 4)[:3, 3],
            axis=-1,
            keepdims=False)  # since g-buffer adopts dynamic radius here.
        frame1_fixed_pos = np.eye(4)
        frame1_fixed_pos[2, -1] = -cam_radius

        transform = frame1_fixed_pos @ np.linalg.inv(
            self.canonical_frame_pos)  # 4,4
        # from LGM, https://github.com/3DTopia/LGM/blob/fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c/core/provider_objaverse.py#L127
        # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(c[[0,4]])

        new_camera_poses = transform[None] @ camera_poses  # [V, 4, 4]

        c = np.concatenate([new_camera_poses.reshape(B, 16), c[:, 16:]],
                           axis=-1)

        return c[0]

    def _post_process_sample(self, data_sample):
        # raw_img, depth, c, bbox, caption, ins = data_sample
        raw_img, c, caption, ins = data_sample

        # c = self.normalize_camera(c) @ if relative pose.

        img = raw_img  # 256x256

        img = torch.from_numpy(img).permute(2, 0, 1) / 127.5 - 1

        # load latent.
        # latent_path = Path(self.mv_latent_dir, ins, 'latent.npy') # ! a converged version, before adding augmentation

        # if random.random() > 0.5:
        #     latent_path = Path(self.mv_latent_dir, ins, 'latent.npy')
        # else: # augmentation, double the dataset
        latent_path = Path(
            self.mv_latent_dir.replace('v=4-final', 'v=4-rotate'), ins,
            'latent.npy')

        latent = np.load(latent_path)

        # return (img_to_encoder, img, c, caption, ins)
        return (latent, img, c, caption, ins)

    def rand_sample_idx(self):
        return random.randint(0, self.instance_data_length - 1)

    def rand_pair(self):
        return (self.rand_sample_idx() for _ in range(2))

    def paired_post_process(self, sample):
        # repeat n times?
        all_inp_list = []
        all_nv_list = []
        caption, ins = sample[-2:]
        # expanded_return = []
        for _ in range(self.pair_per_instance):
            cano_idx, nv_idx = self.rand_pair()
            cano_sample = self._post_process_sample(item[cano_idx]
                                                    for item in sample[:-2])
            nv_sample = self._post_process_sample(item[nv_idx]
                                                  for item in sample[:-2])
            all_inp_list.extend(cano_sample)
            all_nv_list.extend(nv_sample)
        return (*all_inp_list, *all_nv_list, caption, ins)
        # return [cano_sample, nv_sample, caption, ins]
        # return (*cano_sample, *nv_sample, caption, ins)

    # def single_sample_create_dict(self, sample, prefix=''):
    #     # if len(sample) == 1:
    #     #     sample = sample[0]
    #     # assert len(sample) == 6
    #     img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox = sample
    #     return {
    #         # **sample,
    #         f'{prefix}img_to_encoder': img_to_encoder,
    #         f'{prefix}img': img,
    #         f'{prefix}depth_mask': fg_mask_reso,
    #         f'{prefix}depth': depth_reso,
    #         f'{prefix}c': c,
    #         f'{prefix}bbox': bbox,
    #     }

    def single_sample_create_dict(self, sample, prefix=''):
        # if len(sample) == 1:
        #     sample = sample[0]
        # assert len(sample) == 6
        # img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox = sample
        # img_to_encoder, img, c, caption, ins = sample
        # img, c, caption, ins = sample
        latent, img, c, caption, ins = sample
        # load latent
        return {
            # **sample,
            # 'img_to_encoder': img_to_encoder,
            'latent': latent,
            'img': img,
            'c': c,
            'caption': caption,
            'ins': ins
        }

    def decode_zip(self, sample_pyd, shape=(256, 256)):
        if isinstance(sample_pyd, tuple):
            sample_pyd = sample_pyd[0]
        assert isinstance(sample_pyd, dict)

        raw_img = decompress_and_open_image_gzip(
            sample_pyd['raw_img'],
            is_img=True,
            decompress=True,
            decompress_fn=lz4.frame.decompress)

        caption = sample_pyd['caption'].decode('utf-8')
        ins = sample_pyd['ins'].decode('utf-8')

        c = decompress_array(sample_pyd['c'], (25, ),
                             np.float32,
                             decompress=True,
                             decompress_fn=lz4.frame.decompress)

        # bbox = decompress_array(
        #     sample_pyd['bbox'],
        #     (
        #         40,
        #         4,
        #     ),
        #     np.float32,
        #     # decompress=False)
        #     decompress=True,
        #     decompress_fn=lz4.frame.decompress)

        # if self.decode_encode_img_only:
        #     depth = np.zeros(shape=(40, *shape)) # save loading time
        # else:
        #     depth = decompress_array(sample_pyd['depth'], (40, *shape),
        #                             np.float32,
        #                             decompress=True,
        #                             decompress_fn=lz4.frame.decompress)

        # return {'raw_img': raw_img, 'depth': depth, 'bbox': bbox, 'caption': caption, 'ins': ins, 'c': c}
        # return raw_img, depth, c, bbox, caption, ins
        # return raw_img, bbox, caption, ins
        # return bbox, caption, ins
        return raw_img, c, caption, ins
        # ! run single-instance pipeline first
        # return raw_img[0], depth[0], c[0], bbox[0], caption, ins

    def create_dict(self, sample):
        # sample = [item[0] for item in sample] # wds wrap items in []
        # cano_sample_list = [[] for _ in range(6)]
        # nv_sample_list = [[] for _ in range(6)]
        # for idx in range(0, self.pair_per_instance):
        #     cano_sample = sample[6*idx:6*(idx+1)]
        #     nv_sample = sample[6*self.pair_per_instance+6*idx:6*self.pair_per_instance+6*(idx+1)]

        #     for item_idx in range(6):
        #         cano_sample_list[item_idx].append(cano_sample[item_idx])
        #         nv_sample_list[item_idx].append(nv_sample[item_idx])

        #         # ! cycle input/output view for more pairs
        #         cano_sample_list[item_idx].append(nv_sample[item_idx])
        #         nv_sample_list[item_idx].append(cano_sample[item_idx])

        cano_sample = self.single_sample_create_dict(sample, prefix='')
        # nv_sample = self.single_sample_create_dict((torch.cat(item_list) for item_list in nv_sample_list) , prefix='nv_')

        return cano_sample
        # return {
        #     **cano_sample,
        #     # **nv_sample,
        #     'caption': sample[-2],
        #     'ins': sample[-1]
        # }


# test tar loading
def load_wds_diff_ResampledShard(file_path,
                                 batch_size,
                                 num_workers,
                                 reso,
                                 reso_encoder,
                                 test=False,
                                 preprocess=None,
                                 imgnet_normalize=True,
                                 plucker_embedding=False,
                                 decode_encode_img_only=False,
                                 mv_latent_dir='',
                                 **kwargs):

    #     return raw_img, depth, c, bbox, sample_pyd['ins.pyd'], sample_pyd['fname.pyd']

    post_process_cls = PostProcessForDiff(
        reso,
        reso_encoder,
        imgnet_normalize=imgnet_normalize,
        plucker_embedding=plucker_embedding,
        decode_encode_img_only=decode_encode_img_only,
        mv_latent_dir=mv_latent_dir,
    )

    if isinstance(file_path, list):  # lst of shard urls
        all_shards = []
        for url_path in file_path:
            all_shards.extend(wds.shardlists.expand_source(url_path))
        logger.log('all_shards', all_shards)
    else:
        all_shards = file_path  # to be expanded

    dataset = wds.DataPipeline(
        wds.ResampledShards(all_shards),  # url_shard
        # at this point we have an iterator over all the shards
        wds.shuffle(100),
        wds.split_by_worker,  # if multi-node
        wds.tarfile_to_samples(),
        # add wds.split_by_node here if you are using multiple nodes
        wds.shuffle(
            20000
        ),  # shuffles in the memory, leverage large RAM for more efficient loading
        wds.decode(wds.autodecode.basichandlers),  # TODO
        wds.to_tuple("sample.pyd"),  # extract the pyd from top level dict
        wds.map(post_process_cls.decode_zip),
        # wds.map(post_process_cls.paired_post_process), # create input-novelview paired samples
        wds.map(post_process_cls._post_process_sample),
        # wds.detshuffle(1000),  # shuffles in the memory, leverage large RAM for more efficient loading
        wds.batched(
            100,
            partial=True,
            # collation_fn=collate
        )  # streaming more data at once, and rebatch later
    )

    loader_shard = wds.WebLoader(
        dataset,
        num_workers=num_workers,
        drop_last=False,
        batch_size=None,
        shuffle=False,
        persistent_workers=num_workers
        > 0).unbatched().shuffle(2500).batched(batch_size).map(
            post_process_cls.create_dict)

    # persistent_workers=num_workers > 0).unbatched().batched(batch_size).map(post_process_cls.create_dict)
    # 1000).batched(batch_size).map(post_process_cls.create_dict)
    # .map(collate)
    # .map(collate)

    # .batched(batch_size)
    #

    # .unbatched().shuffle(1000).batched(batch_size).map(post_process)
    #     # https://github.com/webdataset/webdataset/issues/187

    # return next(iter(loader_shard))
    #return dataset
    return loader_shard


def load_wds_data(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        num_workers=6,
        plucker_embedding=False,
        decode_encode_img_only=False,
        load_wds_diff=False,
        load_wds_latent=False,
        load_instance=False,  # for evaluation
        mv_input=False,
        split_chunk_input=False,
        duplicate_sample=True,
        mv_latent_dir='',
        append_depth=False,
        gs_cam_format=False,
        orthog_duplicate=False,
        **args):

    if load_wds_diff:
        # assert num_workers == 0  # on aliyun, worker=0 performs much much faster
        wds_loader = load_wds_diff_ResampledShard(
            file_path,
            batch_size=batch_size,
            num_workers=num_workers,
            reso=reso,
            reso_encoder=reso_encoder,
            plucker_embedding=plucker_embedding,
            decode_encode_img_only=decode_encode_img_only,
            mv_input=mv_input,
            split_chunk_input=split_chunk_input,
            append_depth=append_depth,
            mv_latent_dir=mv_latent_dir,
            gs_cam_format=gs_cam_format,
            orthog_duplicate=orthog_duplicate,
        )
    elif load_wds_latent:
        # for diffusion training, cache latent
        wds_loader = load_wds_latent_ResampledShard(
            file_path,
            batch_size=batch_size,
            num_workers=num_workers,
            reso=reso,
            reso_encoder=reso_encoder,
            plucker_embedding=plucker_embedding,
            decode_encode_img_only=decode_encode_img_only,
            mv_input=mv_input,
            split_chunk_input=split_chunk_input,
        )

    # elif load_instance:
    #     wds_loader = load_wds_instance_ResampledShard(
    #         file_path,
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         reso=reso,
    #         reso_encoder=reso_encoder,
    #         plucker_embedding=plucker_embedding,
    #         decode_encode_img_only=decode_encode_img_only
    #     )

    else:
        wds_loader = load_wds_ResampledShard(
            file_path,
            batch_size=batch_size,
            num_workers=num_workers,
            reso=reso,
            reso_encoder=reso_encoder,
            plucker_embedding=plucker_embedding,
            decode_encode_img_only=decode_encode_img_only,
            load_instance=load_instance,
            mv_input=mv_input,
            split_chunk_input=split_chunk_input,
            duplicate_sample=duplicate_sample,
            append_depth=append_depth,
            gs_cam_format=gs_cam_format,
            orthog_duplicate=orthog_duplicate,
        )

    while True:
        yield from wds_loader
        # yield from wds_loader


class PostProcess_forlatent:

    def __init__(
        self,
        reso,
        reso_encoder,
        imgnet_normalize,
        plucker_embedding,
        decode_encode_img_only,
    ) -> None:
        self.plucker_embedding = plucker_embedding
        self.decode_encode_img_only = decode_encode_img_only

        transformations = [
            transforms.ToTensor(),  # [0,1] range
        ]
        if imgnet_normalize:
            transformations.append(
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))  # type: ignore
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)))  # type: ignore

        self.normalize = transforms.Compose(transformations)

        self.reso_encoder = reso_encoder
        self.reso = reso
        self.instance_data_length = 40
        # self.pair_per_instance = 1 # compat
        self.pair_per_instance = 2  # check whether improves IO
        # self.pair_per_instance = 3 # check whether improves IO
        # self.pair_per_instance = 4 # check whether improves IO

    def _post_process_sample(self, data_sample):
        # raw_img, depth, c, bbox, caption, ins = data_sample
        raw_img, c, caption, ins = data_sample

        # bbox = (bbox*(self.reso/256)).astype(np.uint8) # normalize bbox to the reso range

        if raw_img.shape[-2] != self.reso_encoder:
            img_to_encoder = cv2.resize(raw_img,
                                        (self.reso_encoder, self.reso_encoder),
                                        interpolation=cv2.INTER_LANCZOS4)
        else:
            img_to_encoder = raw_img

        img_to_encoder = self.normalize(img_to_encoder)
        if self.plucker_embedding:
            rays_o, rays_d = self.gen_rays(c)
            rays_plucker = torch.cat(
                [torch.cross(rays_o, rays_d, dim=-1), rays_d],
                dim=-1).permute(2, 0, 1)  # [h, w, 6] -> 6,h,w
            img_to_encoder = torch.cat([img_to_encoder, rays_plucker], 0)

        img = cv2.resize(raw_img, (self.reso, self.reso),
                         interpolation=cv2.INTER_LANCZOS4)

        img = torch.from_numpy(img).permute(2, 0, 1) / 127.5 - 1

        return (img_to_encoder, img, c, caption, ins)

    def rand_sample_idx(self):
        return random.randint(0, self.instance_data_length - 1)

    def rand_pair(self):
        return (self.rand_sample_idx() for _ in range(2))

    def paired_post_process(self, sample):
        # repeat n times?
        all_inp_list = []
        all_nv_list = []
        caption, ins = sample[-2:]
        # expanded_return = []
        for _ in range(self.pair_per_instance):
            cano_idx, nv_idx = self.rand_pair()
            cano_sample = self._post_process_sample(item[cano_idx]
                                                    for item in sample[:-2])
            nv_sample = self._post_process_sample(item[nv_idx]
                                                  for item in sample[:-2])
            all_inp_list.extend(cano_sample)
            all_nv_list.extend(nv_sample)
        return (*all_inp_list, *all_nv_list, caption, ins)
        # return [cano_sample, nv_sample, caption, ins]
        # return (*cano_sample, *nv_sample, caption, ins)
    def paired_post_process(self, sample):
        # repeat n times?
        all_inp_list = []
        all_nv_list = []
        caption, ins = sample[-2:]
        # expanded_return = []
        for _ in range(self.pair_per_instance):
            cano_idx, nv_idx = self.rand_pair()
            cano_sample = self._post_process_sample(item[cano_idx]
                                                    for item in sample[:-2])
            nv_sample = self._post_process_sample(item[nv_idx]
                                                  for item in sample[:-2])
            all_inp_list.extend(cano_sample)
            all_nv_list.extend(nv_sample)
        return (*all_inp_list, *all_nv_list, caption, ins)
        # return [cano_sample, nv_sample, caption, ins]
        # return (*cano_sample, *nv_sample, caption, ins)

    # def single_sample_create_dict(self, sample, prefix=''):
    #     # if len(sample) == 1:
    #     #     sample = sample[0]
    #     # assert len(sample) == 6
    #     img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox = sample
    #     return {
    #         # **sample,
    #         f'{prefix}img_to_encoder': img_to_encoder,
    #         f'{prefix}img': img,
    #         f'{prefix}depth_mask': fg_mask_reso,
    #         f'{prefix}depth': depth_reso,
    #         f'{prefix}c': c,
    #         f'{prefix}bbox': bbox,
    #     }

    def single_sample_create_dict(self, sample, prefix=''):
        # if len(sample) == 1:
        #     sample = sample[0]
        # assert len(sample) == 6
        # img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox = sample
        img_to_encoder, img, c, caption, ins = sample
        return {
            # **sample,
            'img_to_encoder': img_to_encoder,
            'img': img,
            'c': c,
            'caption': caption,
            'ins': ins
        }

    def decode_zip(self, sample_pyd, shape=(256, 256)):
        if isinstance(sample_pyd, tuple):
            sample_pyd = sample_pyd[0]
        assert isinstance(sample_pyd, dict)

        latent = sample_pyd['latent']
        caption = sample_pyd['caption'].decode('utf-8')
        c = sample_pyd['c']
        # img = sample_pyd['img']
        # st()

        return latent, caption, c

    def create_dict(self, sample):

        return {
            # **sample,
            'latent': sample[0],
            'caption': sample[1],
            'c': sample[2],
        }


# test tar loading
def load_wds_latent_ResampledShard(file_path,
                                   batch_size,
                                   num_workers,
                                   reso,
                                   reso_encoder,
                                   test=False,
                                   preprocess=None,
                                   imgnet_normalize=True,
                                   plucker_embedding=False,
                                   decode_encode_img_only=False,
                                   **kwargs):

    #     return raw_img, depth, c, bbox, sample_pyd['ins.pyd'], sample_pyd['fname.pyd']

    post_process_cls = PostProcess_forlatent(
        reso,
        reso_encoder,
        imgnet_normalize=imgnet_normalize,
        plucker_embedding=plucker_embedding,
        decode_encode_img_only=decode_encode_img_only,
    )

    if isinstance(file_path, list):  # lst of shard urls
        all_shards = []
        for url_path in file_path:
            all_shards.extend(wds.shardlists.expand_source(url_path))
        logger.log('all_shards', all_shards)
    else:
        all_shards = file_path  # to be expanded

    dataset = wds.DataPipeline(
        wds.ResampledShards(all_shards),  # url_shard
        # at this point we have an iterator over all the shards
        wds.shuffle(50),
        wds.split_by_worker,  # if multi-node
        wds.tarfile_to_samples(),
        # add wds.split_by_node here if you are using multiple nodes
        wds.detshuffle(
            2500
        ),  # shuffles in the memory, leverage large RAM for more efficient loading
        wds.decode(wds.autodecode.basichandlers),  # TODO
        wds.to_tuple("sample.pyd"),  # extract the pyd from top level dict
        wds.map(post_process_cls.decode_zip),
        # wds.map(post_process_cls._post_process_sample),
        # wds.detshuffle(1000),  # shuffles in the memory, leverage large RAM for more efficient loading
        wds.batched(
            150,
            partial=True,
            # collation_fn=collate
        )  # streaming more data at once, and rebatch later
    )

    loader_shard = wds.WebLoader(
        dataset,
        num_workers=num_workers,
        drop_last=False,
        batch_size=None,
        shuffle=False,
        persistent_workers=num_workers
        > 0).unbatched().shuffle(1000).batched(batch_size).map(
            post_process_cls.create_dict)

    # persistent_workers=num_workers > 0).unbatched().batched(batch_size).map(post_process_cls.create_dict)
    # 1000).batched(batch_size).map(post_process_cls.create_dict)
    # .map(collate)
    # .map(collate)

    # .batched(batch_size)
    #

    # .unbatched().shuffle(1000).batched(batch_size).map(post_process)
    #     # https://github.com/webdataset/webdataset/issues/187

    # return next(iter(loader_shard))
    #return dataset
    return loader_shard
