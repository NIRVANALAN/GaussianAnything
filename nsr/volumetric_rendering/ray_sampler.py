# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
The ray sampler is a module that takes in camera matrices and resolution and batches of rays.
Expects cam2world matrices that use the OpenCV camera coordinate system conventions.
"""

import torch
from pdb import set_trace as st
import random

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision


######################################################################################
# wrapper to simplify the use of nerfnet
######################################################################################
# https://github.com/Kai-46/nerfplusplus/blob/ebf2f3e75fd6c5dfc8c9d0b533800daaf17bd95f/ddp_model.py#L16
def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis*p_sphere, dim=-1, keepdim=True) * (1.-torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(
        p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # now calculate conventional depth
    depth_real = 1. / (depth + TINY_NUMBER) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real


class RaySampler(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = None, None, None, None, None

    def create_patch_uv(self,
                        patch_resolution,
                        resolution,
                        cam2world_matrix,
                        fg_bbox=None):

        def sample_patch_uv(fg_bbox=None):
            assert patch_resolution <= resolution

            def sample_patch_range():
                patch_reolution_start = random.randint(
                    0, resolution -
                    patch_resolution)  # alias for randrange(start, stop+1)
                # patch_reolution_end = patch_reolution_start + patch_resolution
                return patch_reolution_start  # , patch_reolution_end

            def sample_patch_range_oversample_boundary(range_start=None,
                                                       range_end=None):
                # left down corner undersampled
                if range_start is None:
                    # range_start = patch_resolution // 2
                    range_start = patch_resolution
                if range_end is None:
                    # range_end = resolution + patch_resolution // 2
                    range_end = resolution + patch_resolution

                # oversample the boundary
                patch_reolution_end = random.randint(
                    range_start,
                    range_end,
                )

                # clip range
                if patch_reolution_end <= patch_resolution:
                    patch_reolution_end = patch_resolution
                elif patch_reolution_end > resolution:
                    patch_reolution_end = resolution

                # patch_reolution_end = patch_reolution_start + patch_resolution
                return patch_reolution_end  # , patch_reolution_end

            # h_start = sample_patch_range()
            # assert fg_bbox is not None
            if fg_bbox is not None and random.random(
            ) > 0.025:  # only train foreground. Has 0.1 prob to sample/train background.
                # if fg_bbox is not None: # only train foreground. Has 0.1 prob to sample/train background.
                # only return one UV here
                top_min, left_min = fg_bbox[:, :2].min(dim=0,
                                                       keepdim=True)[0][0]
                height_max, width_max = fg_bbox[:, 2:].max(dim=0,
                                                           keepdim=True)[0][0]

                left_boundary, right_boundary = patch_resolution // 2, resolution - patch_resolution // 2
                h_mid = random.randint(
                    min(max(top_min, left_boundary), right_boundary),
                    max(min(height_max, right_boundary), left_boundary),
                )

                w_mid = random.randint(
                    min(max(left_min, left_boundary), right_boundary),
                    max(min(width_max, right_boundary), left_boundary),
                )

                h_end = h_mid + patch_resolution // 2
                w_end = w_mid + patch_resolution // 2

                # if top_min + patch_resolution < height_max:
                #     h_end = sample_patch_range_oversample_boundary(
                #         top_min + patch_resolution, height_max)
                # else:
                #     h_end = max(
                #         height_max.to(torch.uint8).item(), patch_resolution)
                # if left_min + patch_resolution < width_max:
                #     w_end = sample_patch_range_oversample_boundary(
                #         left_min + patch_resolution, width_max)
                # else:
                #     w_end = max(
                #         width_max.to(torch.uint8).item(), patch_resolution)

                h_start = h_end - patch_resolution
                w_start = w_end - patch_resolution

                try:
                    assert h_start >= 0 and w_start >= 0
                except:
                    st()

            else:
                h_end = sample_patch_range_oversample_boundary()
                h_start = h_end - patch_resolution
                w_end = sample_patch_range_oversample_boundary()
                w_start = w_end - patch_resolution

                assert h_start >= 0 and w_start >= 0

            uv = torch.stack(
                torch.meshgrid(
                    torch.arange(
                        start=h_start,
                        # end=h_start+patch_resolution,
                        end=h_end,
                        dtype=torch.float32,
                        device=cam2world_matrix.device),
                    torch.arange(
                        start=w_start,
                        #  end=w_start + patch_resolution,
                        end=w_end,
                        dtype=torch.float32,
                        device=cam2world_matrix.device),
                    indexing='ij')) * (1. / resolution) + (0.5 / resolution)

            uv = uv.flip(0).reshape(2, -1).transpose(1, 0)  # ij -> xy

            return uv, (h_start, w_start, patch_resolution, patch_resolution
                        )  # top: int, left: int, height: int, width: int

        all_uv = []
        ray_bboxes = []
        for _ in range(cam2world_matrix.shape[0]):
            uv, bbox = sample_patch_uv(fg_bbox)
            all_uv.append(uv)
            ray_bboxes.append(bbox)

        all_uv = torch.stack(all_uv, 0)  # B patch_res**2 2
        # ray_bboxes = torch.stack(ray_bboxes, 0) # B patch_res**2 2

        return all_uv, ray_bboxes

    def create_uv(self, resolution, cam2world_matrix):

        uv = torch.stack(
            torch.meshgrid(torch.arange(resolution,
                                        dtype=torch.float32,
                                        device=cam2world_matrix.device),
                           torch.arange(resolution,
                                        dtype=torch.float32,
                                        device=cam2world_matrix.device),
                           indexing='ij')) * (1. / resolution) + (0.5 /
                                                                  resolution)

        uv = uv.flip(0).reshape(2, -1).transpose(1, 0)  # why
        uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

        return uv

    def forward(self, cam2world_matrix, intrinsics, resolution, fg_mask=None):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        intrinsics: (N, 3, 3)
        resolution: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 2)
        """
        N, M = cam2world_matrix.shape[0], resolution**2
        cam_locs_world = cam2world_matrix[:, :3, 3]
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        # uv = torch.stack(
        #     torch.meshgrid(torch.arange(resolution,
        #                                 dtype=torch.float32,
        #                                 device=cam2world_matrix.device),
        #                    torch.arange(resolution,
        #                                 dtype=torch.float32,
        #                                 device=cam2world_matrix.device),
        #                    indexing='ij')) * (1. / resolution) + (0.5 /
        #                                                           resolution)
        # uv = uv.flip(0).reshape(2, -1).transpose(1, 0)  # why
        # uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)
        uv = self.create_uv(
            resolution,
            cam2world_matrix,
        )

        x_cam = uv[:, :, 0].view(N, -1)
        y_cam = uv[:, :, 1].view(N, -1)  # [0,1] range
        z_cam = torch.ones((N, M), device=cam2world_matrix.device)

        # basically torch.inverse(intrinsics)
        x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1) *
                  sk.unsqueeze(-1) / fy.unsqueeze(-1) - sk.unsqueeze(-1) *
                  y_cam / fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        cam_rel_points = torch.stack(
            (x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)

        # st()

        world_rel_points = torch.bmm(cam2world_matrix,
                                     cam_rel_points.permute(0, 2, 1)).permute(
                                         0, 2, 1)[:, :, :3]

        ray_dirs = world_rel_points - cam_locs_world[:, None, :]
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

        ray_origins = cam_locs_world.unsqueeze(1).repeat(
            1, ray_dirs.shape[1], 1)

        return ray_origins, ray_dirs, None


class PatchRaySampler(RaySampler):

    def forward(self,
                cam2world_matrix,
                intrinsics,
                patch_resolution,
                resolution,
                fg_bbox=None):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        intrinsics: (N, 3, 3)
        resolution: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 2)
        """
        N, M = cam2world_matrix.shape[0], patch_resolution**2
        cam_locs_world = cam2world_matrix[:, :3, 3]
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        # uv = self.create_uv(resolution, cam2world_matrix)

        # all_uv, ray_bboxes = self.create_patch_uv(
        all_uv_list = []
        ray_bboxes = []
        for idx in range(N):
            uv, bboxes = self.create_patch_uv(
                patch_resolution, resolution, cam2world_matrix[idx:idx + 1],
                fg_bbox[idx:idx + 1]
                if fg_bbox is not None else None)  # for debugging, hard coded
            all_uv_list.append(
                uv
                # cam2world_matrix[idx:idx+1], )[0]  # for debugging, hard coded
            )
            ray_bboxes.extend(bboxes)
        all_uv = torch.cat(all_uv_list, 0)
        # ray_bboxes = torch.cat(ray_bboxes_list, 0)
        # all_uv, _ = self.create_patch_uv(
        #     patch_resolution, resolution,
        #     cam2world_matrix, fg_bbox)  # for debugging, hard coded
        # st()

        x_cam = all_uv[:, :, 0].view(N, -1)
        y_cam = all_uv[:, :, 1].view(N, -1)  # [0,1] range
        z_cam = torch.ones((N, M), device=cam2world_matrix.device)

        # basically torch.inverse(intrinsics)
        x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1) *
                  sk.unsqueeze(-1) / fy.unsqueeze(-1) - sk.unsqueeze(-1) *
                  y_cam / fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        cam_rel_points = torch.stack(
            (x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)

        world_rel_points = torch.bmm(cam2world_matrix,
                                     cam_rel_points.permute(0, 2, 1)).permute(
                                         0, 2, 1)[:, :, :3]

        ray_dirs = world_rel_points - cam_locs_world[:, None, :]
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

        ray_origins = cam_locs_world.unsqueeze(1).repeat(
            1, ray_dirs.shape[1], 1)

        return ray_origins, ray_dirs, ray_bboxes
