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
Helper functions for constructing camera parameter matrices. Primarily used in visualization and inference scripts.
"""

import math

import torch as th
import torch
import torch.nn as nn
import numpy as np

from nsr.volumetric_rendering import math_utils


class GaussianCameraPoseSampler:
    """
    Samples pitch and yaw from a Gaussian distribution and returns a camera pose.
    Camera is specified as looking at the origin.
    If horizontal and vertical stddev (specified in radians) are zero, gives a
    deterministic camera pose with yaw=horizontal_mean, pitch=vertical_mean.
    The coordinate system is specified with y-up, z-forward, x-left.
    Horizontal mean is the azimuthal angle (rotation around y axis) in radians,
    vertical mean is the polar angle (angle from the y axis) in radians.
    A point along the z-axis has azimuthal_angle=0, polar_angle=pi/2.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = GaussianCameraPoseSampler.sample(math.pi/2, math.pi/2, radius=1)
    """

    @staticmethod
    def sample(horizontal_mean,
               vertical_mean,
               horizontal_stddev=0,
               vertical_stddev=0,
               radius=1,
               batch_size=1,
               device='cpu'):
        h = torch.randn((batch_size, 1),
                        device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn(
            (batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2 * v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius * torch.sin(phi) * torch.cos(math.pi -
                                                                     theta)
        camera_origins[:, 2:3] = radius * torch.sin(phi) * torch.sin(math.pi -
                                                                     theta)
        camera_origins[:, 1:2] = radius * torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)


class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """

    @staticmethod
    def sample(horizontal_mean,
               vertical_mean,
               lookat_position,
               horizontal_stddev=0.,
               vertical_stddev=0.,
               radius=1,
               batch_size=1,
               device='cpu'):
        h = torch.randn((batch_size, 1),
                        device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn(
            (batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2 * v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius * torch.sin(phi) * torch.cos(math.pi -
                                                                     theta)
        camera_origins[:, 2:3] = radius * torch.sin(phi) * torch.sin(math.pi -
                                                                     theta)
        camera_origins[:, 1:2] = radius * torch.cos(phi)

        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = math_utils.normalize_vecs(lookat_position -
                                                    camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)


class UniformCameraPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    pose is sampled from a uniform distribution with range +-[horizontal/vertical]_stddev.

    Example:
    For a batch of random camera poses looking at the origin with yaw sampled from [-pi/2, +pi/2] radians:

    cam2worlds = UniformCameraPoseSampler.sample(math.pi/2, math.pi/2, horizontal_stddev=math.pi/2, radius=1, batch_size=16)
    """

    @staticmethod
    def sample(horizontal_mean,
               vertical_mean,
               horizontal_stddev=0,
               vertical_stddev=0,
               radius=1,
               batch_size=1,
               device='cpu'):
        h = (torch.rand((batch_size, 1), device=device) * 2 -
             1) * horizontal_stddev + horizontal_mean
        v = (torch.rand((batch_size, 1), device=device) * 2 -
             1) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2 * v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius * torch.sin(phi) * torch.cos(math.pi -
                                                                     theta)
        camera_origins[:, 2:3] = radius * torch.sin(phi) * torch.sin(math.pi -
                                                                     theta)
        camera_origins[:, 1:2] = radius * torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)


def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = math_utils.normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0],
                             dtype=torch.float,
                             device=origin.device).expand_as(forward_vector)

    right_vector = -math_utils.normalize_vecs(
        torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = math_utils.normalize_vecs(
        torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(
        forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack(
        (right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4,
                                   device=origin.device).unsqueeze(0).repeat(
                                       forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert (cam2world.shape[1:] == (4, 4))
    return cam2world


def FOV_to_intrinsics(fov_degrees, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """

    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = torch.tensor(
        [[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]],
        device=device)
    return intrinsics

def generate_input_camera(r, poses, device='cpu', fov=30):
    def normalize_vecs(vectors): return vectors / (torch.norm(vectors, dim=-1, keepdim=True))
    poses = np.deg2rad(poses)
    poses = torch.tensor(poses).float()
    pitch = poses[:, 0]
    yaw = poses[:, 1]

    z = r*torch.sin(pitch)
    x = r*torch.cos(pitch)*torch.cos(yaw)
    y = r*torch.cos(pitch)*torch.sin(yaw)
    cam_pos = torch.stack([x, y, z], dim=-1).reshape(z.shape[0], -1).to(device)

    forward_vector = normalize_vecs(-cam_pos)
    up_vector = torch.tensor([0, 0, -1], dtype=torch.float,
                                        device=device).reshape(-1).expand_as(forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector,
                                                        dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector,
                                                        dim=-1))
    rotate = torch.stack(
                    (left_vector, up_vector, forward_vector), dim=-1)

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = rotate

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = cam_pos
    cam2world = translation_matrix @ rotation_matrix

    fx = 0.5/np.tan(np.deg2rad(fov/2))
    fxfycxcy = torch.tensor([fx, fx, 0.5, 0.5], dtype=rotate.dtype, device=device)

    return cam2world, fxfycxcy


def uni_mesh_path(frame_number=16, radius=1.8):
    azimuths = []
    elevations = []

    # for elevation in [0,-30,30]:
    # for elevation in [0,-30,30, -65, 65]:
    # for elevation in [0,-30,30, -60, 60]:
    for elevation in [60,30, 0, -30, -60]:

        for i in range(frame_number): # 1030 * 5 * 10, for FID 50K

            # azi, elevation = sample_uniform_cameras_on_sphere()
            # azi, elevation = azi[0] / np.pi * 180, elevation[0] / np.pi * 180
            # azi, elevation = azi[0] / np.pi * 180, 0
            azi = i / frame_number * 360 # [0, 2pi]
            azimuths.append(azi)
            elevations.append(elevation)

    azimuths = np.array(azimuths)
    elevations = np.array(elevations)

    all_frame_number = azimuths.shape[0]

    # azimuths = np.array(list(range(0,360,30))).astype(float)
    # frame_number = azimuths.shape[0]
    # elevations = np.array([10]*azimuths.shape[0]).astype(float)

    zero123pp_pose, _ = generate_input_camera(radius, [[elevations[i], azimuths[i]] for i in range(all_frame_number)], fov=30)
    K = th.Tensor([1.3889, 0.0000, 0.5000, 0.0000, 1.3889, 0.5000, 0.0000, 0.0000, 0.0039]).to(zero123pp_pose) # keeps the same
    mesh_pathes = th.cat([zero123pp_pose.reshape(all_frame_number,-1), K.unsqueeze(0).repeat(all_frame_number,1)], dim=-1).cpu().numpy()

    return mesh_pathes



def sample_uniform_cameras_on_sphere(num_samples=1):
    # Step 1: Sample azimuth angles uniformly from [0, 2*pi)
    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    
    # Step 2: Sample cos(phi) uniformly from [-1, 1]
    cos_phi = np.random.uniform(-1, 1, num_samples)
    
    # Step 3: Calculate the elevation angle (phi) from cos(phi)
    phi = np.arccos(cos_phi)  # phi will be in [0, pi]
    
    # Step 4: Convert spherical coordinates to Cartesian coordinates (x, y, z)
    # x = np.sin(phi) * np.cos(theta)
    # y = np.sin(phi) * np.sin(theta)
    # z = np.cos(phi)
    
    # Combine the x, y, z coordinates into a single array
    # cameras = np.vstack((x, y, z)).T  # Shape: (num_samples, 3)
    
    # return cameras
    return theta, phi
