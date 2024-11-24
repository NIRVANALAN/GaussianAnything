# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import xatlas
import trimesh
import cv2
import numpy as np
from PIL import Image
from functools import partial
import open3d as o3d
import trimesh


# https://github.com/hbb1/2d-gaussian-splatting/blob/19eb5f1e091a582e911b4282fe2832bac4c89f0f/utils/mesh_utils.py#L22C1-L43C18
# def post_process_mesh(mesh, cluster_to_keep=1000):
def post_process_mesh(mesh, cluster_to_keep=None):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    cluster_to_keep = min(len(cluster_n_triangles),10)

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    # print("num vertices raw {}".format(len(mesh.vertices)))
    # print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def smooth_mesh(mesh):
    import copy
    mesh_0 = copy.deepcopy(mesh)
    mesh_0 = mesh_0.filter_smooth_taubin(12)
    return mesh_0


def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj

def to_cam_open3d_compat(gs_foramt_c):
    W = H = image_width = image_height = 512
    projection_matrix = gs_foramt_c['projection_matrix']
    world_view_transform = gs_foramt_c['cam_view']

    # camera_traj = []

    # for i, viewpoint_cam in enumerate(viewpoint_stack):
        # W = viewpoint_cam.image_width
        # H = viewpoint_cam.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W-1) / 2],
        [0, H / 2, 0, (H-1) / 2],
        [0, 0, 0, 1]]).float().T
    intrins =  (projection_matrix @ ndc2pix)[:3,:3].T
    intrinsic=o3d.camera.PinholeCameraIntrinsic(
        width=image_width,
        height=image_height,
        cx = intrins[0,2].item(),
        cy = intrins[1,2].item(), 
        fx = intrins[0,0].item(), 
        fy = intrins[1,1].item()
    )

    extrinsic=np.asarray((world_view_transform.T).cpu().numpy())
    camera = o3d.camera.PinholeCameraParameters()
    camera.extrinsic = extrinsic
    camera.intrinsic = intrinsic

    # camera_traj.append(camera)
    return camera

    # return camera_traj

def save_obj(pointnp_px3, facenp_fx3, colornp_px3, fpath):

    pointnp_px3 = pointnp_px3 @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    facenp_fx3 = facenp_fx3[:, [2, 1, 0]]

    mesh = trimesh.Trimesh(
        vertices=pointnp_px3, 
        faces=facenp_fx3, 
        vertex_colors=colornp_px3,
    )
    mesh.export(fpath, 'obj')


def save_glb(pointnp_px3, facenp_fx3, colornp_px3, fpath):

    pointnp_px3 = pointnp_px3 @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    mesh = trimesh.Trimesh(
        vertices=pointnp_px3, 
        faces=facenp_fx3, 
        vertex_colors=colornp_px3,
    )
    mesh.export(fpath, 'glb')


def save_obj_with_mtl(pointnp_px3, tcoords_px2, facenp_fx3, facetex_fx3, texmap_hxwx3, fname):
    import os
    fol, na = os.path.split(fname)
    na, _ = os.path.splitext(na)

    matname = '%s/%s.mtl' % (fol, na)
    fid = open(matname, 'w')
    fid.write('newmtl material_0\n')
    fid.write('Kd 1 1 1\n')
    fid.write('Ka 0 0 0\n')
    fid.write('Ks 0.4 0.4 0.4\n')
    fid.write('Ns 10\n')
    fid.write('illum 2\n')
    fid.write('map_Kd %s.png\n' % na)
    fid.close()
    ####

    fid = open(fname, 'w')
    fid.write('mtllib %s.mtl\n' % na)

    for pidx, p in enumerate(pointnp_px3):
        pp = p
        fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))

    for pidx, p in enumerate(tcoords_px2):
        pp = p
        fid.write('vt %f %f\n' % (pp[0], pp[1]))

    fid.write('usemtl material_0\n')
    for i, f in enumerate(facenp_fx3):
        f1 = f + 1
        f2 = facetex_fx3[i] + 1
        fid.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
    fid.close()

    # save texture map
    lo, hi = 0, 1
    img = np.asarray(texmap_hxwx3, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = img.clip(0, 255)
    mask = np.sum(img.astype(np.float32), axis=-1, keepdims=True)
    mask = (mask <= 3.0).astype(np.float32)
    kernel = np.ones((3, 3), 'uint8')
    dilate_img = cv2.dilate(img, kernel, iterations=1)
    img = img * (1 - mask) + dilate_img * mask
    img = img.clip(0, 255).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(f'{fol}/{na}.png')


def loadobj(meshfile):
    v = []
    f = []
    meshfp = open(meshfile, 'r')
    for line in meshfp.readlines():
        data = line.strip().split(' ')
        data = [da for da in data if len(da) > 0]
        if len(data) != 4:
            continue
        if data[0] == 'v':
            v.append([float(d) for d in data[1:]])
        if data[0] == 'f':
            data = [da.split('/')[0] for da in data]
            f.append([int(d) for d in data[1:]])
    meshfp.close()

    # torch need int64
    facenp_fx3 = np.array(f, dtype=np.int64) - 1
    pointnp_px3 = np.array(v, dtype=np.float32)
    return pointnp_px3, facenp_fx3


def loadobjtex(meshfile):
    v = []
    vt = []
    f = []
    ft = []
    meshfp = open(meshfile, 'r')
    for line in meshfp.readlines():
        data = line.strip().split(' ')
        data = [da for da in data if len(da) > 0]
        if not ((len(data) == 3) or (len(data) == 4) or (len(data) == 5)):
            continue
        if data[0] == 'v':
            assert len(data) == 4

            v.append([float(d) for d in data[1:]])
        if data[0] == 'vt':
            if len(data) == 3 or len(data) == 4:
                vt.append([float(d) for d in data[1:3]])
        if data[0] == 'f':
            data = [da.split('/') for da in data]
            if len(data) == 4:
                f.append([int(d[0]) for d in data[1:]])
                ft.append([int(d[1]) for d in data[1:]])
            elif len(data) == 5:
                idx1 = [1, 2, 3]
                data1 = [data[i] for i in idx1]
                f.append([int(d[0]) for d in data1])
                ft.append([int(d[1]) for d in data1])
                idx2 = [1, 3, 4]
                data2 = [data[i] for i in idx2]
                f.append([int(d[0]) for d in data2])
                ft.append([int(d[1]) for d in data2])
    meshfp.close()

    # torch need int64
    facenp_fx3 = np.array(f, dtype=np.int64) - 1
    ftnp_fx3 = np.array(ft, dtype=np.int64) - 1
    pointnp_px3 = np.array(v, dtype=np.float32)
    uvs = np.array(vt, dtype=np.float32)
    return pointnp_px3, facenp_fx3, uvs, ftnp_fx3


# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    import nvdiffrast.torch as dr
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')


def xatlas_uvmap(ctx, mesh_v, mesh_pos_idx, resolution):
    import nvdiffrast.torch as dr
    vmapping, indices, uvs = xatlas.parametrize(mesh_v.detach().cpu().numpy(), mesh_pos_idx.detach().cpu().numpy())

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

    uvs = torch.tensor(uvs, dtype=torch.float32, device=mesh_v.device)
    mesh_tex_idx = torch.tensor(indices_int64, dtype=torch.int64, device=mesh_v.device)
    # mesh_v_tex. ture
    uv_clip = uvs[None, ...] * 2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[..., 0:1]), torch.ones_like(uv_clip[..., 0:1])), dim=-1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh_tex_idx.int(), (resolution, resolution))

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh_v[None, ...], rast, mesh_pos_idx.int())
    mask = rast[..., 3:4] > 0
    return uvs, mesh_tex_idx, gb_pos, mask
