# modified from : 2dgs/gaussian_renderer/__init__.py
import numpy as np

from pdb import set_trace as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# from diff_gaussian_rasterization import (
#     GaussianRasterizationSettings,
#     GaussianRasterizer,
# )

from torch.profiler import profile, record_function, ProfilerActivity
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.point_utils import depth_to_normal, depth_to_normal_2

import kiui


class GaussianRenderer2DGS:
    def __init__(self, output_size, out_chans, rendering_kwargs, **kwargs):
        
        # self.opt = opt
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        # self.bg_color = torch.tensor([0,0,1], dtype=torch.float32, device="cuda")

        self.output_size = output_size
        self.out_chans = out_chans
        self.rendering_kwargs = rendering_kwargs
 
        # intrinsics
        # self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        # self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        # self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        # self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        # self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        # self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        # self.proj_matrix[2, 3] = 1
        
    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, tanfov,  bg_color=None, scale_modifier=1, output_size=None):
        # gaussians: [B, N, 14-1]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        if output_size is None:
            output_size = self.output_size

        device = gaussians.device
        B, V = cam_view.shape[:2]
        assert gaussians.shape[2] == 13 # scale with 2dof
        gaussians = gaussians.contiguous().float() # gs rendering in fp32

        # loop of loop...
        images = []
        alphas = []
        depths = []
        # surf_normals = []
        rend_normals = []
        dists = []

        if bg_color is None:
            bg_color = self.bg_color

        for b in range(B):

            # pos, opacity, scale, rotation, shs
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:6].contiguous().float()
            rotations = gaussians[b, :, 6:10].contiguous().float()
            rgbs = gaussians[b, :, 10:13].contiguous().float() # [N, 3]

            for v in range(V):
                
                # render novel views
                view_matrix = cam_view[b, v].float() # world_view_transform
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                # with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU,], record_shapes=True) as prof:

                    # with record_function("rendering"):

                raster_settings = GaussianRasterizationSettings(
                    image_height=output_size,
                    image_width=output_size,
                    tanfovx=tanfov,
                    tanfovy=tanfov,
                    bg=bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                # rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                rendered_image, radii, allmap = rasterizer(
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                    # cov3D_precomp = cov3D_precomp
                )

                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

                # with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU,], record_shapes=True) as prof:

                # ! additional regularizations
                render_alpha = allmap[1:2]

                # get normal map
                # transform normal from view space to world space
                # with record_function("render_normal"):
                render_normal = allmap[2:5]
                # render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
                render_normal = (render_normal.permute(1,2,0) @ (view_matrix[:3,:3].T)).permute(2,0,1)
                
                # with record_function("render_depth"):

                # get median depth map
                render_depth_median = allmap[5:6]
                render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

                # get expected depth map
                render_depth_expected = allmap[0:1]
                render_depth_expected = (render_depth_expected / render_alpha)
                render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
                
                # get depth distortion map
                render_dist = allmap[6:7]

                # psedo surface attributes
                # surf depth is either median or expected by setting depth_ratio to 1 or 0
                # for bounded scene, use median depth, i.e., depth_ratio = 1; 
                # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.

                # ! hard coded depth_ratio = 1 for objaverse
                surf_depth = render_depth_median
                # with record_function("surf_normal"):
                #     depth_ratio = 1
                #     # surf_depth = render_depth_expected * (1-depth_ratio) + (depth_ratio) * render_depth_median
                    
                #     # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
                #     # surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
                #     surf_normal = depth_to_normal_2(world_view_transform=view_matrix, tanfov=tanfov, W=self.output_size, H=self.output_size, depth=surf_depth)
                #     surf_normal = surf_normal.permute(2,0,1)
                #     # remember to multiply with accum_alpha since render_normal is unnormalized.
                #     surf_normal = surf_normal * (render_alpha).detach()

                # ! images
                rendered_image = rendered_image.clamp(0, 1)

                # images.append(rendered_image)
                # alphas.append(rendered_alpha)
                # depths.append(rendered_depth)

                images.append(rendered_image)
                alphas.append(render_alpha)
                depths.append(surf_depth)
                # surf_normals.append(surf_normal)
                rend_normals.append(render_normal)
                dists.append(render_dist)

                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                # st()
                pass

        images = torch.stack(images, dim=0).view(B, V, 3, output_size, output_size)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, output_size, output_size)
        depths = torch.stack(depths, dim=0).view(B, V, 1, output_size, output_size)
        
        # approximated surface normal? No, direct depth supervision here.
        # surf_normals = torch.stack(surf_normals, dim=0).view(B, V, 3, self.output_size, self.output_size)

        # disk normal
        rend_normals = torch.stack(rend_normals, dim=0).view(B, V, 3, output_size, output_size)
        dists = torch.stack(dists, dim=0).view(B, V, 1, output_size, output_size)

        # images = torch.stack(images, dim=0).view(B*V, 3, self.output_size, self.output_size)
        # alphas = torch.stack(alphas, dim=0).view(B*V, 1, self.output_size, self.output_size)
        # depths = torch.stack(depths, dim=0).view(B*V, 1, self.output_size, self.output_size)

        return {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
            "depth": depths,
            # "surf_normal": surf_normals,
            "rend_normal": rend_normals,
            "dist": dists
        }

    # TODO, save/load 2dgs Gaussians

    def save_2dgs_ply(self, path, gaussians, compatible=True):
        # gaussians: [B, N, 13]

        mkdir_p(os.path.dirname(path))
        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:6].contiguous().float()
        rotations = gaussians[0, :, 6:10].contiguous().float()
        shs = gaussians[0, :, 10:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        # xyz = self._xyz.detach().cpu().numpy()
        # normals = np.zeros_like(xyz)
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # opacities = self._opacity.detach().cpu().numpy()
        # scale = self._scaling.detach().cpu().numpy()
        # rotation = self._rotation.detach().cpu().numpy()

        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))

        # save normals also
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))

        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)



    # def save_ply(self, gaussians, path, compatible=True):
    #     # gaussians: [B, N, 14]
    #     # compatible: save pre-activated gaussians as in the original paper

    #     assert gaussians.shape[0] == 1, 'only support batch size 1'

    #     from plyfile import PlyData, PlyElement
     
    #     means3D = gaussians[0, :, 0:3].contiguous().float()
    #     opacity = gaussians[0, :, 3:4].contiguous().float()
    #     scales = gaussians[0, :, 4:7].contiguous().float()
    #     rotations = gaussians[0, :, 7:11].contiguous().float()
    #     shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

    #     # prune by opacity
    #     mask = opacity.squeeze(-1) >= 0.005
    #     means3D = means3D[mask]
    #     opacity = opacity[mask]
    #     scales = scales[mask]
    #     rotations = rotations[mask]
    #     shs = shs[mask]

    #     # invert activation to make it compatible with the original ply format
    #     if compatible:
    #         opacity = kiui.op.inverse_sigmoid(opacity)
    #         scales = torch.log(scales + 1e-8)
    #         shs = (shs - 0.5) / 0.28209479177387814

    #     xyzs = means3D.detach().cpu().numpy()
    #     f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     opacities = opacity.detach().cpu().numpy()
    #     scales = scales.detach().cpu().numpy()
    #     rotations = rotations.detach().cpu().numpy()

    #     l = ['x', 'y', 'z']
    #     # All channels except the 3 DC
    #     for i in range(f_dc.shape[1]):
    #         l.append('f_dc_{}'.format(i))
    #     l.append('opacity')
    #     for i in range(scales.shape[1]):
    #         l.append('scale_{}'.format(i))
    #     for i in range(rotations.shape[1]):
    #         l.append('rot_{}'.format(i))

    #     dtype_full = [(attribute, 'f4') for attribute in l]

    #     elements = np.empty(xyzs.shape[0], dtype=dtype_full)
    #     attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
    #     elements[:] = list(map(tuple, attributes))
    #     el = PlyElement.describe(elements, 'vertex')

    #     PlyData([el]).write(path)
    
    def load_2dgs_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        normal_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians