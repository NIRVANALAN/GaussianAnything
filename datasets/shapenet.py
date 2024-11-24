import os
import torchvision
import pickle
from typing import Any
import lmdb
import cv2
import imageio
import numpy as np
from PIL import Image
import Imath
import OpenEXR
from pdb import set_trace as st
from pathlib import Path

from functools import partial
import io
import gzip
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from guided_diffusion import logger

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
    infi_sampler=True
):
    # st()
    # dataset_cls = {
    #     'input_rec': MultiViewDataset,
    #     'nv': NovelViewDataset,
    # }[trainer_name]
    # st()
    if use_lmdb:
        logger.log('using LMDB dataset')
        # dataset_cls = LMDBDataset_MV #  2.5-3iter/s, but unstable, drops to 1 later.
        if 'nv' in trainer_name:
            dataset_cls = LMDBDataset_NV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
        else:
            dataset_cls = LMDBDataset_MV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
        # dataset = dataset_cls(file_path)
    else:
        if 'nv' in trainer_name:
            dataset_cls = NovelViewDataset  # 1.5-2iter/s
        else:
            dataset_cls = MultiViewDataset

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
    
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        drop_last=False,
                        pin_memory=True,
                        persistent_workers=num_workers > 0, 
                        shuffle=False)
    return loader


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
    infi_sampler=True
):
    # st()
    # dataset_cls = {
    #     'input_rec': MultiViewDataset,
    #     'nv': NovelViewDataset,
    # }[trainer_name]
    # st()
    if use_lmdb:
        logger.log('using LMDB dataset')
        # dataset_cls = LMDBDataset_MV #  2.5-3iter/s, but unstable, drops to 1 later.
        if 'nv' in trainer_name:
            dataset_cls = LMDBDataset_NV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
        else:
            dataset_cls = LMDBDataset_MV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
        # dataset = dataset_cls(file_path)
    else:
        if 'nv' in trainer_name:
            dataset_cls = NovelViewDataset  # 1.5-2iter/s
        else:
            dataset_cls = MultiViewDataset

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
    
    # st()

    if infi_sampler:
        train_sampler = DistributedSampler(dataset=dataset,
                                        shuffle=True,
                                        drop_last=True)

        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=True,
                            pin_memory=True,
                            persistent_workers=num_workers > 0,
                            sampler=train_sampler)

        while True:
            yield from loader

    else:
        # loader = DataLoader(dataset,
        #                     batch_size=batch_size,
        #                     num_workers=num_workers,
        #                     drop_last=False,
        #                     pin_memory=True,
        #                     persistent_workers=num_workers > 0, 
        #                     shuffle=False)
        st()
        return dataset


def load_eval_rays(file_path="",
                   reso=64,
                   reso_encoder=224,
                   imgnet_normalize=True):
    dataset = MultiViewDataset(file_path,
                               reso,
                               reso_encoder,
                               imgnet_normalize=imgnet_normalize)
    pose_list = dataset.single_pose_list
    ray_list = []
    for pose_fname in pose_list:
        # c2w = dataset.get_c2w(pose_fname).reshape(1,4,4)  #[1, 4, 4]
        # rays_o, rays_d = dataset.gen_rays(c2w)
        # ray_list.append(
        #     [rays_o.unsqueeze(0),
        #      rays_d.unsqueeze(0),
        #      c2w.reshape(-1, 16)])

        c2w = dataset.get_c2w(pose_fname).reshape(16)  #[1, 4, 4]

        c = torch.cat([c2w, dataset.intrinsics],
                      dim=0).reshape(25)  # 25, no '1' dim needed.
        ray_list.append(c)

    return ray_list


def load_eval_data(file_path="",
                   reso=64,
                   reso_encoder=224,
                   batch_size=1,
                   num_workers=1,
                   load_depth=False,
                   preprocess=None,
                   imgnet_normalize=True, 
                   interval=1,
                   **kwargs
                   ):

    dataset = MultiViewDataset(file_path,
                               reso,
                               reso_encoder,
                               preprocess=preprocess,
                               load_depth=load_depth,
                               test=True,
                               imgnet_normalize=imgnet_normalize,
                               interval=interval, **kwargs)
    print('eval dataset size: {}'.format(len(dataset)))
    # train_sampler = DistributedSampler(dataset=dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )
    # sampler=train_sampler)
    return loader


def load_memory_data(file_path="",
                     reso=64,
                     reso_encoder=224,
                     batch_size=1,
                     num_workers=1,
                     load_depth=True,
                     preprocess=None,
                     imgnet_normalize=True):
    # load a single-instance into the memory to speed up training IO
    dataset = MultiViewDataset(file_path,
                               reso,
                               reso_encoder,
                               preprocess=preprocess,
                               load_depth=True,
                               test=False,
                               overfitting=True,
                               imgnet_normalize=imgnet_normalize,
                               overfitting_bs=batch_size)
    logger.log('!!!!!!! memory dataset size: {} !!!!!!'.format(len(dataset)))
    # train_sampler = DistributedSampler(dataset=dataset)
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )

    all_data: dict = next(iter(loader))
    while True:
        start_idx = np.random.randint(0, len(dataset) - batch_size + 1)
        yield {
            k: v[start_idx:start_idx + batch_size]
            for k, v in all_data.items()
        }


class MultiViewDataset(Dataset):

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
                 interval=1):
        self.file_path = file_path
        self.overfitting = overfitting
        self.scene_scale = scene_scale
        self.reso = reso
        self.reso_encoder = reso_encoder
        self.classes = False
        self.load_depth = load_depth
        self.preprocess = preprocess
        assert not self.classes, "Not support class condition now."

        # self.ins_list = os.listdir(self.file_path)
        # if test: # TODO

        dataset_name = Path(self.file_path).stem.split('_')[0]
        self.dataset_name = dataset_name

        if test:
            # ins_list_file = Path(self.file_path).parent / f'{dataset_name}_test_list.txt' # ? in domain
            if dataset_name == 'chair':
                self.ins_list = sorted(os.listdir(
                    self.file_path))[1:2]  # more diversity
            else:
                self.ins_list = sorted(os.listdir(self.file_path))[
                    0:1]  # the first 1 instance for evaluation reference.
        else:
            # self.ins_list = sorted(Path(self.file_path).glob('[0-8]*'))
            # self.ins_list = Path(self.file_path).glob('*')
            # self.ins_list = list(Path(self.file_path).glob('*'))[:dataset_size]

            # ins_list_file = Path(
            #     self.file_path).parent / f'{dataset_name}s_train_list.txt'
            # assert ins_list_file.exists(), 'add training list for ShapeNet'
            # with open(ins_list_file, 'r') as f:
            #     self.ins_list = [name.strip() for name in f.readlines()]

            # if dataset_name == 'chair':
            ins_list_file = Path(
                self.file_path).parent / f'{dataset_name}_train_list.txt'
            # st()
            assert ins_list_file.exists(), 'add training list for ShapeNet'
            with open(ins_list_file, 'r') as f:
                self.ins_list = [name.strip()
                                 for name in f.readlines()][:dataset_size]
            # else:
            #     self.ins_list = Path(self.file_path).glob('*')

        if overfitting:
            self.ins_list = self.ins_list[:1]

        self.rgb_list = []
        self.pose_list = []
        self.depth_list = []
        self.data_ins_list = []
        self.instance_data_length = -1
        for ins in self.ins_list:
            cur_rgb_path = os.path.join(self.file_path, ins, 'rgb')
            cur_pose_path = os.path.join(self.file_path, ins, 'pose')

            cur_all_fname = sorted([
                t.split('.')[0] for t in os.listdir(cur_rgb_path)
                if 'depth' not in t
            ][::interval])
            if self.instance_data_length == -1:
                self.instance_data_length = len(cur_all_fname)
            else:
                assert len(cur_all_fname) == self.instance_data_length

            # ! check filtered data
            # for idx in range(len(cur_all_fname)):
            #     fname = cur_all_fname[idx]
            #     if not Path(os.path.join(cur_rgb_path, fname + '.png') ).exists():
            #         cur_all_fname.remove(fname)

            # del cur_all_fname[idx]

            if test:
                mid_index = len(cur_all_fname) // 3 * 2
                cur_all_fname.insert(0, cur_all_fname[mid_index])

            self.pose_list += ([
                os.path.join(cur_pose_path, fname + '.txt')
                for fname in cur_all_fname
            ])
            self.rgb_list += ([
                os.path.join(cur_rgb_path, fname + '.png')
                for fname in cur_all_fname
            ])

            self.depth_list += ([
                os.path.join(cur_rgb_path, fname + '_depth0001.exr')
                for fname in cur_all_fname
            ])
            self.data_ins_list += ([ins] * len(cur_all_fname))

        # validate overfitting on images
        if overfitting:
            # bs=9
            # self.pose_list = self.pose_list[::50//9+1]
            # self.rgb_list = self.rgb_list[::50//9+1]
            # self.depth_list = self.depth_list[::50//9+1]
            # bs=6
            # self.pose_list = self.pose_list[::50//6+1]
            # self.rgb_list = self.rgb_list[::50//6+1]
            # self.depth_list = self.depth_list[::50//6+1]
            # bs=3
            assert overfitting_bs != -1
            # bs=1
            # self.pose_list = self.pose_list[25:26]
            # self.rgb_list = self.rgb_list[25:26]
            # self.depth_list = self.depth_list[25:26]

            # uniform pose sampling
            self.pose_list = self.pose_list[::50//overfitting_bs+1]
            self.rgb_list = self.rgb_list[::50//overfitting_bs+1]
            self.depth_list = self.depth_list[::50//overfitting_bs+1]

            # sequentially sampling pose
            # self.pose_list = self.pose_list[25:25+overfitting_bs]
            # self.rgb_list = self.rgb_list[25:25+overfitting_bs]
            # self.depth_list = self.depth_list[25:25+overfitting_bs]

            # duplicate the same pose
            # self.pose_list = [self.pose_list[25]] * overfitting_bs
            # self.rgb_list = [self.rgb_list[25]] * overfitting_bs
            # self.depth_list = [self.depth_list[25]] * overfitting_bs
            # self.pose_list = [self.pose_list[28]] * overfitting_bs
            # self.rgb_list = [self.rgb_list[28]] * overfitting_bs
            # self.depth_list = [self.depth_list[28]] * overfitting_bs

        self.single_pose_list = [
            os.path.join(cur_pose_path, fname + '.txt')
            for fname in cur_all_fname
        ]

        # st()

        # if imgnet_normalize:
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

        # self.normalize_normalrange = transforms.Compose([
        #     transforms.ToTensor(),# [0,1] range
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])

        fx = fy = 525
        cx = cy = 256  # rendering default K
        factor = self.reso / (cx * 2)  # 128 / 512
        self.fx = fx * factor
        self.fy = fy * factor
        self.cx = cx * factor
        self.cy = cy * factor

        # ! fix scale for triplane ray_sampler(), here we adopt [0,1] uv range, not [0, w] img space range.
        self.cx /= self.reso  # 0.5
        self.cy /= self.reso  # 0.5
        self.fx /= self.reso
        self.fy /= self.reso

        intrinsics = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy],
                               [0, 0, 1]]).reshape(9)
        # self.intrinsics = torch.from_numpy(intrinsics).float()
        self.intrinsics = intrinsics

    def __len__(self):
        return len(self.rgb_list)

    def get_c2w(self, pose_fname):
        with open(pose_fname, 'r') as f:
            cam2world = f.readline().strip()
            cam2world = [float(t) for t in cam2world.split(' ')]
        c2w = torch.tensor(cam2world, dtype=torch.float32).reshape(4, 4)
        return c2w

    def gen_rays(self, c2w):
        # Generate rays
        self.h = self.reso
        self.w = self.reso
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
            indexing='ij')
        xx = (xx - self.cx) / self.fx
        yy = (yy - self.cy) / self.fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(1, -1, 3, 1)
        del xx, yy, zz
        dirs = (c2w[:, None, :3, :3] @ dirs)[..., 0]

        origins = c2w[:, None, :3, 3].expand(-1, self.h * self.w,
                                             -1).contiguous()
        origins = origins.view(-1, 3)
        dirs = dirs.view(-1, 3)

        return origins, dirs

    def read_depth(self, idx):
        depth_path = self.depth_list[idx]
        # image_path = os.path.join(depth_fname, self.image_names[index])
        exr = OpenEXR.InputFile(depth_path)
        header = exr.header()
        size = (header['dataWindow'].max.x - header['dataWindow'].min.x + 1,
                header['dataWindow'].max.y - header['dataWindow'].min.y + 1)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        depth_str = exr.channel('B', FLOAT)
        depth = np.frombuffer(depth_str,
                              dtype=np.float32).reshape(size[1],
                                                        size[0])  # H W
        depth = np.nan_to_num(depth, posinf=0, neginf=0)
        depth = depth.reshape(size)

        def resize_depth_mask(depth_to_resize, resolution):
            depth_resized = cv2.resize(depth_to_resize,
                                       (resolution, resolution),
                                       interpolation=cv2.INTER_LANCZOS4)
            #    interpolation=cv2.INTER_AREA)
            return depth_resized > 0  # type: ignore

        fg_mask_reso = resize_depth_mask(depth, self.reso)
        fg_mask_sr = resize_depth_mask(depth, 128)

        # depth = cv2.resize(depth, (self.reso, self.reso),
        #                    interpolation=cv2.INTER_LANCZOS4)
        #    interpolation=cv2.INTER_AREA)
        # depth_mask = depth > 0
        # depth = np.expand_dims(depth, axis=0).reshape(size)
        # return torch.from_numpy(depth)
        return torch.from_numpy(depth), torch.from_numpy(
            fg_mask_reso), torch.from_numpy(fg_mask_sr)

    def load_bbox(self, mask):
        nonzero_value = torch.nonzero(mask)
        height, width = nonzero_value.max(dim=0)[0]
        top, left = nonzero_value.min(dim=0)[0]
        bbox = torch.tensor([top, left, height, width], dtype=torch.float32)
        return bbox

    def __getitem__(self, idx):
        rgb_fname = self.rgb_list[idx]
        pose_fname = self.pose_list[idx]

        raw_img = imageio.imread(rgb_fname)

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

        img = cv2.resize(raw_img, (self.reso, self.reso),
                         interpolation=cv2.INTER_LANCZOS4)
        #  interpolation=cv2.INTER_AREA)

        # img_sr = cv2.resize(raw_img, (512, 512), interpolation=cv2.INTER_AREA)
        # img_sr = cv2.resize(raw_img, (256, 256), interpolation=cv2.INTER_AREA) # just as refinement, since eg3d uses 64->128 final resolution
        # img_sr = cv2.resize(raw_img, (128, 128), interpolation=cv2.INTER_AREA) # just as refinement, since eg3d uses 64->128 final resolution
        img_sr = cv2.resize(
            raw_img, (128, 128), interpolation=cv2.INTER_LANCZOS4
        )  # just as refinement, since eg3d uses 64->128 final resolution

        # img = torch.from_numpy(img)[..., :3].permute(
        #     2, 0, 1) / 255.0  #[3, reso, reso]

        img = torch.from_numpy(img)[..., :3].permute(
            2, 0, 1
        ) / 127.5 - 1  #[3, reso, reso], normalize to [-1,1], follow triplane range

        img_sr = torch.from_numpy(img_sr)[..., :3].permute(
            2, 0, 1
        ) / 127.5 - 1  #[3, reso, reso], normalize to [-1,1], follow triplane range

        # c2w = self.get_c2w(pose_fname).reshape(1, 4, 4) #[1, 4, 4]
        # rays_o, rays_d = self.gen_rays(c2w)
        # return img_to_encoder, img, rays_o, rays_d, c2w.reshape(-1)

        c2w = self.get_c2w(pose_fname).reshape(16)  #[1, 4, 4] -> [1, 16]
        # c = np.concatenate([c2w, self.intrinsics], axis=0).reshape(25)  # 25, no '1' dim needed.
        c = torch.cat([c2w, torch.from_numpy(self.intrinsics)],
                      dim=0).reshape(25)  # 25, no '1' dim needed.
        ret_dict = {
            # 'rgb_fname': rgb_fname,
            'img_to_encoder': img_to_encoder,
            'img': img,
            'c': c,
            'img_sr': img_sr,
            # 'ins_name': self.data_ins_list[idx]
        }
        if self.load_depth:
            depth, depth_mask, depth_mask_sr = self.read_depth(idx)
            bbox = self.load_bbox(depth_mask)
            ret_dict.update({
                'depth': depth,
                'depth_mask': depth_mask,
                'depth_mask_sr': depth_mask_sr,
                'bbox': bbox
            })
        # rays_o, rays_d = self.gen_rays(c2w)
        # return img_to_encoder, img, c
        return ret_dict


class MultiViewDatasetforLMDB(MultiViewDataset):

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
                 overfitting_bs=-1):
        super().__init__(file_path, reso, reso_encoder, preprocess, classes,
                         load_depth, test, scene_scale, overfitting,
                         imgnet_normalize, dataset_size, overfitting_bs)

    def __len__(self):
        return super().__len__()
        # return 100 # for speed debug

    def __getitem__(self, idx):
        # ret_dict = super().__getitem__(idx)
        rgb_fname = self.rgb_list[idx]
        pose_fname = self.pose_list[idx]
        raw_img = imageio.imread(rgb_fname)[..., :3]

        if raw_img.shape[-1] == 4: # ! set bg to white
            alpha_mask = raw_img[..., -1:] / 255
            raw_img = alpha_mask * raw_img[..., :3] + (1-alpha_mask) * np.ones_like(raw_img[..., :3]) * 255
            raw_img = raw_img.astype(np.uint8)

        raw_img = cv2.resize(raw_img,
                                    (self.reso, self.reso),
                                    interpolation=cv2.INTER_LANCZOS4)

        c2w = self.get_c2w(pose_fname).reshape(16)  #[1, 4, 4] -> [1, 16]
        # c = np.concatenate([c2w, self.intrinsics], axis=0).reshape(25)  # 25, no '1' dim needed.
        c = torch.cat([c2w, torch.from_numpy(self.intrinsics)],
                      dim=0).reshape(25)  # 25, no '1' dim needed.

        depth, depth_mask, depth_mask_sr = self.read_depth(idx)
        bbox = self.load_bbox(depth_mask)
        ret_dict = {
            'raw_img': raw_img,
            'c': c,
            'depth': depth,
            # 'depth_mask': depth_mask, # 64x64 here?
            'bbox': bbox
        }
        return ret_dict


def load_data_dryrun(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        #   shuffle=True,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True):
    # st()
    dataset = MultiViewDataset(file_path,
                               reso,
                               reso_encoder,
                               test=False,
                               preprocess=preprocess,
                               load_depth=load_depth,
                               imgnet_normalize=imgnet_normalize)
    print('dataset size: {}'.format(len(dataset)))
    # st()
    # train_sampler = DistributedSampler(dataset=dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # shuffle=shuffle,
        drop_last=False,
    )
    # sampler=train_sampler)

    return loader


class NovelViewDataset(MultiViewDataset):
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
                 overfitting_bs=-1):
        super().__init__(file_path, reso, reso_encoder, preprocess, classes,
                         load_depth, test, scene_scale, overfitting,
                         imgnet_normalize, dataset_size, overfitting_bs)

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
    dataset_cls = MultiViewDatasetforLMDB

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
        prefetch_factor=2,
        # prefetch_factor=3,
        pin_memory=True,
        persistent_workers=True,
    )
    # sampler=train_sampler)

    # while True:
    #     yield from loader
    return loader, dataset.dataset_name, len(dataset)


class LMDBDataset(Dataset):

    def __init__(self, lmdb_path):
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            max_readers=32,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.num_samples = self.env.stat()['entries']
        # self.start_idx = self.env.stat()['start_idx']
        # self.end_idx = self.env.stat()['end_idx']

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            key = str(idx).encode('utf-8')
            value = txn.get(key)

        sample = pickle.loads(value)
        return sample


def resize_depth_mask(depth_to_resize, resolution):
    depth_resized = cv2.resize(depth_to_resize, (resolution, resolution),
                               interpolation=cv2.INTER_LANCZOS4)
    #    interpolation=cv2.INTER_AREA)
    return depth_resized, depth_resized > 0  # type: ignore


class LMDBDataset_MV(LMDBDataset):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 **kwargs):
        super().__init__(lmdb_path)

        self.reso_encoder = reso_encoder
        self.reso = reso

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

    def _post_process_sample(self, raw_img, depth):
        
        if raw_img.shape[-1] == 4: # ! set bg to white
            alpha_mask = raw_img[..., -1:] / 255
            raw_img = alpha_mask * raw_img[..., :3] + (1-alpha_mask) * np.ones_like(raw_img[..., :3]) * 255
            raw_img = raw_img.astype(np.uint8)

        # if raw_img.shape[-1] == 4: # ! set bg to white
        #     raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGBA2RGB)

        # img_to_encoder = cv2.resize(sample.pop('raw_img'),
        if raw_img.shape[0] != self.reso_encoder:
            img_to_encoder = cv2.resize(raw_img,
                                        (self.reso_encoder, self.reso_encoder),
                                        interpolation=cv2.INTER_LANCZOS4)
        else:
            img_to_encoder = raw_img

        # interpolation=cv2.INTER_AREA)
        # img_to_encoder = img_to_encoder[..., :
        #                                 3]  #[3, reso_encoder, reso_encoder]
        img_to_encoder = self.normalize(img_to_encoder)

        img = cv2.resize(raw_img, (self.reso, self.reso),
                         interpolation=cv2.INTER_LANCZOS4)

        # if img.shape[-1] == 4:
        #     alpha_mask = img[..., -1:] > 0
        #     img = alpha_mask * img[..., :3] + (1-alpha_mask) * np.ones_like(img[..., :3]) * 255

        img = torch.from_numpy(img)[..., :3].permute(
            2, 0, 1
        ) / 127.5 - 1  #[3, reso, reso], normalize to [-1,1], follow triplane range

        # img_sr = torch.from_numpy(raw_img)[..., :3].permute(
        #     2, 0, 1
        # ) / 127.5 - 1  #[3, reso, reso], normalize to [-1,1], follow triplane range

        # depth
        # fg_mask_reso = resize_depth_mask(sample['depth'], self.reso)
        depth_reso, fg_mask_reso = resize_depth_mask(depth, self.reso)

        return {
            # **sample,
            'img_to_encoder': img_to_encoder,
            'img': img,
            'depth_mask': fg_mask_reso,
            # 'img_sr': img_sr, 
            'depth': depth_reso,
            # ! no need to load img_sr for now
        }

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # do transformations online

        return self._post_process_sample(sample['raw_img'], sample['depth'])
        # return sample

def load_bytes(inp_bytes, dtype, shape):
    return np.frombuffer(inp_bytes, dtype=dtype).reshape(shape).copy()

# Function to decompress an image using gzip and open with imageio
def decompress_and_open_image_gzip(compressed_data, is_img=False, decompress=True, decompress_fn=gzip.decompress):
    # Decompress the image data using gzip
    if decompress:
        compressed_data = decompress_fn(compressed_data)

    # Read the decompressed image using imageio
    if is_img:
        compressed_data = imageio.v3.imread(io.BytesIO(compressed_data)).copy()
        # return image
    return compressed_data


# Function to decompress an array using gzip
def decompress_array(compressed_data, shape, dtype, decompress=True, decompress_fn=gzip.decompress):
    # Decompress the array data using gzip
    if decompress:
        # compressed_data = gzip.decompress(compressed_data)
        compressed_data = decompress_fn(compressed_data)

    # Convert the decompressed data to a NumPy array
    # arr = np.frombuffer(decompressed_data, dtype=dtype).reshape(shape)

    return load_bytes(compressed_data, dtype, shape)


class LMDBDataset_MV_Compressed(LMDBDataset_MV):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 **kwargs):
        super().__init__(lmdb_path, reso, reso_encoder, imgnet_normalize,
                         **kwargs)
        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8')) - 40

        self.load_image_fn = partial(decompress_and_open_image_gzip,
                                     is_img=True)

    def __len__(self):
        return self.length
    
    def _load_lmdb_data(self, idx):

        with self.env.begin(write=False) as txn:
            raw_img_key = f'{idx}-raw_img'.encode('utf-8')
            raw_img = self.load_image_fn(txn.get(raw_img_key))

            depth_key = f'{idx}-depth'.encode('utf-8')
            depth = decompress_array(txn.get(depth_key), (512,512), np.float32)

            c_key = f'{idx}-c'.encode('utf-8')
            c = decompress_array(txn.get(c_key), (25, ), np.float32)

            bbox_key = f'{idx}-bbox'.encode('utf-8')
            bbox = decompress_array(txn.get(bbox_key), (4, ), np.float32)

        return raw_img, depth, c, bbox

    def _load_lmdb_data_no_decompress(self, idx):

        with self.env.begin(write=False) as txn:
            raw_img_key = f'{idx}-raw_img'.encode('utf-8')
            # raw_img = txn.get(raw_img_key)
            raw_img = self.load_image_fn(txn.get(raw_img_key), decompress=False)

            depth_key = f'{idx}-depth'.encode('utf-8')
            depth = decompress_array(txn.get(depth_key), (512,512), np.float32, decompress=False)
            # depth = txn.get(depth_key), (512,512)

            # c_key = f'{idx}-c'.encode('utf-8')
            # c = txn.get(c_key), (25, ), np.float32

            # bbox_key = f'{idx}-bbox'.encode('utf-8')
            # bbox = txn.get(bbox_key)

            c_key = f'{idx}-c'.encode('utf-8')
            c = decompress_array(txn.get(c_key), (25, ), np.float32, decompress=False)

            bbox_key = f'{idx}-bbox'.encode('utf-8')
            bbox = decompress_array(txn.get(bbox_key), (4, ), np.float32, decompress=False)

        return raw_img, depth, c, bbox

    def __getitem__(self, idx):
        # sample = super(LMDBDataset).__getitem__(idx)

        # do gzip uncompress online
        '''
        raw_img, depth, c, bbox  = self._load_lmdb_data(idx)

        return {
            **self._post_process_sample(raw_img, depth), 'c': c,
            'bbox': bbox*(self.reso/64.0),
            # 'depth': depth,
        }
        '''
        raw_img, depth, c, bbox  = self._load_lmdb_data_no_decompress(idx)
        return None


class LMDBDataset_NV_Compressed(LMDBDataset_MV_Compressed):
    def __init__(self, lmdb_path, reso, reso_encoder, imgnet_normalize=True, **kwargs):
        super().__init__(lmdb_path, reso, reso_encoder, imgnet_normalize, **kwargs)
        self.instance_data_length = 50 # 

    def __getitem__(self, idx):
        input_view = super().__getitem__(
            idx)  # get previous input view results

        # get novel view of the same instance
        try:
            novel_view = super().__getitem__(
                (idx // self.instance_data_length) * self.instance_data_length +
                random.randint(0, self.instance_data_length - 1))
        except Exception as e:
            raise NotImplementedError(idx)

        assert input_view['ins_name'] == novel_view['ins_name'], 'should sample novel view from the same instance'

        input_view.update({f'nv_{k}': v for k, v in novel_view.items()})
        return input_view