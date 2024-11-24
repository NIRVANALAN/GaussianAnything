# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Streaming images and labels from datasets created with dataset_tool.py."""

import cv2
import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
from torchvision import transforms

from pdb import set_trace as st

from .shapenet import LMDBDataset_MV_Compressed, decompress_array

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------


# copide from eg3d/train.py
def init_dataset_kwargs(data,
                        class_name='datasets.eg3d_dataset.ImageFolderDataset',
                        reso_gt=128):
    # try:
    # if data == 'None':
    #     dataset_kwargs = dnnlib.EasyDict({})  #
    #     dataset_kwargs.name = 'eg3d_dataset'
    #     dataset_kwargs.resolution = 128
    #     dataset_kwargs.use_labels = False
    #     dataset_kwargs.max_size = 70000
    #     return dataset_kwargs, 'eg3d_dataset'

    dataset_kwargs = dnnlib.EasyDict(class_name=class_name,
                                     reso_gt=reso_gt,
                                     path=data,
                                     use_labels=True,
                                     max_size=None,
                                     xflip=False)
    dataset_obj = dnnlib.util.construct_class_by_name(
        **dataset_kwargs)  # Subclass of training.dataset.Dataset.
    dataset_kwargs.resolution = dataset_obj.resolution  # Be explicit about resolution.
    dataset_kwargs.use_labels = dataset_obj.has_labels  # Be explicit about labels.
    dataset_kwargs.max_size = len(
        dataset_obj)  # Be explicit about dataset size.

    return dataset_kwargs, dataset_obj.name
    # except IOError as err:
    #     raise click.ClickException(f'--data: {err}')


class Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            name,  # Name of the dataset.
            raw_shape,  # Shape of the raw image data (NCHW).
            reso_gt=128,
            max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
            use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
            xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
            random_seed=0,  # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # self.reso_gt = 128
        self.reso_gt = reso_gt  # ! hard coded
        self.reso_encoder = 224

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        # self._raw_idx = np.arange(self.__len__(), dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate(
                [self._xflip, np.ones_like(self._xflip)])

        # dino encoder normalizer
        self.normalize_for_encoder_input = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.Resize(size=(self.reso_encoder, self.reso_encoder),
                              antialias=True),  # type: ignore
        ])

        self.normalize_for_gt = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(size=(self.reso_gt, self.reso_gt),
                              antialias=True),  # type: ignore
        ])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels(
            ) if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0],
                                            dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            # assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size
        # return self._get_raw_labels().shape[0]

    def __getitem__(self, idx):
        # print(self._raw_idx[idx], idx)

        matte = self._load_raw_matte(self._raw_idx[idx])
        assert isinstance(matte, np.ndarray)
        assert list(matte.shape)[1:] == self.image_shape[1:]
        if self._xflip[idx]:
            assert matte.ndim == 1  # CHW
            matte = matte[:, :, ::-1]
        # matte_orig = matte.copy().astype(np.float32) / 255
        matte_orig = matte.copy().astype(np.float32) # segmentation version
        # assert matte_orig.max() == 1
        matte = np.transpose(matte,
                            #  (1, 2, 0)).astype(np.float32) / 255  # [0,1] range
                             (1, 2, 0)).astype(np.float32)  # [0,1] range
        matte = cv2.resize(matte, (self.reso_gt, self.reso_gt),
                           interpolation=cv2.INTER_NEAREST)
        assert matte.min() >= 0 and matte.max(
        ) <= 1, f'{matte.min(), matte.max()}'

        if matte.ndim == 3:  # H, W
            matte = matte[..., 0]

        image = self._load_raw_image(self._raw_idx[idx])

        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]

        # blending
        # blending = True
        blending = False
        if blending:
            image = image * matte_orig + (1 - matte_orig) * cv2.GaussianBlur(
                image, (5, 5), cv2.BORDER_DEFAULT)
            # image = image * matte_orig

        image = np.transpose(image, (1, 2, 0)).astype(
            np.float32
        ) / 255  # H W C for torchvision process, normalize to [0,1]

        image_sr = torch.from_numpy(image)[..., :3].permute(
            2, 0, 1) * 2 - 1  # normalize to [-1,1]
        image_to_encoder = self.normalize_for_encoder_input(image)

        image_gt = cv2.resize(image, (self.reso_gt, self.reso_gt),
                              interpolation=cv2.INTER_AREA)
        image_gt = torch.from_numpy(image_gt)[..., :3].permute(
            2, 0, 1) * 2 - 1  # normalize to [-1,1]

        return dict(
            c=self.get_label(idx),
            img_to_encoder=image_to_encoder,  # 224
            img_sr=image_sr,  # 512
            img=image_gt,  # [-1,1] range
            # depth=torch.zeros_like(image_gt)[0, ...] # type: ignore
            depth=matte,
            depth_mask=matte,
            # depth_mask=matte > 0,
            # alpha=matte,
        )  # return dict here

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


#----------------------------------------------------------------------------


class ImageFolderDataset(Dataset):

    def __init__(
            self,
            path,  # Path to directory or zip.
            resolution=None,  # Ensure specific resolution, None = highest available.
            reso_gt=128,
            **super_kwargs,  # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._matte_path = path.replace('unzipped_ffhq_512',
                                        'unzipped_ffhq_matte')
        # self._matte_path = path.replace('unzipped_ffhq_512',
        #                                 'ffhq_512_seg')
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self._path)
                for root, _dirs, files in os.walk(self._path)
                for fname in files
            }
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(
            fname for fname in self._all_fnames
            if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(
            self._load_raw_image(0).shape)
        # raw_shape = [len(self._image_fnames)] + list(
        #     self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution
                                       or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name,
                         raw_shape=raw_shape,
                         reso_gt=reso_gt,
                         **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def _open_matte_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._matte_path, fname), 'rb')
        # if self._type == 'zip':
        #     return self._get_zipfile().open(fname, 'r')
        # return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_matte(self, raw_idx):
        # ! from seg version
        fname = self._image_fnames[raw_idx]
        with self._open_matte_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        # if image.max() != 1:
        image = (image > 0).astype(np.float32) # process segmentation
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_matte_orig(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_matte_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        st() # process segmentation
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            # st()
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels_ = []
        for fname, _ in labels.items():
            # if 'mirror' not in fname:
            labels_.append(labels[fname])
        labels = labels_
        # !
        # labels = [
        #     labels[fname.replace('\\', '/')] for fname in self._image_fnames
        # ]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        self._raw_labels = labels
        return labels


#----------------------------------------------------------------------------


# class ImageFolderDatasetUnzipped(ImageFolderDataset):

#     def __init__(self, path, resolution=None, **super_kwargs):
#         super().__init__(path, resolution, **super_kwargs)


# class ImageFolderDatasetPose(ImageFolderDataset):

#     def __init__(
#             self,
#             path,  # Path to directory or zip.
#             resolution=None,  # Ensure specific resolution, None = highest available.
#             **super_kwargs,  # Additional arguments for the Dataset base class.
#     ):
#         super().__init__(path, resolution, **super_kwargs)
#         # only return labels

#     def __len__(self):
#         return self._raw_idx.size
#         # return self._get_raw_labels().shape[0]

#     def __getitem__(self, idx):
#         # image = self._load_raw_image(self._raw_idx[idx])
#         # assert isinstance(image, np.ndarray)
#         # assert list(image.shape) == self.image_shape
#         # assert image.dtype == np.uint8
#         # if self._xflip[idx]:
#         # assert image.ndim == 3  # CHW
#         # image = image[:, :, ::-1]
#         return dict(c=self.get_label(idx), )  # return dict here


class ImageFolderDatasetLMDB(ImageFolderDataset):
    def __init__(self, path, resolution=None, reso_gt=128, **super_kwargs):
        super().__init__(path, resolution, reso_gt, **super_kwargs)
    
    def __getitem__(self, idx):
        # print(self._raw_idx[idx], idx)

        matte = self._load_raw_matte(self._raw_idx[idx])
        assert isinstance(matte, np.ndarray)
        assert list(matte.shape)[1:] == self.image_shape[1:]
        if self._xflip[idx]:
            assert matte.ndim == 1  # CHW
            matte = matte[:, :, ::-1]
        # matte_orig = matte.copy().astype(np.float32) / 255
        matte_orig = matte.copy().astype(np.float32) # segmentation version
        assert matte_orig.max() <= 1 # some ffhq images are dirty, so may be all zero
        matte = np.transpose(matte,
                            #  (1, 2, 0)).astype(np.float32) / 255  # [0,1] range
                             (1, 2, 0)).astype(np.float32)  # [0,1] range

        # ! load 512 matte
        # matte = cv2.resize(matte, (self.reso_gt, self.reso_gt),
        #                    interpolation=cv2.INTER_NEAREST)

        assert matte.min() >= 0 and matte.max(
        ) <= 1, f'{matte.min(), matte.max()}'

        if matte.ndim == 3:  # H, W
            matte = matte[..., 0]

        image = self._load_raw_image(self._raw_idx[idx])

        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]

        # blending
        # blending = True
        # blending = False
        # if blending:
        #     image = image * matte_orig + (1 - matte_orig) * cv2.GaussianBlur(
        #         image, (5, 5), cv2.BORDER_DEFAULT)
            # image = image * matte_orig

        # image = np.transpose(image, (1, 2, 0)).astype(
        #     np.float32
        # ) / 255  # H W C for torchvision process, normalize to [0,1]

        # image_sr = torch.from_numpy(image)[..., :3].permute(
        #     2, 0, 1) * 2 - 1  # normalize to [-1,1]
        # image_to_encoder = self.normalize_for_encoder_input(image)

        # image_gt = cv2.resize(image, (self.reso_gt, self.reso_gt),
        #                       interpolation=cv2.INTER_AREA)
        # image_gt = torch.from_numpy(image_gt)[..., :3].permute(
        #     2, 0, 1) * 2 - 1  # normalize to [-1,1]

        return dict(
            c=self.get_label(idx),
            # img_to_encoder=image_to_encoder,  # 224
            # img_sr=image_sr,  # 512
            img=image,  # [-1,1] range
            # depth=torch.zeros_like(image_gt)[0, ...] # type: ignore
            # depth=matte,
            depth_mask=matte,
        )  # return dict here

class LMDBDataset_MV_Compressed_eg3d(LMDBDataset_MV_Compressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 **kwargs):
        super().__init__(lmdb_path, reso, reso_encoder, imgnet_normalize,
                         **kwargs)

        self.normalize_for_encoder_input = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.Resize(size=(self.reso_encoder, self.reso_encoder),
                              antialias=True),  # type: ignore
        ])

        self.normalize_for_gt = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(size=(self.reso, self.reso),
                              antialias=True),  # type: ignore
        ])

    def __getitem__(self, idx):
        # sample = super(LMDBDataset).__getitem__(idx)

        # do gzip uncompress online
        with self.env.begin(write=False) as txn:
            img_key = f'{idx}-img'.encode('utf-8')
            image = self.load_image_fn(txn.get(img_key))

            depth_key = f'{idx}-depth_mask'.encode('utf-8')
            # depth = decompress_array(txn.get(depth_key), (512,512), np.float32)
            depth = decompress_array(txn.get(depth_key), (64,64), np.float32)

            c_key = f'{idx}-c'.encode('utf-8')
            c = decompress_array(txn.get(c_key), (25, ), np.float32)

        # ! post processing, e.g., normalizing
        depth = cv2.resize(depth, (self.reso, self.reso),
                           interpolation=cv2.INTER_NEAREST)

        image = np.transpose(image, (1, 2, 0)).astype(
            np.float32
        ) / 255  # H W C for torchvision process, normalize to [0,1]

        image_sr = torch.from_numpy(image)[..., :3].permute(
            2, 0, 1) * 2 - 1  # normalize to [-1,1]
        image_to_encoder = self.normalize_for_encoder_input(image)

        image_gt = cv2.resize(image, (self.reso, self.reso),
                              interpolation=cv2.INTER_AREA)
        image_gt = torch.from_numpy(image_gt)[..., :3].permute(
            2, 0, 1) * 2 - 1  # normalize to [-1,1]


        return {
            'img_to_encoder': image_to_encoder,  # 224
            'img_sr': image_sr,  # 512
            'img': image_gt,  # [-1,1] range
            'c': c,
            'depth': depth,
            'depth_mask': depth,
        }
