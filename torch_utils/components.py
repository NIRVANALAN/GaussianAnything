# https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py
# https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py#L812

import copy
import math
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from pathlib import Path
from random import random

from einops import rearrange, repeat, reduce, pack, unpack

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import einsum, nn
from beartype.typing import List, Union
from beartype import beartype
from tqdm.auto import tqdm
from pdb import set_trace as st

# helper functions, from:
# https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def divisible_by(numer, denom):
    return (numer % denom) == 0


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)

    return inner


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def compact(input_dict):
    return {key: value for key, value in input_dict.items() if exists(value)}


def maybe_transform_dict_key(input_dict, key, fn):
    if key not in input_dict:
        return input_dict

    copied_dict = input_dict.copy()
    copied_dict[key] = fn(copied_dict[key])
    return copied_dict


def cast_uint8_images_to_float(images):
    if not images.dtype == torch.uint8:
        return images
    return images / 255


def module_device(module):
    return next(module.parameters()).device


def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def pad_tuple_to_length(t, length, fillvalue=None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue, ) * remain_length))


# helper classes


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


# tensor helpers


def log(t, eps: float = 1e-12):
    return torch.log(t.clamp(min=eps))


def l2norm(t):
    return F.normalize(t, dim=-1)


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1, ) * padding_dims))


def masked_mean(t, *, dim, mask=None):
    if not exists(mask):
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)


def resize_image_to(image,
                    target_image_size,
                    clamp_range=None,
                    mode='nearest'):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    out = F.interpolate(image, target_image_size, mode=mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out


def calc_all_frame_dims(downsample_factors: List[int], frames):
    if not exists(frames):
        return (tuple(), ) * len(downsample_factors)

    all_frame_dims = []

    for divisor in downsample_factors:
        assert divisible_by(frames, divisor)
        all_frame_dims.append((frames // divisor, ))

    return all_frame_dims


def safe_get_tuple_index(tup, index, default=None):
    if len(tup) <= index:
        return default
    return tup[index]


# image normalization functions
# ddpms expect images to be in the range of -1 to 1


def normalize_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5


# def Upsample(dim, dim_out=None):
#     dim_out = default(dim_out, dim)

#     return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
#                          nn.Conv2d(dim, dim_out, 3, padding=1))



class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(conv, nn.SiLU(), nn.PixelShuffle(2))

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 dim_inter=None,
                 use_norm=True,
                 norm_layer=nn.BatchNorm2d,
                 bias=False):
        super().__init__()
        if dim_inter is None:
            dim_inter = dim_out

        if use_norm:
            self.conv = nn.Sequential(
                norm_layer(dim_in),
                nn.ReLU(True),
                nn.Conv2d(dim_in,
                          dim_inter,
                          3,
                          1,
                          1,
                          bias=bias,
                          padding_mode='reflect'),
                norm_layer(dim_inter),
                nn.ReLU(True),
                nn.Conv2d(dim_inter,
                          dim_out,
                          3,
                          1,
                          1,
                          bias=bias,
                          padding_mode='reflect'),
            )
        else:
            self.conv = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(dim_in, dim_inter, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(dim_inter, dim_out, 3, 1, 1),
            )

        self.short_cut = None
        if dim_in != dim_out:
            self.short_cut = nn.Conv2d(dim_in, dim_out, 1, 1)

    def forward(self, feats):
        feats_out = self.conv(feats)
        if self.short_cut is not None:
            feats_out = self.short_cut(feats) + feats_out
        else:
            feats_out = feats_out + feats
        return feats_out


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class PixelUnshuffleUpsample(nn.Module):
    def __init__(self, output_dim, num_feat=128, num_out_ch=3, sr_ratio=4, *args, **kwargs) -> None:
        super().__init__()

        self.conv_after_body = nn.Conv2d(output_dim, output_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(output_dim, num_feat, 3, 1, 1),
            nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(sr_ratio, num_feat)  # 4 time SR
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x, input_skip_connection=True, *args, **kwargs):
        # x = self.conv_first(x)
        if input_skip_connection:
            x = self.conv_after_body(x) + x
        else:
            x = self.conv_after_body(x)

        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))
        return x


class Conv3x3TriplaneTransformation(nn.Module):
    # used in the final layer before triplane
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        self.conv_after_unpachify = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.conv_before_rendering = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True))

    def forward(self, unpachified_latent):
        latent = self.conv_after_unpachify(unpachified_latent) # no residual connections here
        latent = self.conv_before_rendering(latent) + latent
        return latent


# https://github.com/JingyunLiang/SwinIR/blob/6545850fbf8df298df73d81f3e8cba638787c8bd/models/network_swinir.py#L750
class NearestConvSR(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, output_dim, num_feat=128, num_out_ch=3, sr_ratio=4, *args, **kwargs) -> None:
        super().__init__()

        self.upscale = sr_ratio

        self.conv_after_body = nn.Conv2d(output_dim, output_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(output_dim, num_feat, 3, 1, 1),
                                                    nn.LeakyReLU(inplace=True))
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if self.upscale == 4:
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
    
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, *args, **kwargs):

        # x = self.conv_first(x)
        x = self.conv_after_body(x) + x
        x = self.conv_before_upsample(x)
        x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        if self.upscale == 4:
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.conv_last(self.lrelu(self.conv_hr(x)))


        return x

# https://github.com/yumingj/C2-Matching/blob/fa171ca6707c6f16a5d04194ce866ea70bb21d2b/mmsr/models/archs/ref_restoration_arch.py#L65
class NearestConvSR_Residual(NearestConvSR):
    # learn residual + normalize
    
    def __init__(self, output_dim, num_feat=128, num_out_ch=3, sr_ratio=4, *args, **kwargs) -> None:
        super().__init__(output_dim, num_feat, num_out_ch, sr_ratio, *args, **kwargs)
        # self.mean = torch.Tensor((0.485, 0.456, 0.406)).view(1,3,1,1) # imagenet mean
        self.act = nn.Tanh()

    def forward(self, x, base_x, *args, **kwargs):
        # base_x: low-resolution 3D rendering, for residual addition
        # self.mean = self.mean.type_as(x)
        # x = super().forward(x).clamp(-1,1) 
        x = super().forward(x)
        x = self.act(x) # residual normalize to [-1,1]
        scale = x.shape[-1] // base_x.shape[-1] # 2 or 4
        x = x + F.interpolate(base_x, None, scale, 'bilinear', False) # add residual; [-1,1] range

        # return x  + 2 * self.mean
        return x
    
class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops

# class PixelShuffledDirect(nn.Module):
