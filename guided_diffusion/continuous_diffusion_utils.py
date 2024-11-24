# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import logging
import os
import math
import shutil
import time
import sys
import types

import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
# from util.distributions import PixelNormal
from torch.cuda.amp import autocast

# from tensorboardX import SummaryWriter


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ExpMovingAvgrageMeter(object):

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.avg = 0

    def update(self, val):
        self.avg = (1. - self.momentum) * self.avg + self.momentum * val


class DummyDDP(nn.Module):
    def __init__(self, model):
        super(DummyDDP, self).__init__()
        self.module = model

    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)


def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class Logger(object):
    def __init__(self, rank, save):
        # other libraries may set logging before arriving at this line.
        # by reloading logging, we can get rid of previous configs set by other libraries.
        from importlib import reload
        reload(logging)
        self.rank = rank
        if self.rank == 0:
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(save, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            self.start_time = time.time()

    def info(self, string, *args):
        if self.rank == 0:
            elapsed_time = time.time() - self.start_time
            elapsed_time = time.strftime(
                '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
            if isinstance(string, str):
                string = elapsed_time + string
            else:
                logging.info(elapsed_time)
            logging.info(string, *args)


class Writer(object):
    def __init__(self, rank, save):
        self.rank = rank
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=save, flush_secs=20)

    def add_scalar(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_scalar(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_figure(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_image(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_histogram(*args, **kwargs)

    def add_histogram_if(self, write, *args, **kwargs):
        if write and False:  # Used for debugging.
            self.add_histogram(*args, **kwargs)

    def close(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.close()


def common_init(rank, seed, save_dir):
    # we use different seeds per gpu. But we sync the weights after model initialization.
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    torch.backends.cudnn.benchmark = True

    # prepare logging and tensorboard summary
    logging = Logger(rank, save_dir)
    writer = Writer(rank, save_dir)

    return logging, writer


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def get_stride_for_cell_type(cell_type):
    if cell_type.startswith('normal') or cell_type.startswith('combiner'):
        stride = 1
    elif cell_type.startswith('down'):
        stride = 2
    elif cell_type.startswith('up'):
        stride = -1
    else:
        raise NotImplementedError(cell_type)

    return stride


def get_cout(cin, stride):
    if stride == 1:
        cout = cin
    elif stride == -1:
        cout = cin // 2
    elif stride == 2:
        cout = 2 * cin

    return cout


def kl_balancer_coeff(num_scales, groups_per_scale, fun):
    if fun == 'equal':
        coeff = torch.cat([torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'linear':
        coeff = torch.cat([(2 ** i) * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)],
                          dim=0).cuda()
    elif fun == 'sqrt':
        coeff = torch.cat(
            [np.sqrt(2 ** i) * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)],
            dim=0).cuda()
    elif fun == 'square':
        coeff = torch.cat(
            [np.square(2 ** i) / groups_per_scale[num_scales - i - 1] * torch.ones(groups_per_scale[num_scales - i - 1])
             for i in range(num_scales)], dim=0).cuda()
    else:
        raise NotImplementedError
    # convert min to 1.
    coeff /= torch.min(coeff)
    return coeff


def kl_per_group(kl_all):
    kl_vals = torch.mean(kl_all, dim=0)
    kl_coeff_i = torch.abs(kl_all)
    kl_coeff_i = torch.mean(kl_coeff_i, dim=0, keepdim=True) + 0.01

    return kl_coeff_i, kl_vals


def kl_balancer(kl_all, kl_coeff=1.0, kl_balance=False, alpha_i=None):
    if kl_balance and kl_coeff < 1.0:
        alpha_i = alpha_i.unsqueeze(0)

        kl_all = torch.stack(kl_all, dim=1)
        kl_coeff_i, kl_vals = kl_per_group(kl_all)
        total_kl = torch.sum(kl_coeff_i)

        kl_coeff_i = kl_coeff_i / alpha_i * total_kl
        kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
        kl = torch.sum(kl_all * kl_coeff_i.detach(), dim=1)

        # for reporting
        kl_coeffs = kl_coeff_i.squeeze(0)
    else:
        kl_all = torch.stack(kl_all, dim=1)
        kl_vals = torch.mean(kl_all, dim=0)
        # kl = torch.sum(kl_all, dim=1)
        # kl = torch.mean(kl_all, dim=1)
        kl = torch.mean(kl_all)
        kl_coeffs = torch.ones(size=(len(kl_vals),))

    return kl_coeff * kl, kl_coeffs, kl_vals


def kl_per_group_vada(all_log_q, all_neg_log_p):
    assert len(all_log_q) == len(all_neg_log_p)

    kl_all_list = []
    kl_diag = []
    for log_q, neg_log_p in zip(all_log_q, all_neg_log_p):
        # kl_diag.append(torch.mean(torch.sum(neg_log_p + log_q, dim=[2, 3]), dim=0))
        kl_diag.append(torch.mean(torch.mean(neg_log_p + log_q, dim=[2, 3]), dim=0))
        # kl_all_list.append(torch.sum(neg_log_p + log_q, dim=[1, 2, 3]))
        kl_all_list.append(torch.mean(neg_log_p + log_q, dim=[1, 2, 3]))

    # kl_all = torch.stack(kl_all, dim=1)   # batch x num_total_groups
    kl_vals = torch.mean(torch.stack(kl_all_list, dim=1), dim=0)   # mean per group

    return kl_all_list, kl_vals, kl_diag


def kl_coeff(step, total_step, constant_step, min_kl_coeff, max_kl_coeff):
    # return max(min(max_kl_coeff * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)
    return max(min(min_kl_coeff + (max_kl_coeff - min_kl_coeff) * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)


def log_iw(decoder, x, log_q, log_p, crop=False):
    recon = reconstruction_loss(decoder, x, crop)
    return - recon - log_q + log_p


def reconstruction_loss(decoder, x, crop=False):
    from util.distributions import DiscMixLogistic

    recon = decoder.log_p(x)
    if crop:
        recon = recon[:, :, 2:30, 2:30]

    if isinstance(decoder, DiscMixLogistic):
        return - torch.sum(recon, dim=[1, 2])  # summation over RGB is done.
    else:
        return - torch.sum(recon, dim=[1, 2, 3])


def vae_terms(all_log_q, all_eps):
    from util.distributions import log_p_standard_normal

    # compute kl
    kl_all = []
    kl_diag = []
    log_p, log_q = 0., 0.
    for log_q_conv, eps in zip(all_log_q, all_eps):
        log_p_conv = log_p_standard_normal(eps)
        kl_per_var = log_q_conv - log_p_conv
        kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3]), dim=0))
        kl_all.append(torch.sum(kl_per_var, dim=[1, 2, 3]))
        log_q += torch.sum(log_q_conv, dim=[1, 2, 3])
        log_p += torch.sum(log_p_conv, dim=[1, 2, 3])
    return log_q, log_p, kl_all, kl_diag


def sum_log_q(all_log_q):
    log_q = 0.
    for log_q_conv in all_log_q:
        log_q += torch.sum(log_q_conv, dim=[1, 2, 3])

    return log_q


def cross_entropy_normal(all_eps):
    from util.distributions import log_p_standard_normal

    cross_entropy = 0.
    neg_log_p_per_group = []
    for eps in all_eps:
        neg_log_p_conv = - log_p_standard_normal(eps)
        neg_log_p = torch.sum(neg_log_p_conv, dim=[1, 2, 3])
        cross_entropy += neg_log_p
        neg_log_p_per_group.append(neg_log_p_conv)

    return cross_entropy, neg_log_p_per_group


def tile_image(batch_image, n, m=None):
    if m is None:
        m = n
    assert n * m == batch_image.size(0)
    channels, height, width = batch_image.size(1), batch_image.size(2), batch_image.size(3)
    batch_image = batch_image.view(n, m, channels, height, width)
    batch_image = batch_image.permute(2, 0, 3, 1, 4)  # n, height, n, width, c
    batch_image = batch_image.contiguous().view(channels, n * height, m * width)
    return batch_image


def average_gradients_naive(params, is_distributed):
    """ Gradient averaging. """
    if is_distributed:
        size = float(dist.get_world_size())
        for param in params:
            if param.requires_grad:
                param.grad.data /= size
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)


def average_gradients(params, is_distributed):
    """ Gradient averaging. """
    if is_distributed:
        if isinstance(params, types.GeneratorType):
            params = [p for p in params]

        size = float(dist.get_world_size())
        grad_data = []
        grad_size = []
        grad_shapes = []
        # Gather all grad values
        for param in params:
            if param.requires_grad:
                grad_size.append(param.grad.data.numel())
                grad_shapes.append(list(param.grad.data.shape))
                grad_data.append(param.grad.data.flatten())
        grad_data = torch.cat(grad_data).contiguous()

        # All-reduce grad values
        grad_data /= size
        dist.all_reduce(grad_data, op=dist.ReduceOp.SUM)

        # Put back the reduce grad values to parameters
        base = 0
        for i, param in enumerate(params):
            if param.requires_grad:
                param.grad.data = grad_data[base:base + grad_size[i]].view(grad_shapes[i])
                base += grad_size[i]


def average_params(params, is_distributed):
    """ parameter averaging. """
    if is_distributed:
        size = float(dist.get_world_size())
        for param in params:
            param.data /= size
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)


def average_tensor(t, is_distributed):
    if is_distributed:
        size = float(dist.get_world_size())
        dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
        t.data /= size


def broadcast_params(params, is_distributed):
    if is_distributed:
        for param in params:
            dist.broadcast(param.data, src=0)


def num_output(dataset):
    if dataset in {'mnist',  'omniglot'}:
        return 28 * 28
    elif dataset == 'cifar10':
        return 3 * 32 * 32
    elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
        size = int(dataset.split('_')[-1])
        return 3 * size * size
    elif dataset == 'ffhq':
        return 3 * 256 * 256
    else:
        raise NotImplementedError


def get_input_size(dataset):
    if dataset in {'mnist', 'omniglot'}:
        return 32
    elif dataset == 'cifar10':
        return 32
    elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
        size = int(dataset.split('_')[-1])
        return size
    elif dataset == 'ffhq':
        return 256
    else:
        raise NotImplementedError


def get_bpd_coeff(dataset):
    n = num_output(dataset)
    return 1. / np.log(2.) / n


def get_channel_multiplier(dataset, num_scales):
    if dataset in {'cifar10', 'omniglot'}:
        mult = (1, 1, 1)
    elif dataset in {'celeba_256', 'ffhq', 'lsun_church_256'}:
        if num_scales == 3:
            mult = (1, 1, 1)        # used for prior at 16
        elif num_scales == 4:
            mult = (1, 2, 2, 2)     # used for prior at 32
        elif num_scales == 5:
            mult = (1, 1, 2, 2, 2)  # used for prior at 64
    elif dataset == 'mnist':
        mult = (1, 1)
    else:
        raise NotImplementedError

    return mult


def get_attention_scales(dataset):
    if dataset in {'cifar10', 'omniglot'}:
        attn = (True, False, False)
    elif dataset in {'celeba_256', 'ffhq', 'lsun_church_256'}:
        # attn = (False, True, False, False) # used for 32
        attn = (False, False, True, False, False)  # used for 64
    elif dataset == 'mnist':
        attn = (True, False)
    else:
        raise NotImplementedError

    return attn


def change_bit_length(x, num_bits):
    if num_bits != 8:
        x = torch.floor(x * 255 / 2 ** (8 - num_bits))
        x /= (2 ** num_bits - 1)
    return x


def view4D(t, size, inplace=True):
    """
     Equal to view(-1, 1, 1, 1).expand(size)
     Designed because of this bug:
     https://github.com/pytorch/pytorch/pull/48696
    """
    if inplace:
        return t.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1).expand(size)
    else:
        return t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(size)


def get_arch_cells(arch_type, use_se):
    if arch_type == 'res_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['down_enc'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_dec'] = {'conv_branch': ['mconv_e6k5g0'], 'se': use_se}
        arch_cells['up_dec'] = {'conv_branch': ['mconv_e6k5g0'], 'se': use_se}
        arch_cells['normal_pre'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['down_pre'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_post'] = {'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
        arch_cells['up_post'] = {'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_bnswish':
        arch_cells = dict()
        arch_cells['normal_enc'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['down_enc'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_dec'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['up_dec'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_pre'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['down_pre'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_post'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['up_post'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_bnswish2':
        arch_cells = dict()
        arch_cells['normal_enc'] = {'conv_branch': ['res_bnswish_x2'], 'se': use_se}
        arch_cells['down_enc'] = {'conv_branch': ['res_bnswish_x2'], 'se': use_se}
        arch_cells['normal_dec'] = {'conv_branch': ['res_bnswish_x2'], 'se': use_se}
        arch_cells['up_dec'] = {'conv_branch': ['res_bnswish_x2'], 'se': use_se}
        arch_cells['normal_pre'] = {'conv_branch': ['res_bnswish_x2'], 'se': use_se}
        arch_cells['down_pre'] = {'conv_branch': ['res_bnswish_x2'], 'se': use_se}
        arch_cells['normal_post'] = {'conv_branch': ['res_bnswish_x2'], 'se': use_se}
        arch_cells['up_post'] = {'conv_branch': ['res_bnswish_x2'], 'se': use_se}
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_mbconv_attn':
        arch_cells = dict()
        arch_cells['normal_enc'] = {'conv_branch': ['res_bnswish', 'res_bnswish', ], 'se': use_se, 'attn_type': 'attn'}
        arch_cells['down_enc'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se, 'attn_type': 'attn'}
        arch_cells['normal_dec'] = {'conv_branch': ['mconv_e6k5g0'], 'se': use_se, 'attn_type': 'attn'}
        arch_cells['up_dec'] = {'conv_branch': ['mconv_e6k5g0'], 'se': use_se, 'attn_type': 'attn'}
        arch_cells['normal_pre'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['down_pre'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_post'] = {'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
        arch_cells['up_post'] = {'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_mbconv_attn_half':
        arch_cells = dict()
        arch_cells['normal_enc'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['down_enc'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_dec'] = {'conv_branch': ['mconv_e6k5g0'], 'se': use_se, 'attn_type': 'attn'}
        arch_cells['up_dec'] = {'conv_branch': ['mconv_e6k5g0'], 'se': use_se, 'attn_type': 'attn'}
        arch_cells['normal_pre'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['down_pre'] = {'conv_branch': ['res_bnswish', 'res_bnswish'], 'se': use_se}
        arch_cells['normal_post'] = {'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
        arch_cells['up_post'] = {'conv_branch': ['mconv_e3k5g0'], 'se': use_se}
        arch_cells['ar_nn'] = ['']
    else:
        raise NotImplementedError

    return arch_cells


def groups_per_scale(num_scales, num_groups_per_scale):
    g = []
    n = num_groups_per_scale
    for s in range(num_scales):
        assert n >= 1
        g.append(n)
    return g


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, scale):
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.scale = scale

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1
        timesteps = timesteps * self.scale
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class RandomFourierEmbedding(nn.Module):
    def __init__(self, embedding_dim, scale):
        super(RandomFourierEmbedding, self).__init__()
        self.w = nn.Parameter(torch.randn(size=(1, embedding_dim // 2)) * scale, requires_grad=False)

    def forward(self, timesteps):
        emb = torch.mm(timesteps[:, None], self.w * 2 * 3.14159265359)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


def init_temb_fun(embedding_type, embedding_scale, embedding_dim):
    if embedding_type == 'positional':
        temb_fun = PositionalEmbedding(embedding_dim, embedding_scale)
    elif embedding_type == 'fourier':
        temb_fun = RandomFourierEmbedding(embedding_dim, embedding_scale)
    else:
        raise NotImplementedError

    return temb_fun

def get_dae_model(args, num_input_channels):
    if args.dae_arch == 'ncsnpp':
        # we need to import NCSNpp after processes are launched on the multi gpu training.
        from score_sde.ncsnpp import NCSNpp
        dae = NCSNpp(args, num_input_channels)
    else:
        raise NotImplementedError

    return dae

def symmetrize_image_data(images):
    return 2.0 * images - 1.0


def unsymmetrize_image_data(images):
    return (images + 1.) / 2.


def normalize_symmetric(images):
    """
    Normalize images by dividing the largest intensity. Used for visualizing the intermediate steps.
    """
    b = images.shape[0]
    m, _ = torch.max(torch.abs(images).view(b, -1), dim=1)
    images /= (m.view(b, 1, 1, 1) + 1e-3)

    return images


@torch.jit.script
def soft_clamp5(x: torch.Tensor):
    return x.div(5.).tanh_().mul(5.)  # 5. * torch.tanh(x / 5.) <--> soft differentiable clamp between [-5, 5]

@torch.jit.script
def soft_clamp(x: torch.Tensor, a: torch.Tensor):
    return x.div(a).tanh_().mul(a)

class SoftClamp5(nn.Module):
    def __init__(self):
        super(SoftClamp5, self).__init__()

    def forward(self, x):
        return soft_clamp5(x)


def override_architecture_fields(args, stored_args, logging):
    # list of architecture parameters used in NVAE:
    architecture_fields = ['arch_instance', 'num_nf', 'num_latent_scales', 'num_groups_per_scale',
                           'num_latent_per_group', 'num_channels_enc', 'num_preprocess_blocks',
                           'num_preprocess_cells', 'num_cell_per_cond_enc', 'num_channels_dec',
                           'num_postprocess_blocks', 'num_postprocess_cells', 'num_cell_per_cond_dec',
                           'decoder_dist', 'num_x_bits', 'log_sig_q_scale',
                           'progressive_input_vae', 'channel_mult']

    # backward compatibility
    """ We have broken backward compatibility. No need to se these manually
    if not hasattr(stored_args, 'log_sig_q_scale'):
        logging.info('*** Setting %s manually ****', 'log_sig_q_scale')
        setattr(stored_args, 'log_sig_q_scale', 5.)

    if not hasattr(stored_args, 'latent_grad_cutoff'):
        logging.info('*** Setting %s manually ****', 'latent_grad_cutoff')
        setattr(stored_args, 'latent_grad_cutoff', 0.)

    if not hasattr(stored_args, 'progressive_input_vae'):
        logging.info('*** Setting %s manually ****', 'progressive_input_vae')
        setattr(stored_args, 'progressive_input_vae', 'none')

    if not hasattr(stored_args, 'progressive_output_vae'):
        logging.info('*** Setting %s manually ****', 'progressive_output_vae')
        setattr(stored_args, 'progressive_output_vae', 'none')
    """

    if not hasattr(stored_args, 'num_x_bits'):
        logging.info('*** Setting %s manually ****', 'num_x_bits')
        setattr(stored_args, 'num_x_bits', 8)

    if not hasattr(stored_args, 'channel_mult'):
        logging.info('*** Setting %s manually ****', 'channel_mult')
        setattr(stored_args, 'channel_mult', [1, 2])

    for f in architecture_fields:
        if not hasattr(args, f) or getattr(args, f) != getattr(stored_args, f):
            logging.info('Setting %s from loaded checkpoint', f)
            setattr(args, f, getattr(stored_args, f))


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    dist.barrier()
    dist.destroy_process_group()


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape, device='cuda') * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y, device='cuda')


def trace_df_dx_hutchinson(f, x, noise, no_autograd):
    """
    Hutchinson's trace estimator for Jacobian df/dx, O(1) call to autograd
    """
    if no_autograd:
        # the following is compatible with checkpointing
        torch.sum(f * noise).backward()
        # torch.autograd.backward(tensors=[f], grad_tensors=[noise])
        jvp = x.grad
        trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
        x.grad = None
    else:
        jvp = torch.autograd.grad(f, x, noise, create_graph=False)[0]
        trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
        # trJ = torch.einsum('bijk,bijk->b', jvp, noise)  # we could test if there's a speed difference in einsum vs sum

    return trJ

def different_p_q_objectives(iw_sample_p, iw_sample_q):
    assert iw_sample_p in ['ll_uniform', 'drop_all_uniform', 'll_iw', 'drop_all_iw', 'drop_sigma2t_iw', 'rescale_iw',
                           'drop_sigma2t_uniform']
    assert iw_sample_q in ['reweight_p_samples', 'll_uniform', 'll_iw']
    # In these cases, we reuse the likelihood-based p-objective (either the uniform sampling version or the importance
    # sampling version) also for q.
    if iw_sample_p in ['ll_uniform', 'll_iw'] and iw_sample_q == 'reweight_p_samples':
        return False
    # In these cases, we are using a non-likelihood-based objective for p, and hence definitly need to use another q
    # objective.
    else:
        return True


# def decoder_output(dataset, logits, fixed_log_scales=None):
#     if dataset in {'cifar10', 'celeba_64', 'celeba_256', 'imagenet_32', 'imagenet_64', 'ffhq',
#                    'lsun_bedroom_128', 'lsun_bedroom_256', 'mnist', 'omniglot',
#                    'lsun_church_256'}:
#         return PixelNormal(logits, fixed_log_scales)
#     else:
#         raise NotImplementedError


def get_mixed_prediction(mixed_prediction, param, mixing_logit, mixing_component=None):
    if mixed_prediction:
        assert mixing_component is not None, 'Provide mixing component when mixed_prediction is enabled.'
        coeff = torch.sigmoid(mixing_logit)
        param = (1 - coeff) * mixing_component + coeff * param

    return param


def set_vesde_sigma_max(args, vae, train_queue, logging, is_distributed):
    logging.info('')
    logging.info('Calculating max. pairwise distance in latent space to set sigma2_max for VESDE...')

    eps_list = []
    vae.eval()
    for step, x in enumerate(train_queue):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()
        x = symmetrize_image_data(x)

        # run vae
        with autocast(enabled=args.autocast_train):
            with torch.set_grad_enabled(False):
                logits, all_log_q, all_eps = vae(x)
                eps = torch.cat(all_eps, dim=1)

        eps_list.append(eps.detach())

    # concat eps tensor on each GPU and then gather all on all GPUs
    eps_this_rank = torch.cat(eps_list, dim=0)
    if is_distributed:
        eps_all_gathered = [torch.zeros_like(eps_this_rank)] * dist.get_world_size()
        dist.all_gather(eps_all_gathered, eps_this_rank)
        eps_full = torch.cat(eps_all_gathered, dim=0)
    else:
        eps_full = eps_this_rank

    # max pairwise distance squared between all latent encodings, is computed on CPU
    eps_full = eps_full.cpu().float()
    eps_full = eps_full.flatten(start_dim=1).unsqueeze(0)
    max_pairwise_dist_sqr = torch.cdist(eps_full, eps_full).square().max()
    max_pairwise_dist_sqr = max_pairwise_dist_sqr.cuda()

    # to be safe, we broadcast to all GPUs if we are in distributed environment. Shouldn't be necessary in principle.
    if is_distributed:
        dist.broadcast(max_pairwise_dist_sqr, src=0)

    args.sigma2_max = max_pairwise_dist_sqr.item()

    logging.info('Done! Set args.sigma2_max set to {}'.format(args.sigma2_max))
    logging.info('')
    return args


def mask_inactive_variables(x, is_active):
    x = x * is_active
    return x


def common_x_operations(x, num_x_bits):
    x = x[0] if len(x) > 1 else x
    x = x.cuda()

    # change bit length
    x = change_bit_length(x, num_x_bits)
    x = symmetrize_image_data(x)

    return x
