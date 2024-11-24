EPS = 1e-7

import kornia
from typing import Dict, Iterator, List, Optional, Tuple, Union
import torchvision
from guided_diffusion import dist_util, logger
from pdb import set_trace as st
from torch.nn import functional as F
import numpy as np
import torch
import torch.nn as nn
import lpips

from . import *

from .sdfstudio_losses import ScaleAndShiftInvariantLoss
from ldm.util import default, instantiate_from_config
from .vqperceptual import hinge_d_loss, vanilla_d_loss
from torch.autograd import Variable

from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Main loss function used for ZoeDepth. Copy/paste from AdaBins repo (https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7)
def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction


class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""

    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self,
                input,
                target,
                mask=None,
                interpolate=True,
                return_interpolated=False):
        # input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(input,
                                              target.shape[-2:],
                                              mode='bilinear',
                                              align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        # with torch.amp.autocast(enabled=False):  # amp causes NaNs in this loss function

        alpha = 1e-7
        g = torch.log(input + alpha) - torch.log(target + alpha)

        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

        loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss

        return loss, intr_input


def get_outnorm(x: torch.Tensor, out_norm: str = '') -> torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if 'b' in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if 'c' in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[-3]
    if 'i' in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1] * img_shape[-2]

    return norm


class CharbonnierLoss(torch.nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, out_norm: str = 'bci'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm

    def forward(self, x, y):
        norm = get_outnorm(x, self.out_norm)
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss * norm


def feature_vae_loss(feature):
    # kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    # feature dim: B C H W
    mu = feature.mean(1)
    var = feature.var(1)
    log_var = torch.log(var)
    kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - var, dim=1), dim=0)
    return kld


def kl_coeff(step, total_step, constant_step, min_kl_coeff, max_kl_coeff):
    # return max(min(max_kl_coeff * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)
    kl_lambda = max(
        min(
            min_kl_coeff + (max_kl_coeff - min_kl_coeff) *
            (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)
    return torch.tensor(kl_lambda, device=dist_util.dev())


def depth_smoothness_loss(alpha_pred, depth_pred):
    # from PesonNeRF paper.
    # all Tensor shape B 1 H W
    geom_loss = (
        alpha_pred[..., :-1] * alpha_pred[..., 1:] * (
            depth_pred[..., :-1] - depth_pred[..., 1:]  # W dim
        ).square()).mean()  # mean of ([8, 1, 64, 63])

    geom_loss += (alpha_pred[..., :-1, :] * alpha_pred[..., 1:, :] *
                  (depth_pred[..., :-1, :] - depth_pred[..., 1:, :]).square()
                  ).mean()  # H dim, ([8, 1, 63, 64])

    return geom_loss


# https://github.com/elliottwu/unsup3d/blob/master/unsup3d/networks.py#L140
class LPIPSLoss(torch.nn.Module):

    def __init__(
        self,
        loss_weight=1.0,
        use_input_norm=True,
        range_norm=True,
        # n1p1_input=True,
    ):
        super(LPIPSLoss, self).__init__()
        # self.perceptual = lpips.LPIPS(net="alex", spatial=False).eval()
        self.perceptual = lpips.LPIPS(net="vgg", spatial=False).eval()
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        # if self.use_input_norm:
        #     # the mean is for image with range [0, 1]
        #     self.register_buffer(
        #         'mean',
        #         torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        #     # the std is for image with range [0, 1]
        #     self.register_buffer(
        #         'std',
        #         torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target, conf_sigma_percl=None):
        # st()
        # ! add large image support, only sup 128x128 patch
        lpips_loss = self.perceptual(target.contiguous(), pred.contiguous())
        return self.loss_weight * lpips_loss.mean()


# mask-aware perceptual loss
class PerceptualLoss(nn.Module):

    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)

        vgg_pretrained_features = torchvision.models.vgg16(
            pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        out = x / 2 + 0.5
        out = (out - self.mean_rgb.view(1, 3, 1, 1)) / self.std_rgb.view(
            1, 3, 1, 1)
        return out

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = torch.cat([im1, im2], 0)
        im = self.normalize(im)  # normalize input

        ## compute features
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        for f1, f2 in feats[2:3]:  # use relu3_3 features only
            loss = (f1 - f2)**2
            if conf_sigma is not None:
                loss = loss / (2 * conf_sigma**2 + EPS) + (conf_sigma +
                                                           EPS).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm // h, wm // w
                mask0 = nn.functional.avg_pool2d(mask,
                                                 kernel_size=(sh, sw),
                                                 stride=(sh,
                                                         sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)


# add confidence support, unsup3d version
def photometric_loss_laplace(im1, im2, mask=None, conf_sigma=None):
    loss = (im1 - im2).abs()
    # loss = (im1 - im2).square()
    if conf_sigma is not None:
        loss = loss * 2**0.5 / (conf_sigma + EPS) + (conf_sigma + EPS).log()

    if mask is not None:
        mask = mask.expand_as(loss)
        loss = (loss * mask).sum() / mask.sum()

    else:
        loss = loss.mean()

    return loss


# gaussian likelihood version, What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
# also used in the mask-aware vgg loss
def photometric_loss(im1, im2, mask=None, conf_sigma=None):
    # loss = torch.nn.functional.mse_loss(im1, im2, reduce='none')
    loss = (im1 - im2).square()

    if conf_sigma is not None:
        loss = loss / (2 * conf_sigma**2 + EPS) + (conf_sigma + EPS).log()

    if mask is not None:
        mask = mask.expand_as(loss)
        loss = (loss * mask).sum() / mask.sum()

    else:
        loss = loss.mean()

    return loss


class E3DGELossClass(torch.nn.Module):

    def __init__(self, device, opt) -> None:
        super().__init__()

        self.opt = opt
        self.device = device
        self.criterionImg = {
            'mse': torch.nn.MSELoss(),
            'l1': torch.nn.L1Loss(),
            'charbonnier': CharbonnierLoss(),
        }[opt.color_criterion]

        self.criterion_latent = {
            'mse': torch.nn.MSELoss(),
            'l1': torch.nn.L1Loss(),
            'vae': feature_vae_loss
        }[opt.latent_criterion]

        # self.criterionLPIPS = LPIPS(net_type='alex', device=device).eval()
        if opt.lpips_lambda > 0:
            self.criterionLPIPS = LPIPSLoss(loss_weight=opt.lpips_lambda)
        # self.criterionLPIPS = torch.nn.MSELoss()

        if opt.id_lambda > 0:
            self.criterionID = IDLoss(device=device).eval()
        self.id_loss_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

        # define 3d rec loss, for occupancy
        # self.criterion3d_rec = torch.nn.SmoothL1Loss(reduction='none')
        # self.criterion_alpha = torch.nn.SmoothL1Loss()

        # self.criterion3d_rec = torch.nn.MSELoss(reduction='none')
        self.criterion_alpha = torch.nn.L1Loss()

        if self.opt.xyz_lambda > 0:
            # self.criterion_xyz = torch.nn.SmoothL1Loss()
            self.criterion_xyz = torch.nn.L1Loss() # follow LION, but noisy xyz here...

        if self.opt.depth_lambda > 0:
        # ! this depth loss not converging, no idea why
            self.criterion3d_rec = ScaleAndShiftInvariantLoss(alpha=0.5,
                                                              scales=1)
        else:
            self.criterion3d_rec = torch.nn.SmoothL1Loss(reduction='none')

        #     self.silog_loss = SILogLoss()

        if self.opt.lambda_opa_reg > 0:
            # self.beta_mvp_dist = torch.distributions.beta.Beta(torch.tensor(0.5, device=device), torch.tensor(0.5, device=device))
            # self.beta_mvp_base_dist = torch.distributions.beta.Beta(torch.tensor(10, device=device), torch.tensor(0.5, device=device)) # force close to 1 for base
            # self.beta_mvp_base_dist = torch.distributions.beta.Beta(torch.tensor(0.6, device=device), torch.tensor(0.2, device=device)) # force close to 1 for base
            self.beta_mvp_base_dist = torch.distributions.beta.Beta(torch.tensor(0.5, device=device), torch.tensor(0.25, device=device)) # force close to 1 for base

        logger.log('init loss class finished', )

    def calc_scale_invariant_depth_loss(self, pred_depth: torch.Tensor,
                                        gt_depth: torch.Tensor,
                                        gt_depth_mask: torch.Tensor):
        """apply 3d shape reconstruction supervision. Basically supervise the depth with L1 loss
        """

        shape_loss_dict = {}
        assert gt_depth_mask is not None
        shape_loss = self.criterion3d_rec(pred_depth, gt_depth, gt_depth_mask)

        # if shape_loss > 0.2:  # hinge loss, avoid ood gradient
        #     shape_loss = torch.zeros_like(shape_loss)
        # else:
        shape_loss = shape_loss.clamp(0.04) # g-buffer depth is very noisy
        shape_loss *= self.opt.depth_lambda

        shape_loss_dict['loss_depth'] = shape_loss
        # shape_loss_dict['depth_fgratio'] = gt_depth_mask.mean()

        # return l_si, shape_loss_dict
        return shape_loss, shape_loss_dict

    def calc_depth_loss(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor,
                        gt_depth_mask: torch.Tensor):
        """apply 3d shape reconstruction supervision. Basically supervise the depth with L1 loss
        """

        shape_loss_dict = {}
        shape_loss = self.criterion3d_rec(pred_depth, gt_depth)
        assert gt_depth_mask is not None

        shape_loss *= gt_depth_mask
        shape_loss = shape_loss.sum() / gt_depth_mask.sum()

        # else:
        #     shape_loss /= pred_depth.numel()
        # l_si = self.silog_loss(pred_depth, gt_depth, mask=None, interpolate=True, return_interpolated=False)

        # l_si *= self.opt.depth_lambda
        # shape_loss_dict['loss_depth'] = l_si
        shape_loss_dict['loss_depth'] = shape_loss.clamp(
            min=0, max=0.1) * self.opt.depth_lambda

        # shape_loss_dict['loss_depth'] = shape_loss.clamp(
        #     min=0, max=0.5) * self.opt.depth_lambda

        # return l_si, shape_loss_dict
        return shape_loss, shape_loss_dict

    @torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False)
    def calc_alpha_loss(self, pred_alpha, gt_depth_mask):
        # return self.criterionImg(alpha, gt_depth_mask.float())

        if gt_depth_mask.ndim == 3:
            gt_depth_mask = gt_depth_mask.unsqueeze(1)

        if gt_depth_mask.shape[1] == 3:
            gt_depth_mask = gt_depth_mask[:, 0:1, ...]  # B 1 H W

        assert pred_alpha.shape == gt_depth_mask.shape

        alpha_loss = self.criterion_alpha(pred_alpha, gt_depth_mask)
        # st()

        return alpha_loss

    @torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False)
    def calc_mask_mse_loss(
            self,
            input,
            gt,
            gt_depth_mask,
            #    conf_sigma=None,
            conf_sigma_l1=None,
            #    conf_sigma_percl=None,
            use_fg_ratio=False):
        if gt_depth_mask.ndim == 3:
            gt_depth_mask = gt_depth_mask.unsqueeze(1).repeat_interleave(3, 1)
        else:
            assert gt_depth_mask.shape == input.shape
        gt_depth_mask = gt_depth_mask.float()

        if conf_sigma_l1 is None:
            rec_loss = torch.nn.functional.mse_loss(
                input.float(), gt.float(),
                reduction='none')  # 'sum' already divide by batch size n
        else:
            rec_loss = photometric_loss(
                input, gt, gt_depth_mask, conf_sigma_l1
            )  # ! only cauclate laplace on the foreground, or bg confidence low, large gradient.
            return rec_loss
            # rec_loss = torch.nn.functional.l1_loss( # for laplace loss
            #     input.float(), gt.float(),
            #     reduction='none')  # 'sum' already divide by batch size n
        # gt_depth_mask = torch.ones_like(gt_depth_mask) # ! DEBUGGING

        # if conf_sigma is not None: # from unsup3d, but a L2 version
        #     rec_loss = rec_loss * 2**0.5 / (conf_sigma + EPS) + (conf_sigma +
        #                                                          EPS).log()
        #     return rec_loss.mean()
        # rec_loss = torch.exp(-(rec_loss * 2**0.5 / (conf_sigma + EPS))) * 1/(conf_sigma +
        #                                                      EPS) / (2**0.5)

        fg_size = gt_depth_mask.sum()
        # fg_ratio = fg_size / torch.ones_like(gt_depth_mask).sum() if use_fg_ratio else 1
        fg_loss = rec_loss * gt_depth_mask
        fg_loss = fg_loss.sum() / fg_size  # * fg_ratio

        if self.opt.bg_lamdba > 0:
            bg_loss = rec_loss * (1 - gt_depth_mask)
            bg_loss = bg_loss.sum() / (1 - gt_depth_mask).sum()
            rec_loss = fg_loss + bg_loss * self.opt.bg_lamdba
        else:
            rec_loss = fg_loss

        return rec_loss

    @torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False)
    def calc_2d_rec_loss(
        self,
        input,
        gt,
        depth_fg_mask,
        test_mode=True,
        step=1,
        ignore_lpips=False,
        # conf_sigma=None,
        conf_sigma_l1=None,
        conf_sigma_percl=None,
        pred_alpha=None,
    ):
        opt = self.opt
        loss_dict = {}

        # logger.log(test_mode)
        # logger.log(input.min(), input.max(), gt.min(), gt.max())
        if test_mode or not opt.fg_mse:
            rec_loss = self.criterionImg(input, gt)
        else:
            rec_loss = self.calc_mask_mse_loss(
                input,
                gt,
                depth_fg_mask,
                conf_sigma_l1=conf_sigma_l1,
            )
            #   conf_sigma_percl=conf_sigma_percl)
            #    conf_sigma)

        # if step == 300:
        #     st()

        if opt.lpips_lambda > 0 and step >= opt.lpips_delay_iter and not ignore_lpips:  # tricky solution to avoid NAN in LPIPS loss

            # with torch.autocast(device_type='cuda',
            #                     dtype=torch.float16,
            #                     enabled=False):
            # if test_mode or not opt.fg_mse:  # no need to calculate background lpips for ease of computation
            # inp_for_lpips = input * pred_alpha + torch.ones_like(input) * (1-pred_alpha)
            # gt_for_lpips = gt * depth_fg_mask + torch.ones_like(gt) * (1-depth_fg_mask)
            
            inp_for_lpips = input * pred_alpha
            gt_for_lpips = gt * depth_fg_mask

            width = input.shape[-1]
            if width == 192: # triplane here
                lpips_loss = self.criterionLPIPS( # loss on 128x128 center crop
                    inp_for_lpips[:, :, width//2-64:width//2+64, width//2-64:width//2+64],
                    gt_for_lpips[:, :, width//2-64:width//2+64, width//2-64:width//2+64],
                    conf_sigma_percl=conf_sigma_percl,
                )
            elif width >256:
            # elif width >192:
                # lpips_loss = self.criterionLPIPS(
                #     F.interpolate(inp_for_lpips, (256,256), mode='bilinear'),
                #     F.interpolate(gt_for_lpips, (256,256), mode='bilinear'),
                #     conf_sigma_percl=conf_sigma_percl,
                # )

                # patch = 80

                # patch = 128
                patch = 144
                middle_point = width // 2
                lpips_loss = self.criterionLPIPS( # loss on 128x128 center crop
                    inp_for_lpips[:, :, middle_point-patch:middle_point+patch, middle_point-patch:middle_point+patch],
                    gt_for_lpips[:, :,  middle_point-patch:middle_point+patch, middle_point-patch:middle_point+patch],
                    conf_sigma_percl=conf_sigma_percl,
                )

            else: # directly supervise when <= 256
                # ! add foreground mask
                assert pred_alpha is not None
                lpips_loss = self.criterionLPIPS(
                    inp_for_lpips,
                    gt_for_lpips,
                    # conf_sigma_percl=conf_sigma_percl,
                )
            # else:  # fg lpips
            #     assert depth_fg_mask.shape == input.shape
            #     lpips_loss = self.criterionLPIPS(
            #         input.contiguous() * depth_fg_mask,
            #         gt.contiguous() * depth_fg_mask).mean()
        else:
            lpips_loss = torch.tensor(0., device=input.device)

        if opt.ssim_lambda > 0:
            loss_ssim = self.ssim_loss(input, gt)  #?
        else:
            loss_ssim = torch.tensor(0., device=input.device)

        loss_psnr = self.psnr((input / 2 + 0.5), (gt / 2 + 0.5), 1.0)

        if opt.id_lambda > 0:
            loss_id = self._calc_loss_id(input, gt)
        else:
            loss_id = torch.tensor(0., device=input.device)

        if opt.l1_lambda > 0:
            loss_l1 = F.l1_loss(input, gt)
        else:
            loss_l1 = torch.tensor(0., device=input.device)

        # loss = rec_loss * opt.l2_lambda + lpips_loss * opt.lpips_lambda + loss_id * opt.id_lambda + loss_ssim * opt.ssim_lambda
        rec_loss = rec_loss * opt.l2_lambda
        loss = rec_loss + lpips_loss + loss_id * opt.id_lambda + loss_ssim * opt.ssim_lambda + opt.l1_lambda * loss_l1

        # if return_dict:
        loss_dict['loss_l2'] = rec_loss
        loss_dict['loss_id'] = loss_id
        loss_dict['loss_lpips'] = lpips_loss
        loss_dict['loss'] = loss
        loss_dict['loss_ssim'] = loss_ssim

        # metrics to report, not involved in training
        loss_dict['mae'] = loss_l1
        loss_dict['PSNR'] = loss_psnr
        loss_dict['SSIM'] = 1 - loss_ssim  # Todo
        loss_dict['ID_SIM'] = 1 - loss_id

        return loss, loss_dict

    @torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False)
    def calc_shape_rec_loss(
        self,
        pred_shape: dict,
        gt_shape: dict,
        device,
    ):
        """apply 3d shape reconstruction supervision. Basically supervise the densities with L1 loss

        Args:
            pred_shape (dict): dict contains reconstructed shape information
            gt_shape (dict): dict contains gt shape information
            supervise_sdf (bool, optional): whether supervise sdf rec. Defaults to True.
            supervise_surface_normal (bool, optional): whether supervise surface rec. Defaults to False.

        Returns:
            dict: shape reconstruction loss
        """

        shape_loss_dict = {}
        shape_loss = 0
        # assert supervise_sdf or supervise_surface_normal, 'should at least supervise one types of shape reconstruction'
        # todo, add weights

        if self.opt.shape_uniform_lambda > 0:
            shape_loss_dict['coarse'] = self.criterion3d_rec(
                pred_shape['coarse_densities'].squeeze(),
                gt_shape['coarse_densities'].squeeze())
            shape_loss += shape_loss_dict[
                'coarse'] * self.opt.shape_uniform_lambda

        if self.opt.shape_importance_lambda > 0:
            shape_loss_dict['fine'] = self.criterion3d_rec(
                pred_shape['fine_densities'].squeeze(),  # ? how to supervise
                gt_shape['fine_densities'].squeeze())
            shape_loss += shape_loss_dict[
                'fine'] * self.opt.shape_importance_lambda

        loss_depth = self.criterion_alpha(pred_shape['image_depth'],
                                          gt_shape['image_depth'])

        shape_loss += loss_depth * self.opt.shape_depth_lambda
        shape_loss_dict.update(dict(loss_depth=loss_depth))
        # TODO, add on surface pts supervision ?

        return shape_loss, shape_loss_dict

    @torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False)
    def psnr(self, input, target, max_val):
        return kornia.metrics.psnr(input, target, max_val)

    # @torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False)
    def ssim_loss(self, img1, img2, window_size=11, size_average=True):
        channel = img1.size(-3)
        window = create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return 1 - _ssim(img1, img2, window, window_size, channel, size_average)

    @torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False)
    def forward(self,
                pred,
                gt,
                test_mode=True,
                step=1,
                return_fg_mask=False,
                conf_sigma_l1=None,
                conf_sigma_percl=None,
                ignore_kl=False,
                ignore_lpips=False,
                *args,
                **kwargs):

        with torch.autocast(device_type='cuda',
                            dtype=torch.float16,
                            enabled=False):

            loss = torch.tensor(0., device=self.device)
            loss_dict = {}

            if 'image_mask' in pred:
                pred_alpha = pred['image_mask']  # B 1 H W
            else:
                N, _, H, W = pred['image_depth'].shape
                pred_alpha = pred['weights_samples'].permute(0, 2, 1).reshape(
                    N, 1, H, W)

            # balance rec_loss with logvar
            # if 'depth_mask' in gt:
            if self.opt.online_mask:
                # https://github.com/elliottwu/unsup3d/blob/dc961410d61684561f19525c2f7e9ee6f4dacb91/unsup3d/model.py#L193
                margin = (self.opt.max_depth - self.opt.min_depth) / 2
                fg_mask = (pred['image_depth']
                           < self.opt.max_depth + margin).float()  # B 1 H W
                fg_mask = fg_mask.repeat_interleave(3, 1).float()

            else:
                if 'depth_mask' in gt:
                    if gt['depth_mask'].shape[1] != 1:
                        fg_mask = gt['depth_mask'].unsqueeze(1)
                    else:
                        fg_mask = gt['depth_mask']
                    fg_mask = fg_mask.repeat_interleave(
                        3, 1).float()
                else:
                    fg_mask = None

            loss_2d, loss_2d_dict = self.calc_2d_rec_loss(
                pred['image_raw'],
                gt['img'],
                fg_mask,
                test_mode=test_mode,
                step=step,
                ignore_lpips=ignore_lpips,
                conf_sigma_l1=conf_sigma_l1,
                conf_sigma_percl=conf_sigma_percl,
                pred_alpha=pred_alpha,
                )
            #   ignore_lpips=self.opt.fg_mse)

            if self.opt.kl_lambda > 0 and not ignore_kl:
                # assert 'posterior' in pred, 'logvar' in pred
                assert 'posterior' in pred

                if self.opt.kl_anneal:
                    kl_lambda = kl_coeff(
                        step=step,
                        constant_step=5e3,  # 1w steps
                        total_step=25e3,  # 5w steps in total
                        min_kl_coeff=max(1e-9, self.opt.kl_lambda / 1e4),
                        max_kl_coeff=self.opt.kl_lambda)
                    loss_dict['kl_lambda'] = kl_lambda
                else:
                    loss_dict['kl_lambda'] = torch.tensor(
                        self.opt.kl_lambda, device=dist_util.dev())

                if self.opt.pt_ft_kl:
                    pt_kl, ft_kl = pred['posterior'].kl(pt_ft_separate=True)
                    kl_batch = pt_kl.shape[0]

                    # loss_dict['kl_loss_pt'] = pt_kl.sum() * loss_dict['kl_lambda'] * 0.01 / kl_batch
                    loss_dict['kl_loss_pt'] = pt_kl.sum() * loss_dict['kl_lambda'] * 0 # no compression at all.
                    loss_dict['kl_loss_ft'] = ft_kl.sum() * loss_dict['kl_lambda'] / kl_batch
                    loss = loss + loss_dict['kl_loss_pt'] + loss_dict['kl_loss_ft']

                    loss_dict['latent_mu_pt'] = pred['posterior'].mean[:, :3].mean()
                    loss_dict['latent_std_pt'] = pred['posterior'].std[:, :3].mean()

                    loss_dict['latent_mu_ft'] = pred['posterior'].mean[:, 3:].mean()
                    loss_dict['latent_std_ft'] = pred['posterior'].std[:, 3:].mean()

                elif self.opt.ft_kl:

                    ft_kl = pred['posterior'].kl(ft_separate=True)
                    kl_batch = ft_kl.shape[0]

                    loss_dict['kl_loss_ft'] = ft_kl.sum() * loss_dict['kl_lambda'] / kl_batch
                    loss = loss + loss_dict['kl_loss_ft']

                    loss_dict['latent_mu_ft'] = pred['posterior'].mean[:, :].square().mean().float().detach()
                    loss_dict['latent_std_ft'] = pred['posterior'].std[:, :].mean().float().detach()

                else:
                    
                    kl_loss = pred['posterior'].kl()
                    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                    loss_dict['kl_loss'] = kl_loss * loss_dict['kl_lambda']
                    loss += loss_dict['kl_loss']
                    loss_dict['latent_mu'] = pred['posterior'].mean.mean()
                    loss_dict['latent_std'] = pred['posterior'].std.mean()

                # nll_loss = loss_2d / torch.exp(pred['logvar']) + pred['logvar'] # nll_loss
                nll_loss = loss_2d
                loss += nll_loss

                loss_dict.update(dict(nll_loss=nll_loss))

                # loss_dict['latent_mu'] = pred['latent_normalized'].mean()
                # loss_dict['latent_max'] = pred['latent_normalized'].max()
                # loss_dict['latent_min'] = pred['latent_normalized'].min()
                # loss_dict['latent_std'] = pred['latent_normalized'].std()
                #  pred[
                #     'latent_normalized_2Ddiffusion'].mean()
                # loss_dict['latent_std'] = pred[
                #     'latent_normalized_2Ddiffusion'].std()
                # loss_dict['latent_max'] = pred[
                #     'latent_normalized_2Ddiffusion'].max()
                # loss_dict['latent_min'] = pred[
                #     'latent_normalized_2Ddiffusion'].min()

            else:
                loss += loss_2d

            # if 'image_sr' in pred and pred['image_sr'].shape==gt['img_sr']:
            if 'image_sr' in pred:

                if 'depth_mask_sr' in gt:
                    depth_mask_sr = gt['depth_mask_sr'].unsqueeze(
                        1).repeat_interleave(3, 1).float()
                else:
                    depth_mask_sr = None

                loss_sr, loss_sr_dict = self.calc_2d_rec_loss(
                    pred['image_sr'],
                    gt['img_sr'],
                    depth_fg_mask=depth_mask_sr,
                    # test_mode=test_mode,
                    test_mode=True,
                    step=step)
                loss_sr_lambda = 1
                if step < self.opt.sr_delay_iter:
                    loss_sr_lambda = 0
                loss += loss_sr * loss_sr_lambda
                for k, v in loss_sr_dict.items():
                    loss_dict['sr_' + k] = v * loss_sr_lambda

            if self.opt.depth_lambda > 0:  # TODO, switch to scale-agnostic depth loss
                assert 'depth' in gt
                pred_depth = pred['image_depth']
                if pred_depth.ndim == 4:
                    pred_depth = pred_depth.squeeze(1)  # B H W

                # _, shape_loss_dict = self.calc_depth_loss(
                #     pred_depth, gt['depth'], fg_mask[:, 0, ...])
                _, shape_loss_dict = self.calc_scale_invariant_depth_loss(
                    pred_depth, gt['depth'], fg_mask[:, 0, ...])
                loss += shape_loss_dict['loss_depth']
                loss_dict.update(shape_loss_dict)

            # if self.opt.latent_lambda > 0:  # make sure the latent suits diffusion learning
            #     latent_mu = pred['latent'].mean()
            #     loss_latent = self.criterion_latent(
            #         latent_mu, torch.zeros_like(
            #             latent_mu))  # only regularize the mean value here
            #     loss_dict['loss_latent'] = loss_latent
            #     loss += loss_latent * self.opt.latent_lambda

            if self.opt.alpha_lambda > 0 and 'image_depth' in pred:
                loss_alpha = self.calc_alpha_loss(pred_alpha, fg_mask)
                loss_dict['loss_alpha'] = loss_alpha * self.opt.alpha_lambda
                loss += loss_alpha * self.opt.alpha_lambda

            if self.opt.depth_smoothness_lambda > 0:
                loss_depth_smoothness = depth_smoothness_loss(
                    pred_alpha,
                    pred['image_depth']) * self.opt.depth_smoothness_lambda
                loss_dict['loss_depth_smoothness'] = loss_depth_smoothness
                loss += loss_depth_smoothness

            loss_2d_dict['all_loss'] = loss
            loss_dict.update(loss_2d_dict)

            # if return_fg_mask:
            return loss, loss_dict, fg_mask
            # else:
            #     return loss, loss_dict

    def _calc_loss_id(self, input, gt):
        if input.shape[-1] != 256:
            arcface_input = self.id_loss_pool(input)
            id_loss_gt = self.id_loss_pool(gt)
        else:
            arcface_input = input
            id_loss_gt = gt

        loss_id, _, _ = self.criterionID(arcface_input, id_loss_gt, id_loss_gt)

        return loss_id

    def calc_2d_rec_loss_misaligned(self, input, gt):
        """id loss + vgg loss

        Args:
            input (_type_): _description_
            gt (_type_): _description_
            depth_mask (_type_): _description_
            test_mode (bool, optional): _description_. Defaults to True.
        """
        opt = self.opt
        loss_dict = {}

        if opt.lpips_lambda > 0:
            with torch.autocast(
                    device_type='cuda', dtype=torch.float16,
                    enabled=False):  # close AMP for lpips to avoid nan
                lpips_loss = self.criterionLPIPS(input, gt)
        else:
            lpips_loss = torch.tensor(0., device=input.device)

        if opt.id_lambda > 0:
            loss_id = self._calc_loss_id(input, gt)
        else:
            loss_id = torch.tensor(0., device=input.device)

        loss_dict['loss_id_real'] = loss_id
        loss_dict['loss_lpips_real'] = lpips_loss

        loss = lpips_loss * opt.lpips_lambda + loss_id * opt.id_lambda

        return loss, loss_dict


class E3DGE_with_AdvLoss(E3DGELossClass):
    # adapted from sgm/modules/autoencoding/losses/discriminator_loss.py

    def __init__(
        self,
        device,
        opt,
        discriminator_config: Optional[Dict] = None,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_start: int = 0,
        disc_loss: str = "hinge",
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        regularization_weights: Union[None, Dict[str, float]] = None,
        dtype=torch.float32,
        # additional_log_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            device,
            opt,
        )

        # ! initialize GAN loss
        discriminator_config = default(
            discriminator_config,
            {
                "target":
                "nsr.losses.disc.NLayerDiscriminator",
                "params": {
                    "input_nc": disc_in_channels,
                    "n_layers": disc_num_layers,
                    "use_actnorm": False,
                },
            },
        )

        self.discriminator = instantiate_from_config(
            discriminator_config).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight # self.regularization_weights = default(regularization_weights, {})

        # self.forward_keys = [
        #     "optimizer_idx",
        #     "global_step",
        #     "last_layer",
        #     "split",
        #     "regularization_log",
        # ]

        # self.additional_log_keys = set(default(additional_log_keys, []))
        # self.additional_log_keys.update(set(
        #     self.regularization_weights.keys()))

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.discriminator.parameters()

    def forward(self,
                pred,
                gt,
                behaviour: str,
                test_mode=True,
                step=1,
                return_fg_mask=False,
                conf_sigma_l1=None,
                conf_sigma_percl=None,
                ignore_d_loss=False,
                *args,
                **kwargs):

        # now the GAN part
        reconstructions = pred['image_raw']
        inputs = gt['img']

        if behaviour == 'g_step':

            nll_loss, loss_dict, fg_mask = super().forward(
                pred,
                gt,
                test_mode,
                step, 
                return_fg_mask,
                conf_sigma_l1,
                conf_sigma_percl,
                *args,
                **kwargs)

            # generator update
            if not ignore_d_loss and (step >= self.discriminator_iter_start or not self.training):
                logits_fake = self.discriminator(reconstructions.contiguous())
                g_loss = -torch.mean(logits_fake)
                if self.training:
                    d_weight = torch.tensor(self.discriminator_weight)
                    # d_weight = self.calculate_adaptive_weight(
                    #     nll_loss, g_loss, last_layer=last_layer)
                else:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, requires_grad=True)
            
            g_loss = g_loss * d_weight * self.disc_factor

            loss = nll_loss + g_loss

            # TODO
            loss_dict.update({
                f"loss/g": g_loss.detach().mean(),
            })

            # return loss, log
            return loss, loss_dict, fg_mask

        elif behaviour == 'd_step' and not ignore_d_loss:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(
                reconstructions.contiguous().detach())

            if step >= self.discriminator_iter_start or not self.training:
                d_loss = self.disc_factor * self.disc_loss(
                    logits_real, logits_fake)
            else:
                d_loss = torch.tensor(0.0, requires_grad=True)

            loss_dict = {}

            loss_dict.update({
                "loss/disc": d_loss.clone().detach().mean(),
                "logits/real": logits_real.detach().mean(),
                "logits/fake": logits_fake.detach().mean(),
            })

            return d_loss, loss_dict, None
        else:
            raise NotImplementedError(f"Unknown optimizer behaviour {behaviour}")
