# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli as Bern
import numpy as np
from pdb import set_trace as st
# from util import utils
from .continuous_diffusion_utils import view4D

@torch.jit.script
def sample_normal_jit(mu, sigma):
    rho = mu.mul(0).normal_()
    z = rho.mul_(sigma).add_(mu)
    return z, rho


@torch.jit.script
def log_p_standard_normal(samples):
    log_p = - 0.5 * torch.square(samples) - 0.9189385332  # 0.5 * np.log(2 * np.pi)
    return log_p


def log_p_var_normal(samples, var):
    log_p = - 0.5 * torch.square(samples) / var - 0.5 * np.log(var) - 0.9189385332  # 0.5 * np.log(2 * np.pi)
    return log_p


def one_hot(indices, depth, dim):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size).cuda()
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)

    return y_onehot

# TODO: merge this with the next class
class PixelNormal(object):
    def __init__(self, param, fixed_log_scales=None):
        size = param.size()
        C = size[1]
        if fixed_log_scales is None:
            self.num_c = C // 2
            self.means = param[:, :self.num_c, :, :]                                    # B, 1 or 3, H, W
            self.log_scales = torch.clamp(param[:, self.num_c:, :, :], min=-7.0)        # B, 1 or 3, H, W
            raise NotImplementedError
        else:
            self.num_c = C
            self.means = param                                                          # B, 1 or 3, H, W
            self.log_scales = view4D(fixed_log_scales, size)                      # B, 1 or 3, H, W

    def get_params(self):
        return self.means, self.log_scales, self.num_c

    def log_prob(self, samples):
        B, C, H, W = samples.size()
        assert C == self.num_c

        log_probs = -0.5 * torch.square(self.means - samples) * torch.exp(-2.0 * self.log_scales) - self.log_scales - 0.9189385332  # -0.5*log(2*pi)
        return log_probs

    def sample(self, t=1.):
        z, rho = sample_normal_jit(self.means, torch.exp(self.log_scales)*t)  # B, 3, H, W
        return z

    def log_prob_discrete(self, samples):
        """
        Calculates discrete pixel probabilities.
        """
        # samples should be in [-1, 1] already
        B, C, H, W = samples.size()
        assert C == self.num_c

        centered = samples - self.means
        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / 255.)
        cdf_plus = torch.distributions.Normal(0, 1).cdf(plus_in)
        min_in = inv_stdv * (centered - 1. / 255.)
        cdf_min = torch.distributions.Normal(0, 1).cdf(min_in)
        log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
        log_one_minus_cdf_min = torch.log(torch.clamp(1. - cdf_min, min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.999, log_one_minus_cdf_min,
            torch.log(torch.clamp(cdf_delta, min=1e-12))))

        assert log_probs.size() == samples.size()
        return log_probs

    def mean(self):
        return self.means


class Normal:
    def __init__(self, mu, log_sigma):
        self.mu = mu
        self.log_sigma = log_sigma
        self.sigma = torch.exp(log_sigma)

    def sample(self, t=1.):
        return sample_normal_jit(self.mu, self.sigma * t)

    def sample_given_rho(self, rho):
        return rho * self.sigma + self.mu

    def log_p(self, samples):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - self.log_sigma
        return log_p

    def kl(self, normal_dist):
        term1 = (self.mu - normal_dist.mu) / normal_dist.sigma
        term2 = self.sigma / normal_dist.sigma

        return 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(self.log_sigma) + normal_dist.log_sigma

    def mean(self):
        return self.mu


class Bernoulli:
    def __init__(self, logits):
        self.dist = Bern(logits=logits)

    def log_p(self, samples):
        # convert samples to {0, 1}
        samples = (samples + 1.) / 2
        return self.dist.log_prob(samples)

    def mean(self):
        # map the mean to [-1, 1]
        return 2 * self.dist.mean - 1.

class DiscLogistic:
    def __init__(self, param):
        B, C, H, W = param.size()
        self.num_c = C // 2
        self.means = param[:, :self.num_c, :, :]                              # B, 3, H, W
        self.log_scales = torch.clamp(param[:, self.num_c:, :, :], min=-7.0)  # B, 3, H, W

    def log_p(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= -1.0

        B, C, H, W = samples.size()
        assert C == self.num_c

        centered = samples - self.means                                         # B, 3, H, W
        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / 255.)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - np.log(127.5))

        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.999, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))   # B, 3, H, W

        return log_probs

    def sample(self):
        u = torch.Tensor(self.means.size()).uniform_(1e-5, 1. - 1e-5).cuda()                        # B, 3, H, W
        x = self.means + torch.exp(self.log_scales) * (torch.log(u) - torch.log(1. - u))            # B, 3, H, W
        x = torch.clamp(x, -1, 1.)
        return x

    def mean(self):
        return self.means


class DiscMixLogistic:
    def __init__(self, param, num_mix=10, num_bits=8):
        B, C, H, W = param.size()
        self.num_mix = num_mix
        self.logit_probs = param[:, :num_mix, :, :]                                   # B, M, H, W
        l = param[:, num_mix:, :, :].view(B, 3, 3 * num_mix, H, W)                    # B, 3, 3 * M, H, W
        self.means = l[:, :, :num_mix, :, :]                                          # B, 3, M, H, W
        self.log_scales = torch.clamp(l[:, :, num_mix:2 * num_mix, :, :], min=-7.0)   # B, 3, M, H, W
        self.coeffs = torch.tanh(l[:, :, 2 * num_mix:3 * num_mix, :, :])              # B, 3, M, H, W
        self.max_val = 2. ** num_bits - 1

    def log_p(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= -1.0

        B, C, H, W = samples.size()
        assert C == 3, 'only RGB images are considered.'

        samples = samples.unsqueeze(4)                                                  # B, 3, H , W
        samples = samples.expand(-1, -1, -1, -1, self.num_mix).permute(0, 1, 4, 2, 3)   # B, 3, M, H, W
        mean1 = self.means[:, 0, :, :, :]                                               # B, M, H, W
        mean2 = self.means[:, 1, :, :, :] + \
                self.coeffs[:, 0, :, :, :] * samples[:, 0, :, :, :]                     # B, M, H, W
        mean3 = self.means[:, 2, :, :, :] + \
                self.coeffs[:, 1, :, :, :] * samples[:, 0, :, :, :] + \
                self.coeffs[:, 2, :, :, :] * samples[:, 1, :, :, :]                     # B, M, H, W

        mean1 = mean1.unsqueeze(1)                          # B, 1, M, H, W
        mean2 = mean2.unsqueeze(1)                          # B, 1, M, H, W
        mean3 = mean3.unsqueeze(1)                          # B, 1, M, H, W
        means = torch.cat([mean1, mean2, mean3], dim=1)     # B, 3, M, H, W
        centered = samples - means                          # B, 3, M, H, W

        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / self.max_val)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - np.log(self.max_val / 2))

        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.999, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))   # B, 3, M, H, W

        log_probs = torch.sum(log_probs, 1) + F.log_softmax(self.logit_probs, dim=1)         # B, M, H, W
        return torch.logsumexp(log_probs, dim=1)                                             # B, H, W

    def sample(self, t=1.):
        gumbel = -torch.log(- torch.log(torch.Tensor(self.logit_probs.size()).uniform_(1e-5, 1. - 1e-5).cuda()))  # B, M, H, W
        sel = one_hot(torch.argmax(self.logit_probs / t + gumbel, 1), self.num_mix, dim=1)          # B, M, H, W
        sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)                                             # B, 3, H, W
        log_scales = torch.sum(self.log_scales * sel, dim=2)                                   # B, 3, H, W
        coeffs = torch.sum(self.coeffs * sel, dim=2)                                           # B, 3, H, W

        # cells from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = torch.Tensor(means.size()).uniform_(1e-5, 1. - 1e-5).cuda()                        # B, 3, H, W
        x = means + torch.exp(log_scales) * t * (torch.log(u) - torch.log(1. - u))             # B, 3, H, W

        x0 = torch.clamp(x[:, 0, :, :], -1, 1.)                                                # B, H, W
        x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)                       # B, H, W
        x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W

        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        x = torch.cat([x0, x1, x2], 1)
        return x

    def mean(self):
        sel = torch.softmax(self.logit_probs, dim=1)                                           # B, M, H, W
        sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)                                             # B, 3, H, W
        coeffs = torch.sum(self.coeffs * sel, dim=2)                                           # B, 3, H, W

        # we don't sample from logistic components, because of the linear dependencies, we use mean
        x = means                                                                              # B, 3, H, W
        x0 = torch.clamp(x[:, 0, :, :], -1, 1.)                                                # B, H, W
        x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)                       # B, H, W
        x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W

        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        x = torch.cat([x0, x1, x2], 1)
        return x


