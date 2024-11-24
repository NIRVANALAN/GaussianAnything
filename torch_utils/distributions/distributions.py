# https://raw.githubusercontent.com/CompVis/latent-diffusion/e66308c7f2e64cb581c6d27ab6fbeb846828253b/ldm/modules/distributions/distributions.py

import torch
import numpy as np
from pdb import set_trace as st


class AbstractDistribution:

    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):

    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


@torch.jit.script
def soft_clamp20(x: torch.Tensor):
    return x.div(20.).tanh().mul(
        20.
    )  # 5. * torch.tanh(x / 5.) <--> soft differentiable clamp between [-5, 5]


# @torch.jit.script
# def soft_clamp(x: torch.Tensor, a: torch.Tensor):
#     return x.div(a).tanh_().mul(a)


class DiagonalGaussianDistribution(object):

    def __init__(self, parameters, deterministic=False, soft_clamp=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)

        if soft_clamp:
            # self.mean, self.logvar = soft_clamp5(self.mean), soft_clamp5(self.logvar) # as in LSGM, bound the range. needs re-training?
            self.logvar = soft_clamp20(
                self.logvar)  # as in LSGM, bound the range. [-20, 20]
        else:
            self.logvar = torch.clamp(self.logvar, -30.0, 20.0)

        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(
            self.mean.shape).to(device=self.parameters.device)
        return x

    # https://github.dev/NVlabs/LSGM/util/distributions.py
    def log_p(self, samples):
        # for calculating the negative encoder entropy term
        normalized_samples = (samples - self.mean) / self.var
        log_p = -0.5 * normalized_samples * normalized_samples - 0.5 * np.log(
            2 * np.pi) - self.logvar  #

        return log_p  # ! TODO

    def normal_entropy(self):
        # for calculating normal entropy. Motivation: supervise logvar directly.
        # normalized_samples = (samples - self.mean) / self.var
        # log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - self.logvar #
        # entropy = torch.sum(self.logvar + 0.5 * (np.log(2 * np.pi) + 1),
        #                     dim=[1, 2, 3]).mean(0)
        # entropy = torch.mean(self.logvar + 0.5 * (np.log(2 * np.pi) + 1)) # follow eps loss tradition here, average overall dims.
        entropy = self.logvar + 0.5 * (np.log(2 * np.pi) + 1) # follow eps loss tradition here, average overall dims.
                            

        return entropy  # ! TODO

    def kl(self, other=None, pt_ft_separate=False, ft_separate=False):

        def kl_fn(mean, var, logvar):
            return 0.5 * torch.sum(
                torch.pow(mean, 2) + var - 1.0 - logvar,
                dim=list(range(1,mean.ndim))) # support B L C-like VAE latent

        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                if pt_ft_separate: # as in LION
                    pt_kl = kl_fn(self.mean[:, :3], self.var[:, :3], self.logvar[:, :3]) # (B C L) input
                    ft_kl = kl_fn(self.mean[:, 3:], self.var[:, 3:], self.logvar[:, 3:]) # (B C L) input
                    return pt_kl, ft_kl
                elif ft_separate:
                    ft_kl = kl_fn(self.mean[:, :], self.var[:, :], self.logvar[:, :]) # (B C L) input
                    return ft_kl
                else:
                    return 0.5 * torch.sum(
                        torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                        dim=list(range(1,self.mean.ndim))) # support B L C-like VAE latent
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var +
                    self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar +
                               torch.pow(sample - self.mean, 2) / self.var,
                               dim=dims)

    def mode(self):
        return self.mean


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) +
                  ((mean1 - mean2)**2) * torch.exp(-logvar2))
