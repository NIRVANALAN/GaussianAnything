# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from pdb import set_trace as st
from abc import ABC, abstractmethod
import numpy as np
import torch
import gc
from .continuous_distributions import log_p_standard_normal, log_p_var_normal
from .continuous_diffusion_utils import trace_df_dx_hutchinson, sample_gaussian_like, sample_rademacher_like, get_mixed_prediction
from torchdiffeq import odeint
from torch.cuda.amp import autocast
from timeit import default_timer as timer

from guided_diffusion import dist_util, logger


def make_diffusion(args):
    """ simple diffusion factory function to return diffusion instances. Only use this to create continuous diffusions """
    if args.sde_sde_type == 'geometric_sde':
        return DiffusionGeometric(args)
    elif args.sde_sde_type == 'vpsde':
        return DiffusionVPSDE(args)
    elif args.sde_sde_type == 'sub_vpsde':
        return DiffusionSubVPSDE(args)
    elif args.sde_sde_type == 'vesde':
        return DiffusionVESDE(args)
    else:
        raise ValueError("Unrecognized sde type: {}".format(args.sde_sde_type))


class DiffusionBase(ABC):
    """
    Abstract base class for all diffusion implementations.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sigma2_0 = args.sde_sigma2_0
        self.sde_type = args.sde_sde_type

    @abstractmethod
    def f(self, t):
        """ returns the drift coefficient at time t: f(t) """
        pass

    @abstractmethod
    def g2(self, t):
        """ returns the squared diffusion coefficient at time t: g^2(t) """
        pass

    @abstractmethod
    def var(self, t):
        """ returns variance at time t, \sigma_t^2"""
        pass

    @abstractmethod
    def e2int_f(self, t):
        """ returns e^{\int_0^t f(s) ds} which corresponds to the coefficient of mean at time t. """
        pass

    @abstractmethod
    def inv_var(self, var):
        """ inverse of the variance function at input variance var. """
        pass

    @abstractmethod
    def mixing_component(self, x_noisy, var_t, t, enabled):
        """ returns mixing component which is the optimal denoising model assuming that q(z_0) is N(0, 1) """
        pass

    def sample_q(self, x_init, noise, var_t, m_t):
        """ returns a sample from diffusion process at time t """
        return m_t * x_init + torch.sqrt(var_t) * noise

    def log_snr(self, m_t, var_t):
        return torch.log((torch.square(m_t) / var_t))

    def _predict_x0_from_eps(self, z, eps, logsnr):
        """eps = (z - alpha * x0) / sigma
        """
        return torch.sqrt(1 + torch.exp(-logsnr)) * (
            z - eps * torch.rsqrt(1 + torch.exp(logsnr)))

    def _predict_eps_from_x0(self, z, x0, logsnr):
        """x = (z - sigma * eps) / alpha
        """
        return torch.sqrt(1 + torch.exp(logsnr)) * (
            z - x0 * torch.rsqrt(1 + torch.exp(-logsnr)))
    
    def _predict_eps_from_z_and_v(self, v_t, var_t, z, m_t):
        # TODO, use logsnr here?
        return torch.sqrt(var_t) * z + m_t * v_t

    def _predict_x0_from_z_and_v(self, v_t, var_t, z, m_t):
        return torch.sqrt(var_t) * v_t + m_t * z

    def cross_entropy_const(self, ode_eps):
        """ returns cross entropy factor with variance according to ode integration cutoff ode_eps """
        # _, c, h, w = x_init.shape
        return 0.5 * (1.0 + torch.log(2.0 * np.pi * self.var(
            t=torch.tensor(ode_eps, device=dist_util.dev()))))

    def compute_ode_nll(self, dae, eps, ode_eps, ode_solver_tol,
                        enable_autocast, no_autograd, num_samples, report_std):
        """ calculates NLL based on ODE framework, assuming integration cutoff ode_eps """
        # ODE solver starts consuming the CPU memory without this on large models
        # https://github.com/scipy/scipy/issues/10070
        gc.collect()

        dae.eval()

        def ode_func(t, state):
            """ the ode function (including log probability integration for NLL calculation) """
            global nfe_counter
            nfe_counter = nfe_counter + 1

            x = state[0].detach()
            x.requires_grad_(True)
            noise = sample_gaussian_like(
                x)  # could also use rademacher noise (sample_rademacher_like)
            with torch.set_grad_enabled(True):
                with autocast(enabled=enable_autocast):
                    variance = self.var(t=t)
                    mixing_component = self.mixing_component(
                        x_noisy=x,
                        var_t=variance,
                        t=t,
                        enabled=dae.mixed_prediction)
                    pred_params = dae(x=x, t=t)
                    params = get_mixed_prediction(dae.mixed_prediction,
                                                  pred_params,
                                                  dae.mixing_logit,
                                                  mixing_component)
                    dx_dt = self.f(t=t) * x + 0.5 * self.g2(
                        t=t) * params / torch.sqrt(variance)

                with autocast(enabled=False):
                    dlogp_x_dt = -trace_df_dx_hutchinson(
                        dx_dt, x, noise, no_autograd).view(x.shape[0], 1)

            return (dx_dt, dlogp_x_dt)

        # NFE counter
        global nfe_counter

        nll_all, nfe_all = [], []
        for i in range(num_samples):
            # integrated log probability
            logp_diff_t0 = torch.zeros(eps.shape[0], 1, device=dist_util.dev())

            nfe_counter = 0

            # solve the ODE
            x_t, logp_diff_t = odeint(
                ode_func,
                (eps, logp_diff_t0),
                torch.tensor([ode_eps, 1.0], device=dist_util.dev()),
                atol=ode_solver_tol,
                rtol=ode_solver_tol,
                method="scipy_solver",
                options={"solver": 'RK45'},
            )
            # last output values
            x_t0, logp_diff_t0 = x_t[-1], logp_diff_t[-1]

            # prior
            if self.sde_type == 'vesde':
                logp_prior = torch.sum(log_p_var_normal(x_t0,
                                                        var=self.sigma2_max),
                                       dim=[1, 2, 3])
            else:
                logp_prior = torch.sum(log_p_standard_normal(x_t0),
                                       dim=[1, 2, 3])

            log_likelihood = logp_prior - logp_diff_t0.view(-1)

            nll_all.append(-log_likelihood)
            nfe_all.append(nfe_counter)

        nfe_mean = np.mean(nfe_all)
        nll_all = torch.stack(nll_all, dim=1)
        nll_mean = torch.mean(nll_all, dim=1)
        if num_samples > 1 and report_std:
            nll_stddev = torch.std(nll_all, dim=1)
            nll_stddev_batch = torch.mean(nll_stddev)
            nll_stderror_batch = nll_stddev_batch / np.sqrt(num_samples)
        else:
            nll_stddev_batch = None
            nll_stderror_batch = None
        return nll_mean, nfe_mean, nll_stddev_batch, nll_stderror_batch

    def sample_model_ode(self,
                         dae,
                         num_samples,
                         shape,
                         ode_eps,
                         ode_solver_tol,
                         enable_autocast,
                         temp,
                         noise=None):
        """ generates samples using the ODE framework, assuming integration cutoff ode_eps """
        # ODE solver starts consuming the CPU memory without this on large models
        # https://github.com/scipy/scipy/issues/10070
        gc.collect()

        dae.eval()

        def ode_func(t, x):
            """ the ode function (sampling only, no NLL stuff) """
            global nfe_counter
            nfe_counter = nfe_counter + 1
            with autocast(enabled=enable_autocast):
                variance = self.var(t=t)
                mixing_component = self.mixing_component(
                    x_noisy=x,
                    var_t=variance,
                    t=t,
                    enabled=dae.mixed_prediction)
                pred_params = dae(x=x, t=t)
                params = get_mixed_prediction(dae.mixed_prediction,
                                              pred_params, dae.mixing_logit,
                                              mixing_component)
                dx_dt = self.f(t=t) * x + 0.5 * self.g2(
                    t=t) * params / torch.sqrt(variance)

            return dx_dt

        # the initial noise
        if noise is None:
            noise = torch.randn(size=[num_samples] + shape,
                                device=dist_util.dev())

        if self.sde_type == 'vesde':
            noise_init = temp * noise * np.sqrt(self.sigma2_max)
        else:
            noise_init = temp * noise

        # NFE counter
        global nfe_counter
        nfe_counter = 0

        # solve the ODE
        start = timer()
        samples_out = odeint(
            ode_func,
            noise_init,
            torch.tensor([1.0, ode_eps], device=dist_util.dev()),
            atol=ode_solver_tol,
            rtol=ode_solver_tol,
            method="scipy_solver",
            options={"solver": 'RK45'},
        )
        end = timer()
        ode_solve_time = end - start

        return samples_out[-1], nfe_counter, ode_solve_time

    # def iw_quantities(self, size, time_eps, iw_sample_mode, iw_subvp_like_vp_sde):
    def iw_quantities(self, iw_sample_mode, size=None):

        args = self.args
        time_eps, iw_subvp_like_vp_sde = args.sde_time_eps, args.iw_subvp_like_vp_sde
        if size is None:
            size = args.batch_size

        if self.sde_type in ['geometric_sde', 'vpsde']:
            return self._iw_quantities_vpsdelike(size, time_eps,
                                                 iw_sample_mode)
        elif self.sde_type in ['sub_vpsde']:
            return self._iw_quantities_subvpsdelike(size, time_eps,
                                                    iw_sample_mode,
                                                    iw_subvp_like_vp_sde)
        elif self.sde_type in ['vesde']:
            return self._iw_quantities_vesde(size, time_eps, iw_sample_mode)
        else:
            raise NotImplementedError

    def _iw_quantities_vpsdelike(self, size, time_eps, iw_sample_mode):
        """
        For all SDEs where the underlying SDE is of the form dz = -0.5 * beta(t) * z * dt + sqrt{beta(t)} * dw, like
        for the VPSDE.
        """
        rho = torch.rand(size=[size], device=dist_util.dev())

        # In the following, obj_weight_t corresponds to the weight in front of the l2 loss for the given iw_sample_mode.
        # obj_weight_t_ll corresponds to the weight that converts the weighting scheme in iw_sample_mode to likelihood
        # weighting.

        if iw_sample_mode == 'll_uniform':
            # uniform t sampling - likelihood obj. for both q and p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'll_iw': # ! q-obj
            # importance sampling for likelihood obj. - likelihood obj. for both q and p
            ones = torch.ones_like(rho, device=dist_util.dev())
            sigma2_1, sigma2_eps = self.var(ones), self.var(time_eps * ones)
            log_sigma2_1, log_sigma2_eps = torch.log(sigma2_1), torch.log(
                sigma2_eps)
            var_t = torch.exp(rho * log_sigma2_1 +
                              (1 - rho) * log_sigma2_eps)  # sigma square
            t = self.inv_var(var_t)
            m_t, g2_t = self.e2int_f(t), self.g2(t)  # m_t is alpha_bar
            obj_weight_t = obj_weight_t_ll = 0.5 * (
                log_sigma2_1 - log_sigma2_eps) / (1.0 - var_t)

        elif iw_sample_mode == 'drop_all_uniform':
            # uniform t sampling - likelihood obj. for q, all-prefactors-dropped obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = torch.ones(1, device=dist_util.dev())
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'drop_all_iw':
            # importance sampling for all-pref.-dropped obj. - likelihood obj. for q, all-pref.-dropped obj. for p
            assert self.sde_type == 'vpsde', 'Importance sampling for fully unweighted objective is currently only ' \
                                             'implemented for the regular VPSDE.'
            t = torch.sqrt(1.0 / self.delta_beta_half) * torch.erfinv(
                rho * self.const_norm_2 + self.const_erf) - self.beta_frac
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = self.const_norm / (1.0 - var_t)
            obj_weight_t_ll = obj_weight_t * g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'drop_sigma2t_iw': # ! default mode for p
            # importance sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
            ones = torch.ones_like(rho, device=dist_util.dev())
            sigma2_1, sigma2_eps = self.var(ones), self.var(time_eps * ones)
            var_t = rho * sigma2_1 + (1 - rho) * sigma2_eps # ! sigma square
            t = self.inv_var(var_t)
            m_t, g2_t = self.e2int_f(t), self.g2(t) # ! m_t: alpha_bar sqrt
            obj_weight_t = 0.5 * (sigma2_1 - sigma2_eps) / (1.0 - var_t)
            obj_weight_t_ll = obj_weight_t / var_t

        elif iw_sample_mode == 'drop_sigma2t_uniform':
            # uniform sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = g2_t / 2.0
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'rescale_iw':
            # importance sampling for 1/(1-sigma2_t) resc. obj. - likelihood obj. for q, 1/(1-sigma2_t) resc. obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = 0.5 / (1.0 - var_t)
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        else:
            raise ValueError(
                "Unrecognized importance sampling type: {}".format(
                    iw_sample_mode))

        return t, var_t.view(-1, 1, 1, 1), m_t.view(-1, 1, 1, 1), obj_weight_t.view(-1, 1, 1, 1), \
            obj_weight_t_ll.view(-1, 1, 1, 1), g2_t.view(-1, 1, 1, 1)

    def _iw_quantities_subvpsdelike(self, size, time_eps, iw_sample_mode,
                                    iw_subvp_like_vp_sde):
        """
        For all SDEs where the underlying SDE is of the form
        dz = -0.5 * beta(t) * z * dt + sqrt{beta(t) * (1 - exp[-2 * betaintegral])} * dw, like for the Sub-VPSDE.
        When iw_subvp_like_vp_sde is True, then we define the importance sampling distributions based on an analogous
        VPSDE, while stile using the Sub-VPSDE. The motivation is that deriving the correct importance sampling
        distributions for the Sub-VPSDE itself is hard, but the importance sampling distributions from analogous VPSDEs
        probably already significantly reduce the variance also for the Sub-VPSDE.
        """
        rho = torch.rand(size=[size], device=dist_util.dev())

        # In the following, obj_weight_t corresponds to the weight in front of the l2 loss for the given iw_sample_mode.
        # obj_weight_t_ll corresponds to the weight that converts the weighting scheme in iw_sample_mode to likelihood
        # weighting.
        if iw_sample_mode == 'll_uniform':
            # uniform t sampling - likelihood obj. for both q and p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'll_iw':
            if iw_subvp_like_vp_sde:
                # importance sampling for vpsde likelihood obj. - sub-vpsde likelihood obj. for both q and p
                ones = torch.ones_like(rho, device=dist_util.dev())
                sigma2_1, sigma2_eps = self.var_vpsde(ones), self.var_vpsde(
                    time_eps * ones)
                log_sigma2_1, log_sigma2_eps = torch.log(sigma2_1), torch.log(
                    sigma2_eps)
                var_t_vpsde = torch.exp(rho * log_sigma2_1 +
                                        (1 - rho) * log_sigma2_eps)
                t = self.inv_var_vpsde(var_t_vpsde)
                var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
                obj_weight_t = obj_weight_t_ll = g2_t / (2.0 * var_t) * \
                    (log_sigma2_1 - log_sigma2_eps) * var_t_vpsde / (1 - var_t_vpsde) / self.beta(t)
            else:
                raise NotImplementedError

        elif iw_sample_mode == 'drop_all_uniform':
            # uniform t sampling - likelihood obj. for q, all-prefactors-dropped obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = torch.ones(1, device=dist_util.dev())
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'drop_all_iw':
            if iw_subvp_like_vp_sde:
                # importance sampling for all-pref.-dropped obj. - likelihood obj. for q, all-pref.-dropped obj. for p
                assert self.sde_type == 'sub_vpsde', 'Importance sampling for fully unweighted objective is ' \
                                                     'currently only implemented for the Sub-VPSDE.'
                t = torch.sqrt(1.0 / self.delta_beta_half) * torch.erfinv(
                    rho * self.const_norm_2 + self.const_erf) - self.beta_frac
                var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
                obj_weight_t = self.const_norm / (1.0 - self.var_vpsde(t))
                obj_weight_t_ll = obj_weight_t * g2_t / (2.0 * var_t)
            else:
                raise NotImplementedError

        elif iw_sample_mode == 'drop_sigma2t_iw':
            if iw_subvp_like_vp_sde:
                # importance sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
                ones = torch.ones_like(rho, device=dist_util.dev())
                sigma2_1, sigma2_eps = self.var_vpsde(ones), self.var_vpsde(
                    time_eps * ones)
                var_t_vpsde = rho * sigma2_1 + (1 - rho) * sigma2_eps
                t = self.inv_var_vpsde(var_t_vpsde)
                var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
                obj_weight_t = 0.5 * g2_t / self.beta(t) * (
                    sigma2_1 - sigma2_eps) / (1.0 - var_t_vpsde)
                obj_weight_t_ll = obj_weight_t / var_t
            else:
                raise NotImplementedError

        elif iw_sample_mode == 'drop_sigma2t_uniform':
            # uniform sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = g2_t / 2.0
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'rescale_iw':
            # importance sampling for 1/(1-sigma2_t) resc. obj. - likelihood obj. for q, 1/(1-sigma2_t) resc. obj. for p
            # Note that we use the sub-vpsde variance to scale the p objective! It's not clear what's optimal here!
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = 0.5 / (1.0 - var_t)
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        else:
            raise ValueError(
                "Unrecognized importance sampling type: {}".format(
                    iw_sample_mode))

        return t, var_t.view(-1, 1, 1, 1), m_t.view(-1, 1, 1, 1), obj_weight_t.view(-1, 1, 1, 1), \
            obj_weight_t_ll.view(-1, 1, 1, 1), g2_t.view(-1, 1, 1, 1)

    def _iw_quantities_vesde(self, size, time_eps, iw_sample_mode):
        """
        For the VESDE.
        """
        rho = torch.rand(size=[size], device=dist_util.dev())

        # In the following, obj_weight_t corresponds to the weight in front of the l2 loss for the given iw_sample_mode.
        # obj_weight_t_ll corresponds to the weight that converts the weighting scheme in iw_sample_mode to likelihood
        # weighting.
        if iw_sample_mode == 'll_uniform':
            # uniform t sampling - likelihood obj. for both q and p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'll_iw':
            # importance sampling for likelihood obj. - likelihood obj. for both q and p
            ones = torch.ones_like(rho, device=dist_util.dev())
            nsigma2_1, nsigma2_eps, sigma2_eps = self.var_N(ones), self.var_N(
                time_eps * ones), self.var(time_eps * ones)
            log_frac_sigma2_1, log_frac_sigma2_eps = torch.log(
                self.sigma2_max / nsigma2_1), torch.log(nsigma2_eps /
                                                        sigma2_eps)
            var_N_t = (1.0 - self.sigma2_min) / (
                1.0 - torch.exp(rho *
                                (log_frac_sigma2_1 + log_frac_sigma2_eps) -
                                log_frac_sigma2_eps))
            t = self.inv_var_N(var_N_t)
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = obj_weight_t_ll = 0.5 * (
                log_frac_sigma2_1 +
                log_frac_sigma2_eps) * self.var_N(t) / (1.0 - self.sigma2_min)

        elif iw_sample_mode == 'drop_all_uniform':
            # uniform t sampling - likelihood obj. for q, all-prefactors-dropped obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = torch.ones(1, device=dist_util.dev())
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'drop_all_iw':
            # importance sampling for all-pref.-dropped obj. - likelihood obj. for q, all-pref.-dropped obj. for p
            ones = torch.ones_like(rho, device=dist_util.dev())
            nsigma2_1, nsigma2_eps, sigma2_eps = self.var_N(ones), self.var_N(
                time_eps * ones), self.var(time_eps * ones)
            log_frac_sigma2_1, log_frac_sigma2_eps = torch.log(
                self.sigma2_max / nsigma2_1), torch.log(nsigma2_eps /
                                                        sigma2_eps)
            var_N_t = (1.0 - self.sigma2_min) / (
                1.0 - torch.exp(rho *
                                (log_frac_sigma2_1 + log_frac_sigma2_eps) -
                                log_frac_sigma2_eps))
            t = self.inv_var_N(var_N_t)
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t_ll = 0.5 * (log_frac_sigma2_1 +
                                     log_frac_sigma2_eps) * self.var_N(t) / (
                                         1.0 - self.sigma2_min)
            obj_weight_t = 2.0 * obj_weight_t_ll / np.log(
                self.sigma2_max / self.sigma2_min)

        elif iw_sample_mode == 'drop_sigma2t_iw':
            # importance sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
            ones = torch.ones_like(rho, device=dist_util.dev())
            nsigma2_1, nsigma2_eps = self.var_N(ones), self.var_N(time_eps *
                                                                  ones)
            var_N_t = torch.exp(rho * torch.log(nsigma2_1) +
                                (1 - rho) * torch.log(nsigma2_eps))
            t = self.inv_var_N(var_N_t)
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = 0.5 * torch.log(
                nsigma2_1 / nsigma2_eps) * self.var_N(t)
            obj_weight_t_ll = obj_weight_t / var_t

        elif iw_sample_mode == 'drop_sigma2t_uniform':
            # uniform sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = g2_t / 2.0
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'rescale_iw':
            # uniform sampling for 1/(1-sigma2_t) resc. obj. - likelihood obj. for q, 1/(1-sigma2_t) resc. obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = 0.5 / (1.0 - var_t)
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        else:
            raise ValueError(
                "Unrecognized importance sampling type: {}".format(
                    iw_sample_mode))

        return t, var_t.view(-1, 1, 1, 1), m_t.view(-1, 1, 1, 1), obj_weight_t.view(-1, 1, 1, 1), \
            obj_weight_t_ll.view(-1, 1, 1, 1), g2_t.view(-1, 1, 1, 1)


class DiffusionGeometric(DiffusionBase):
    """
    Diffusion implementation with dz = -0.5 * beta(t) * z * dt + sqrt(beta(t)) * dW SDE and geometric progression of
    variance. This is our new diffusion.
    """

    def __init__(self, args):
        super().__init__(args)
        self.sigma2_min = args.sde_sigma2_min
        self.sigma2_max = args.sde_sigma2_max

    def f(self, t):
        return -0.5 * self.g2(t)

    def g2(self, t):
        sigma2_geom = self.sigma2_min * (
            (self.sigma2_max / self.sigma2_min)**t)
        log_term = np.log(self.sigma2_max / self.sigma2_min)
        return sigma2_geom * log_term / (1.0 - self.sigma2_0 +
                                         self.sigma2_min - sigma2_geom)

    def var(self, t):
        return self.sigma2_min * ((self.sigma2_max / self.sigma2_min)**
                                  t) - self.sigma2_min + self.sigma2_0

    def e2int_f(self, t):
        return torch.sqrt(1.0 + self.sigma2_min *
                          (1.0 - (self.sigma2_max / self.sigma2_min)**t) /
                          (1.0 - self.sigma2_0))

    def inv_var(self, var):
        return torch.log(
            (var + self.sigma2_min - self.sigma2_0) /
            self.sigma2_min) / np.log(self.sigma2_max / self.sigma2_min)

    def mixing_component(self, x_noisy, var_t, t, enabled):
        if enabled:
            return torch.sqrt(var_t) * x_noisy
        else:
            return None


class DiffusionVPSDE(DiffusionBase):
    """
    Diffusion implementation of the VPSDE. This uses the same SDE like DiffusionGeometric but with linear beta(t).
    Note that we need to scale beta_start and beta_end by 1000 relative to JH's DDPM values, since our t is in [0,1].
    """

    def __init__(self, args):
        super().__init__(args)
        # self.beta_start = args.sde_beta_start # 0.1
        # self.beta_end = args.sde_beta_end # 20

        # ! hard coded, in the scale of 1000.
        # beta_start = scale * 0.0001
        # beta_end = scale * 0.02

        self.beta_start = 0.1
        self.beta_end = 20

        # auxiliary constants
        self.time_eps = args.sde_time_eps  # 0.01 by default in LSGM. Any influence?
        self.delta_beta_half = torch.tensor(0.5 *
                                            (self.beta_end - self.beta_start),
                                            device=dist_util.dev())
        self.beta_frac = torch.tensor(self.beta_start /
                                      (self.beta_end - self.beta_start),
                                      device=dist_util.dev())
        self.const_aq = (1.0 - self.sigma2_0) * torch.exp(
            0.5 * self.beta_frac) * torch.sqrt(
                0.25 * np.pi / self.delta_beta_half)
        self.const_erf = torch.erf(
            torch.sqrt(self.delta_beta_half) *
            (self.time_eps + self.beta_frac))
        self.const_norm = self.const_aq * (torch.erf(
            torch.sqrt(self.delta_beta_half) *
            (1.0 + self.beta_frac)) - self.const_erf)
        self.const_norm_2 = torch.erf(
            torch.sqrt(self.delta_beta_half) *
            (1.0 + self.beta_frac)) - self.const_erf

    def f(self, t):
        return -0.5 * self.g2(t)

    def g2(self, t):
        return self.beta_start + (self.beta_end - self.beta_start) * t

    def var(self, t):
        return 1.0 - (1.0 - self.sigma2_0
                      ) * torch.exp(-self.beta_start * t - 0.5 *
                                    (self.beta_end - self.beta_start) * t * t)

    def e2int_f(self, t):
        return torch.exp(-0.5 * self.beta_start * t - 0.25 *
                         (self.beta_end - self.beta_start) * t * t)

    def inv_var(self, var):
        c = torch.log((1 - var) / (1 - self.sigma2_0))
        a = self.beta_end - self.beta_start
        t = (-self.beta_start +
             torch.sqrt(np.square(self.beta_start) - 2 * a * c)) / a
        return t

    def mixing_component(self, x_noisy, var_t, t, enabled):
        if enabled:
            return torch.sqrt(var_t) * x_noisy
        else:
            return None

    def mixing_component_x0(self, x_noisy, var_t, t, enabled):
        if enabled:
            # return torch.sqrt(var_t) * x_noisy
            return torch.sqrt(1-var_t) * x_noisy # zt * alpha_t
        else:
            return None


class DiffusionSubVPSDE(DiffusionBase):
    """
    Diffusion implementation of the sub-VPSDE. Note that this uses a different SDE compared to the above two diffusions.
    """

    def __init__(self, args):
        super().__init__(args)
        self.beta_start = args.sde_beta_start
        self.beta_end = args.sde_beta_end

        # auxiliary constants (assumes regular VPSDE)
        self.time_eps = args.sde_time_eps
        self.delta_beta_half = torch.tensor(0.5 *
                                            (self.beta_end - self.beta_start),
                                            device=dist_util.dev())
        self.beta_frac = torch.tensor(self.beta_start /
                                      (self.beta_end - self.beta_start),
                                      device=dist_util.dev())
        self.const_aq = (1.0 - self.sigma2_0) * torch.exp(
            0.5 * self.beta_frac) * torch.sqrt(
                0.25 * np.pi / self.delta_beta_half)
        self.const_erf = torch.erf(
            torch.sqrt(self.delta_beta_half) *
            (self.time_eps + self.beta_frac))
        self.const_norm = self.const_aq * (torch.erf(
            torch.sqrt(self.delta_beta_half) *
            (1.0 + self.beta_frac)) - self.const_erf)
        self.const_norm_2 = torch.erf(
            torch.sqrt(self.delta_beta_half) *
            (1.0 + self.beta_frac)) - self.const_erf

    def f(self, t):
        return -0.5 * self.beta(t)

    def g2(self, t):
        return self.beta(t) * (
            1.0 - torch.exp(-2.0 * self.beta_start * t -
                            (self.beta_end - self.beta_start) * t * t))

    def var(self, t):
        int_term = torch.exp(-self.beta_start * t - 0.5 *
                             (self.beta_end - self.beta_start) * t * t)
        return torch.square(1.0 - int_term) + self.sigma2_0 * int_term

    def e2int_f(self, t):
        return torch.exp(-0.5 * self.beta_start * t - 0.25 *
                         (self.beta_end - self.beta_start) * t * t)

    def beta(self, t):
        """ auxiliary beta function """
        return self.beta_start + (self.beta_end - self.beta_start) * t

    def inv_var(self, var):
        raise NotImplementedError

    def mixing_component(self, x_noisy, var_t, t, enabled):
        if enabled:
            int_term = torch.exp(-self.beta_start * t - 0.5 *
                                 (self.beta_end - self.beta_start) * t *
                                 t).view(-1, 1, 1, 1)
            return torch.sqrt(var_t) * x_noisy / (
                torch.square(1.0 - int_term) + int_term)
        else:
            return None

    def var_vpsde(self, t):
        return 1.0 - (1.0 - self.sigma2_0
                      ) * torch.exp(-self.beta_start * t - 0.5 *
                                    (self.beta_end - self.beta_start) * t * t)

    def inv_var_vpsde(self, var):
        c = torch.log((1 - var) / (1 - self.sigma2_0))
        a = self.beta_end - self.beta_start
        t = (-self.beta_start +
             torch.sqrt(np.square(self.beta_start) - 2 * a * c)) / a
        return t


class DiffusionVESDE(DiffusionBase):
    """
    Diffusion implementation of the VESDE with dz = sqrt(beta(t)) * dW
    """

    def __init__(self, args):
        super().__init__(args)
        self.sigma2_min = args.sde_sigma2_min
        self.sigma2_max = args.sde_sigma2_max
        assert self.sigma2_min == self.sigma2_0, "VESDE was proposed implicitly assuming sigma2_min = sigma2_0!"

    def f(self, t):
        return torch.zeros_like(t, device=dist_util.dev())

    def g2(self, t):
        return self.sigma2_min * np.log(self.sigma2_max / self.sigma2_min) * (
            (self.sigma2_max / self.sigma2_min)**t)

    def var(self, t):
        return self.sigma2_min * ((self.sigma2_max / self.sigma2_min)**
                                  t) - self.sigma2_min + self.sigma2_0

    def e2int_f(self, t):
        return torch.ones_like(t, device=dist_util.dev())

    def inv_var(self, var):
        return torch.log(
            (var + self.sigma2_min - self.sigma2_0) /
            self.sigma2_min) / np.log(self.sigma2_max / self.sigma2_min)

    def mixing_component(self, x_noisy, var_t, t, enabled):
        if enabled:
            return torch.sqrt(var_t) * x_noisy / (self.sigma2_min * (
                (self.sigma2_max / self.sigma2_min)**t.view(-1, 1, 1, 1)) -
                                                  self.sigma2_min + 1.0)
        else:
            return None

    def var_N(self, t):
        return 1.0 - self.sigma2_min + self.sigma2_min * (
            (self.sigma2_max / self.sigma2_min)**t)

    def inv_var_N(self, var):
        return torch.log(
            (var + self.sigma2_min - 1.0) / self.sigma2_min) / np.log(
                self.sigma2_max / self.sigma2_min)
