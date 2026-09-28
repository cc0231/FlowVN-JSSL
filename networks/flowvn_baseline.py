"""
Adopted from
https://github.com/rixez/pytorch_mri_variationalnetwork

BASELINE FlowVN -- faithful reproduction of the ORIGINAL paper architecture
(Jacobs et al. 2025, "FlowMRI-Net", the FlowVN variant), for a clean
no-optimization baseline comparison.

"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils.misc_utils import *


def zero_mean_norm_ball(x, zero_mean=True, normalize=True, norm_bound=1.0, mask=None, axis=(0, ...)):
    """ https://github.com/VLOGroup/mri-variationalnetwork/blob/master/vn/proxmaps.py """
    if mask is None:
        shape = []
        for i in range(len(x.shape)):
            if i in axis:
                shape.append(x.shape[i])
            else:
                shape.append(1)
        mask = torch.ones(shape, dtype=torch.float32)

    x_masked = x * mask

    if zero_mean:
        x_mean = torch.mean(x_masked, dim=axis, keepdim=True)
        x_zm = x_masked - x_mean
    else:
        x_zm = x_masked

    if normalize:
        magnitude = torch.sqrt(torch.sum(torch.square(x_zm), dim=axis, keepdim=True))
        x_proj = x_zm / magnitude * norm_bound
        return x_proj

    return x_zm


class RBFActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w, mu, sigma):
        """
        input: input tensor ((N*D)xCxTxHxW)
        w: weight of the RBF kernels (1 x C x 1 x 1 x 1 x # of RBF kernels)
        mu: center of the RBF (# of RBF kernels)
        sigma: std of the RBF (1)
        """
        output = input.new_zeros(input.shape)  # initialize output tensor
        rbf_grad_input = input.new_zeros(input.shape)  # initialize
        for i in range(w.shape[-1]):
            tmp = w[:, :, :, :, :, i] * torch.exp(-torch.square(input - mu[i]) / (2 * sigma ** 2))
            output += tmp
            rbf_grad_input += tmp * (-(input - mu[i])) / (sigma ** 2)  # derivative wrt (input - mu[i])
        del tmp

        ctx.save_for_backward(input, w, mu, sigma, rbf_grad_input)
        return output  # y

    @staticmethod
    def backward(ctx, grad_output):
        input, w, mu, sigma, rbf_grad_input = ctx.saved_tensors

        grad_input = grad_output * rbf_grad_input

        grad_w = w.new_zeros(w.shape)
        for i in range(w.shape[-1]):
            tmp = (grad_output * torch.exp(-torch.square(input - mu[i]) / (2 * sigma ** 2))).sum((0, 2, 3, 4))
            grad_w[:, :, :, :, :, i] = tmp.view(w.shape[0:-1])

        return grad_input, grad_w, None, None


class RBFActivation(nn.Module):
    def __init__(self, feat, **kwargs):
        super().__init__()
        self.options = kwargs

        x_0 = np.linspace(kwargs['vmin'], kwargs['vmax'], kwargs['num_act_weights'], dtype=np.float32)  # weight of RBF kernels
        mu = np.linspace(kwargs['vmin'], kwargs['vmax'], kwargs['num_act_weights'], dtype=np.float32)  # center of RBF
        sigma = 2 * kwargs['vmax'] / (kwargs['num_act_weights'] - 1)

        w_0 = kwargs['weight'] * x_0
        w_0 = np.reshape(w_0, (1, 1, 1, 1, 1, kwargs['num_act_weights']))
        w_0 = np.repeat(w_0, feat, 1)
        self.w = torch.nn.Parameter(torch.from_numpy(w_0))

        # buffers (were plain tensors moved to cuda:0 in the original forward)
        self.register_buffer('mu', torch.from_numpy(mu))
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))

        self.rbf_act = RBFActivationFunction.apply

    def forward(self, x):
        return self.rbf_act(x, self.w, self.mu, self.sigma)


class LinearActivationFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size):
        x_clamped = x.clamp(min=-(grid.item() * (size // 2)), max=(grid.item() * (size // 2 - 1)))

        floored_x = torch.floor(x_clamped / grid)  # left coefficient
        fracs = x_clamped / grid - floored_x  # distance to left coefficient

        indexes = (zero_knot_indexes.view(1, -1, 1, 1, 1) + floored_x).long()
        ctx.save_for_backward(fracs, coefficients_vect, indexes, grid)

        out = coefficients_vect[indexes + 1] * fracs + coefficients_vect[indexes] * (1 - fracs)
        out[x < -(grid.item() * (size // 2))] = x[x < -(grid.item() * (size // 2))]
        out[x > (grid.item() * (size // 2 - 1))] = x[x > (grid.item() * (size // 2 - 1))]
        return out

    @staticmethod
    def backward(ctx, grad_out):
        fracs, coefficients_vect, indexes, grid = ctx.saved_tensors
        grad_x = ((coefficients_vect[indexes + 1] - coefficients_vect[indexes]) / grid * grad_out)

        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1) + 1, (fracs * grad_out).view(-1))
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1), ((1 - fracs) * grad_out).view(-1))
        return grad_x, grad_coefficients_vect, None, None, None


class LinearActivation(torch.nn.Module):
    def __init__(self, num_activations, **kwargs):
        super().__init__()
        self.options = kwargs

        # buffers (were tensors hard-pinned to cuda:0 in the original)
        self.register_buffer('grid', torch.tensor([self.options['grid']], dtype=torch.float32))
        grid_tensor = (
            torch.arange(-(self.options['num_act_weights'] // 2), (self.options['num_act_weights'] // 2) + 1,
                         dtype=torch.float32).mul(self.grid)
        ).expand((num_activations, self.options['num_act_weights']))
        grid_tensor = grid_tensor * 0.01  # lower initial values for training stability (span unchanged)
        self.coefficients_vect = torch.nn.Parameter(grid_tensor.contiguous().view(-1))

        self.register_buffer(
            'zero_knot_indexes',
            torch.arange(0, num_activations, dtype=torch.long) * self.options['num_act_weights']
            + (self.options['num_act_weights'] // 2),
        )

    def forward(self, input):
        return LinearActivationFunc.apply(input, self.coefficients_vect, self.grid,
                                          self.zero_knot_indexes, self.options['num_act_weights'])


class VnMriReconCell(nn.Module):
    def __init__(self, block_id, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options

        conv_kernel_xyz = torch.normal(mean=0, std=1, size=(options['features_out'], options['features_in'], options['kernel_size'], options['kernel_size'], options['kernel_size']))
        conv_kernel_xyz /= np.sqrt(options['kernel_size'] ** 2 * options['features_in'])
        conv_kernel_xyz -= torch.mean(conv_kernel_xyz, axis=(1, 2, 3, 4), keepdims=True)
        conv_kernel_xyz = zero_mean_norm_ball(conv_kernel_xyz, axis=(1, 2, 3, 4))
        self.conv_kernel_xyz = torch.nn.Parameter(conv_kernel_xyz)

        conv_kernel_xyt = torch.normal(mean=0, std=1, size=(options['features_out'], options['features_in'], options['kernel_size'], options['kernel_size'], options['kernel_size']))
        conv_kernel_xyt /= np.sqrt(options['kernel_size'] ** 2 * options['features_in'])
        conv_kernel_xyt -= torch.mean(conv_kernel_xyt, axis=(1, 2, 3, 4), keepdims=True)
        conv_kernel_xyt = zero_mean_norm_ball(conv_kernel_xyt, axis=(1, 2, 3, 4))
        self.conv_kernel_xyt = torch.nn.Parameter(conv_kernel_xyt)

        conv_kernel_yzt = torch.normal(mean=0, std=1, size=(options['features_out'], options['features_in'], options['kernel_size'], options['kernel_size'], options['kernel_size']))
        conv_kernel_yzt /= np.sqrt(options['kernel_size'] ** 2 * options['features_in'])
        conv_kernel_yzt -= torch.mean(conv_kernel_yzt, axis=(1, 2, 3, 4), keepdims=True)
        conv_kernel_yzt = zero_mean_norm_ball(conv_kernel_yzt, axis=(1, 2, 3, 4))
        self.conv_kernel_yzt = torch.nn.Parameter(conv_kernel_yzt)

        conv_kernel_xzt = torch.normal(mean=0, std=1, size=(options['features_out'], options['features_in'], options['kernel_size'], options['kernel_size'], options['kernel_size']))
        conv_kernel_xzt /= np.sqrt(options['kernel_size'] ** 2 * options['features_in'])
        conv_kernel_xzt -= torch.mean(conv_kernel_xzt, axis=(1, 2, 3, 4), keepdims=True)
        conv_kernel_xzt = zero_mean_norm_ball(conv_kernel_xzt, axis=(1, 2, 3, 4))
        self.conv_kernel_xzt = torch.nn.Parameter(conv_kernel_xzt)

        if options['act'] == "rbf":
            self.activation1 = RBFActivation(feat=options['features_out'], **options)
            self.activation2 = RBFActivation(feat=options['features_out'], **options)
            self.activation3 = RBFActivation(feat=options['features_out'], **options)
            self.activation4 = RBFActivation(feat=options['features_out'], **options)
            self.activation5 = RBFActivation(feat=options['features_in'], **options)
        elif options['act'] == "linear":
            self.activation1 = LinearActivation(num_activations=options['features_out'], **options)
            self.activation2 = LinearActivation(num_activations=options['features_out'], **options)
            self.activation3 = LinearActivation(num_activations=options['features_out'], **options)
            self.activation4 = LinearActivation(num_activations=options['features_out'], **options)
            self.activation5 = LinearActivation(num_activations=options['features_in'], **options)
        else:
            raise ValueError("baseline FlowVN: act should be either 'rbf' or 'linear'")

        self.lamb_ru = torch.nn.Parameter(torch.tensor(1., dtype=torch.float32))
        self.lamb_du = torch.nn.Parameter(torch.tensor(1., dtype=torch.float32))
        self.alpha = torch.nn.Parameter(torch.tensor(1., dtype=torch.float32))
        self.block_id = block_id
        self.pad = int((self.options['kernel_size'] - 1) / 2)
        self.pad_d = int((self.options['D_size'] - 1) / 2)
        self.sgd_momentum = self.options['sgd_momentum']

    def forward(self, u_t_1, f, c, S_prev=None):
        N, V, T, D, H, W, _ = u_t_1.shape

        # convolutions (real and imag parts in batch dim -> same weights applied independently)
        u_k_xyz = F.conv3d(u_t_1.permute(0, 2, 6, 1, 3, 4, 5).contiguous().view(N * T * 2, V, D, H, W), self.conv_kernel_xyz, padding=(self.pad_d, self.pad, self.pad))
        u_k_xyt = F.conv3d(u_t_1.permute(0, 5, 6, 1, 3, 4, 2).contiguous().view(N * W * 2, V, D, H, T), self.conv_kernel_xyt, padding=(self.pad_d, self.pad, self.pad))
        u_k_yzt = F.conv3d(u_t_1.permute(0, 3, 6, 1, 4, 5, 2).contiguous().view(N * D * 2, V, H, W, T), self.conv_kernel_yzt, padding=(self.pad, self.pad, self.pad))
        u_k_xzt = F.conv3d(u_t_1.permute(0, 4, 6, 1, 3, 5, 2).contiguous().view(N * H * 2, V, D, W, T), self.conv_kernel_xzt, padding=(self.pad_d, self.pad, self.pad))

        # activations
        f_u_k_xyz = self.activation1(u_k_xyz)
        f_u_k_xyt = self.activation2(u_k_xyt)
        f_u_k_yzt = self.activation3(u_k_yzt)
        f_u_k_xzt = self.activation4(u_k_xzt)

        # transpose convolutions
        u_k_T_xyz = F.conv_transpose3d(f_u_k_xyz, self.conv_kernel_xyz, padding=(self.pad_d, self.pad, self.pad))
        u_k_T_xyt = F.conv_transpose3d(f_u_k_xyt, self.conv_kernel_xyt, padding=(self.pad_d, self.pad, self.pad))
        u_k_T_yzt = F.conv_transpose3d(f_u_k_yzt, self.conv_kernel_yzt, padding=(self.pad, self.pad, self.pad))
        u_k_T_xzt = F.conv_transpose3d(f_u_k_xzt, self.conv_kernel_xzt, padding=(self.pad_d, self.pad, self.pad))

        # fuse and normalize
        Ru = (u_k_T_xyz.view(N, T, 2, V, D, H, W).permute(0, 3, 1, 4, 5, 6, 2)
              + u_k_T_xyt.view(N, W, 2, V, D, H, T).permute(0, 3, 6, 4, 5, 1, 2)
              + u_k_T_yzt.view(N, D, 2, V, H, W, T).permute(0, 3, 6, 1, 4, 5, 2)
              + u_k_T_xzt.view(N, H, 2, V, D, W, T).permute(0, 3, 6, 4, 1, 5, 2))
        Ru = Ru / self.options['features_out']

        # data consistency
        Au = mri_forward_op(torch.view_as_complex(u_t_1), c, abs(f[:, :, 0, :, 0, :, :]) != 0)
        residual = Au - f
        C = residual.shape[2]
        residual = (self.activation5(residual.real.view(N, V, C * T, D, H * W)).view(N, V, C, T, D, H, W)
                    + 1j * self.activation5(residual.imag.view(N, V, C * T, D, H * W)).view(N, V, C, T, D, H, W))
        Du = torch.view_as_real(mri_adjoint_op(residual, c))

        # gradient-descent update (+ heavy-ball momentum, threaded via S_prev)
        S = Ru * self.lamb_ru + Du * self.lamb_du
        if self.sgd_momentum and S_prev is not None:
            S = S + self.alpha * S_prev
        return u_t_1 - S, S


class FlowVN(nn.Module):
    def __init__(self, **kwargs):
        super(FlowVN, self).__init__()
        options = kwargs
        self.options = options
        self.nc = options['num_stages']  # number of iterations

        self.cell_list = nn.ModuleList([VnMriReconCell(block_id=i, **options) for i in range(self.nc)])

    def forward(self, x, f, c, usrate=None):
        # dtype safety: adjoint/FFT can yield complex128; convs are float32.
        x = x.to(torch.complex64)
        f = f.to(torch.complex64)
        c = c.to(torch.complex64)

        x = torch.view_as_real(x)
        S_prev = None
        for i in range(self.nc):
            x, S_prev = self.cell_list[i](x, f, c, S_prev)
        return torch.view_as_complex(x)
