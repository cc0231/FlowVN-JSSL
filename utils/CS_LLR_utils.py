import time 
import torch
import einops
import numpy as np
from loguru import logger
from types import SimpleNamespace
import scipy
torch.backends.cudnn.benchmark = True

def k2i_numpy(x, ax=None, sens=None):
    if ax is None:
        ax = [-2, -1]
    if sens is None:
        return scipy.fft.fftshift(scipy.fft.ifftn(scipy.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho", workers=-1),
                            axes=ax)
    else:
        x = scipy.fft.fftshift(scipy.fft.ifftn(scipy.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho", workers=-1),
                            axes=ax)
        x = np.sum(x * np.conj(sens[..., None, None]), axis=3)
        return x
    
def DerivOp_1d(x, dim):
    Dx = x.roll(1, dim) - x
    return Dx

def DerivOpT_1d(Dx, dim):
    return Dx.roll(-1, dim) - Dx

def get_vtv_reg_multi_v_f_lowmem(V, T, lamb_tv, lamb_tv_deps=0.001, lamb_tv_w=(1, 0.05, 0.05, 0.05) ):
    def vtv_reg_multi_v_f_lowmem(img):
        img = einops.rearrange(img, '(V T) X Y Z -> V T X Y Z', V=V, T=T)
        sDx = 0.
        for di in range(4):
            _d = DerivOp_1d(img, 1 + di) * torch.tensor(lamb_tv_w[di]).sqrt()
            sDx += einops.reduce(_d.abs().square(), 'V T X Y Z -> 1 T X Y Z', 'sum')
            _d = None
        sDx = torch.sqrt(sDx + lamb_tv_deps)
        fval = sDx.sum() * lamb_tv
        grad = 0.
        for di in range(4):
            _d = DerivOp_1d(img, 1 + di) * torch.tensor(lamb_tv_w[di]).sqrt()
            _d /= sDx
            grad += lamb_tv * DerivOpT_1d(_d, 1 + di) * torch.tensor(lamb_tv_w[di]).sqrt()
            _d = None
        grad = einops.rearrange(grad, 'V T X Y Z -> (V T) X Y Z')
        return fval, grad
    return vtv_reg_multi_v_f_lowmem


def k2i(X, dims=-1, force_fftn_device=None):
    """
    X: (..., H, W)
    """
    Xdev = X.device
    if isinstance(dims, int):
        dims = [dims]
    if not X.is_complex():
        X = X.to(torch.complex64)
    for dim in dims:
        X = torch.fft.ifftshift(X, dim=dim)
        if force_fftn_device is not None:
            X = torch.fft.ifftn(X.to(force_fftn_device), dim=dim, norm="ortho").to(Xdev)
        else:
            X = torch.fft.ifftn(X, dim=dim, norm="ortho")
        X = torch.fft.fftshift(X, dim=dim)
    return X

def i2k(X, dims=-1, force_fftn_device=None):
    """
    X: (..., H, W) 
    """
    Xdev = X.device
    if isinstance(dims, int):
        dims = [dims]
    if not X.is_complex():
        X = X.to(torch.complex64)
    for dim in dims:
        X = torch.fft.ifftshift(X, dim=dim)
        if force_fftn_device is not None:
            X = torch.fft.fftn(X.to(force_fftn_device), dim=dim, norm="ortho").to(Xdev)
        else:
            X = torch.fft.fftn(X, dim=dim, norm="ortho")
        X = torch.fft.fftshift(X, dim=dim)
    return X

def return_cartesian_acq_operator(mask, kspc, dims=-1, coils=None, **i2k_kwargs):
    if np.any(np.array(dims) >= 0):
        raise ValueError(f'Found dims>=0! You have to expect dimension padding from the left. Better use negative dims. Look at this code!!!')
    if torch.is_grad_enabled():
        raise ValueError(f'Gradient enabled! You should disable it for this function.')
     
    if coils is None:
        def A(x):
            # x: T X Y
            return i2k(x, dims=dims, **i2k_kwargs) * mask
        def At(y): 
            # x: T X Y
            tmp = k2i(mask * y, dims=dims, **i2k_kwargs) 
            return tmp
    else:
        assert coils.is_complex()
        C, T, X, Y, Z = kspc.shape
        def A(x):
            # x: T X Y
            xc = einops.rearrange(x, '... -> 1 ...') * coils
            return i2k(xc, dims=dims) * mask
        def At(y): 
            # y: C T X Y
            C, T, X, Y, Z = y.shape
            Aty = torch.zeros([T, X, Y, Z], dtype=y.dtype, device=y.device)
            for ic in range(C):
                Aty += k2i(mask[0, ...] * y[ic, ...], dims=dims, **i2k_kwargs) * coils[ic, ...].conj()
            return Aty
        def AtA(x):
            # x: T X Y Z
            C = coils.shape[0]
            AtAx = 0.
            for ic in range(C):
                Ax_b = i2k(x * coils[ic, ...], dims=dims, **i2k_kwargs) * mask[0, ...]
                AtAx += k2i(Ax_b, dims=dims, **i2k_kwargs) * coils[ic, ...].conj()
            return AtAx
        def AtA_wcost(x):
            # x: T X Y Z
            C = coils.shape[0]
            cost = 0.
            AtAx_b = torch.zeros_like(x)
            for ic in range(C):
                Ax_b = i2k(x * coils[ic, ...], dims=dims, **i2k_kwargs) * mask[0, ...] - kspc[ic, ...]
                AtAx_b += k2i(Ax_b, dims=dims, **i2k_kwargs) * coils[ic, ...].conj()
                cost = cost + Ax_b.abs().square().sum()/2.
                
                # AtAx_b = AtAx_b + k2i(Ax_b, dims=dims, **i2k_kwargs) * coils[ic, ...].conj()
            return AtAx_b, cost
    operator = SimpleNamespace(A=A, At=At, AtA_wcost=AtA_wcost, AtA=AtA)
    return operator


def get_paddings(imsz, bsz):
    imsz = torch.tensor(imsz, dtype=torch.float32)
    bsz = torch.tensor(bsz, dtype=torch.float32)
    return torch.tensor(bsz * torch.ceil(imsz / bsz), dtype=torch.long)


def mask_ref_grad(x, x_ref, mask):
    # ||mask * (x - x_ref)||_2^2
    dif = mask * (x - x_ref)
    cost = dif.abs().square().sum() / 2.
    grad = mask * dif
    return grad, cost

def mbtorch(x):
    if isinstance(x, torch.Tensor):
        return x
    else:
        return torch.tensor(x)
    
def DerivOp(x, dim_weights=(1,0.1,0.1,0.1)):
    Dx1 = Dx2 = Dx3 = Dx4 = None
    if dim_weights[0] > 0:
        Dx1 = (x.roll(1, 0) - x) * mbtorch(dim_weights[0]).sqrt()
    if dim_weights[1] > 0:
        Dx2 = (x.roll(1, 1) - x) * mbtorch(dim_weights[1]).sqrt()
    if dim_weights[2] > 0:
        Dx3 = (x.roll(1, 2) - x) * mbtorch(dim_weights[2]).sqrt()
    if x.ndim == 4 and dim_weights[3] > 0:
        Dx4 = (x.roll(1, 3) - x) * mbtorch(dim_weights[3]).sqrt()
    return Dx1, Dx2, Dx3, Dx4


def _DerivOpT_inplace(Dx1, Dx2, Dx3, Dx4, dim_weights=(1,0.1,0.1,0.1)):
    for notNone in [Dx1, Dx2, Dx3, Dx4]:
        if notNone is not None:
            break
    ret = torch.zeros_like(notNone)

    if Dx1 is not None:
        ret += (Dx1.roll(-1, 0) - Dx1) * mbtorch(dim_weights[0]).sqrt()
    if Dx2 is not None:
        ret += (Dx2.roll(-1, 1) - Dx2) * mbtorch(dim_weights[1]).sqrt()
    if Dx3 is not None:
        ret += (Dx3.roll(-1, 2) - Dx3) * mbtorch(dim_weights[2]).sqrt()
    if Dx4 is not None:
        ret += (Dx4.roll(-1, 3) - Dx4) * mbtorch(dim_weights[3]).sqrt()
    return ret

def _DerivOpT(Dx1, Dx2, Dx3, Dx4, dim_weights=(1,0.1,0.1,0.1)):
    for notNone in [Dx1, Dx2, Dx3, Dx4]:
        if notNone is not None:
            break
    ret = 0.

    if Dx1 is not None:
        ret = ret + (Dx1.roll(-1, 0) - Dx1) * mbtorch(dim_weights[0]).sqrt()
    if Dx2 is not None:
        ret = ret + (Dx2.roll(-1, 1) - Dx2) * mbtorch(dim_weights[1]).sqrt()
    if Dx3 is not None:
        ret = ret + (Dx3.roll(-1, 2) - Dx3) * mbtorch(dim_weights[2]).sqrt()
    if Dx4 is not None:
        ret = ret + (Dx4.roll(-1, 3) - Dx4) * mbtorch(dim_weights[3]).sqrt()
    return ret

def DerivOpT(Dx1, Dx2, Dx3, Dx4, dim_weights=(1,0.1,0.1,0.1)):
    if torch.is_grad_enabled():
        return _DerivOpT(Dx1, Dx2, Dx3, Dx4, dim_weights=dim_weights)
    else:
        return _DerivOpT_inplace(Dx1, Dx2, Dx3, Dx4, dim_weights=dim_weights)
    

def to_torch_tensor(data):
    """

    Args:
        data: The data to convert. Can be a list, numpy array or another PyTorch tensor.
        device (str, optional): The device to store the tensor on. Default is 'cpu'.
        dtype (torch.dtype, optional): The desired data type of the tensor. Default is torch.float32.

    Returns:
        torch.Tensor: The data as a PyTorch tensor.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, list) or isinstance(data, tuple):
        return torch.tensor(data)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    

def generate_shifts(bsz):
    if bsz.numel() == 2:
        return to_torch_tensor((torch.randint(0, bsz[0], (1,)), torch.randint(0, bsz[1], (1,)))).long()
    elif bsz.numel() == 3:
        return to_torch_tensor((torch.randint(0, bsz[0], (1,)), 
                                torch.randint(0, bsz[1], (1,)), 
                                torch.randint(0, bsz[2], (1,)))).long()
    else:
        print(bsz)
        raise ValueError(f'Invalid bsz {bsz} its size is {bsz.numel()}')

    
def apply_shifts(img, shifts):
    if shifts.numel() == 2:
        return torch.roll(img, (shifts[-2], shifts[-1]), dims=(-2, -1))
    elif shifts.numel() == 3:
        return torch.roll(img, (shifts[-3], shifts[-2], shifts[-1]), dims=(-3, -2, -1))

def im2blocks_2d(img, bsz, buffer=None):
    """
    img: (..., H, W)
    returns: (..., H*W, b1*b2) with padding
    """
    bsz = to_torch_tensor(bsz).long()
    if bsz.numel() == 1:
        bsz = to_torch_tensor([bsz, bsz]).long()
    H, W = img.shape[-2:]

    pad1 = bsz[0] * torch.ceil(H / bsz[0]) - H
    pad2 = bsz[1] * torch.ceil(W / bsz[1]) - W
    if pad1 > 0 or pad2 > 0:
        img = torch.nn.functional.pad(img, (0, pad2.long(), 0, pad1.long()), mode='constant', value=0)
    H_pad, W_pad = img.shape[-2:]
    blocks = einops.rearrange(img, 'T (h b1) (w b2) -> (h w) T (b1 b2)', b1=bsz[0], b2=bsz[1])
    return blocks, (H_pad, W_pad)

def im2blocks_3d(img, bsz, buffer=None):
    """
    img: (..., H, W, D)
    returns: (..., H*W*D, b1*b2*b3) with padding
    """
    bsz = to_torch_tensor(bsz).long()
    if bsz.numel() == 1:
        bsz = to_torch_tensor([bsz, bsz, bsz]).long()
    H, W, D = img.shape[-3:]

    pad1 = bsz[0] * torch.ceil(H / bsz[0]) - H
    pad2 = bsz[1] * torch.ceil(W / bsz[1]) - W
    pad3 = bsz[2] * torch.ceil(D / bsz[2]) - D
    # print('im shape 0', img.shape)
    if buffer is None:
        if pad1 > 0 or pad2 > 0 or pad3 > 0:
            img = torch.nn.functional.pad(img, (0, pad3.long(), 0, pad2.long(), 0, pad1.long()), mode='constant', value=0)
    else:
        buffer[:, :H, :W, :D] = img
        img = buffer
        
    sz_pad = img.shape[-3:]
    blocks = einops.rearrange(img, 'T (h b1) (w b2) (d b3) -> (h w d) T (b1 b2 b3)', b1=bsz[0], b2=bsz[1], b3=bsz[2])
    return blocks, sz_pad

from torch.autograd.functional import vjp
class eigh_custom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        s, u = torch.linalg.eigh(A)
        ctx.save_for_backward(s.clone(), u.clone(), A.clone())
        return s, u

    @staticmethod
    def backward(ctx, ds, du):
        N = ds.shape[-1]
        s, u, A = ctx.saved_tensors

        VhgV = torch.matmul(u.mH, du)
        VhgV = (VhgV - VhgV.mH) / 2
        Ld = einops.rearrange(s, 'b i -> b 1 i') - einops.rearrange(s, 'b i-> b i 1')
        Ld.diagonal(dim1=-2, dim2=-1).fill_(1.)
        gA = VhgV / Ld
        gA[gA.isnan()] = 0
        gA[gA.isinf()] = 0
        gA.diagonal(dim1=-2, dim2=-1).copy_(ds)
        gA = torch.matmul(u, torch.matmul(gA, u.mH))
        try:
            _, v = vjp(torch.linalg.eigh, A, (ds, du))
            if (v - gA).abs().max() > 1e-4:
                logger.error('Inconsistency with torchs vjp ', (v - gA).abs().max(), (v - gA).abs().mean())
        except:
            logger.warning('Default VJP failed!')
        return gA
    
def _eigh_maybe_cpu(C):
    """torch.linalg.eigh, but if the CUDA linalg backend fails to load
    (missing libtorch_cuda_linalg.so / cusolver on some nodes), transparently
    run the decomposition on CPU and move the result back to the original device.
    Only the fallback path pays the host<->device copy; healthy setups are unaffected."""
    try:
        return torch.linalg.eigh(C)
    except RuntimeError as e:
        msg = str(e).lower()
        if 'linalg' not in msg and 'dlopen' not in msg and 'cusolver' not in msg:
            raise
        dev = C.device
        S, U = torch.linalg.eigh(C.cpu())
        return S.to(dev), U.to(dev)


def svdecon_par(blocks, force_eig_on_cpu=False, deps=1e-6):
    B, M, N = blocks.shape
    if M <= N:
        C = blocks @ blocks.conj().transpose(-1, -2)
        if force_eig_on_cpu:
            C = C.to('cpu')

        if torch.is_grad_enabled():
            Sb, Ub = eigh_custom.apply(C)
        else:
            Sb, Ub = _eigh_maybe_cpu(C)

        if force_eig_on_cpu:
            Sb, Ub = Sb.to(blocks.device), Ub.to(blocks.device)
        Sb = Sb.clamp(0).sqrt()
        Vb = Ub.conj().transpose(-1, -2) @ blocks
        if torch.is_grad_enabled():
            Vb = Vb / (Sb.unsqueeze(-1) + deps)
        else:
            Vb /= (Sb.unsqueeze(-1) + deps)
    else:
        C = blocks.conj().transpose(-1, -2) @ blocks
        if force_eig_on_cpu:
            C = C.to('cpu')
        if torch.is_grad_enabled():
            Sb, Vb = eigh_custom.apply(C)
        else:
            Sb, Vb = _eigh_maybe_cpu(C)
        if force_eig_on_cpu:
            Sb, Vb = Sb.to(blocks.device), Vb.to(blocks.device)
        Sb = Sb.clamp(0).sqrt()
        Ub = blocks @ Vb
        if torch.is_grad_enabled():
            Ub = Ub / (Sb.unsqueeze(-2) + deps)
        else:
            Ub /= (Sb.unsqueeze(-2) + deps)
        Vb = Vb.conj().transpose(-1, -2)
    return Ub, Sb, Vb

def svdecon_mul(U, S, Vh):
    if U.shape[-1] == U.shape[-2]:
        return (U * S.unsqueeze(-2)) @ Vh
    elif Vh.shape[-1] == Vh.shape[-2]:
        return U @ (Vh * S.unsqueeze(-1))
    else:
        raise ValueError(f'Invalid U shape {U.shape}, Vh shape {Vh.shape}')

def blocks2im_2d(blocks, bsz, sz, sz_pad):
    """
    blocks: (..., Hpad*Wpad, b1*b2) 
    returnsL (..., H, W)
    """
    H, W = sz
    Hpad, Wpad = sz_pad
    bsz = to_torch_tensor(bsz).long()
    if bsz.numel() == 1:
        bsz = to_torch_tensor([bsz, bsz]).long()
    blocks = einops.rearrange(blocks, '(h w) T (b1 b2) -> T (h b1) (w b2)', 
                              b1=bsz[0], b2=bsz[1], 
                              h=(Hpad/bsz[0]).long(), w=(Wpad/bsz[1]).long())
    blocks = blocks[..., :H, :W]
    return blocks

def blocks2im_3d(blocks, bsz, sz, sz_pad):
    """
    blocks: (..., Hpad*Wpad*Dpad, b1*b2*b3) 
    returnsL (..., H, W, D)
    """
    bsz = to_torch_tensor(bsz).long()
    if bsz.numel() == 1:
        bsz =to_torch_tensor([bsz, bsz, bsz]).long()
    
    H, W, D = sz
    Hpad, Wpad, Dpad = sz_pad

    blocks = einops.rearrange(blocks, '(h w d) T (b1 b2 b3) -> T (h b1) (w b2) (d b3)', 
                              b1=bsz[0], b2=bsz[1], b3=bsz[2], 
                              h=(Hpad/bsz[0]).long(), w=(Wpad/bsz[1]).long(), d=(Dpad/bsz[2]).long())
    blocks = blocks[..., :H, :W, :D]
    return blocks


def singular_proj(y, lamb, bsz, buffer_image=None, force_eig_on_cpu=False):
    """
    y: T, H, W or T, H, W, D
    """
    if y.ndim == 3:
        nd = 2
    elif y.ndim == 4:
        nd = 3
    else:
        raise ValueError(f'Invalid y shape {y.shape}')

    bsz = to_torch_tensor(bsz).long()
    if bsz.numel() == 1:
        if nd == 3:
            bsz = to_torch_tensor([bsz, bsz, bsz]).long()
        elif nd == 2:
            bsz = to_torch_tensor([bsz, bsz]).long()

    shifts = generate_shifts(bsz)
    y = apply_shifts(y, shifts)

    imsz = y.shape[-nd:]
    if nd == 2:
        blocks, sz_pad = im2blocks_2d(y, bsz, buffer=buffer_image)
    elif nd == 3:
        blocks, sz_pad = im2blocks_3d(y, bsz, buffer=buffer_image)
    # print(f'blocks shape {blocks.shape} {blocks.dtype} {blocks.device}')
    

    u, s, v = svdecon_par(blocks, force_eig_on_cpu=force_eig_on_cpu)

    s = torch.clamp(s - lamb, min=0)
    nuclear_cost = s.sum()
    s = s.to(torch.complex64)
    x = svdecon_mul(u, s, v)
    if nd == 2:
        x = blocks2im_2d(x, bsz, imsz, sz_pad)
    elif nd == 3:
        x = blocks2im_3d(x, bsz, imsz, sz_pad)
    x = apply_shifts(x, -shifts)
    return x, nuclear_cost

def pgd_recon(ksp, acq_operator, lamb, bsz=[8, 8], maxiters=120, alpha=0.8, tau=0.98, alpha_min=1e-6, 
              l2_ref_penalty=0., l2_ref_image=None, l2_ref_mask=None, debug=False, debug_upd_iter=1,
               itinfo_to_item=True, compute_cost_every=1, l2_dt=0., l1_dt=0, l1_dt_eps=1e-6, l1_dt_w=(1, 0.1, 0.1, 0.1),
               return_x_per_iter=False, force_eig_on_cpu=False, custom_regularizer_fn=None,
               x0=None, opt_type='nesterov'):
    
    bsz = torch.tensor(bsz)
    itinfo = SimpleNamespace(cost=[], data_cost=[], nuclear_cost=[], l2_ref_cost = [], l2_dt_cost=[], 
                             l1_dt_cost=[], custom_reg_cost=[], alphas=[], x_per_iter=[])
    
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=torch.float)
    t_prev = torch.tensor(1., dtype=torch.float)

    if x0 is None:
        x = acq_operator.At(ksp) 
    else:
        x = x0
    z = x.clone()
    z_prev = z.clone()

    T = x.shape[0]
    buffer_image = None
    if x.ndim == 4:
        szpad = get_paddings(x.shape[-3:], bsz)
        if not torch.is_grad_enabled():
            buffer_image = torch.zeros([T, szpad[0], szpad[1], szpad[2]], dtype=x.dtype, device=x.device)
    elif x.ndim == 3:
        szpad = get_paddings(x.shape[-2:], bsz)
        if not torch.is_grad_enabled():
            buffer_image = torch.zeros([T, szpad[0], szpad[1]], dtype=x.dtype, device=x.device)
    
    fval_custom_reg = l2_dt_cost = l2_ref_cost = l1_dt_cost = None

    for it in range(maxiters):
        tm0 = time.time()
        if it % compute_cost_every == 0:
            grad, data_cost = acq_operator.AtA_wcost(x)
        else:
            raise NotImplementedError("This is bug!")
            grad = acq_operator.AtA(x)
            data_cost = None
        # torch.cuda.synchronize()
        time_data = time.time() - tm0
        
        tm0 = time.time()
        if l2_ref_penalty > 0:
            grad_ref, l2_ref_cost = mask_ref_grad(x, l2_ref_image, l2_ref_mask)
            grad = grad + l2_ref_penalty * grad_ref
            l2_ref_cost = l2_ref_penalty * l2_ref_cost
            grad_ref = None

        if l2_dt > 0:
            Dx = DerivOp_1d(x, 0)
            if torch.is_grad_enabled():
                grad = grad + l2_dt * DerivOpT_1d(Dx, 0)
            else:
                grad += grad + l2_dt * DerivOpT_1d(Dx, 0)
            l2_dt_cost = l2_dt * Dx.abs().square().sum() / 2.
            Dx = None
        
        if l1_dt > 0:
            Dx = DerivOp(x, dim_weights=l1_dt_w)
            sDx = 0.
            for d in Dx:
                if d is not None:
                    if torch.is_grad_enabled():
                        sDx = sDx + d.abs().square()
                    else:
                        sDx += d.abs().square()
            sDx = torch.sqrt(sDx + l1_dt_eps)
            l1_dt_cost = l1_dt * sDx.sum()
            ssDx = [dx / sDx if dx is not None else None for dx in Dx]
            gg = DerivOpT(*ssDx, dim_weights=l1_dt_w)
            grad = grad + l1_dt * gg
            sDx = Dx = gg = None
        
        if custom_regularizer_fn is not None:
            fval_custom_reg, _grad = custom_regularizer_fn(x)
            if torch.is_grad_enabled():
                grad = grad + _grad
            else:
                grad += _grad
            _grad = None
        time_regs = time.time() - tm0

        tm0 = time.time()
        if opt_type == 'nesterov':
            z_prev, z = z, z_prev
            tm1 = time.time()
            if torch.is_grad_enabled():
                x = x - alpha * grad
            else:
                x -= alpha * grad        
            grad = None
            z, nuclear_cost = singular_proj(x, lamb * alpha, bsz, buffer_image=buffer_image, 
                                            force_eig_on_cpu=force_eig_on_cpu)
            time_reg = time.time() - tm1
            del grad
            t = (1. + torch.sqrt(1. + 4. * t_prev**2)) / 2.
            x = z + ((t_prev - 1.) / t) * (z - z_prev)
            t_prev = t.clone()
        elif opt_type == 'fista': 
            z_prev, z = z, z_prev
            z, nuclear_cost = singular_proj(x - alpha * grad, lamb * alpha, bsz, buffer_image=buffer_image, 
                                            force_eig_on_cpu=force_eig_on_cpu)
            x = z + (it - 2) / (it + 1) * (z - z_prev)
        elif opt_type == 'gd': 
            x = x - alpha * grad
            x, nuclear_cost = singular_proj(x, lamb * alpha, bsz, buffer_image=buffer_image, 
                                            force_eig_on_cpu=force_eig_on_cpu)
            z = x
        time_prox = time.time() - tm0

        alpha = torch.clamp(alpha * tau, min=alpha_min)
        cost = data_cost + nuclear_cost * lamb
        if l2_ref_cost is not None:
            cost = cost + l2_ref_cost
        if l2_dt_cost is not None:
            cost = cost + l2_dt_cost
        if l1_dt_cost is not None:
            cost = cost + l1_dt_cost
        if fval_custom_reg is not None:
            cost = cost + fval_custom_reg
        # cost = data_cost + fval_custom_reg + nuclear_cost * lamb + l2_dt_cost + l1_dt_cost + fval_custom_reg if data_cost is not None else None
        
        if cost is not None:
            if it == 0:
                cost_max = cost.clone()
                cost_min = cost.clone()
            else:
                cost_max = torch.max(cost_max, cost)
                cost_min = torch.min(cost_min, cost)

        
        if itinfo_to_item:
            cost = cost.item() if cost is not None else None
            data_cost = data_cost.item() if cost is not None else None
            l2_ref_cost = l2_ref_cost.item() if l2_ref_cost is not None else None
            l2_dt_cost = l2_dt_cost.item() if l2_dt_cost is not None else None
            l1_dt_cost = l1_dt_cost.item() if l1_dt_cost is not None else None
            fval_custom_reg = fval_custom_reg.item() if fval_custom_reg is not None else None
            nuclear_cost = nuclear_cost.item()
            alpha_i = alpha.item()
        else:
            alpha_i = alpha

        itinfo.cost.append(cost)
        itinfo.data_cost.append(data_cost)
        itinfo.nuclear_cost.append(nuclear_cost)
        itinfo.l2_ref_cost.append(l2_ref_cost)
        itinfo.l2_dt_cost.append(l2_dt_cost)
        itinfo.l1_dt_cost.append(l1_dt_cost)
        itinfo.custom_reg_cost.append(fval_custom_reg)
        itinfo.alphas.append(alpha_i)
        if return_x_per_iter:
            itinfo.x_per_iter.append(z.clone())


        if debug and it % debug_upd_iter == 0:
            if cost is not None:
                print(f'it {it} cost={cost} data_cost {data_cost} nuclear_cost {nuclear_cost}')
                print(f'\ttime_data={time_data:.3f} time_regs={time_regs:.3f} time_prox={time_prox:.3f}')
    return z, itinfo
