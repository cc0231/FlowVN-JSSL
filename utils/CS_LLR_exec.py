import os
import re
import time
import torch
import einops
import argparse
import numpy as np
import h5py
from CS_LLR_utils import *
from ktgaussian import ktgaussian_mask


def _ifft_readout(x):
    """Centered orthonormal inverse FFT along the readout axis (last axis = FE).
    Mirrors utils/dataloader._ifft_readout so the mat path matches the network path."""
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=-1), axis=-1, norm="ortho"), axes=-1)


def _load_mat_complex(path, key):
    """Load an HDF5 (MATLAB v7.3) real/imag compound field as complex64."""
    with h5py.File(path, "r") as f:
        arr = np.array(f[key])
    return (arr["real"] + arr["imag"] * 1j).astype(np.complex64)


def _prep_from_mat(subj_dir, usrate, seed):
    """Reconstruct CS_LLR's raw .npy inputs on the fly from the shipped .mat files.

    Loads kdata_full.mat + coilmap.mat, brings the readout into image domain, and
    applies a fresh k-t Gaussian mask at acceleration `usrate` -- matching the axis
    order, readout iFFT, unit-RSS coil norm, and mask broadcast of utils/dataloader.
    Nothing is written to disk. Returns:
        ksp_full : (V, C, T, FE, PE, SPE) complex64, unsampled (PE,SPE) zeroed (shared across V)
        coils    : (C, FE, PE, SPE)       complex64, unit-RSS normalized
    """
    k = _load_mat_complex(os.path.join(subj_dir, "kdata_full.mat"), "kdata_full")
    k = _ifft_readout(k)                                # readout (last axis, FE) -> image domain
    k = np.transpose(k, (0, 2, 1, 5, 4, 3))            # (V, C, T, FE, PE, SPE)
    V, C, T, FE, PE, SPE = k.shape

    c = _load_mat_complex(os.path.join(subj_dir, "coilmap.mat"), "coilmap")
    c = np.transpose(c, (0, 3, 2, 1))                  # (C, FE, PE, SPE)
    c = c / (np.sqrt(np.sum(np.abs(c) ** 2, axis=0, keepdims=True)) + 1e-8)  # unit-RSS coil norm

    np.random.seed(int(seed))                          # deterministic -> reproducible figure
    mask_tkk = ktgaussian_mask(PE, SPE, T, int(usrate))            # (T, PE, SPE)
    sim_mask = np.broadcast_to(mask_tkk[np.newaxis], (V, T, PE, SPE)).copy()
    k = k * sim_mask[:, np.newaxis, :, np.newaxis]     # zero unsampled (PE,SPE), same mask for all V
    return k.astype(np.complex64), c.astype(np.complex64)


def CS_LLR(lamb_tv, lamb_llr, input_dir, input, output_dir, vol, usrate=None, seed=0, eig_on_cpu=True):
    """
    LLR and spatiotemporal TV are implementated here (for the FlowMRI_net paper lamb_tv=0)

    You could reconstruct all segments (velocity-encodings) at once by removing the seg loop and indexing
    This will be faster but requires more GPU memory

    If <input> (e.g. kdata_ktGaussian{R}.npy) is not present in the subject dir, CS_LLR falls
    back to reconstructing directly from the shipped kdata_full.mat + coilmap.mat, undersampling
    on the fly at rate `usrate` (inferred from the digits in <input> when not given). No
    intermediate files are written -- only the final rec0..rec3.npy.
    """

    os.makedirs(output_dir, exist_ok=True)
    subj_dir = os.path.join(input_dir, vol)
    npy_path = os.path.join(subj_dir, input)
    use_mat = not os.path.isfile(npy_path)
    ksp_full = coils_full = None
    if use_mat:
        if usrate is None:
            m = re.search(r"(\d+)", input)
            if m is None:
                raise ValueError(
                    f"cannot infer undersampling rate from --input '{input}'; pass --usrate")
            usrate = int(m.group(1))
        print(f"[CS_LLR] '{npy_path}' not found -> reconstructing from kdata_full.mat "
              f"at R={usrate} (seed={seed})", flush=True)
        ksp_full, coils_full = _prep_from_mat(subj_dir, usrate, seed)

    for seg in [0,1,2,3]:
        if use_mat:
            ksp = ksp_full[[seg]].astype(np.complex64)  # (V, C, T, D, H, W) complex64, D=FE (image domain)
        else:
            ksp = np.load(npy_path, mmap_mode='c')[[seg], :, :, 0, :, :, :].astype(np.complex64)  # (V, C, T, D, H, W) complex64
        ksp = np.fft.ifftshift(np.fft.fft(np.fft.ifftshift(ksp, axes=3), axis=3, norm="ortho"), axes=3)
        retro_prosp = False  # retrospectively undersampling of prospectively-undersampled data (for brain)
        if retro_prosp:
            m = np.load(input_dir + vol + '/retro_mask_R24.npy')[[seg], :, :, :]
            ksp *= m[:, np.newaxis, :, np.newaxis, :, :]

        if use_mat:
            coils = coils_full[:, np.newaxis, :, :, :].astype(np.complex64)  # (C, 1, D, H, W) complex64
        else:
            coils = np.load(subj_dir+'/coilmap_full.npy', mmap_mode='c')[:, np.newaxis, :, :, :].astype(np.complex64)  # (C, 1, D, H, W) complex64
        mask = ksp[:,0:1,:,0:1,:,:] !=  0 #(V, 1, T, 1, H, W) bool

        V, C, T, X, Y, Z = ksp.shape
        norm_k = np.median(np.abs(ksp[:,:,:, X//2, Y//2, Z//2])) / 10
        ksp = ksp / norm_k

        with torch.no_grad():
            ksp = torch.from_numpy(ksp).to(torch.complex64)
            ksp = k2i(ksp, -3)
            coils = torch.from_numpy(coils).to(torch.complex64)
            mask = torch.from_numpy(mask).to(torch.float32)
            mask = (ksp.abs().sum(dim=(1, -3), keepdims=True) > 0).float()

        # fixed CS-LLR hyperparameters
        lamb_tv_deps = 0.005            # epsilon to prevent zero gradient and smooth the optimization
        lamb_tv_w = (1, 0.1, 0.1, 0.1)  # weightings for gradients over (t, x, y, z)
        bsz = 16                        # LLR block/patch size

        vtv_reg_multi_v_f_lowmem = get_vtv_reg_multi_v_f_lowmem(V, T, lamb_tv, lamb_tv_deps, lamb_tv_w)
                
        with torch.no_grad():
            dtype = torch.complex64
            dev = 'cuda'
            coils = coils.to(dev).to(dtype)
            kspc = ksp.to(dev).to(dtype)
            
            kspc = einops.rearrange(kspc, 'V C T X Y Z -> C (V T) X Y Z')
            mask = (einops.reduce(kspc.abs(), 'C VT X Y Z -> 1 VT 1 Y Z', 'sum') > 0.).float()
            mask = mask.to(dev)
            
            acq_op = return_cartesian_acq_operator(mask, kspc, dims=[-1, -2], coils=coils, force_fftn_device=None)

            torch.cuda.synchronize()
            torch.cuda.reset_max_memory_allocated()
            start_max_memory = torch.cuda.max_memory_allocated()
            tm0 = time.time()
            rec, itinfo = pgd_recon(kspc, acq_op, lamb_llr, alpha=0.3, tau=0.97, 
                                                    bsz=bsz, debug=True, maxiters=70, 
                                                    custom_regularizer_fn=vtv_reg_multi_v_f_lowmem,
                                                    force_eig_on_cpu=eig_on_cpu) # CPU eig avoids libtorch_cuda_linalg.so (cusolver) load failures on some nodes
            
            torch.cuda.synchronize()
            print(f'Elapsed {time.time() - tm0}')
            end_max_memory = torch.cuda.max_memory_allocated()
            
            rec = rec.cpu()
            rec = einops.rearrange(rec, '(V T) X Y Z -> V T X Y Z', V=V, T=T)
            kspc = acq_op = None
            
            rec = rec * norm_k

            print('Mem: ', (end_max_memory - start_max_memory)/1024/1024, end_max_memory/1024/1024, start_max_memory/1024/1024)

        # save reconstruction
        np.save(output_dir + 'rec{}.npy'.format(seg), rec)

    pass
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS-LLR arguments')

    parser.add_argument('--lambda_llr', type=float, default=0.0, help='weighting of LLR regularization')
    parser.add_argument('--lambda_tv',  type=float, default=0.0, help='weighting of TV regularization')
    parser.add_argument('--input_dir',	type=str,   default='',  help='directory of the input data')
    parser.add_argument('--input',	    type=str,   default='',  help='name of network input file (e.g. rawdata_full_1_R16.npy)')
    parser.add_argument('--output_dir',	type=str,   default='',  help='directory of the output data')
    parser.add_argument('--vol',	    type=str,   default='',  help='volunteer to reconstruct')
    parser.add_argument('--usrate',	    type=int,   default=None, help='acceleration for the on-the-fly kdata_full.mat fallback (default: digits parsed from --input)')
    parser.add_argument('--seed',	    type=int,   default=0,   help='RNG seed for the on-the-fly k-t Gaussian mask (reproducible recon)')
    parser.add_argument('--gpu_eig',    action='store_true',     help='run the LLR eigendecomposition on GPU (faster, but needs a working libtorch_cuda_linalg.so/cusolver); default runs eig on CPU')

    args = parser.parse_args()
    args = vars(args)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms = True

    CS_LLR(lamb_tv=args["lambda_tv"], lamb_llr=args["lambda_llr"], input_dir=args["input_dir"], input=args["input"], output_dir=args["output_dir"], vol=args["vol"], usrate=args["usrate"], seed=args["seed"], eig_on_cpu=not args["gpu_eig"])
