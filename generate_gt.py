"""
Pre-compute gt_imdata.mat for supervised training.

For each subject directory found under --root_dir, loads k-space data
(kdata_full.mat if present, else kdata_ktGaussian*.mat) and coilmap.mat,
performs a fully-sampled adjoint reconstruction, and saves the result as
gt_imdata.mat with shape (V, T, kx, ky, kz) in the same HDF5 compound
format (real/imag fields) as the existing kdata files.

Usage:
    python generate_gt.py --root_dir /path/to/TrainSet/Aorta/Center008
    python generate_gt.py --root_dir /path/to/ValidationSet/Aorta/Center011
"""

import argparse
import glob
import re
import numpy as np
import h5py
from pathlib import Path


def load_kdata_full(path):
    with h5py.File(path, 'r') as f:
        arr = np.array(f['kdata_full'])
    return (arr['real'] + arr['imag'] * 1j).astype(np.complex64)


def load_kdata_ktgaussian(path):
    with h5py.File(path, 'r') as f:
        arr = np.array(f['kdata_ktGaussian'])
    return (arr['real'] + arr['imag'] * 1j).astype(np.complex64)


def load_coilmap(path):
    with h5py.File(path, 'r') as f:
        arr = np.array(f['coilmap'])
    return (arr['real'] + arr['imag'] * 1j).astype(np.complex64)


def reconstruct_gt(kdata, coilmap):
    """
    Adjoint reconstruction (ifft2 + coil-combine) using the same convention
    as mriAdjointOp in utils/misc_utils.py.

    kdata:   (V, C, T, kx, ky, kz)  after transpose(0,2,1,4,5,3)
    coilmap: (C, kx, ky, kz)        after transpose(0,2,3,1)
    Returns: (V, T, kx, ky, kz) complex64
    """
    # ifftshift → ifft2 → fftshift  (same as mriAdjointOp per coil)
    k = np.fft.ifftshift(kdata, axes=(-2, -1))
    img = np.fft.ifft2(k, axes=(-2, -1), norm='ortho')
    img = np.fft.fftshift(img, axes=(-2, -1))
    gt = np.sum(img * np.conj(coilmap)[np.newaxis, :, np.newaxis, :, :, :], axis=1)
    return gt.astype(np.complex64)


def save_gt_mat(path: Path, gt: np.ndarray):
    """Save complex64 array as HDF5 .mat with separate real/imag fields,
    matching the compound format of the existing kdata_full.mat files."""
    dt = np.dtype([('real', np.float32), ('imag', np.float32)])
    arr = np.empty(gt.shape, dtype=dt)
    arr['real'] = gt.real
    arr['imag'] = gt.imag
    with h5py.File(path, 'w') as f:
        f.create_dataset('gt_imdata', data=arr)


def process_subject(subj_dir: Path, overwrite: bool = False):
    gt_path = subj_dir / 'gt_imdata.mat'
    if gt_path.exists() and not overwrite:
        print(f'  skip (exists): {gt_path}')
        return

    coilmap_path = subj_dir / 'coilmap.mat'
    if not coilmap_path.exists():
        print(f'  skip (no coilmap.mat): {subj_dir}')
        return

    # Prefer fully-sampled data; fall back to ktGaussian
    kdata_full_path = subj_dir / 'kdata_full.mat'
    if kdata_full_path.exists():
        print(f'  loading kdata_full.mat  …', end=' ', flush=True)
        raw = load_kdata_full(kdata_full_path)
    else:
        pattern = str(subj_dir / 'kdata_ktGaussian*.mat')
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f'  skip (no k-space file): {subj_dir}')
            return
        print(f'  loading {Path(matches[0]).name}  …', end=' ', flush=True)
        raw = load_kdata_ktgaussian(matches[0])

    # raw shape from .mat:  original HDF5 layout varies; transpose to (V,C,T,kx,ky,kz)
    raw = np.transpose(raw, (0, 2, 1, 4, 5, 3))

    print(f'kdata {raw.shape}', end=' ', flush=True)

    coilmap = load_coilmap(coilmap_path)
    coilmap = np.transpose(coilmap, (0, 2, 3, 1))  # → (C, kx, ky, kz)

    gt = reconstruct_gt(raw, coilmap)              # (V, T, kx, ky, kz)
    print(f'→ gt {gt.shape}', end=' ', flush=True)

    save_gt_mat(gt_path, gt)
    print(f'→ saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory to search for subject folders')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing gt_imdata.mat files')
    args = parser.parse_args()

    root = Path(args.root_dir)
    # Find all directories that contain coilmap.mat
    subj_dirs = sorted({p.parent for p in root.rglob('coilmap.mat')})
    print(f'Found {len(subj_dirs)} subject(s) under {root}')

    for subj_dir in subj_dirs:
        print(f'Processing {subj_dir}')
        process_subject(subj_dir, overwrite=args.overwrite)
