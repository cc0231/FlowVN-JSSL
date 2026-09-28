import numpy as np
import torch
from cmr_helper_scripts.evaluation2026.pytorch_ssim import *

def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

def SSIM(pred, gt, segmask=None):
    """
    SSIM within segmask (3D). Returns the mean SSIM over voxels where segmask==1.

    pred, gt: shape (Nv, Nt, SPE, PE, FE)
    segmask : shape (SPE, PE, FE)

    Note:
        This implementation is adapted from:
        https://github.com/jinh0park/pytorch-ssim-3D
    """
    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    ssim_fn = SSIM3D(window_size=11, size_average=False)

    # Mask to ROI (broadcast across Nv and Nt).
    pred = pred * segmask[None, None]
    gt   = gt   * segmask[None, None]

    # Normalize by masked GT maximum.
    gt_max = np.max(gt)
    pred = pred / gt_max
    gt   = gt   / gt_max

    pred_t = _to_tensor(pred).float()
    gt_t   = _to_tensor(gt).float()

    # SSIM map: (Nv, Nt, SPE, PE, FE)
    ssim_map = ssim_fn(pred_t, gt_t)
    # Mean SSIM over ROI voxels (segmask == 1), per (Nv, Nt).
    roi = _to_tensor(segmask.astype(bool)).to(ssim_map.device)  # (SPE, PE, FE)
    roi = roi.unsqueeze(0).unsqueeze(0)                         # (1, 1, SPE, PE, FE)

    roi_sum = (ssim_map * roi).sum()
    roi_cnt = roi.sum().clamp_min(1.0) * gt.shape[1] * gt.shape[0]

    return (roi_sum / roi_cnt).item()


def nRMSE(pred, gt, segmask=None, eps=1e-12):
    """
    Normalized RMSE within segmask.

    pred, gt: shape (Nv, Nt, SPE, PE, FE)
    segmask : shape (SPE, PE, FE)
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt   = np.asarray(gt, dtype=np.float32)

    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    segmask = np.asarray(segmask, dtype=bool)

    # (SPE,PE,FE) -> (1,1,SPE,PE,FE) -> broadcast to (Nv,Nt,SPE,PE,FE)
    while segmask.ndim < gt.ndim:
        segmask = segmask[None, ...]
    mask = np.broadcast_to(segmask, gt.shape).astype(np.float32)

    n = np.sum(mask)
    mse = np.sum(((pred - gt) ** 2) * mask) / (n + eps)
    denom = np.max(gt * mask) + eps
    return np.sqrt(mse) / denom

def RelErr(pred, gt, segmask=None, eps=1e-12):
    """
    Relative magnitude error for vector fields.

    pred, gt: shape (Nv, Nt, SPE, PE, FE)
    segmask : shape (SPE, PE, FE)

    Computes || |v_gt| - |v_pred| ||_2 / || |v_gt| ||_2 within mask, aggregated over (Nt,SPE,PE,FE).
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt   = np.asarray(gt, dtype=np.float32)

    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    segmask = np.asarray(segmask, dtype=bool)

    gt_mag   = np.linalg.norm(gt, axis=0)    # (Nt,SPE,PE,FE)
    pred_mag = np.linalg.norm(pred, axis=0)  # (Nt,SPE,PE,FE)

    # (SPE,PE,FE) -> (1,SPE,PE,FE) -> broadcast to (Nt,SPE,PE,FE)
    while segmask.ndim < gt_mag.ndim:
        segmask = segmask[None, ...]
    mask = np.broadcast_to(segmask, gt_mag.shape).astype(np.float32)

    numerator = np.sum(((gt_mag - pred_mag) ** 2) * mask)
    denominator = np.sum((gt_mag ** 2) * mask) + eps
    return np.sqrt(numerator / denominator)

def AngErr(pred, gt, segmask=None, eps=1e-8):
    """
    Angular error (degrees) for vector fields.

    pred, gt: shape (Nv, Nt, SPE, PE, FE)
    segmask : shape (SPE, PE, FE)

    Angle per voxel/time:
        acos( <p,g> / (||p||·||g|| + eps) )
    then averaged inside mask over (Nt,SPE,PE,FE).
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt   = np.asarray(gt, dtype=np.float32)

    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    segmask = np.asarray(segmask, dtype=bool)

    dot = np.sum(pred * gt, axis=0)  # (Nt,SPE,PE,FE)
    norm_p = np.linalg.norm(pred, axis=0)
    norm_g = np.linalg.norm(gt, axis=0)
    cos_sim = np.clip(dot / (norm_p * norm_g + eps), -1.0, 1.0)

    error_map = np.arccos(cos_sim)           # radians

    while segmask.ndim < error_map.ndim:
        segmask = segmask[None, ...]
    mask = np.broadcast_to(segmask, error_map.shape).astype(np.float32)

    n_valid = np.sum(mask) + 1e-12
    return (np.sum(error_map * mask) / n_valid) / np.pi * 180.0

def ComplexDiffErr(pred, ref, segmask, eps=1e-12):
    """
    Symmetric bounded complex diff error in [0, 1]:
        E = ||pred - ref||_2 / (||pred||_2 + ||ref||_2)
    computed inside ROI.
    """
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    segmask = np.asarray(segmask, dtype=bool)

    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {ref.shape}")

    while segmask.ndim < pred.ndim:
        segmask = segmask[None, ...]
    mask = np.broadcast_to(segmask, pred.shape)

    if not np.any(mask):
        raise ValueError("segmask has no True elements.")

    err = np.sqrt(np.sum(np.abs(pred[mask] - ref[mask]) ** 2))
    predn = np.sqrt(np.sum(np.abs(pred[mask]) ** 2))
    refn = np.sqrt(np.sum(np.abs(ref[mask]) ** 2))

    return float(err / (predn + refn + eps))