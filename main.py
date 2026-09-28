from pathlib import Path
from datetime import timedelta
import argparse
import numpy as np
from glob import glob
import re
import os

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from utils.misc_utils import *
from utils.dataloader import OwnDataset, mat_to_npy_segmask, JSSLDataset, SubjectGroupedSampler
import pl_bolts

from networks.flowvn import FlowVN
from networks.flowvn_baseline import FlowVN as FlowVN_baseline
import torch
import csv
try:
    from cmr_helper_scripts.evaluation2026.utils_metrics import SSIM, nRMSE, RelErr, AngErr
    from cmr_helper_scripts.evaluation2026.utils_flow import complex2magflow
    from cmr_helper_scripts.evaluation2026.pytorch_ssim import SSIM3D
    _HAVE_EVAL_METRICS = True
except Exception:
    _HAVE_EVAL_METRICS = False

try:
    from cmr_helper_scripts.evaluation2026.utils_bgc import execute_MSAC
    _HAVE_BGC = True
except Exception:
    _HAVE_BGC = False


def _save_coo_npz(path, arr):
    """Save a dense complex array as a sparse COO .npz (coords/data/shape)."""
    arr = np.asarray(arr).astype('complex64')
    coords = np.argwhere(arr != 0).astype(np.int32)
    data = arr[tuple(coords.T)] if coords.size else arr.reshape(-1)[:0]
    np.savez_compressed(path, coords=coords, data=data, shape=np.array(arr.shape, dtype=np.int64))


class UnrolledNetwork(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.options = kwargs
        self.log_img_count = 0

        if self.options["network"] != "FlowVN":
            raise ValueError("only --network FlowVN is supported")
        self.network = (FlowVN_baseline if self.options.get("baseline", False) else FlowVN)(**self.options)

        self.L1_loss = nn.L1Loss()
        self._test_cache = {}
        self._ssim3d = SSIM3D(window_size=11, size_average=False) if (_HAVE_EVAL_METRICS and self.options.get('ssim_weight', 0) > 0) else None
        if self.options.get('ssim_weight', 0) > 0 and not _HAVE_EVAL_METRICS:
            raise RuntimeError(
                "--ssim_weight > 0 but 'cmr_helper_scripts.evaluation2026' failed to import, so the "
                "SSIM term would contribute exactly 0 for this entire run. Fix the import (it also "
                "gates the val SSIM/nRMSE/RelErr/AngErr metrics) or pass --ssim_weight 0."
            )
        self._vis_acc = {}
        self._vis_subj = None
        self.vis_nv_idx = 0
        self.vis_nt_idx = 0
        self._weight_snapshot = None      # CPU snapshot of last clean weights for skip-revert
        self._snap_steps = 0

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Compute the zero-filled initial image imdata_p1 = adjoint(kdata_p1) on-GPU,
        after the batch is moved to device. The CPU dataloader no longer does this
        per-sample FFT; this is the same value (the adjoint is linear and kdata_p1 is
        already masked + normalized). Handles flat and JSSL-nested (sup/ssl) batches."""
        def _add(d):
            if isinstance(d, dict) and 'kdata_p1' in d and 'imdata_p1' not in d:
                d['imdata_p1'] = mri_adjoint_op(d['kdata_p1'], d['coil_sens'])
        if isinstance(batch, dict):
            if 'sup' in batch or 'ssl' in batch:
                _add(batch.get('sup')); _add(batch.get('ssl'))
            else:
                _add(batch)
        return batch

    def _net_forward(self, imdata, kdata, coil_sens, usrate_true):
        if imdata.shape[1] > 1:
            return torch.cat(
                [self.network(Variable(imdata[:, [v]], requires_grad=True),
                              kdata[:, [v]], coil_sens, usrate_true)
                 for v in range(imdata.shape[1])], dim=1)
        return self.network(Variable(imdata, requires_grad=True), kdata, coil_sens, usrate_true)

    @staticmethod
    def _norm01(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = x - x.min()
        d = x.max() - x.min()
        return x / (d if d > 0 else 1.0)

    def _collect_val_slab(self, recon, gt, batch, usrate):
        """Accumulate the mid-SPE strip of every readout slab for ONE fixed subject,
        so on_validation_epoch_end can stitch a full 2D image (not a single-slice strip).
        gt may be None (ssdu)."""
        if self.logger is None or not hasattr(self.logger, "experiment") or self.global_rank != 0:
            return
        subj = batch["subj"][0] if isinstance(batch["subj"], (list, tuple)) else batch["subj"]
        if self._vis_subj is None:
            self._vis_subj = subj
        if subj != self._vis_subj:
            return
        slice_start = int(batch["slice_start"][0]) if hasattr(batch["slice_start"], "__len__") else int(batch["slice_start"])
        pred = recon.detach().cpu().numpy()
        if pred.ndim == 6:
            pred = pred[0]                                  # (Nv, Nt, FE, PE, SPE)
        nv = int(np.clip(self.vis_nv_idx, 0, pred.shape[0] - 1))
        nt = int(np.clip(self.vis_nt_idx, 0, pred.shape[1] - 1))
        spe = pred.shape[4] // 2
        acc = self._vis_acc.setdefault(int(usrate), {"pred": [], "gt": []})
        acc["pred"].append((slice_start, np.abs(pred[nv, nt, :, :, spe])))   # (FE_slab, PE)
        if gt is not None:
            g = gt.detach().cpu().numpy()
            if g.ndim == 6:
                g = g[0]
            acc["gt"].append((slice_start, np.abs(g[nv, nt, :, :, spe])))

    def _log_eval_metrics(self, recon, gt, batch, usrate):
        """Leaderboard-style metrics within the segmask (reuses evaluation2026):
        magnitude SSIM/nRMSE always; velocity RelErr/AngErr when >=2 vencs are reconstructed.
        No-op unless a real segmask is present (held-out-train val via --explicit_val_dirs)."""
        if not _HAVE_EVAL_METRICS:
            return
        seg = batch.get("segmentation", None)
        if seg is None:
            return
        seg = seg.detach().cpu().numpy() if torch.is_tensor(seg) else np.asarray(seg)
        seg = seg[0]
        if seg.size <= 1:   # dummy/absent mask -> cannot score within ROI
            return
        seg = seg.astype(bool)                                                 # (SPE, PE, FE)
        if not seg.any():   # slab outside the segmentation ROI -> ROI metrics undefined, skip
            return
        pred = np.transpose(recon[0].detach().cpu().numpy(), (0, 1, 4, 3, 2))  # (Nv, Nt, SPE, PE, FE)
        gtn = np.transpose(gt[0].detach().cpu().numpy(), (0, 1, 4, 3, 2))
        kw = dict(on_step=False, on_epoch=True, add_dataloader_idx=False)
        ssim_v = float(SSIM(np.abs(pred), np.abs(gtn), seg))
        nrmse_v = float(nRMSE(np.abs(pred), np.abs(gtn), seg))
        self.log(f"val/SSIM_usrate{usrate}", ssim_v, **kw)
        self.log(f"val/nRMSE_usrate{usrate}", nrmse_v, **kw)
        self.log("val_SSIM_mean", ssim_v, **kw)    # epoch-mean over all R -- monitorable
        self.log("val_nRMSE_mean", nrmse_v, **kw)
        if pred.shape[0] >= 2:   # velocity needs the reference + flow encodings
            pc, gc = pred, gtn
            if _HAVE_BGC and self.options.get("val_bgc", True):
                try:
                    _st = np.random.get_state()       # execute_MSAC reseeds np.random globally
                    try:
                        corr = np.asarray(execute_MSAC(gtn, corr_fit_order=3, th=0.1)) * seg
                    finally:
                        np.random.set_state(_st)      # don't perturb training RNG
                    factor = np.exp(-1j * corr)
                    gc = gtn.copy(); gc[1:] = gc[1:] * factor
                    pc = pred.copy(); pc[1:] = pc[1:] * factor
                except Exception as e:
                    if not getattr(self, "_bgc_warned", False):
                        print(f"[val_bgc] MSAC correction failed ({e}); AngErr will not match the board")
                        self._bgc_warned = True
            _, fp = complex2magflow(pc)
            _, fg = complex2magflow(gc)
            rel_v = float(RelErr(fp, fg, seg)); ang_v = float(AngErr(fp, fg, seg))
            self.log(f"val/RelErr_usrate{usrate}", rel_v, **kw)
            self.log(f"val/AngErr_usrate{usrate}", ang_v, **kw)
            self.log("val_RelErr_mean", rel_v, **kw)
            self.log("val_AngErr_mean", ang_v, **kw)

    def _angerr_w(self):
        """Effective angerr_weight with optional epoch warmup (--angerr_warmup_start /
        --angerr_warmup_epochs): 0 before the start epoch, then linearly ramped to the full
        --angerr_weight. Defaults (0,0) => full weight from epoch 0 (unchanged). Only the
        angle loss is scheduled; L1 stays at 1 and SSIM stays constant."""
        w = float(self.options.get("angerr_weight", 0.0) or 0.0)
        if w <= 0:
            return 0.0
        start = int(self.options.get("angerr_warmup_start", 0) or 0)
        ramp = int(self.options.get("angerr_warmup_epochs", 0) or 0)
        e = self.current_epoch
        if e < start:
            return 0.0
        if ramp > 0 and e < start + ramp:
            return w * (e - start + 1) / ramp   # +1 so it is already nonzero on the first ramp epoch
        return w

    def _angerr_term(self, recon, gt, batch):
        """Differentiable angular-error supervision inside the segmask -- directly targets the
        leaderboard AngErr (angle between the 3-component velocity vectors). Velocity is the
        inter-encoding phase difference v_k = angle(x_k * conj(x_0)); we penalize 1 - cos(angle)
        between the pred and GT velocity vectors, magnitude-weighted by GT velocity (down-weights
        low-velocity voxels whose direction is noise -- this also stabilizes the phase gradient).

        Needs the reference + >=2 flow encodings in ONE recon, i.e. --flowvn_allvenc (all four
        encodings reconstructed per step); it is a no-op otherwise. No background correction is
        applied: the scorer applies the SAME GT-derived correction to gt and pred, so matching RAW
        velocity direction is exactly what minimizes the scored AngErr (and MSAC is
        non-differentiable anyway)."""
        if recon.shape[1] < 3:   # reference + >=2 flow encodings needed for a velocity vector
            if not getattr(self, "_angerr_warned", False):
                print(f"[angerr] recon has V={recon.shape[1]}; AngErr supervision needs all encodings "
                      f"(ref + >=2 flow encodings) -> no-op. Pass --flowvn_allvenc.")
                self._angerr_warned = True
            return recon.new_zeros((), dtype=torch.float32)
        seg = batch.get("segmentation", None)
        if seg is None or (torch.is_tensor(seg) and seg.numel() <= 1):
            return recon.new_zeros((), dtype=torch.float32)
        eps = 1e-6
        vp = torch.angle(recon[:, 1:] * recon[:, :1].conj())   # (B,K,T,D,H,W) pred velocity comps
        vg = torch.angle(gt[:, 1:]    * gt[:, :1].conj())      # (B,K,T,D,H,W) GT velocity comps
        dot = (vp * vg).sum(dim=1)                             # (B,T,D,H,W)
        ang = 1.0 - (dot / (vp.norm(dim=1) * vg.norm(dim=1) + eps)).clamp(-1.0, 1.0)  # 0..2
        m = seg.float().permute(0, 3, 2, 1).unsqueeze(1).expand_as(ang)   # (B,T,D,H,W) segmask
        w = m * vg.norm(dim=1).detach() if self.options.get("angerr_magweight", True) else m
        return (ang * w).sum() / w.sum().clamp_min(eps)

    def _vel_term(self, recon, gt, batch):
        """Complex-velocity supervision. Penalizes the per-component WRAPPED phase-difference
        velocity v_k = angle(x_k * conj(x_0)) between pred and GT inside the segmask. Unlike
        _angerr_term (direction only, 1-cos), this penalizes the velocity vector *error itself*
        -> drives down RelErr (velocity magnitude) AND AngErr (direction) together. The wrapped
        error is angle(pp * conj(pg)) so it is correct across the +-pi phase boundary. Needs the
        reference + >=1 flow encoding in ONE recon, i.e. --flowvn_allvenc; no-op otherwise."""
        if recon.shape[1] < 2:
            if not getattr(self, "_vel_warned", False):
                print(f"[vel] recon has V={recon.shape[1]}; --vel_weight needs all encodings "
                      f"(ref + >=1 flow encoding) -> no-op. Pass --flowvn_allvenc.", flush=True)
                self._vel_warned = True
            return recon.new_zeros((), dtype=torch.float32)
        seg = batch.get("segmentation", None)
        if seg is None or (torch.is_tensor(seg) and seg.numel() <= 1):
            return recon.new_zeros((), dtype=torch.float32)
        eps = 1e-6
        pp = recon[:, 1:] * recon[:, :1].conj()      # (B,K,T,D,H,W) complex velocity phasor (pred)
        pg = gt[:, 1:]    * gt[:, :1].conj()          # (B,K,T,D,H,W) complex velocity phasor (GT)
        dphi = torch.angle(pp * pg.conj())            # wrapped per-component velocity error in (-pi,pi]
        m = seg.float().permute(0, 3, 2, 1).unsqueeze(1).unsqueeze(1)   # (B,1,1,D,H,W) segmask
        w = m.expand_as(dphi)
        if self.options.get("angerr_magweight", True):
            w = w * torch.angle(pg).abs().detach()   # down-weight low-velocity (noisy-phase) components
        return (dphi.abs() * w).sum() / w.sum().clamp_min(eps)

    def _r_weight(self, batch):
        """Per-sample loss multiplier that upweights high-acceleration samples (--r_loss_pow):
        (usrate_true / 30)^pow, 30 = mid of the R=10-50 range. pow=0 => 1 (off). Concentrates
        gradient where velocity error is worst (high R). batch_size 1 => the batch is one sample."""
        p = float(self.options.get("r_loss_pow", 0.0) or 0.0)
        if p <= 0:
            return 1.0
        u = batch.get("usrate_true", None)
        if u is None:
            return 1.0
        r = float(u.reshape(-1).float().mean().item())
        return (max(r, 1.0) / 30.0) ** p

    def _ssim_term(self, recon, gt, batch):
        """1 - SSIM inside the segmask (matches the scorer), on magnitude.
        ssim_levels>1 -> multi-scale (MS-SSIM approximation: weighted sum over pooled scales)."""
        if self._ssim3d is None:
            return recon.new_zeros((), dtype=torch.float32)
        seg = batch.get("segmentation", None)
        if seg is None or (torch.is_tensor(seg) and seg.numel() <= 1):
            return recon.new_zeros((), dtype=torch.float32)
        B, V, T, D, H, W = recon.shape
        pred = recon.abs().reshape(B * V * T, 1, D, H, W)
        g = gt.abs().reshape(B * V * T, 1, D, H, W)
        m = seg.float().permute(0, 3, 2, 1)[:, None, None].expand(B, V, T, D, H, W).reshape(B * V * T, 1, D, H, W)
        gmax = (g * m).amax().clamp_min(1e-8).detach()
        pred = pred / gmax; g = g / gmax
        levels = max(1, int(self.options.get("ssim_levels", 1) or 1))
        wts = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:levels]
        wts = [x / sum(wts) for x in wts]
        acc = 0.0
        for lvl in range(levels):
            smap = self._ssim3d(pred * m, g * m)
            acc = acc + wts[lvl] * (smap * m).sum() / m.sum().clamp_min(1.0)
            if lvl < levels - 1:
                pred = F.avg_pool3d(pred, (1, 2, 2)); g = F.avg_pool3d(g, (1, 2, 2)); m = F.avg_pool3d(m, (1, 2, 2))
        return 1.0 - acc

    def _pix_loss(self, pred, gt):
        return self.L1_loss(pred - gt, torch.zeros_like(pred))

    def on_before_optimizer_step(self, optimizer, *args):
        """Log the PRE-clip total grad-norm once per optimizer step. If this sits above
        --grad_clip_val, the clip is throttling the effective LR (a common cause of a
        slow/plateauing loss) -> raise --grad_clip_val (keep --skip_loss_thresh for spikes)."""
        if self.global_rank != 0:
            return
        sq = 0.0
        for p in self.parameters():
            if p.grad is not None:
                sq = sq + p.grad.detach().norm(2) ** 2
        self.log("train/grad_norm", sq ** 0.5, on_step=True, on_epoch=False, prog_bar=False)

    def training_step(self, batch):
        _jssl_logs = {}
        _epoch_logs = {}
        if self.options["loss"] == "ssdu":
            kdata_p2 = batch["kdata_p2"]
            loss_mask = abs(kdata_p2[:, :, 0, :, 0, :, :]) != 0
            kp2 = torch.view_as_real(kdata_p2)
            d2 = torch.norm(kp2, p=2).clamp_min(1e-8)   # denominators are recon-independent:
            d1 = torch.norm(kp2, p=1).clamp_min(1e-8)   # compute once, shared across stages

            def _ssdu_consistency(recon):
                """Normalized L2+L1 k-space consistency of `recon` on the held-out p2 lines.
                Identical formula to the original single-output ssdu loss."""
                kp1 = torch.view_as_real(mri_forward_op(recon, batch["coil_sens"], loss_mask.float()))
                return 0.5 * torch.norm(kp2 - kp1, p=2) / d2 + 0.5 * torch.norm(kp2 - kp1, p=1) / d1

            recon_img_p1 = self._net_forward(
                batch["imdata_p1"], batch["kdata_p1"], batch["coil_sens"], batch["usrate_true"])
            loss = _ssdu_consistency(recon_img_p1)

        elif self.options["loss"] == "supervised":
            recon_img_p1 = self._net_forward(
                batch["imdata_p1"], batch["kdata_p1"],
                batch["coil_sens"], batch["usrate_true"],
            )
            loss = self._pix_loss(recon_img_p1, batch["gt"][:, 0])

            if (not torch.isfinite(loss)) or (loss.item() > 10):
                print(
                    "[LOSS SPIKE]",
                    loss.item(),
                    batch["norm"],
                    batch["case_dir"],
                    batch["slice_start"],
                    "seg",
                    batch["seg_idx"],
                    "usrate",
                    batch["usrate"],
                )
            if self.options.get("ssim_weight", 0) > 0:
                loss = loss + self.options["ssim_weight"] * self._ssim_term(recon_img_p1, batch["gt"][:, 0], batch)
            _aw = self._angerr_w()
            self.log("train/angerr_weight", _aw, on_step=False, on_epoch=True)   # visible warmup ramp
            if _aw > 0:
                loss = loss + _aw * self._angerr_term(recon_img_p1, batch["gt"][:, 0], batch)
            if self.options.get("vel_weight", 0) > 0:
                _vl = self._vel_term(recon_img_p1, batch["gt"][:, 0], batch)
                _epoch_logs["train/loss_vel"] = _vl
                loss = loss + self.options["vel_weight"] * _vl
            if self.options.get("ksp_weight", 0) > 0:
                loss_mask = abs(batch["kdata_p1"][:, :, 0, :, 0, :, :]) != 0
                k_pred = mri_forward_op(recon_img_p1, batch["coil_sens"], loss_mask.float())
                kr_meas = torch.view_as_real(batch["kdata_p1"])
                kr_pred = torch.view_as_real(k_pred)
                loss_ksp = (
                    0.5 * torch.norm(kr_meas - kr_pred, p=2) / torch.norm(kr_meas, p=2).clamp_min(1e-8)
                    + 0.5 * torch.norm(kr_meas - kr_pred, p=1) / torch.norm(kr_meas, p=1).clamp_min(1e-8)
                )
                _epoch_logs["train/ksp_loss"] = loss_ksp
                loss = loss + self.options["ksp_weight"] * loss_ksp
            loss = loss * self._r_weight(batch)   # high-R emphasis (--r_loss_pow); 1.0 when off

        elif self.options["loss"] == "jssl":
            sup, ssl = batch["sup"], batch["ssl"]
            recon_sup = self._net_forward(sup["imdata_p1"], sup["kdata_p1"], sup["coil_sens"], sup["usrate_true"])
            loss_sup = self._pix_loss(recon_sup, sup["gt"][:, 0])
            if self.options.get("ssim_weight", 0) > 0:
                loss_sup = loss_sup + self.options["ssim_weight"] * self._ssim_term(recon_sup, sup["gt"][:, 0], sup)
            _aw = self._angerr_w()
            self.log("train/angerr_weight", _aw, on_step=False, on_epoch=True)   # visible warmup ramp
            if _aw > 0:
                loss_sup = loss_sup + _aw * self._angerr_term(recon_sup, sup["gt"][:, 0], sup)
            if self.options.get("vel_weight", 0) > 0:
                _vl = self._vel_term(recon_sup, sup["gt"][:, 0], sup)
                _epoch_logs["train/loss_vel"] = _vl
                loss_sup = loss_sup + self.options["vel_weight"] * _vl
            if self.options.get("ksp_weight", 0) > 0:
                _lm = abs(sup["kdata_p1"][:, :, 0, :, 0, :, :]) != 0
                _kp = mri_forward_op(recon_sup, sup["coil_sens"], _lm.float())
                _km_r, _kp_r = torch.view_as_real(sup["kdata_p1"]), torch.view_as_real(_kp)
                loss_ksp = (
                    0.5 * torch.norm(_km_r - _kp_r, p=2) / torch.norm(_km_r, p=2).clamp_min(1e-8)
                    + 0.5 * torch.norm(_km_r - _kp_r, p=1) / torch.norm(_km_r, p=1).clamp_min(1e-8)
                )
                _epoch_logs["train/ksp_loss"] = loss_ksp
                loss_sup = loss_sup + self.options["ksp_weight"] * loss_ksp
            # High-R emphasis on the supervised stream only; the ssl stream carries its own R.
            loss_sup = loss_sup * self._r_weight(sup)
            v = int(np.random.randint(ssl["imdata_p1"].shape[1]))
            recon_ssl = self.network(Variable(ssl["imdata_p1"][:, [v]], requires_grad=True),
                                     ssl["kdata_p1"][:, [v]], ssl["coil_sens"], ssl["usrate_true"])
            kdata_p2 = ssl["kdata_p2"][:, [v]]
            loss_mask = abs(kdata_p2[:, :, 0, :, 0, :, :]) != 0
            kdata_p1 = mri_forward_op(recon_ssl, ssl["coil_sens"], loss_mask.float())
            loss_ssl = (0.5 * torch.norm(torch.view_as_real(kdata_p2) - torch.view_as_real(kdata_p1), p=2) / torch.norm(torch.view_as_real(kdata_p2), p=2).clamp_min(1e-8)
                        + 0.5 * torch.norm(torch.view_as_real(kdata_p2) - torch.view_as_real(kdata_p1), p=1) / torch.norm(torch.view_as_real(kdata_p2), p=1).clamp_min(1e-8))
            loss = loss_sup + loss_ssl
            _jssl_logs = {"train/loss_sup": loss_sup, "train/loss_ssl": loss_ssl}
            try:
                _r = int(round(float(ssl["usrate"].reshape(-1)[0]))) if torch.is_tensor(ssl["usrate"]) \
                     else int(ssl["usrate"])
                _epoch_logs[f"train/loss_ssl_usrate{_r}"] = loss_ssl
            except Exception:
                pass

        _thr = self.options.get("skip_loss_thresh", 0) or 0
        if (not torch.isfinite(loss)) or (_thr > 0 and loss.item() > _thr):
            def _attr(d):
                def g(k):
                    try:
                        v = d[k]
                        return v.tolist() if torch.is_tensor(v) else v
                    except Exception:
                        return "?"
                return f"case={g('case_dir')} usrate={g('usrate')} norm={g('norm')}"
            if isinstance(batch, dict) and "sup" in batch:
                _info = f"SUP[{_attr(batch['sup'])}] SSL[{_attr(batch['ssl'])}]"
            else:
                _info = _attr(batch)
            print(f"[SKIP STEP] loss={loss.item():.4g} (thresh={_thr}) {_info} -> batch skipped, no grad", flush=True)
            if self._weight_snapshot is not None:
                device = next(self.parameters()).device
                self.load_state_dict({k: v.to(device) for k, v in self._weight_snapshot.items()}, strict=False)
                print(f"[SKIP STEP] reverted to last clean snapshot", flush=True)
            return None

        for tag, val in _jssl_logs.items():
            self.log(tag, val, on_step=True, on_epoch=True)
        for tag, val in _epoch_logs.items():
            self.log(tag, val, on_step=False, on_epoch=True)
        self.log_dict(
            {"train_loss_epoch": loss, "step": self.current_epoch * 1.0},
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs is None:
            return   # skip fired this step — don't overwrite the snapshot we just restored
        _thr = self.options.get("skip_loss_thresh", 0) or 0
        if _thr > 0:   # only bother snapshotting when skip-revert is in use
            self._snap_steps += 1
            if self._snap_steps >= 20:
                self._weight_snapshot = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                self._snap_steps = 0

    def on_train_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        m = self.trainer.callback_metrics
        tl = m.get('train_loss_epoch', m.get('train/loss_epoch'))
        tl = float(tl) if tl is not None else float('nan')
        print(f"[epoch {self.current_epoch}] lr={lr:.3e} train_loss={tl:.4f}", flush=True)

    def on_validation_epoch_start(self):
        self._vis_acc = {}
        self._vis_subj = None

    def on_validation_epoch_end(self):
        if self.logger is not None and hasattr(self.logger, "experiment") and self.global_rank == 0:
            for usrate, acc in self._vis_acc.items():
                for tag, slabs in (("pred", acc["pred"]), ("gt", acc["gt"])):
                    if not slabs:
                        continue
                    D = slabs[0][1].shape[0]; PE = slabs[0][1].shape[1]
                    FE = max(st for st, _ in slabs) + D
                    full = np.zeros((FE, PE), np.float32); cnt = np.zeros((FE, 1), np.float32)
                    for st, strip in slabs:
                        full[st:st + D] += strip; cnt[st:st + D] += 1
                    full /= np.maximum(cnt, 1.0)
                    self.logger.experiment.add_image(
                        f"val_images/usrate{usrate}/{self._vis_subj}/{tag}",
                        torch.from_numpy(self._norm01(full)[None, :, :]), self.global_step)
        self._vis_acc = {}
        self._vis_subj = None

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        usrate = int(batch["usrate"][0]) if hasattr(batch["usrate"], "__len__") else int(batch["usrate"])

        if self.options.get('zero_filled', False):
            recon = batch["imdata_p1"]
        else:
            recon = self._test_recon(self.network, batch)

        if self.options['loss'] == 'ssdu' and 'kdata_p2' in batch:   # external val (no GT) -> self-supervised k-space consistency
            kdata_p2 = batch['kdata_p2']
            loss_mask = abs(kdata_p2[:, :, 0, :, 0, :, :]) != 0
            kdata_p1 = mri_forward_op(recon, batch['coil_sens'], loss_mask.float())
            kp2 = torch.view_as_real(kdata_p2)
            kp1 = torch.view_as_real(kdata_p1)
            loss = 0.5 * torch.norm(kp2 - kp1, p=2) / torch.norm(kp2, p=2).clamp_min(1e-8) + 0.5 * torch.norm(kp2 - kp1, p=1) / torch.norm(kp2, p=1).clamp_min(1e-8)
            self._collect_val_slab(recon * batch["norm"], None, batch, usrate)

        else:  # supervised
            gt = batch["gt"][:, 0]   # (B, V, T, D, H, W) -- drop the leading newaxis dim
            loss = self.L1_loss(recon - gt, torch.zeros_like(recon))
            self._collect_val_slab(recon * batch["norm"], gt * batch["norm"], batch, usrate)
            self._log_eval_metrics(recon, gt, batch, usrate)

        self.log(f"val/loss_usrate{usrate}", loss, on_step=False, on_epoch=True, add_dataloader_idx=False)
        # self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        self.log_dict({"val_loss_epoch": loss, "step": self.current_epoch * 1.}, on_step=True, on_epoch=True, prog_bar=True)
        return {"avg_val_loss": loss, "usrate": usrate}

    def _test_recon(self, network, batch):
        return torch.cat([network(batch['imdata_p1'][:, [v]], batch['kdata_p1'][:, [v]], batch['coil_sens'], batch["usrate_true"])
                          for v in range(batch['imdata_p1'].shape[1])], dim=1)

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        recon_img = self._test_recon(self.network, batch)

        recon_ms = None
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            recon_ms = float(start_event.elapsed_time(end_event))

        slab = (recon_img[0] * batch['norm']).cpu().numpy()  # (Nv, T, D, H, W)
        key = (str(batch['case_dir'][0]), int(batch['usrate'][0]))
        pack = self._test_cache.setdefault(key, {'slabs': [], 'ms': 0.0})
        pack['slabs'].append((int(batch['slice_start'][0]), slab))
        if recon_ms is not None:
            pack['ms'] += recon_ms
        return {}

    def on_test_epoch_end(self):
        val_dir = Path(self.options.get('val_dir', '') or '')
        if self.options.get('in_base_dir'):
            in_base = self.options['in_base_dir']
        else:
            _task = next((p for p in (val_dir, *val_dir.parents) if p.name.startswith('Task')), None)
            if _task is not None:
                in_base = str(_task.parent)
            else:
                in_base = str(val_dir.parents[2]) if len(val_dir.parents) >= 3 else str(val_dir)

        _ckpt = self.options.get('ckpt_path')
        _default_out = (Path(_ckpt).resolve().parent.parent if _ckpt else self.trainer.log_dir) or 'results/corrected_runs/test_out'
        out_base = Path(self.options.get('out_base_dir') or _default_out)
        D_size = self.options['D_size']
        for (case_dir, usrate), pack in self._test_cache.items():
            slabs = pack['slabs']
            Nx = max(ss for ss, _ in slabs) + D_size
            Nv, T, _, ky, kz = slabs[0][1].shape
            accum = np.zeros((Nv, T, Nx, ky, kz), dtype=np.complex64)
            count = np.zeros(Nx, dtype=np.float32)
            for ss, slab in slabs:
                accum[:, :, ss:ss+D_size] += slab
                count[ss:ss+D_size] += 1
            accum /= np.maximum(count, 1.0)[None, None, :, None, None]
            img = np.transpose(accum, (0, 1, 4, 3, 2))  # (Nv, Nt, SPE, PE, FE)
            seg = mat_to_npy_segmask(Path(case_dir) / 'segmask.mat')  # (SPE, PE, FE)
            img = img * seg[None, None].astype(np.float32)
            try:
                rel = os.path.relpath(case_dir, in_base)
            except ValueError:
                rel = Path(case_dir).name
            out_dir = out_base / rel                       # per-case dir mirrors the input layout (no overwrite)
            out_dir.mkdir(parents=True, exist_ok=True)
            _save_coo_npz(str(out_dir / ('img_ktGaussian%s.npz' % usrate)), img)
            with open(out_dir / ('recontime_ktGaussian%s.csv' % usrate), 'w', newline='') as fcsv:
                _w = csv.writer(fcsv); _w.writerow(['recontime']); _w.writerow([pack['ms'] / 1000.0])
            print('[test] saved', out_dir / ('img_ktGaussian%s.npz' % usrate), 'shape', img.shape)
        self._test_cache.clear()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.options["lr"])
        if self.options.get("lr_sched", "cosine") == "plateau":

            class _WarmupPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
                def __init__(self, optimizer, warmup_epochs, peak_lr, **kw):
                    self.warmup_epochs = int(warmup_epochs)
                    self.peak_lr = float(peak_lr)
                    self.warm_done = 0
                    super().__init__(optimizer, **kw)
                    if self.warmup_epochs > 0:               # set epoch-0 LR to the ramp bottom
                        self._set_lr(self.peak_lr / self.warmup_epochs)
                        self.warm_done = 1
                def _set_lr(self, lr):
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr
                def step(self, metrics=None, epoch=None):
                    if self.warm_done < self.warmup_epochs:  # still ramping: ignore the metric
                        self.warm_done += 1
                        self._set_lr(self.peak_lr * self.warm_done / self.warmup_epochs)
                        self.best, self.num_bad_epochs = self.mode_worse, 0
                        self.last_epoch += 1
                        return
                    super().step(metrics)                    # peak reached -> plateau takes over
            sch = _WarmupPlateau(
                opt, warmup_epochs=self.options.get("warmup_epochs", 10),
                peak_lr=self.options["lr"],
                mode=("max" if "SSIM" in self.options["monitor"] else "min"),
                factor=self.options.get("lr_factor", 0.5),
                patience=self.options.get("lr_patience", 3),
                threshold=self.options.get("lr_threshold", 1e-4), threshold_mode="rel",
                min_lr=self.options.get("lr_min", 0.0))
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sch, "monitor": self.options["monitor"],
                                     "interval": "epoch", "frequency": 1}}
        lr_sch = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
                    opt, warmup_epochs=self.options.get("warmup_epochs", 10),
                    max_epochs=self.options["epoch"])
        return {"optimizer": opt, "lr_scheduler": lr_sch}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network arguments')

    # Data IO
    parser.add_argument('--D_size',     type=int,  	default=1,	    			help='number of slices per volume (FlowVN only)')
    parser.add_argument('--T_size',     type=int,  	default=5,	    			help='number of cardiac bins per volume')
    parser.add_argument('--root_dir',   type=str,	default='',	help='training data root, e.g. $DATA/TaskR1R2/TrainSet/Aorta (scanned for subjects with params.csv)')
    parser.add_argument('--ckpt_path',	type=str,   default=None,				help='checkpoint to test, warm-start, or resume training from')
    parser.add_argument('--resume', action='store_true', help='train: full resume of --ckpt_path (weights + optimizer + epoch + LR schedule); without it, --ckpt_path is a weights-only warm start')
    parser.add_argument('--input',	    type=str,   default='', 				help='name of network input file')
    parser.add_argument('--val_dir',   type=str,	default='',	help='validation/test data root, e.g. $DATA/TaskR1R2/ValidationSet/Aorta')
    parser.add_argument('--explicit_train_dirs', type=str, nargs='+', default=None, help='train: explicit patient dirs (overrides the --root_dir scan); pair with --explicit_val_dirs for a stratified split')
    parser.add_argument('--train_undersampled_dirs', type=str, nargs='+', default=None, help='extra patient dirs whose ALREADY-undersampled kdata_ktGaussian{R} is used as-is (SSDU-split, no re-undersampling). Globs OK.')
    parser.add_argument('--explicit_val_dirs', type=str, nargs='+', default=None, help='val: held-out TRAIN patient dirs (with kdata_full) for real-GT validation')
    parser.add_argument('--ssl_dir', type=str, default=None, help='jssl (REQUIRED): reference-free source for the self-supervised stream. Point at TRAINING subjects held out from --root_dir/--explicit_train_dirs (must be disjoint; enforced). Subjects with kdata_full are undersampled here (k-t Gaussian, random R); subjects with kdata_ktGaussian{R} are used as acquired. NO LONGER defaults to --val_dir -- that silently trained the SSL stream on the official ValidationSet, which the rules forbid.')
    parser.add_argument('--mask_pool', type=int, default=0, help='dataloader speed: reuse a per-worker pool of N k-t Gaussian masks per (ky,kz,T,R) instead of generating a fresh one every sample. ktgaussian_mask rejection-samples in a Python loop per accepted point PER FRAME, so it is a real per-item cost -- and --loss jssl pays it twice per step (supervised + ssl streams both simulate). The SSDU split and all augmentations stay per-sample random; only the set of distinct acquisition masks becomes finite. 0 = off (fresh every sample, original). Try 16 if the loader is the bottleneck (nvidia-smi util well below 100%%).')
    parser.add_argument('--zero_filled', action='store_true', help='baseline floor: bypass the network in validation_step and score the zero-filled reconstruction (the adjoint A^H(k)=imdata_p1) through the SAME MSAC-corrected, seg-masked, per-R scorer used for the model. Run with --mode train and --explicit_val_dirs (held-out train subjects with real GT); main.py then runs a validation-only pass (no training, no checkpoint needed -- the network is never called) and prints val SSIM/nRMSE/RelErr/AngErr and their per-R variants. Use --loss supervised. This is the learning-free lower bound every reported number must beat.')
    parser.add_argument('--flowvn_allvenc', action='store_true', help='supervised stream: reconstruct all 4 velocity encodings per step (each with its own forward) so the velocity losses --angerr_weight/--vel_weight can be computed. The network stays per-encoding. The self-supervised stream stays single-encoding')
    parser.add_argument('--ssdu_rho', type=float, default=0.2, help='SSDU k-space partitioning ratio rho = |loss set|/|sampled set|: fraction of acquired points held out as the loss (p2) split; the rest become the training (p1) input. 0.2 = original default.')
    parser.add_argument('--seed', type=int, default=None, help='global seed for reproducibility; unset = fresh random run (PL logs the chosen seed)')
    parser.add_argument('--ssim_weight', type=float, default=0.0, help='supervised loss: weight of (1 - segmask SSIM) added to L1; 0 = pure L1')
    parser.add_argument('--ksp_weight', type=float, default=0.0, help='weight of the dual-domain k-space term L_ksp = d(A x, k) on the ACQUIRED lines (normalized L2+L1, same form as the SSDU loss). 0 = off')
    parser.add_argument('--ssim_levels', type=int, default=1, help='SSIM scales: 1 = single-scale; >1 = MS-SSIM approximation')
    parser.add_argument('--augment', action='store_true', help='train-time augmentation of the fully-sampled source (readout flip + global phase); off by default so baselines are clean')
    parser.add_argument('--aug_noise', type=float, default=0.0, help='input-only complex k-space noise: per-sample sigma ~ U(0, aug_noise) x sampled-kspace RMS (clean GT). ~0.02-0.08 for SNR/field-strength (S1) robustness; 0 = off')
    parser.add_argument('--aug_phase_ramp', type=float, default=0.0, help='#12 vendor aug: B0/eddy-current phase-ramp on the SOURCE (requires --augment). Per-sample linear image-domain phase ramp along the readout (FE) of up to +/- this many cycles across the FE extent, applied to all vencs (magnitude- & velocity-invariant; GT inherits it). ~0.5-2.0 mimics GE/Siemens/Philips phase differences; 0 = off')
    parser.add_argument('--aug_coil', type=float, default=0.0, help='coil-map (CSM) perturbation aug: per-sample sigma ~ U(0, aug_coil) multiplicative complex jitter on coil_sens (GT uses true maps). ~0.02-0.1 for cross-scanner CSM robustness (#3); 0 = off')
    parser.add_argument('--monitor', type=str, default='val_loss_epoch', help='checkpoint metric: val_loss_epoch | val_nRMSE_mean | val_SSIM_mean (eval metrics need --explicit_val_dirs + supervised)')
    parser.add_argument('--save_top_k', type=int, default=-1, help='checkpoints to keep ranked by --monitor (-1 = keep all)')
    parser.add_argument('--ckpt_every_min', type=float, default=20.0, help='rolling MID-epoch checkpoint every N minutes (node-crash insurance; epoch-end saves alone lose the whole in-progress epoch). Keeps only the newest one (<version>-rolling-*.ckpt, previous is deleted). Resume with --ckpt_path as usual: weights/optimizer/LR state restored, the interrupted epoch restarts from its beginning. 0 = off')
    parser.add_argument('--in_base_dir', type=str, default=None, help='test: base input dir; case path relative to this is mirrored under --out_base_dir')
    parser.add_argument('--out_base_dir', type=str, default=None, help='test: submission output root; each case is written to <out_base_dir>/<rel>/img_ktGaussian{R}.npz (defaults to the run log dir)')

    # General configuration 
    parser.add_argument('--skip_loss_thresh',   type=float, default=0.0,       help='training: skip a batch (no gradient) when its loss is non-finite or exceeds this value (PaLM/OPT-style spike guard). 0 = disabled. Typical normal loss ~0.5-1, so e.g. 5 catches spikes without dropping legit batches.')
    parser.add_argument('--no_usrate_modulation', action='store_true',        help="FlowVN: DISABLE acceleration conditioning, i.e. restore the original network's fixed scalar lamb_ru/lamb_du instead of learning them as spline functions of the true R over [9,51]. On by default (it is a proposed technique). Pass this to ablate it: it is the ONLY difference between the two runs, so the pair isolates acceleration conditioning.")
    parser.add_argument('--refine_csm',         action='store_true',          help='learnable in-unroll refinement of the precomputed coil maps (cross-scanner robustness); zero-initialized, so a fresh run starts from the unrefined maps')
    parser.add_argument('--baseline',           action='store_true',          help='use the paper-faithful FlowVN without our changes (networks/flowvn_baseline.py) for the baseline comparison; ignores --refine_csm, --act linear_flowvn and acceleration conditioning')
    parser.add_argument('--robust_norm',        action='store_true',          help='dataloader: winsorized-RMS k-space normalization (clip top 0.5pct magnitudes) instead of plain RMS -> robust to vendor hot samples for cross-center generalization; off by default')
    parser.add_argument('--scanner_reps',        type=str, default=None,       help="dataloader: scanner-balanced oversampling, e.g. 'UIH_30T_uMR890:4,Philips_30T_Ingenia:2,Philips_15T_Ambition:2'. Each 'scanner_tag:rep' duplicates that scanner's subject groups rep times per epoch (tag = the immediate parent dir of the patient dir). Counters the GE-majority bias so the net learns scanner-invariant recon (helps Special Task 1 / new-site aorta). Duplicates still reshuffle their own slices each epoch. Unlisted scanners = rep 1. None = off. Only affects the SubjectGroupedSampler train order; val/test untouched. Warning: you have few minority subjects (e.g. 6 UIH) so heavy reps overfit them -- a sqrt-frequency rule (UIH~4, Philips~2, GE~1) is the safe middle.")
    parser.add_argument('--network',   		  	type=str,  	default='FlowVN',    	  help='network to use (FlowVN)')
    parser.add_argument('--num_stages',       	type=int,  	default=10,    	  help='number of stages in the network')
    parser.add_argument('--features_in',      	type=int,  	default=1,    	  help='number of input dimensions')
    parser.add_argument('--features_out',    	type=int,  	default=24,    	  help='number of filters for convolutional kernel')

    # FlowVN specific configuration
    parser.add_argument('--kernel_size',       	type=int,  	default=7,    	  help='xyz kernel size')
    parser.add_argument('--act',            	type=str,  	default='linear', choices=['rbf', 'linear', 'linear_flowvn'], help='activation: rbf (slow, bounded), linear (fast but unbounded -> identity passthrough outside grid = more unstable), linear_flowvn (FlowVN adaptive interpolator: bounded/saturating like rbf but at linear speed; recommended for stability)')
    parser.add_argument('--num_act_weights',	type=int,  	default=71,    	  help='number of basis functions / interpolation knots for activation')
    parser.add_argument('--vmax_dc',        	type=float,	default=7.0,   	  help='linear_flowvn: response range (+/-vmax_dc) of the data-consistency activation; wider than --vmax since k-space residuals have larger dynamic range')
    parser.add_argument('--stddev_init',    	type=float,	default=0.05,  	  help='linear_flowvn: stddev of the truncated-normal init for the interpolator knots (near-zero init = near-identity regularizer at start)')
    parser.add_argument('--grid',    	        type=float, default=0.25,	  help='grid size for linear act')
    parser.add_argument('--weight',           	type=float, default=0.025,	  help='scale weights for RBF kernel')
    parser.add_argument('--vmin',    		  	type=float,	default=-3.5,	  help='min value of filter response for rbf activation')
    parser.add_argument('--vmax',    		  	type=float, default=3.5, 	  help='max value of filter response for rbf activation')
    parser.add_argument('--sgd_momentum',	    type=bool, 	default=True,	  help='heavy-ball momentum in the unrolled cascade. Note type=bool, so `--sgd_momentum False` still evaluates to True; use --no_sgd_momentum to disable')
    parser.add_argument('--no_sgd_momentum',    action='store_true',          help='disable the unrolled-cascade momentum (flowvn.py bounds its coefficient to (0,1) with a sigmoid; flowvn_baseline.py uses the raw coefficient)')
    parser.add_argument('--grad_clip_val', type=float, default=1.0, help='gradient clipping value (was hardcoded 1.0). If train/grad_norm routinely exceeds this, the clip is throttling the effective LR and slowing convergence -> raise it (e.g. 5) and keep --skip_loss_thresh for true spikes.')

    # Training and Testing Configuration
    parser.add_argument('--mode',           		type=str,   default='train',	help='train or test')
    parser.add_argument('--lr',             		type=float, default=1e-4,     	help='peak learning rate (reached at the end of warmup, then cosine-decayed)')
    parser.add_argument('--epoch',          		type=int,   default=100,      	help='number of training epoch')
    parser.add_argument('--warmup_epochs',  		type=int,   default=0,       	help='epochs to linearly ramp LR from ~0 to peak --lr (both schedules). Longer = gentler ramp (helps avoid mid-warmup divergence at higher peak LR). Set 0 to start hot.')
    parser.add_argument('--lr_sched',       		type=str,   default='cosine', choices=['cosine','plateau'], help='LR schedule after warmup: cosine = CosineAnnealing on a fixed clock (needs a correct --epoch; distorted by resumes that change the budget). plateau = ReduceLROnPlateau, reactive: halves LR after --lr_patience stalled epochs on --monitor; robust to resumes / unknown epoch count.')
    parser.add_argument('--lr_patience',    		type=int,   default=3,        	help='plateau only: epochs with no --monitor improvement before LR *= --lr_factor (counted ONLY after warmup). Your epochs are long, so keep this small (2-3), not the paper\'s 5.')
    parser.add_argument('--lr_factor',      		type=float, default=0.5,      	help='plateau only: multiplicative LR cut per trigger')
    parser.add_argument('--lr_min',         		type=float, default=0.0,      	help='plateau only: floor the LR can decay to via ReduceLROnPlateau (min_lr). 0 = no floor. e.g. 1e-5 keeps it training after repeated cuts.')
    parser.add_argument('--lr_threshold',   		type=float, default=1e-4,     	help='plateau only: relative improvement (threshold_mode=rel) needed to count as progress; below it for --lr_patience epochs = plateau -> LR cut. Default 1e-4=0.01%% (lenient). ssdu ~1e-3 (0.1%%), supervised ~2e-3-3e-3 (faster descent).')
    parser.add_argument('--batch_size',     		type=int,   default=1,        	help='batch size')
    parser.add_argument('--loss',       		    type=str,   default='',       	help='training regime: supervised, ssdu, or jssl (joint supervised + self-supervised)')
    parser.add_argument('--vel_weight', type=float, default=0.0, help='weight of the wrapped per-component velocity loss L_vel (phase difference v_k=angle(x_k*conj(x_0)) vs GT, segmask + GT-velocity-magnitude weighted). Needs --flowvn_allvenc. 0 = off')
    parser.add_argument('--r_loss_pow', type=float, default=0.0, help='supervised/jssl: upweight high-acceleration samples by (usrate_true/30)^pow so the optimizer focuses where velocity error is worst (high R). 0 = uniform (off). Try 1-2. R=50 gets (50/30)^pow, R=10 gets (10/30)^pow.')
    parser.add_argument('--num_workers',       		type=int,   default=4,       	help='number of workers')
    parser.add_argument('--gpu',                    type=int,   default=0,           help='single-GPU device index, e.g. --gpu 3. Ignored if --gpus is set.')
    parser.add_argument('--gpus',                   type=str,   default=None,        help='multi-GPU DDP: comma-separated device indices, e.g. --gpus 0,1,2,3. ~N x faster with FULL data coverage and identical math/quality (sampler shards subjects disjointly per rank). NOTE: effective batch = batch_size * accumulate_grad_batches * N_gpus, so 4 GPUs => 4x the effective batch; if optimization changes, scale --lr by ~N (or sqrt(N)) and/or lower --accumulate_grad_batches. Overrides --gpu.')
    parser.add_argument('--accumulate_grad_batches', type=int, default=16, help='gradient accumulation steps. Effective batch = batch_size * this * N_gpus. For a clean DDP speedup with IDENTICAL optimization, divide this by the GPU count (e.g. 4 GPUs -> 4) so effective batch stays 16.')
    parser.add_argument('--val_num_workers', type=int, default=2, help='val loader workers; kept low because the per-subject cache holds a full volume per worker')
    parser.add_argument('--val_check_interval',       		type=float,   default=1.0,       	help='val check interval')
    parser.add_argument('--limit_train_batches', type=float, default=1.0, help='cap samples per epoch. Each subject contributes ~Nx (~117) near-adjacent readout slabs, so a full epoch is num_subjects*Nx (~14.6k) and very slow. A value >1 = exact sample count per epoch (e.g. 1500); a value in (0,1] = fraction. The subject sampler reshuffles every epoch, so capping still covers the whole dataset over several epochs while keeping the per-subject .mat cache efficient. 1.0 = no cap (old behavior).')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='run validation (and the GT metrics) every N epochs instead of every epoch. Pure speedup: nothing training-critical depends on val (checkpoint + plateau LR both monitor train_loss_epoch). With a small --limit_train_batches, epochs are short, so e.g. 3-5 keeps val overhead low.')
    parser.add_argument('--val_bgc', type=lambda s: str(s).lower() not in ('0','false','no'), default=True, help='apply the scorer\'s MSAC background-phase correction (from GT, to both gt+pred) before val RelErr/AngErr, so val_AngErr matches the leaderboard. Deterministic (fixed seed) but adds a per-subject MSAC fit to validation; set --val_bgc 0 to skip if val is too slow (then val_AngErr is on uncorrected phase and will read higher than the board).')
    parser.add_argument('--angerr_weight', type=float, default=0.0, help='weight of the differentiable angular-error loss L_ang (1-cos between predicted and GT velocity vectors, in the segmask). Needs --flowvn_allvenc. 0 = off')
    parser.add_argument('--angerr_magweight', type=lambda s: str(s).lower() not in ('0','false','no'), default=True, help='weight the angular-error loss by GT velocity magnitude (down-weights low-velocity noise voxels, stabilizes the phase gradient). Default on; --angerr_magweight 0 for an unweighted mean matching the raw metric.')
    parser.add_argument('--angerr_warmup_start', type=int, default=0, help='epoch at which the angular-error loss begins fading in (weight = 0 before it). Phase/velocity direction is unreliable until the magnitude recon is decent, so a garbage early AngErr gradient just hurts -- delay it. 0 = no delay (full weight from epoch 0, unchanged). Try ~5.')
    parser.add_argument('--angerr_warmup_epochs', type=int, default=0, help='epochs over which --angerr_weight is linearly ramped 0 -> full, starting at --angerr_warmup_start. 0 = step on at full weight (no ramp). Only the angerr loss is warmed up; L1 stays at weight 1 and SSIM stays constant. Try ~5.')
    parser.add_argument('--precision', type=str, default='32', choices=['32', '16', 'bf16'], help='Trainer precision. 32 = fp32 (current/default, unchanged). Mixed precision reduces PER-SAMPLE compute (~1.3-1.7x faster) with FULL data coverage and no fewer samples -> genuine wall-clock saving when COMPUTE-bound (check nvidia-smi util ~100%%). bf16 preferred over 16 for this complex-valued net (fp32 range, no loss-scaling). CAVEAT: autocast handles complex (modReLU/FFT/complex-conv) unevenly -> VERIFY heldout SSIM/nRMSE match fp32 on a short run before a long job.')


    argsorg = parser.parse_args()
    
    args = vars(argsorg)
    args['_num_devices'] = 1
    if args.get('no_sgd_momentum'):
        args['sgd_momentum'] = False
    for _key in ('explicit_val_dirs', 'explicit_train_dirs', 'train_undersampled_dirs'):
        if args.get(_key):
            expanded = []
            for p in args[_key]:
                matches = sorted(glob(p))
                expanded.extend(matches if matches else [p])
            args[_key] = expanded
    pl.seed_everything(args.get('seed'), workers=True)   # None -> random+logged; set --seed to reproduce

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True)

    _resume_version = None
    if args['mode'] == 'train' and args.get('resume') and args.get('ckpt_path'):
        _vname = Path(args['ckpt_path']).resolve().parent.parent.name   # .../version_N/checkpoints/x.ckpt
        _m = re.match(r'version_(\d+)$', _vname)
        if _m:
            _resume_version = int(_m.group(1))
    logger = (TensorBoardLogger("./results/corrected_runs/lightning_logs", name="", version=_resume_version)
              if args['mode'] == 'train' else None)

    if args['mode'] == 'train':
        print_options(parser,argsorg,logger)

    _callbacks = []
    if args['mode'] == 'train':   # checkpointing (and the logger it needs) only exist for training
        _mon = args['monitor']
        _mode = 'max' if 'SSIM' in _mon else 'min'   # SSIM higher is better; loss/nRMSE lower is better
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            mode = _mode,
            monitor = _mon,
            save_top_k=args['save_top_k'],
            every_n_epochs=1,
            save_weights_only=False,
            save_last=True,                       # always keep last.ckpt for easy --resume
            filename=str(logger.version)+'-{epoch}'+'-{'+_mon+':.5f}'
            )

        _callbacks = [checkpoint_callback, pl.callbacks.LearningRateMonitor(logging_interval='epoch')]
        if args.get('ckpt_every_min', 0) and args['ckpt_every_min'] > 0:

            _callbacks.append(pl.callbacks.ModelCheckpoint(
                monitor=None,
                save_top_k=1,
                train_time_interval=timedelta(minutes=args['ckpt_every_min']),
                save_weights_only=False,             
                save_last=False,                     
                filename=str(logger.version)+'-rolling-{epoch}-{step}'
                ))

    _precision = 'bf16' if args['precision'] == 'bf16' else int(args['precision'])   # PL 1.9: 'bf16' str or 16/32 int
    if args.get('gpus'):
        _devices = [int(g) for g in str(args['gpus']).split(',') if g.strip() != '']
    else:
        _devices = [args['gpu']]
    _strategy = 'ddp' if len(_devices) > 1 else None   # PL 1.9: None = single-device default ('auto' is 2.0+)
    trainer = pl.Trainer(default_root_dir = "results/corrected_runs/lightning_logs/",
                        accelerator=('gpu' if torch.cuda.is_available() else 'cpu'),
                        devices=(_devices if torch.cuda.is_available() else 1),
                        strategy=_strategy,


                        replace_sampler_ddp=False,
                        max_epochs=args['epoch'],
                        precision=_precision,
                        accumulate_grad_batches=args['accumulate_grad_batches'],
                        logger=logger, gradient_clip_val=args['grad_clip_val'], callbacks=_callbacks, num_sanity_val_steps=0,
                        log_every_n_steps=5, val_check_interval = args["val_check_interval"],
                        # >1 -> exact sample count per epoch (int); <=1 -> fraction. Caps the ~14.6k-slab epoch.
                        limit_train_batches=(int(args['limit_train_batches']) if args['limit_train_batches'] > 1
                                             else args['limit_train_batches']),
                        check_val_every_n_epoch=args['check_val_every_n_epoch'])

    if args['mode'] == 'train':
        if args['loss'] == 'jssl':
            if not args.get('ssl_dir'):
                raise ValueError(
                    "--loss jssl requires an explicit --ssl_dir: the reference-free source for "
                    "the self-supervised stream. Point it at TRAINING subjects held out from "
                    "--root_dir / --explicit_train_dirs (e.g. one or more whole centers). It no "
                    "longer defaults to --val_dir (the official ValidationSet): training on that "
                    "violates the challenge rule that methods be developed solely on the training set."
                )

            ssl_root = Path(args['ssl_dir'])
            ssl_pats = sorted({p.parent for p in ssl_root.rglob('params.csv')})
            if not ssl_pats:
                raise ValueError(f"--ssl_dir {ssl_root}: no subjects found (no params.csv beneath it).")
            ssl_full = [str(p) for p in ssl_pats if (p / args['input']).is_file()]
            ssl_us = [str(p) for p in ssl_pats
                      if not (p / args['input']).is_file() and any(p.glob('kdata_ktGaussian[0-9]*.mat'))]
            _unusable = len(ssl_pats) - len(ssl_full) - len(ssl_us)
            if _unusable:
                raise ValueError(f"--ssl_dir {ssl_root}: {_unusable} subject(s) have neither "
                                 f"{args['input']} nor kdata_ktGaussian*.mat")
            print(f"[jssl] ssl stream <- {ssl_root}: {len(ssl_full)} subject(s) undersampled here, "
                  f"{len(ssl_us)} already undersampled", flush=True)

            sup_ds = OwnDataset(**dict(args, loss='supervised', mode='train'))

            ssl_ds = OwnDataset(**dict(args, loss='ssdu', mode='train',
                                       explicit_train_dirs=ssl_full,
                                       train_undersampled_dirs=ssl_us,
                                       augment=False, aug_noise=0.0, aug_coil=0.0,
                                       aug_phase_ramp=0.0))

            _key = lambda p: tuple(Path(p).parts[-3:])
            _sup_subj = {_key(e[0]) for e in sup_ds.filename}
            _both = _sup_subj & {_key(e[0]) for e in ssl_ds.filename}
            if _both:
                raise ValueError(
                    f"{len(_both)} subject(s) feed BOTH the supervised and self-supervised streams "
                    f"(e.g. {sorted(_both)[:3]}). The SSL stream stands in for subjects whose "
                    f"references we do not have; if the supervised stream already carries them the "
                    f"setup is self-defeating. --ssl_dir must be disjoint from --root_dir / "
                    f"--explicit_train_dirs (note Aorta_train/Aorta_heldout copy the same subjects)."
                )

            dataset = JSSLDataset(sup_ds, ssl_ds)
            val_dataset = OwnDataset(**dict(args, loss='supervised', mode='val'))
        else:
            dataset = OwnDataset(**args)
            args_val = vars(parser.parse_args())
            args_val["mode"] = "val"
            val_dataset = OwnDataset(**args_val)

        _pw = args['num_workers'] > 0   
        
        _scanner_reps = None
        _sr = args.get('scanner_reps')
        if _sr:
            _scanner_reps = {}
            for _tok in str(_sr).split(','):
                _tok = _tok.strip()
                if not _tok:
                    continue
                _k, _v = _tok.rsplit(':', 1)
                _scanner_reps[_k.strip()] = int(_v)
            print(f"[scanner_reps] oversampling {_scanner_reps}", flush=True)
        if hasattr(dataset, 'subject_groups'):
            train_sampler = SubjectGroupedSampler(dataset, shuffle=True, seed=args.get('seed'),
                                                  scanner_reps=_scanner_reps)
            dataloader = DataLoader(dataset, batch_size=args['batch_size'], num_workers=args['num_workers'],
                                    pin_memory=True, persistent_workers=_pw, sampler=train_sampler)
        else:
            dataloader = DataLoader(dataset, batch_size=args['batch_size'], num_workers=args['num_workers'],
                                    pin_memory=True, persistent_workers=_pw, shuffle=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args['batch_size'], num_workers=args['val_num_workers'], pin_memory=False, persistent_workers=args['val_num_workers'] > 0)

        fit_ckpt = None
        if args['ckpt_path'] is not None and args.get('resume'):
            # full resume: Lightning restores weights + optimizer + epoch + LR schedule from the ckpt
            model = UnrolledNetwork(**args)
            fit_ckpt = args['ckpt_path']
            print(f"[resume] continuing training from {args['ckpt_path']}")
        elif args['ckpt_path'] is not None:
            # weights-only warm start (fresh optimizer / epoch / schedule)
            model = UnrolledNetwork.load_from_checkpoint(args["ckpt_path"], **args)
        else:
            model = UnrolledNetwork(**args)

        if args.get('zero_filled'):
            print("[zero_filled] validation-only pass; network bypassed, scoring A^H(k)", flush=True)
            trainer.validate(model, dataloaders=dataloader_val)
        else:
            trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader_val, ckpt_path=fit_ckpt)

    elif args['mode'] == 'test':
        dataset = OwnDataset(**args)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=args['num_workers'], pin_memory=False)
        
        model = UnrolledNetwork.load_from_checkpoint(args['ckpt_path'], **args)
        trainer.test(model, dataloaders=dataloader)
        
