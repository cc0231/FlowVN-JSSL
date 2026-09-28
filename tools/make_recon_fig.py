#!/usr/bin/env python
"""
Qualitative reconstruction figure for the CMRx4DFlow / STACOM paper (Fig. 2).

LAYOUT  2 rows x 5 columns
    rows : magnitude | velocity component v_j
    cols : reference | baseline | proposed | |baseline-ref| | |proposed-ref|

The two error columns share a colour scale and an error gain, so the baseline and
proposed error maps are directly comparable by eye -- that side-by-side is what
carries the argument; the reconstructions themselves all look broadly plausible.

INPUT
-----
Complex volumes in the challenge scorer layout  (Nv, Nt, SPE, PE, FE), Nv = 4
(one flow-compensated reference encoding + three directional encodings), plus a
boolean segmentation mask (SPE, PE, FE), as written by board_eval.py --dump_npz.

USAGE
-----
  python tools/make_recon_fig.py \
      --ref_npz  gt.npz --base_npz baseline.npz --prop_npz proposed.npz \
      --seg_npz  seg.npz \
      --frame    systole \
      --out      paper/fig_recon.pdf

  # let it pick the frame with the highest in-mask velocity (usually systole):
  python tools/make_recon_fig.py ... --frame auto

MULTI-R SWEEP  (rows {magnitude, velocity} x columns {Reference, R=10..50})
--------------
  # one command, straight from a checkpoint (board-faithful; matches Table 1):
  python tools/make_recon_fig.py --multi_r \
      --prop_ckpt <version_N>/checkpoints/best.ckpt \
      --prop_train_opt <version_N>/train_opt.txt \
      --subj_dir '$DATA/TaskR1R2/TrainSet/Aorta_heldout/Center012/Philips_15T_Ambition/P009' \
      --accels 10,20,30,40,50 --gpu 1 --out paper/fig_multir.pdf

  # or from an existing board_eval.py --dump_npz directory:
  python tools/make_recon_fig.py --multi_r --npz_dir results/npz --out paper/fig_multir.pdf
"""
import argparse
import glob
import os
import re
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- challenge evaluation helpers --------------------------------------------------------
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (this file lives in tools/)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_EVAL_DIR = os.path.join(_HERE, "cmr_helper_scripts", "evaluation2026")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
try:
    from utils_flow import complex2magflow
    from utils_metrics import SSIM, nRMSE, RelErr, AngErr
    _HAVE_EVAL = True
except Exception as _e:                                   # figure still renders without them
    print(f"[warn] challenge eval helpers unavailable ({_e}); panel annotations disabled")
    _HAVE_EVAL = False


def _load(path):
    a = np.load(path)
    return a[a.files[0]] if hasattr(a, "files") else a


def apply_view(view, gt, seg, rows):
    """Permute the SPATIAL axes so the chosen plane is the first two (displayed) axes and
    the remaining axis becomes the slice index pick_slice runs along.

    Volumes are (Nv, Nt, SPE, PE, FE); seg is (SPE, PE, FE). All drawing code slices the
    LAST spatial axis and shows the first two, so the display plane is chosen purely by
    which axis we move to the end:
      * 'spe_pe' (default) -> unchanged: the 24x104 phase-encoded plane, FE as slice axis.
      * 'pe_fe'            -> move SPE to the end: the ~104x128 PE-readout plane (squarer,
                             larger panels), SPE as slice axis.
    The SAME permutation is applied to gt/seg/preds, so every whole-volume in-mask metric
    is invariant (SSIM's cubic 11^3 window included) -- only the cross-section changes.
    """
    if view == "spe_pe":
        return gt, seg, rows
    if view != "pe_fe":
        raise ValueError(f"unknown --view {view!r}")
    gt = np.moveaxis(gt, 2, -1)                       # (Nv, Nt, SPE, PE, FE) -> (Nv, Nt, PE, FE, SPE)
    seg = np.moveaxis(np.asarray(seg), 0, -1)         # (SPE, PE, FE) -> (PE, FE, SPE)
    rows = [(lab, np.moveaxis(x, 2, -1)) for lab, x in rows]
    return gt, seg, rows


def velocity(x):
    """v_j = angle(x_j * conj(x_0)) -> (3, Nt, SPE, PE, FE). Matches eq. (2) in the paper."""
    if _HAVE_EVAL:
        _, flow = complex2magflow(x)
        return np.asarray(flow)
    return np.angle(x[1:] * np.conj(x[:1]))


def magnitude(x):
    """Anatomical image = magnitude of the flow-compensated reference encoding."""
    return np.abs(x[0])


def load_venc(subj_dir):
    """Per-direction VENC for a subject, from the challenge params CSV in its directory.

    VENC is a LIST (one value per flow encoding, with venc_order naming the directions),
    not a scalar -- usually isotropic, but the format does not require it. Returns a
    length-3 array, or None if no CSV with a VENC field is found (then everything stays
    in radians, which is exact: +-pi IS +-venc).
    """
    if not subj_dir:
        return None
    try:
        from utils_datasl import read_params_csv
    except Exception as e:
        print(f"[venc] utils_datasl unavailable ({e}); using radians")
        return None
    for p in sorted(glob.glob(os.path.join(subj_dir, "*.csv"))):
        try:
            params = read_params_csv(p)
        except Exception:
            continue
        if params and params.get("VENC"):
            v = np.asarray(params["VENC"], float).ravel()
            print(f"[venc] {os.path.basename(p)}: {v} order={params.get('venc_order')}")
            return v
    print(f"[venc] no params CSV with a VENC field in {subj_dir}; using radians")
    return None


def venc_from_dir(npz_dir, subj=None):
    """Auto-load the true VENC stamped into a board_eval dump (<tag>_venc.npy), or None.
    Lets make_recon_fig scale velocity to cm/s with no manual --venc / --subj_dir."""
    if not npz_dir:
        return None
    pats = sorted(glob.glob(os.path.join(npz_dir, "*_venc.npy")))
    if subj is not None:
        pats = [p for p in pats if os.path.basename(p).startswith(str(subj))] or pats
    if pats:
        v = np.asarray(np.load(pats[0]), float).ravel()
        print(f"[venc] auto-loaded from dump {os.path.basename(pats[0])} = {v}")
        return v
    return None


def pick_frame(v_ref, seg, mode):
    """'auto' -> frame with the largest mean in-mask speed (peak systole in practice)."""
    if mode != "auto":
        return int(mode)
    speed = np.linalg.norm(v_ref, axis=0)                 # (Nt, SPE, PE, FE)
    m = np.broadcast_to(seg, speed.shape[1:])
    per_frame = [speed[t][m].mean() for t in range(speed.shape[0])]
    t = int(np.argmax(per_frame))
    print(f"[frame] auto-selected t={t} (mean in-mask speed {per_frame[t]:.4f})")
    return t


def pick_slice(seg, mode):
    """'auto' -> the readout (FE) slice carrying the most segmented voxels.

    Scorer layout is (SPE, PE, FE) = (W, H, D) after the transpose in reconstruct().
    The anatomical image plane is SPE x PE -- the undersampled phase-encoded plane
    (k_y, k_z) -- and FE is the fully-sampled readout, only D_size slices deep
    (7 here). So the slice index runs along FE. Slicing along SPE instead yields a
    PE x readout sliver a handful of voxels wide, which is not an image.
    """
    if mode != "auto":
        return int(mode)
    counts = seg.sum(axis=(0, 1))
    f = int(np.argmax(counts))
    print(f"[slice] auto-selected FE={f} of {seg.shape[2]} ({counts[f]} in-mask voxels)")
    return f


def crop_box(seg2d, margin=24, full_fov=False):
    """Bounding box around the mask + margin.

    A tight crop is right for a final paper panel but useless for inspection: with a
    small margin you see the vessel lumen and a rim, no surrounding anatomy, and no
    way to tell what you are looking at. Default margin is therefore generous; pass
    full_fov=True to disable cropping entirely.
    """
    if full_fov:
        return slice(None), slice(None)
    idx = np.argwhere(seg2d)
    if idx.size == 0:
        print("[crop] empty mask on this slice; showing full FOV")
        return slice(None), slice(None)
    (r0, c0), (r1, c1) = idx.min(0), idx.max(0) + 1
    r0, c0 = max(r0 - margin, 0), max(c0 - margin, 0)
    r1 = min(r1 + margin, seg2d.shape[0])
    c1 = min(c1 + margin, seg2d.shape[1])
    return slice(r0, r1), slice(c0, c1)


def wrapped_diff(a, b):
    """Phase-correct velocity difference: angle(e^{ia} * conj(e^{ib})) in (-pi, pi]."""
    return np.angle(np.exp(1j * a) * np.conj(np.exp(1j * b)))


def reconstruct(ckpt, train_opt, subj_dir, gpu=0, usrate=None, network=None):
    """QUICK LOOK ONLY -- not the submission path. Prefer the .npz route below.

    This inherits D_size from train_opt, so it returns a NORMALIZED READOUT-SLAB CROP
    (7 slices for version_411) carrying the dataloader's mask. board_eval.py instead
    reconstructs the full volume, overlap-averages the slabs, de-normalizes to physical
    units and loads the real segmask.mat -- i.e. what the leaderboard actually scores.
    For anything that goes in the paper:

        python board_eval.py --version_dir <V> --subj_dirs '<glob>' \
               --accels 50 --dump_npz results/npz
        python tools/make_recon_fig.py --ref_npz ..._gt.npz --prop_npz ..._pred.npz \
               --seg_npz ..._seg.npz --base_npz <baseline pred>

    -> (gt, pred, seg, usrate) in scorer layout (Nv, Nt, SPE, PE, FE).
    usrate: pick the batch at this acceleration (e.g. 50). None = first valid batch.
    """
    print("[warn] reconstruct(): readout-slab crop, normalized, dataloader mask -- "
          "quick look only. Use board_eval.py --dump_npz for paper figures.")
    import torch
    from torch.utils.data import DataLoader
    from utils.train_opt import opts_from_train_opt      # parses train_opt.txt
    from main import UnrolledNetwork
    from utils.dataloader import OwnDataset

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    opts = opts_from_train_opt(train_opt)
    if network:
        opts["network"] = network

    model = UnrolledNetwork(**opts)
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    ds = OwnDataset(**dict(opts, loss="supervised", mode="val",
                           explicit_val_dirs=[subj_dir], num_workers=0, batch_size=1))
    for batch in DataLoader(ds, batch_size=1, num_workers=0):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch = model.on_after_batch_transfer(batch, 0)
        r = int(batch["usrate"][0]) if hasattr(batch["usrate"], "__len__") else int(batch["usrate"])
        if usrate is not None and r != int(usrate):
            continue
        with torch.no_grad():
            recon = model._test_recon(model.network, batch)   # per-encoding: own forward, then stack
        gt = batch["gt"][:, 0]
        pred = np.transpose(recon[0].cpu().numpy(), (0, 1, 4, 3, 2))
        gtn = np.transpose(gt[0].cpu().numpy(), (0, 1, 4, 3, 2))
        seg = np.asarray(batch["segmentation"][0].cpu().numpy()
                         if torch.is_tensor(batch["segmentation"])
                         else batch["segmentation"][0], bool)
        if seg.size <= 1 or not seg.any() or pred.shape[0] < 2:
            continue
        print(f"[recon] {os.path.basename(ckpt)}  R~{r}  vencs={pred.shape[0]}")
        return gtn, pred, seg, r
    raise RuntimeError(f"no usable batch for {subj_dir}"
                       + (f" at R={usrate}" if usrate is not None else ""))


def build(ref, base, prop, seg, frame="auto", sl="auto", comp=0,
          gain=5.0, venc=None, out="fig_recon.pdf", dpi=400,
          margin=24, full_fov=False, title=None):
    """base=None -> single-model view: 2 rows x 3 cols (ref | proposed | error).

    comp: 0/1/2 for a signed velocity component, or "speed" for |v| -- speed reads
    far more like anatomy than a single signed component and is the better choice
    when you are trying to tell whether the reconstruction is physiological.
    """
    seg = np.asarray(seg, bool)
    solo = base is None
    speed_mode = str(comp).lower() == "speed"
    ci = 0 if speed_mode else int(comp)
    v_ref, v_prop = velocity(ref), velocity(prop)
    v_base = v_prop if solo else velocity(base)

    print(f"[shape] volume (Nv,Nt,SPE,PE,FE)={ref.shape}  seg={seg.shape}")
    t = pick_frame(v_ref, seg, frame)
    f = pick_slice(seg, sl)
    rs, cs = crop_box(seg[:, :, f], margin=margin, full_fov=full_fov)
    fov = seg.shape[:2]

    if venc is None:
        vencs, vscale, vunit = None, 1.0, "rad"
    else:
        vencs = (np.full(3, float(venc)) if np.ndim(venc) == 0
                 else np.asarray(venc, float).ravel())
        if vencs.size != 3:
            vencs = np.resize(vencs, 3)
        vscale, vunit = float(vencs[ci]) / np.pi, "cm/s"
        if not np.allclose(vencs, vencs[0]):
            print(f"[venc] anisotropic {vencs}: component view uses venc[{ci}]={vencs[ci]:g}")

    def mag2d(x):
        return magnitude(x)[t][:, :, f][rs, cs]

    def vel2d_rad(v):
        """Raw radians -- the wrapped difference must be taken before any venc scaling."""
        return v[ci, t][:, :, f][rs, cs]

    def speed2d(v):
        """|v| with EACH component scaled by its own venc before the norm."""
        comps = v[:, t][:, :, :, f]                       # (3, SPE, PE)
        if vencs is not None:
            comps = comps * (vencs[:, None, None] / np.pi)
        return np.linalg.norm(comps, axis=0)[rs, cs]

    seg2d = seg[:, :, f][rs, cs]

    m_ref, m_prop = mag2d(ref), mag2d(prop)
    m_base = m_prop if solo else mag2d(base)
    r_ref, r_prop = vel2d_rad(v_ref), vel2d_rad(v_prop)
    r_base = r_prop if solo else vel2d_rad(v_base)
    u_ref, u_prop, u_base = r_ref * vscale, r_prop * vscale, r_base * vscale
    print(f"[panel] {m_ref.shape[0]}x{m_ref.shape[1]} px of {fov[0]}x{fov[1]} FOV"
          f"  (t={t}, FE={f}{', full FOV' if full_fov else f', margin={margin}'})")

    if speed_mode:                       # |v| reads like anatomy; a signed component does not
        u_ref, u_prop = speed2d(v_ref) * vscale, speed2d(v_prop) * vscale
        u_base = u_prop if solo else speed2d(v_base) * vscale
        vel_err_base, vel_err_prop = np.abs(u_base - u_ref), np.abs(u_prop - u_ref)
        vcmap, vlo = "magma", 0.0
        vlabel = "Speed $|v|$"
    else:
        vel_err_base = np.abs(wrapped_diff(r_base, r_ref)) * vscale
        vel_err_prop = np.abs(wrapped_diff(r_prop, r_ref)) * vscale
        vcmap, vlo = "RdBu_r", None
        vlabel = f"Velocity $v_{ci+1}$"

    # ---- shared windows: never auto-scale per panel, it destroys comparability --------
    mag_hi = float(np.percentile(m_ref, 99.5))
    vlim = float(np.percentile(np.abs(u_ref), 99.5)) or 1.0
    vlo = 0.0 if speed_mode else -vlim
    mag_err_base, mag_err_prop = np.abs(m_base - m_ref), np.abs(m_prop - m_ref)
    if gain and gain > 0:
        mag_err_hi, vel_err_hi = mag_hi / gain, vlim / gain
        gtag = rf"$\times{gain:g}$"
    else:
        mag_err_hi = float(np.percentile(np.maximum(mag_err_base, mag_err_prop), 99)) or 1.0
        vel_err_hi = float(np.percentile(np.maximum(vel_err_base, vel_err_prop), 99)) or 1.0
        gtag = "auto"

    if solo:
        mag_row = [m_ref, m_prop, mag_err_prop]
        vel_row = [u_ref, u_prop, vel_err_prop]
        titles = ["Reference", "Reconstruction", r"$|$Recon$-$Ref$|$"]
        n_err = 1                                     # trailing columns that are error maps
    else:
        mag_row = [m_ref, m_base, m_prop, mag_err_base, mag_err_prop]
        vel_row = [u_ref, u_base, u_prop, vel_err_base, vel_err_prop]
        titles = ["Reference", "Baseline", "Proposed",
                  r"$|$Base$-$Ref$|$", r"$|$Prop$-$Ref$|$"]
        n_err = 2
    print(f"[window] image {mag_hi:.4g} / velocity +-{vlim:.4g} {vunit}; "
          f"error window {gtag} -> {mag_err_hi:.4g} / {vel_err_hi:.4g}")

    rows = [
        dict(panels=mag_row, cmap="gray", vmin=0, vmax=mag_hi,
             ecmap="inferno", evmin=0, evmax=mag_err_hi,
             label="Magnitude", unit="a.u."),
        dict(panels=vel_row, cmap=vcmap, vmin=vlo, vmax=vlim,
             ecmap="inferno", evmin=0, evmax=vel_err_hi,
             label=vlabel, unit=vunit),
    ]

    from matplotlib.gridspec import GridSpec
    ncol = len(titles)
    first_err = ncol - n_err
    h, w = m_ref.shape
    CBW, SPC = 0.065, 0.16                     # colourbar width and a dead spacer column
    widths = ([1.0] * first_err + [SPC, CBW, SPC]
              + [1.0] * n_err + [SPC, CBW])
    panel_in = 6.5 / sum(widths)
    fig = plt.figure(figsize=(7.2, 2 * panel_in * h / w + 0.8))
    gs = GridSpec(2, len(widths), figure=fig, width_ratios=widths,
                  wspace=0.05, hspace=0.10)
    col_of = (list(range(first_err))
              + [first_err + 3 + i for i in range(n_err)])
    cb_cols = (first_err + 1, len(widths) - 1)

    axes = np.empty((2, ncol), dtype=object)
    for r in range(2):
        for c in range(ncol):
            axes[r, c] = fig.add_subplot(gs[r, col_of[c]])
    handles = [[None] * ncol for _ in range(2)]

    for r, spec in enumerate(rows):
        for c, img in enumerate(spec["panels"]):
            ax = axes[r, c]
            is_err = c >= first_err
            handles[r][c] = ax.imshow(
                      img, cmap=spec["ecmap"] if is_err else spec["cmap"],
                      vmin=spec["evmin"] if is_err else spec["vmin"],
                      vmax=spec["evmax"] if is_err else spec["vmax"],
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=7, pad=3)
            if c == 0:
                ax.set_ylabel(spec["label"], fontsize=7)
                # segmentation contour on the reference panel only -- on all five it is noise
                ax.contour(seg2d, levels=[0.5], colors="lime", linewidths=0.5)

    # ---- colourbars into their own reserved columns -----------------------------------
    for r, spec in enumerate(rows):
        for cbc, src in zip(cb_cols, (handles[r][0], handles[r][first_err])):
            cax = fig.add_subplot(gs[r, cbc])
            cb = fig.colorbar(src, cax=cax)
            cb.ax.tick_params(labelsize=5, length=2, pad=1)
            cb.set_label(spec["unit"], fontsize=5, labelpad=1)
            cb.outline.set_linewidth(0.3)

    _annotate(axes, ref, base, prop, seg, t, ci, solo)

    if title:
        fig.suptitle(title, fontsize=8, y=0.995)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    print(f"[write] {out}")
    return out


def _annotate(axes, ref, base, prop, seg, t, comp, solo=False):
    """SSIM on the magnitude row, AngErr on the velocity row, reconstruction columns only."""
    if not _HAVE_EVAL:
        return
    kw = dict(fontsize=5.5, labelpad=1.5)
    try:
        v_ref, v_prop = velocity(ref), velocity(prop)
        pairs = [(prop, v_prop)] if solo else [(base, velocity(base)), (prop, v_prop)]
        for col, (x, v) in enumerate(pairs, start=1):
            mtxt = (f"SSIM {float(SSIM(magnitude(x), magnitude(ref), seg)):.3f}\n"
                    f"nRMSE {float(nRMSE(magnitude(x), magnitude(ref), seg)):.3f}")
            vtxt = (f"RelErr {float(RelErr(v[:, [t]], v_ref[:, [t]], seg)):.3f}\n"
                    f"AngErr {float(AngErr(v[:, [t]], v_ref[:, [t]], seg)):.1f}$^\\circ$")
            axes[0, col].set_xlabel(mtxt, **kw)
            axes[1, col].set_xlabel(vtxt, **kw)
    except Exception as e:
        print(f"[warn] annotation failed ({e}); check SSIM/AngErr signatures")


def build_multi_r(cases, comp=0, venc=None, gain=0.0, out="fig_multir.pdf",
                  title=None, margin=24, full_fov=False, dpi=400, row_h=1.0):
    """Acceleration sweep: rows = {magnitude, velocity}, columns = {Reference, R=...}.

    cases: sorted [(R, gt, pred, seg), ...] for ONE subject. Evidences the claim that a
    single model spans R=10-50, which the method-comparison panel cannot show. Far more
    compact per unit of information than one figure per R.
    """
    Rs = [c[0] for c in cases]
    gt0, seg = cases[0][1], np.asarray(cases[0][3], bool)
    speed_mode = str(comp).lower() == "speed"
    ci = 0 if speed_mode else int(comp)

    if venc is None:
        vencs, vunit = None, "rad"
    else:
        vencs = (np.full(3, float(venc)) if np.ndim(venc) == 0
                 else np.asarray(venc, float).ravel())
        vencs = np.resize(vencs, 3)
        vunit = "cm/s"

    t = pick_frame(velocity(gt0), seg, "auto")
    f = pick_slice(seg, "auto")
    rs, cs = crop_box(seg[:, :, f], margin=margin, full_fov=full_fov)

    def mg(x):
        return magnitude(x)[t][:, :, f][rs, cs]

    def vl(x):
        v = velocity(x)
        if speed_mode:
            c3 = v[:, t][:, :, :, f]
            if vencs is not None:
                c3 = c3 * (vencs[:, None, None] / np.pi)
            return np.linalg.norm(c3, axis=0)[rs, cs]
        s = 1.0 if vencs is None else vencs[ci] / np.pi
        return v[ci, t][:, :, f][rs, cs] * s

    mag_row = [mg(gt0)] + [mg(c[2]) for c in cases]
    vel_row = [vl(gt0)] + [vl(c[2]) for c in cases]
    titles = ["Reference"] + [f"$R={r}$" for r in Rs]

    mag_hi = float(np.percentile(mag_row[0], 99.5))
    vhi = float(np.percentile(np.abs(vel_row[0]), 99.5)) or 1.0
    vlo = 0.0 if speed_mode else -vhi
    vcmap = "magma" if speed_mode else "RdBu_r"
    vlabel = "Speed $|v|$" if speed_mode else f"Velocity $v_{ci+1}$"

    from matplotlib.gridspec import GridSpec
    ncol = len(titles)
    h, w = mag_row[0].shape
    CBW, SPC = 0.065, 0.16
    widths = [1.0] * ncol + [SPC, CBW]
    panel_in = 6.6 / sum(widths)
    fig = plt.figure(figsize=(7.2, 2 * panel_in * h / w * row_h + 0.6))
    gs = GridSpec(2, len(widths), figure=fig, width_ratios=widths, wspace=0.05, hspace=0.08)

    for r, (row, cmap, lo, hi, lab) in enumerate([
            (mag_row, "gray", 0.0, mag_hi, "Magnitude"),
            (vel_row, vcmap, vlo, vhi, vlabel)]):
        im = None
        for c, img in enumerate(row):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=7, pad=3)
            if c == 0:
                ax.set_ylabel(lab, fontsize=7)
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[r, len(widths) - 1]))
        cb.ax.tick_params(labelsize=5, length=2, pad=1)
        cb.set_label("a.u." if r == 0 else vunit, fontsize=5, labelpad=1)
        cb.outline.set_linewidth(0.3)

    if title:
        fig.suptitle(title, fontsize=8, y=0.995)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    print(f"[write] {out}  ({ncol} cols: Reference + R={Rs})")
    return out


def build_multi_r_byrow(cases, venc=None, out="fig_multir_rows.pdf", title=None,
                        margin=24, full_fov=False, dpi=400,
                        diff_clip=50.0, ang_clip=45.0, vel_clip=100.0):
    """R BY ROW: rows = Reference, R=10..50; cols =
       Magnitude | Velocity magnitude | Mag error (a.u.) | RelErr | AngErr (deg).

    The RelErr column is the velocity-MAGNITUDE relative error and AngErr(deg) the
    DIRECTION error -- the headline metric a speed map cannot show. Magnitude error is
    shown in native a.u. (signed: blue under-, red over-estimate), RelErr as a fraction,
    AngErr in degrees. Velocity-magnitude and all error columns are shown inside the
    segmentation ROI (NaN elsewhere). Consumes the SAME board-faithful
    cases=[(R,gt,pred,seg)] as build_multi_r.
    """
    import matplotlib.colors as mcolors
    Rs = [c[0] for c in cases]
    gt0, seg = cases[0][1], np.asarray(cases[0][3], bool)
    if venc is None:
        vencs, vunit = None, "rad"
    else:
        vencs = (np.full(3, float(venc)) if np.ndim(venc) == 0
                 else np.resize(np.asarray(venc, float).ravel(), 3))
        vunit = "cm/s"

    t = pick_frame(velocity(gt0), seg, "auto")
    f = pick_slice(seg, "auto")
    rs, cs = crop_box(seg[:, :, f], margin=margin, full_fov=full_fov)
    seg2d = seg[:, :, f][rs, cs]
    segn = np.where(seg2d, 1.0, np.nan)

    def mg(x):
        return magnitude(x)[t][:, :, f][rs, cs]

    def spd(x):
        c3 = velocity(x)[:, t][:, :, :, f]
        if vencs is not None:
            c3 = c3 * (vencs[:, None, None] / np.pi)
        return np.linalg.norm(c3, axis=0)[rs, cs]

    def ang_deg(xp, xr):
        vp = velocity(xp)[:, t][:, :, :, f][:, rs, cs]     # (3, H, W)
        vr = velocity(xr)[:, t][:, :, :, f][:, rs, cs]
        n = np.linalg.norm(vp, axis=0) * np.linalg.norm(vr, axis=0) + 1e-8
        return np.degrees(np.arccos(np.clip((vp * vr).sum(0) / n, -1.0, 1.0)))

    def mag_err(x):
        """Pure image-magnitude error, pred - ref, in native a.u. (signed:
        blue = under-estimate, red = over-estimate). No normalization."""
        return mg(x) - ref_mag

    def relerr(xp, ff=0.05):
        """Per-voxel RelErr = |norm(v_pred) - norm(v_ref)| / norm(v_ref): velocity
        MAGNITUDE relative error (the leaderboard RelErr, a fraction). Direction error
        is the separate AngErr column; this one is speed only."""
        sr = ref_spd
        den = np.maximum(sr, ff * np.nanmax(sr))
        return np.abs(spd(xp) - sr) / den

    ref_mag, ref_spd = mg(gt0), spd(gt0)
    mag_hi = float(np.percentile(ref_mag, 99.5))
    spd_hi = float(np.percentile(ref_spd[seg2d], 99.5)) or 1.0
    merr_hi = float(np.percentile(np.abs(mg(cases[-1][2]) - ref_mag)[seg2d], 99.0)) or 1.0

    cols = ["Magnitude", "Velocity magnitude", "Mag error",
            "RelErr", r"AngErr ($^\circ$)"]
    rows = ["Reference"] + [f"$R={r}$" for r in Rs]
    nr, nc = len(rows), len(cols)
    h, w = ref_mag.shape
    fig, ax = plt.subplots(nr, nc, figsize=(1.35 * nc, 1.35 * (h / w) * nr + 0.45),
                           constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)
    mnorm = mcolors.TwoSlopeNorm(0.0, -merr_hi, merr_hi)
    roi_lw = 0.9   # yellow ROI-contour width (thinner than the previous 1.2)

    def show(a, img, cmap, vmin=None, vmax=None, norm=None):
        return a.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm,
                        interpolation="nearest")

    ims = {}
    for r, lab in enumerate(rows):
        x = gt0 if r == 0 else cases[r - 1][2]             # pred for this R
        ims["mag"] = show(ax[r, 0], mg(x), "gray", 0, mag_hi)
        ims["spd"] = show(ax[r, 1], spd(x) * segn, "magma", 0, spd_hi)
        ax[r, 1].set_facecolor("black")
        ims["spd"].cmap.set_bad("black")
        # yellow ROI contour on the velocity-magnitude column, every row (incl. GT)
        ax[r, 1].contour(seg2d, levels=[0.5], colors="yellow", linewidths=roi_lw)
        if r == 0:
            for c in (2, 3, 4):
                ax[r, c].axis("off")
            ax[0, 0].contour(seg2d, levels=[0.5], colors="yellow", linewidths=roi_lw)
        else:
            ims["dm"] = show(ax[r, 2], mag_err(x) * segn, "coolwarm", norm=mnorm)
            ims["ds"] = show(ax[r, 3], relerr(x) * segn, "inferno", 0, 1.0)
            ims["an"] = show(ax[r, 4], ang_deg(x, gt0) * segn, "inferno", 0, ang_clip)
            # right 3 cols: black background outside ROI + yellow closed-loop ROI contour
            for c in (2, 3, 4):
                ax[r, c].set_facecolor("black")
                ims[{2: "dm", 3: "ds", 4: "an"}[c]].cmap.set_bad("black")
                ax[r, c].contour(seg2d, levels=[0.5], colors="yellow", linewidths=roi_lw)
        ax[r, 0].set_ylabel(lab, fontsize=7, labelpad=1)
        for c in range(nc):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for sp in ax[r, c].spines.values():
                sp.set_visible(False)
    for c, ttl in enumerate(cols):
        ax[0, c].set_title(ttl, fontsize=7, pad=2)

    def cbar(im, col, lab):
        cb = fig.colorbar(im, ax=ax[:, col], location="bottom", shrink=.9, pad=.015,
                          aspect=28)
        cb.set_label(lab, fontsize=6, labelpad=1)
        cb.ax.tick_params(labelsize=5, length=2, pad=1)
    cbar(ims["mag"], 0, "a.u."); cbar(ims["spd"], 1, vunit)
    cbar(ims["dm"], 2, "a.u."); cbar(ims["ds"], 3, "ratio"); cbar(ims["an"], 4, "deg")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[write] {out}  (R by row: {Rs}; cols mag|velmag|magerr|relerr|angerr)")
    return out


def build_pair_byrow(gt, base, prop, seg, R, venc=None, out="fig_pair_rows.pdf",
                     title=None, margin=24, full_fov=False, dpi=400, ang_clip=90.0):
    """PAIR BY ROW at a single R: rows = Proposed, Baseline, Ground truth; cols =
       Magnitude | Velocity magnitude | Mag error (a.u.) | RelErr | AngErr (deg).

    Same column semantics and masking as build_multi_r_byrow, but the rows compare TWO
    reconstructions (proposed vs a baseline -- CS-LLR, zero-filled, or an ablation model)
    against the shared ground truth at ONE acceleration. The Proposed and Baseline rows
    carry error columns vs GT; the Ground-truth row shows only Magnitude + Velocity
    magnitude (no error against itself). Velocity magnitude and every error column are
    masked to the segmentation ROI on black, and every panel gets a thin yellow ROI
    contour.
    """
    import matplotlib.colors as mcolors
    seg = np.asarray(seg, bool)
    if venc is None:
        vencs, vunit = None, "rad"
    else:
        vencs = (np.full(3, float(venc)) if np.ndim(venc) == 0
                 else np.resize(np.asarray(venc, float).ravel(), 3))
        vunit = "cm/s"

    t = pick_frame(velocity(gt), seg, "auto")
    f = pick_slice(seg, "auto")
    rs, cs = crop_box(seg[:, :, f], margin=margin, full_fov=full_fov)
    seg2d = seg[:, :, f][rs, cs]
    segn = np.where(seg2d, 1.0, np.nan)

    def mg(x):
        return magnitude(x)[t][:, :, f][rs, cs]

    def spd(x):
        c3 = velocity(x)[:, t][:, :, :, f]
        if vencs is not None:
            c3 = c3 * (vencs[:, None, None] / np.pi)
        return np.linalg.norm(c3, axis=0)[rs, cs]

    def ang_deg(xp, xr):
        vp = velocity(xp)[:, t][:, :, :, f][:, rs, cs]
        vr = velocity(xr)[:, t][:, :, :, f][:, rs, cs]
        n = np.linalg.norm(vp, axis=0) * np.linalg.norm(vr, axis=0) + 1e-8
        return np.degrees(np.arccos(np.clip((vp * vr).sum(0) / n, -1.0, 1.0)))

    ref_mag, ref_spd = mg(gt), spd(gt)
    mag_hi = float(np.percentile(ref_mag, 99.5))
    spd_hi = float(np.percentile(ref_spd[seg2d], 99.5)) or 1.0

    def mag_err(x):
        return mg(x) - ref_mag

    def relerr(xp, ff=0.05):
        den = np.maximum(ref_spd, ff * np.nanmax(ref_spd))
        return np.abs(spd(xp) - ref_spd) / den

    merr_hi = float(max(np.percentile(np.abs(mag_err(prop))[seg2d], 99.0),
                        np.percentile(np.abs(mag_err(base))[seg2d], 99.0))) or 1.0

    cols = ["Magnitude", "Velocity magnitude", "Mag error", "RelErr", r"AngErr ($^\circ$)"]
    row_specs = [("Proposed", prop), ("Baseline", base), ("Ground truth", gt)]
    nr, nc = len(row_specs), len(cols)
    h, w = ref_mag.shape
    fig, ax = plt.subplots(nr, nc, figsize=(1.35 * nc, 1.35 * (h / w) * nr + 0.45),
                           constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)
    mnorm = mcolors.TwoSlopeNorm(0.0, -merr_hi, merr_hi)
    roi_lw = 0.9

    def show(a, img, cmap, vmin=None, vmax=None, norm=None):
        return a.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm,
                        interpolation="nearest")

    ims = {}
    for r, (lab, x) in enumerate(row_specs):
        ims["mag"] = show(ax[r, 0], mg(x), "gray", 0, mag_hi)
        ims["spd"] = show(ax[r, 1], spd(x) * segn, "magma", 0, spd_hi)
        ax[r, 1].set_facecolor("black")
        ims["spd"].cmap.set_bad("black")
        # yellow ROI contour on magnitude + velocity magnitude, every row
        ax[r, 0].contour(seg2d, levels=[0.5], colors="yellow", linewidths=roi_lw)
        ax[r, 1].contour(seg2d, levels=[0.5], colors="yellow", linewidths=roi_lw)
        if lab == "Ground truth":
            for c in (2, 3, 4):
                ax[r, c].axis("off")
        else:
            ims["dm"] = show(ax[r, 2], mag_err(x) * segn, "coolwarm", norm=mnorm)
            ims["ds"] = show(ax[r, 3], relerr(x) * segn, "inferno", 0, 1.0)
            ims["an"] = show(ax[r, 4], ang_deg(x, gt) * segn, "inferno", 0, ang_clip)
            for c in (2, 3, 4):
                ax[r, c].set_facecolor("black")
                ims[{2: "dm", 3: "ds", 4: "an"}[c]].cmap.set_bad("black")
                ax[r, c].contour(seg2d, levels=[0.5], colors="yellow", linewidths=roi_lw)
        ax[r, 0].set_ylabel(lab, fontsize=7, labelpad=1)
        for c in range(nc):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for sp in ax[r, c].spines.values():
                sp.set_visible(False)
    for c, ttl in enumerate(cols):
        ax[0, c].set_title(ttl, fontsize=7, pad=2)

    def cbar(im, col, lab):
        cb = fig.colorbar(im, ax=ax[:, col], location="bottom", shrink=.9, pad=.015,
                          aspect=28)
        cb.set_label(lab, fontsize=6, labelpad=1)
        cb.ax.tick_params(labelsize=5, length=2, pad=1)
    cbar(ims["mag"], 0, "a.u."); cbar(ims["spd"], 1, vunit)
    cbar(ims["dm"], 2, "a.u."); cbar(ims["ds"], 3, "ratio"); cbar(ims["an"], 4, "deg")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[write] {out}  (pair rows proposed|baseline|gt at R={R}; "
          f"cols mag|velmag|magerr|relerr|angerr)")
    return out


def build_multi_model_byrow(gt, rows, seg, R, venc=None, gt_top=False,
                            out="fig_models_rows.pdf", title=None, margin=24,
                            full_fov=False, dpi=400, ang_clip=90.0):
    """MULTI-MODEL BY ROW at a single R: one row per reconstruction plus a ground-truth
    row (top if gt_top else bottom). Columns = Magnitude | Velocity magnitude |
    Mag error (a.u.) | RelErr | AngErr (deg). Model rows carry error columns vs GT; the
    GT row shows only Magnitude + Velocity magnitude.

    Following FlowVN Fig. 3, the recon Magnitude and its Mag-error map are shown over the
    ENTIRE field of view; Velocity magnitude, RelErr and AngErr stay masked to the
    segmentation ROI on black (outside the vessel the reference speed ~0 makes them pure
    noise). Every panel carries the yellow ROI contour.

    rows: list of (label, pred_array), in top-to-bottom order among the model rows
    (e.g. [("CS-LLR", cs), ("v455", v455), ("Proposed", prop)]).
    """
    import matplotlib.colors as mcolors
    seg = np.asarray(seg, bool)
    if venc is None:
        vencs, vunit = None, "rad"
    else:
        vencs = (np.full(3, float(venc)) if np.ndim(venc) == 0
                 else np.resize(np.asarray(venc, float).ravel(), 3))
        vunit = "cm/s"

    t = pick_frame(velocity(gt), seg, "auto")
    f = pick_slice(seg, "auto")
    rs, cs = crop_box(seg[:, :, f], margin=margin, full_fov=full_fov)
    seg2d = seg[:, :, f][rs, cs]
    segn = np.where(seg2d, 1.0, np.nan)

    def mg(x):
        return magnitude(x)[t][:, :, f][rs, cs]

    def spd(x):
        c3 = velocity(x)[:, t][:, :, :, f]
        if vencs is not None:
            c3 = c3 * (vencs[:, None, None] / np.pi)
        return np.linalg.norm(c3, axis=0)[rs, cs]

    def ang_deg(xp, xr):
        vp = velocity(xp)[:, t][:, :, :, f][:, rs, cs]
        vr = velocity(xr)[:, t][:, :, :, f][:, rs, cs]
        n = np.linalg.norm(vp, axis=0) * np.linalg.norm(vr, axis=0) + 1e-8
        return np.degrees(np.arccos(np.clip((vp * vr).sum(0) / n, -1.0, 1.0)))

    ref_mag, ref_spd = mg(gt), spd(gt)
    mag_hi = float(np.percentile(ref_mag, 99.5))
    spd_hi = float(np.percentile(ref_spd[seg2d], 99.5)) or 1.0

    def mag_err(x):
        return mg(x) - ref_mag

    def relerr(xp, ff=0.05):
        den = np.maximum(ref_spd, ff * np.nanmax(ref_spd))
        return np.abs(spd(xp) - ref_spd) / den

    # symmetric magnitude-error clip from the WORST model row, so every row shares scale
    merr_hi = max([float(np.percentile(np.abs(mag_err(x))[seg2d], 99.0))
                   for _, x in rows] + [1e-6])

    gt_row = ("GT", gt)
    row_specs = ([gt_row] + list(rows)) if gt_top else (list(rows) + [gt_row])

    cols = ["Magnitude", "Velocity magnitude", "Mag error", "RelErr", r"AngErr ($^\circ$)"]
    nr, nc = len(row_specs), len(cols)
    h, w = ref_mag.shape
    fig, ax = plt.subplots(nr, nc, figsize=(1.35 * nc, 1.35 * (h / w) * nr + 0.18 * nr + 0.45),
                           constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.05, wspace=0.01, hspace=0.06)
    mnorm = mcolors.TwoSlopeNorm(0.0, -merr_hi, merr_hi)
    roi_lw = 0.9

    def show(a, img, cmap, vmin=None, vmax=None, norm=None):
        return a.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm,
                        interpolation="nearest")

    def _corner(a, txt):
        """Metric label in a panel's bottom-right corner (clear of the centred ROI),
        white on a translucent dark box so it reads over both bright anatomy and the
        black error-map background."""
        a.text(0.97, 0.05, txt, transform=a.transAxes, ha="right", va="bottom",
               fontsize=6, color="white",
               bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.45))

    ims = {}
    for r, (lab, x) in enumerate(row_specs):
        ims["mag"] = show(ax[r, 0], mg(x), "gray", 0, mag_hi)
        ims["spd"] = show(ax[r, 1], spd(x) * segn, "magma", 0, spd_hi)
        ax[r, 1].set_facecolor("black")
        ims["spd"].cmap.set_bad("black")
        ax[r, 0].contour(seg2d, levels=[0.5], colors="yellow", linewidths=roi_lw)
        ax[r, 1].contour(seg2d, levels=[0.5], colors="yellow", linewidths=roi_lw)
        if lab == "GT":
            for c in (2, 3, 4):
                ax[r, c].axis("off")
        else:
            ims["dm"] = show(ax[r, 2], mag_err(x), "coolwarm", norm=mnorm)
            ims["ds"] = show(ax[r, 3], relerr(x) * segn, "inferno", 0, 1.0)
            ims["an"] = show(ax[r, 4], ang_deg(x, gt) * segn, "inferno", 0, ang_clip)
            ax[r, 2].contour(seg2d, levels=[0.5], colors="yellow", linewidths=roi_lw)
            for c in (3, 4):
                ax[r, c].set_facecolor("black")
                ims[{3: "ds", 4: "an"}[c]].cmap.set_bad("black")
                ax[r, c].contour(seg2d, levels=[0.5], colors="yellow", linewidths=roi_lw)
            if _HAVE_EVAL:
                try:
                    mx, mref = magnitude(x), magnitude(gt)
                    vx, vref = velocity(x), velocity(gt)
                    _corner(ax[r, 0], f"SSIM {float(SSIM(mx, mref, seg)):.3f}")
                    _corner(ax[r, 2], f"nRMSE {float(nRMSE(mx, mref, seg)):.3f}")
                    _corner(ax[r, 3], f"RelErr {float(RelErr(vx, vref, seg)):.3f}")
                    _corner(ax[r, 4], f"AngErr {float(AngErr(vx, vref, seg)):.1f}$^\\circ$")
                except Exception as e:
                    print(f"[warn] corner metric annotation failed ({e})")
        ax[r, 0].set_ylabel(lab, fontsize=11, fontweight="bold", labelpad=1)
        for c in range(nc):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for sp in ax[r, c].spines.values():
                sp.set_visible(False)
    for c, ttl in enumerate(cols):
        trow = 1 if (gt_top and c >= 2) else 0
        ax[trow, c].set_title(ttl, fontsize=11, fontweight="bold", pad=2)

    def cbar(im, col, lab):
        cb = fig.colorbar(im, ax=ax[:, col], location="bottom", shrink=.9, pad=.015,
                          aspect=28)
        cb.set_label(lab, fontsize=9, fontweight="bold", labelpad=1)
        cb.ax.tick_params(labelsize=5, length=2, pad=1)
    cbar(ims["mag"], 0, "a.u."); cbar(ims["spd"], 1, vunit)
    cbar(ims["dm"], 2, "a.u."); cbar(ims["ds"], 3, "ratio"); cbar(ims["an"], 4, "deg")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    labs = "|".join(l for l, _ in row_specs)
    print(f"[write] {out}  (rows {labs} at R={R}; cols mag|velmag|magerr|relerr|angerr)")
    return out


def build_multi_model_bycol(gt, rows, seg, R, venc=None, gt_left=True,
                            out="fig_models_cols.pdf", title=None, margin=24,
                            full_fov=False, dpi=400, ang_clip=90.0, merr_clip=0.15):
    """MULTI-MODEL BY COLUMN -- transpose of build_multi_model_byrow into the FlowVN
    Fig. 3 orientation: one COLUMN per model (GT plus each reconstruction), the five
    quantities Magnitude | Velocity magnitude | Mag error | RelErr | AngErr as ROWS. The
    GT column shows only Magnitude + Velocity magnitude. Same masking / full-FOV / corner
    annotations as build_multi_model_byrow, laid out portrait so the panels are larger.
    """
    import matplotlib.colors as mcolors
    seg = np.asarray(seg, bool)
    if venc is None:
        vencs, vunit = None, "rad"
    else:
        vencs = (np.full(3, float(venc)) if np.ndim(venc) == 0
                 else np.resize(np.asarray(venc, float).ravel(), 3))
        vunit = "cm/s"

    t = pick_frame(velocity(gt), seg, "auto")
    f = pick_slice(seg, "auto")
    rs, cs = crop_box(seg[:, :, f], margin=margin, full_fov=full_fov)
    seg2d = seg[:, :, f][rs, cs]
    segn = np.where(seg2d, 1.0, np.nan)

    def mg(x):
        return magnitude(x)[t][:, :, f][rs, cs]

    def spd(x):
        c3 = velocity(x)[:, t][:, :, :, f]
        if vencs is not None:
            c3 = c3 * (vencs[:, None, None] / np.pi)
        return np.linalg.norm(c3, axis=0)[rs, cs]

    def ang_deg(xp, xr):
        vp = velocity(xp)[:, t][:, :, :, f][:, rs, cs]
        vr = velocity(xr)[:, t][:, :, :, f][:, rs, cs]
        n = np.linalg.norm(vp, axis=0) * np.linalg.norm(vr, axis=0) + 1e-8
        return np.degrees(np.arccos(np.clip((vp * vr).sum(0) / n, -1.0, 1.0)))

    _mg_raw = mg
    mag_hi = float(np.percentile(_mg_raw(gt), 99.5)) or 1.0
    mg = lambda x: _mg_raw(x) / mag_hi
    ref_mag, ref_spd = mg(gt), spd(gt)
    spd_hi = float(np.percentile(ref_spd[seg2d], 99.5)) or 1.0

    def mag_err(x):
        return mg(x) - ref_mag

    def relerr(xp, ff=0.05):
        den = np.maximum(ref_spd, ff * np.nanmax(ref_spd))
        return np.abs(spd(xp) - ref_spd) / den

    merr_hi = float(merr_clip)

    gt_col = ("GT", gt)
    col_specs = ([gt_col] + list(rows)) if gt_left else (list(rows) + [gt_col])
    metrics = ["Magnitude", "Velocity magnitude", "Mag error", "RelErr", r"AngErr ($^\circ$)"]
    units = ["", vunit, "", "ratio", "deg"]      # magnitude + mag error are normalized -> unitless
    nr, nc = len(metrics), len(col_specs)
    h, w = ref_mag.shape

    fig, ax = plt.subplots(nr, nc, figsize=(1.9 * nc + 0.6, 1.9 * (h / w) * nr + 0.6),
                           squeeze=False, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.008, wspace=0.03, hspace=0.01)
    mnorm = mcolors.TwoSlopeNorm(0.0, -merr_hi, merr_hi)
    roi_lw = 0.9
    merr_lw = roi_lw + 1.6   # bold ROI contour specifically on the Mag-error row

    def show(a, img, cmap, vmin=None, vmax=None, norm=None):
        return a.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm,
                        interpolation="nearest")

    def _corner(a, txt):
        a.text(0.97, 0.06, txt, transform=a.transAxes, ha="right", va="bottom",
               fontsize=9, color="white",
               bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.45))

    rowim = {}
    for k, (lab, x) in enumerate(col_specs):
        rowim[0] = show(ax[0, k], mg(x), "gray", 0, 1.0)
        ax[0, k].contour(seg2d, levels=[0.5], colors="yellow", linewidths=roi_lw)
        im_spd = show(ax[1, k], spd(x) * segn, "magma", 0, spd_hi)   # background=0 outside ROI
        ax[1, k].set_facecolor("black"); im_spd.cmap.set_bad("black")
        ax[1, k].contour(seg2d, levels=[0.5], colors="yellow", linewidths=roi_lw)
        rowim[1] = im_spd
        if lab != "GT":
            im_dm = show(ax[2, k], mag_err(x), "coolwarm", norm=mnorm)
            ax[2, k].contour(seg2d.astype(float), levels=[0.5], colors="#00008b", linewidths=merr_lw)
            ax[2, k].contour(seg2d.astype(float), levels=[0.85], colors="white", linewidths=merr_lw * 0.55)
            im_ds = show(ax[3, k], relerr(x) * segn, "inferno", 0, 1.0)   # background=0 outside ROI
            ax[3, k].set_facecolor("black"); im_ds.cmap.set_bad("black")
            ax[3, k].contour(seg2d, levels=[0.5], colors="yellow", linewidths=roi_lw)
            im_an = show(ax[4, k], ang_deg(x, gt) * segn, "inferno", 0, ang_clip)   # background=0 outside ROI
            ax[4, k].set_facecolor("black"); im_an.cmap.set_bad("black")
            ax[4, k].contour(seg2d, levels=[0.5], colors="yellow", linewidths=roi_lw)
            rowim[2], rowim[3], rowim[4] = im_dm, im_ds, im_an
            if _HAVE_EVAL:
                try:
                    mx, mref = magnitude(x), magnitude(gt)
                    vx, vref = velocity(x), velocity(gt)
                    _corner(ax[0, k], f"SSIM {float(SSIM(mx, mref, seg)):.3f}")
                    _corner(ax[2, k], f"nRMSE {float(nRMSE(mx, mref, seg)):.3f}")
                    _corner(ax[3, k], f"RelErr {float(RelErr(vx, vref, seg)):.3f}")
                    _corner(ax[4, k], f"AngErr {float(AngErr(vx, vref, seg)):.1f}$^\\circ$")
                except Exception as e:
                    print(f"[warn] corner metric annotation failed ({e})")
        else:
            for m in (2, 3, 4):          # GT column: no error rows, leave blank
                ax[m, k].set_facecolor("white")
        ax[0, k].set_title(lab, fontsize=14, fontweight="bold", pad=3)
        for m in range(nr):
            ax[m, k].set_xticks([]); ax[m, k].set_yticks([])
            for sp in ax[m, k].spines.values():
                sp.set_visible(False)

    err_lab_col = 1 if gt_left else 0
    for m, name in enumerate(metrics):
        if m >= 2:
            axe = ax[m, err_lab_col]
            axe.text(-0.06, 0.5, name, transform=axe.transAxes, rotation=90,
                     ha="right", va="center", fontsize=10, fontweight="bold")
        else:
            ax[m, 0].set_ylabel(name, fontsize=10, fontweight="bold", labelpad=2)

    for m in range(nr):
        cb = fig.colorbar(rowim[m], ax=list(ax[m, :]), location="right", shrink=0.9,
                          pad=0.01, aspect=10)
        cb.set_label(units[m], fontsize=12, fontweight="bold", labelpad=1)
        cb.ax.tick_params(labelsize=5, length=2, pad=1)
        cb.outline.set_linewidth(0.3)

    if title:
        fig.suptitle(title, fontsize=9)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    labs = "|".join(l for l, _ in col_specs)
    print(f"[write] {out}  (cols {labs} at R={R}; rows mag|velmag|magerr|relerr|angerr)")
    return out


def collect_pair(prop_dir, base_dir, subj=None, usrate=None):
    """Resolve a paired panel from two board_eval --dump_npz dirs, no manual tag needed.

    Returns (ref, base, prop, seg, subj, R). gt/seg come from the proposed dir (they are
    the subject's, identical either side). Picks a subject+R present in BOTH dirs; with
    subj/usrate None it takes the first common one (or the requested acceleration).
    """
    def index(d):
        out = {}
        for g in glob.glob(os.path.join(d, "*_gt.npz")):
            m = re.match(r"(.*)_R(\d+)$", os.path.basename(g)[:-len("_gt.npz")])
            if m:
                out[(m.group(1), int(m.group(2)))] = g[:-len("_gt.npz")]
        return out

    pi, bi = index(prop_dir), index(base_dir)
    if not pi:
        raise SystemExit(f"no *_gt.npz in {prop_dir!r} -- run board_eval.py --dump_npz first")
    if not bi:
        raise SystemExit(f"no *_gt.npz in {base_dir!r} -- run board_eval.py --dump_npz first")
    common = sorted(set(pi) & set(bi))
    if not common:
        raise SystemExit(f"no (subject,R) in both dirs. prop={sorted(pi)} base={sorted(bi)}")
    keys = common
    if subj is not None:
        keys = [k for k in keys if k[0] == subj] or keys
    if usrate is not None:
        keys = [k for k in keys if k[1] == int(usrate)] or keys
    key = keys[0]
    print(f"[pair] subject '{key[0]}' at R={key[1]}  ({len(common)} common cases)")
    ps, bs = pi[key], bi[key]
    return (_load(ps + "_gt.npz"), _load(bs + "_pred.npz"), _load(ps + "_pred.npz"),
            _load(ps + "_seg.npz").astype(bool), key[0], key[1])


def collect_models(specs, subj=None, usrate=None):
    """Gather several board_eval --dump_npz dirs (one per model) for a multi-row panel.

    specs: list of (label, dir). Returns (gt, seg, R, subj, [(label, pred), ...]) for a
    subject+R present in EVERY dir. gt/seg come from the first dir (the subject's,
    identical across models).
    """
    def index(d):
        out = {}
        for g in glob.glob(os.path.join(d, "*_gt.npz")):
            m = re.match(r"(.*)_R(\d+)$", os.path.basename(g)[:-len("_gt.npz")])
            if m:
                out[(m.group(1), int(m.group(2)))] = g[:-len("_gt.npz")]
        return out

    indexed = [(lab, index(d), d) for lab, d in specs]
    for lab, idx, d in indexed:
        if not idx:
            raise SystemExit(f"no *_gt.npz in {d!r} (row '{lab}') -- run board_eval.py --dump_npz first")
    common = set.intersection(*[set(idx) for _, idx, _ in indexed])
    if not common:
        raise SystemExit("no (subject,R) present in ALL row dirs:\n  "
                         + "\n  ".join(f"{lab}: {sorted(idx)}" for lab, idx, _ in indexed))
    keys = sorted(common)
    if subj is not None:
        keys = [k for k in keys if k[0] == subj] or keys
    if usrate is not None:
        keys = [k for k in keys if k[1] == int(usrate)] or keys
    key = keys[0]
    print(f"[models] subject '{key[0]}' at R={key[1]}  ({len(common)} common cases, "
          f"{len(indexed)} model rows)")
    gt = _load(indexed[0][1][key] + "_gt.npz")
    seg = _load(indexed[0][1][key] + "_seg.npz").astype(bool)
    rows = [(lab, _load(idx[key] + "_pred.npz")) for lab, idx, _ in indexed]
    venc = None
    for _, _, d in indexed:                                # true VENC from whichever dump stamped it
        venc = venc_from_dir(d, key[0])
        if venc is not None:
            break
    return gt, seg, key[1], key[0], rows, venc


def collect_multi_r(npz_dir, subj=None):
    """Gather board_eval --dump_npz output as [(R, gt, pred, seg), ...] for one subject."""
    gts = sorted(glob.glob(os.path.join(npz_dir, "*_gt.npz")))
    if not gts:
        raise SystemExit(f"no *_gt.npz under {npz_dir!r} -- run board_eval.py --dump_npz first")
    bysubj = {}
    for g in gts:
        base = os.path.basename(g)[:-len("_gt.npz")]
        m = re.match(r"(.*)_R(\d+)$", base)
        if not m:
            continue
        bysubj.setdefault(m.group(1), []).append((int(m.group(2)), g))
    if not bysubj:
        raise SystemExit("filenames do not match <subj>_R<accel>_gt.npz")
    if subj is None:
        subj = max(bysubj, key=lambda s: len(bysubj[s]))       # the one with most accelerations
        print(f"[multi-R] subject '{subj}' ({len(bysubj[subj])} accelerations)")
    if subj not in bysubj:
        raise SystemExit(f"subject {subj!r} not found; have: {sorted(bysubj)}")
    out = []
    for R, g in sorted(bysubj[subj]):
        stem = g[:-len("_gt.npz")]
        out.append((R, _load(g), _load(stem + "_pred.npz"), _load(stem + "_seg.npz").astype(bool)))
    return out, subj


def recon_multi_r(ckpt, train_opt, subj_dir, gpu=0, accels=(10, 20, 30, 40, 50),
                  network=None, dump_npz=None, d_size=None):
    """One subject, board-faithful full-volume recon at EVERY R -> [(R, gt, pred, seg), ...].

    This is the fix for "make_recon_fig is still one R": it drives board_eval.eval_checkpoint
    (the submission-grade assembly -- per-slab norm rescale, overlap-average over D_size
    windows, real segmask.mat) with --dump_npz across all accelerations, then reads the dumps
    straight back via collect_multi_r. So the sweep panel is byte-identical to what the
    leaderboard scores -- unlike reconstruct() above, which is a normalized readout-slab crop
    for quick looks only and, by construction, one R per call.

    dump_npz: where to cache the assembled arrays. None -> a temp dir (discarded on exit; the
    figure is already written). Pass --npz_dir to keep them for re-plotting without re-running
    the network.
    """
    if not train_opt:
        raise SystemExit("--multi_r from a checkpoint needs --prop_train_opt (the run's train_opt.txt)")
    from utils.train_opt import opts_from_train_opt      # parses train_opt.txt
    from board_eval import eval_checkpoint               # submission-grade full-volume assembly

    opts = opts_from_train_opt(train_opt)
    if network:
        opts["network"] = network
    dirs = sorted(glob.glob(subj_dir)) if any(c in subj_dir for c in "*?[") else [subj_dir]
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise SystemExit(f"no subject directory matched {subj_dir!r}")
    dump = dump_npz or tempfile.mkdtemp(prefix="multir_npz_")
    os.makedirs(dump, exist_ok=True)
    print(f"[multi-R] reconstructing {len(dirs)} subject(s) at R={list(accels)} -> {dump}")
    eval_checkpoint(ckpt, opts, dirs, gpu=gpu, accels=list(accels),
                    d_size=d_size, dump_npz=dump)
    return collect_multi_r(dump, subj=None)             # picks the subject with the most R present


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- mode A: reconstruct from two checkpoints (preferred) -------------------------
    ap.add_argument("--base_ckpt", help="baseline FlowVN checkpoint")
    ap.add_argument("--base_train_opt", help="version_N/train_opt.txt for --base_ckpt")
    ap.add_argument("--prop_ckpt", help="proposed checkpoint")
    ap.add_argument("--prop_train_opt", help="version_N/train_opt.txt for --prop_ckpt")
    ap.add_argument("--network", default=None,
                    help="override the network name from train_opt (e.g. FlowVN); default = as trained")
    ap.add_argument("--subj_dir", "--subj_dirs", dest="subj_dir",
                    help="held-out-train subject dir or glob (has kdata_full -> real GT)")
    ap.add_argument("--usrate", type=int, default=50, help="acceleration to show (default 50)")
    ap.add_argument("--gpu", type=int, default=0)
    # --- mode B: pre-dumped scorer-format arrays --------------------------------------
    ap.add_argument("--ref_npz", help="reference (fully-sampled) complex volume")
    ap.add_argument("--base_npz", help="baseline FlowVN reconstruction")
    ap.add_argument("--prop_npz", help="proposed reconstruction")
    ap.add_argument("--seg_npz", help="boolean segmentation mask")
    ap.add_argument("--frame", default="auto", help="'auto' (peak in-mask speed) or an index")
    ap.add_argument("--slice", dest="sl", default="auto", help="'auto' (largest mask) or an SPE index")
    ap.add_argument("--comp", default="0",
                    help="velocity component 0/1/2, or 'speed' for |v| (reads more like anatomy)")
    ap.add_argument("--margin", type=int, default=24,
                    help="pixels of context around the mask (default 24; small values hide anatomy)")
    ap.add_argument("--full_fov", action="store_true", help="no cropping at all")
    ap.add_argument("--gain", type=float, default=5.0, help="error-map gain; state it in the caption")
    ap.add_argument("--venc", type=float, default=None,
                    help="encoding velocity, to display cm/s instead of radians. OPTIONAL and "
                         "normally omit it: nothing in this repo stores a venc (AngErr and RelErr "
                         "are both scale-invariant, so the scorer never needs one), and radians are "
                         "exact -- +-pi IS +-venc. Only pass a value you took from the acquisition "
                         "header. Note utils_flow expects venc PER DIRECTION (shape (3,)); this "
                         "flag applies one number to all three, valid only for isotropic venc.")
    ap.add_argument("--out", default="fig_recon.pdf")
    ap.add_argument("--title", default=None,
                    help="explicit figure title, overriding the auto subject/R/checkpoint one")
    ap.add_argument("--no_title", action="store_true",
                    help="no title at all -- use this for the paper panel, where the LNCS "
                         "caption does that job and a suptitle just duplicates it")
    ap.add_argument("--multi_r", action="store_true",
                    help="acceleration sweep: rows {magnitude, velocity} x columns "
                         "{Reference, R=...}. Source is either --npz_dir (pre-dumped) or, in "
                         "one command, --prop_ckpt --prop_train_opt --subj_dir (reconstructs "
                         "every R in --accels here, board-faithful)")
    ap.add_argument("--npz_dir", help="board_eval --dump_npz output dir to read for --multi_r; "
                         "when reconstructing from a checkpoint, doubles as the dir the assembled "
                         "arrays are cached in (else a temp dir is used)")
    ap.add_argument("--accels", default="10,20,30,40,50",
                    help="accelerations for --multi_r reconstruction from a checkpoint")
    ap.add_argument("--d_size", type=int, default=None,
                    help="readout-slab depth for --multi_r reconstruction; default = the run's "
                         "D_size (use the SAME D_size you submit with, e.g. 7)")
    ap.add_argument("--byrow", action="store_true",
                    help="--multi_r layout with R BY ROW + difference columns: "
                         "cols = Magnitude | Speed |v| | dMag(%%) | dSpeed(%%) | AngErr(deg). "
                         "Speed/dSpeed carry RelErr, AngErr carries direction (headline).")
    ap.add_argument("--diff_clip", type=float, default=50.0,
                    help="+/- %% clip for the --byrow difference columns (default 50)")
    ap.add_argument("--ang_clip", type=float, default=45.0,
                    help="deg clip for the --byrow AngErr column (default 90)")
    ap.add_argument("--merr_clip", type=float, default=0.15,
                    help="+/- clip for the Mag error diverging map in the --rows panel. Magnitude "
                         "is normalized to [0,1], so this is a fraction of GT max intensity "
                         "(default 0.15 = 15%%). Fixed value -> stable colorbar across subjects.")
    ap.add_argument("--vel_clip", type=float, default=120.0,
                    help="%% clip for the --byrow velocity-difference column (default 100)")
    ap.add_argument("--pair_dirs", nargs=2, metavar=("PROP_DIR", "BASE_DIR"),
                    help="two board_eval --dump_npz dirs; auto-resolves a common subject+R "
                         "into a baseline-vs-proposed panel (no manual filename tag)")
    ap.add_argument("--pair_byrow", action="store_true",
                    help="with --pair_dirs: ROWS = Proposed, Baseline, Ground truth at ONE R "
                         "(pick R with --usrate); cols = Magnitude | Velocity magnitude | "
                         "Mag error | RelErr | AngErr. venc from --venc or --subj_dir's CSV.")
    ap.add_argument("--rows", nargs="+", metavar="LABEL:DIR",
                    help="multi-model panel: one 'LABEL:DUMP_DIR' per reconstruction row, in "
                         "top-to-bottom order (LABEL must contain no ':'). A ground-truth row "
                         "is added (bottom, or top with --gt_top). Same 5 columns as --pair_byrow; "
                         "pick R with --usrate, venc from --venc or --subj_dir's CSV.")
    ap.add_argument("--gt_top", action="store_true",
                    help="place the ground-truth row at the TOP of a --rows panel (default: bottom); "
                         "with --vertical, places the GT COLUMN on the LEFT")
    ap.add_argument("--vertical", action="store_true",
                    help="--rows panel: transpose to FlowVN Fig. 3 orientation -- one COLUMN "
                         "per model, the five quantities as rows (portrait, larger panels)")
    ap.add_argument("--subj", default=None, help="subject tag; default = auto (most accels / first common)")
    ap.add_argument("--view", choices=["spe_pe", "pe_fe"], default="spe_pe",
                    help="display plane for --rows: 'spe_pe' (default, the 24x104 phase-encoded "
                         "plane, FE as slice axis) or 'pe_fe' (the ~104x128 PE-readout plane, SPE "
                         "as slice axis -- squarer, larger panels). Whole-volume metrics unchanged.")
    a = ap.parse_args()

    if a.multi_r:
        if a.prop_ckpt and a.subj_dir:
            # one command: reconstruct THIS subject at every R, board-faithful, then sweep
            accels = [int(x) for x in str(a.accels).split(",")]
            cases, subj = recon_multi_r(a.prop_ckpt, a.prop_train_opt, a.subj_dir,
                                        gpu=a.gpu, accels=accels, network=a.network,
                                        dump_npz=a.npz_dir, d_size=a.d_size)
        elif a.npz_dir:
            cases, subj = collect_multi_r(a.npz_dir, a.subj)
        else:
            ap.error("--multi_r needs either --npz_dir (a board_eval.py --dump_npz dir) "
                     "OR --prop_ckpt --prop_train_opt --subj_dir (reconstruct every R here)")
        ttl = None if a.no_title else (a.title or f"{subj}   |   acceleration sweep")
        if a.byrow:
            venc = a.venc if a.venc is not None else venc_from_dir(a.npz_dir, subj)
            if venc is None:
                venc = load_venc(a.subj_dir)
            build_multi_r_byrow(cases, venc=venc, out=a.out, title=ttl,
                                margin=a.margin, full_fov=a.full_fov,
                                diff_clip=a.diff_clip, ang_clip=a.ang_clip, vel_clip=a.vel_clip)
        else:
            build_multi_r(cases, comp=a.comp, venc=a.venc, gain=a.gain, out=a.out,
                          title=ttl, margin=a.margin, full_fov=a.full_fov)
        return

    if a.rows:
        specs = []
        for spec in a.rows:
            if ":" not in spec:
                ap.error(f"--rows entry {spec!r} must be LABEL:DIR")
            lab, d = spec.split(":", 1)                       # first ':' splits label from path
            specs.append((lab, d))
        gt, seg, R, subj, rows, venc_auto = collect_models(specs, subj=a.subj, usrate=a.usrate)
        gt, seg, rows = apply_view(a.view, gt, seg, rows)     # choose display plane (SPE x PE vs PE x FE)
        title = None if a.no_title else (a.title or f"{subj}   |   R={R}")
        venc = a.venc if a.venc is not None else (venc_auto if venc_auto is not None
                                                  else load_venc(a.subj_dir))
        if a.vertical:
            build_multi_model_bycol(gt, rows, seg, R, venc=venc, gt_left=a.gt_top,
                                    out=a.out, title=title, margin=a.margin,
                                    full_fov=a.full_fov, ang_clip=a.ang_clip,
                                    merr_clip=a.merr_clip)
        else:
            build_multi_model_byrow(gt, rows, seg, R, venc=venc, gt_top=a.gt_top,
                                    out=a.out, title=title, margin=a.margin,
                                    full_fov=a.full_fov, ang_clip=a.ang_clip)
        return

    if a.pair_dirs:
        ref, base, prop, seg, subj, R = collect_pair(a.pair_dirs[0], a.pair_dirs[1],
                                                     subj=a.subj, usrate=a.usrate)
        title = None if a.no_title else (a.title or f"{subj}   |   R={R}")
        if a.pair_byrow:
            venc = a.venc if a.venc is not None else venc_from_dir(a.pair_dirs[0], subj)
            if venc is None:
                venc = load_venc(a.subj_dir)
            build_pair_byrow(ref, base, prop, seg, R, venc=venc, out=a.out, title=title,
                             margin=a.margin, full_fov=a.full_fov, ang_clip=a.ang_clip)
        else:
            build(ref, base, prop, seg, frame=a.frame, sl=a.sl, comp=a.comp, gain=a.gain,
                  venc=a.venc, out=a.out, margin=a.margin, full_fov=a.full_fov, title=title)
        return

    if a.prop_ckpt and a.subj_dir and not a.base_ckpt:
        # single-model look: reference | reconstruction | error
        ref, prop, seg, _ = reconstruct(a.prop_ckpt, a.prop_train_opt, a.subj_dir,
                                        gpu=a.gpu, usrate=a.usrate)
        base = None
    elif a.base_ckpt and a.prop_ckpt and a.subj_dir:
        ref, base, seg, r = reconstruct(a.base_ckpt, a.base_train_opt, a.subj_dir,
                                        gpu=a.gpu, usrate=a.usrate)
        _, prop, _, r2 = reconstruct(a.prop_ckpt, a.prop_train_opt, a.subj_dir,
                                     gpu=a.gpu, usrate=a.usrate)
        assert r == r2, f"acceleration mismatch: baseline R={r}, proposed R={r2}"
    elif a.ref_npz and a.base_npz and a.prop_npz and a.seg_npz:
        ref, base, prop, seg = (_load(a.ref_npz), _load(a.base_npz),
                                _load(a.prop_npz), _load(a.seg_npz).astype(bool))
    else:
        ap.error("provide one of:\n"
                 "  --prop_ckpt --prop_train_opt --subj_dir            (single model)\n"
                 "  + --base_ckpt --base_train_opt                     (paired comparison)\n"
                 "  --ref_npz --base_npz --prop_npz --seg_npz          (pre-dumped arrays)")

    if a.no_title:
        title = None                       # paper panel: the LNCS caption does this job
    elif a.title:
        title = a.title
    else:
        bits = []
        if a.subj_dir:
            bits.append("/".join(os.path.normpath(a.subj_dir).split(os.sep)[-3:]))
        if a.usrate:
            bits.append(f"R={a.usrate}")
        title = "   |   ".join(bits) if bits else None

    build(ref, base, prop, seg, frame=a.frame, sl=a.sl, comp=a.comp, gain=a.gain,
          venc=a.venc, out=a.out, margin=a.margin, full_fov=a.full_fov, title=title)


if __name__ == "__main__":
    main()
