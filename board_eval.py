#!/usr/bin/env python
"""
Board-faithful evaluation of a trained FlowVN checkpoint.

Reconstructs FULL-VOLUME recons for held-out-train subjects (which have kdata_full -> real
GT + segmask) at every acceleration R in {10,20,30,40,50}, and scores them with the EXACT
leaderboard pipeline: MSAC background-phase correction estimated from GT and applied to both
gt & pred, then SSIM / nRMSE / RelErr / AngErr inside the segmask
(cmr_helper_scripts/evaluation2026/score.py::compute_metrics_standard, replicated here to
avoid score.py's heavy import side-effects -- byte-equivalent).

Why not just read val_AngErr? Training-time validation runs on single readout slices with a
random R per slice, so it can only ever be a proxy. This evaluator rebuilds the full case
volume at a fixed R (exactly what the board scores), so the numbers are leaderboard-grade.

USAGE
-----
  # one-shot, all five R, all held-out subjects:
  python board_eval.py \
      --ckpt results/corrected_runs/lightning_logs/version_5/checkpoints/5-epoch=10-....ckpt \
      --train_opt results/corrected_runs/lightning_logs/version_5/train_opt.txt \
      --subj_dirs '$DATA/TaskR1R2/TrainSet/Aorta_heldout/*/*/*' \
      --gpu 0 --out_csv board_scores.csv

  # periodic: re-score the LATEST checkpoint under a version dir every 30 min, log to TensorBoard
  python board_eval.py \
      --version_dir results/corrected_runs/lightning_logs/version_5 \
      --subj_dirs '$DATA/TaskR1R2/TrainSet/Aorta_heldout/*/*/*' \
      --gpu 0 --tb_logdir results/corrected_runs/lightning_logs/version_5 --interval 1800
"""
import os
import sys
import csv
import time
import glob
import argparse
import numpy as np
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.join(_HERE, "cmr_helper_scripts", "evaluation2026")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
from utils_bgc import execute_MSAC                                  # noqa: E402
from utils_flow import complex2magflow                              # noqa: E402
from utils_metrics import SSIM, nRMSE, RelErr, AngErr               # noqa: E402
from utils.train_opt import opts_from_train_opt                    # noqa: E402  (parses train_opt.txt)

ACCELS = [10, 20, 30, 40, 50]


def score_case(gt, pred, seg):
    """gt, pred: complex (Nv, Nt, SPE, PE, FE); seg: (SPE, PE, FE) bool. Board metrics."""
    st = np.random.get_state()                                     # execute_MSAC reseeds np.random
    try:
        corr = np.asarray(execute_MSAC(gt, corr_fit_order=3, th=0.1)) * seg
    finally:
        np.random.set_state(st)
    factor = np.exp(-1j * corr)
    gt_c = gt.copy();   gt_c[1:]   = gt_c[1:]   * factor
    pred_c = pred.copy(); pred_c[1:] = pred_c[1:] * factor
    mag_gt, flow_gt = complex2magflow(gt_c)
    mag_pred, flow_pred = complex2magflow(pred_c)
    return {
        "SSIM":   float(SSIM(mag_pred, mag_gt, seg)),
        "nRMSE":  float(nRMSE(mag_pred, mag_gt, seg)),
        "RelErr": float(RelErr(flow_pred, flow_gt, seg)),
        "AngErr": float(AngErr(flow_pred, flow_gt, seg)),
    }


def _load_model(ckpt, opts, device):
    import torch
    from main import UnrolledNetwork
    model = UnrolledNetwork(**opts)
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


def _recon_slab(model, batch, network, zero_filled=False):
    import torch
    with torch.no_grad():
        if zero_filled:                                # learning-free floor: A^H(k)=imdata_p1
            return batch["imdata_p1"]                  # already coil-combined adjoint (see main.py)
        return model._test_recon(model.network, batch)   # per-encoding recon (matches test)


def _venc_for_case(cdir):
    """True per-direction VENC (cm/s) from the case's params CSV, or None. Saved next to
    the dump so make_recon_fig can auto-scale velocity without a manual --venc."""
    import glob as _glob
    try:
        try:
            from utils_datasl import read_params_csv
        except Exception:
            from utils.utils_datasl import read_params_csv
        for p in sorted(_glob.glob(os.path.join(str(cdir), "*.csv"))):
            params = read_params_csv(p)
            if params and params.get("VENC"):
                return np.asarray(params["VENC"], float).ravel()
    except Exception as e:
        print(f"  [venc] could not read VENC for {cdir}: {e}")
    return None


def eval_checkpoint(ckpt, opts, subj_dirs, gpu=0, accels=ACCELS, d_size=None, dump_npz=None,
                    zero_filled=False):
    """Reconstruct FULL case volumes exactly like the submission path (on_test_epoch_end:
    per-slab norm rescale + overlap-average over D_size windows + segmask.mat), then score
    each (subject, R) with the real leaderboard scorer. Returns list of dict rows."""
    import torch
    from torch.utils.data import DataLoader
    from utils.dataloader import OwnDataset, mat_to_npy_segmask
    from utils.misc_utils import mri_forward_op

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    network = opts["network"]
    D = int(d_size if d_size is not None else (opts.get("D_size", 1) or 1))   # match the submission's D_size
    model = _load_model(ckpt, opts, device)
    rows = []

    for R in accels:
        ds_opts = dict(opts, loss="supervised", mode="val", explicit_val_dirs=list(subj_dirs),
                       target_acc=int(R), D_size=D, batch_size=1, num_workers=0)
        ds = OwnDataset(**ds_opts)
        dl = DataLoader(ds, batch_size=1, num_workers=0)

        acc = {}                                       # case_dir -> {"slabs": [(ss, pslab, gslab)], "subj"}
        for batch in dl:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch = model.on_after_batch_transfer(batch, 0)
            recon = _recon_slab(model, batch, network, zero_filled=zero_filled)
            nrm = float(batch["norm"][0]) if hasattr(batch["norm"], "__len__") else float(batch["norm"])
            pslab = (recon[0] * nrm).cpu().numpy()                    # (Nv,T,D,H,W) physical units
            gslab = (batch["gt"][:, 0][0] * nrm).cpu().numpy()        # de-normalized GT (im/norm * norm)
            ss = int(batch["slice_start"][0]) if hasattr(batch["slice_start"], "__len__") else int(batch["slice_start"])
            cdir = batch["case_dir"][0] if isinstance(batch["case_dir"], (list, tuple)) else batch["case_dir"]
            subj = batch["subj"]; subj = subj[0] if isinstance(subj, (list, tuple)) else subj
            d = acc.setdefault(str(cdir), {"slabs": [], "subj": subj, "resid": []})
            d["slabs"].append((ss, pslab, gslab))

            with torch.no_grad():
                m_acq = (batch["kdata_p1"][:, :, 0, :, 0, :, :].abs() != 0).to(recon.real.dtype)
                k_pred = mri_forward_op(recon, batch["coil_sens"], m_acq)
                den = torch.linalg.vector_norm(batch["kdata_p1"])
                d["resid"].append(float(torch.linalg.vector_norm(k_pred - batch["kdata_p1"])
                                        / den.clamp_min(1e-12)))

        for cdir, d in acc.items():
            slabs = d["slabs"]
            Nx = max(ss for ss, _, _ in slabs) + D
            Nv, T, _, H, W = slabs[0][1].shape
            pa = np.zeros((Nv, T, Nx, H, W), np.complex64); ga = np.zeros_like(pa)
            cnt = np.zeros(Nx, np.float32)
            for ss, ps, gs in slabs:                                 # overlap-average, as on_test_epoch_end
                pa[:, :, ss:ss + D] += ps; ga[:, :, ss:ss + D] += gs; cnt[ss:ss + D] += 1
            cinv = (1.0 / np.maximum(cnt, 1.0))[None, None, :, None, None]
            pred = np.transpose(pa * cinv, (0, 1, 4, 3, 2))          # (Nv, Nt, SPE, PE, FE)
            gtv = np.transpose(ga * cinv, (0, 1, 4, 3, 2))
            seg = mat_to_npy_segmask(Path(cdir) / "segmask.mat").astype(bool)   # (SPE, PE, FE), same as test
            if seg.sum() == 0 or pred.shape[0] < 2:
                continue
            if dump_npz:                                             # save assembled arrays for offline re-scoring
                os.makedirs(dump_npz, exist_ok=True)

                parts = [p for p in Path(cdir).parts[-3:] if p]
                stem = "_".join(parts).replace(" ", "-") or str(d["subj"])
                tag = f"{stem}_R{R}"
                np.savez(os.path.join(dump_npz, tag + "_gt.npz"), gtv)
                np.savez(os.path.join(dump_npz, tag + "_pred.npz"), pred)
                np.savez(os.path.join(dump_npz, tag + "_seg.npz"), seg)
                _v = _venc_for_case(cdir)                    # true VENC from the case params CSV
                if _v is not None:
                    np.save(os.path.join(dump_npz, tag + "_venc.npy"), _v)
                    print(f"  [dump] {os.path.join(dump_npz, tag)}_{{gt,pred,seg}}.npz + _venc.npy {_v}")
                else:
                    print(f"  [dump] {os.path.join(dump_npz, tag)}_{{gt,pred,seg}}.npz  (no VENC csv found)")
            m = score_case(gtv, pred, seg)
            m.update(subj=d["subj"], R=R)
            m["KspResid"] = float(np.mean(d["resid"])) if d.get("resid") else float("nan")
            rows.append(m)
            print(f"  R={R:>2}  {d['subj']:<28}  SSIM={m['SSIM']:.4f}  nRMSE={m['nRMSE']:.4f}  "
                  f"RelErr={m['RelErr']:.4f}  AngErr={m['AngErr']:.3f}  "
                  f"KspResid={m['KspResid']:.4f}")
    return rows


def summarize(rows):
    """Adjusted-style means (plain mean over available cases) overall and per R."""
    out = {}
    if not rows:
        return out

    keys = ("SSIM", "nRMSE", "RelErr", "AngErr", "KspResid")
    for key in keys:
        vals = [r[key] for r in rows if key in r]
        out[key] = float(np.nanmean(vals)) if vals else float("nan")
    per_R = {}
    for R in sorted({r["R"] for r in rows}):
        sub = [r for r in rows if r["R"] == R]
        per_R[R] = {k: (float(np.nanmean([r[k] for r in sub if k in r]))
                        if any(k in r for r in sub) else float("nan")) for k in keys}
    out["per_R"] = per_R
    return out


def _latest_ckpt(version_dir):
    cks = glob.glob(os.path.join(version_dir, "checkpoints", "*.ckpt"))
    cks = [c for c in cks if os.path.basename(c) != "last.ckpt"] or cks
    return max(cks, key=os.path.getmtime) if cks else None


def _log_tb(tb_logdir, summary, step):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return
    w = SummaryWriter(tb_logdir)
    for k in ("SSIM", "nRMSE", "RelErr", "AngErr"):
        w.add_scalar(f"board/{k}_mean", summary[k], step)
    for R, d in summary.get("per_R", {}).items():
        for k, v in d.items():
            w.add_scalar(f"board/{k}_R{R}", v, step)
    w.close()


def run_once(args):
    ckpt = args.ckpt or _latest_ckpt(args.version_dir)
    if not ckpt:
        print("[board_eval] no checkpoint found"); return
    train_opt = args.train_opt or os.path.join(args.version_dir or os.path.dirname(os.path.dirname(ckpt)), "train_opt.txt")
    opts = opts_from_train_opt(train_opt)
    if args.network:
        opts["network"] = args.network
    subj_dirs = sorted(glob.glob(args.subj_dirs)) if any(c in args.subj_dirs for c in "*?[") else [args.subj_dirs]
    subj_dirs = [d for d in subj_dirs if os.path.isdir(d)]
    accels = [int(x) for x in args.accels.split(",")]
    print(f"[board_eval] ckpt={os.path.basename(ckpt)}  subjects={len(subj_dirs)}  R={accels}")

    rows = eval_checkpoint(ckpt, opts, subj_dirs, gpu=args.gpu, accels=accels,
                           d_size=args.d_size, dump_npz=args.dump_npz,
                           zero_filled=args.zero_filled)
    summary = summarize(rows)
    if summary:
        print("\n== board-faithful means ==")
        print("  overall : SSIM %.4f  nRMSE %.4f  RelErr %.4f  AngErr %.3f  KspResid %.4f"
              % (summary["SSIM"], summary["nRMSE"], summary["RelErr"], summary["AngErr"],
                 summary["KspResid"]))
        for R, d in summary["per_R"].items():
            print("  R=%-2d    : SSIM %.4f  nRMSE %.4f  RelErr %.4f  AngErr %.3f  KspResid %.4f"
                  % (R, d["SSIM"], d["nRMSE"], d["RelErr"], d["AngErr"], d["KspResid"]))
    if args.out_csv and rows:
        newf = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as f:
            wr = csv.writer(f)
            if newf:

                wr.writerow(["time", "ckpt", "R", "subj", "SSIM", "nRMSE", "RelErr", "AngErr",
                             "KspResid"])
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            for r in rows:
                wr.writerow([ts, os.path.basename(ckpt), r["R"], r["subj"],
                             r["SSIM"], r["nRMSE"], r["RelErr"], r["AngErr"],
                             r.get("KspResid", "")])
        print(f"[board_eval] appended {len(rows)} rows -> {args.out_csv}")
    if args.tb_logdir and summary:
        import torch
        step = torch.load(ckpt, map_location="cpu").get("global_step", 0)
        _log_tb(args.tb_logdir, summary, step)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="specific checkpoint; or use --version_dir for the latest")
    ap.add_argument("--version_dir", help="lightning version_N dir; picks the newest checkpoint")
    ap.add_argument("--train_opt", help="train_opt.txt (defaults to <version_dir>/train_opt.txt)")
    ap.add_argument("--subj_dirs", required=True, help="held-out subject dir or glob (has kdata_full)")
    ap.add_argument("--network", default=None)
    ap.add_argument("--accels", default="10,20,30,40,50")
    ap.add_argument("--d_size", type=int, default=None, help="readout-slab depth for reconstruction; default = the run's D_size. Use the SAME D_size you submit with (e.g. 7) so the recon matches the leaderboard exactly.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--dump_npz", default=None, help="dir to save assembled full-volume gt/pred/seg per case (re-score offline with tools/score_npz.py)")
    ap.add_argument("--zero_filled", action="store_true", help="score the learning-free floor A^H(k)=imdata_p1 instead of the model output (no network run; ckpt/train_opt still needed for dataloader + D_size)")
    ap.add_argument("--tb_logdir", default=None, help="log board/AngErr etc. scalars for TensorBoard")
    ap.add_argument("--interval", type=int, default=0, help="seconds between re-scores of the latest ckpt (0=one-shot)")
    args = ap.parse_args()
    if not (args.ckpt or args.version_dir):
        ap.error("provide --ckpt or --version_dir")

    if args.interval > 0:
        last = None
        while True:
            ck = args.ckpt or _latest_ckpt(args.version_dir)
            if ck and ck != last:
                run_once(args); last = ck
            else:
                print(f"[board_eval] no new checkpoint; sleeping {args.interval}s")
            time.sleep(args.interval)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
