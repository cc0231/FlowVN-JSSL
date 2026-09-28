#!/usr/bin/env python
"""Score every zero-filled dump in results/zf_npz into results/zf_scores.csv
(board_eval format), so plot_rsweep_3method.py can draw the zero-filled floor curve."""
import glob, os, csv, time, numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
import board_eval as be

def load(p):
    a = np.load(p); return a[a.files[0]]

rows = []
for gtf in sorted(glob.glob("results/zf_npz/*_gt.npz")):
    stem = gtf[:-7]
    base = os.path.basename(stem)
    subj = base.rsplit("_R", 1)[0]
    R = int(base.rsplit("_R", 1)[1])
    gt = load(gtf); pred = load(stem + "_pred.npz"); seg = load(stem + "_seg.npz").astype(bool)
    m = be.score_case(gt, pred, seg)
    m.update(subj=subj, R=R); rows.append(m)
    print(f"{subj:40s} R{R:<2} SSIM={m['SSIM']:.3f} nRMSE={m['nRMSE']:.3f} "
          f"RelErr={m['RelErr']:.3f} AngErr={m['AngErr']:.2f}", flush=True)

with open("results/zf_scores.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["time", "ckpt", "R", "subj", "SSIM", "nRMSE", "RelErr", "AngErr"])
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    for r in rows:
        w.writerow([ts, "zero_filled", r["R"], r["subj"], r["SSIM"], r["nRMSE"], r["RelErr"], r["AngErr"]])

print(f"\nwrote results/zf_scores.csv ({len(rows)} rows)")
for R in (10, 20, 30, 40, 50):
    vs = [r for r in rows if r["R"] == R]
    if vs:
        print(f"  R{R}: " + " ".join(f"{k}={np.mean([r[k] for r in vs]):.3f}"
                                     for k in ("SSIM", "nRMSE", "RelErr", "AngErr")))
