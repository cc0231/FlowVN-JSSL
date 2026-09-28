#!/usr/bin/env python
"""Score the CS-LLR dumps in results/cs_llr_-725allnpz/ with the EXACT board scorer
used by board_eval.py (MSAC bg-phase correction -> SSIM/nRMSE/RelErr/AngErr in-mask),
so CS-LLR lines up with the proposed/baseline numbers. CPU-only."""
import os, glob, numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from board_eval import score_case   # reuses the byte-identical leaderboard scorer

DUMP = "results/cs_llr_-725allnpz"

def _load(p):
    z = np.load(p); return z[z.files[0]] if hasattr(z, "files") else z

rows = []
for gtp in sorted(glob.glob(os.path.join(DUMP, "*_gt.npz"))):
    tag = os.path.basename(gtp)[:-7]              # strip _gt.npz
    R = int(tag.split("_R")[-1])
    gt = _load(gtp)
    pred = _load(os.path.join(DUMP, tag + "_pred.npz"))
    seg = _load(os.path.join(DUMP, tag + "_seg.npz")).astype(bool)
    m = score_case(gt, pred, seg)
    m.update(tag=tag, R=R)
    rows.append(m)
    print(f"  R={R:>2}  {tag:<40}  SSIM={m['SSIM']:.4f}  nRMSE={m['nRMSE']:.4f}  "
          f"RelErr={m['RelErr']:.4f}  AngErr={m['AngErr']:.3f}", flush=True)

print("\n== CS-LLR board-faithful means ==")
allm = {k: np.mean([r[k] for r in rows]) for k in ("SSIM","nRMSE","RelErr","AngErr")}
print("  overall : SSIM %.4f  nRMSE %.4f  RelErr %.4f  AngErr %.3f" % tuple(allm[k] for k in ("SSIM","nRMSE","RelErr","AngErr")))
for R in sorted({r["R"] for r in rows}):
    sub = [r for r in rows if r["R"] == R]
    print("  R=%-2d    : SSIM %.4f  nRMSE %.4f  RelErr %.4f  AngErr %.3f" %
          (R, *[np.mean([s[k] for s in sub]) for k in ("SSIM","nRMSE","RelErr","AngErr")]))

import csv
with open("paper/cs_llr_scores.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["tag","R","SSIM","nRMSE","RelErr","AngErr"])
    w.writeheader()
    for r in rows: w.writerow({k: r[k] for k in w.fieldnames})
print("\nwrote paper/cs_llr_scores.csv")
