#!/usr/bin/env python
"""Re-score a board_eval.py --dump_npz directory WITHOUT re-running the network.

The dumps already contain the assembled full-volume gt/pred/seg per (case, R), so all
that's left is board_eval.score_case (the byte-identical leaderboard scorer: MSAC bg-phase
correction -> SSIM/nRMSE/RelErr/AngErr in-mask). This is the fast path to regenerate the
means + per-R table if the console log was lost -- CPU-only, a couple minutes for 13x5.

USAGE
    python tools/score_npz.py results/prop_v473ep21 [--out_csv paper/prop_v473ep21_scores.csv]
"""
import os, glob, csv, argparse, numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from board_eval import score_case          # reuses the byte-identical leaderboard scorer

METR = ("SSIM", "nRMSE", "RelErr", "AngErr")


def _load(p):
    z = np.load(p); return z[z.files[0]] if hasattr(z, "files") else z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="board_eval --dump_npz dir (has *_gt.npz/_pred.npz/_seg.npz)")
    ap.add_argument("--out_csv", default=None)
    a = ap.parse_args()

    gtps = sorted(glob.glob(os.path.join(a.dump, "*_gt.npz")))
    if not gtps:
        raise SystemExit(f"no *_gt.npz under {a.dump!r} -- run board_eval.py --dump_npz first")

    rows = []
    for gtp in gtps:
        tag = os.path.basename(gtp)[:-7]                  # strip _gt.npz
        R = int(tag.split("_R")[-1])
        gt = _load(gtp)
        pred = _load(os.path.join(a.dump, tag + "_pred.npz"))
        seg = _load(os.path.join(a.dump, tag + "_seg.npz")).astype(bool)
        m = score_case(gt, pred, seg)
        m.update(tag=tag, R=R)
        rows.append(m)
        print(f"  R={R:>2}  {tag:<44}  SSIM={m['SSIM']:.4f}  nRMSE={m['nRMSE']:.4f}  "
              f"RelErr={m['RelErr']:.4f}  AngErr={m['AngErr']:.3f}", flush=True)

    print("\n== board-faithful means ==")
    allm = {k: float(np.mean([r[k] for r in rows])) for k in METR}
    print("  overall : SSIM %.4f  nRMSE %.4f  RelErr %.4f  AngErr %.3f" % tuple(allm[k] for k in METR))
    for R in sorted({r["R"] for r in rows}):
        sub = [r for r in rows if r["R"] == R]
        print("  R=%-2d    : SSIM %.4f  nRMSE %.4f  RelErr %.4f  AngErr %.3f" %
              (R, *[np.mean([s[k] for s in sub]) for k in METR]))

    if a.out_csv:
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tag", "R", *METR])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})
        print(f"\nwrote {a.out_csv}")


if __name__ == "__main__":
    main()
