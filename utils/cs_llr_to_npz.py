#!/usr/bin/env python
"""Convert a CS-LLR reconstruction into a board_eval-compatible npz dump, so it can be
used as the BASELINE side of `make_recon_fig.py --pair_byrow`.

CS_LLR_exec.py saves one file per velocity encoding -- rec0.npy .. rec3.npy -- each of
shape (1, T, X, Y, Z) with image axes (X,Y,Z) = (readout, ky, kz), complex, already
multiplied by norm_k (physical units).

board_eval.py --dump_npz saves <tag>_pred.npz as a single array of shape
(Nv, T, SPE, PE, FE) = transpose(<(Nv,T,readout,ky,kz)>, (0,1,4,3,2)).

This script stacks the four encodings, applies the SAME (0,1,4,3,2) transpose, optionally
rescales the magnitude to the ground truth (CS-LLR and the network normalize
differently, so the Magnitude column and Mag-error are otherwise not comparable -- phase
/ velocity is scale-invariant and unaffected), and writes a self-consistent BASE_DIR by
copying <tag>_{gt,seg}.npz from the proposed dump.

VERIFY after running: render the pair figure and check the CS-LLR magnitude is spatially
aligned with the ground-truth magnitude. A same-shape but wrong permutation (e.g. ky<->kz
if they are equal-sized) will pass the shape check but look transposed -- fix --transpose.

Usage:
  python utils/cs_llr_to_npz.py --cs_dir <CS_LLR_exec output_dir> \
      --prop_dir results/fig_byrow_npz \
      --tag Center012_Philips_15T_Ambition_P009_R50 \
      --out_dir results/cs_llr_npz
"""
import os
import argparse
import numpy as np


def _load(path):
    z = np.load(path)
    return z[z.files[0]] if hasattr(z, "files") else z


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cs_dir", required=True,
                    help="dir holding rec0.npy..rec3.npy written by CS_LLR_exec.py")
    ap.add_argument("--prop_dir", required=True,
                    help="proposed board_eval --dump_npz dir (source of <tag>_gt/seg.npz)")
    ap.add_argument("--tag", required=True,
                    help="case tag, e.g. Center012_Philips_15T_Ambition_P009_R50")
    ap.add_argument("--out_dir", required=True, help="output BASE_DIR for the CS-LLR dump")
    ap.add_argument("--transpose", default="0,1,4,3,2",
                    help="axis permutation to match board_eval order, or 'none' (default 0,1,4,3,2)")
    ap.add_argument("--match_scale", action="store_true", default=True,
                    help="rescale CS-LLR magnitude to the GT median in-mask magnitude (default on)")
    ap.add_argument("--no_match_scale", dest="match_scale", action="store_false")
    a = ap.parse_args()

    # 1. stack the four per-encoding recons -> (V=4, T, X, Y, Z) complex
    recs = []
    for s in range(4):
        r = np.load(os.path.join(a.cs_dir, f"rec{s}.npy"))
        recs.append(r[0] if r.ndim == 5 else r)            # drop the leading V=1 axis
    pred = np.stack(recs, axis=0).astype(np.complex64)
    print(f"[cs->npz] stacked CS-LLR recon: {pred.shape} {pred.dtype}")

    # 2. match board_eval axis order (Nv, T, SPE, PE, FE)
    if a.transpose.lower() != "none":
        perm = tuple(int(x) for x in a.transpose.split(","))
        pred = np.transpose(pred, perm)
        print(f"[cs->npz] transposed with {perm} -> {pred.shape}")

    # 3. GT + seg come from the proposed dump (they are the subject's, identical either side)
    gt = _load(os.path.join(a.prop_dir, a.tag + "_gt.npz"))
    seg = _load(os.path.join(a.prop_dir, a.tag + "_seg.npz")).astype(bool)
    if pred.shape != gt.shape:
        raise SystemExit(f"shape mismatch after transpose: CS-LLR {pred.shape} vs GT {gt.shape}. "
                         f"Adjust --transpose so CS-LLR matches the GT axis order.")

    if a.match_scale:
        mag_gt, mag_cs = np.abs(gt[0]), np.abs(pred[0])    # (T, SPE, PE, FE)
        s = float(np.median(mag_gt[:, seg]) / max(np.median(mag_cs[:, seg]), 1e-8))
        pred = (pred * s).astype(np.complex64)
        print(f"[cs->npz] scale-matched CS-LLR magnitude to GT (x{s:.4g})")

    # 5. write a self-consistent BASE_DIR (collect_pair indexes by *_gt.npz, loads base *_pred)
    os.makedirs(a.out_dir, exist_ok=True)
    np.savez(os.path.join(a.out_dir, a.tag + "_pred.npz"), pred)
    np.savez(os.path.join(a.out_dir, a.tag + "_gt.npz"), gt)
    np.savez(os.path.join(a.out_dir, a.tag + "_seg.npz"), seg)
    vpath = os.path.join(a.prop_dir, a.tag + "_venc.npy")   # carry the true VENC across if stamped
    if os.path.exists(vpath):
        np.save(os.path.join(a.out_dir, a.tag + "_venc.npy"), np.load(vpath))
    print(f"[cs->npz] wrote {os.path.join(a.out_dir, a.tag)}_{{gt,pred,seg}}.npz")


if __name__ == "__main__":
    main()
