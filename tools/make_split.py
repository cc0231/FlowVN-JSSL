#!/usr/bin/env python
"""Stratified train / held-out split of the CMRx4DFlow 2026 aortic TrainSet.

Copies ~10% of the subjects of EACH center into <src>_heldout and the rest into
<src>_train, preserving the <Center>/<Vendor>/<Patient> layout. These are real copies
(not symlinks), so deleting the split folders can never touch the originals -- but they
DUPLICATE the k-space on disk (~1x the dataset). Seeded, so the split is reproducible.

USAGE
    python tools/make_split.py --src $DATA/TaskR1R2/TrainSet/Aorta
"""
import argparse
import glob
import os
import random
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="TrainSet anatomy dir, e.g. $DATA/TaskR1R2/TrainSet/Aorta")
ap.add_argument("--frac", type=float, default=0.1, help="held-out fraction per center")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

random.seed(a.seed)
src = a.src.rstrip("/\\")
train_root, held_root = src + "_train", src + "_heldout"
pats = sorted({os.path.dirname(p) for p in glob.glob(f"{src}/**/params.csv", recursive=True)})
by_center = {}
for p in pats:
    by_center.setdefault(os.path.relpath(p, src).split(os.sep)[0], []).append(p)   # 1st level = center

n_tr = n_h = 0
for c, ps in sorted(by_center.items()):
    random.shuffle(ps)
    k = max(1, int(len(ps) * a.frac + 1e-9))   # == len(ps) // 10 at the default frac (paper split)
    for i, p in enumerate(ps):
        dst = os.path.join(held_root if i < k else train_root, os.path.relpath(p, src))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(p, dst, dirs_exist_ok=True)
        n_h, n_tr = n_h + int(i < k), n_tr + int(i >= k)
print(f"copied {n_tr} train -> {train_root}\n       {n_h} heldout -> {held_root}")
