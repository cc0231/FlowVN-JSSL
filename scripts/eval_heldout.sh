#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob
: "${DATA:?set DATA to the CMRx4DFlow root that contains TaskR1R2/}"
: "${VERSION_DIR:?set VERSION_DIR to results/corrected_runs/lightning_logs/version_N}"
HELDOUT=$DATA/TaskR1R2/TrainSet/Aorta_heldout

CKPT_ARG=(--version_dir "$VERSION_DIR")
[[ -n "${CKPT:-}" ]] && CKPT_ARG=(--ckpt "$CKPT" --train_opt "$VERSION_DIR/train_opt.txt")

# Center011 is excluded: it is the reference-free self-supervised stream in training.
for subj in $HELDOUT/!(Center011)/*/*; do
    python -u board_eval.py "${CKPT_ARG[@]}" --subj_dirs "$subj" \
        --gpu "${GPU:-0}" --dump_npz results/heldout_npz
done
python tools/score_npz.py results/heldout_npz --out_csv results/heldout_scores.csv

