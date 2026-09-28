#!/usr/bin/env bash
# ---------------------------------------------------------------------------

set -euo pipefail
: "${DATA:?set DATA to the CMRx4DFlow root that contains TaskR1R2/}"
: "${CKPT:?set CKPT to the trained .ckpt}"
OUT=${OUT:-results/submission}

python -u main.py --mode test --loss supervised --input kdata_full.mat \
    --network FlowVN --features_in 1 --D_size 7 --T_size -1 \
    --num_stages 8 --features_out 24 --kernel_size 7 --num_act_weights 71 --act linear_flowvn \
    --refine_csm --robust_norm \
    --ckpt_path "$CKPT" \
    --val_dir "$DATA/TaskR1R2/ValidationSet/Aorta" \
    --in_base_dir "$DATA" --out_base_dir "$OUT" \
    --num_workers 4 --gpu "${GPU:-0}"
