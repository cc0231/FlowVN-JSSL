#!/usr/bin/env bash

set -euo pipefail
shopt -s extglob

: "${DATA:?set DATA to the CMRx4DFlow root that contains TaskR1R2/}"
AORTA=$DATA/TaskR1R2/TrainSet/Aorta
SSL_CENTER=Center011
GPU=${GPU:-0}

mkdir -p results
python -u main.py --mode train --input kdata_full.mat \
    --network FlowVN --features_in 1 --D_size 7 --T_size -1 \
    --num_stages 8 --features_out 24 --kernel_size 7 --num_act_weights 71 --act linear_flowvn \
    --loss jssl --flowvn_allvenc --ssdu_rho 0.2 \
    --explicit_train_dirs ${AORTA}_train/!($SSL_CENTER)/*/* \
    --ssl_dir "$AORTA/$SSL_CENTER" \
    --explicit_val_dirs ${AORTA}_heldout/!($SSL_CENTER)/*/* \
    --refine_csm --robust_norm \
    --augment --aug_coil 0.1 --aug_noise 0.04 --aug_phase_ramp 1.0 \
    --scanner_reps Philips_30T_Ingenia:2,Philips_15T_Ambition:2 \
    --ssim_weight 0.16 --ssim_levels 3 --ksp_weight 0.3 \
    --angerr_weight 1.0 --angerr_warmup_epochs 4 --vel_weight 1.0 \
    --lr 5e-4 --lr_sched plateau --lr_factor 0.7 --lr_patience 2 \
    --monitor train_loss_epoch --save_top_k -1 \
    --epoch 16 --batch_size 1 --accumulate_grad_batches 1 --grad_clip_val 1.0 \
    --skip_loss_thresh 8 --seed 42 \
    --num_workers 8 --val_num_workers 2 --gpu "$GPU" \
    2>&1 | tee "results/train_jssl_$(date +%Y%m%d_%H%M%S).log"

