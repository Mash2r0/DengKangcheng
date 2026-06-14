#!/usr/bin/bash

datasets=("ceilnet_table2" "real20" "objects" "postcard" "wild")

for d in "${datasets[@]}"; do
    output=$(python test_errnet.py \
      --name errnet_dual_fusion_gated_r3_ft20 \
      --dataset "$d" \
      -r \
      --icnn_path checkpoints/errnet_dual_fusion_gated_r3_ft20/errnet_latest.pt \
      --hyper \
      --inet errnet_dual_fusion \
      --expert0_inet errnet_r3lite_gated \
      --expert1_inet errnet_r3lite \
      --result_dir results_errnet_dual_fusion_gated_r3_ft20 2>&1 \
      --no-verbose | tail -1)
    echo "| $d | ${output:0:-3} |"
done