#!/usr/bin/bash

datasets=("ceilnet_table2" "real20" "objects" "postcard" "wild")

for d in "${datasets[@]}"; do
    output=$(python test_errnet.py \
      --name errnet_r3lite_gated \
      --dataset "$d" \
      -r \
      --icnn_path checkpoints/errnet_r3lite_gated_supervised_ft20/errnet_latest.pt \
      --hyper \
      --inet errnet_r3lite_gated \
      --result_dir results_gated_supervised_ft20 2>&1 \
      --no-verbose | tail -1)
    echo "| $d | ${output:0:-3} |"
done