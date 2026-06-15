# images_5 Gated FT20-4 Inference Metrics

Checkpoint:

```text
checkpoints/errnet_r3lite_gated_supervised_ft20_4/checkpoints/errnet_r3lite_gated_supervised_ft20_4/errnet_latest.pt
```

Inference command:

```powershell
conda run -n errnet python test_errnet.py --name errnet_r3lite_gated_supervised_ft20_4_images5 --dataset custom --input_dir images5_gated_ft20_4/input --result_dir images5_gated_ft20_4/inference_768 --max_long_edge 768 -r --icnn_path checkpoints/errnet_r3lite_gated_supervised_ft20_4/checkpoints/errnet_r3lite_gated_supervised_ft20_4/errnet_latest.pt --hyper --inet errnet_r3lite_gated --nThreads 0 --display_id 0 --no-verbose
```

Outputs:

```text
images5_gated_ft20_4/inference_768/custom/{1..5}/errnet_r3lite_gated_supervised_ft20_4_images5.png
images5_gated_ft20_4/effect_grid.png
```

Notes:

- The run uses `--max_long_edge 768`.
- For metric calculation, each model output was resized to the corresponding `images_5/groundtruth/{1..5}.png` size with bicubic interpolation.
- Metrics are computed with `util/index.py`, using `float32` arrays to avoid integer overflow in LMSE.

| image | LMSE | NCC | PSNR | SSIM |
|---|---:|---:|---:|---:|
| 1 | 0.0042 | 0.9380 | 20.5431 | 0.8381 |
| 2 | 0.0170 | 0.7154 | 15.8491 | 0.7761 |
| 3 | 0.0093 | 0.9121 | 19.3075 | 0.8060 |
| 4 | 0.0140 | 0.9624 | 20.6536 | 0.7591 |
| 5 | 0.0048 | 0.9588 | 22.1877 | 0.8216 |
| mean | 0.0098 | 0.8973 | 19.7082 | 0.8002 |
