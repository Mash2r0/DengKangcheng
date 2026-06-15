# images_5 ERRNet baseline Inference Metrics

Checkpoint:

```text
checkpoints/errnet/errnet_060_00463920.pt
```

Inference command:

```powershell
conda run -n errnet python test_errnet.py --name errnet_baseline_images5 --dataset custom --input_dir images5_baseline/input --result_dir images5_baseline/inference_768 --max_long_edge 768 -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper --inet errnet --nThreads 0 --display_id 0 --no-verbose
```

Outputs:

```text
images5_baseline/inference_768/custom/{1..5}/errnet_baseline_images5.png
images5_baseline/effect_grid.png
```

Notes:

- The run uses `--max_long_edge 768`.
- For metric calculation, each model output was resized to the corresponding `images_5/groundtruth/{1..5}.png` size with bicubic interpolation.
- Metrics are computed with `util/index.py`, using `float32` arrays to avoid integer overflow in LMSE.

| image | LMSE | NCC | PSNR | SSIM |
|---|---:|---:|---:|---:|
| 1 | 0.0041 | 0.9378 | 20.5517 | 0.8411 |
| 2 | 0.0175 | 0.7156 | 15.8404 | 0.7772 |
| 3 | 0.0122 | 0.9062 | 19.0679 | 0.7679 |
| 4 | 0.0169 | 0.9574 | 20.3132 | 0.7172 |
| 5 | 0.0087 | 0.9471 | 21.4066 | 0.7378 |
| mean | 0.0119 | 0.8928 | 19.4360 | 0.7682 |
