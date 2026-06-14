# TRACE - RR Project Session Summary

Last updated: 2026-06-14

## Project Context

Workspace:

```text
D:\Documents\Fdu\3B\DIP\PJ\DengKangcheng
```

Task: complete a DIP course project on single-image reflection removal using the cloned ERRNet repository as baseline. The conda environment is `errnet`. Raw data has already been prepared through:

```text
datasets\prepare_test_data.py
datasets\prepare_train_data.py
```

The goal has shifted from only reproducing ERRNet to producing an improved method whose metrics are at least competitive with, and ideally better than, the baseline.

## Main Work Completed

1. Baseline setup and evaluation
   - Ran baseline test-set evaluation.
   - Created first metric tables and inspected output directories/images.
   - Baseline checkpoint in active use:

```text
checkpoints/errnet/errnet_060_00463920.pt
```

2. Literature-guided improvement plan
   - Investigated recent single-image reflection removal ideas, including component separation, location/mask awareness, cascaded refinement, physically based rendering, and in-the-wild methods.
   - Chose a practical direction compatible with the ERRNet codebase and local 16 GB GPU:
     - R3Lite component-aware extension.
     - Baseline-preserving gated refinement.
     - Dual-expert fusion for safer improvement over baseline.
   - Plan file:

```text
workspace\RR_project_execution_plan.md
```

3. ERRNet-R3Lite implementation
   - Implemented `errnet_r3lite`.
   - Added auxiliary reflection/residual outputs and losses:
     - `lambda_rec`
     - `lambda_r`
     - `lambda_excl`
   - Added mixed synthesis option.
   - Ran smoke tests confirming loss/checkpoint path works.

4. Training/evaluation follow-up
   - User completed:
     - stage 1 aligned/pretrain.
     - stage 2 unaligned finetune.
   - Important R3Lite checkpoint:

```text
checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_080_00623560.pt
```

5. Checkpoint sweep and ablation
   - Performed checkpoint sweep and lambda ablations.
   - Report:

```text
workspace\results\checkpoint_sweep_ablation_report.md
```

6. Baseline-preserving gated refiner
   - Implemented `errnet_r3lite_gated`.
   - Main idea:

```text
T = T_base + M * Delta_T
```

   - Added:
     - `lambda_base`
     - `lambda_mask`
     - `mask_reflect_scale`
     - `fusion_mask_bias`
     - `no_freeze_gated_base`
   - Added `--reset_epoch_on_load` support for clean finetune epoch counting.

7. Training metric visualization
   - Added CSV/PNG metric curve logging during training/evaluation.
   - Main outputs are under:

```text
checkpoints\<run_name>\metric_plots\
```

   - Added options:
     - `--no_metric_plot`
     - `--metric_plot_freq`

8. Dual-expert fusion
   - Implemented `errnet_dual_fusion`.
   - Initial design fused frozen ERRNet baseline expert and frozen R3Lite expert:

```text
T = T_expert0 + M * (T_expert1 - T_expert0)
```

   - Then extended it to configurable experts:
     - `--expert0_inet`
     - `--expert1_inet`
     - `--expert0_path`
     - `--expert1_path`
   - Supported expert architectures:

```text
errnet
errnet_r3lite
errnet_r3lite_gated
```

   - This allows the previously finetuned gated model to be used as expert0.

## Important Files Modified

Code:

```text
models\arch\default.py
models\arch\__init__.py
models\errnet_model.py
options\errnet\train_options.py
options\base_option.py
train_errnet.py
engine.py
util\metric_visualizer.py
```

Docs/results:

```text
workspace\RR_project_execution_plan.md
workspace\results\dual_expert_fusion_implementation.md
workspace\results\checkpoint_sweep_ablation_report.md
```

## Current Key Code Entry Points

Network registry:

```text
models\arch\__init__.py
```

Implemented network names:

```text
errnet
errnet_r3lite
errnet_r3lite_gated
errnet_dual_fusion
```

Dual fusion implementation:

```text
models\arch\default.py
class DualExpertFusionNet
```

Model loading/loss integration:

```text
models\errnet_model.py
```

Training options:

```text
options\errnet\train_options.py
```

## Important Checkpoints

ERRNet baseline:

```text
checkpoints/errnet/errnet_060_00463920.pt
```

R3Lite unaligned finetune:

```text
checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_080_00623560.pt
```

Gated supervised finetune produced by the user:

```text
checkpoints/errnet_r3lite_gated_supervised_ft20/errnet_r3lite_gated_supervised_ft20/errnet_latest.pt
```

Note: this checkpoint is nested one extra directory level because of the command/output path used during training.

Dual fusion smoke checkpoints:

```text
checkpoints/dual_fusion_smoke/errnet_latest.pt
checkpoints/dual_fusion_flexible_smoke/errnet_latest.pt
```

## User-Reported Gated FT20 Metrics

Command used by user:

```powershell
conda run -n errnet python train_errnet.py --name errnet_r3lite_gated_supervised_ft20 --hyper -r --reset_epoch_on_load --icnn_path checkpoints/errnet/errnet_060_00463920.pt --inet errnet_r3lite_gated --synthesis_model ceilnet --lambda_gan 0 --lambda_rec 0 --lambda_r 0 --lambda_excl 0 --lambda_base 0.05 --lambda_mask 0.3 --mask_reflect_scale 0.12 --nEpochs 20 --nThreads 0 --display_id 0 --save_epoch_freq 5 --no-verbose
```

Metrics shown by user:

```text
ceilnet_table2 | LMSE 0.0046 | NCC 0.9810 | PSNR 27.9669 | SSIM 0.9419
real20         | LMSE 0.0200 | NCC 0.8877 | PSNR 23.5661 | SSIM 0.8298
objects        | LMSE 0.0030 | NCC 0.9818 | PSNR 24.8188 | SSIM 0.8980
postcard       | LMSE 0.0044 | NCC 0.9465 | PSNR 22.0781 | SSIM 0.8782
wild           | LMSE 0.0079 | NCC 0.9360 | PSNR 25.2038 | SSIM 0.8885
```

Observation: gated FT20 is stable and close to baseline, but the gain is small. This motivated dual-expert fusion.

## Dual Fusion Validation

Initial dual fusion test with default experts:

```text
expert0 = errnet baseline
expert1 = errnet_r3lite unaligned
dataset = real20
```

Result:

```text
LMSE 0.0200 | NCC 0.8883 | PSNR 23.5790 | SSIM 0.8292
```

Flexible dual fusion smoke test:

```text
expert0 = errnet_r3lite_gated
expert1 = errnet_r3lite
```

Training smoke passed:

```text
checkpoints/dual_fusion_flexible_smoke/errnet_001_00000003.pt
checkpoints/dual_fusion_flexible_smoke/errnet_latest.pt
checkpoints/dual_fusion_flexible_smoke/metric_plots/train.png
```

Reloaded flexible fusion checkpoint and tested real20 successfully:

```text
LMSE 0.0198 | NCC 0.8882 | PSNR 23.5907 | SSIM 0.8305
```

## Recommended Next Training Command

Use the user-trained gated model as the stable expert0 and R3Lite unaligned as expert1:

```powershell
conda run -n errnet python train_errnet.py --name errnet_dual_fusion_gated_r3_ft20 --hyper --expert0_inet errnet_r3lite_gated --expert0_path checkpoints/errnet_r3lite_gated_supervised_ft20/errnet_r3lite_gated_supervised_ft20/errnet_latest.pt --expert1_inet errnet_r3lite --expert1_path checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_080_00623560.pt --inet errnet_dual_fusion --synthesis_model ceilnet --lambda_gan 0 --gan_start_epoch -1 --lambda_rec 0 --lambda_r 0 --lambda_excl 0 --lambda_base 0.03 --lambda_mask 0.2 --mask_reflect_scale 0.12 --fusion_mask_bias -4 --nEpochs 20 --nThreads 16 --display_id 0 --save_epoch_freq 5 --metric_plot_freq 1 --no-verbose
```

If mask barely moves, use a more aggressive variant:

```powershell
conda run -n errnet python train_errnet.py --name errnet_dual_fusion_gated_r3_ft20_aggressive --hyper --expert0_inet errnet_r3lite_gated --expert0_path checkpoints/errnet_r3lite_gated_supervised_ft20/errnet_r3lite_gated_supervised_ft20/errnet_latest.pt --expert1_inet errnet_r3lite --expert1_path checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_080_00623560.pt --inet errnet_dual_fusion --synthesis_model ceilnet --lambda_gan 0 --gan_start_epoch -1 --lambda_rec 0 --lambda_r 0 --lambda_excl 0 --lambda_base 0.01 --lambda_mask 0.1 --mask_reflect_scale 0.12 --fusion_mask_bias -2.5 --nEpochs 20 --nThreads 16 --display_id 0 --save_epoch_freq 5 --metric_plot_freq 1 --no-verbose
```

## Recommended Test Command

Important: when testing or resuming a flexible dual-fusion checkpoint, pass the same expert architecture names used during training. The expert paths are not needed when loading a full dual-fusion checkpoint with `-r --icnn_path`, but `--expert0_inet/--expert1_inet` are needed so the model structure matches the checkpoint.

```powershell
conda run -n errnet python test_errnet.py --name errnet_dual_fusion_gated_r3_ft20 --dataset real20 --hyper --inet errnet_dual_fusion --expert0_inet errnet_r3lite_gated --expert1_inet errnet_r3lite -r --icnn_path checkpoints/errnet_dual_fusion_gated_r3_ft20/errnet_latest.pt --result_dir workspace/results/errnet_dual_fusion_gated_r3_ft20/eval --nThreads 0 --no-verbose
```

For initial, untrained fusion evaluation, both expert paths are needed:

```powershell
conda run -n errnet python test_errnet.py --name errnet_dual_fusion_init --dataset real20 --hyper --inet errnet_dual_fusion --expert0_inet errnet_r3lite_gated --expert0_path checkpoints/errnet_r3lite_gated_supervised_ft20/errnet_r3lite_gated_supervised_ft20/errnet_latest.pt --expert1_inet errnet_r3lite --expert1_path checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_080_00623560.pt --result_dir workspace/results/dual_fusion_gated_r3_init/eval --nThreads 0 --no-verbose
```

## Important Cautions

1. `errnet_r3lite_gated_supervised_ft20` is not a plain `DRNet` checkpoint.
   - It must be loaded with `--expert0_inet errnet_r3lite_gated`.
   - Loading it into a plain `errnet` expert will fail or lose the trained gated heads.

2. GAN schedule is now controlled by:

```text
--lambda_gan
--gan_start_epoch
```

   Use `--lambda_gan 0 --gan_start_epoch -1` to keep GAN disabled for all epochs.

3. For dual fusion, default experts are still backward compatible:

```text
expert0_inet = errnet
expert1_inet = errnet_r3lite
```

4. Testing saved flexible dual-fusion checkpoints requires the same expert architecture names as training, even when using `-r --icnn_path`.

5. `workspace\results\dual_expert_fusion_implementation.md` contains more detailed notes and commands for dual fusion.

## Current Git/Workspace Notes

At the time this trace was written, expected modified files include:

```text
models/arch/__init__.py
models/arch/default.py
models/errnet_model.py
options/errnet/train_options.py
train_errnet.py
workspace/TRACE.md
```

There are also generated checkpoints/results under `checkpoints\...` and `workspace\results\...`.
