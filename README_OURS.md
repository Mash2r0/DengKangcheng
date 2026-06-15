# 图像反射去除课程项目复现实验说明

本项目以 ERRNet 为 baseline，在此基础上实现并实验了三类改进模型：

- `errnet_r3lite`：增加反射层/残差重建分支的轻量 R3 版本。
- `errnet_r3lite_gated`：baseline-preserving 的单模型 gated refiner。
- `errnet_dual_fusion`：冻结两个专家模型，只训练融合 mask 的双专家融合模型。

以下命令默认在项目根目录运行：

```powershell
cd D:\Documents\Fdu\3B\DIP\PJ\DengKangcheng
conda activate errnet
```

如果不用 `conda activate`，也可以在每条命令前加：

```powershell
conda run -n errnet
```

## 1. 数据准备

第一次运行前需要把原始数据处理成训练/测试脚本使用的目录结构：

```powershell
python datasets\prepare_test_data.py
python datasets\prepare_train_data.py
```

处理后的主要目录为：

```text
datasets/processed_data/
  VOCdevkit/VOC2012/PNGImages/
  real_train/
  testdata_CEILNET_table2/
  real20/
  postcard/
  objects/
  wild/
  sir2_withgt/
```

## 2. Benchmark 测试命令

测试脚本为 `test_errnet.py`。支持的数据集名称：

```text
ceilnet_table2
real20
objects
postcard
wild
sir2_withgt
custom
```

### 2.1 测试 ERRNet baseline

```powershell
python test_errnet.py --name errnet --dataset real20 -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper --inet errnet --result_dir results_baseline --nThreads 0 --no-verbose
```

其中 `--dataset real20` 可替换为其他 benchmark 名称。

### 2.2 (BEST) 测试单 gated refiner 

```powershell
python test_errnet.py --name errnet_r3lite_gated_supervised_ft20_4 --dataset real20 -r --icnn_path checkpoints/errnet_r3lite_gated_supervised_ft20_4/errnet_latest.pt --hyper --inet errnet_r3lite_gated --result_dir results_gated_ft20_4 --nThreads 0 --no-verbose
```

### 2.3 测试 dual expert fusion

如果加载的是已经训练好的 `errnet_dual_fusion` checkpoint，需要传入训练时使用的专家结构名，确保模型结构和 checkpoint 匹配：

```powershell
python test_errnet.py --name errnet_dual_fusion_gated_r3_ft20 --dataset real20 --hyper --inet errnet_dual_fusion --expert0_inet errnet_r3lite_gated --expert1_inet errnet_r3lite -r --icnn_path checkpoints/errnet_dual_fusion_gated_r3_ft20/errnet_latest.pt --result_dir results_dual_fusion --nThreads 0 --no-verbose
```

如果只是测试未训练的初始 dual fusion，则还需要传入两个专家 checkpoint：

```powershell
python test_errnet.py --name errnet_dual_fusion_init --dataset real20 --hyper --inet errnet_dual_fusion --expert0_inet errnet_r3lite_gated --expert0_path checkpoints/errnet_r3lite_gated_supervised_ft20_4/errnet_latest.pt --expert1_inet errnet_r3lite --expert1_path checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_080_00623560.pt --result_dir results_dual_fusion_init --nThreads 0 --no-verbose
```

### 2.4 测试自己的图片

自定义图片使用 `--dataset custom`。例如测试 `images_5/{1..5}.jpeg`：

```powershell
python test_errnet.py --name errnet_baseline_images5 --dataset custom --input_dir images5_baseline/input --result_dir images5_baseline/inference_768 --max_long_edge 768 -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper --inet errnet --nThreads 0 --display_id 0 --no-verbose
```

输出会保存在：

```text
images5_baseline/inference_768/custom/<image_name>/
  m_input.png
  errnet_baseline_images5.png
```

`--max_long_edge 768` 会把长边缩放到不超过 768，适合显存有限时推理较大图片。如果需要全分辨率结果，可以去掉该参数，但显存和耗时会明显增加。

## 3. 训练命令

### 3.1 R3Lite 第一阶段：aligned pretraining

从头训练 `errnet_r3lite`：

```powershell
python train_errnet.py --name errnet_r3lite_scratch --hyper --inet errnet_r3lite --synthesis_model mixed --lambda_rec 0.2 --lambda_r 0.1 --lambda_excl 0.01 --nEpochs 60 --nThreads 16 --display_id 0 --save_epoch_freq 5 --metric_plot_freq 1 --no-verbose
```

该阶段使用合成/有配对数据训练，目标是让 R3Lite 学会输出透射层、反射层和残差重建。

### 3.2 R3Lite 第二阶段：unaligned finetuning

基于第一阶段 checkpoint 继续在 unaligned 数据上微调：

```powershell
python train_errnet_unaligned.py --name errnet_r3lite_scratch_unaligned_ft --hyper -r --icnn_path checkpoints/errnet_r3lite_scratch/errnet_latest.pt --inet errnet_r3lite --synthesis_model mixed --unaligned_loss vgg --lambda_rec 0.2 --lambda_r 0.1 --lambda_excl 0.01 --nEpochs 80 --nThreads 16 --display_id 0 --save_epoch_freq 5 --metric_plot_freq 1 --no-verbose
```

常用的第二阶段 checkpoint：

```text
checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_080_00623560.pt
```

### 3.3 单模型 gated refiner

单 gated refiner 从 ERRNet baseline 初始化，并冻结 baseline 主干，只训练 `delta/mask/reflection/residual` 等轻量头部：

```powershell
python train_errnet.py --name errnet_r3lite_gated_supervised_ft20 --hyper -r --reset_epoch_on_load --icnn_path checkpoints/errnet/errnet_060_00463920.pt --inet errnet_r3lite_gated --synthesis_model ceilnet --lambda_gan 0 --lambda_rec 0 --lambda_r 0 --lambda_excl 0 --lambda_base 0.05 --lambda_mask 0.3 --mask_reflect_scale 0.12 --nEpochs 20 --nThreads 16 --display_id 0 --save_epoch_freq 5 --metric_plot_freq 1 --no-verbose
```

更激进的版本会减小 baseline 约束，使模型更容易偏离 baseline：

```powershell
python train_errnet.py --name errnet_r3lite_gated_supervised_ft20_aggressive --hyper -r --reset_epoch_on_load --icnn_path checkpoints/errnet/errnet_060_00463920.pt --inet errnet_r3lite_gated --synthesis_model ceilnet --lambda_gan 0 --lambda_rec 0 --lambda_r 0 --lambda_excl 0 --lambda_base 0.005 --lambda_mask 0.05 --mask_reflect_scale 0.08 --fusion_mask_bias -2.0 --nEpochs 20 --nThreads 16 --display_id 0 --save_epoch_freq 5 --metric_plot_freq 1 --no-verbose
```

已保存并用于 `images_5` 对比的 gated checkpoint：

```text
checkpoints/errnet_r3lite_gated_supervised_ft20_4/checkpoints/errnet_r3lite_gated_supervised_ft20_4/errnet_latest.pt
```

### 3.4 双专家融合 dual expert fusion

dual fusion 冻结两个专家，只训练一个融合 mask：

```text
T = T_expert0 + M * (T_expert1 - T_expert0)
```

推荐使用较稳定的 gated model 作为 `expert0`，R3Lite unaligned model 作为 `expert1`：

```powershell
python train_errnet.py --name errnet_dual_fusion_gated_r3_ft20 --hyper --expert0_inet errnet_r3lite_gated --expert0_path checkpoints/errnet_r3lite_gated_supervised_ft20_4/checkpoints/errnet_r3lite_gated_supervised_ft20_4/errnet_latest.pt --expert1_inet errnet_r3lite --expert1_path checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_080_00623560.pt --inet errnet_dual_fusion --synthesis_model ceilnet --lambda_gan 0 --gan_start_epoch -1 --fusion_mask_bias -4 --lambda_rec 0 --lambda_r 0 --lambda_excl 0 --lambda_base 0.03 --lambda_mask 0.2 --mask_reflect_scale 0.12 --nEpochs 20 --nThreads 16 --display_id 0 --save_epoch_freq 5 --metric_plot_freq 1 --no-verbose
```

更激进版本：

```powershell
python train_errnet.py --name errnet_dual_fusion_gated_r3_ft20_aggressive --hyper --expert0_inet errnet_r3lite_gated --expert0_path checkpoints/errnet_r3lite_gated_supervised_ft20_4/checkpoints/errnet_r3lite_gated_supervised_ft20_4/errnet_latest.pt --expert1_inet errnet_r3lite --expert1_path checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_080_00623560.pt --inet errnet_dual_fusion --synthesis_model ceilnet --lambda_gan 0 --gan_start_epoch -1 --fusion_mask_bias -2.5 --lambda_rec 0 --lambda_r 0 --lambda_excl 0 --lambda_base 0.01 --lambda_mask 0.1 --mask_reflect_scale 0.08 --nEpochs 20 --nThreads 16 --display_id 0 --save_epoch_freq 5 --metric_plot_freq 1 --no-verbose
```

## 4. 常用参数说明

### 4.1 通用参数

| 参数 | 作用 |
|---|---|
| `--name` | 实验名。checkpoint、日志、可视化结果会保存到 `checkpoints/<name>/`。测试时也决定输出文件名。 |
| `--gpu_ids` | 使用的 GPU。默认 `0`；CPU 测试可设为 `-1`。 |
| `--hyper` | 启用 ERRNet 原始 hypercolumn/VGG 特征输入。baseline 和本项目模型一般都需要打开。 |
| `--inet` | 选择网络结构。可用：`errnet`、`errnet_r3lite`、`errnet_r3lite_gated`、`errnet_dual_fusion`。 |
| `-r` / `--resume` | 从 checkpoint 加载模型。测试和 finetune 通常都需要。 |
| `--icnn_path` | 指定要加载的生成器 checkpoint。 |
| `--reset_epoch_on_load` | 从已有 checkpoint 初始化权重，但把 epoch/iteration 计数清零，适合 finetune 新实验。 |
| `--nEpochs` | 训练总 epoch。注意如果不加 `--reset_epoch_on_load`，会从 checkpoint 里的 epoch 继续计数。 |
| `--nThreads` | DataLoader 线程数。Windows 上排查问题可用 `0`，正常训练可用 `8` 或 `16`。 |
| `--display_id 0` | 关闭 visdom 显示，避免无 visdom 服务时报错。 |
| `--save_epoch_freq` | 每隔多少 epoch 保存一次 checkpoint。 |
| `--metric_plot_freq` | 每隔多少次记录/评测更新训练曲线图。输出在 `checkpoints/<name>/metric_plots/`。 |
| `--no-verbose` | 减少命令行输出。 |
| `--max_dataset_size` | smoke test 时限制训练样本数，例如 `--max_dataset_size 4`。 |

### 4.2 训练数据和合成参数

| 参数 | 作用 |
|---|---|
| `--synthesis_model ceilnet` | 使用 CEILNet 风格反射合成。gated refiner 通常使用它，比较稳定。 |
| `--synthesis_model mixed` | 混合多种反射合成方式，R3Lite 从头训练时更有多样性。 |
| `--low_sigma` / `--high_sigma` | 控制合成反射模糊核强度范围。 |
| `--low_gamma` / `--high_gamma` | 控制合成反射强度变化。 |
| `--batchSize` | batch size。默认 1，16GB 显存下保持 1 比较稳。 |
| `--loadSize` / `--fineSize` | 训练时多尺度缩放和裁剪大小。默认 `fineSize=224,224`。 |

### 4.3 损失权重参数

| 参数 | 作用 | 调参建议 |
|---|---|---|
| `--lambda_gan` | GAN loss 权重。gated/dual fusion 中建议设为 `0`，避免轻量 refiner 不稳定。 |
| `--gan_start_epoch` | `train_errnet.py` 中控制第几轮启用 GAN。设为 `-1` 表示全程禁用 GAN schedule。 |
| `--lambda_vgg` | 透射层 VGG perceptual loss 权重，默认 `0.1`。 |
| `--lambda_rec` | R3Lite 的重建一致性损失，约束 `T + R + residual` 接近输入。 |
| `--lambda_r` | R3Lite 的反射层辅助损失。 |
| `--lambda_excl` | 透射层和反射层梯度排斥损失，鼓励两层分离。 |
| `--lambda_base` | baseline-preserving 损失，约束输出不要过度偏离 baseline/expert0。越大越保守。 |
| `--lambda_mask` | mask 监督损失，约束 mask 更集中在反射区域。越大越保守。 |
| `--mask_reflect_scale` | 用 `abs(input-target)` 生成软反射 mask 时的尺度。越小，软 mask 越容易变大。 |
| `--fusion_mask_bias` | mask head 最后一层 bias 初始值。`-4` 很保守，`-2.5/-2` 更激进，`0` 接近一开始平均融合。 |

### 4.4 dual expert fusion 参数

| 参数 | 作用 |
|---|---|
| `--expert0_inet` | expert0 的网络结构。支持 `errnet`、`errnet_r3lite`、`errnet_r3lite_gated`。 |
| `--expert0_path` | expert0 checkpoint。训练初始 dual fusion 时必须提供。 |
| `--expert1_inet` | expert1 的网络结构。支持同上。 |
| `--expert1_path` | expert1 checkpoint。训练初始 dual fusion 时必须提供。 |

注意：测试已训练好的 `errnet_dual_fusion` checkpoint 时，`--expert0_path/--expert1_path` 可以不传，因为 checkpoint 内已经包含两个专家权重；但 `--expert0_inet/--expert1_inet` 仍应保持和训练时一致，否则模型结构可能对不上。

## 5. 测试参数说明

| 参数 | 作用 |
|---|---|
| `--dataset` | 指定测试数据集。benchmark 用 `real20` 等名称，自定义图片用 `custom`。 |
| `--data_root` | benchmark 数据根目录，默认 `datasets/processed_data`。 |
| `--input_dir` | `custom` 或 `internet` 模式下的输入图片目录。注意目录第一层应只放待测试图片。 |
| `--result_dir` | 测试结果保存目录。 |
| `--save_subdir` | 覆盖 `result_dir` 下默认子目录名。 |
| `--max_long_edge` | 推理时限制图片最长边，降低显存占用。benchmark 的 `real20` 默认会限制到 512。 |