# 图像反射去除课程项目执行方案 v0.3

依据文件：`docs/2026-DIP课程项目-RR.pptx`、`README_DIP26.md`、当前 ERRNet 代码仓库、`workspace/RR_literature_survey.md`、`workspace/results/r3lite_result_analysis.md`、`workspace/results/checkpoint_sweep_ablation_report.md`。

DDL：2026-06-16 17:00，邮件发送至 `qxiang24@m.fudan.edu.cn`，主题 `DIP课程论文-学号-姓名`。

## 1. 项目目标

完成单图像反射去除算法实践，交付：

- 可运行代码仓库，基于 ERRNet baseline 完成复现与改进。
- baseline 与改进方法的定量对比：PSNR、SSIM、NCC、LMSE。
- 指定测试集与自采 5 张照片的可视化结果。
- 课程论文，包含背景综述、方法、实验设置、定量/定性结果、结论与个人贡献。
- 汇报 PPT。
- 训练得到的改进模型权重，并在论文中提供网盘或 OneDrive 链接。

## 2. 当前仓库状态

已完成：

- ERRNet 仓库已克隆到当前目录。
- conda 环境名为 `errnet`。
- `datasets/prepare_test_data.py` 和 `datasets/prepare_train_data.py` 已运行。
- 预训练权重存在：`checkpoints/errnet/errnet_060_00463920.pt`。
- Baseline 已完成多测试集评测，结果见 `workspace/results/metrics_baseline.md` 和 `测试结果.md`。
- `ERRNet-R3Lite` 已实现并完成 smoke test。
- `ERRNet-R3Lite` scratch aligned pretraining 已完成，主要 checkpoint：
  - `checkpoints/errnet_r3lite_scratch/errnet_055_00425260.pt`
  - `checkpoints/errnet_r3lite_scratch/errnet_060_00463920.pt`
  - `checkpoints/errnet_r3lite_scratch/errnet_latest.pt`
- `ERRNet-R3Lite` unaligned finetuning 已完成，主要 checkpoint：
  - `checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_075_00583650.pt`
  - `checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_080_00623560.pt`
  - `checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_latest.pt`
- 已完成 checkpoint sweep 与短程 loss ablation，结论见 `workspace/results/checkpoint_sweep_ablation_report.md`。
- 处理后的主要数据集已在位：
  - `testdata_CEILNET_table2`：100 张。
  - `real20`：20 张。
  - `objects`：200 张。
  - `postcard`：179 张。
  - `wild`：101 张。
  - `sir2_withgt`：480 张，可作为补充评测。
  - `real_train`：89 张。
  - VOC 处理图像：15287 张，训练列表实际使用 7643 张。
  - DSLR unaligned train：250 张。

注意：当前 worktree 已有模型实现、实验脚本、结果目录和部分 notebook checkpoint 的未提交/未跟踪内容，后续不要误覆盖。

## 3. 总体技术路线

采用“已闭环实验结果为依据，完成最终交付”的路线：

1. 固化 baseline 与 improved 定量结果
   - Baseline 使用课程/ERRNet checkpoint：`checkpoints/errnet/errnet_060_00463920.pt`。
   - Improved 使用已完成的 `ERRNet-R3Lite` 两阶段训练结果。
   - 所有主测试集指标以 `测试结果.md` 和 `workspace/results/checkpoint_sweep_ablation_report.md` 为准。

2. 选择最终 checkpoint
   - 若论文主表强调 PSNR：使用 `aligned60`，即 `checkpoints/errnet_r3lite_scratch/errnet_060_00463920.pt`。
   - 若论文主表强调真实场景结构/视觉质量：使用 `unaligned80`，即 `checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_080_00623560.pt`。
   - 当前推荐最终 improved checkpoint：`unaligned80`；同时在论文中报告 `aligned60` 是平均 PSNR 最优的 R3Lite checkpoint。

3. 方法叙述定位
   - `ERRNet-R3Lite` 作为主要改进方法：ERRNet 主干 + reflection 辅助分支 + learnable residual reconstruction + mixed synthesis。
   - 结果不能表述为“全面超越 baseline”；应表述为“在 postcard、wild/sir2 的结构指标上改善明显，但在 CEILNet synthetic 和 objects 上存在保真度退化”。
   - 后续不再继续做短程全局 loss 权重搜索，因为 `lambda_rec/lambda_r/lambda_excl` ablation 未带来收益。

4. 完成统一可视化和自采数据
   - 同一测试集、同一 resize/预处理、同一指标函数。
   - 输出 CSV/Markdown 表格、样例拼图和论文可用图片。
   - 尽快补齐自采 5 组近似配对图像；若不能严格配对，则只作为定性展示并明确说明。

5. 完成论文与 PPT
   - 论文强调方法动机、轻量改进、训练成本、定量/定性分析和失败案例。
   - PPT 控制为 8-12 页，覆盖背景、baseline、改进、实验、结论。

## 4. 改进方法设计：ERRNet-R3Lite

### 4.1 动机

文献调查见 `workspace/RR_literature_survey.md`。调查后不再采用上一版单纯 residual smoothness 的 `ERRNet-RS`，原因是它缺少对近年工作的直接对应。后续方法更强调三个方向：

- DSRNet：用更一般的 superposition model 和 learnable residue term 捕获简单线性叠加无法表示的残差信息。
- IBCLN：通过 cascaded refinement 和 residual reconstruction 约束逐步改善 transmission/reflection 分离。
- Beyond Linearity、Absorption Effect、Physically-Based Training Images：真实玻璃反射不是简单 `T + blur(R)`，训练合成需要考虑非线性、空间变化、吸收、ghosting、defocus 等因素。

因此最终改进方法命名为 `ERRNet-R3Lite`，即 Reflection branch + Residual Reconstruction + Realistic synthesis 的轻量化版本。

### 4.2 网络输出

baseline ERRNet 只输出：

`T_hat = G(I)`

`ERRNet-R3Lite` 输出：

`T_hat, R_hat = G_r3(I)`

其中 `T_hat` 是最终反射去除结果，`R_hat` 是 auxiliary reflection layer，只用于训练约束、可视化分析和论文解释。

### 4.3 可学习残差重建

增加一个很小的 residual head：

`S_hat = H(I, T_hat, R_hat)`

用它约束输入重建：

`I ≈ T_hat + R_hat + S_hat`

这里 `S_hat` 吸收非线性叠加、玻璃吸收、颜色偏移和 ghosting 等无法由简单 `T+R` 表示的误差。推理时仍只保存 `T_hat`。

### 4.4 损失函数

保留 ERRNet 原有损失：

- aligned：pixel + gradient + VGG + GAN。
- unaligned：VGG/contextual loss。

新增：

1. residual reconstruction loss

   `L_rec = L1(I, clamp(T_hat + R_hat + S_hat)) + L_grad(I, clamp(T_hat + R_hat + S_hat))`

2. reflection auxiliary loss

   `L_r = L1(R_hat, R) + L_grad(R_hat, R)`

   只对 synthetic aligned 数据启用，避免把 real89 中 fake `target_r` 当作真实反射监督。

3. T/R gradient exclusion loss

   `L_excl = mean(sigmoid(|grad(T_hat)|) * sigmoid(|grad(R_hat)|))`

   作用是降低 transmission 和 reflection 的结构泄漏。

最终：

`L_total = L_ERRNet + lambda_rec * L_rec + lambda_r * L_r + lambda_excl * L_excl`

初始超参：

- `lambda_rec = 0.2`
- `lambda_r = 0.1`
- `lambda_excl = 0.01`

已完成短程 loss ablation：

- `lambda_rec=0.10, lambda_r=0.05, lambda_excl=0.005`
- `lambda_rec=0.05, lambda_r=0.05, lambda_excl=0.000`
- `lambda_rec=0.10, lambda_r=0.00, lambda_excl=0.005`

结论：三组短程 continuation 均未超过 `aligned60` reference，平均 PSNR 下降约 `0.11-0.13 dB`，SSIM 下降约 `0.0005-0.0008`。因此后续不再把时间投入到继续微调全局 `lambda_rec/lambda_r/lambda_excl`，而是把它作为报告中的 ablation 结果：单纯全局权重搜索不能解决 fidelity 下降问题。

下一步如继续做方法改进，优先方向改为：

- reflection confidence / mask-weighted loss：用 `abs(I - T)` 或 synthetic `target_r` 生成 soft reflection mask，只在反射区域强化重建/分离约束，在非反射区域保护 transmission 细节。
- curriculum mixed synthesis：先用原 CEILNet 风格合成稳定训练，再逐步引入 mixed synthesis，避免训练初期分布过复杂。

### 4.5 非线性/物理启发合成增强

当前 `ReflectionSythesis_1` 主要是 blur 后叠加。新增 `--synthesis_model`：

- `ceilnet`：原始 `ReflectionSythesis_1`，用于 baseline 可比。
- `perceptual`：使用已有 `ReflectionSythesis_2`。
- `mixed`：训练时随机混合 `ReflectionSythesis_1`、`ReflectionSythesis_2` 和新增轻量 `ReflectionSythesis_3`。

`ReflectionSythesis_3` 近似模拟：

- spatially varying alpha；
- transmission attenuation；
- reflection color shift；
- small shift ghosting；
- blur/defocus 随机化。

### 4.6 代码落点

已完成主要修改：

- `options/errnet/train_options.py`
  - 增加 `--lambda_rec`
  - 增加 `--lambda_r`
  - 增加 `--lambda_excl`
  - 增加 `--synthesis_model`

- `models/losses.py`
  - 复用现有 `compute_gradient` / `GradientLoss`。
  - 增加 `ExclusionLoss`，或先在 model 内部直接计算。

- `models/errnet_model.py`
  - 增加 R3Lite model 或在当前 model 中按 `--r3lite` 分支处理双输出。
  - 在 `backward_G()` 中追加 `Rec`、`R`、`Excl`。
  - 在 `get_current_errors()` 中记录 `Rec`、`RLayer`、`Excl`。

- `models/arch/default.py` 或新增 `models/arch/r3lite.py`
  - 增加 6-channel output 的 ERRNet variant。
  - 增加 residual head `H(I,T_hat,R_hat)`。

- `data/transforms.py`、`data/reflect_dataset.py`
  - 增加 `ReflectionSythesis_3`。
  - 增加 `--synthesis_model mixed` 的选择逻辑。

- 新增实验脚本：
  - `workspace/scripts/eval_r3lite_checkpoints.py`：批量评测 checkpoint 并输出 CSV/Markdown。
  - `workspace/scripts/summarize_eval_outputs.py`：从已保存输出图像恢复指标表。

待补充脚本：

- `workspace/scripts/make_visual_grid.py`：生成 input/baseline/improved/gt 拼图。

## 5. 实验设计

### 5.1 方法对比

最终论文至少比较两组，建议报告三组：

1. Baseline：ERRNet 官方/课程 checkpoint
   - `checkpoints/errnet/errnet_060_00463920.pt`

2. Improved-A：ERRNet-R3Lite aligned scratch
   - 推荐 checkpoint：`checkpoints/errnet_r3lite_scratch/errnet_060_00463920.pt`
   - 平均 PSNR 在 R3Lite sweep 中最高，适合作为 pixel-fidelity 对照。

3. Improved-B：ERRNet-R3Lite unaligned finetuning
   - 推荐 checkpoint：`checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_080_00623560.pt`
   - 平均 SSIM/NCC 最高，且在 `postcard`、`wild`、`sir2_withgt` 更符合真实场景结构改善。
   - 当前建议作为最终 improved 主模型。

补充 ablation：

4. Loss-weight ablation
   - 三组短程 continuation 均未优于 `aligned60`。
   - 作为负结果写入报告，用来说明“继续调全局 loss 权重不是有效改进方向”。

### 5.2 测试集

主表必须包含：

- `ceilnet_table2`
- `real20`
- `objects`
- `postcard`
- `wild`
- 自采 5 张照片

补充表可包含：

- `sir2_withgt`

### 5.3 自采 5 张照片方案

课程要求对自采 5 张照片也给出 PSNR、SSIM、NCC、LMSE。真实社交媒体图片通常没有 GT，无法严谨计算这些指标，因此优先采集“近似配对”的 5 组数据：

- `blended/`：隔着玻璃拍摄，保留反射。
- `transmission_layer/`：同一场景尽量保持机位，打开玻璃门/移除反射源/遮挡反射源后拍摄，作为近似 GT。

目录建议：

```text
datasets/processed_data/my5/
  blended/
    001.png
    ...
  transmission_layer/
    001.png
    ...
```

如果只能获得无 GT 的社交媒体图片，则只做定性可视化，并在论文中说明这些图片不参与 full-reference 指标计算。为了满足课程指标要求，仍建议自己拍摄 5 组可近似配对照片。

### 5.4 统一评测命令

baseline 示例：

```powershell
conda run -n errnet python test_errnet.py --name errnet --dataset ceilnet_table2 -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper
conda run -n errnet python test_errnet.py --name errnet --dataset real20 -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper
conda run -n errnet python test_errnet.py --name errnet --dataset objects -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper
conda run -n errnet python test_errnet.py --name errnet --dataset postcard -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper
conda run -n errnet python test_errnet.py --name errnet --dataset wild -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper
```

improved 示例：

```powershell
conda run -n errnet python test_errnet.py --name errnet_r3lite --dataset ceilnet_table2 -r --icnn_path checkpoints/errnet_r3lite/latest.pt --hyper --inet errnet_r3lite
```

后续统一使用 `workspace/scripts/eval_r3lite_checkpoints.py` 和已有 `test_errnet.py` 生成/复查结果，输出：

```text
workspace/results/metrics_baseline.csv
workspace/results/metrics_errnet_r3lite.csv
workspace/results/metrics_compare.md
workspace/results/checkpoint_sweep/combined_checkpoint_sweep.csv
workspace/results/ablation_loss/eval/checkpoint_sweep.csv
```

## 6. 训练计划

### 6.1 Baseline 复现

状态：已完成 baseline 测试集评测。后续只在需要复查结果或补自采数据时重新运行测试命令。

保留命令：

```powershell
conda run -n errnet python test_errnet.py --name errnet --dataset ceilnet_table2 -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper --result_dir results_baseline
```

不再安排重新训练 baseline。课程项目中 baseline 使用提供 checkpoint，并在报告中说明 checkpoint 来源。

### 6.2 Improved 训练

状态：已完成 `ERRNet-R3Lite` 两阶段训练，不再把继续训练作为主线任务。

已完成训练：

- aligned scratch：

```powershell
python train_errnet.py --name errnet_r3lite_scratch --hyper --inet errnet_r3lite --synthesis_model mixed --lambda_rec 0.2 --lambda_r 0.1 --lambda_excl 0.01 --nEpochs 60 --nThreads 0 --display_id 0 --save_epoch_freq 5 --no-verbose
```

- unaligned finetuning：

```powershell
python train_errnet_unaligned.py --name errnet_r3lite_scratch_unaligned_ft --hyper -r --icnn_path checkpoints/errnet_r3lite_scratch/errnet_latest.pt --inet errnet_r3lite --synthesis_model mixed --unaligned_loss vgg --lambda_rec 0.2 --lambda_r 0.1 --lambda_excl 0.01 --nEpochs 80 --nThreads 0 --display_id 0 --save_epoch_freq 5 --no-verbose
```

最终推荐 checkpoint：

- 主模型：`checkpoints/errnet_r3lite_scratch_unaligned_ft/errnet_080_00623560.pt`
- PSNR 对照：`checkpoints/errnet_r3lite_scratch/errnet_060_00463920.pt`

### 6.3 Loss Ablation

状态：已完成短程诊断，不再继续扩大该方向。

已完成三组：

- `rec=0.10, r=0.05, excl=0.005`
- `rec=0.05, r=0.05, excl=0`
- `rec=0.10, r=0, excl=0.005`

结论：短程 continuation 没有超过 `aligned60`，所以后续不建议继续搜索全局 loss 权重。

### 6.4 后续可选方法改进

如果还要继续做代码级改进，优先级如下：

1. `ReflectionConfidence-R3Lite`
   - 新增 soft reflection confidence map，用合成数据的 `abs(I - T)` 或 `target_r` 监督。
   - 用 mask 加权 `L_rec/L_r/L_excl`，反射区域强约束，非反射区域保护背景。
   - 预计更能解决 `objects/ceilnet_table2` 保真度下降问题。

2. `Curriculum Mixed Synthesis`
   - 训练前半段使用 `ceilnet` synthesis，后半段逐步引入 `mixed`。
   - 目标是缓解当前 mixed 从 epoch 0 开始导致的 synthetic/object fidelity 退化。

3. 轻量 cascaded refinement
   - 增加第二阶段 residual refiner：输入 `[I, T_hat, R_hat]`，输出 `delta_T`。
   - 成本较高，只有在论文/PPT基本完成后再考虑。

## 7. 结果产物规范

建议目录：

```text
workspace/
  RR_project_execution_plan.md
  logs/
  results/
    metrics_baseline.csv
    metrics_errnet_r3lite.csv
    metrics_compare.md
    r3lite_result_analysis.md
    checkpoint_sweep_ablation_report.md
    checkpoint_sweep/
    ablation_loss/
  scripts/
    eval_r3lite_checkpoints.py
    summarize_eval_outputs.py
    make_visual_grid.py
  figures/
    qualitative_ceilnet.png
    qualitative_real20.png
    qualitative_sir2.png
    qualitative_my5.png
  paper/
    DIP_RR_report.md
  slides/
    DIP_RR_presentation_outline.md
```

每次实验记录：

- git commit hash。
- checkpoint path。
- 训练命令。
- 测试命令。
- 数据集版本/路径。
- 每个数据集指标均值。
- 至少 3-5 组可视化样例。

## 8. 论文结构

建议课程论文结构：

1. 摘要
2. 背景与任务定义
   - 单图反射去除问题。
   - ill-posed 难点。
   - 传统先验方法与深度学习方法概述。
3. Baseline：ERRNet
   - VGG hypercolumn。
   - Channel-wise context。
   - Multi-scale spatial context。
   - aligned / unaligned loss。
4. 改进方法：ERRNet-R3Lite
   - 文献调查结论与方法选择依据。
   - Reflection auxiliary branch。
   - Learnable residual reconstruction。
   - 非线性/物理启发合成增强。
   - 损失函数与训练方式。
5. 实验设置
   - 训练集、测试集、自采数据。
   - 指标：PSNR、SSIM、NCC、LMSE。
   - 训练环境与超参。
6. 实验结果
   - 定量表格。
   - 数据集逐项分析。
   - 可视化对比。
   - 失败案例。
7. 结论与感悟
8. 个人贡献说明
9. 代码和权重链接

## 9. PPT 结构

建议 8-12 页：

1. 标题页
2. 任务背景与难点
3. ERRNet baseline
4. 改进方法 ERRNet-R3Lite
5. 数据集与指标
6. 定量结果总表
7. 可视化对比
8. 自采照片结果
9. 失败案例与分析
10. 结论与贡献

## 10. 时间安排

### 2026-06-11

- 已完成执行方案初版。
- 已跑通 baseline 所有测试集，生成第一版指标表。
- 已检查输出图像质量和结果目录。

### 2026-06-12

- 已实现 ERRNet-R3Lite 网络、损失项、参数开关和 mixed synthesis。
- 已完成 smoke test：小数据/短 epoch 确认 loss 正常、checkpoint 可保存。
- 已启动并完成 `errnet_r3lite_scratch` 从头训练。

### 2026-06-13

- 已完成 improved 两阶段训练。
- 已完成 baseline/improved 总表分析。
- 已完成 checkpoint sweep 和 loss ablation。
- 已确定最终 checkpoint 推荐：主模型 `unaligned80`，PSNR 对照 `aligned60`。

### 2026-06-14

- 采集并整理 5 组自采照片。
- 完成自采数据评测与可视化。
- 生成论文用核心图表：
  - baseline vs aligned60 vs unaligned80 定量表。
  - postcard/wild/sir2/objects/ceilnet 代表样例拼图。
  - loss ablation 负结果表。
- 若时间允许，补一个 reflection confidence / mask-weighted loss 的设计小节，作为未来工作或可选扩展，不再默认训练。

### 2026-06-15

- 完成课程论文初稿。
- 完成 PPT 初稿。
- 整理代码、README、权重上传链接。

### 2026-06-16 上午

- 最终复查代码可运行性、论文链接、权重链接、PPT。
- 2026-06-16 17:00 前发送邮件。

## 11. 风险与备选方案

1. GPU 时间不足
   - 当前主要训练已完成，GPU 风险下降。
   - 后续只保留评测、自采图和可视化任务；不再依赖新的长训练。
   - 若新增 reflection confidence 改进来不及，只作为未来工作写入论文。

2. 改进指标不稳定
   - 当前结果已经确认：R3Lite 不全面优于 baseline，但在 `postcard` 和部分真实结构指标上有清晰收益。
   - 论文中不宣称全面 SOTA，只强调数据集相关收益、失败案例和原因分析。
   - 用 checkpoint sweep 解释：`aligned60` 更偏 PSNR，`unaligned80` 更偏 SSIM/NCC 和真实场景适配。
   - 用 loss ablation 说明：简单调全局 loss 权重没有解决 fidelity 退化，后续应做位置感知/加权 loss。

3. 自采照片无法获得严格 GT
   - 优先自己拍摄配对图。
   - 若仍无法严格对齐，指标作为近似参考，同时强调定性对比。

4. 论文/PPT 时间不足
   - 先完成表格和 4 张核心可视化，再补综述。
   - PPT 从论文图表直接复用，避免重复制作。

## 12. 下一步立即执行清单

- [x] 跑 baseline 五个主测试集，保存指标。
- [x] 实现 `errnet_r3lite`、`lambda_rec`、`lambda_r`、`lambda_excl`。
- [x] 实现 `--synthesis_model mixed` 和 `ReflectionSythesis_3`。
- [x] 完成 `errnet_r3lite_scratch` aligned 从头训练。
- [x] 完成 `errnet_r3lite_scratch_unaligned_ft` 第二阶段训练。
- [x] 完成 checkpoint sweep。
- [x] 完成 `lambda_rec/lambda_r/lambda_excl` 短程 ablation。
- [ ] 新增或整理 `my5` 自采数据支持。
- [ ] 生成 baseline/aligned60/unaligned80 对比表和论文图片。
- [ ] 实现或手工生成可视化拼图。
- [ ] 撰写论文初稿。
- [ ] 制作 PPT 初稿。
- [ ] 上传最终权重并整理 README/运行命令。
