# 图像反射去除课程项目执行方案 v0.2

依据文件：`docs/2026-DIP课程项目-RR.pptx`、`README_DIP26.md`、当前 ERRNet 代码仓库、`workspace/RR_literature_survey.md`。

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

注意：当前 worktree 已有 `.gitignore` 和 `datasets/prepare_test_data.py` 的未提交修改，后续不要误覆盖。

## 3. 总体技术路线

采用“先闭环，后改进，再统一产物”的路线：

1. 建立 baseline 闭环
   - 使用课程提供的 ERRNet 预训练权重跑完整测试集。
   - 将 stdout 指标、输出图像、运行参数统一保存，作为论文 baseline 表格。

2. 复现/微调 baseline
   - 优先从 `checkpoints/errnet/errnet_060_00463920.pt` 出发。
   - 如 GPU 时间允许，运行 aligned 训练或短程 finetune；如时间不足，使用提供权重作为复现 baseline，并说明来源和命令。

3. 实现轻量改进方法
   - 不换大模型，不引入扩散/Transformer 等显著增加训练成本的方法。
   - 基于 DSRNet、IBCLN、Beyond Linearity、Absorption Effect 和 Physically-Based Training Images 的调查结果，将最终改进方法确定为 `ERRNet-R3Lite`。
   - 改进方向为：ERRNet 主干 + reflection 辅助分支 + learnable residual reconstruction + 非线性/物理启发合成增强。
   - 本地 RTX 5070 Ti 16G 支持 improved 从头训练；同时保留从 baseline checkpoint 局部加载/finetune 的备选路径。

4. 做统一评测和可视化
   - 同一测试集、同一 resize/预处理、同一指标函数。
   - 输出 CSV/Markdown 表格、样例拼图和论文可用图片。

5. 完成论文与 PPT
   - 论文强调方法动机、轻量改进、训练成本、定量/定性分析。
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

若 PSNR 下降或背景边缘变弱，优先降低 `lambda_excl` 到 `0.005`；若输出仍残留明显反射，优先提高 `lambda_rec` 到 `0.3`。

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

预计修改：

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

- 新增评测/汇总脚本，建议放在 `tools/`：
  - `tools/eval_all.py`：批量跑所有数据集并输出 CSV/Markdown。
  - `tools/make_visual_grid.py`：生成 input/baseline/improved/gt 拼图。

## 5. 实验设计

### 5.1 方法对比

至少比较两组：

1. Baseline：ERRNet 官方/课程 checkpoint
   - `checkpoints/errnet/errnet_060_00463920.pt`

2. Improved：ERRNet-R3Lite finetune
   - 从 baseline checkpoint finetune。
   - 只局部加载兼容层；6-channel output head 和 residual head 随机初始化。
   - 保存至 `checkpoints/errnet_r3lite/`。

可选第三组：

3. Improved-from-scratch：ERRNet-R3Lite 从头训练
   - 利用本地 RTX 5070 Ti 16G 运行完整 aligned 训练，作为更有说服力的改进模型。
   - 训练成本预计与 baseline 同量级，优先安排夜间长任务。
   - 与 baseline checkpoint 和 ERRNet-R3Lite finetune 一起比较，区分“训练策略收益”和“方法收益”。

可选第四组：

4. Baseline finetune：原始 loss 从同一 checkpoint 短程 finetune
   - 用于证明提升来自改进 loss，而不是单纯多训练。
   - 如果时间不足，此项作为 ablation 可省略。

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

后续会用 `tools/eval_all.py` 把这些命令自动化，输出：

```text
workspace/results/metrics_baseline.csv
workspace/results/metrics_errnet_r3lite.csv
workspace/results/metrics_compare.md
```

## 6. 训练计划

### 6.1 Baseline 复现

优先级：

1. 先用预训练权重跑评测，保证指标和 README 表格接近。
2. 若 GPU 时间充足，再执行原始 aligned 训练：

```powershell
conda run -n errnet python train_errnet.py --name errnet_reproduce --hyper
```

3. 若有时间再执行 unaligned finetune：

```powershell
conda run -n errnet python train_errnet_unaligned.py --name errnet_unaligned_ft --hyper -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --unaligned_loss vgg
```

### 6.2 Improved 训练

由于本地有 RTX 5070 Ti 16G，可以把 improved 从头训练列为可选主实验路径；同时保留短程 finetune 作为快速验证和时间兜底。

#### 6.2.1 快速验证：从 baseline checkpoint finetune

```powershell
conda run -n errnet python train_errnet.py --name errnet_r3lite --hyper -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --inet errnet_r3lite --synthesis_model mixed --lambda_rec 0.2 --lambda_r 0.1 --lambda_excl 0.01
```

由于 output head 与 baseline 不完全兼容，加载 checkpoint 时需要支持 partial load：兼容层加载 baseline，新增 head 随机初始化。如果训练脚本仍固定跑 60 epoch，代码实现时需要增加 `--nEpochs` 生效或新增短程 finetune 脚本，建议先跑 10-20 epoch 观察验证集结果。

#### 6.2.2 可选主实验：ERRNet-R3Lite 从头训练

从头训练命令：

```powershell
conda run -n errnet python train_errnet.py --name errnet_r3lite_scratch --hyper --inet errnet_r3lite --synthesis_model mixed --lambda_rec 0.2 --lambda_r 0.1 --lambda_excl 0.01
```

建议训练策略：

- 先运行 debug/smoke test，确认新增损失没有 NaN、显存可承受。
- 再按 baseline 的 60 epoch aligned 训练协议完整训练。
- 若训练时间允许，再从 scratch checkpoint 继续运行 unaligned finetune：

```powershell
conda run -n errnet python train_errnet_unaligned.py --name errnet_r3lite_scratch_unaligned_ft --hyper -r --icnn_path checkpoints/errnet_r3lite_scratch/latest.pt --inet errnet_r3lite --unaligned_loss vgg --synthesis_model mixed --lambda_rec 0.2 --lambda_r 0.1 --lambda_excl 0.01
```

实现时需要确保 `train_errnet_unaligned.py` 也能解析并使用新增 loss 参数；对 unaligned 数据默认启用 `L_rec` 和 `L_excl`，不启用 `L_r`。

建议保存：

- latest checkpoint
- best PSNR checkpoint
- best SSIM checkpoint

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

- 完成执行方案。
- 跑通 baseline 所有测试集，生成第一版指标表。
- 检查输出图像质量和结果目录。

### 2026-06-12

- 实现 ERRNet-R3Lite 网络、损失项、参数开关和 mixed synthesis。
- 完成 smoke test：小数据/短 epoch 确认 loss 正常、checkpoint 可保存。
- 开始短程 finetune；若 smoke test 稳定，夜间启动 `errnet_r3lite_scratch` 从头训练。

### 2026-06-13

- 完成 improved finetune 主要训练。
- 检查 `errnet_r3lite_scratch` 训练进度；如果结果稳定，继续完整训练，否则回退到 finetune 版本作为主结果。
- 跑所有测试集评测。
- 初步比较 baseline 与 improved，必要时调整 `lambda_rec`、`lambda_r`、`lambda_excl`。

### 2026-06-14

- 采集并整理 5 组自采照片。
- 完成自采数据评测与可视化。
- 生成论文用图表。

### 2026-06-15

- 完成课程论文初稿。
- 完成 PPT 初稿。
- 整理代码、README、权重上传链接。

### 2026-06-16 上午

- 最终复查代码可运行性、论文链接、权重链接、PPT。
- 2026-06-16 17:00 前发送邮件。

## 11. 风险与备选方案

1. GPU 时间不足
   - 直接使用课程 baseline checkpoint 做 baseline。
   - Improved 只做 checkpoint finetune 10 epoch。
   - 如果 finetune 来不及，保留代码实现和小规模训练结果，但论文要明确训练预算。
   - RTX 5070 Ti 16G 支持尝试从头训练，但从头训练不是唯一交付路径；若完整 scratch 训练未收敛，使用 finetune 结果提交。

2. 改进指标不稳定
   - 若背景边缘被削弱，降低 `lambda_excl`。
   - 若重建约束过强导致输出偏向输入，降低 `lambda_rec`。
   - 若 reflection 分支不稳定，先关闭或降低 `lambda_r`。
   - 保留“可视化改善但 PSNR 小幅下降”的分析，因为反射去除中视觉质量和 full-reference 指标可能不完全一致。
   - 增加 baseline-finetune ablation，避免把训练轮数差异误当作方法差异。

3. 自采照片无法获得严格 GT
   - 优先自己拍摄配对图。
   - 若仍无法严格对齐，指标作为近似参考，同时强调定性对比。

4. 论文/PPT 时间不足
   - 先完成表格和 4 张核心可视化，再补综述。
   - PPT 从论文图表直接复用，避免重复制作。

## 12. 下一步立即执行清单

- [ ] 跑 baseline 五个主测试集，保存指标。
- [ ] 新增 `my5` 数据集支持。
- [ ] 实现 `errnet_r3lite`、`lambda_rec`、`lambda_r`、`lambda_excl`。
- [ ] 实现 `--synthesis_model mixed` 和 `ReflectionSythesis_3`。
- [ ] 新增批量评测和可视化拼图脚本。
- [ ] 训练 `errnet_r3lite` 短程版本。
- [ ] 启动并跟踪 `errnet_r3lite_scratch` 从头训练。
- [ ] 生成对比表和论文图片。
- [ ] 撰写论文与 PPT。
