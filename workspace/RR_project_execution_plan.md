# 图像反射去除课程项目执行方案 v0.1

依据文件：`docs/2026-DIP课程项目-RR.pptx`、`README_DIP26.md`、当前 ERRNet 代码仓库。

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
   - 在 ERRNet 的训练目标上加入反射层平滑先验与边缘保持约束，命名为 `ERRNet-RS`（Reflection-Smoothness ERRNet）。
   - 改进只增加少量损失项，可从 ERRNet checkpoint 直接加载并 finetune，符合“禁止使用过多显卡”的限制。

4. 做统一评测和可视化
   - 同一测试集、同一 resize/预处理、同一指标函数。
   - 输出 CSV/Markdown 表格、样例拼图和论文可用图片。

5. 完成论文与 PPT
   - 论文强调方法动机、轻量改进、训练成本、定量/定性分析。
   - PPT 控制为 8-12 页，覆盖背景、baseline、改进、实验、结论。

## 4. 改进方法设计：ERRNet-RS

### 4.1 动机

反射去除可近似看作图像分层：

`I = T + R`

其中 `I` 是输入混合图，`T` 是透射层，`R` 是反射层。传统方法通常假设透射层保留较清晰边缘，而反射层更模糊、更平滑。ERRNet 已使用 VGG hypercolumn、通道注意力、多尺度空间上下文和 misaligned loss；本项目在不增加明显计算量的前提下，将“反射残差应更平滑、透射边缘应保留”的先验加入训练。

### 4.2 损失函数

baseline 对 aligned 数据已有 pixel/VGG/GAN loss，对 unaligned 数据已有 VGG 或 contextual loss。改进方法在此基础上增加：

1. 反射残差平滑损失

   令 `R_hat = clamp(I - T_hat, -1, 1)`，约束 `R_hat` 的梯度：

   `L_ref_smooth = |grad_x(R_hat)| + |grad_y(R_hat)|`

   作用：降低输出中残留的高频反射纹理和局部鬼影。

2. 边缘保持损失

   对 aligned 数据使用目标透射层边缘作为监督：

   `L_edge = |Edge(T_hat) - Edge(T)|`

   作用：防止残差平滑损失把真实背景边缘一起抹掉。

3. 可选颜色一致性损失

   用低权重约束全局颜色漂移：

   `L_color = |mean(T_hat) - mean(T)|`

   只在 aligned 数据启用。

最终：

`L_total = L_ERRNet + lambda_ref_smooth * L_ref_smooth + lambda_edge * L_edge + lambda_color * L_color`

初始超参建议：

- `lambda_ref_smooth = 0.02`
- `lambda_edge = 0.05`
- `lambda_color = 0.01`

若 PSNR 下降或图像过平滑，优先降低 `lambda_ref_smooth` 到 `0.005` 或 `0.01`。

### 4.3 代码落点

预计修改：

- `options/errnet/train_options.py`
  - 增加 `--lambda_ref_smooth`
  - 增加 `--lambda_edge`
  - 增加 `--lambda_color`

- `models/losses.py`
  - 复用现有 `compute_gradient` / `GradientLoss`。
  - 增加 `ResidualSmoothLoss` 或在 model 内部直接计算。

- `models/errnet_model.py`
  - 在 `backward_G()` 中按 aligned/unaligned 状态追加改进损失。
  - 在 `get_current_errors()` 中记录 `RefSmooth`、`Edge`、`Color`。

- 新增评测/汇总脚本，建议放在 `tools/`：
  - `tools/eval_all.py`：批量跑所有数据集并输出 CSV/Markdown。
  - `tools/make_visual_grid.py`：生成 input/baseline/improved/gt 拼图。

## 5. 实验设计

### 5.1 方法对比

至少比较两组：

1. Baseline：ERRNet 官方/课程 checkpoint
   - `checkpoints/errnet/errnet_060_00463920.pt`

2. Improved：ERRNet-RS
   - 从 baseline checkpoint finetune。
   - 保存至 `checkpoints/errnet_rs/`。

可选第三组：

3. Baseline finetune：原始 loss 从同一 checkpoint 短程 finetune
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
conda run -n errnet python test_errnet.py --name errnet_rs --dataset ceilnet_table2 -r --icnn_path checkpoints/errnet_rs/latest.pt --hyper
```

后续会用 `tools/eval_all.py` 把这些命令自动化，输出：

```text
workspace/results/metrics_baseline.csv
workspace/results/metrics_errnet_rs.csv
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

推荐先短程 finetune，而不是从头训练：

```powershell
conda run -n errnet python train_errnet.py --name errnet_rs --hyper -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --lambda_ref_smooth 0.02 --lambda_edge 0.05 --lambda_color 0.01
```

如果训练脚本仍固定跑 60 epoch，代码实现时需要增加 `--nEpochs` 生效或新增短程 finetune 脚本，建议先跑 10-20 epoch 观察验证集结果。

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
    metrics_errnet_rs.csv
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
4. 改进方法：ERRNet-RS
   - 反射残差平滑先验。
   - 边缘保持约束。
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
4. 改进方法 ERRNet-RS
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

- 实现 ERRNet-RS 损失项和参数开关。
- 完成 smoke test：小数据/短 epoch 确认 loss 正常、checkpoint 可保存。
- 开始短程 finetune。

### 2026-06-13

- 完成 improved 主要训练。
- 跑所有测试集评测。
- 初步比较 baseline 与 improved，必要时调整 `lambda_ref_smooth`。

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

2. 改进指标不稳定
   - 降低 `lambda_ref_smooth`。
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
- [ ] 实现 `lambda_ref_smooth`、`lambda_edge`、`lambda_color`。
- [ ] 新增批量评测和可视化拼图脚本。
- [ ] 训练 `errnet_rs` 短程版本。
- [ ] 生成对比表和论文图片。
- [ ] 撰写论文与 PPT。
