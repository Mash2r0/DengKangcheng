# 反射去除文献调查与改进方法选择

调查日期：2026-06-11。

目标：根据课程 PPT 和 awesome-reflection-removal 中的 single image methods，选择一个能在当前 ERRNet baseline 上落地、训练成本可控、论文中可解释的改进方法。

## 调查结论

最终选择：`ERRNet-R3Lite`，即在 ERRNet 上加入“Reflection 分支 + learnable residual reconstruction + 非线性/物理启发合成增强”。

不再采用上一版 `ERRNet-RS` 的单纯残差平滑方案。原因是残差平滑虽然来自传统先验，但后续深度方法更强调：

- 预测 transmission 与 reflection 两个分量，并让二者互相约束。
- 用 residual/reconstruction 项弥补简单线性叠加模型的误差。
- 训练数据合成要比简单 `I=T+blur(R)` 更接近真实玻璃反射，包括空间变化、衰减、ghosting、defocus、吸收等因素。
- 反射位置感知有价值，但完整实现需要反射位置标注或大规模真实配对数据，当前课程周期内不适合作为主线。

## 文献方法对比

| 方法 | 核心思想 | 对当前项目的启发 | 是否作为主改进 |
| --- | --- | --- | --- |
| Revisiting Single Image Reflection Removal In the Wild, CVPR 2024 | 构建 RRW 大规模真实配对数据集，提出 MaxRF 描述 reflection location，并设计 location-aware cascaded framework。 | 说明反射位置感知很重要，但需要大规模真实配对数据和位置滤波监督；当前可作为讨论和可选扩展。 | 否，作为论文综述和未来工作。 |
| DSRNet, ICCV 2023 | 提出更一般的 superposition model，引入 learnable residue term；用 dual-stream interaction 分离 transmission/reflection。 | 最适合移植：在 ERRNet 中加入 reflection 辅助输出和 residual reconstruction loss。 | 是，作为主方法核心来源。 |
| Location-aware SIRR, ICCV 2021 | 显式利用反射位置感知信息做反射去除。 | 可启发后续加入 reflection confidence map，但完整模型和训练流程迁移成本较高。 | 否，作为可选扩展。 |
| Absorption Effect, CVPR 2021 | 成像模型考虑玻璃吸收效应：`I = ΩT + ΦR`，两步估计吸收并恢复透射层。 | 可在数据合成中加入 transmission attenuation / color attenuation，提升真实场景泛化。 | 部分采用，作为合成增强。 |
| Trash or Treasure?, NeurIPS 2021 | 交互式 dual-stream，利用用户线索区分 transmission/reflection。 | 思想上支持双流，但课程要求自动评测，不适合依赖交互。 | 否。 |
| Physically-based Training Images, CVPR 2020 | 用物理渲染生成更真实训练图，包含空间变化、ghosting、attenuation、blur/defocus。 | 不做完整渲染管线，但在合成增强中近似模拟这些退化。 | 部分采用，作为合成增强。 |
| IBCLN, CVPR 2020 | cascaded refinement，逐步估计 T/R，使用 residual reconstruction loss。 | residual reconstruction 是低成本且有效的训练约束；完整 LSTM cascade 迁移成本较高。 | 部分采用。 |
| ERRNet, CVPR 2019 | baseline：VGG hypercolumn、channel-wise context、multi-scale spatial context、misaligned training loss。 | 保留主干和训练框架，避免重写项目。 | baseline。 |
| Beyond Linearity, CVPR 2019 | 指出线性叠加模型不足，采用非线性成像/合成思想。 | 直接支持改造当前 `ReflectionSythesis_1`，加入非线性合成策略。 | 部分采用。 |
| Perceptual Losses, CVPR 2018 | 用 perceptual loss 和真实配对数据提升反射分离。 | 当前 ERRNet 已有 VGG loss 和 real89 数据；继续沿用。 | 已由 baseline 覆盖。 |
| CRRN / bidirectional / CEILNet 等早期方法 | 多尺度、并发、双向或边缘指导。 | 主要作为历史综述；部分思想已被 ERRNet/IBCLN/DSRNet 吸收。 | 否。 |

## 最终方法：ERRNet-R3Lite

`R3Lite` 表示 Reflection branch + Residual Reconstruction + Realistic synthesis 的轻量化版本。

### 1. 双层输出

现有 ERRNet 只输出 transmission：

`T_hat = G(I)`

改进后输出：

`T_hat, R_hat = G_r3(I)`

其中 `T_hat` 仍作为最终反射去除结果，`R_hat` 是辅助分支，只在训练和可视化分析中使用。

### 2. 可学习残差重建

基于 DSRNet 的 learnable residue term 与 IBCLN 的 residual reconstruction，增加一个小 CNN：

`S_hat = H(I, T_hat, R_hat)`

约束：

`I ≈ T_hat + R_hat + S_hat`

其中 `S_hat` 用来吸收真实成像中非线性叠加、吸收、颜色偏移和 ghosting 等无法由简单 `T+R` 表示的误差。

推荐损失：

`L_rec = L1(I, clamp(T_hat + R_hat + S_hat)) + L_grad(I, clamp(T_hat + R_hat + S_hat))`

初始权重：`lambda_rec = 0.2`。

### 3. Reflection 辅助监督

合成数据中有 `target_r`，可加入：

`L_r = L1(R_hat, R) + L_grad(R_hat, R)`

只对 synthetic aligned 数据启用。真实 aligned 数据中的 `target_r` 目前是 fake reflection gt，不用于 `L_r`。

初始权重：`lambda_r = 0.1`。

### 4. T/R 梯度排斥

为了减少 transmission 与 reflection 的结构互相泄漏，引入轻量 exclusion loss：

`L_excl = mean(sigmoid(|∇T_hat|) * sigmoid(|∇R_hat|))`

初始权重：`lambda_excl = 0.01`。如果 PSNR 下降或边缘被削弱，优先降到 `0.005`。

### 5. 非线性/物理启发合成增强

当前代码主要使用 `ReflectionSythesis_1`，即较简单的 blur 后叠加。改进训练中增加 `--synthesis_model mixed`：

- 保留 `ReflectionSythesis_1`，保证 baseline 可比。
- 混入已有 `ReflectionSythesis_2`，其包含 gamma、vignetting mask、reflection attenuation。
- 新增轻量 `ReflectionSythesis_3`，近似模拟：
  - spatially varying alpha；
  - transmission attenuation；
  - reflection color shift；
  - small shift ghosting；
  - blur/defocus 强度随机化。

这条路线对应 Beyond Linearity、Absorption Effect、Physically-based Training Images 的共同结论：训练合成分布比简单线性叠加更接近真实玻璃成像。

## 为什么不直接复现某篇完整模型

- DSRNet 完整网络需要迁移独立代码和数据设置，容易偏离“基于 ERRNet baseline 改进”的课程要求；但它的 residual term 和 component synergy 很适合轻量移植。
- IBCLN 完整 cascade/LSTM 重写成本高，且 6 月 16 日前调参风险大；保留 residual reconstruction 更稳。
- 2024 RRW/MaxRF 很强，但需要额外大规模数据下载和 location-aware pipeline，适合写进综述或未来工作，不适合作为当前主交付。
- YTMT 是交互式方法，不适合自动指标评测。

## 参考来源

- CVPR 2024 Revisiting SIRR in the Wild: https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Revisiting_Single_Image_Reflection_Removal_In_the_Wild_CVPR_2024_paper.html
- DSRNet ICCV 2023: https://arxiv.org/abs/2308.10027
- DSRNet code: https://github.com/mingcv/DSRNet
- Location-aware SIRR code: https://github.com/zdlarr/Location-aware-SIRR
- Absorption Effect CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Single_Image_Reflection_Removal_With_Absorption_Effect_CVPR_2021_paper.pdf
- IBCLN CVPR 2020: https://arxiv.org/abs/1911.06634
- IBCLN code: https://github.com/JHL-HUST/IBCLN
- Physically-based Training Images: https://arxiv.org/abs/1904.11934
- Beyond Linearity code: https://github.com/csqiangwen/Single-Image-Reflection-Removal-Beyond-Linearity
- Perceptual Losses code/data: https://github.com/ceciliavision/perceptual-reflection-removal
