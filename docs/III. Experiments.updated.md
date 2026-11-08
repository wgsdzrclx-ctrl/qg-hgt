可以。结合参考论文的三段式结构，以及你这篇论文“质量估计 + 证据一致性”的核心贡献，我建议把实验压缩为下面这套会议论文版本。

## III. Experiments

### A. Dataset and Base Methods

这一节只说明实验设置，不新增实验：

- 数据集：TEP extended500，正常状态与 IDV1–IDV20，共 21 类、52 个过程变量。
- 严格按独立 run 划分训练/验证/测试，再生成滑动窗口。
- 窗口长度、步长、训练样本预算和模型超参数统一从 YAML 读取。
- 所有结果使用 seed=0/1/2/3/4，报告均值±标准差。
- 保留四个有代表性的模型：
  - CNN：非图时序基线；
  - GCN：同质图基线；
  - HGT：异构图基线；
  - QC-HGT：本文完整模型。

不再运行 MLP、LSTM、TCN、GraphSAGE、GAT、R-GCN、HAN 等重复性基线。

### B. Performance Comparison

#### 表 I：干净训练—单一退化测试

使用协议一，训练：

| 模型   | 作用               |
| ------ | ------------------ |
| CNN    | 非图模型对照       |
| GCN    | 同质图对照         |
| HGT    | 最重要的异构图对照 |
| QC-HGT | 本文模型           |

表 I 结果（更新于 2026-10-08 21:23，北京时间）：

| Model | Clean | Noise | Missing | Drift | Time Shift | Avg. Robustness |
| ----- | ----- | ----- | ------- | ----- | ---------- | --------------- |
| CNN | 0.575650 ± 0.005809 | 0.572761 ± 0.008081 | 0.570094 ± 0.009042 | 0.567748 ± 0.006249 | 0.545748 ± 0.009070 | 0.565289 ± 0.008120 |
| GCN | 0.145871 ± 0.020364 | 0.136729 ± 0.018268 | 0.136024 ± 0.016435 | 0.145694 ± 0.019838 | 0.146538 ± 0.021698 | 0.140202 ± 0.018413 |
| HGT | 0.434612 ± 0.034280 | 0.407562 ± 0.020493 | 0.422703 ± 0.030850 | 0.431148 ± 0.031674 | 0.433082 ± 0.033978 | 0.423440 ± 0.028756 |
| QC-HGT | 0.449269 ± 0.031141 | 0.412584 ± 0.027095 | 0.429018 ± 0.029936 | 0.442730 ± 0.031039 | 0.450048 ± 0.030031 | 0.432680 ± 0.029339 |

表注：所有数值均为 Macro-F1（0–1），报告 seed=0/1/2/3/4 的均值±样本标准差（ddof=1）。先在每个 seed 内按以下规则汇总，再计算五 seed 的均值和标准差，不直接平均标准差。协议一四个模型的五 seed 均已完成，数据来源为 `E:/Desktop/paper_code/results/stage5/protocol_1/main/{model}_{seed}.json` 中 `complete=true` 的结果。

具体要求：

- 每种退化仍测试全部四档强度；
- Noise、Drift、Time Shift 分别报告其四档强度的平均 Macro-F1；Missing 报告随机缺失四档与连续缺失四档共八个条件的等权平均 Macro-F1；
- `Avg. Robustness` 对 Gaussian Noise、Random Missing、Continuous Missing、Drift、Time Shift 五个子类型等权平均，每个子类型内部四档强度等权；即每个 seed 内对 20 个单一退化条件平均，不含 Clean 或混合退化，也不是归一化 AUC。等价计算为 `(Noise + 2 × Missing + Drift + Time Shift) / 5`；
- 正文主指标只使用 Macro-F1；
- Accuracy、Weighted-F1、每类指标保留在结果文件中，不必都放入正文；
- 可以补充 `ΔF1`，说明模型相对 Clean 的性能下降。

这一部分需要：

$ 4\text{ models}\times5\text{ seeds}=20\text{ training runs} $

不同退化和不同强度只需要推理，不需要重新训练。

#### 表 II：混合退化训练与未知组合泛化

协议二只比较与论文贡献直接相关的模型：

| Model             | Clean | Seen Degradation | Unseen Combination |
| ----------------- | ----- | ---------------- | ------------------ |
| HGT               | 0.416190 ± 0.043063 | 0.413019 ± 0.042967 | 0.412991 ± 0.040243 |
| HGT + Quality     | 0.409087 ± 0.024026 | 0.403986 ± 0.025286 | 0.406407 ± 0.024675 |
| HGT + Consistency | 待补充（3/5 seed 完整） | 待补充（3/5 seed 完整） | 待补充（3/5 seed 完整） |
| QC-HGT            | 待补充（0/5 seed 完整） | 待补充（0/5 seed 完整） | 待补充（0/5 seed 完整） |

表注：更新时点同表 I；Macro-F1 使用 0–1，± 为五 seed 的样本标准差（ddof=1）。Clean、Seen Degradation、Unseen Combination 分别取 `clean`、`protocol_group=known`、`protocol_group=unseen_combination` 的组级 Macro-F1，再在五 seed 间汇总。来源为 `E:/Desktop/paper_code/results/stage5/protocol_2/main/{model}_{seed}.json`。HGT 和 HGT + Quality 的 seed=0–4 已全部完成；HGT + Consistency 的 seed=0/1/2 已完成，seed=3 正在测试，seed=4 尚未完成；QC-HGT 尚无完整结果。不足五个完整 seed 的行暂不计算正式均值±标准差。

设计说明：两表对应新版 III.B 的两个协议及原框架 §5.4 的 Macro-F1；每个 seed 内先汇总条件，再计算五 seed 统计。下文原有实验计划中的历史进度描述不代表本次表格更新时的状态。

这张表同时承担“混合退化性能比较”和“核心消融实验”两个作用：

- HGT：两个模块都没有；
- HGT + Quality：只使用质量模块；
- HGT + Consistency：只使用一致性模块；
- QC-HGT：两个模块协同使用。

因此不再单独运行完整的八行消融表。

这一部分需要：

$ 4\text{ models}\times5\text{ seeds}=20\text{ training runs} $

### C. Robustness and Component Analysis

参考论文第三节使用超参数分析，但你的论文更需要验证两个核心模块。因此建议将其改为 “Robustness and Component Analysis”，并全部复用前面的 checkpoint。

正文只保留一张双子图：

- 图 2(a)：缺失率—Macro-F1 曲线，比较 CNN、GCN、HGT、QC-HGT；
- 图 2(b)：QC-HGT 预测的传感器可靠性 \(r_j\) 随退化强度变化曲线。

另外保留一张紧凑的混淆矩阵图：

- QC-HGT 在 Clean 与 Unseen Mixed 条件下的混淆矩阵；
- 用于分析哪些 TEP 故障类别在退化后容易混淆。

这些图都由表 I、表 II 的结果生成，不新增训练任务。

## 最终计算量

| 实验                                      | 训练任务数 |
| ----------------------------------------- | ---------- |
| 协议一：CNN、GCN、HGT、QC-HGT             | 20         |
| 协议二：HGT、Quality、Consistency、QC-HGT | 20         |
| 曲线和可视化                              | 0          |
| 合计                                      | **40**     |

目前 CNN 的协议一 5 个 seed 已经完成，因此实际剩余约为：

$ 40-5=\mathbf{35}\text{ 个训练任务} $

## 建议删除的实验

会议正文暂时不做：

- Product Fusion 等非核心消融；
- 八行完整消融；
- \(W,\lambda_q,\lambda_c\) 网格敏感性；
- ECE、Brier Score、NLL；
- 检测延迟时间曲线；
- t-SNE/UMAP；
- 多种图层数、隐藏维度分析；
- 所有模型都重复运行协议二。

指标代码可以继续保留，但不需要全部写进论文。

因此，当前 [stage5.yaml](E:/Desktop/paper_code/configs/stage5.yaml) 还应进一步收缩为上述 **40 次训练的会议矩阵**。这比当前配置的 55 次训练减少约 27%，而且保留了两个协议、主要基线以及质量/一致性两个核心贡献的直接证据。当前训练仍保持停止状态。
