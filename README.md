# TEP 鲁棒故障诊断项目

本项目实现 TEP 数据退化协议、异构过程图、QC-HGT 及阶段 4 全部基线。论文主实验使用 Rieth `extended500`（正常工况与 IDV1–IDV20，共 21 类）；Braatz 标准版仅用于数据加载兼容性验证，二者不会混合。

## 环境与配置

运行环境为 Python 3.10+、PyTorch 2.x、PyG 2.x。安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

数据、模型和训练超参数统一位于 `configs/data.yaml`、`configs/stage2.yaml` 与阶段配置。`stage2.yaml` 的小样本限制仅用于模块自检；阶段 5 使用 `stage5.yaml` 中明确记录的固定窗口预算，以便在保持 run 级无泄漏划分和五随机种子的前提下完成完整实验矩阵。

## 数据与异构图

数据首先按独立运行序列划分 train/val/test，然后才在每条 run 内切滑动窗口，避免窗口级随机划分泄漏。每个窗口生成三类节点：41 个 `sensor`、11 个 `actuator` 和 6 个 `process_unit`。节点特征统一为：

```text
[均值, 标准差, 最小值, 最大值, 线性趋势, 末值, 首末差值]
```

六类关系为 `monitors`、`controls`、`flows_to`、`co_monitors`、`monitored_by` 和 `controlled_by`。完整逐变量映射及中文注释见 `data/tep_hetero_graph.py`。各工艺单元的映射摘要如下。

| 工艺单元 | XMEAS | XMV |
| --- | --- | --- |
| 进料系统 | 1–4 | 1–4 |
| 反应器 | 6–9、21、23–28 | 10 |
| 冷凝器 | 22 | 11 |
| 产品分离器 | 10–14、29–36 | 6–7 |
| 循环压缩机 | 5、20 | 5 |
| 汽提塔 | 15–19、37–41 | 8–9 |

## 阶段 4 基线

所有模型由 `models/baselines.py` 注册，并统一调用：

```python
logits_or_output = model(x_dict, edge_index_dict)
```

普通基线返回 logits；包含质量或一致性辅助损失的变体返回含 `logits` 与 `loss` 的字典。训练器会自动采用对应损失。

| 命令名 | 文件 | 输入与方法 |
| --- | --- | --- |
| `mlp` | `models/mlp.py` | 展平 52×7 统计特征后分类 |
| `cnn` | `models/cnn.py` | 52 个变量作为 Conv1d 通道，时间为卷积轴 |
| `tcn` | `models/tcn.py` | 因果空洞卷积与残差连接 |
| `lstm` | `models/lstm.py` | 单个多变量 LSTM 编码完整窗口 |
| `gcn` | `models/gcn.py` | 六类边压成同一边集后执行 GCN |
| `sage` | `models/sage.py` | 同质图上的 GraphSAGE |
| `gat` | `models/gat.py` | 同质图上的多头 GAT |
| `rgcn` | `models/rgcn.py` | 原生异构关系，每类边拥有独立变换 |
| `hgt` | `models/hgt.py` | PyG 原生 HGTConv |
| `hgt_fixed_quality` | `models/hgt_fixed_quality.py` | Oracle：以 `q*` 直接作为质量可靠性 |
| `hgt_learnable_quality` | `models/hgt_learnable_quality.py` | 可观测质量估计与质量注意力，不含一致性 |
| `hgt_consistency` | `models/hgt_consistency.py` | 一致性注意力与 `L_c`，不含质量项 |
| `qc_hgt` | `models/qc_hgt_full.py` | 可学习质量与证据一致性协同的完整模型 |

选择单个多变量 LSTM，是因为它能在每一时刻联合建模 52 个变量的耦合关系，且参数量显著少于 52 个相互独立的 LSTM。

图模型严格使用相同的七维窗口统计特征。CNN、TCN 与 LSTM 为实现“变量为通道/时间序列编码”，读取生成这些统计量的同一份 W 步原始窗口；它们不使用额外样本、标签或退化参数。这个差异属于时序基线定义本身，应在论文实验设置中明确说明。

### Fixed Quality 的使用边界

`hgt_fixed_quality` 已实现，但不能执行普通验证/测试流程。原因是 `q*` 由人为注入的噪声、缺失、漂移和时间偏移强度计算；在推理阶段读取它会违反实验铁律并导致信息泄漏。因此：

- 该模型只允许作为带明确 Oracle 标记的训练诊断；
- 测试图不携带 `q*`，模型会主动抛出错误；
- 可部署且公平的替代方案是 `hgt_learnable_quality`，它只从原始窗口可观测统计量估计质量。

## QC-HGT 与公式对应

`models/qc_hgt.py` 对应论文框架 4.5–4.8 节：

- 4.5：从缺失率、差分方差、FFT 高频能量、历史均值偏差、连续相同值比例、时间戳延迟计算 `a_j`，学习 `r_j` 和 `L_q`；
- 4.6：基于真实边、错误连接、跨窗口/run 和时间错位样本学习 `c_ij` 与 `L_c`；
- 4.7：将 `λ_r log(r_j+ε)` 与 `γ_r c_ij` 加入关系注意力；
- 4.8：三类节点分别 mean-pool，拼接分类，并优化联合损失。

推理时质量描述特征只由窗口观测数据和观测掩码计算，不读取注入退化强度。模型文字流程图见 `docs/qc_hgt_flow.txt`。

## 训练和五随机种子汇总

单个种子示例：

```powershell
.\.venv\Scripts\python.exe train.py `
  --model hgt `
  --config configs\stage2.yaml `
  --seed 0
```

可选模型名：

```text
mlp, cnn, lstm, tcn, gcn, sage, gat, rgcn, hgt,
hgt_fixed_quality, hgt_learnable_quality, hgt_consistency, qc_hgt
```

执行 seed=0/1/2/3/4 并汇总均值±标准差：

```powershell
.\.venv\Scripts\python.exe scripts\run_five_seeds.py `
  --model hgt `
  --config configs\stage2.yaml
```

单种子结果写入 `results/{model}_{seed}.json`，同时保存最佳验证 checkpoint 和日志；五种子汇总写入 `results/{model}_summary.json`。

## 自检

```powershell
# 阶段 1：退化、质量单调性、run 泄漏与数据格式
.\.venv\Scripts\python.exe scripts\check_data_pipeline.py --config configs\data.yaml --check-all-variants

# 阶段 2：统计特征、异构图、映射和 q* 可见性
.\.venv\Scripts\python.exe scripts\check_stage2.py --config configs\stage2.yaml --check-dataset

# 阶段 3：QC-HGT 前向/反向、消融和推理保护
.\.venv\Scripts\python.exe scripts\check_stage3.py --config configs\stage2.yaml

# 阶段 4：全部基线前向/反向和 Fixed Quality 保护
.\.venv\Scripts\python.exe scripts\check_stage4.py --config configs\stage2.yaml
```

## 文件与目录作用

| 文件或目录 | 作用 |
| --- | --- |
| `data/tep_dataset.py` | 标准版/extended500 加载、run 级划分、窗口索引和泄漏报告 |
| `data/corruptions.py` | 五类退化及训练质量真值 `q*` |
| `data/tep_hetero_graph.py` | 变量映射、七维统计特征、异构图和同质图转换 |
| `data/feature_dataset.py` | 统计特征视图及固定种子调试子集 |
| `models/common.py` | 统一输入适配、图压缩、窗口标准化和分类型池化 |
| `models/mlp.py` | MLP 基线 |
| `models/cnn.py` | 1D-CNN 基线 |
| `models/tcn.py` | TCN 基线 |
| `models/lstm.py` | 多变量 LSTM 基线 |
| `models/gcn.py` | 同质 GCN 基线 |
| `models/sage.py` | 同质 GraphSAGE 基线 |
| `models/gat.py` | 同质 GAT 基线 |
| `models/rgcn.py` | 原生异构 R-GCN 基线 |
| `models/hgt.py` | 原生异构 HGT 基线 |
| `models/hgt_fixed_quality.py` | Fixed Quality Oracle 变体 |
| `models/hgt_learnable_quality.py` | Learnable Quality 变体 |
| `models/hgt_consistency.py` | Consistency 变体 |
| `models/qc_hgt.py` | 本文 QC-HGT 完整模型 |
| `models/qc_hgt_full.py` | 完整模型的统一训练接口适配器 |
| `models/baselines.py` | 模型注册与 YAML 配置路由 |
| `train.py` | 统一训练、联合损失、早停和命令行入口 |
| `eval/runtime.py` | 流式推理、校准与过程监测评估 |
| `eval/metrics.py` | 分类、鲁棒性、校准和检测指标 |
| `eval/summarize.py` | 五种子表格、LaTeX 与敏感性汇总 |
| `eval/visualize.py` | 表示、混淆矩阵、可靠性和检测曲线 |
| `main.py` | 数据、模型、训练、验证、测试和结果写入编排 |
| `utils/` | 随机种子、日志、早停与 checkpoint |
| `configs/data.yaml` | 数据协议、退化档位与质量权重 |
| `configs/stage2.yaml` | 全部模型与训练超参数 |
| `scripts/check_stage4.py` | 阶段 4 全模型真实图批次自检 |
| `scripts/run_five_seeds.py` | 五种子运行及均值±标准差汇总 |
| `docs/qc_hgt_flow.txt` | QC-HGT 文字流程图 |
| `论文框架.md` | 论文问题定义、方法公式与实验协议 |

## 正式实验要求

正式结果必须关闭调试抽样，完整执行 seed=0/1/2/3/4，报告 Accuracy 和 Macro-F1 的均值±标准差，并保留 YAML、日志、checkpoint、单种子 JSON 与汇总 JSON 以便复核。

## 阶段 5：完整协议、汇总与可视化

阶段 5 由 `configs/stage5.yaml` 统一控制。40 次训练的会议论文矩阵在协议一保留 CNN、GCN、HGT 与 QC-HGT，在协议二保留 HGT、Quality Only、Consistency Only 与 QC-HGT；协议二结果同时用于核心消融，避免重复训练。所有实验仍执行 seed=0/1/2/3/4。参数敏感性及非核心结构变体默认关闭。

当前阶段 5 计算预算为每个 seed 固定抽取 train/val/test = 5,000/2,000/2,000 个窗口、最多 30 epoch、早停 patience=5。抽样在独立 run 划分并切窗后进行：先按故障场景平衡选择 run，每条 run 固定抽取 20 个窗口；训练时在约 21 条活动 run 间交错 4 窗口缓存块。不同退化测试场景共享相同 clean run 缓存，但各自动态施加退化。抽样索引由 YAML 固定种子控制并在模型间共享；论文报告中必须披露该预算，不能表述为全窗口普查实验。

```bash
# 可用 PYTHON_BIN 覆盖 Python 解释器路径
bash run_protocol1.sh
bash run_protocol2.sh

# Windows + CUDA 环境：顺序执行精简的两套协议、汇总和绘图
powershell -ExecutionPolicy Bypass -File .\run_stage5_gpu.ps1

# 汇总表格与 LaTeX
python -m eval.summarize --config configs/stage5.yaml

# 默认生成 protocol_2、seed=0 的图；method 在 stage5.yaml 中选择 tsne/umap
python -m eval.visualize --config configs/stage5.yaml --protocol protocol_2 --seed 0

# 指标单元测试
python -m unittest tests.test_metrics
```

运行目录约定：

```text
results/stage5/
├─ protocol_1/{main,ablation,sensitivity}/       # 每模型、每 seed 的 JSON
├─ protocol_2/{main,ablation,sensitivity}/
├─ checkpoints/<protocol>/<experiment_type>/     # 最佳验证 Macro-F1 checkpoint
├─ logs/<protocol>/<experiment_type>/             # 训练及场景评估日志
├─ artifacts/<protocol>/<experiment_type>/        # 可视化抽样与故障时序 NPZ
├─ summary/                                       # CSV、JSON、LaTeX 表格
└─ figures/<protocol>/                            # 论文图件
```

预期汇总输出：

- `main_results.{csv,json,tex}`：论文单一主表；Clean/单退化取协议一，Mixed 取协议二未见组合退化；
- `main_results_protocol_1/2.{csv,json,tex}`：Clean、Noise、Missing、Drift、TimeShift、Mixed 的五种子 Macro-F1；
- `ablation_protocol_2.{csv,tex}`：HGT、仅质量、仅一致性与完整模型；
- `protocol_groups_protocol_2.{csv,tex}`：协议二已见退化与未见组合退化；
- `sensitivity.{csv,json}`：仅在重新启用敏感性实验时生成有效数据；
- `embedding_clean_*`、`embedding_mixed_*`、`confusion_clean_vs_degraded.png`；
- `sensor_reliability_vs_strength.png`、`missing_rate_macro_f1_all_models.png`；
- `fault_probability_detection_delay.png`。

协议运行器采用“一次训练、多场景测试”，同一模型/seed 的 clean 与各退化强度共享 checkpoint。中断后再次执行会依据完整 JSON 或协议隔离的 checkpoint 继续运行，不会因切换测试场景重复训练。
