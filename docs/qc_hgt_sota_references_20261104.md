# QC-HGT 先进图模型对比候选与复现边界

核验日期：2026-11-04。此表是候选实验设计，不是同一协议下的性能排行榜。本轮未训练这些模型；文献指标不能与本项目结果直接比较。

## 优先复现：任务贴近且有公开代码

| 优先级 | 方法与来源 | 代码核验 | 对比价值与协议差异 |
|---|---|---|---|
| 1 | **TEP-ASTGCN**，Digital Chemical Engineering 2024。[期刊原文](https://www.sciencedirect.com/science/article/pii/S2772508124000206)，DOI 10.1016/j.dche.2024.100158 | 已确认[作者公开仓库](https://github.com/Parth-B7999/TEP-Fault-Detection-Diagnosis-GAE-ASTGCN)，账号主页为作者 Parth Brahmbhatt；仓库含模型和数据准备 notebooks。 | 工艺图结合空间与时间注意力，适合检验原始时序编码的贡献。仓库明确诊断模型为**20类故障**、窗口20、52变量；本项目为**正常+20故障的21类**、窗口30。必须统一类别、窗口与run划分后重新训练。仓库列出的97.63%准确率不能直接作为本项目对比结果。实现成本中等。 |
| 2 | **GNN-TAM**，IEEE Access 2024。[作者论文预印本](https://arxiv.org/abs/2210.11164)，[可读全文](https://arxiv.org/html/2210.11164v1) | [AIRI-Institute/gnn-tam](https://github.com/AIRI-Institute/gnn-tam)明确标为官方实现，含模型、图结构学习、训练与评估代码。 | 可训练邻接矩阵及多图融合，直接检验固定工艺拓扑的局限。注意arXiv v1实验使用**正常+28种故障**的另一TEP版本，窗口100，run长2000、故障起点600；不能沿用其数据协议或文献分数。改接本项目窗口与21类输出，成本低至中。期刊年份2024、预印本首发2022应区别标注。 |

## 最新任务相关候选：先核实全文与复现条件

| 优先级 | 方法与来源 | 代码核验 | 对比价值与协议差异 |
|---|---|---|---|
| 3 | **LGF-MLTG**：Multi-Level Temporal Graph Networks with Local-Global Fusion for Industrial Fault Diagnosis，**2026-04-20 arXiv预印本**。[原文](https://arxiv.org/html/2604.18765v1) | 本轮未找到官方代码，不代表作者一定未公开。 | 每窗口Pearson相关图、LSTM、GraphSAGE、DiffPool及局部/全局融合。全文Sec.4.1使用100个训练run和10个测试run、窗口100、stride1，表1列故障1–20而未列正常类。需要按本项目21类与相同窗口重现。可对照动态图、时序和池化设计，但应明确标为预印本，复现成本中至高。 |
| 4 | **LSTM-GatedTAG**：Text-attributed dynamic graph learning for structure-aware fault diagnosis in chemical processes，Journal of Loss Prevention in the Process Industries，**2026-10-16在线预发表**。[出版商页面](https://www.sciencedirect.com/science/article/pii/S0950423026003049)，DOI 10.1016/j.jlp.2026.106215 | 本轮未确认官方代码；出版商正文直接访问403，仅核验到出版商搜索索引中的摘要和部分正文。 | 21类TEP、共享变量级LSTM、窗口相关图、固定变量描述/所属单元/变量类型与门控融合；摘要指出窗口30、top-k=6。主题与本项目高度接近。摘要报告Macro-F1 0.8894±0.0015、Accuracy 0.8821±0.0012，**仅为外部背景，不能直接比较**。完整采样预算、归一化和标签规则仍需核对；未获得全文及实现前优先作为相关工作，复现成本高。 |

## 可选通用图架构：不称为TEP领域SOTA

| 优先级 | 方法与来源 | 代码核验 | 适配判断 |
|---|---|---|---|
| 5 | **GraphGPS**，NeurIPS 2022。[会议原文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html) | [官方仓库](https://github.com/rampasek/GraphGPS)已确认。 | 局部消息传递、全局注意力、位置/结构编码的成熟强基线。可用节点/边类型编码适配58节点图，并与QC-HGT共享时序编码器；其原论文没有验证本项目TEP协议，不能称为2026最新TEP SOTA。适配成本中。 |
| 6 | **Graph-Mamba**，2024预印本。[原论文](https://arxiv.org/abs/2402.00789) | [bowang-lab/Graph-Mamba官方仓库](https://github.com/bowang-lab/Graph-Mamba)已确认，论文直接给出链接。 | Mamba与节点优先级/置换用于长程图依赖。对58节点、30步窗口，长序列与大图效率优势未必明显，增加实现和环境成本，优先级低于任务专用时空图模型。不是同名高光谱图像GraphMamba。 |

## 不直接混入21类分类主表的研究

- [PHM Europe 2026时空GNN/GATv2](https://papers.phmsociety.org/index.php/phme/article/view/4949)仅用正常数据训练，研究故障检测及结构解释；其检测指标不能与21类分类Accuracy/Macro-F1混比。论文还指出学习注意力并未可靠恢复全部已知物理关系，可支持对图解释的审慎讨论。
- [SSDSTG，PSEP 2026](https://www.sciencedirect.com/science/article/pii/S0957582026001928)为半监督动态图、伪标签与对比学习方案，适合后续低标签实验；当前完全监督主表若引入，需要统一额外无标签数据预算。本轮未确认官方代码。
- LFA-GTF（IEEE TIM 2025，DOI [10.1109/TIM.2025.3612657](https://doi.org/10.1109/TIM.2025.3612657)）检索到候选，但本轮无法读取IEEE原文且未确认官方代码；不依据二手摘要分数下结论。
- MTGNN、Graph WaveNet、GDN等原为预测或异常检测的方法，若改分类头，应明确写成“分类适配版本”，并重新训练评估。

## 建议实验设计

1. **隔离图机制贡献**：用相同7维统计特征比较HGT、QC-HGT、GNN-TAM、GraphGPS。
2. **比较完整系统**：补充Temporal-HGT；使改进QC-HGT与Temporal-HGT、Temporal-GraphGPS共享相同的原始窗口编码器。另加入原生TEP-ASTGCN和一个2026候选。这样能够区分时序信息收益与质量/一致性机制收益。
3. 固定21类、run级划分、正常期标签规则、窗口、训练采样预算、测试退化、调参预算和五随机种子；训练预算扩容应同步给基线。
4. 报告Accuracy、Macro-F1、各类召回、退化前后下降值、参数量和推理时间。论文原分数仅放相关工作，不写入本项目统一实测表。
5. 对同一测试run做配对比较，置信区间按run聚类bootstrap；不要把重叠滑窗当独立样本。先取得实验结果，再判断是否显著提升或超过某个基线。
