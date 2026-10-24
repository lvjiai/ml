# 需求规格说明书（SRS）

**项目**：sim2real-rul —— 涡扇发动机剩余寿命预测与可校准区间
**版本**：v0.2（实现完成后的回填版；v0.1 的跨设备需求已由 ADR 0001 撤销）
**日期**：2026-10-24
**状态**：与实现对齐，待评审

---

## 1. 引言

### 1.1 目的

本文档定义「sim2real-rul」项目的需求边界、功能与非功能需求、验收标准与里程碑。v0.1 写于项目开始前；本版在 W1–W18 全部完成之后回填，**记录实际做了什么、每一条需求如何被验证**，而不是只记录当初打算做什么。两者不一致处以本文档 `实现状态` 列为准，差异在 §9 中列出。

### 1.2 范围

**做（已交付）**：

- 在 NASA C-MAPSS 四个子集上构建剩余寿命（RUL）点预测，覆盖基线、树模型、循环网络与卷积-Transformer；
- 用 split conformal 与 CQR 给出带覆盖保证的预测区间，报告边际与分段覆盖率；
- 在**同一仿真器的跨工况迁移**（E1–E5 五级阶梯）上量化"换工况后模型与区间各坏多少"，并拆出"校准失效"这一独立机制；
- 用三套观测口径（末周期、每周期、60–80% 寿命全量）评估同一批模型，报告排名随口径的翻转；
- 用校准集完成模型选择（W18），报告"守规矩的代价"；
- 交付 Streamlit 单曲线演示与中文报告。

**不做（明确撤销或降级）**：

- ~~在 MetroPT-3 上做跨设备迁移~~：ADR 0001 撤销。跨设备结论容易被"当然迁移不过去"消解，且两台设备的失效语义不可比。MetroPT-3 仅保留加载器、数据侧写与附录素材。
- 不做实时流式推理、云端部署、多设备联合调度；
- 不采集非公开数据、不使用个人数据；
- 不追求 SOTA 名次。

### 1.3 术语

| 术语 | 定义 |
|---|---|
| RUL | Remaining Useful Life，从当前时刻到失效的剩余循环数 |
| 截断点 | 训练与评估中把 RUL 截断在 125 的上限（CAP=125） |
| 校准集 | 从训练集按发动机切出的 20%，只用于 conformal 校准与模型选择 |
| 工况（condition） | C-MAPSS 的运行状态；FD001/FD003 一种，FD002/FD004 六种 |
| 域偏移 | 训练与部署的输入分布不同；本项目指同一仿真器内不同工况 |
| 无监督条件归一化 | 在目标域上只用传感器值拟合 KMeans 工况并做 z-score，不使用标签（transductive） |
| 少样本重校准 | 只用目标域少量有标签发动机重算 conformal 分位数 q，不重训模型 |
| split conformal | 用留出校准集残差分位数构造区间的 conformal 方法 |
| CQR | Conformalized Quantile Regression，先做分位数回归再用 conformal 校准 |
| 覆盖率 | 测试样本中真实值落入预测区间的比例（边际或分段） |
| 卡平台 | 预测 RUL ≥ 100 而真实 RUL ≤ 80 的情形，W17 的计数口径 |
| 选型后悔值 | 用校准集选出的模型在测试集上的 RMSE 减去测试集事后最优模型的 RMSE |

### 1.4 参考

- Saxena et al., *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*
- C-MAPSS 数据包内 `readme.txt` 与 `Damage Propagation Modeling.pdf`
- Romano, Patterson, Candès, *Conformalized Quantile Regression*, NeurIPS 2019
- Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*, 2021
- Lei et al., *Distribution-Free Predictive Inference for Regression*, JASA 2018
- UCI Machine Learning Repository: MetroPT-3（id 791，仅附录使用）
- 本仓库：`docs/adr/0001-domain-shift-within-simulation.md`、`docs/adr/0002-transductive-target-normalisation.md`

---

## 2. 总体描述

### 2.1 产品视角

一个**分析型工程项目**：主线是"模型给出的剩余寿命与区间，在换观测位置、换工况之后还剩多少可信度"。模型是证据，区间与协议是交付物。演示把结论变得可触摸，不是产品。

### 2.2 用户与角色

| 角色 | 关注点 |
|---|---|
| 项目作者（学生） | 理解每一行代码、能讲清方法与结论 |
| 验收者 | 三条硬线是否达标、数字能否复现 |
| 演示观众 | 输入一条曲线，能否得到 RUL 与区间，并知道区间何时不可信 |

### 2.3 运行环境

- 操作系统：Windows 11
- 硬件：仅 CPU（无 GPU），内存 ≥ 16 GB；实测四子集完整训练约 2 小时（含深模型）
- 语言/运行时：Python 3.13
- 依赖（锁定版本见 `requirements.txt`）：pandas 2.3.1、numpy 2.3.2、scikit-learn 1.9.1、lightgbm 4.7.0、torch 2.8.0（CPU 轮子）、streamlit 1.66.1、matplotlib 3.10.5、pytest 9.1.1、joblib 1.6.0
- 网络：仅一次性下载数据；训练与演示全程离线可复现

### 2.4 设计约束

- C1：数据必须可公开获取且许可允许再分发结果（C-MAPSS 为 NASA 公开数据；MetroPT-3 为 CC BY 4.0，仅附录）
- C2：全流程必须能在 CPU 上完成（见 NFR-1）
- C3：不设硬性截止；按周交付可验收增量
- C4：报告为中文、偏学术，需包含方法、实验、讨论与局限
- C5：所有评估数字必须能由 `reports/tables/*.csv` 复现，禁止手写数字

### 2.5 假设与依赖

- A1：C-MAPSS 的官方训练/测试划分与 `RUL_FD00x.txt` 可直接使用
- A2：RUL 截断于 125（CAP），该假设贯穿训练、区间与评估三处，不得只在其中一处生效
- A3：测试单元的轨迹在末周期截断，且只在末周期给出真值标签
- A4：跨域只做同一仿真器内的工况迁移（ADR 0001）

---

## 3. 功能需求

优先级：M = 必须，S = 应当，C = 可选。`实现状态` 列：✅ 已完成并有产物，◐ 部分完成，✗ 未做（并说明）。

### 3.1 数据管道

| ID | 需求 | 优先级 | 实现状态 | 验收方式 |
|---|---|---|---|---|
| FR-1.1 | 下载并校验 C-MAPSS（外层 zip 内含 `CMAPSSData.zip`），校验文件大小与行数 | M | ✅ | 四子集行数 20631 / 53759 / 24720 / 61249；测试 13096 / 33991 / 16596 / 41214 |
| FR-1.2 | 解析 train/test/RUL，统一为长表 `unit, cycle, setting_1..3, s_1..s_21` | M | ✅ | `tests/test_data.py` |
| FR-1.3 | 训练标签 `RUL = max_cycle(unit) − cycle`，截断 125 | M | ✅ | `tests/test_data.py` 与抽样核对 |
| FR-1.4 | 处理常数列/近常数列，列出剔除清单 | S | ✅ | EDA 图 `eda_sensor_std_*.png`；深模型用 14 个传感器 |
| FR-1.5 | 可复现划分（fit 80% / calib 20%，按 unit，固定种子 42） | M | ✅ | `tests/test_split.py`；两次运行一致 |
| FR-1.6 | 数据搬入项目内 `data/`，不依赖机器绝对路径 | M | ✅ | `src/config.py` 默认 `PROJECT_ROOT/data/cmapss` |

### 3.2 建模

| ID | 需求 | 优先级 | 实现状态 | 验收方式 |
|---|---|---|---|---|
| FR-2.1 | 特征：工况归一化 + 多尺度滚动统计（窗口 5/10/20/30 × 均值/标准差/差分/斜率）+ cummax，379 维 | M | ✅ | `reports/tables/w2_features.json`（特征名）、`importance_*.csv` |
| FR-2.2 | 主模型：LightGBM，超参由 fit 内 GroupKFold(3) 选 | M | ✅ | `reports/tables/w2_gbdt.csv` |
| FR-2.3 | 对照模型：LSTM（窗口 30 与 50）、ConvTransformer 深模型 | S | ✅ | `w3_lstm.csv`、`w14_lstm_tta.csv`、`w11_deep.csv`、`w15_deep_tta.csv` |
| FR-2.4 | 基线：常数、仅 cycle 的线性、原始特征 ridge | M | ✅ | `w1_baseline.csv` |
| FR-2.5 | 模型选择只在 fit/calib 内进行；测试集只读一次 | M | ✅（W18 补齐） | `scripts/w18_select.py`，`w18_select.csv` |
| FR-2.6 | 推理入口：加载工件即可预测，不重训 | M | ✅ | `src/pipeline.py`（`load_bundle` / `predict_one`），`tests/test_pipeline.py` |
| FR-2.7 | 推理时对外推（TTA）：平移 −2…+2 取平均 | S | ✅ | `src/models/loaders.py`；W14/W15 收益 0.5–1.0 |

### 3.3 评估

| ID | 需求 | 优先级 | 实现状态 | 验收方式 |
|---|---|---|---|---|
| FR-3.1 | 指标：RMSE、NASA 分、分段 MAE（0–30 / 30–80 / 80+） | M | ✅ | `src/eval/metrics.py` |
| FR-3.2 | 四子集分别报告并给出合并口径 | M | ✅ | `w12_final.csv`、`w16_pooled.csv` |
| FR-3.3 | 误差与真实 RUL、发动机编号的关系分析 | S | ✅ | `reports/figures/`、W17 汇总表 |
| FR-3.4 | 每周期评估（W13） | S | ✅ | `w13_per_cycle.csv` |
| FR-3.5 | 中段全量对照：709 台 × 5 个位置，不抽样（W17） | M | ✅ | `w17_midlife.csv`、`w17_midlife_summary.csv` |
| FR-3.6 | 配对 bootstrap 检验差异是否显著 | M | ✅ | `w16_bootstrap.csv`、`w16_pooled.csv` |
| FR-3.7 | 选型协议：候选只在校准集上比，测试读一次；报告后悔值（W18） | M | ✅ | `w18_select.csv`、`w18_selected.csv` |

### 3.4 不确定性

| ID | 需求 | 优先级 | 实现状态 | 验收方式 |
|---|---|---|---|---|
| FR-4.1 | split conformal（残差分位）与 CQR 双实现 | M | ✅ | `src/intervals/conformal.py`，`tests/test_conformal.py` |
| FR-4.2 | 报告边际覆盖率、平均/中位宽度、分段覆盖率 | M | ✅ | `w4_intervals.csv`、`w4_alpha_sweep.csv` |
| FR-4.3 | α ∈ {0.05, 0.10, 0.20} 敏感性 | S | ✅ | `w4_alpha_sweep.csv` |
| FR-4.4 | 校准与测试严格分离；校准点只取 RUL ≤ 150、目标截断 125 | M | ✅ | `scripts/w4_intervals.py` |
| FR-4.5 | 讨论可交换性假设在时序上的违反 | S | ✅ | REPORT §3.4、§7 |
| FR-4.6 | CQR 与 split 的对比结论必须如实报告（含 CQR 未获胜） | M | ✅ | REPORT §5.2 |

### 3.5 域偏移协议（原"跨域"）

| ID | 需求 | 优先级 | 实现状态 | 验收方式 |
|---|---|---|---|---|
| FR-5.1 | 定义同仿真器跨工况的五级阶梯 E1–E5（同分布 / 朴素迁移 / 无监督条件归一化 / 少样本重校准 / 加噪与传感器丢失） | M | ✅ | ADR 0001，`w5_domain_shift.csv` |
| FR-5.2 | 量化落差：覆盖率、宽度、点 RMSE | M | ✅ | 同上 |
| FR-5.3 | 拆解落差来源：点预测退化 vs 校准失效（宽度不变） | S | ✅ | REPORT §5.3、W5.md §3 |
| FR-5.4 | 少样本重校准的 k 敏感性（k=1,2,3,5,10,20,50） | M | ✅ | `w5_domain_shift.csv`、`w5_fewshot.png` |
| FR-5.5 | transductive 设定必须显式声明 | M | ✅ | ADR 0002 |

### 3.6 演示

| ID | 需求 | 优先级 | 实现状态 | 验收方式 |
|---|---|---|---|---|
| FR-6.1 | Streamlit 单页：选机器 / 拖观测周期 → RUL 点预测 + 90% 区间 | M | ✅ | `tests/test_app.py`（AppTest 真跑） |
| FR-6.2 | 场景名用外行可读的说明，不直接暴露 FD00x | M | ✅ | 场景 1–4 文案（一种/六种运转状态、两种损坏方式） |
| FR-6.3 | 真实值可显示/隐藏；隐藏时曲线仍需完整寿命横轴 | M | ✅ | `test_hidden_truth_keeps_full_life_trajectories` |
| FR-6.4 | 真值 >125 时显示 `≥125`，误差按 125 计 | M | ✅ | 截断语义修复（ef63e2a） |
| FR-6.5 | 上传 CSV 推理，缺列给出明确错误 | M | ✅ | `src/demo.py::coerce_uploaded`，`tests/test_demo.py` |
| FR-6.6 | 区间宽度可解释（页内说明"90% 范围"如何用校准集得到） | S | ✅ | 页内说明块 |
| FR-6.7 | 双击启动脚本，自动查依赖、挑空闲端口、开浏览器 | S | ✅ | `打开demo.bat`，`--check` 输出环境 |
| FR-6.8 | 单次推理 < 2 秒 | S | ✅ | 实测；`test_app.py` 通过 AppTest |

### 3.7 报告与可复现

| ID | 需求 | 优先级 | 实现状态 | 验收方式 |
|---|---|---|---|---|
| FR-7.1 | 中文报告：问题 → 数据 → 方法 → 评估 → 结果 → 负结果 → 工程 → 局限 → 结论 | M | ✅ | `docs/reports/REPORT.md` |
| FR-7.2 | 每个数字都能由脚本重新生成 | M | ✅ | 全部来自 `reports/tables/*.csv` |
| FR-7.3 | 一条命令复现（含 `--fast` 冒烟） | M | ✅ | `scripts/run_all.py`（`--fast` 约 32 秒） |
| FR-7.4 | README 说明环境、数据、运行、目录 | M | ✅ | `README.md` |
| FR-7.5 | 设计文档与 ADR 记录取舍 | M | ✅ | `docs/DESIGN.md`、`docs/adr/0001–0002` |

---

## 4. 非功能需求

| ID | 需求 | 指标 | 实现状态 |
|---|---|---|---|
| NFR-1 | 性能 | 全流程 CPU < 2 小时（四子集含深模型）；`--fast` < 1 分钟；单次推理 < 2 秒 | ✅ 实测约 2 小时 / 32 秒 |
| NFR-2 | 可复现 | 固定种子、依赖锁定、工件落盘 | ✅ `requirements.txt`、`artifacts/` |
| NFR-3 | 可解释 | 特征重要性、误差分段、区间来源说明 | ✅ `importance_*.png`、REPORT §5 |
| NFR-4 | 代码质量 | 类型注解、docstring、单元测试、无硬编码路径 | ✅ 48 个测试；grep 确认无绝对路径 |
| NFR-5 | 可维护 | 模块边界 data / features / models / intervals / eval / app | ✅ `src/` 六个子包 |
| NFR-6 | 文档 | 术语表、ADR、周报 | ✅ `GLOSSARY.md`、`docs/adr/`、`docs/reports/W*.md` |

---

## 5. 数据需求

| 数据集 | 内容 | 规模 | 许可 | 获取 |
|---|---|---|---|---|
| C-MAPSS | 4 子集，709 台训练发动机，160,359 行训练 / 104,897 行测试，26 列 | 12.4 MB（zip） | NASA 公开数据 | 已下载，`data/cmapss` |
| MetroPT-3 | 地铁空压机 17 列时序，1,516,948 行，2020-02-01 → 2020-09-01 | 218 MB | CC BY 4.0 | 已下载，`data/metropt`，仅附录 |

存储布局（实际）：

```
data/
  cmapss/        # train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt
  metropt/       # MetroPT3.csv
artifacts/       # 模型、归一化器、校准参数、预测
reports/
  tables/        # 所有数字的来源
  figures/       # 所有图
```

`data/` 已加入 `.gitignore`；仓库只含代码与文档。

---

## 6. 外部接口

- **训练/评估 CLI**：`scripts/w1_*.py` … `scripts/w18_*.py`，每个脚本可单独执行，参数见 `--help`
- **一键复现**：`python scripts/run_all.py`（`--fast` 只跑 W1 + W2/FD001）
- **推理入口**：`src.pipeline.load_bundle` + `predict_one`（demo 与测试都走这里）
- **Demo**：`打开demo.bat` 或 `python -m streamlit run app/streamlit_app.py`
- **产物**：`reports/tables/*.csv`、`reports/figures/*.png`、`artifacts/`

---

## 7. 验收标准（三条硬线）

1. **演示可用**：任取一条测试曲线（或上传同格式 CSV），界面在 2 秒内给出 RUL 点预测与 90% 区间；真实值超过 125 时按 `≥125` 显示且误差按上限计；错误输入不崩溃。
   - 证据：`tests/test_app.py`、`tests/test_demo.py`、AppTest 实测。
2. **落差可量化**：报告包含同仿真器跨工况的 E1–E5 对比表（覆盖率、宽度、RMSE），并给出至少两条有数据支撑的解释（点预测退化、校准失效）。
   - 证据：`w5_domain_shift.csv`、`w5_fewshot.png`、REPORT §5.3。
3. **一键复现**：干净环境执行 `python scripts/run_all.py`，从原始数据生成报告所需表格；抽查 3 个数字与 REPORT 一致。
   - 证据：`--fast` 冒烟 32 秒；`run_all.py` 失败即停。

补充验收（v0.2 新增）：

4. **选型守规矩**：候选模型只在校准集上比较，测试集只读一次，并报告后悔值。
   - 证据：`w18_select.csv`、`w18_selected.csv`。
5. **中段结论**：任何"模型更好"的结论必须标明观测口径；中段全量对照可复现。
   - 证据：`w17_midlife_summary.csv`。

---

## 8. 里程碑（每周一个可验收交付，实际状态）

| 周 | 交付 | 状态 | 关键产物 |
|---|---|---|---|
| W1 | 数据管道 + EDA + 朴素基线 | ✅ | `w1_baseline.csv`、12 张 EDA 图 |
| W2 | 379 维特征 + LightGBM | ✅ | `w2_gbdt.csv`、`importance_*.png` |
| W3 | LSTM 对照 | ✅ | `w3_lstm.csv` |
| W4 | split conformal + CQR | ✅ | `w4_intervals.csv`、`w4_alpha_sweep.csv` |
| W5 | 域偏移阶梯 E1–E5 | ✅ | `w5_domain_shift.csv`、ADR 0001/0002 |
| W6 | Streamlit demo | ✅ | `app/streamlit_app.py`、`打开demo.bat` |
| W7 | 中文报告 | ✅ | `docs/reports/REPORT.md` |
| W8 | 打包与复现 | ✅ | `README.md`、`run_all.py`、`requirements.txt` |
| W9–W10 | 结构变体与等权组合 | ✅ | `w9_optimize.csv`、`w10_blend.csv` |
| W11–W12 | 深模型 + 口径修正 | ✅ | `w11_deep.csv`、`w12_final.csv` |
| W13–W16 | 每周期、窗口集成、TTA、显著性 | ✅ | `w13`–`w16` 表 |
| W17 | 中段全量对照 | ✅ | `w17_midlife_summary.csv` |
| W18 | 选型协议闭环 | ✅ | `w18_select.csv`、`w18_selected.csv` |

---

## 9. 风险与开放问题

| ID | 风险/问题 | 状态 | 处置 |
|---|---|---|---|
| R1 | 用测试集调参导致数字偏乐观 | 部分残留 | W18 已把选型挪到校准集；FD001 历史污染无法追溯，REPORT §7 已声明 |
| R2 | 跨设备结论不可比 | 已关闭 | ADR 0001 撤销跨设备，改同仿真器阶梯 |
| R3 | conformal 在时间序列上的可交换性 | 已讨论 | REPORT §3.4；用按发动机切分缓解 |
| R4 | 中段"卡平台"（FD004 23% 位置） | 未解决 | 列为后续方向；W17 已量化 |
| R5 | 分区/条件覆盖率不足 | 未解决 | 后续方向：Mondrian by RUL band |
| R6 | 无真实数据 | 未解决 | 报告中作为最大局限声明 |
| R7 | CQR 未胜过 split | 已如实报告 | 负结果写入 REPORT §5.2 |

---

## 10. 技术清单（本项目实际用到的技术）

| 技术 | 用在哪 | 关键参数/实现 |
|---|---|---|
| KMeans 工况聚类 | 工况归一化 | settings 三列聚类；FD001/3 取 1，FD002/4 取 6；`src/features/conditions.py` |
| 按工况 z-score | 特征与深模型输入 | 均值/标准差按工况统计，退化 std 置 1 |
| 多尺度滚动特征 | GBDT 输入 | 窗口 5/10/20/30 × 均值/总体标准差/差分/斜率 + cummax；379 维；`src/features/rolling.py` |
| LightGBM 回归 | 主模型 | `num_leaves` 31 或 63、`min_data_in_leaf` 50 或 20、早停 100、上限 1500 轮；`src/models/gbdt.py` |
| GroupKFold | 选超参/轮数 | 3 折，按发动机分组；`select_model` |
| LSTM | 序列对照 | 21 维输入、hidden 64、窗口 30/50；Huber δ=10；`src/models/lstm.py` |
| 空洞卷积 + Transformer | 深模型 | 1×1 投影、dilation 1/2/4 三层卷积、2 层 4 头 Transformer（d=64，FFN 128，pre-norm）、注意力池化 + 工况嵌入；14 个传感器；`src/models/deep.py` |
| TTA | 推理增强 | 窗口平移 −2…+2 取平均；`src/models/loaders.py` |
| 等权组合 | 集成 | 各模型预测算术平均（不做权重搜索，避免过拟合测试集） |
| split conformal | 区间基线 | 残差分位，k = ⌈(n+1)(1−α)⌉；`src/intervals/conformal.py` |
| CQR | 区间主方法 | 分位数 LightGBM（400 轮）+ conformal 校准；E = max(lo−y, y−hi) |
| 截断（CAP=125） | 标签与区间 | 训练目标、区间上下界、评估真值三处一致 |
| 配对 bootstrap | 显著性 | 按发动机重采样 2000–3000 次；`scripts/w16_blend_final.py`、W17 |
| 少样本重校准 | 域偏移 E4 | 只重算 q，不重训；k ∈ {1,2,3,5,10,20,50} |
| 无监督条件归一化 | 域偏移 E3 | 目标域上 KMeans（transductive，ADR 0002） |
| 特征加噪 / 传感器丢失 | 域偏移 E5 | 噪声 0.1/0.3/0.5；置零 s_4/s_11/s_12/s_7 |
| 每周期评估 | 稳健性 | 失败时间 = 末周期 + RUL；真值 min(F−t,125) |
| 中段全量对照 | 稳健性 | 60/65/70/75/80% 寿命，5 个位置 × 709 台 |
| 校准集选型 | 协议 | 75% 寿命截断，8 候选，选后测试只读一次 |
| Streamlit | 演示 | 单页；AppTest 端到端测试 |
| pytest | 测试 | 48 个用例，覆盖数据、特征、模型、区间、demo、选型 |
| 子进程编排 | 一键复现 | `scripts/run_all.py` 逐步执行，失败即停 |

---

## 11. 附录：目录结构（实际）

```
rul-sim2real/
  app/streamlit_app.py       # 演示
  src/
    data/                     # 读取、校验、划分、截断工具
    features/                 # 工况归一化、滚动特征
    models/                   # 基线、GBDT、LSTM、深模型、加载器
    intervals/                # split conformal、CQR
    eval/                     # 指标、图、重要性
    pipeline.py               # 推理入口
    demo.py                   # demo 的纯逻辑（可测）
  scripts/                    # W1–W18 实验脚本 + run_all
  tests/                      # 48 个测试
  data/                       # 本地数据（gitignore）
  artifacts/                  # 模型与预测
  reports/                    # tables / figures / 周报
  docs/                       # SRS、DESIGN、ADR、REPORT、GLOSSARY
```
