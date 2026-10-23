# rul-sim2real

在 NASA C-MAPSS 涡扇发动机退化数据上做**剩余寿命（RUL）点预测 + 可校准区间**，并把"换工况后区间还准不准"当作核心问题。

- 主线：C-MAPSS FD001–FD004，按**发动机单元**划分训练/校准/测试，绝不按行随机切分。
- 点预测：工况条件归一化 + 379 维多尺度滚动特征 → LightGBM；LSTM 与卷积-Transformer（带工况嵌入）作对照/融合。
- 区间：split conformal 与 CQR，用 20% 训练单元校准。
- 压力测试：同一仿真器内的跨工况迁移（FD001 → FD002/FD004），量化"区间静默失效"。
- 交付：可复现的实验脚本 + Streamlit demo + 中文报告。

## 快速开始

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 1. 准备数据

下载 NASA C-MAPSS 并解压到 `data/cmapss`：

```
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
```

解压后目录应为：

```
data/cmapss/
├── train_FD001.txt ... train_FD004.txt
├── test_FD001.txt  ... test_FD004.txt
└── RUL_FD001.txt   ... RUL_FD004.txt
```

路径可用环境变量覆盖：`CMAPSS_DIR`（默认 `data/cmapss`）。数据目录不入库。

### 2. 一键跑通

```bash
python scripts/run_all.py            # 全部：数据校验 → 特征/GBDT → LSTM → 区间 → 域偏移 → 汇总
python scripts/run_all.py --fast     # 只跑 W1+W2（约 1 分钟，用于冒烟）
```

### 3. Demo

```bash
streamlit run app/streamlit_app.py
```

或在 Windows 上直接双击仓库根目录的 `打开demo.bat`：它会检查 Python 与依赖、挑一个空闲端口、启动服务并自动打开浏览器。关闭那个命令行窗口即停止服务（`start_demo.bat` 是旧版，功能相同）。

选一台发动机、拖动"只看前 N 个周期"，看 RUL 点预测与 90% 区间如何随证据变化。

### 4. 测试

```bash
pytest -q        # 40 个测试，含一个真正启动 Streamlit 的端到端测试
```

## 结果（测试单元末周期，RUL 截断 125）

| 模型 | FD001 | FD002 | FD003 | FD004 | 合并 RMSE |
|---|---:|---:|---:|---:|---:|
| ridge（原始特征） | 15.00 | 15.58 | 15.77 | 17.57 | 15.98 |
| gbdt（379 维特征） | 16.20 | 15.61 | 14.15 | 17.34 | 15.87 |
| lstm（30 周期窗口，W3） | 14.25 | 15.28 | 18.21 | 14.92 | 15.66 |
| lstm（30 周期 + TTA，W14） | 13.61 | 14.94 | 17.19 | 15.38 | 15.33 |
| lstm（50 周期 + TTA，W14） | 14.14 | 13.65 | 13.94 | 13.60 | 13.83 |
| 深模型（卷积 + Transformer，3 种子，W11） | 15.28 | 14.61 | 13.68 | 15.51 | 14.77 |
| 深模型 + TTA（单种子，W15） | 14.44 | 15.01 | 15.14 | 16.04 | 15.17 |
| lstm(50) + 深模型（等权，W16） | **13.15** | **13.22** | **13.30** | **13.66** | **13.33** |

"合并 RMSE" 是把四个子集的平方误差按单元数加权后开根号，便于比较一行模型。

**怎么读**：

- 窗口长度是这一轮最大的变量。同一个 LSTM，窗口 30 换成 50，合并 RMSE 从 15.33 降到 13.83，配对 bootstrap（3000 次，按发动机重采样）区间 `[-2.43, -0.63]` 不含 0，这一条是站得住的。
- 窗口在 20% 校准单元上选，不按测试集选；FD001 是例外，校准挑了 50，测试上 30 更好（13.61 vs 14.14），报告里如实写了。
- lstm(50) + 深模型再降 0.50（13.83 → 13.33），但合并 bootstrap 区间 `[-0.88, +0.14]` 跨零，只能算"方向一致、未证明"。
- TTA（把最后窗口平移 −2…+2 周期后平均）对深模型和 LSTM 都有 0.5–1.0 的收益，成本只有推理时间。

**口径修正（W12）**：W2–W10 的旧表用原始 RUL 标签（FD002/FD004 最大 195）算误差，而所有模型的训练目标都截断在 125，等于让模型为"不可能预测的 130–195"背锅，FD002/FD004 被高估约 10 个点。W12 起统一按 `min(RUL, 125)` 评估；W13 另外补了按每个观测周期打分的结果（模型在中段明显差于末端）。

区间（α=0.10，gbdt 点模型）：覆盖率 0.86–0.96，宽度 42–54 周期。

域偏移（FD001 → FD002/FD004）：覆盖率从 0.86 塌到 0.37–0.39，而**区间宽度不变**——这是本项目最重要的结论，详见 `docs/reports/REPORT.md`。

## 目录

```
app/                 Streamlit demo（只调 src.pipeline）
src/
├── config.py        路径与常量（env 可覆盖）
├── data/            cmapss.py（读取/校验）, metropt.py（附录数据集）
├── features/        conditions.py（工况归一化）, rolling.py（379 维特征）, rul.py
├── models/          gbdt.py（含 GroupKFold 选参）, lstm.py, deep.py（卷积+Transformer）
├── intervals/       conformal.py（split + CQR + 分段指标）
├── eval/            metrics.py, plots.py, importance.py
├── pipeline.py      统一推理入口（load_bundle / predict_one）
└── demo.py          demo 辅助（窗口截断、上传校验）
scripts/             w1–w5 实验脚本 + compare_models.py + run_all.py
tests/               40 个测试
artifacts/           模型、归一化器、特征列表、校准参数（演示用，可重建）
reports/             tables/ 与 figures/
docs/                SRS、DESIGN、ADR、周报、REPORT.md
```

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/SRS.md`](docs/SRS.md) | 需求规格：功能/非功能需求、验收硬线 |
| [`docs/DESIGN.md`](docs/DESIGN.md) | 详细设计：模块、函数签名、协议、实现坑（§15–§18 为后续追加） |
| [`docs/reports/REPORT.md`](docs/reports/REPORT.md) | 中文项目报告（结论、负结果、局限） |
| [`docs/reports/W2.md`](docs/reports/W2.md)–[`W5.md`](docs/reports/W5.md) | 每周实验记录 |
| [`docs/adr/`](docs/adr) | 两个架构决策：放弃跨设备、transductive 归一化 |
| [`GLOSSARY.md`](GLOSSARY.md) | 术语表 |

## 可复现性说明

- 所有划分由 `seed=42` 固定；`split_units`、`GroupKFold`、LSTM 初始化均有种子。
- LightGBM 模型以文本形式保存（`model_to_string`），避免中文路径导致的 C++ 写入失败。
- 区间校准参数落盘在 `artifacts/calibration_*.json`，`src.pipeline.load_bundle` 会读取它，demo 与报告用同一套数字。
- 已知偏差：FD001 在开发阶段被用于超参探索与深模型冒烟测试，数字偏乐观；FD002–FD004 未受污染（详见 REPORT §7）。
- 口径：所有 RMSE 按 `min(RUL, 125)` 计算；W2–W10 旧表使用未截断标签，修正说明见 REPORT §5.6。

## 数据许可

C-MAPSS 由 NASA PCoE 发布，可公开用于研究。仓库不附带数据文件。MetroPT-3（UCI 791）为 CC BY 4.0，仅作附录素材，加载器在 `src/data/metropt.py`。
