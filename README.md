# rul-sim2real

在 NASA C-MAPSS 涡扇发动机退化数据上做**剩余寿命（RUL）点预测 + 可校准区间**，并把"换工况后区间还准不准"当作核心问题。

- 主线：C-MAPSS FD001–FD004，按**发动机单元**划分训练/校准/测试，绝不按行随机切分。
- 点预测：工况条件归一化 + 379 维多尺度滚动特征 → LightGBM 主模型，LSTM 与 Ridge 作对照/融合。
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

或在 Windows 上直接双击仓库根目录的 `start_demo.bat`。启动后浏览器打开 <http://localhost:8501>（这是一个网页界面，不是双击 `.py` 就能看的东西）。

选一台发动机、拖动"只看前 N 个周期"，看 RUL 点预测与 90% 区间如何随证据变化。

### 4. 测试

```bash
pytest -q        # 40 个测试，含一个真正启动 Streamlit 的端到端测试
```

## 结果（测试单元末周期）

| 模型 | FD001 RMSE | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| ridge_raw | 22.44 | 31.89 | 22.57 | 36.04 |
| gbdt_engineered | 17.40 | 27.13 | 15.60 | 28.78 |
| lstm | 15.18 | 26.94 | 19.11 | 26.06 |
| lstm ⊕ blend（等权） | **14.23** | **26.29** | **16.38** | **26.71** |
| gbdt + linear_tree（W9） | 17.25 | 27.73 | 19.26 | 28.66 |
| 分工况专家（W9） | 17.40 | 27.18 | 15.60 | 28.16 |
| 分工况专家 + linear（W9） | 17.25 | 27.75 | 19.26 | **27.59** |
| gbdt + lstm（等权，W10） | 14.91 | **26.28** | 16.36 | 26.77 |

W9/W10 的优化结论是**没有稳定的赢家**：按工况拆专家只在 FD004 有效（28.78→27.59），linear_tree 在 FD003 反而退化（15.60→19.26），等权 `gbdt+lstm` 四子集平均最好（21.08，gbdt 22.23、lstm 21.82）但 FD003 变差。详见 `docs/reports/REPORT.md` §5.5。

区间（α=0.10，gbdt 点模型）：覆盖率 0.86–0.96，宽度 42–54 周期。

域偏移（FD001 → FD002/FD004）：覆盖率从 0.86 塌到 0.37–0.39，而**区间宽度不变**——这是本项目最重要的结论，详见 `docs/reports/REPORT.md`。

## 目录

```
app/                 Streamlit demo（只调 src.pipeline）
src/
├── config.py        路径与常量（env 可覆盖）
├── data/            cmapss.py（读取/校验）, metropt.py（附录数据集）
├── features/        conditions.py（工况归一化）, rolling.py（379 维特征）, rul.py
├── models/          gbdt.py（含 GroupKFold 选参）, lstm.py
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
- 已知偏差：FD001 在开发阶段被用于超参探索，数字偏乐观；FD002–FD004 未受污染（详见 REPORT §7）。

## 数据许可

C-MAPSS 由 NASA PCoE 发布，可公开用于研究。仓库不附带数据文件。MetroPT-3（UCI 791）为 CC BY 4.0，仅作附录素材，加载器在 `src/data/metropt.py`。
