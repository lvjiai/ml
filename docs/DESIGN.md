# 详细设计报告

**项目**：sim2real-rul —— 涡扇发动机剩余寿命预测与可校准区间
**版本**：v0.2（实现完成后的回填版，对应 SRS v0.2）
**日期**：2026-10-24
**读者**：实现者（本项目即作者本人）、验收者、答辩评审

> **怎么读这份文档**：§1–§14 是开工前的**初始设计**（保留原样，便于对照"当初怎么想"）；§15–§21 是 W2–W18 的**实施记录**（每个阶段追加，记录实际做法、与初始设计的差异、踩过的坑）；§22–§24 是**演示层、工程化与技术总表**；§25 是**与 v0.1 的差异清单**。两者冲突时，以实施记录与 §25 为准。

---

## 1. 设计目标与原则

本设计要满足 SRS 的三个硬线，因此所有取舍服从三条原则：

1. **可验证优先于先进**：每个模块都能被一个测试或一条命令单独验证；宁可少一个模型，不可多一处"跑通了但不知道为什么"。
2. **落差必须可归因**：跨域实验不是"跑一下看掉多少"，而是把落差拆成特征漂移、标签构造、失效模式三块，每块有对应函数与图表。
3. **区间是一等公民**：点预测与区间走同一条数据流，区间不是事后补的装饰。

对应的技术选型：

| 决策 | 选择 | 理由 |
|---|---|---|
| 主模型 | LightGBM 回归 | 表格特征上强、CPU 快、特征重要性可直接解释；SRS FR-2.2 |
| 对照模型 | LSTM（滑窗） | 提供"时序建模是否值得"的证据；FR-2.3 |
| 区间主方法 | CQR | 分位数回归能随 RUL 自适应变宽，conformal 校准给覆盖保证；FR-4.1 |
| 区间对照 | split conformal | 实现最简单、假设最弱，作为下界参照；FR-4.2 |
| 划分 | 按发动机单元（unit）分组 | 同一台发动机的相邻循环高度相关，按行随机切分等于泄漏；FR-1.5 |
| RUL 标签 | 截断于 125 | 早期循环的"精确 RUL"无信息量，截断是 C-MAPSS 文献惯例，需做敏感性检查（SRS A2） |

---

## 2. 总体架构

```
                    ┌─────────────────────────────────────────┐
                    │              run_all.py                  │
                    │  data → features → train → eval → report │
                    └───────┬─────────┬─────────┬──────────────┘
                            │         │         │
        ┌───────────────────▼──┐  ┌───▼─────────▼───────────────┐
        │  src/data            │  │  src/features               │
        │  loader / validate   │→ │  rul / conditions / rolling │
        └──────────┬───────────┘  └───────────┬─────────────────┘
                   │                          │
        ┌──────────▼───────────┐  ┌───────────▼─────────────────┐
        │ src/models           │  │ src/intervals               │
        │ gbdt / lstm / base   │→ │ split_conformal / cqr       │
        └──────────┬───────────┘  └───────────┬─────────────────┘
                   │                          │
        ┌──────────▼──────────────────────────▼─────────────────┐
        │ src/eval：metrics / plots / cross_domain              │
        └──────────┬────────────────────────────────────────────┘
                   │
        ┌──────────▼───────────┐   ┌───────────────────────────┐
        │ src/app (streamlit)  │   │ reports/{figures,tables}  │
        └──────────────────────┘   └───────────────────────────┘
```

模块依赖是单向的：`app` 只依赖 `models` + `features` 的**推理接口**，不依赖训练代码；`eval` 不反向依赖 `models`，只吃"预测结果表"。这样换模型不需要改评估，符合 NFR-5。

### 2.1 数据流（一次完整运行）

1. `data.loader` 读入 4 个子集，产出统一长表 `DataFrame`（训练集/测试集/RUL 标签）。
2. `features.rul` 生成训练标签；`features.conditions` 做工况归一化；`features.rolling` 生成滑窗特征。
3. 按 unit 划分 fit / calib 两部分（校准集只用于 conformal）。
4. `models` 在 fit 上训练；`intervals` 在 calib 上校准。
5. `eval` 在官方测试集上算指标、画图；`cross_domain` 跑真实域协议。
6. `report` 汇总表格与图，`app` 加载模型产物提供交互。

### 2.2 目录结构

```
src/
├── config.py          # 全局配置（dataclass）
├── data/
│   ├── loader.py      # 下载、解压、解析、schema 校验
│   └── schema.py      # 列名常量、Dataset dataclass
├── features/
│   ├── rul.py         # 标签生成与截断
│   ├── conditions.py  # 工况聚类与归一化
│   ├── rolling.py     # 滑窗统计特征
│   └── windows.py     # 序列窗口构造（LSTM 用）
├── models/
│   ├── base.py        # 基线：线性回归、cycle-only
│   ├── gbdt.py        # LightGBM 点预测
│   └── lstm.py        # LSTM 点预测
├── intervals/
│   ├── split_conformal.py
│   └── cqr.py
├── eval/
│   ├── metrics.py     # RMSE / NASA score / 覆盖率
│   ├── plots.py       # 图
│   └── cross_domain.py# 跨域协议（接口先定，实现待 TBD-1）
└── app/
    └── streamlit_app.py
```

---

## 3. 数据层设计

### 3.1 数据结构

```python
@dataclass(frozen=True)
class Dataset:
    name: str                 # "FD001" … "FD004"
    train: pd.DataFrame       # unit, cycle, setting_1..3, s_1..s_21
    test: pd.DataFrame        # 同上
    rul: np.ndarray           # 测试集每台发动机的真实 RUL（与 test 的 unit 顺序一致）
```

约定：`unit` 为 int，`cycle` 从 1 开始；列顺序固定，便于断言。

### 3.2 函数设计

#### `download_and_extract(dest: Path) -> Path`

- **职责**：若 `dest/cmapssdata` 已存在且四个子集文件齐全则直接返回；否则下载 NASA zip（12,429,152 B），解外层得到 `CMAPSSData.zip`，再解内层到 `dest/cmapssdata`。
- **校验**：下载后比对文件大小；解压后检查 12 个文件（4 子集 × train/test/RUL）齐全。
- **异常**：网络失败抛 `DownloadError`（含 URL 与目标路径）；大小不符抛 `ChecksumError`。
- **幂等**：重复调用不重复下载。

#### `load_subset(name: str, root: Path) -> Dataset`

- **职责**：读取 `train_FD00x.txt` / `test_FD00x.txt` / `RUL_FD00x.txt`。
- **解析**：空白分隔、无表头，按 `schema.COLUMNS`（26 列）赋名；`dtype` 统一为 float32（`unit`/`cycle` 为 int32）。
- **校验**：列数 == 26；无 NaN；`unit` 连续且 `cycle` 在每个 unit 内严格递增；RUL 文件行数 == 测试集 unit 数。
- **已知表**（来自实测，用于回归测试）：train 行数 20631/53759/24720/61249，test 行数 13096/33991/16596/41214，RUL 行数 100/259/100/248。

#### `validate_dataset(ds: Dataset) -> None`

逐条执行上面的断言，失败抛 `DataValidationError(stage, detail)`。它是 W1 的验收物。

---

## 4. 特征层设计

### 4.1 标签：`add_rul(df, clip=125)`

```
RUL_i = max_cycle(unit_i) - cycle_i
RUL_i = min(RUL_i, clip)
```

- 训练集用真实失效点，测试集不用（测试集标签来自 RUL 文件）。
- `clip` 做成参数，W2 跑一次 `clip ∈ {100, 125, 150}` 的敏感性对比（SRS A2）。

### 4.2 工况归一化：`assign_condition(df, n_clusters=6)`

FD002/FD004 有 6 种工况。做法：

1. 对 `(setting_1, setting_2, setting_3)` 做 `StandardScaler`；
2. `KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)` 聚成 6 类，写回 `condition` 列；
3. 对每个 `(subset, condition, sensor)` 计算均值/标准差，做 z-score。

- FD001/FD003 只有 1 种工况，`n_clusters=1`，退化为普通标准化。
- 风险：聚类标签不稳定 → 固定 seed，并在报告中报告各簇样本量。
- 替代方案（备用）：不用聚类，直接把 3 个 setting 作为特征喂给 GBDT，让它自己学。W2 两种都跑，取测试集更好者，记入 ADR。

### 4.3 滑窗特征：`rolling_features(df, sensors, windows=(5, 10, 20, 30))`

对每个 unit 按 cycle 排序后，对每个传感器列生成：

| 特征 | 定义 |
|---|---|
| `{s}_mean_w{w}` | 窗口内均值 |
| `{s}_std_w{w}` | 窗口内标准差 |
| `{s}_slope_w{w}` | 对窗口内 cycle 做一元线性回归的斜率 |
| `{s}_diff_w{w}` | 当前值 − 窗口均值 |
| `{s}_cummax` | 历史最大值（单调退化信号） |

- 窗口不足时 `min_periods=1`，不产生 NaN；但 `std`/`slope` 在窗口 < 2 时置 0 并打标 `short_window=1`。
- 特征数估算：21 传感器 × 4 窗口 × 4 统计 + 21 cummax ≈ 357 列；加 setting 与 condition one-hot，约 370 列。
- **常数传感器剔除**：在训练集上算每列 std，std < 1e-8 的列剔除并记录清单（FD001 中 s_1/s_5/s_10/s_16/s_18/s_19 恒定）。必须在 fit 集上统计，避免用测试集信息。

### 4.4 序列窗口：`make_windows(df, window=30, stride=1)`

- 为 LSTM 构造 `(N, window, n_features)` 张量，标签取每个窗口**最后一个** cycle 的 RUL。
- 只用长度 ≥ window 的尾部窗口；不足 window 的早期记录不进入 LSTM 训练（GBDT 不受影响）。
- 返回 `X, y, unit_ids`，`unit_ids` 用于分组划分与误差归因。

---

## 5. 模型层设计

### 5.1 划分策略

```
train units ──80%──> fit set（训练）
            └─20%──> calib set（conformal 校准）
官方 test units ────> 只用于最终评估（一次）
```

- 按 unit 用 `GroupShuffleSplit`，`random_state=seed`。
- 禁止：用 test 做早停、用 test 选超参、用 test 挑特征。违反即视为流程缺陷（SRS FR-2.5）。

### 5.2 基线：`src/models/base.py`

| 函数 | 说明 |
|---|---|
| `cycle_only_predict(df, max_cycle=125)` | `RUL = clip(max_cycle - cycle)`，不用传感器。检验"传感器到底有没有用" |
| `linear_baseline(X_train, y_train)` | Ridge，标准化后回归 |

### 5.3 LightGBM：`src/models/gbdt.py`

```python
def train_gbdt(X, y, X_val, y_val, params, seed) -> lgb.Booster
def predict_gbdt(model, X) -> np.ndarray
```

默认超参（W2 起调，记录在 ADR）：

```python
{
  "objective": "regression",
  "metric": "rmse",
  "learning_rate": 0.05,
  "num_leaves": 31,
  "min_data_in_leaf": 50,
  "feature_fraction": 0.8,
  "bagging_fraction": 0.8,
  "bagging_freq": 1,
  "num_threads": 8,
  "deterministic": True,
  "seed": 42,
}
```

- 早停：`early_stopping(100)`，验证集 = fit 内再切 10% 的 unit（不是 calib，不是 test）。
- 产物：`artifacts/gbdt_FD00x.txt` + `feature_importance.csv`。

### 5.4 LSTM：`src/models/lstm.py`

```python
class RULNet(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2)
    def forward(self, x)  # x: (B, W, F) -> (B,)
```

- 训练：MSE on clipped RUL，Adam lr=1e-3，batch=256，最多 60 epoch，早停 patience=8（验证 RMSE）。
- 标准化：用 fit 集的 `StandardScaler`，对每个特征维度做，**不用**每台发动机单独归一化（测试时拿不到整条序列的未来）。
- CPU 预算：FD001 约 5–10 分钟；若超时，先减到 1 层 + hidden 32，仍超时则改用 GRU（记 ADR）。

### 5.5 模型对比协议

同一测试集、同一指标表、同一划分。对比表列为：模型 / RMSE / NASA score / MAE(0–30) / MAE(30–80) / MAE(80+)。

---

## 6. 区间层设计

### 6.1 split conformal（对照）

给定点预测 `f` 与校准集 `{(X_i, y_i)}_{i=1..n}`：

```
s_i = |y_i - f(X_i)|
q   = Quantile(s, ceil((n+1)(1-α)) / n)        # 有限样本修正
C(x) = [ f(x) - q, f(x) + q ]
```

```python
class SplitConformal:
    def fit(self, X_cal, y_cal, model) -> Self   # 计算并缓存 q
    def predict(self, X) -> tuple[np.ndarray, np.ndarray]
```

- `q` 用 `np.quantile(scores, level, method="higher")`，等价于取第 ⌈(n+1)(1−α)⌉ 个次序统计量。
- 保证在可交换性下 `P(Y ∈ C(X)) ≥ 1−α`，无需分布假设。

### 6.2 CQR（主方法）

两个分位数模型 `q_lo`（α/2）、`q_hi`（1−α/2），`objective="quantile"`，其余超参与主模型一致。

**训练**：在 fit 集上分别训练两个 LightGBM 分位数回归器。

**校准**（在 calib 集上）：

```
E_i = max( q_lo(X_i) - y_i ,  y_i - q_hi(X_i) )
Q   = Quantile(E, ceil((n+1)(1-α)) / n)
C(x) = [ q_lo(x) - Q ,  q_hi(x) + Q ]
```

```python
class CQR:
    def fit(self, X_fit, y_fit, X_cal, y_cal) -> Self
    def predict(self, X) -> tuple[np.ndarray, np.ndarray]
```

**实现要点**

1. `q_lo`/`q_hi` 分开训练，不能用同一个模型套两套分位数（LightGBM 的 quantile 目标只支持单 α）。
2. 区间下界截断到 0、上界截断到 125：真实 RUL 必在 [0,125] 内，截断不降低覆盖率，只会收窄宽度——这本身是"领域知识换区间质量"的一个可写进报告的论点。
3. 允许 `q_lo(x) > q_hi(x)` 的交叉（校准后仍可能），此时交换两端而不是报错。
4. `Q` 恒非负；若 `Q` 为负说明分位数模型已经系统性覆盖，取 0。

### 6.3 评估：`src/eval/metrics.py`

| 函数 | 返回 |
|---|---|
| `rmse(y, p)` | float |
| `nasa_score(y, p)` | float，`d = p − y`；`d < 0` 时 `exp(−d/13) − 1`，否则 `exp(d/10) − 1` |
| `mae_by_bin(y, p, bins)` | dict，分段 MAE |
| `coverage(y, lo, hi)` | float |
| `mean_width(lo, hi)` | float |
| `coverage_by_bin(y, lo, hi, bins)` | dict，检验"区间在最需要的地方是否也覆盖" |

- 分段默认 `[0, 30, 80, 200]`（W1 实测：测试集 RUL 标签未经截断，FD001/FD003 最大 145，FD002/FD004 最大 194/195；顶格 bins 停在 125 或 150 都会把长寿命样本静默丢掉）。
- 覆盖率报告：边际覆盖率 + 分段覆盖率 + 宽度随 RUL 变化的曲线（FR-4.3）。

### 6.4 校准集大小的敏感性

校准集从 10% / 20% / 30% 三档跑，画"覆盖率 vs 校准集比例"曲线，说明有限样本修正的必要性（FR-4.4）。

---

## 7. 评估与报告层

### 7.1 结果表结构

`reports/tables/main_results.csv`：

```
subset, model, rmse, nasa_score, mae_0_30, mae_30_80, mae_80_125
```

`reports/tables/intervals.csv`：

```
subset, method, alpha, coverage, mean_width, coverage_0_30, coverage_30_80, coverage_80_125
```

### 7.2 图

| 图 | 文件 | 说明 |
|---|---|---|
| 预测 vs 真实散点 | `pred_vs_true_FD00x.png` | 按 unit 着色 |
| 区间宽度随 RUL | `width_vs_rul.png` | CQR vs split |
| 误差分布 | `error_hist.png` | 低估/高估分开 |
| 特征重要性 Top20 | `importance_FD00x.png` | 解释"模型在看什么" |
| 覆盖率 vs 校准集比例 | `coverage_vs_calib.png` | FR-4.4 |

---

## 8. 跨域层设计（§9 协议待定）

接口先定，避免 TBD-1 解决后返工：

```python
class DomainAdapter(Protocol):
    def load(self) -> pd.DataFrame: ...          # 统一成同一 schema
    def make_target(self, df) -> pd.DataFrame: ...  # 生成与 RUL 可比的目标
    def eval(self, model, df) -> dict: ...        # 指标口径与仿真域一致
```

`src/eval/cross_domain.py` 只依赖这个 Protocol，因此两种候选协议（直接 RUL 迁移 / 异常预警提前量）都能插进来：

- **协议 A（RUL 迁移）**：在 MetroPT-3 上构造"从某时刻到故障的剩余时间"目标，用仿真域模型直接推理，比较误差分布与区间覆盖率。
- **协议 B（预警提前量）**：把"在故障前 T 小时发出警报"当任务，比较仿真域阈值策略与真实域重校准策略的提前量/误报率。

代码里用 `--protocol {A,B}` 切换，ADR 定稿后删掉未选中的分支（连同其测试）。

---

## 9. 配置与编排

### 9.1 配置

`src/config.py`：

```python
@dataclass(frozen=True)
class Config:
    data_root: Path = Path("data")
    subsets: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")
    rul_clip: int = 125
    windows: tuple[int, ...] = (5, 10, 20, 30)
    calib_frac: float = 0.2
    alpha: float = 0.1
    seed: int = 42
    n_threads: int = 8
```

- CLI 覆盖：`--subsets FD001 --alpha 0.05`，用 `dataclasses.replace` 生成新配置，不就地改。
- 所有随机性只从 `config.seed` 派生（`numpy` / `torch` / `lightgbm` 各设一次），保证 NFR-2。

### 9.2 `run_all.py`

```
python run_all.py --stage data      # 下载 + 校验，打印行数
python run_all.py --stage features  # 生成特征表，缓存到 data/processed
python run_all.py --stage train     # 训练全部模型 + 校准，产物落 artifacts/
python run_all.py --stage eval      # 生成 reports/tables 与 figures
python run_all.py --stage report    # 拼装 report.md
python run_all.py --stage app       # 启动 streamlit（仅本地）
python run_all.py                   # 全流程
```

- 每个 stage 先检查前置产物，缺失则报错并提示应该先跑哪个 stage，而不是静默重算。
- 中间产物带配置哈希（`features_<hash>.parquet`），配置变了自动失效。

### 9.3 日志与错误

- `logging`，格式 `%(asctime)s %(levelname)s %(name)s: %(message)s`，默认 INFO，`--verbose` 开 DEBUG。
- 自定义异常：`DataValidationError`、`DownloadError`、`NotFittedError`（区间对象未 fit 就 predict）。
- 不允许 `except Exception: pass`；顶层捕获后打印上下文并 `sys.exit(1)`。

---

## 10. 演示层设计

`src/app/streamlit_app.py`：

```python
@st.cache_resource
def load_bundle(subset: str) -> Bundle      # 模型 + 分位数模型 + scaler + 特征列

def predict_curve(df: pd.DataFrame, bundle: Bundle, alpha: float) -> dict:
    # 返回 {"rul": float, "lo": float, "hi": float, "width": float,
    #       "top_features": list[tuple[str, float]]}
```

- 输入：CSV，至少含 `cycle` 与传感器列；列缺失 → `st.error` 列出缺失列；行数 < 窗口长度 → 提示"至少 N 行"。
- 输出：大号 RUL 数字 + 区间条 + 宽度随 α 的小表 + 该曲线的传感器趋势图。
- 缓存：模型只加载一次；推理走单条曲线，响应 < 2 秒（NFR-1）。
- 关键点：**demo 与评估共用 `features.rolling` 的同一份代码**，禁止在 app 里重写特征逻辑（否则演示与报告数字对不上，验收第 3 条会挂）。

---

## 11. 测试策略

| 层 | 测试 | 判据 |
|---|---|---|
| data | `test_loader.py` | 四子集行数等于 §3.2 已知表；schema 列数 26；无 NaN |
| features | `test_features.py` | 窗口特征列数符合预期；常数传感器被剔除；无未来信息泄漏（用未来行扰动，过去特征不变） |
| intervals | `test_conformal.py` | 合成数据（已知分布）上覆盖率 ≥ 1−α−0.03；CQR 在异方差数据上宽度随噪声增大 |
| models | `test_models.py` | 小样本过拟合测试：训练集 RMSE 单调下降；预测形状正确 |
| app | `test_app.py` | 缺列/短序列/NaN 输入给出错误而非崩溃 |

**泄漏测试**是重点：构造一个"把第 t 行改成极大值"的用例，断言第 t 行之前的所有特征不变——这是分组时序数据最容易犯的错。

---

## 12. 性能预算

| 阶段 | 预算（8 核 CPU） |
|---|---|
| 数据解析 | < 10 s |
| 特征生成（4 子集） | < 3 min |
| LightGBM（4 子集 × 2 分位数 + 点预测） | < 10 min |
| LSTM（FD001 + FD002） | < 20 min |
| 评估与作图 | < 2 min |
| 单次 demo 推理 | < 2 s |

超预算的处理顺序：减窗口数 → 减传感器 → LSTM 降级 GRU/减层 → 最后才考虑抽样。

---

## 13. 与 SRS 的追溯

| SRS 需求 | 设计落点 |
|---|---|
| FR-1.1–1.3 数据与标签 | §3.2 `load_subset`、§4.1 `add_rul` |
| FR-1.5 可复现划分 | §5.1 分组划分、§9.1 种子 |
| FR-2.1–2.4 特征与模型 | §4.3、§5.2–5.4 |
| FR-2.5 测试集只用一次 | §5.1 禁令 + `run_all` 流程 |
| FR-3.1–3.3 评估 | §6.3、§7.1 |
| FR-4.1–4.5 区间 | §6.1–6.4 |
| FR-5.x 跨域 | §8（Protocol 接口 + 两套协议） |
| FR-6.x demo | §10 |
| FR-7.x 报告与复现 | §7、§9.2 |
| NFR-1 性能 | §12 |
| NFR-3 可解释 | §7.2 特征重要性图、§10 top_features |
| NFR-5 可维护 | §2 单向依赖 |

---

## 14. 开放问题

1. **TBD-1**：跨域协议 A/B 未定，§8 的接口先按双协议设计，定稿后裁剪。
2. 工况聚类 vs setting 直接入模，W2 用测试集比一次后定，写 ADR。（W2 实际只实现了"按工况归一化"这一条，未做对照，此项保留。）
3. LSTM 在 CPU 上的实际耗时未经实测，W3 首日先跑通一条 FD001 再决定是否上 FD002。
4. 是否加"非对称 conformal"（对低估更严格）作为第三个区间方法——取决于报告篇幅，暂列 C 级。

---

## 15. 实施记录（W2）

### 15.1 与 §4.3 的差异

- 实际实现为 `src/features/rolling.py:add_rolling_features`，全向量化（`cumsum` + `groupby.shift`），无逐单元循环；21 传感器 × 4 窗口 × 4 统计量 + `cummax` + 原始值 + settings + condition = **379 维**。
- 列名规范为 `{sensor}_mean_w{w}` / `_std_w{w}` / `_diff_w{w}` / `_trend_w{w}` / `_cummax`，不是 §4.3 里的简写。
- `trend` 定义为窗口首尾差 /(n−1)，窗口未满时按已有行数计算（不是 NaN）。
- `std` 为总体标准差（ddof=0），由原始一阶/二阶矩计算，避免逐窗口 `rolling().std()` 的开销。

### 15.2 特征因果性

`tests/test_features.py::test_rolling_features_are_causal` 把某单元轨迹截断到第 80 个周期后重算，逐列比对第 80 行特征，确保任何特征都不使用未来信息。这是 FR-2.5 与测试期截断轨迹的前提。

### 15.3 模型选择协议（替代 §5.5 的"固定 500 轮"）

- 划分：`fit`（80% 单元）/`calib`（20% 单元），W4 的 conformal 复用同一 `calib`。
- 选择：在 `fit` 内部做 `GroupKFold(3)`，网格 `{num_leaves:31,min_data_in_leaf:50}` 与 `{63, 20}`，早停 100 轮、上限 1500 轮；取各折 RMSE 均值最低者。
- 实测：四个子集都选中 `31/50`，轮数 73 / 112 / 119 / 72。
- **注意**：CV 的 RMSE 是在整条训练轨迹所有行上算的（大量行处于 RUL=125 平台，很容易），与测试指标（仅末周期）不可比，只用于选型，不能当作性能估计。

### 15.4 W2 实测（测试单元末周期，RMSE / NASA score）

| 子集 | W1 ridge_raw | GBDT+特征 | Ridge+特征 | 50/50 融合 |
|---|---|---|---|---|
| FD001 | 22.44 / 1290 | 17.40 / 498 | 16.34 / 375 | **15.81 / 353** |
| FD002 | 31.89 / 2900 | 27.13 / 9518 | 27.80 / 8400 | **26.90 / 8086** |
| FD003 | 22.57 / 1715 | **15.60 / 776** | 17.42 / 570 | 15.59 / 536 |
| FD004 | 36.04 / 15647 | 28.78 / 5686 | 29.53 / 6450 | **28.65 / 5681** |

结论：

1. 特征工程带来 20–30% 的 RMSE 下降（对比 W1 的 `ridge_raw`），且四个子集一致。
2. **GBDT 与线性模型互有胜负**：FD001/FD002 上 Ridge 略好（线性外推 + 末端样本稀疏时更稳），FD003/FD004 上 GBDT 明显好。50/50 融合在四个子集上都不差于两者中较弱者，FD001 达到 15.81。
3. FD002/FD004 仍是 27–29，难点在 6 工况混合 + RUL 分布更宽（测试标签最高 194/195）+ 高 RUL 区间样本少——这是 W3（LSTM）与 W5（跨域）需要回答的。
4. 特征重要性：滑窗**均值**主导（s_4、s_11、s_3、s_17 的 `mean_w5/w10`），`cummax` 次之；这与 EDA 中"退化表现为缓慢漂移、瞬时值噪声大"一致。
5. Ridge 训练时出现的 `LinAlgWarning: ill-conditioned` 来自 379 维强共线特征，不影响结果，未处理。

### 15.5 工件

- 结果表：`reports/tables/w2_results.csv`
- 预测明细：`artifacts/preds/w2_preds_{FD00x}.csv`
- 模型与归一化器：`artifacts/gbdt_{FD00x}.txt`、`artifacts/norm_{FD00x}.joblib`、`artifacts/features_{FD00x}.json`
- 重要性：`artifacts/importance_{FD00x}.csv`、`reports/figures/importance_{FD00x}.png`

### 15.6 已知实现坑

- LightGBM 的 C++ 层无法写入含非 ASCII 字符的路径（`Model file ... is not available for writes`），因此模型文本用 `booster.model_to_string()` + Python `write_text` 落盘。
- 评估函数的分段 bins 已按实测改为 `(0, 30, 80, 200)`（测试集 RUL 未截断）。

### 15.7 测试集接触记录（诚实性备注）

W2 开发过程中，为诊断"为什么 GBDT 跑不过 Ridge"，曾在 **FD001 测试集**上探索性比较过：不同 `num_leaves/min_data_in_leaf`、`linear_tree`、训练子集（全量 / rul<125 / rul<100 / 末 120 周期）、GBDT/Ridge 融合权重。这些比较**没有**进入最终选型（最终选型改为 fit 单元内的 GroupKFold），但 FD001 的测试集因此不再是"只看过一次"。

影响与处理：

- FD001 的 W2 数字（15.81 / 17.40 / 16.34）可能略有乐观偏差，应在报告中标注为"开发期已接触"。
- FD002–FD004 的测试集在选型阶段未被查看，是干净的主结果。
- 若需要严格版，可在最终报告前对 FD001 用冻结的超参重跑一次，并只引用那一次的数字。

---

## 16. 实施记录（W3）

### 16.1 模型与序列构造

- `src/models/lstm.py:RULNet`：`LSTM(21→64)` → `Linear(64→32)` → `ReLU` → `Linear(32→1)`，输入为 30×21 的窗口。
- `build_sequences`：每个单元逐周期切片，窗口 30、步长 1，标签取窗口末周期的截断 RUL；窗口完全因果。
- `last_window`：测试路径，每台发动机取最后 30 个周期；不足 30 的（FD002 有 6 台）在左端复制首个读数补齐，**不丢样本**。
- `train_lstm`：Huber(δ=10)，Adam(1e-3)，batch 128，最多 40 epoch，patience 6，保留验证 RMSE 最优权重。
- 输入只有归一化传感器，**settings 未入模**；这是 FD002/FD004 的已知改进空间。

### 16.2 协议

与 W2 共用 `fit/calib`（80/20 单元）；在 `fit` 内再切 15% 单元做早停验证；测试集只在最后报告时用一次（末周期）。

### 16.3 实测（测试末周期，RMSE / NASA score）

| 子集 | GBDT+特征 | Ridge+特征 | 50/50 融合 | LSTM | LSTM⊕融合 |
|---|---|---|---|---|---|
| FD001 | 17.40 / 498 | 16.34 / 375 | 15.81 / 353 | **15.18 / 415** | 14.23 / 295 |
| FD002 | 27.13 / 9518 | 27.80 / 8400 | 26.90 / 8086 | **26.94 / 7968** | 26.29 / 7347 |
| FD003 | **15.60 / 776** | 17.42 / 570 | 15.59 / 536 | 19.11 / 1618 | 16.38 / 827 |
| FD004 | 28.78 / 5686 | 29.53 / 6450 | 28.65 / 5681 | **26.06 / 4475** | 26.71 / 4556 |

### 16.4 结论

1. 验收线（FD001 < 15.8）达成：LSTM 15.18，LSTM⊕融合 14.23。
2. 序列信息在 FD001/FD004 最有用，FD002 与融合打平，**FD003 退化到 19.11**。
3. FD003 的退化定位为**验证集太小**（fit 仅 68 台训练、12 台验证，可能漏掉两种故障模式之一），早停 epoch 8 时验证 RMSE 14.44、测试 19.11，是典型的小验证集早停偏差，不是 LSTM 本身不行。
4. 等权融合（LSTM + `blend_50`，未用测试集调权重）在 3/4 子集不差于最好的单模型。
5. CPU 成本：FD001 32 s、FD002 44 s、FD003 35 s、FD004 103 s。

### 16.5 工件

`artifacts/lstm_{FD00x}.pt`、`artifacts/lstm_{FD00x}.json`（含逐 epoch 验证 RMSE）、`artifacts/preds/w3_preds_{FD00x}.csv`、`reports/tables/w3_results.csv`、`reports/tables/w3_ensemble.csv`。

### 16.6 待改进

- 早停验证按故障模式分层，或改 5 折 CV 选 epoch。
- settings 入模、窗口长度（20/30/50）消融。
- FD001 测试集污染说明见 §15.7。

---

## 17. 实施记录（W4）

### 17.1 区间方法与实现

- `src/intervals/conformal.py`：
  - `conformal_quantile(scores, alpha)`：取第 `ceil((n+1)(1−alpha))` 小的分数（有限样本保证，直接用次序统计量，不用 `np.quantile` 的线性位置，避免偏一位）。
  - `SplitConformal`：对点模型绝对残差做校准，`[p−q, p+q]`。
  - `CQR`：训练 5%/95% 分位回归器，校准 `E=max(lo−y, y−hi)` 的 conformal 分位数，输出 `[lo−Q, hi+Q]`。
  - `interval_metrics`：边际覆盖率、宽度、按真实 RUL 分段的覆盖率与宽度。
- `scripts/w4_intervals.py`：加载 W2 工件（`gbdt_*.txt`、`norm_*.joblib`、`features_*.json`）与 W3 的 `lstm_*.pt`，在 `calib` 单元上按截断点采样校准，测试集末周期评估；α∈{0.05,0.1,0.2}。

### 17.2 两个关键决定

1. **校准点取"截断点"而不是末周期**：测试单元是截断轨迹，末周期处标签恒为 0，无法用来校准误差分布。改为从每台校准单元第 30 周期起、每 10 周期取一个截断点，标签 = 到失效的周期数（≤125），只保留剩余寿命 ≤150 的点（FD001 得 291 个校准点，FD002 761 个）。
2. **目标量截断在 125**：点模型训练标签就截断在 125，超过 125 只能说"至少还有 125 个周期"；因此覆盖率的评估目标是 `min(RUL,125)`，W2/W3 的 RMSE 用未截断标签，两者分栏引用。

### 17.3 实测（α=0.10，测试末周期）

| 子集 | 方法 | 覆盖率 | 平均宽度 | 0–30 段覆盖率 |
|---|---|---|---|---|
| FD001 | gbdt_split | 0.86 | 42.3 | 1.00 |
| FD001 | gbdt_cqr | 0.88 | 49.3 | 0.96 |
| FD002 | gbdt_split | 0.95 | 54.1 | 1.00 |
| FD002 | gbdt_cqr | 0.94 | 51.9 | 1.00 |
| FD003 | gbdt_split | 0.96 | 47.5 | 1.00 |
| FD003 | gbdt_cqr | 0.95 | 55.1 | 1.00 |
| FD004 | gbdt_split | 0.91 | 50.0 | 1.00 |
| FD004 | gbdt_cqr | 0.90 | 50.1 | 0.94 |

α 扫描（四子集平均）：0.05→覆盖 0.954/0.962、0.10→0.920/0.916、0.20→0.835/0.812（split/CQR）。

### 17.4 结论与负结果

- 区间覆盖率贴近名义水平（0.05→0.95，0.10→0.92，0.20→0.83），说明 conformal 在本任务的分布假设下工作正常。
- **CQR 没有在宽度上赢过 split conformal**（每个 α 宽 3–6 周期）：低 RUL 段点模型已很准（0–30 段 MAE 7.4），固定宽度够用；分位模型在 400 轮、379 维、291–761 个校准点下低估不确定性，校准后 Q 反而更大。按 DESIGN §6.2 的预期，结论记为"负结果 + 归因"，不剪裁掉。
- FD001 coverage 0.86（n=100，SE≈3%）在 0.90±2σ 内，不算异常。

### 17.5 工件

`reports/tables/w4_intervals.csv`、`w4_alpha_sweep.csv`、`reports/figures/w4_intervals_FD001.png`、`artifacts/preds/w4_preds_{FD00x}.csv`；`tests/test_conformal.py` 6 个测试，全套 23 passed。

### 17.6 待改进

- 校准分布与测试分布对齐（基准测试集偏后段截断）：当前用"剩余寿命 ≤150"的窗口缓解，未完全解决。
- CQR 分位模型的超参（轮数、leaves）未搜索。
- 目标截断 125 是评估约定，需在最终报告的"指标口径"里单独说明。

---

## 18. 实施记录（W5）

### 18.1 范围修订：只做仿真内部的域偏移

原计划是 C-MAPSS → MetroPT-3 的跨设备迁移。用户否掉了跨设备（ADR 0001），理由充分：涡扇发动机仿真与地铁空压机只有压力/温度/负载的语义相通，跨过去测出的差距既归因不清、也无法复现。现行范围：**只在 C-MAPSS 内部做阶梯式域偏移**，每一步只加一个变量，落差可归因。

MetroPT-3 的处理：`src/data/metropt.py` 与 `tests/test_metropt.py` 保留（loader、4 次漏气事件窗口、10 s 重采样），`reports/w5_metropt_profile.md` 作为附录素材；不再进入主线。

### 18.2 五级阶梯

`scripts/w5_domain_shift.py` 定义并跑通 E1–E5：

| 阶段 | 变量 | 目的 |
|---|---|---|
| E1 | 无 | 同分布参照 |
| E2 | 目标域变了，适配为零 | 测"朴素迁移"的崩坏幅度 |
| E3 | + 目标域无监督条件归一化 | 测"只靠无标签适配"能救回多少 |
| E4 | + k 台目标域有标签单元重校准 `q` | 测"少样本重校准"的代价 |
| E5 | + 特征噪声 / 传感器丢失 | 测输入退化下的鲁棒性 |

关键实现选择：

- **便携特征**：去掉 `setting_1..3` 与 `condition` 后的 378 维。settings 的取值集合在域间不同，带上它们等于把"工况身份"当特征。
- **E3 是 transductive 的**：工况归一化只在目标域训练集的传感器上做 KMeans，不需要标签，但用到了测试前的目标域数据——ADR 0002 明确声明，不允许隐式使用。
- **E4 的 `q` 只在 k 台单元上重算**，模型权重不动；所以 E3/E4 的 RMSE 相同，差异全部来自区间。
- E5 的 dropout 固定去掉 s_4、s_11、s_12、s_7（W2 重要性图里前四），是对"最有用的传感器坏了"的最坏情况测试。

### 18.3 实测（α=0.10）

完整表见 `reports/tables/w5_domain_shift.csv`；摘录：

| 阶段 | FD002 覆盖/宽/RMSE | FD004 覆盖/宽/RMSE |
|---|---|---|
| E1 | 0.860 / 48.2 / 16.20 | 0.860 / 48.2 / 16.20 |
| E2 | 0.390 / 48.2 / 40.00 | 0.367 / 48.2 / 39.36 |
| E3 | 0.672 / 48.2 / 26.50 | 0.673 / 48.2 / 24.07 |
| E4 k=10 | 0.876 / 92.9 / 26.50 | 0.968 / 94.0 / 24.07 |
| E5 噪声 0.3 | 0.649 / 48.2 / 26.85 | 0.657 / 48.2 / 26.10 |
| E5 丢传感器 | 0.548 / 48.2 / 28.62 | 0.504 / 48.2 / 32.08 |

### 18.4 结论

- 目标域一变、适配为零时，名义 90% 的区间实际只覆盖 37–39%，且**没有变宽**——共形区间在协变量偏移下会"过度自信"，这是前提失效，不是实现 bug。
- 无监督条件归一化救回约一半（0.67），剩下的靠少样本重校准；把覆盖率拉回 0.88–0.97 的代价是宽度 1.7–2.4 倍，且随 k 非单调（k 小时的 `q` 由抽到的单元决定）。
- 输入退化（噪声、丢关键传感器）会让覆盖率再掉 0.02–0.35，丢传感器时低 RUL 段最惨。

### 18.5 工件

`reports/tables/w5_domain_shift.csv`、`reports/figures/w5_fewshot.png`、`docs/reports/W5.md`；`docs/adr/0001-domain-shift-within-simulation.md`、`0002-transductive-target-normalisation.md`；`GLOSSARY.md` 收录 10 个术语。

### 18.6 待改进

- 只有 FD001→FD002/FD004 两个方向，单 seed，E4 的 k 抽样噪声未做置信区间。
- E5 的噪声是特征上的高斯噪声，不是传感器物理退化模型。
- 未测"目标域有少量标签就重训模型"的上限，E3/E4 的 RMSE 不代表目标域可达上限。

## 19. 实施记录（W11–W12）：深模型与评估口径修正

### 19.1 动机

W1–W10 的结论是：树模型与 LSTM 在单工况上够用，FD002/FD004 仍偏高，且两类模型的误差结构不同（树模型末端准、序列模型中段稳）。同时发现点预测表与区间表的口径不一致：区间实验用 `min(RUL,125)`，点预测表用原始标签（最高 195）。

### 19.2 口径修正（W12）

- 所有点预测指标改为 `min(RUL, 125)`，与训练目标、W4 区间口径统一。
- 影响：FD001 17.40 -> 16.20、FD002 27.13 -> 15.61、FD003 15.60 -> 14.15、FD004 28.78 -> 17.34。
- `scripts/w12_compare.py` 生成 `reports/tables/w12_final.csv`，是唯一口径统一的对照表；旧表保留在各周报，不再作为结论依据。

### 19.3 深模型（W11）

- `src/models/deep.py`：14 个传感器（去 7 个近似常量）-> 工况内标准化 -> 1x1 投影 -> 三层空洞卷积（d=64, dilation 1/2/4）-> 2 层 Transformer 编码器（4 头, FFN 128）-> 注意力池化 -> 拼接可学习工况嵌入 -> MLP -> RUL。
- 训练：窗口 30、Huber delta=10、AdamW lr=1e-3、余弦退火、batch 256、30 轮上限、fit 内 15% 单元早停、3 个种子平均。
- `scripts/w11_deep.py` 复用 W2 的 fit 80% / calib 20% 划分与 `ConditionNormalizer`（只在 fit 单元上拟合），测试集只在最后预测一次。
- 结果（末周期 RMSE）：15.28 / 14.61 / 13.68 / 15.51；等权组合 `deep+gbdt+lstm` 13.64 / 13.74 / 13.93 / 14.87，四子集平均 14.04。

### 19.4 诚实性

- FD001 在 5 轮冒烟阶段被看过一次，FD001 数字偏乐观；FD002–FD004 未受影响。
- 组合权重为事先等权，未在测试集上调权；"逐子集取最优组合"仅用于报告。
- 深模型尚未接入 demo 与区间，后续需要做。

### 19.5 工件

`artifacts/preds/w11_preds_FD00x.csv`、`reports/tables/w11_deep.csv`、`reports/tables/w12_final.csv`、`reports/w11_run.log`、`tests/test_deep.py`。

## 20. 实施记录（W13–W16）：每周期评估、窗口集成与 TTA

### 20.1 动机

§19 之后剩下的两个问题：① 所有数字都只取末周期，模型在中段是否同样成立；② 窗口长度和推理扰动从来没被当成变量，而 §19 的负结果表明"加容量"不解决问题。W13–W16 按顺序回答，规则仍是"校准选型、测试只看一次"。

### 20.2 每周期评估（W13，`scripts/w13_per_cycle.py`）

- 失败时间 = 测试轨迹最后观测周期 + `RUL_FD00x`；第 t 周期真值 `min(F - t, 125)`，与训练目标、区间口径一致。
- GBDT 的 379 维特征全部因果（滚动窗口与 cummax 只看过去），可直接逐行预测；LSTM 用 stride=1 滑窗。
- 输出 `late`（最后 30 周期）与 `all`（全部周期）两个视图。结论：模型在中段明显更差；`all` 视图数字更低是早期目标恒为 125 造成的假象。NASA 分在该口径下被早期误差放大到数万，故本阶段只报 RMSE。

### 20.3 窗口集成与 TTA（W14，`scripts/w14_lstm_tta.py`）

- 窗口 {20, 30, 50} 各训一个 LSTM；TTA 把最后窗口平移 −2…+2 周期后平均（等价于对"最近若干个可能起点"做集成）。
- 选型：在 20% 校准单元上按 stride 20 取截断样本打分（`calib_windows`），选窗口与组合方式；测试集只预测一次。
- 结果（test / calib RMSE）：FD001 17.45/18.03、13.61/15.44、14.14/12.80，集成 13.68；FD002 15.95、14.94、13.65/14.76，集成 13.44；FD003 21.91、17.19、13.94/13.92，集成 15.80；FD004 16.51、15.38、13.60/15.42，集成 14.12。
- FD001 是"校准选型失败"的样本：calib 选 50（12.80 vs 15.44），test 上 30 更好（13.61 vs 14.14）。如实保留，不做二次选择。
- 实现坑：训练时漏了 `norm.transform(fit)`（训练原始量纲 / 测试归一化），RMSE 82–86；`calib_windows` 定义在 `__main__` 之后导致 NameError；`add_rul` 未生成导致 KeyError。

### 20.4 深模型 TTA（W15，`scripts/w15_deep_tta.py`）

- 复用 §19 的 ConvTransformer，单种子训练，权重存 `artifacts/deep_{subset}.pt`（配 `.json` 存结构参数），TTA 与 W14 相同。
- 结果：14.44 / 15.01 / 15.14 / 16.04。与 W11 三种子（15.28/14.61/13.68/15.51）的差异说明单种子方差有 0.5–1.5 个 RMSE，**不能拿单次训练当模型能力**。

### 20.5 最终对比与显著性（W16，`scripts/w16_blend_final.py`）

- 读 W2（gbdt）、W14（逐窗口与集成）、W15（深模型 TTA）预测，按 unit 对齐后做等权组合。
- 配对 bootstrap：3000 次，按发动机重采样（同一台机器的多周期保持成组），同时给逐子集区间与**跨四子集合并**区间（合并 = 把四个子集的平方误差按单元数汇总后开根号）。
- 最终表：gbdt 15.87；w30 15.33；w50 13.83；deep 15.17；w50+deep **13.33**（逐子集 13.15/13.22/13.30/13.66）。
- 合并 bootstrap：w30→w50 −1.50 [−2.43,−0.63] 显著；w50→w50+deep −0.50 [−0.88,+0.14] 不显著；gbdt 比 w50 差 +2.03 显著；deep 单模比 w50 差 +1.33 显著。
- 结论：本轮唯一可宣称的改进是"窗口 50 + TTA"；组合是候选而非结论；模型容量的边际收益低于窗口/推理策略。

### 20.6 诚实性

- FD001 在 W11 冒烟与 W14 选型中各被看过一次（后者是 test 上发现 calib 选错，属于事后观察，已声明）。
- 深模型 TTA 是单种子，与 W11 三种子不是同一口径，文中分列不混用。
- 深模型与 w50 LSTM 均未接入 demo 与区间；demo 仍是 GBDT + CQR。
- "严格选型验证"（calib 上多组划分重复）未做，留作后续。

### 20.7 工件

`scripts/w13_per_cycle.py`、`w14_lstm_tta.py`、`w15_deep_tta.py`、`w16_blend_final.py`；`reports/tables/w13_per_cycle.csv`、`w14_lstm_tta.csv`、`w15_deep_tta.csv`、`w16_final.csv`、`w16_bootstrap.csv`、`w16_pooled.csv`；`artifacts/preds/w14_preds_w*_FD00x.csv`、`w15_preds_FD00x.csv`、`artifacts/deep_FD00x.pt/.json`。


## 21. 实施记录（W18）：选型协议闭环

### 21.1 动机

W2–W16 的所有模型对比都在测试集上读，违反了"测试集只看一次"的协议。W18 补齐：候选模型只在校准单元的 20% 上比较，选中之后测试集才被读一次。

### 21.2 实现

- `src/models/loaders.py`：把已训练模型从工件重建为可预测对象——
  - `load_norm` / `deep_normalizer`：GBDT 与 LSTM 用 `norm_FD00x.joblib`（全训练集拟合），深模型用 80% fit 单元拟合的独立归一化；
  - `load_lstm`（窗口 30 / 50，隐藏层从 state_dict 推出）、`predict_lstm`（TTA 平移 −2…+2）；
  - `load_deep`（读 `deep_FD00x.json` 恢复结构）、`predict_deep`（同样 TTA）。
- `scripts/w18_select.py`：每台校准发动机取 75% 寿命处截断（真值 `min(剩余,125)`），8 个候选（4 个单模 + 4 个等权组合）在同一校准集上算 RMSE，取最小者；随后在测试集上只读选中的那一行。
- 输出 `reports/tables/w18_select.csv`（每子集 × 候选的 calib/test RMSE）与 `w18_selected.csv`（选中、事后最优、后悔值、Spearman）。

### 21.3 结果

选中模型平均测试 RMSE 13.65（事后最优 13.24，单独 GBDT 15.87）；后悔值 FD001 0.83、FD002 0.18、FD003 0.64、FD004 0.00；校准-测试排序 Spearman 0.41–0.81。

### 21.4 结论与诚实性

- 校准选型可用，代价（后悔值）小于选错模型的代价；但 FD001 只有 20 个校准单元，排序不稳，需要多组划分重复才能报方差。
- 深模型在任何子集都没被选中；FD002 选中 `w50+deep`，与 W16 的"不显著"并不矛盾（前者比平均误差，后者检验差值区间）。
- W18 不能修复 FD001 历史上在测试集上调参的污染；它只给出"若全程守规矩会选什么"的参考。

---

## 22. 演示层实现（W6）

### 22.1 页面结构

单页 Streamlit，左侧栏三个输入、主区两图四卡，底部三个展开块：

- 侧栏：场景选择（radio，显示"场景 1：一种运转状态"而不是 FD001）、机器编号、观测周期滑杆、真实值开关（显示/不显示）。
- 主区：指标卡四张（剩余周期数、90% 范围、范围宽度、真实值）；图 1 为 4 条关键传感器读数（按窗口内 z-score）；图 2 为"预测值随观测周期增长"的曲线 + 真值斜线 + 当前截止竖线。
- 展开块：页面说明、这个模型在什么情况下不能用、上传传感器数据（CSV）。

### 22.2 数据流

```
load_subset(name)  →  with_targets(train, 125)  →  observable_window(unit, cutoff)
                                                   →  pipeline.predict_one
                                                   →  指标卡 / 图 2
```

- `app/streamlit_app.py::get_dataset` 永远返回**完整寿命**的训练轨迹（带 rul_true）。真值是否显示由页面决定，不在数据层截断。
- `src/demo.py::observable_window` 只做一件事：把轨迹截到 `cycle ≤ cutoff`，并保留 `rul_true`（若有）。它是纯函数，可单测。
- `rul_curve` 在 18 个递增截止点上重复调用 `predict_one`，得到"观测越长、预测越准"的轨迹；缓存键含场景与机器，避免重复推理。

### 22.3 三个关键决定

1. **真值可见性与数据截断解耦**。曾经的 bug：`show_truth=False` 时返回 `ds.test`（本就截断），图 2 的横轴只到截断点，曲线塌成一个点。修复后隐藏真值只影响覆盖层，横轴始终是完整寿命。
2. **截断语义在界面上明说**。模型上限 125；真值 176 时显示 `≥125`，误差按 125 计，图 2 的 >125 区域涂灰。否则用户会看到"误差 −54.9"这种由口径造成的假误差。
3. **场景名映射**。`SCENARIOS` 把 FD001–FD004 映射为"一种/六种运转状态、两种损坏方式"的外行描述，FD 编号只在展开块的"使用限制"里出现。

### 22.4 启动器

`打开demo.bat`（ASCII only，避免 cmd 的 GBK 解析）：

- 依次探测 `python`、`py -3`；检查 `streamlit/lightgbm/sklearn/pandas/numpy/torch` 是否可导入；
- 端口从 8501 起探测，占用则顺延到 8502、8503；
- 后台 powershell 延迟 6 秒开浏览器，前台前台跑 streamlit；
- `--check` 只做环境检查并打印 `OK: python` / `PORT: 8502`；`DEMO_NO_BROWSER=1` 跳过开浏览器。

### 22.5 测试

- `tests/test_demo.py`（9 个）：`observable_window` 截断、`settings_defaults` 中位数、`coerce_uploaded` 的缺列/缺设置/缺 cycle/空表。
- `tests/test_app.py`（4 个，AppTest 真跑）：页面能渲染、点预测落在区间内、值域 0–125、隐藏真值时横轴仍是完整寿命。

---

## 23. 工程化与复现（W7–W8）

### 23.1 报告数字溯源

REPORT 的每一张表都标注 `reports/tables/*.csv` 来源；`scripts/run_all.py` 按 W1 → W5 → compare 的顺序重跑，生成同样的表。摘要里的数字（如"合并 RMSE 13.33"）由 `w16_pooled.csv` 直接读出，不手写。

### 23.2 编排

`scripts/run_all.py`：

- 子进程逐步执行 `w1_data_baseline.py`、`w2_features_gbdt.py`、`w3_lstm.py`、`w4_intervals.py`、`w5_domain_shift.py`、`compare_models.py`；
- 任一步非零退出即停，打印失败阶段；
- `--fast` 只跑 W1 与 W2/FD001（约 32 秒），用于冒烟。

### 23.3 依赖与数据布局

- `requirements.txt` 锁定版本（含 torch 的 CPU 轮子安装说明）；
- 数据从机器绝对路径搬到项目内：`data/cmapss`、`data/metropt`；`src/config.py` 默认按 `PROJECT_ROOT` 解析，并支持 `RUL_DATA_DIR` / `METROPT_CSV` 覆盖；
- `data/` 加入 `.gitignore`；仓库只含代码、文档与小型工件（`artifacts/` 里的模型文本与预测）。

### 23.4 版本管理

Git 仓库，远程 `https://github.com/lvjiai/ml`；每个工作包一次提交，提交信息带阶段前缀（`W2:`、`W6:`、`W18:`）。没有 CI：测试靠本地 `pytest`（48 个）。

### 23.5 测试清单

| 文件 | 覆盖 |
|---|---|
| `test_data.py` | 解析、行数、标签截断、常数列 |
| `test_split.py` | 按 unit 切分、比例、种子可复现 |
| `test_features.py` | 滚动特征因果性、工况归一化 |
| `test_metrics.py` | RMSE、NASA、分段 MAE、边界 |
| `test_conformal.py` | 有限样本分位、边际覆盖、CQR 单调性、截断 |
| `test_lstm.py` | 窗口形状、补齐、前向与梯度 |
| `test_deep.py` | 窗口形状、截断目标、前向与梯度 |
| `test_pipeline.py` | 加载工件、单条推理、区间 |
| `test_demo.py` / `test_app.py` | demo 纯逻辑、页面端到端 |
| `test_select.py` | 截断工具、标签上限、非法比例 |

### 23.6 已知工程债

- 无 CI、无 lint/format 配置、未打包成 wheel；
- 深模型只有单种子工件，没有多组划分重复；
- demo 单页无鉴权（仅本地使用，不部署）。

---

## 24. 技术总表（设计与实现对照）

| 技术 | 实现位置 | 输入 → 输出 | 复杂度/成本 | 设计理由 |
|---|---|---|---|---|
| 工况聚类与归一化 | `src/features/conditions.py`（`ConditionNormalizer`） | 原始表 → 按工况 z-score 的表 | 训练一次 KMeans，推理 O(n) | 六工况子集里，不归一化的话同一传感器含义不同 |
| 多尺度滚动特征 | `src/features/rolling.py`（`add_rolling_features`） | 归一化表 → 379 维特征表 | 向量化 cumsum + shift；4 窗口 × 21 传感器 × 4 统计 | 退化是趋势不是瞬时值；多窗口兼顾早期与末期 |
| LightGBM | `src/models/gbdt.py`（`train_gbdt` / `select_model`） | 379 维 → 点预测 | CPU 秒级；3 折 GroupKFold 选轮数 | 表格特征强、可解释、无需 GPU |
| LSTM | `src/models/lstm.py`（`RULNet`） | 30/50 × 21 → 点预测 | CPU 每子集 0.5–1.5 分钟 | 提供"序列建模是否值得"的证据 |
| 卷积-Transformer | `src/models/deep.py`（`ConvTransformer`） | 30 × 14 + 工况 → 点预测 | CPU 每子集 15–45 分钟 | 检验容量是否是瓶颈（结论：不是） |
| TTA | `src/models/loaders.py`（`_tta_lstm` / `_tta_deep`） | 窗口 → 平均预测 | 推理 ×5 | 成本低、稳定 0.5–1.0 收益 |
| split conformal | `src/intervals/conformal.py` | 校准残差 → q、区间 | O(n log n) | 假设最弱，作为下界 |
| CQR | 同上 + `scripts/w4_intervals.py` | 分位模型 + 校准 → 区间 | 多训两个 LightGBM | 检验"自适应宽度"是否值回成本（结论：未胜） |
| 配对 bootstrap | `scripts/w16_blend_final.py`、W17 | 两条误差向量 → 差值区间 | 2000–3000 次重采样 | 避免把噪声当改进 |
| 校准集选型 | `scripts/w18_select.py` + `src/models/loaders.py` | 8 候选 → 选中的模型 | 推理级开销 | 把测试集从选择回路里摘出去 |
| 中段评估 | `scripts/w17_midlife.py` + `truncate_at_fraction` | 全训练集 → 3545 条记录 | 与推理同量级 | 末周期数字会掩盖中段失效 |
| Streamlit | `app/streamlit_app.py` + `src/demo.py` | 曲线 → 指标卡与两图 | 单次 < 2 秒 | 让结论可触摸 |
| 一键复现 | `scripts/run_all.py` | 原始数据 → 全部表 | `--fast` 32 秒，全量约 2 小时 | 数字可追溯 |

---

## 25. 与 v0.1 设计的差异清单

| 项 | v0.1 设计 | v0.2 实际 | 原因 |
|---|---|---|---|
| 跨域范围 | C-MAPSS → MetroPT-3 跨设备 | 同仿真器 E1–E5 阶梯 | 跨设备不可比、结论易被消解（ADR 0001） |
| MetroPT-3 | 主线之一 | 仅加载器与附录 | 同上 |
| 区间主方法 | CQR 为主、split 为对照 | 结论反转：split 覆盖率相当、宽度更小 3–6 周期；REPORT §5.2 如实写 | CQR 的分位模型在低 RUL 段低估不确定性 |
| demo 区间 | 未定 | 仍展示 CQR 的 90% 范围 | 与 REPORT 结论存在口径差，**待切换为 split**（见下） |
| 演示形态 | 时间轴滑杆（14/28/42/56 天） | 选机器 + 观测周期 + 预测轨迹 | 训练集是完整寿命，"选一台机器看它怎么坏"更有说服力 |
| 特征 | 滑动统计 + 工况归一化（未定维数） | 379 维多尺度滚动 + cummax | 早期/末期需要不同窗口 |
| 模型 | GBDT + LSTM/TCN | GBDT + LSTM(w30/w50) + ConvTransformer + 等权组合 | 需要检验"容量是否瓶颈" |
| 选型 | 只写"测试集只读一次" | W18 真正闭环，报告后悔值 | 之前所有对比都在测试集上读，属于协议漏洞 |
| 评估口径 | 末周期 RMSE/NASA | 末周期 + 每周期 + 中段全量 + bootstrap | 末周期会美化模型（§20、§21） |
| 数据位置 | `data/raw` | `data/cmapss`、`data/metropt` | 与加载器默认路径一致 |
| 测试 | 计划"核心逻辑有单测" | 48 个用例 | 覆盖到 demo 与选型工具 |

**已知不一致（需要后续处理）**：REPORT §5.2 的结论是 split conformal 比 CQR 更窄、覆盖率相当，而 demo 仍用 CQR 的 `cqr_q` 与分位模型出区间。两者都来自同一份校准集，属于"报告结论未落到产品"的偏差。修正方式：在 `RULBundle` 增加 `interval="split"` 选项，demo 默认用它（split 的 q 已存在 `conformal_FD00x.json` 的 `split_q` 字段），并补一个测试。
