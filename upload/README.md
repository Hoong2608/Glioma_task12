# 任务一：真人体 vs 伪造影像（`IsNotHumanBodyProb`）

本目录实现**评审目标一**：在服务器上用赛方数据**从零训练**判别模型，再把训练产物
接进现有比赛推理管线（替换 Dummy Goal1）。

> 本目录以外的文件（`app/`、`core/`、`data/`、`pipeline/`、`output/`、`tasks/`、
> `scripts/`、`README.md`、`requirements.txt`、`Dockerfile`、`start.sh` 等）**一行都不用改**。
> 所有改动都限制在 `task1/` 内，接入方式只是设置环境变量。

## 1. 需要上传哪些文件

只上传 **`task1/` 整个目录**到服务器的仓库根目录（即与 `app/`、`core/` 同级）：

```text
<服务器仓库根>/task1/            ← 上传这个目录
├── __init__.py
├── _bootstrap.py               # 兼容直接执行脚本的路径兜底
├── config.py                   # 运行/训练配置（TASK1_* 环境变量）
├── dataset.py                  # 赛方数据扫描、标签、切分、切片数据集  ★训练新增
├── metrics.py                  # AP / 部分 AUC-PR / ROC-AUC / Recall@FPR  ★训练新增
├── middle_slice.py             # 任务二：全库中间层指纹与重复索引          ★任务二
├── stitched.py                 # 任务二：拼接分数 + 阈值标定 CLI          ★任务二
├── goal2.py                    # 任务二：检测任务、闸门、重复对记录        ★任务二
├── model.py                    # 网络结构（ConvNeXt-Tiny + 频域分支）
├── preprocess.py               # 体数据 -> 切片（与主文件夹推理口径逐位一致）
├── scorer.py                   # 权重加载与打分（检查级聚合）
├── goal1.py                    # 管线 StudyTask：产出 Goal1Result
├── pipeline_factory.py         # COMPETITION_PIPELINE_FACTORY 入口
├── train.py                    # 训练主脚本                     ★训练新增
├── run_local_eval.py           # 本地端到端跑一次（等于一次 /call）
├── score_dataset.py            # 批量打分 + 指标（自测用）
├── requirements.txt            # 额外依赖：torch / timm / numpy / nibabel
├── configs/task1.env.example   # 环境变量样例（可直接改成生产值）
├── tests/test_task1.py         # 11 项自检
├── tests/test_task2.py         # 7 项自检（拼接/重复/闸门/端到端）
├── weights/                    # 可选：公开代理数据权重（兜底/可选初始化，213 MB）
└── artifacts/                  # 可选：本地验证证据（上传与否都不影响运行）
```

上传建议：

| 内容 | 是否必须 | 说明 |
| --- | --- | --- |
| `task1/*.py`、`task1/requirements.txt`、`task1/configs/` | **必须** | 训练与推理的全部代码 |
| `task1/tests/` | 建议 | 服务器上先跑自检，确认环境 OK |
| `task1/weights/`（213 MB） | 可选 | 公开代理数据训练的权重；服务器从零训练时可不上传。留着有两个用途：训练失败时的兜底、或用 `--backbone-init` 做初始化对比 |
| `task1/artifacts/` | 可选 | 本地验证记录（打分 jsonl、demo 输出），纯证据 |
| 管线其他目录 | **不需要** | 服务器已有，且不允许改动 |

上传方式二选一：

```bash
# 方式 A：直接 tar 整个目录（保留相对结构）
tar czf task1.tar.gz task1
scp task1.tar.gz <user>@<server>:/2026aicompetition/workspace/Glioma_recognition-main/
ssh <user>@<server> 'cd /2026aicompetition/workspace/Glioma_recognition-main && tar xzf task1.tar.gz'

# 方式 B：sftp / 平台文件上传，把 task1 目录整体放到仓库根目录
```

> 服务器仓库根目录以你方实际部署为准（本仓库解包后是
> `Glioma_recognition-main/Glioma_recognition-main/`）。放好后确认
> `<仓库根>/task1/train.py` 与 `<仓库根>/app/server.py` 同级。

## 2. 服务器上要改的东西

**不需要改任何 py 文件**，只做「装依赖 + 设环境变量 + 跑训练」。可选的、纯属本目录内的
调整只有三处：

| 文件 | 要不要动 | 说明 |
| --- | --- | --- |
| `task1/configs/task1.env.example` | 可改成生产值 | 数据目录、训练产物目录、日志目录；也可以完全不改，直接在命令行/启动脚本里 export |
| `task1/requirements.txt` | 一般不动 | 只是依赖清单；若服务器 CUDA 版本不同，按注释换 torch 安装源 |
| `task1/README.md` | 不动 | 本说明 |

管线的 `start.sh` / `Dockerfile` / `requirements.txt` **不改**，用环境变量注入即可
（`start.sh` 里是 `exec python -m uvicorn app.server:app`，会读取环境变量）。

### 2.1 装依赖

```bash
cd <仓库根>
python -m pip install -r task1/requirements.txt
# CUDA 12.1 环境推荐（避免 pip 换成 CPU 版 torch）：
# python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### 2.2 设置环境变量

```bash
# 训练/数据（本目录专用）
export TASK1_DATA_ROOT=/2026aicompetition/datasets/training   # 含 annotation/ 的根目录
export TASK1_RUN_DIR=/2026aicompetition/workspace/task1_runs  # 训练产物目录
export TASK1_LOG_DIR=/2026aicompetition/workspace/logs        # 赛方要求的日志目录

# 推理（管线侧，仅环境变量）
export COMPETITION_PIPELINE_FACTORY=task1.pipeline_factory:build_pipeline
export TASK1_WEIGHTS=/2026aicompetition/workspace/task1_runs/best.pt
```

## 3. 服务器训练流程

### 步骤 0：数据体检（几秒钟，不需要 GPU）

```bash
cd <仓库根>
python -m task1.dataset --data-root $TASK1_DATA_ROOT --out-dir $TASK1_RUN_DIR
```

输出会告诉你扫描到多少检查/序列、正类（`fake/`）多少、按检查号切分后的
train/val 数量，并写出 `manifest.jsonl`。**先把这一步的 JSON 看清楚**：

* `positives` 为 0 → 目录里没找到 `fake/`，用 `--annotation-root` 指定标注根目录；
* `by_split.val.positives` 为 0 → 阳性太少，调大 `--val-fraction` 或换切分种子。

### 步骤 1：从零训练

```bash
python -m task1.train \
  --data-root $TASK1_DATA_ROOT \
  --out-dir  $TASK1_RUN_DIR \
  --log-dir  $TASK1_LOG_DIR \
  --epochs 40 --batch-size 4 --slices-per-case 16 --image-size 224 \
  --num-workers 8 --val-fraction 0.15 --seed 42
```

关键参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--pretrained-backbone` | `0` | **0 = 从零随机初始化**（本任务要求）；`1` = 用 timm 预训练权重（需联网或本地缓存） |
| `--backbone-init PATH` | 无 | 用本地已有权重初始化骨干（可选：`weights/authenticity_seed0.pt` 做对比实验） |
| `--composition` / `--duplicate` | `exclude` | `fake/` 之外的两类特殊影像；`exclude` 不参与目标一训练，`negative` 当作真人体负类 |
| `--batch-size` | 4 | 每批序列数；24 GB 显存可加到 8~16，6 GB 显存用 2 |
| `--slices-per-case` / `--image-size` | 16 / 224 | 与推理口径一致，改了要重新训练 |
| `--pos-weight` | -1 | 自动取 neg/pos；正类极少时可显式指定 |
| `--patience` | 10 | 按检查级 AP 早停 |
| `--limit-per-class` | 0 | 每类只取 N 条序列，用于冒烟调试 |
| `--smoke-test` | 关 | 跑 2 步就退出，验证链路 |

先跑一次冒烟（1~2 分钟）确认服务器环境正常：

```bash
python -m task1.train --data-root $TASK1_DATA_ROOT --out-dir $TASK1_RUN_DIR/smoke \
  --log-dir $TASK1_LOG_DIR --smoke-test --epochs 1 --batch-size 2 --slices-per-case 8 --image-size 128
```

### 步骤 2：看训练结果

产物都在 `$TASK1_RUN_DIR`：

```text
task1_runs/
├── best.pt        # 验证集检查级 AP 最优权重（推理服务用这个）
├── last.pt        # 最后一个 epoch
├── manifest.jsonl # 本次训练用到的全部数据 + train/val 划分（可复现）
├── summary.json   # 数据统计、超参、best_case_ap、耗时
└── logs/…         # 若未设 TASK1_LOG_DIR
$TASK1_LOG_DIR/training.jsonl   # 赛方规范日志（timestamp/epoch/step/phase/mode/loss/lr/data_source/checkpoint/pretrained_from）
```

`best.pt` 里除了 `model` 还记录了 `backbone` / `image_size` / `slices_per_case` /
`frequency_branch` / `val_ap` / `pretrained_from`，推理侧会自动按这些值复现输入几何。

### 步骤 3：让推理服务用新权重

```bash
export TASK1_WEIGHTS=$TASK1_RUN_DIR/best.pt
export TASK1_DEVICE=auto
cd <仓库根> && ./start.sh          # 或 python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

自测：

```bash
curl http://127.0.0.1:8000/health
python -m task1.score_dataset --dataset <测试集目录> --out-dir $TASK1_RUN_DIR/predict
python -m task1.run_local_eval --dataset <测试集目录> --output $TASK1_RUN_DIR/answer --evaluation-id demo-001
```

## 4. 训练/推理的数据口径

* 一个训练样本 = 一个检查的**一条序列**（K 张 2.5D 切片）；检查级分数 = 该检查所有序列
  概率的**最大值**（推理侧一致：`TASK1_SERIES_AGGREGATION=max`）。
* 切片选择：训练随机抽 K 层（含翻转/亮度/偏置场/模糊/噪声/gamma 增广），验证与推理用
  uniform 抽层；归一化为 0.5/99.5 百分位截断。
* 检查级聚合 logits 的方式训练与推理严格一致：取 `K//2` 个最高 **logit** 求均值后 sigmoid。
* 目标一标签：`annotation/fake/` 下为阳性（1 = 非真人体/伪造），其余正常影像为阴性（0）；
  `Composition/`（拼接）与 `duplicate/`（重复）默认**不参与**目标一训练，可用
  `--composition negative --duplicate negative` 改为负类。
* 切分按**检查号**分层，同一检查不会跨 train/val。

## 5. 自检与本地验证结果

服务器上先跑：

```bash
cd <仓库根> && python -m unittest discover -s task1/tests -t . -v
```

本地（Windows + RTX 3060 Laptop）已完成的验证：

* 11 项自检全部通过：赛方目录扫描/标签/切分、特殊目录策略、后缀错标的 NIfTI 兜底读取、
  从零训练→`best.pt`→推理可加载的完整闭环、与主文件夹推理口径的逐位等价、Goal1 契约、
  管线端到端输出通过 `OutputValidator`。
* 与主文件夹等价性：预处理 `volume_to_slices` 结果逐位相等（`rtol=0, atol=0`）；
  打分与主文件夹 `AuthenticityPredictor.score_volume` 一致（误差 < 1e-6）。
* 真实体数据短训练（公开代理数据，按赛方目录结构摆放 `annotation/fake|NORMAL…`）：
  36 例 → train 27 / val 9，3 epoch，1.14 分钟，验证集检查级 AP = 1.0，日志字段合规。
* 推理端到端（GPU，含 I/O）：真人体 `0.00004`、TCIA 体模 `0.99997`、BrainWeb 合成脑 `0.6377`；
  单例 0.36–2.66 s。

注意：公开代理数据与赛方 `annotation/fake/` 的真实分布差别较大，上面这些数字只说明
「链路正确、方向正确、能收敛」，**不等于比赛得分**。

## 6. 常见问题

| 现象 | 处理 |
| --- | --- |
| `cannot find the 'fake' annotation folder` | 数据不在默认位置：`python -m task1.dataset --data-root <上层层级> --annotation-root <含 fake 的目录>` |
| 某文件 `is not a gzip file` | 已自动兜底（按实际格式改读），只需留意日志里的 warning |
| CUDA OOM | 降 `--batch-size`（4→2→1）、`--slices-per-case`（16→8）、或 `--image-size`（224→160，注意需重训） |
| val 没有阳性样本 / AP 显示 `null` | 数据里阳性检查太少，调大 `--val-fraction`、换 `--seed`，或补 `fake/` 数据 |
| 训练很慢 | 加大 `--num-workers`（I/O 为主）、开 `--amp 1`（默认开）、用 SSD 存放数据 |
| 想复现上一次训练 | 用产物里的 `manifest.jsonl`：`--manifest $TASK1_RUN_DIR/manifest.jsonl` |
| 权重加载失败但服务仍能启动 | 默认 `TASK1_STRICT=0`，会逐例输出兜底概率 `0.5` 并在日志报错；自检时设 `TASK1_STRICT=1` 直接报错 |

## 7. 任务二：拼接影像 + 重复影像

按你的思路实现，全部在 `task1/` 内，管线零改动。

### 7.1 重复影像（`middle_slice.py` + `goal2.py`）

1. **先扫全库**：处理第一个检查时，从 `Series.source_path` 反推数据集根目录，遍历全库，
   每个检查每条序列只抽**中间一层**（最短轴 `n // 2`），算 SHA1 指纹建立倒排索引；
2. **逐例比对**：当前检查各序列的中间层指纹命中索引中**其它**检查的指纹时判为重复；
3. 重复检查写入 `duplicate_pairs.jsonl`（`PairProb = 1.0`，每例最多 200 个候选），
   并在写 `prediction.json` **之前**给下游任务上闸。

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `TASK2_DUPLICATE` | `1` | 是否启用重复检测 |
| `TASK2_DUPLICATE_SCAN` | `1` | `1` = 先扫全库（按你的思路）；`0` = 只按已见过的检查增量比对 |
| `TASK2_DUPLICATE_SCAN_MAX_VOLUMES` | `0` | 全库扫描的条数上限（0 = 不限），超大测试集可用来限时 |
| `TASK2_DUPLICATE_GATE` | `1` | 命中重复后是否对下游任务上闸 |
| `TASK2_MAX_PAIRS` | `200` | 每例最多提交的候选对数（赛方稀疏性约束） |

### 7.2 拼接影像（`stitched.py` + `goal2.py`）

对每条序列的每一层，取上下两层求平均再与当前层比较：

```
d_i = mean(|I_i - (I_{i-1} + I_{i+1}) / 2|) / mean(|I|)      # 无量纲
score = max_i d_i                                            # 一层拼接即判定
p     = sigmoid(scale * (score / threshold - 1))             # 写入 IsStitchedProb
```

默认丢弃层方向两端各 10% 的层（`TASK2_STITCHED_BAND=0.1`）：任何体数据在 FOV 边缘
（颈部、空气进出视野）本来就有大的层间变化，计入会误判。

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `TASK2_STITCHED` | `1` | 是否启用拼接检测 |
| `TASK2_STITCHED_THRESHOLD` | `0.16` | 判定阈值（**必须用赛方数据重新标定**） |
| `TASK2_STITCHED_METRIC` | `curvature` | `curvature`（你的规则）/ `adjacent` / `local`（相对自身层间差） |
| `TASK2_STITCHED_STAT` | `max` | 层分数汇总：`max` / `p99` / `mean` |
| `TASK2_STITCHED_BAND` | `0.1` | 两端丢弃比例 |
| `TASK2_STITCHED_SCALE` | `8.0` | 概率映射斜率 |
| `TASK2_STITCHED_GATE` | `1` | 命中后是否上闸 |

### 7.3 闸门（「该病人不向下进入后续任务」）

`goal2_stitched` 排在任务链**最前面**，命中拼接或重复时写 `context.diagnostics["goal2_gate"]`；
下游任务被 `GatedStudyTask` 包装，看到闸门就直接返回中性结果、**不执行模型**。

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `TASK2_GATED_FIELDS` | `goal3,goal4,goal5` | 被闸门拦下的任务（可加 `goal1`；`goal2` 是检测器本身，不能拦） |

命中情况记录在两个地方，便于赛后核对：`context.diagnostics["goal2"]`（分数、命中检查号、扫描统计）
与 `context.diagnostics["goal2_skipped_tasks"]`（哪些任务被跳过）。`prediction.json` 的字段结构
由管线 Writer 固定，不可新增字段，因此闸门效果体现为「下游字段是中性值」。

### 7.4 在赛方数据上自检与标定

```bash
# 两条规则的准确率自检：Composition 召回（拼接）/ duplicate 金标准召回（重复）/ 正常影像误报
python -m task1.goal2 --data-root $TASK1_DATA_ROOT --out-dir $TASK1_RUN_DIR

# 只用 Composition vs 正常影像重新标定拼接阈值（按目标 FPR 取分位）
python -m task1.stitched --data-root $TASK1_DATA_ROOT --out-dir $TASK1_RUN_DIR --target-fpr 0.05
# 也可以换口径对比：--metric adjacent|local --stat p99|mean --band 0.2
```

两个命令都会打印 JSON 并写出 `goal2_evaluation.json` / `stitched_calibration.json`，
把标定出的 `threshold` 设到 `TASK2_STITCHED_THRESHOLD` 即可。

### 7.5 本地验证结果与诚实的边界

* 自检 7/7 通过：拼接分数（人工拼缝 > 正常体数据）、概率单调有界、中间层索引能命中
  同一份像素的不同容器/后缀拷贝、闸门确实跳过下游任务、端到端输出通过 `OutputValidator`
  且 `duplicate_pairs.jsonl` 恰好一行。
* 在本地按赛方目录结构造的自检集上（`annotation/Composition` = 两个真实体数据各取一半拼接，
  正常 = 10 例脑提取 T1，重复 = 同一份像素两份 + 金标准）：
  拼接 AP = 1.0（正类中位 0.141 vs 负类 0.006，建议阈值 0.008~0.16 都能全召回）、
  重复金标准召回 = 1.0、正常影像误报 = 0。证据见 `artifacts/task2_selfcheck/`。
* 端到端样例（`artifacts/task2_demo_answer/demo-task2`，4 例：1 对重复 + 1 例拼接 + 2 例正常）：
  `duplicate_pairs.jsonl` 恰好一行 `CASE_A / CASE_A_COPY, PairProb = 1.0`；
  重复例被上闸（下游任务未执行），拼接例在标定阈值 0.05 下 `IsStitchedProb = 1.0000`、
  正常例 0.0008–0.0013。若沿用默认阈值 0.16，该拼接例只有 0.1268、不会被判为拼接——
  这正是必须先标定的直接证据。
* **必须注意**：拼接分数强烈依赖影像内容。在公开代理数据上，脑提取后的临床 T1 分数在
  0.005–0.01，而带颈部/全头视野或体模的体数据能到 0.4–1.5（见 `artifacts` 记录）。
  因此 `TASK2_STITCHED_THRESHOLD` 默认值只是占位，**上服务器后必须用 `python -m task1.stitched`
  在赛方数据上标定**；如果标定显示 AP 很低，直接换 `--metric local` 或 `--metric adjacent`
  再标一次，命令会同时给出 AP / ROC-AUC / Recall@10%FPR 供比较。
* 重复检测要求「逐像素完全一致」，因此对**重采样/重新压缩过**的重复影像可能漏检；
  需要放宽时可在此模块上加相似度检索（当前结构已预留：`DuplicateIndex` 只存指纹集合）。
