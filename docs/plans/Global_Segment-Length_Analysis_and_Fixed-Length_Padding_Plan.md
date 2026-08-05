# Global Segment-Length Analysis and Fixed-Length Padding Plan

## 1. 总体目标

新增两个互相独立的脚本：

```text
scripts/analyze_segment_lengths.py
scripts/pad_segmented_imu.py
```

完整流程：

```text
已完成的 variable-length segmentation 根目录
→ 扫描全部 user/action
→ 验证 segment 输出完整性
→ 统计全局长度分布
→ 标记异常长 segments
→ 推荐 padding target length
→ 人工确认或直接读取推荐值
→ 批量生成 fixed-length arrays
```

两个脚本均不读取：

* 原始 `ring_0.bin`；
* Board chunks；
* alignment offset；
* timestamp label 原始文件；
* gravity-removal 配置。

它们只处理已完成的 segmentation 输出。

---

# 2. 输入根目录的定义

一次分析只处理一种 segmentation 和预处理方法。

例如 label-only low-pass：

```text
outputs/segmentedIMU_LowPassFiltering/
```

Board-assisted low-pass：

```text
outputs/boardAssistSegmentedIMU_LowPassFilterin/
```

Madgwick 和 Raw IMU 应分别分析：

```text
outputs/segmentedIMU_Madgwick/
outputs/segmentedIMU_RawIMU/
outputs/boardAssistSegmentedIMU_Madgwick/
outputs/boardAssistSegmentedIMU_RawIMU/
```

不能把不同 gravity-removal 方法的目录混在一次分析中，因为它们代表不同的数据版本，后续模型训练也应分别管理。

---

# 3. 输入目录结构

脚本递归查找：

```text
{input_root}/
└── user_*/
    └── action_*/
        ├── {stem}_rawIMU.npy
        ├── {stem}_labels.npy
        ├── {stem}_segment_offsets.npy
        ├── {stem}_segment_lengths.npy
        └── ...
```

其中：

```text
stem = {user}_action_{action}
```

例如：

```text
user_0_action_0_rawIMU.npy
user_0_action_0_labels.npy
user_0_action_0_segment_offsets.npy
user_0_action_0_segment_lengths.npy
```

Board-assisted 目录还可能包含：

```text
user_0_action_0_board_event_targets.npy
```

分析脚本只需要 lengths、offsets、labels 和 manifest；padding 脚本还需要 `rawIMU.npy` 和可选的 Board event targets。

---

# 4. 共享实现模块

新增：

```text
src/writingring/segment_padding.py
```

该模块供两个 CLI 共用，避免两套脚本使用不同的文件发现和验证逻辑。

建议数据结构：

```python
@dataclass(frozen=True, slots=True)
class SegmentedDatasetPaths:
    input_dir: Path
    user: str
    action: str
    stem: str

    raw_imu_path: Path
    labels_path: Path
    segment_offsets_path: Path
    segment_lengths_path: Path

    manifest_path: Path | None
    board_event_targets_path: Path | None
```

```python
@dataclass(frozen=True, slots=True)
class SegmentLengthRecord:
    global_segment_index: int
    local_segment_index: int

    user: str
    action: str
    dataset_id: int | None
    label: str

    sample_count: int
    source_directory: Path
```

---

# 5. 输入发现规则

递归查找：

```text
*_segment_lengths.npy
```

找到 lengths 文件后，根据同一 stem 推导其他文件。

例如找到：

```text
user_0_action_0_segment_lengths.npy
```

必须同时存在：

```text
user_0_action_0_rawIMU.npy
user_0_action_0_labels.npy
user_0_action_0_segment_offsets.npy
```

如果缺少任何必需文件：

```text
默认直接失败
```

不能静默跳过一个 user/action，否则全局 maximum 和长度分布会不完整。

可选 Board targets：

```text
如果存在 → 验证并记录
如果不存在 → 视为 label-only 数据
```

---

# 6. 共享输入一致性验证

每个 user/action 必须满足：

```text
rawIMU.ndim == 2
rawIMU.shape[1] == 6

labels.ndim == 1

segment_lengths.ndim == 1
segment_offsets.ndim == 1

len(labels) == len(segment_lengths)
len(segment_offsets) == len(segment_lengths) + 1

segment_offsets[0] == 0
segment_offsets[-1] == len(rawIMU)

segment_offsets 单调非递减
segment_lengths == np.diff(segment_offsets)

所有 segment length > 0
```

Board-assisted 数据额外满足：

```text
board_event_targets.ndim == 2
board_event_targets.shape == (len(rawIMU), 4)
board_event_targets.dtype 可安全转换为 bool
```

发现不一致时，错误信息必须指出：

```text
user
action
文件路径
预期 shape
实际 shape
```

---

# 7. Script 1：全局长度分析

脚本：

```text
scripts/analyze_segment_lengths.py
```

基本命令：

```bash
python scripts/analyze_segment_lengths.py \
    --input-root outputs/segmentedIMU_LowPassFiltering
```

Board-assisted：

```bash
python scripts/analyze_segment_lengths.py \
    --input-root outputs/boardAssistSegmentedIMU_LowPassFilterin
```

该脚本只分析，不生成 padded arrays。

---

# 8. 分析脚本需要计算的统计量

对根目录下所有 segments 的长度：

[
L_1,L_2,\ldots,L_N
]

计算：

```text
segment count
user/action count
minimum
maximum
mean
standard deviation
median
P75
P90
P95
P97.5
P99
P99.5
```

同时计算秒数版本：

[
duration_i=\frac{L_i}{f_s}
]

默认：

```text
sampling rate = 200 Hz
```

但提供：

```text
--sampling-rate 200
```

报告同时显示：

```text
600 samples
3.000 seconds
```

---

# 9. 候选 Padding Length 分析

默认候选值：

```text
256
320
400
480
512
600
640
768
800
1024
```

允许用户指定：

```bash
--candidate-lengths 400 500 600 700 800
```

对每个候选长度 (T)，计算：

## Coverage

[
Coverage(T)=
\frac{#{i:L_i\le T}}{N}
]

表示无需截断即可容纳的 segment 比例。

## Overflow

[
OverflowCount(T)=
#{i:L_i>T}
]

## 有效样本比例

若所有 segments 都 padding 到 (T)，且仅考虑能够容纳的情况：

[
Efficiency(T)=
\frac{\sum_iL_i}{N\cdot T}
]

对于 (T<\max L_i)，该值只能作为假设性指标，并必须标记：

```text
requires overflow handling
```

## Padding 比例

[
PaddingRatio(T)=1-Efficiency(T)
]

例如：

```text
target = 600
coverage = 99.2%
overflow = 18
valid sample ratio = 71.4%
padding ratio = 28.6%
```

---

# 10. 推荐长度的三种结果

分析脚本不只输出一个模糊的“最佳长度”，而是给出三个语义明确的推荐。

## 10.1 Pure-padding Recommendation

如果不允许丢弃或截断任何 segment：

[
T_{pure}=\max_iL_i
]

可选择将其向上取整：

```text
--round-to 1
--round-to 8
--round-to 16
--round-to 32
```

计算：

```python
recommended_pure_padding = (
    math.ceil(maximum_length / round_to) * round_to
)
```

默认：

```text
round_to = 1
```

这是唯一能够保证：

```text
所有 segments 只 padding、不 truncation
```

的推荐值。

---

## 10.2 P99 Recommendation

[
T_{p99}=\lceil P99\rceil
]

同样可按 `round_to` 向上取整。

该值通常更节省计算，但必须明确显示：

```text
有多少 segments 超过该长度
```

它不能直接用于纯 padding，除非后续对 overflow segments 另行处理。

---

## 10.3 Balanced Recommendation

在候选长度中选择满足：

```text
coverage >= requested minimum
```

的最小值。

默认：

```text
--minimum-coverage 0.99
```

例如：

```text
400 → 82.1%
512 → 96.7%
600 → 99.3%
640 → 99.8%
800 → 100%
```

Balanced recommendation 为：

```text
600
```

但报告必须注明：

```text
该推荐仍有 overflow segments，不能用于 strict pure padding。
```

---

# 11. 异常长 Segment 检测

如果 maximum 远高于主体分布，不能直接让所有样本 pad 到 maximum。

至少使用两种规则标记异常候选：

## Quantile Rule

```text
length > P99
```

## IQR Rule

[
length>Q_3+3\cdot IQR
]

其中：

[
IQR=Q_3-Q_1
]

输出异常 segment identity：

```text
user
action
local segment index
dataset ID
label
sample count
duration seconds
source directory
```

例如：

```text
user_7/action_2
segment 83
label = "A"
length = 1842
duration = 9.21 s
```

分析脚本不删除这些 segments，只负责报告。

---

# 12. 分析脚本输出

默认在：

```text
{input_root}/padding_analysis/
```

生成：

```text
segment_length_analysis.json
segment_length_analysis.csv
segment_length_candidates.csv
segment_length_outliers.csv
```

建议同时生成：

```text
segment_length_histogram.png
segment_length_ecdf.png
```

其中 JSON 是后续 padding 脚本可读取的机器接口。

---

# 13. Analysis JSON 核心结构

```json
{
  "input_root": "outputs/segmentedIMU_LowPassFiltering",
  "sampling_rate_hz": 200.0,
  "segment_count": 2184,
  "user_action_count": 126,

  "length_statistics": {
    "minimum": 184,
    "median": 411.0,
    "p95": 563.0,
    "p99": 618.0,
    "maximum": 742
  },

  "recommendations": {
    "pure_padding": {
      "target_length": 742,
      "coverage": 1.0,
      "overflow_count": 0
    },
    "p99": {
      "target_length": 618,
      "coverage": 0.99,
      "overflow_count": 22
    },
    "balanced": {
      "target_length": 640,
      "coverage": 0.996,
      "overflow_count": 9
    }
  }
}
```

---

# 14. 分析阶段的推荐决策

如果当前需求是：

```text
所有 segment 必须保留
只允许 padding
不允许 truncation
```

则应使用：

```text
recommendations.pure_padding.target_length
```

如果 pure-padding target 因少数异常值过大，应先人工检查 `segment_length_outliers.csv`。

不能直接使用 P99，然后仍声称“所有数据只进行了 padding”。

---

# 15. Script 2：固定长度 Padding

脚本：

```text
scripts/pad_segmented_imu.py
```

它读取已完成 segmentation 根目录，并输出独立的 fixed-length 数据根目录。

基本命令：

```bash
python scripts/pad_segmented_imu.py \
    --input-root outputs/segmentedIMU_LowPassFiltering \
    --target-length 742
```

也可以读取分析报告：

```bash
python scripts/pad_segmented_imu.py \
    --input-root outputs/segmentedIMU_LowPassFiltering \
    --analysis-report \
      outputs/segmentedIMU_LowPassFiltering/padding_analysis/segment_length_analysis.json \
    --recommendation pure-padding
```

---

# 16. Padding 脚本的 Target Length 来源

以下方式二选一：

## 显式指定

```text
--target-length 742
```

## 从报告读取

```text
--analysis-report PATH
--recommendation {pure-padding,p99,balanced}
```

默认推荐只允许：

```text
pure-padding
```

如果选择：

```text
p99
balanced
```

且存在 overflow，脚本默认失败。

这是为了避免分析建议被误当作自动 truncation 策略。

---

# 17. Padding 脚本的严格原则

第一版只实现：

```text
right-side zero padding
overflow = error
```

即：

```text
短于 target → 右侧补零
等于 target → 原样保留
长于 target → 整个任务失败
```

不实现：

```text
truncation
cropping
resampling
segment splitting
silent skipping
```

这些属于独立的数据处理决策，不应混入 padding。

---

# 18. 输出根目录

默认不修改输入目录。

如果输入：

```text
outputs/segmentedIMU_LowPassFiltering
```

目标长度：

```text
742
```

默认输出：

```text
outputs/segmentedIMU_LowPassFiltering_padded_742
```

也可显式指定：

```bash
--output-root outputs/fixedLengthIMU_LowPass_742
```

保留原始层级：

```text
output_root/
└── user_0/
    └── action_0/
        └── ...
```

---

# 19. 每个 User/Action 的输出

## Label-only

```text
output_root/
└── user_0/
    └── action_0/
        ├── user_0_action_0_paddedIMU.npy
        ├── user_0_action_0_labels.npy
        ├── user_0_action_0_valid_lengths.npy
        ├── user_0_action_0_valid_mask.npy
        ├── user_0_action_0_padding_manifest.csv
        └── user_0_action_0_padding_summary.json
```

## Board-assisted

额外生成：

```text
user_0_action_0_padded_board_event_targets.npy
```

---

# 20. 输出数组定义

## Padded IMU

```text
paddedIMU.shape == (N, T, 6)
```

其中：

* (N)：该 user/action 的 segment 数；
* (T)：全根目录统一 target length。

## Labels

```text
labels.shape == (N,)
```

与原 labels 完全一致。

## Valid Lengths

```text
valid_lengths.shape == (N,)
```

定义：

```python
valid_lengths[i] = original segment length
```

## Valid Mask

```text
valid_mask.shape == (N, T)
```

定义：

```python
valid_mask[i, t] = t < valid_lengths[i]
```

## Board Event Targets

如果原目录存在 targets：

```text
padded_board_event_targets.shape == (N, T, 4)
```

padding 区域全部为 `False`。

---

# 21. Padding 算法

```python
padded_imu = np.zeros(
    (segment_count, target_length, 6),
    dtype=raw_imu.dtype,
)

valid_mask = np.zeros(
    (segment_count, target_length),
    dtype=np.bool_,
)

valid_lengths = segment_lengths.astype(
    np.int32,
    copy=True,
)
```

逐段处理：

```python
for index in range(segment_count):
    source_start = segment_offsets[index]
    source_stop = segment_offsets[index + 1]
    length = source_stop - source_start

    if length > target_length:
        raise SegmentPaddingError(...)

    padded_imu[index, :length, :] = (
        raw_imu[source_start:source_stop]
    )

    valid_mask[index, :length] = True
```

Board targets：

```python
padded_targets[index, :length, :] = (
    board_event_targets[source_start:source_stop]
)
```

---

# 22. Padding Manifest

每个 segment 一行：

```text
segment_index
user
action
dataset_id
label

original_length
target_length
padding_length
valid_fraction

was_padded
board_event_targets_present
source_input_directory
```

计算：

```python
padding_length = target_length - original_length
valid_fraction = original_length / target_length
was_padded = original_length < target_length
```

---

# 23. Padding Summary

每个 user/action 输出：

```json
{
  "target_length": 742,
  "segment_count": 24,

  "minimum_original_length": 271,
  "maximum_original_length": 689,

  "padded_segment_count": 24,
  "exact_length_segment_count": 0,

  "total_valid_samples": 10583,
  "total_padding_samples": 7225,
  "valid_sample_ratio": 0.5943,

  "padding_side": "right",
  "padding_value": 0.0,
  "overflow_policy": "error",

  "board_event_targets_present": true
}
```

---

# 24. 根目录级 Padding Summary

批处理完成后，在 output root 生成：

```text
padding_dataset_summary.json
padding_dataset_manifest.csv
```

记录：

```text
input root
output root
target length
sampling rate
processed user/action count
segment count
total valid samples
total padding samples
global valid sample ratio
Board-assisted package count
label-only package count
failed package count
```

成功时：

```text
failed package count = 0
```

---

# 25. Transactional Publishing

必须先完成整个根目录的验证：

```text
发现全部 input packages
→ 验证全部 files
→ 验证 global maximum <= target
```

只有全部通过后才开始写正式结果。

建议：

```text
先写入临时 output root
全部完成后 rename 到最终 output root
```

例如：

```text
.fixedLengthIMU_742.tmp/
→
fixedLengthIMU_742/
```

如果任何 user/action 失败：

* 删除临时 root；
* 不留下部分 padded 数据；
* 不修改原 segmentation 结果。

---

# 26. Padding 前全局预检查

Padding 脚本开始时必须重新计算当前输入 root 的：

```text
package count
segment count
global maximum
```

并与 analysis report 比较。

如果分析后输入数据发生变化：

```text
segment count 不一致
global maximum 不一致
input root 不一致
```

则拒绝使用旧报告，提示重新运行 analysis。

建议 analysis JSON 保存：

```text
每个 lengths 文件的相对路径
文件大小
mtime
或 SHA-256
```

第一版至少保存：

```text
relative path
segment count
maximum length
```

用于检测明显变化。

---

# 27. 分析脚本 CLI

```text
python scripts/analyze_segment_lengths.py
```

参数：

```text
--input-root PATH
--output-dir PATH

--sampling-rate FLOAT
--candidate-lengths INTEGER [INTEGER ...]

--minimum-coverage FLOAT
--round-to INTEGER

--overwrite
```

默认：

```text
sampling rate = 200
minimum coverage = 0.99
round to = 1
```

---

# 28. Padding 脚本 CLI

```text
python scripts/pad_segmented_imu.py
```

参数：

```text
--input-root PATH
--output-root PATH

--target-length INTEGER

--analysis-report PATH
--recommendation {pure-padding,p99,balanced}

--padding-value FLOAT
--overwrite
```

约束：

```text
--target-length
与
--analysis-report + --recommendation
二选一
```

第一版固定：

```text
padding side = right
overflow policy = error
```

无需开放多种 padding side 或 overflow policy。

---

# 29. 推荐实际运行流程

## Step 1：分析 Label-only Low-pass

```bash
python scripts/analyze_segment_lengths.py \
    --input-root outputs/segmentedIMU_LowPassFiltering \
    --candidate-lengths 400 480 512 600 640 768 800
```

## Step 2：检查异常值

查看：

```text
padding_analysis/segment_length_outliers.csv
padding_analysis/segment_length_histogram.png
padding_analysis/segment_length_candidates.csv
```

## Step 3：决定 Target

如果不允许截断：

```text
使用 pure_padding.target_length
```

## Step 4：生成固定长度数据

```bash
python scripts/pad_segmented_imu.py \
    --input-root outputs/segmentedIMU_LowPassFiltering \
    --analysis-report \
      outputs/segmentedIMU_LowPassFiltering/padding_analysis/segment_length_analysis.json \
    --recommendation pure-padding
```

## Step 5：分别处理 Board-assisted Root

```bash
python scripts/analyze_segment_lengths.py \
    --input-root outputs/boardAssistSegmentedIMU_LowPassFilterin
```

然后：

```bash
python scripts/pad_segmented_imu.py \
    --input-root outputs/boardAssistSegmentedIMU_LowPassFilterin \
    --analysis-report \
      outputs/boardAssistSegmentedIMU_LowPassFilterin/padding_analysis/segment_length_analysis.json \
    --recommendation pure-padding
```

---

# 30. 测试计划

## Input Discovery

验证：

* 能发现多个 users/actions；
* 不会把 `padded_*` 目录再次作为输入；
* 缺少必需文件时失败；
* duplicate stem 时失败。

## Length Analysis

使用长度：

```text
[100, 200, 300, 1000]
```

验证：

```text
minimum = 100
maximum = 1000
median = 250
pure-padding recommendation = 1000
```

## Round Up

```text
maximum = 601
round_to = 32
```

结果：

```text
pure-padding target = 608
```

## Candidate Coverage

验证每个候选长度的：

```text
coverage
overflow count
valid sample ratio
padding ratio
```

## Outlier Report

异常 segment 必须保留：

```text
user/action
index
label
length
```

## Padding Content

长度：

```text
[3, 5]
target = 6
```

输出：

```text
shape = (2, 6, 6)
valid_lengths = [3, 5]
mask sums = [3, 5]
```

## Overflow

若：

```text
maximum length = 601
target = 600
```

则在任何正式输出前失败。

## Board Targets

验证：

* shape 为 `(N,T,4)`；
* 有效部分与原 targets 相同；
* padding 部分全部为 False。

## Input Immutability

输入 root 中所有原文件 hash 在运行前后相同。

## Transactional Failure

任一 user/action 失败时：

```text
最终 output root 不存在
```

或保持原有完整版本不变。

---

# 31. 文件修改清单

新增：

```text
src/writingring/segment_padding.py

scripts/analyze_segment_lengths.py
scripts/pad_segmented_imu.py

tests/test_segment_padding.py
tests/test_analyze_segment_lengths_cli.py
tests/test_pad_segmented_imu_cli.py

docs/notes/SEGMENT_PADDING.md
```

修改：

```text
src/writingring/__init__.py
README.md
```

不修改：

```text
src/writingring/segmentation.py
src/writingring/board_event_segmentation.py
scripts/segment_ring_imu.py
```

Segmentation 与 fixed-length preparation 保持完全解耦。

---

# 32. 实施顺序

1. 实现 segmentation package discovery。
2. 实现 package input validation。
3. 实现 segment metadata extraction。
4. 实现全局 length statistics。
5. 实现 candidate coverage/efficiency 分析。
6. 实现 outlier identification。
7. 实现三类 recommendation。
8. 实现 JSON、CSV 和分析图输出。
9. 实现分析脚本 CLI。
10. 实现 fixed-length array builder。
11. 实现 valid lengths 和 valid mask。
12. 实现 Board event targets 同步 padding。
13. 实现 per-package manifest 和 summary。
14. 实现 root-level manifest 和 summary。
15. 实现 transactional root publishing。
16. 实现 padding CLI。
17. 添加 unit tests。
18. 添加 CLI integration tests。
19. 在真实根目录运行分析。
20. 人工检查异常长 segments。
21. 确定最终 target。
22. 生成 fixed-length dataset。
23. 验证所有输出第二维完全一致。

---

# 33. 完成标准

实现完成后必须满足：

* 分析脚本一次扫描整个 segmentation root；
* 所有 user/action 的长度共同决定 target；
* 输出 maximum、quantiles、coverage 和 padding efficiency；
* 明确区分 pure-padding、P99 和 balanced 推荐；
* pure-padding recommendation 可以覆盖全部 segments；
* 异常长 segments 可追溯到 user/action/label；
* padding 脚本只读取已完成 segmentation；
* padding 脚本不重新运行任何 segmentation；
* 输入 variable-length 数据完全不被修改；
* 输出使用独立 fixed-length root；
* 所有 IMU 输出形状为 `(N,T,6)`；
* 所有 packages 使用同一个 (T)；
* labels、valid lengths 和 valid masks 完整输出；
* Board targets 自动同步为 `(N,T,4)`；
* 默认只使用 right zero padding；
* 默认禁止 truncation 和 silent skipping；
* 任一 overflow 在写文件前被发现；
* 根目录输出事务性发布；
* 所有新增与现有测试通过。


