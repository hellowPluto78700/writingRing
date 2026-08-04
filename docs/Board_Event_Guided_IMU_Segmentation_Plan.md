# Dual-Mode Alignment-Aware Board-Event-Guided IMU Segmentation Plan

## 1. 设计结论

系统保留两套彼此独立、可同时存在的 segmentation 方法：

```text
label
aligned-board-events
```

最终设计遵循以下原则：

1. `label` 模式是默认模式，完整保留当前实现和输出行为。
2. `aligned-board-events` 模式必须显式选择，不得隐式改变旧结果。
3. 两种模式使用独立代码路径、配置类型、结果类型和输出目录。
4. 两种模式不得在同一聚合输出中混合。
5. `timestamp.txt` 始终负责类别定义和 label interval 的硬隔离语义。
6. Board press/lift events 只在 `aligned-board-events` 模式中用于细化动作窗口。
7. Ring–Board alignment offset 只读取和应用，不在 segmentation 阶段重新估计。
8. 所有 segment 保持 variable-length，并继续采用 contiguous storage、offsets 和 lengths。
9. `aligned-board-events` 模式必须为每个 recording 生成完整 recording verification PNG。
10. 原始 Ring、Board 和 label timestamps 不得修改。

---

## 2. 两种模式的职责边界

### 2.1 Label-only 模式

使用原始边界：

```text
[label_i_timestamp, label_(i+1)_timestamp)
```

最后一个 label 使用：

```text
[label_last_timestamp, ring_end_timestamp)
```

该模式：

- 只读取 `ring_0.bin` 和 `timestamp.txt`；
- 不读取 Board 数据；
- 不读取 alignment offset；
- 不检测 press/lift events；
- 不生成 Board event targets；
- 不生成 Board events CSV；
- 默认不生成 verification PNG；
- 保持当前 public API、数组内容、segment 顺序、skip rules、manifest 和输出路径不变。

### 2.2 Aligned-board-events 模式

该模式同时使用：

- `timestamp.txt` 中的 labels；
- 已成功计算的 Ring–Board alignment offset；
- 完整 Board recording 中的 press/lift touch sequence。

其中：

- label 决定类别；
- label interval 决定事件归属范围和相邻类别的硬隔离边界；
- Board events 决定动作的实际开始与结束；
- verification figure 用于验证 alignment、events、labels 和 final segments 是否一致。

新模式默认规则：

```text
pre-press context  = 0.2 s
post-lift context  = 0.2 s
missing event      = skip
cross-label touch  = skip
verification       = required
```

---

## 3. 向后兼容要求

### 3.1 默认模式

配置和 CLI 默认均为：

```text
boundary_mode = label
```

因此现有命令：

```bash
python scripts/segment_ring_imu.py \
    --data-root data \
    --user user_0 \
    --action 0 \
    --output-root outputs/segmentedIMU
```

必须保持当前行为。

### 3.2 旧 API 不变

保留现有 recording-level API：

```python
def segment_recording_by_labels(
    *,
    ring_imu: np.ndarray,
    ring_timestamps_us: np.ndarray,
    labels: Sequence[SegmentLabel],
    config: SegmentationConfig = SegmentationConfig(),
) -> tuple[SegmentedSample, ...]:
    ...
```

该函数不得：

- 增加 Board 参数；
- 增加 alignment 参数；
- 改变 label interval 语义；
- 改变 skip rules；
- 改变现有返回值语义。

保留现有 user/action API：

```python
def segment_user_action(
    *,
    data_root: Path,
    user: str,
    action: str,
    output_root: Path,
    config: SegmentationConfig = SegmentationConfig(),
    overwrite: bool = False,
) -> UserActionSegmentationResult:
    ...
```

### 3.3 Label-only 严格隔离

运行 `label` 模式时：

1. 不调用 Board loader；
2. 不调用 Board event detector；
3. 不读取 alignment TXT；
4. 不检查 offset 是否存在；
5. 不要求 alignment 成功；
6. 不生成 Board sidecar 文件；
7. 不进入 aligned-mode 输出目录；
8. 即使缺少 Board 文件或 offset，仍可正常运行。

---

## 4. 推荐模块结构

### 4.1 保留模块

```text
src/writingring/segmentation.py
```

仅负责当前 label-only segmentation。

```text
src/writingring/event_alignment.py
src/writingring/alignment_io.py
```

继续负责 alignment 相关基础逻辑和 offset 读取。

### 4.2 新增模块

```text
src/writingring/board_event_segmentation.py
```

负责：

- Board event/touch pair 输入验证；
- eligible pair assignment；
- provisional window 生成；
- 相邻窗口 collision resolution；
- final segment 提取；
- Board event target 生成；
- aligned-mode manifest rows 和 skipped records。

```text
src/writingring/segmentation_verification.py
```

负责：

- 完整 recording verification figure；
- aligned-mode 强制 verification；
- 可选的 label-only verification；
- verification 输入验证和结果统计。

### 4.3 CLI 仅负责 dispatch

```python
if args.boundary_mode == "label":
    result = segment_user_action(...)
elif args.boundary_mode == "aligned-board-events":
    result = segment_user_action_by_aligned_board_events(...)
else:
    raise ValueError(...)
```

CLI 不应在两种模式间共享隐式 fallback 行为。

---

## 5. 配置类型分离

### 5.1 Label-only 配置

保持现有配置最小化：

```python
@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    output_dtype: str = "float32"
    include_last_label: bool = True
    minimum_label_interval_us: float = 100_000.0
    maximum_segment_duration_us: float = 5_000_000.0
```

不得加入 Board-only 必填字段。

### 5.2 Board-event-guided 配置

新增：

```python
@dataclass(frozen=True, slots=True)
class BoardEventSegmentationConfig:
    output_dtype: str = "float32"
    include_last_label: bool = True

    pre_press_context_us: float = 200_000.0
    post_lift_context_us: float = 200_000.0

    missing_event_policy: str = "skip"
    crossing_touch_policy: str = "skip"

    minimum_label_interval_us: float = 100_000.0
    maximum_segment_duration_us: float = 5_000_000.0

    require_successful_alignment: bool = True
    label_time_domain: str = "ring"
```

配置校验至少包括：

```text
pre_press_context_us >= 0
post_lift_context_us >= 0
minimum_label_interval_us > 0
maximum_segment_duration_us > 0
missing_event_policy in allowed values
crossing_touch_policy in allowed values
label_time_domain in {ring, board}
```

### 5.3 Verification 配置

```python
@dataclass(frozen=True, slots=True)
class SegmentationVerificationConfig:
    panel_duration_s: float = 10.0
    output_dpi: int = 200

    valid_event_linewidth: float = 1.5
    transient_event_linewidth: float = 0.6

    valid_event_alpha: float = 0.80
    transient_event_alpha: float = 0.35

    segment_alpha: float = 0.10
    overwrite: bool = False
```

校验：

```text
panel_duration_s > 0
output_dpi > 0
valid_event_linewidth > transient_event_linewidth
0 < segment_alpha <= 0.12
```

---

## 6. 统一时间域和 Alignment 规则

Ring–Board alignment 统一采用：

```text
ring_timestamp_us = board_timestamp_us + alignment_offset_us
```

因此 Board event 映射为：

```python
aligned_event_timestamp_us = (
    frame_timestamp_raw + alignment_offset_us
)
```

Label timestamp 默认已经属于 Ring/shared 时间域：

```python
aligned_label_timestamp_us = label_timestamp_us
```

只有显式配置：

```text
label_time_domain = board
```

时，才对 label 加同一个 offset。

必须验证：

- alignment 结果成功；
- offset 是有限值；
- offset 文件中的 user、action、dataset ID 与 recording 一致；
- offset 只应用一次；
- Board timestamps、label timestamps 和 Ring timestamps 不被原地修改；
- segmentation 阶段不得重新估计 offset。

---

## 7. 完整 Board Recording 和 Stale Tail

Alignment 过程可能只使用 recording 前部的部分事件估计 offset，但 segmentation 必须使用完整 Board recording：

```text
加载完整 Board recording
→ 检测 timestamp backward jump
→ 删除 backward jump 后的 stale tail
→ 在剩余完整有效时间段上检测 occupancy transitions
→ 将全部事件映射到 Ring 时间
```

不得只使用：

- alignment interval 中的事件；
- 前若干个 press；
- 固定前 20 秒或前 60 秒；
- alignment verification 图中显示的子集。

---

## 8. Board Occupancy、Events 和 Touch Pairs

### 8.1 Occupancy

```python
occupied = board_frames["contact_count"].to_numpy() > 0
```

### 8.2 Transition 定义

```text
False → True  = press
True  → False = lift
```

### 8.3 Pairing

每个 press 与其后的第一个 lift 配对。

禁止将当前 label interval 的 press 与下一个 label interval 的 lift 错误拼接为同一个有效 touch。

### 8.4 Touch 分类

#### Valid touch

```text
press 和 lift 完整配对
duration_frames >= minimum_duration_frames
transient == false
incomplete_touch == false
```

用途：参与 segment boundary、写入 event target、写入 Board events CSV，并显示在 verification figure。

#### Transient/bounce touch

```text
press 和 lift 完整配对
duration_frames < minimum_duration_frames
transient == true
```

用途：不参与 segment boundary，但写入 transient target、Board events CSV，并显示在 verification figure。

#### Incomplete touch

包括 press 后没有 lift、独立 lift、recording 在 touch 中途结束或 pairing 输入不完整。

用途：不参与 boundary，默认不写入 target，但写入 Board events CSV 和 summary，默认不显示在主图中。

#### Crossing touch

若：

```text
press < next_label_timestamp <= lift
```

则该 pair 跨越 label boundary。默认：

```text
crossing_touch_policy = skip
```

该 pair 不属于任一相邻 label，记录为 `touch_crosses_next_label`，保留在 audit 和 verification 中。

---

## 9. Label Validity 和 Label Interval

在使用 Board events 前，继续执行现有 label validity rules：

- label 等于 `wrong`，忽略大小写：不生成 segment；
- 相邻 label 间隔小于 `0.1 s`：相关 start 无效；
- 当前 label interval 大于 `5 s`：当前 start 无效；
- 最后一个 label 到 Ring 结尾超过 `5 s`：最后 start 无效。

无效 label 仍保留为时间边界、manifest/audit 中的 skipped record 和 verification 图中的 marker。

对于第 `i` 个 label：

```text
I_i = [L_i, L_(i+1))
```

最后一个 label：

```text
I_last = [L_last, ring_end]
```

Label interval 表示类别归属和硬隔离范围，但在 aligned mode 中不直接等于最终 IMU segment。

---

## 10. Eligible Touch Pair Assignment

### 10.1 非最后一个 Label

完整 valid pair 只有满足以下条件时才属于当前 label：

```text
L_i <= press < lift < L_(i+1)
```

建议实现：

```python
eligible_pairs = touch_pairs[
    touch_pairs["valid_touch"]
    & ~touch_pairs["transient"]
    & ~touch_pairs["incomplete_touch"]
    & (touch_pairs["aligned_press_us"] >= current_label_us)
    & (touch_pairs["aligned_press_us"] < next_label_us)
    & (touch_pairs["aligned_lift_us"] > touch_pairs["aligned_press_us"])
    & (touch_pairs["aligned_lift_us"] < next_label_us)
]
```

### 10.2 最后一个 Label

要求：

```text
L_last <= press < lift <= ring_end_timestamp_us
```

### 10.3 Assignment 输出

每个 label interval 的分析结果应至少包含：

```text
label validity status
eligible valid pair count
transient pair count
incomplete pair count
crossing pair count
skip reason, if any
first valid press
last valid lift
provisional window, if available
```

建议先完成所有 label 的 assignment，再统一处理相邻窗口冲突。

---

## 11. 两阶段 Boundary 生成算法

为避免边界逻辑与相邻 segment 状态相互耦合，使用两阶段算法。

### 11.1 第一阶段：生成每个 Label 的 Provisional Window

如果 label interval 中至少存在一个 eligible valid pair：

```python
first_press_us = eligible_pairs.iloc[0]["aligned_press_us"]
last_lift_us = eligible_pairs.iloc[-1]["aligned_lift_us"]
```

候选开始和结束：

```text
provisional_start = first_press - pre_press_context
provisional_end   = last_lift + post_lift_context
```

先 clamp 到 Ring 范围：

```python
provisional_start_us = max(
    ring_start_us,
    first_press_us - config.pre_press_context_us,
)

provisional_end_us = min(
    ring_end_us,
    last_lift_us + config.post_lift_context_us,
)
```

一个 label interval 中存在多个 touches 时：

- 第一个 valid press 决定开始；
- 最后一个 valid lift 决定结束；
- 中间所有 valid touches 均包含在同一 segment 中。

### 11.2 第二阶段：解析相邻 Provisional Window 冲突

仅在所有 label 已完成第一阶段后进行。

#### 情况 A：相邻窗口不重叠

如果：

```text
provisional_end_i <= provisional_start_(i+1)
```

则保留：

```python
final_end_i = provisional_end_i
final_start_next = provisional_start_next
```

即使 `final_end_i` 稍微超过 `L_(i+1)`，只要仍处于两个动作窗口之间的空闲区且没有与下一个 provisional window 重叠，也可保留完整 post-roll。

#### 情况 B：相邻窗口发生重叠

如果：

```text
provisional_end_i > provisional_start_(i+1)
```

则使用下一个 label timestamp 作为硬隔离边界：

```python
final_end_i = min(
    provisional_end_i,
    next_label_timestamp_us,
)

final_start_next = max(
    provisional_start_next,
    next_label_timestamp_us,
)
```

必须满足：

```text
final_end_i <= final_start_(i+1)
```

#### 情况 C：下一个 Label 没有可导出 Segment

如果下一个 label 无效、没有完整 valid touch、只有 transient、存在 crossing touch、缺少 Board 或缺少 alignment，则无法建立可靠的 next provisional start。

当前 segment 采用保守结束边界：

```python
final_end_i = min(
    provisional_end_i,
    next_label_timestamp_us,
)
```

当前类别不得跨入一个未知或无效的下一个 label interval。

### 11.3 Final Boundary Invariants

每个成功 segment 必须满足：

```text
ring_start <= final_start < final_end <= ring_end
```

相邻成功 segments 必须满足：

```text
final_end_i <= final_start_(i+1)
```

Provisional boundaries 仅保存在 metadata 中，不用于最终数据提取。

---

## 12. 没有 Eligible Pair 时的策略

默认：

```text
missing_event_policy = skip
```

不生成 segment，并记录精确原因：

```text
no_complete_touch_pair
only_transient_touches
incomplete_touch
touch_crosses_next_label
board_data_missing
alignment_missing
label_is_wrong
adjacent_interval_lt_0.1s
segment_duration_gt_5s
```

不建议默认回退到 label interval，因为这会在同一训练集中混合两种 boundary 定义。

如保留实验性 fallback：

```text
missing_event_policy = fallback-to-label
```

则必须显式启用，在 manifest 中记录 `boundary_source = label_fallback`，在 summary 中统计 fallback 数量，并最好写入独立实验目录：

```text
aligned_board_events_with_label_fallback/
```

生产数据默认不得启用 fallback。

---

## 13. 时间边界到 Ring Sample Index

继续使用 left-side `searchsorted`：

```python
start_index = np.searchsorted(
    ring_timestamps_us,
    final_start_timestamp_us,
    side="left",
)

stop_index = np.searchsorted(
    ring_timestamps_us,
    final_end_timestamp_us,
    side="left",
)
```

提取：

```python
segment_imu = ring_imu[start_index:stop_index]
```

最终 segment 使用半开区间：

```text
[final_start, final_end)
```

这样 duplicate Ring timestamps 不会在相邻 segments 中重复导出。

提取后再次验证：

```text
0 <= start_index < stop_index <= len(ring_imu)
sample_count == stop_index - start_index
```

---

## 14. Board Event Targets

Aligned mode 输出四通道 event targets：

```text
channel 0 = valid press
channel 1 = valid lift
channel 2 = transient press
channel 3 = transient lift
```

聚合后：

```python
board_event_targets.shape == (
    total_segment_samples,
    4,
)
```

事件映射到 segment 内第一个 timestamp 不早于 event 的 sample：

```python
local_event_index = np.searchsorted(
    segment_ring_timestamps_us,
    aligned_event_timestamp_us,
    side="left",
)
```

仅当：

```text
0 <= local_event_index < segment_sample_count
```

时写入：

```python
event_targets[local_event_index, channel] = 1
```

要求：

- event 不得映射到 event timestamp 之前；
- 同一 event 不得重复映射到多个 segment；
- segment 外事件只保留在 audit CSV；
- incomplete events 默认不写入 target；
- 精确微秒时间保留在 Board events CSV，避免 sample quantization 丢失。

---

## 15. Result 和 Dataclass 兼容设计

### 15.1 保留现有类型

```text
SegmentedSample
UserActionSegmentationResult
SegmentationOutputPaths
```

继续仅用于 label-only 模式。不得向现有 dataclass 添加必填 Board 字段。

### 15.2 新增 Aligned-mode 类型

```python
@dataclass(frozen=True, slots=True)
class BoardEventSegmentedSample:
    imu: np.ndarray
    board_event_targets: np.ndarray
    label: str

    source_label_index: int
    sample_count: int

    start_sample_index: int
    stop_sample_index_exclusive: int

    label_timestamp_us: float
    next_label_timestamp_us: float | None

    first_press_timestamp_us: float
    last_lift_timestamp_us: float

    provisional_start_timestamp_us: float
    provisional_end_timestamp_us: float

    final_start_timestamp_us: float
    final_end_timestamp_us: float

    valid_touch_pair_count: int
    transient_touch_pair_count: int
    incomplete_touch_pair_count: int
    crossing_touch_pair_count: int

    boundary_source: str
    boundary_collision_resolved: bool
```

```python
@dataclass(frozen=True, slots=True)
class SkippedBoardEventSegment:
    source_label_index: int
    label: str
    label_timestamp_us: float
    next_label_timestamp_us: float | None
    skip_reason: str
```

新增：

```text
BoardEventSegmentationResult
BoardEventSegmentationOutputPaths
BoardEventUserActionSegmentationResult
```

---

## 16. 输出目录隔离

### 16.1 Label-only 输出

路径和文件名保持不变：

```text
outputs/segmentedIMU/
└── user_0/
    └── action_0/
        ├── user_0_action_0_rawIMU.npy
        ├── user_0_action_0_labels.npy
        ├── user_0_action_0_segment_offsets.npy
        ├── user_0_action_0_segment_lengths.npy
        ├── user_0_action_0_segments.csv
        └── user_0_action_0_segmentation_summary.json
```

### 16.2 Aligned-board-events 输出

进入独立方法目录：

```text
outputs/segmentedIMU/
└── aligned_board_events/
    └── user_0/
        └── action_0/
            ├── user_0_action_0_rawIMU.npy
            ├── user_0_action_0_labels.npy
            ├── user_0_action_0_segment_offsets.npy
            ├── user_0_action_0_segment_lengths.npy
            ├── user_0_action_0_board_event_targets.npy
            ├── user_0_action_0_segments.csv
            ├── user_0_action_0_board_events.csv
            ├── user_0_action_0_segmentation_summary.json
            ├── 0_ring_0_segmentation_verification.png
            ├── 1_ring_0_segmentation_verification.png
            └── ...
```

Verification 图片必须与该方法最终 arrays、CSV 和 summary 位于同一目录。不得写回 `data/` 或 `data_sample/`。

---

## 17. Contiguous Variable-Length Storage

两种模式均继续使用：

```python
raw_imu.shape == (total_samples, 6)
segment_offsets.shape == (N + 1,)
segment_lengths.shape == (N,)
labels.shape == (N,)
```

Aligned mode 额外要求：

```python
board_event_targets.shape == (total_samples, 4)
```

第 `i` 个 segment：

```python
start = segment_offsets[i]
stop = segment_offsets[i + 1]

segment_imu = raw_imu[start:stop]
segment_label = labels[i]
```

Aligned mode：

```python
segment_events = board_event_targets[start:stop]
```

本阶段不进行 padding 或 truncation。固定形状模型输入应由后续独立 preprocessing 阶段生成。

---

## 18. Manifest 设计

### 18.1 Label-only Manifest

保持当前列和语义不变，不加入伪造 Board 字段。

### 18.2 Aligned-mode Manifest

推荐在同一个 manifest 中同时保存 exported 和 skipped labels，以保持 source label 数量可审计。

字段至少包括：

```text
segment_index
exported
skip_reason

user
action
dataset_id
source_label_index
label

label_timestamp_us
next_label_timestamp_us

alignment_offset_us
alignment_event_coverage_ratio

first_press_timestamp_us
last_lift_timestamp_us

provisional_start_timestamp_us
provisional_end_timestamp_us
final_start_timestamp_us
final_end_timestamp_us

start_sample_index
stop_sample_index_exclusive
sample_count

valid_touch_pair_count
transient_touch_pair_count
incomplete_touch_pair_count
crossing_touch_pair_count

boundary_source
boundary_collision_resolved

ring_source_path
label_source_path
offset_source_path
segmentation_verification_path
```

对于 skipped label：

```text
exported = false
segment_index = empty
sample indices = empty
final boundaries = empty
skip_reason = explicit reason
```

---

## 19. Board Events CSV

输出：

```text
user_0_action_0_board_events.csv
```

字段至少包括：

```text
user
action
dataset_id

event_index
event_type
paired_touch_index

frame_timestamp_raw
aligned_event_timestamp_us
aligned_event_elapsed_s

valid_touch
transient
incomplete_touch
crossing_touch
duration_frames

source_global_frame_index
used_for_segment_boundary
assigned_segment_index
event_target_channel
```

规则：

- valid、transient 和 incomplete events 全部写入；
- crossing events 明确标记；
- incomplete events 的 `used_for_segment_boundary = false` 且 `event_target_channel` 为空；
- segment 外事件仍写入 CSV；
- `assigned_segment_index` 仅在事件确实属于成功导出的 segment 时填写。

---

## 20. Summary 设计

### 20.1 两种模式共有字段

```json
{
  "boundary_mode": "label"
}
```

或：

```json
{
  "boundary_mode": "aligned_board_events"
}
```

### 20.2 Label-only Summary

其余字段保持当前格式，不伪造 Board/alignment 字段。

### 20.3 Aligned-mode Summary

至少增加：

```text
alignment_required
alignment_offset_root
pre_press_context_us
post_lift_context_us

source_label_count
exported_segment_count
skipped_segment_count
skipped_segment_counts_by_reason

valid_touch_pair_count
transient_touch_pair_count
incomplete_touch_pair_count
crossing_touch_pair_count

board_event_target_channels
boundary_source_counts

recording_count
verification_image_count
recordings_without_verification
```

成功完成时：

```text
verification_image_count == recording_count
recordings_without_verification == []
```

---

## 21. Verification 行为按模式区分

### 21.1 Aligned-board-events 模式

Verification 是强制输出。每个 `{dataset_id}_ring_0.bin` 生成：

```text
{dataset_id}_ring_0_segmentation_verification.png
```

图片必须覆盖完整 Ring recording。

图片生成失败时：

- 当前 recording 不标记为成功；
- 不发布该 recording 的部分 aggregate 数据；
- 不留下空图片；
- 返回明确错误。

### 21.2 Label-only 模式

默认不生成 verification PNG。

可选参数：

```text
--write-label-verification
```

该图只显示 transient score、timestamp labels、label-only segment 背景和 skipped labels，不需要 Board 或 alignment。

可选 overlay：

```text
--overlay-aligned-board-events
```

该参数只影响绘图，不改变 label-only boundaries。若缺少 alignment，应报告绘图配置错误，不得改变 segmentation mode。

---

## 22. Verification 时间轴和布局

统一使用 Ring elapsed time：

```python
ring_start_us = float(ring_timestamps_us[0])
ring_elapsed_s = (
    ring_timestamps_us - ring_start_us
) / 1_000_000.0
```

Board events、labels 和 segments 都相对于同一个 `ring_start_us` 转换为 elapsed seconds。

完整 recording 每 10 秒一个 panel：

```python
panel_count = max(
    1,
    math.ceil(recording_duration_s / config.panel_duration_s),
)
```

例如 53.4 秒：

```text
0–10 s
10–20 s
20–30 s
30–40 s
40–50 s
50–53.4 s
```

纵向排列：

```python
figure, axes = plt.subplots(
    panel_count,
    1,
    figsize=(16, 3.0 * panel_count),
    sharey=True,
    layout="constrained",
)
```

---

## 23. Verification 显示语义

### 23.1 Transient Score

显示完整 recording 六轴 transient score：

```text
连续实线
linewidth ≈ 0.8
```

显示上限：

```python
y_max = np.quantile(transient_score, 0.995) * 1.1
```

仅限制显示范围，不修改原始 score。

### 23.2 Board Events

所有 Board event markers 使用虚线。

```text
Valid press:      红色虚线，linewidth 1.5，alpha 0.80
Valid lift:       橙色虚线，linewidth 1.5，alpha 0.80
Transient press:  红色虚线，linewidth 0.6，alpha 0.35
Transient lift:   橙色虚线，linewidth 0.6，alpha 0.35
```

颜色区分 press/lift，线宽区分 valid/transient。Incomplete events 默认不显示在主图中。

### 23.3 Labels

成功导出 segment 的 label：

```text
灰色虚线
linewidth = 0.8
alpha = 0.65
```

Label line 只表示 `timestamp.txt` marker，不表示 segment start。

Skipped label：

```text
红色实线
linewidth = 1.4
alpha = 0.90
```

显示：

```text
a [skipped:no_complete_touch_pair]
```

### 23.4 Exported Segments

成功 segment 使用低透明度背景：

```python
axis.axvspan(
    visible_segment_start_s,
    visible_segment_end_s,
    alpha=config.segment_alpha,
)
```

要求：

- 左边缘对应 final start；
- 右边缘对应 final end；
- 不绘制独立 segment start/end 竖线；
- 不显示 provisional boundaries；
- 可标注 `003:a`；
- segment label 文字只显示一次。

### 23.5 跨 Panel Segment

```python
visible_start = max(segment_start_s, panel_start_s)
visible_stop = min(segment_end_s, panel_stop_s)
```

若 `visible_start < visible_stop`，则绘制该 panel 内的 segment 部分。

### 23.6 Panel 边界去重

除最后一个 panel 外使用半开区间：

```text
[panel_start, panel_stop)
```

最后一个 panel：

```text
[panel_start, recording_end]
```

位于恰好 `20.0 s` 的 event、label 或 annotation 只能显示在 `20–30 s` panel。

### 23.7 Legend

只在第一个 panel 显示：

```text
Ring transient score
Valid Board press
Valid Board lift
Transient Board press
Transient Board lift
Timestamp label
Skipped label
Exported segment
```

`Exported segment` 使用 rectangle patch。

---

## 24. Verification 输入验证

生成图片前必须验证：

```text
Ring timestamps 非递减
Ring timestamps 与 transient score 长度相同
aligned Board timestamps 有限
labels 严格递增
final_start < final_end
segments 按时间排序
相邻 segments 不重叠
sample indices 位于 Ring 范围内
alignment offset 有限
```

发现 overlap 时必须报错：

```python
raise SegmentationVerificationError(
    "final segmentation windows overlap"
)
```

不得仅将错误边界画出来。

Verification 函数不得重新估计 offset、重新计算 segmentation boundaries、修改 caller-owned DataFrame 或修改原始 timestamps。

---

## 25. Recording-level Aligned API

新增：

```python
def segment_recording_by_aligned_board_events(
    *,
    ring_imu: np.ndarray,
    ring_timestamps_us: np.ndarray,
    labels: Sequence[SegmentLabel],
    aligned_board_events: pd.DataFrame,
    aligned_touch_pairs: pd.DataFrame,
    config: BoardEventSegmentationConfig,
) -> BoardEventSegmentationResult:
    ...
```

职责：

1. 校验 inputs；
2. 执行现有 label validity checks；
3. 为每个 label 分配 eligible pairs；
4. 生成 provisional windows；
5. 解析相邻冲突；
6. 生成 final windows；
7. 转换 sample indices；
8. 提取 variable-length IMU；
9. 生成 event targets；
10. 生成 exported/skipped metadata。

该 API 不负责读取磁盘文件、读取 offset、重新运行 alignment、写出图片或聚合多个 recordings。

---

## 26. User/Action-level Aligned API

新增：

```python
def segment_user_action_by_aligned_board_events(
    *,
    data_root: Path,
    user: str,
    action: str,
    output_root: Path,
    alignment_offset_root: Path,
    config: BoardEventSegmentationConfig,
    verification_config: SegmentationVerificationConfig,
    overwrite: bool = False,
) -> BoardEventUserActionSegmentationResult:
    ...
```

职责：discovery，Ring、labels、Board 和 offset 加载，identity validation，完整 Board event detection，recording-level segmentation，verification figure，aggregate arrays，manifest、events CSV、summary 和 transactional publication。

---

## 27. 每个 Recording 的完整 Aligned 流程

```text
发现 recording
→ 加载 ring_0
→ 加载 timestamp labels
→ 执行 label validity checks
→ 读取并验证 alignment offset
→ 加载完整 Board recording
→ 删除 backward-jump stale tail
→ 检测完整 press/lift events 和 touch pairs
→ 将全部 Board events 映射到 Ring 时间
→ 为每个 label 选择 eligible valid touch pairs
→ 生成 provisional windows
→ 解析相邻 window overlap
→ 建立 final segment windows
→ 将 final boundaries 转为 Ring sample indices
→ 提取 variable-length IMU
→ 生成四通道 Board event targets
→ 生成 Board event audit rows
→ 计算完整 recording transient score
→ 生成 verification PNG
→ 验证 PNG 存在且非空
→ 将该 recording 加入 user/action 聚合结果
```

缺少 Board 或成功 offset 时：

- 默认返回明确错误；
- 不自动切换为 label-only；
- 不在 aligned output 中静默混入旧模式结果。

---

## 28. Transactional 输出发布策略

Aligned mode 的 verification 是成功条件。

每个 recording：

1. 在内存中计算 segmentation result；
2. 在内存中计算 event targets 和 audit rows；
3. 将 verification 写入临时路径；
4. 验证图片存在且文件大小大于 0；
5. 将 recording 标记为成功；
6. 将 recording 数据加入 aggregate buffers。

User/action 完成后：

1. 将 arrays、CSV 和 summary 写入临时目录；
2. 校验 offsets、lengths、targets 和 manifest 一致；
3. 校验 verification 数量等于 recording 数量；
4. 原子发布到最终方法目录。

若任一关键步骤失败：

- 不发布部分完成的 aggregate outputs；
- 不覆盖现有有效结果；
- 清理当前运行生成的临时文件；
- 返回明确错误。

---

## 29. CLI 设计

新增：

```text
--boundary-mode {label,aligned-board-events}
```

默认：

```text
label
```

### 29.1 Aligned-only 参数

```text
--alignment-offset-root PATH
--pre-press-context-seconds 0.2
--post-lift-context-seconds 0.2
--missing-event-policy {skip,fallback-to-label}
--crossing-touch-policy {skip,clip-at-label}
--verification-panel-seconds 10
--verification-dpi 200
--overwrite-verification
```

建议初始实现中只启用：

```text
crossing_touch_policy = skip
```

`clip-at-label` 可保留为后续扩展，只有在实现、测试和文档完整后才启用。

### 29.2 Label-only 可选参数

```text
--write-label-verification
--overlay-aligned-board-events
```

### 29.3 无效组合

CLI 应直接报错，而不是忽略参数。例如：

```text
--boundary-mode label
--pre-press-context-seconds 0.2
```

应报告该参数只适用于 `aligned-board-events`。

### 29.4 Aligned-mode 示例

```bash
python scripts/segment_ring_imu.py \
    --data-root data \
    --user user_0 \
    --action 0 \
    --output-root outputs/segmentedIMU \
    --boundary-mode aligned-board-events \
    --alignment-offset-root outputs/alignment/offsets
```

---

## 30. Verification 主接口

```python
def create_segmentation_verification_figure(
    *,
    ring_dataframe: pd.DataFrame,
    ring_timestamps_us: np.ndarray,
    transient_score: np.ndarray,

    aligned_board_events: pd.DataFrame | None,

    labels: Sequence[SegmentLabel],
    label_skip_reasons: Sequence[str | None],

    segmented_samples: Sequence[
        SegmentedSample | BoardEventSegmentedSample
    ],

    output_path: Path,
    config: SegmentationVerificationConfig,

    user: str,
    action: str,
    dataset_id: int,
    boundary_mode: str,
    alignment_offset_us: float | None,
) -> SegmentationVerificationResult:
    ...
```

其中：

- aligned mode 要求 `aligned_board_events` 和有限 offset；
- label mode 允许二者为 `None`；
- 图中仅使用 caller 提供的 final boundaries；
- 函数不得重算 segmentation。

---

## 31. 测试计划

### 31.1 Label-only Regression

使用修改前固定 fixtures，验证 raw IMU、labels、offsets、lengths 以及原 manifest 和 summary 关键字段完全一致。

### 31.2 Label 模式不访问 Board

仅提供：

```text
ring_0.bin
timestamp.txt
```

不提供 Board chunks 和 offset TXT。默认 CLI 必须成功。

通过 monkeypatch 验证：

```text
load_board()
read_alignment_offset_txt()
detect_board_events()
```

均未被调用。

### 31.3 Aligned 模式依赖检查

缺少 offset 时必须失败：

```text
alignment offset is required for aligned-board-events segmentation
```

缺少 Board 时必须失败并指出 recording identity。不得静默运行 label-only。

### 31.4 Event 分类

覆盖：

- valid touch；
- transient touch；
- incomplete press；
- independent lift；
- touch crossing label boundary；
- Board stale tail。

### 31.5 正常单 Touch

```text
label_i     = 10.0 s
press       = 10.5 s
lift        = 11.5 s
next_label  = 13.0 s
```

结果：

```text
start = 10.3 s
end   = 11.7 s
```

### 31.6 多个 Touch

```text
10.5/10.9
11.2/11.8
12.0/12.5
```

结果：

```text
start = 10.3 s
end   = 12.7 s
```

全部 valid events 写入 targets。

### 31.7 无相邻冲突

```text
current provisional end = 13.10 s
next provisional start  = 13.30 s
```

保留当前 end 为 `13.10 s`。

### 31.8 相邻冲突

```text
current provisional end = 13.40 s
next label               = 13.00 s
next provisional start   = 13.30 s
```

结果：

```text
current final end = 13.00 s
next final start  = 13.30 s
```

### 31.9 下一个 Segment 被跳过

当前 provisional end 超过 next label 时：

```text
current final end = next label timestamp
```

### 31.10 无完整 Touch

结果：

```text
segment skipped
reason = no_complete_touch_pair
```

Verification 显示红色实线 label marker，不显示 segment background。

### 31.11 Transient-only Interval

结果：

```text
segment skipped
reason = only_transient_touches
```

Transient press/lift 仍显示并写入 transient targets。

### 31.12 Crossing Touch

```text
press < next_label <= lift
```

默认：

```text
segment skipped
reason = touch_crosses_next_label
```

### 31.13 Event Target Mapping

验证：

- 四 channels 正确；
- event 不漏映射；
- event 不重复映射；
- event 不落到其 timestamp 之前的 sample；
- segment 外 event 仅进入 CSV；
- targets 行数等于 aggregate raw IMU 行数。

### 31.14 Offset 只应用一次

```text
Board event = 10.0 s
offset      = +0.5 s
```

Aligned event 必须为 `10.5 s`。

### 31.15 两种模式共存

依次运行 `label` 和 `aligned-board-events`，验证：

- label outputs 未修改；
- aligned outputs 位于方法目录；
- 两组 `rawIMU.npy` 可具有不同 lengths；
- 两组结果互不覆盖；
- summary 明确记录 boundary mode。

### 31.16 Verification

验证：

```text
53 s → 6 panels
20 s → 2 panels
7 s  → 1 panel
```

并检查完整 recording 覆盖、panel boundary 去重、线型语义、axvspan、跨 panel segment、输出路径和非空 PNG。

### 31.17 CLI 参数隔离

验证 aligned-only 参数不能用于 label mode，label-only plotting 参数不能改变 aligned boundaries，aligned mode 缺少 required arguments 时错误明确，默认 CLI 仍为 label mode。

---

## 32. 文件修改清单

### 32.1 保留并回归测试

```text
src/writingring/segmentation.py
tests/test_segmentation.py
```

### 32.2 新增

```text
src/writingring/board_event_segmentation.py
src/writingring/segmentation_verification.py

tests/test_board_event_segmentation.py
tests/test_segmentation_verification.py

docs/BOARD_EVENT_GUIDED_SEGMENTATION.md
```

### 32.3 修改

```text
src/writingring/__init__.py
scripts/segment_ring_imu.py
tests/test_segment_ring_imu_cli.py
README.md
```

### 32.4 保留 Label 文档

```text
docs/IMU_SEGMENTATION.md
```

用于描述当前 label-only 行为。

### 32.5 复用

```text
src/writingring/event_alignment.py
src/writingring/alignment_io.py
```

---

## 33. 分阶段实施顺序

### Phase 0：冻结 Label-only 基线

1. 为当前 label-only outputs 建立 regression fixtures。
2. 记录当前 public API、CLI 默认行为、manifest 和 summary schema。
3. 确认现有 segmentation tests 全部通过。

### Phase 1：建立 Aligned-mode 基础数据层

1. 新增 `BoardEventSegmentationConfig`。
2. 新增 Board event/touch pair 标准 schema。
3. 加入 alignment offset identity validation。
4. 实现完整 Board loading 和 stale-tail trimming。
5. 实现 valid/transient/incomplete/crossing touch 分类。

### Phase 2：实现 Recording-level Segmentation Core

1. 实现 label validity 复用。
2. 实现 eligible pair assignment。
3. 实现 provisional window 生成。
4. 实现两阶段 collision resolution。
5. 实现 final boundary invariants。
6. 实现 sample index conversion。
7. 实现 aligned sample 和 skipped result types。
8. 实现四通道 Board event targets。

### Phase 3：实现 User/Action Aggregation 和输出

 1. 新增 aligned user/action API。
 2. 新增方法隔离输出目录。
 3. 实现 aggregate arrays 和 offsets。
 4. 实现 aligned manifest。
 5. 实现 Board events CSV。
 6. 实现 aligned summary。

### Phase 4：实现 Verification

 1. 新增 verification path builder。
 2. 实现完整 recording、10 秒 panels。
 3. 实现 Board events、labels、segments 和 skipped annotations。
 4. 实现 verification input validation。
 5. 将 verification 设为 aligned recording 成功条件。
 6. 实现 transactional publication。

### Phase 5：CLI、文档和兼容测试

 1. 加入 `--boundary-mode` dispatch。
 2. 确认默认仍为 `label`。
 3. 拒绝无效参数组合。
 4. 更新 README 中两种运行命令。
 5. 更新 label-only 和 aligned-mode 独立文档。

### Phase 6：集成验证和批量运行

 1. 对 `user_0/action_0` 进行人工 verification 检查。
 2. 对边界冲突、跨 label touch 和 transient-only 情况人工抽查。
 3. 确认 label-only outputs 无变化。
 4. 批量运行全部 aligned recordings。
 5. 检查每个 recording 均有非空 verification PNG。

---

## 34. 完成标准

### 34.1 Label-only 模式

必须满足：

- 默认命令仍运行当前 label-only segmentation；
- 原函数签名不变；
- 原输出目录不变；
- 原数组格式和内容不变；
- 原 segment 顺序不变；
- 原 skip rules 不变；
- 不依赖 Board；
- 不依赖 alignment；
- 现有 tests 全部通过。

### 34.2 Aligned-board-events 模式

必须满足：

- 必须显式选择；
- 类别仍来自 `timestamp.txt`；
- label interval 仍提供硬隔离语义；
- Board events 来自完整有效 Board recording；
- offset 只应用一次；
- start 使用第一个 valid press 前 `0.2 s`；
- end 使用最后一个 valid lift 后 `0.2 s`；
- 相邻成功 segments 不重叠；
- overlap 时使用 next label timestamp 隔离；
- 下一个 segment 无效时当前 end 不越过 next label；
- 没有完整 valid touch 的 label 默认跳过；
- transient/bounce 不参与 segment boundary；
- valid/transient press/lift 分别写入四通道 targets；
- incomplete events 保留在 audit 中；
- 所有 Board events 写入 CSV；
- segments 保持 variable-length；
- 继续使用 contiguous raw IMU、offsets 和 lengths；
- aligned outputs 写入独立方法目录；
- 每个 recording 生成独立 verification PNG；
- verification 与 final boundaries 完全一致；
- verification 覆盖完整 recording；
- verification 与 arrays、CSV、summary 位于同一目录；
- verification 失败时不发布部分结果；
- 新增 unit、integration、CLI 和 plotting tests 全部通过。

### 34.3 两种模式共同要求

- 两套结果可同时存在；
- 不在同一 aggregate 文件中混合 boundary mode；
- summary 明确记录 boundary mode；
- 原始 Ring、Board 和 label 数据不被修改；
- 所有时间域转换和输出边界均可审计。
