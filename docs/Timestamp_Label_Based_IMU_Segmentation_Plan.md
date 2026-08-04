# Timestamp-Label-Based IMU Segmentation Plan

## 1. 目标

基于每个 recording 的 `{dataset_id}_timestamp.txt` 对 Ring IMU 数据进行切分。

每一行 label marker 定义一个 segment 的起点：

[
\text{segment}_i
================

[t_i,\ t_{i+1})
]

也就是：

* 当前 label 的 timestamp 是当前 segment 的开始；
* 下一个 label 的 timestamp 是下一个 segment 的开始；
* 当前 segment 不包含 timestamp 等于下一个 label timestamp 的 IMU samples；
* 最后一个 label 从其 timestamp 开始，延伸到该 recording 的 Ring 数据结尾。

每个 segment 最终固定为：

```text
600 samples × 6 IMU channels
```

其中：

```text
600 = 3 seconds × 200 Hz
6 = acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
```

对于同一个 user、同一个 action 下的所有 dataset，将所有 segments 按确定顺序合并，输出：

```python
subject_action_rawIMU.npy.shape == (N, 600, 6)
subject_action_labels.npy.shape == (N,)
```

其中：

* `N`：该 user/action 下所有 recording 产生的 segment 总数；
* `rawIMU[i]`：第 `i` 个固定长度 IMU segment；
* `labels[i]`：第 `i` 个 segment 对应的类别；
* `rawIMU[i]` 和 `labels[i]` 必须严格一一对应。

---

# 2. 数据来源

使用项目中已确认的 recording 结构：

```text
data/
└── user_0/
    └── 0/
        ├── 0_ring_0.bin
        ├── 0_timestamp.txt
        ├── 1_ring_0.bin
        ├── 1_timestamp.txt
        └── ...
```

同一个 recording 由以下身份唯一确定：

```text
user
action
dataset_id
```

当前计划只使用：

```text
{dataset_id}_ring_0.bin
```

不使用：

```text
{dataset_id}_ring_1.bin
```

Ring binary 每行有七个 `float64`：

```text
acc_x
acc_y
acc_z
gyr_x
gyr_y
gyr_z
timestamp
```

前六列作为模型输入，最后一列用于 segmentation。

Timestamp 文件格式为：

```text
{integer_timestamp} {single_character_label}
```

其中 marker timestamps 严格递增，并与 Ring timestamps 处于可比较的微秒级时间域。

---

# 3. 输出单位

输出单位定义为：

```text
一个 user × 一个 action
```

例如：

```text
user_0/action_0
```

下的所有 dataset：

```text
dataset 0
dataset 1
dataset 2
dataset 3
...
```

产生的 segments 合并为一个数组。

建议输出目录：

```text
outputs/segmentedIMU/
└── user_0/
    └── action_0/
        ├── user_0_action_0_rawIMU.npy
        ├── user_0_action_0_labels.npy
        ├── user_0_action_0_valid_lengths.npy
        ├── user_0_action_0_valid_mask.npy
        └── user_0_action_0_segments.csv
```

# 4. 固定 Segment 长度

配置：

```python
sampling_rate_hz = 200.0
segment_duration_s = 3.0
target_samples = 600
channel_count = 6
```

要求：

```python
target_samples = round(
    sampling_rate_hz * segment_duration_s
)
```

必须验证：

```python
target_samples == 600
```

原描述中的：

```text
不足 300 samples 时补零
```

应统一改为：

```text
不足 600 samples 时补零
```

否则输出 shape 会与 `(N, 600, 6)` 冲突。

---

# 5. Label 解析

新增结构：

```python
@dataclass(frozen=True, slots=True)
class SegmentLabel:
    timestamp_us: float
    label: str
    source_line_number: int
```

解析接口：

```python
def load_timestamp_labels(
    path: Path,
) -> tuple[SegmentLabel, ...]:
    ...
```

解析规则：

1. 忽略空行；
2. 可忽略以 `#` 开头的注释；
3. 第一列作为 timestamp；
4. 剩余文本作为 label；
5. timestamp 必须是有限数值；
6. label 不允许为空；
7. timestamps 必须严格递增；
8. malformed line 必须报告文件路径和行号；
9. 保留 label 的原始大小写。

例如：

```text
1720476263355268 a
1720476264859021 b
1720476267021453 c
```

解析为：

```python
[
    SegmentLabel(1720476263355268, "a", 1),
    SegmentLabel(1720476264859021, "b", 2),
    SegmentLabel(1720476267021453, "c", 3),
]
```

---

# 6. Ring 数据加载

加载：

```python
raw = np.fromfile(
    ring_path,
    dtype=np.float64,
)

ring = raw.reshape(-1, 7)
```

拆分：

```python
imu = ring[:, :6]
timestamps_us = ring[:, 6]
```

验证：

```text
Ring 文件存在
数值总数能被 7 整除
至少有一个 sample
所有值有限
timestamps 非递减
IMU shape 为 (M, 6)
```

注意：当前 Ring timestamps 可能重复，因此只能要求：

```python
timestamps_us[i + 1] >= timestamps_us[i]
```

不能要求严格递增。样例 Ring 数据存在大量重复 timestamp，因此 segmentation 实现必须能处理 duplicate timestamps。

---

# 7. Timestamp 到 Sample Index 的映射

不要简单使用“距离 timestamp 最近的 sample”，因为那可能选择 label timestamp 之前的 sample。

当前 label 定义的是 segment 开始，因此使用：

```python
start_index = np.searchsorted(
    ring_timestamps_us,
    label_timestamp_us,
    side="left",
)
```

其语义是：

```text
第一个 timestamp >= label timestamp 的 Ring sample
```

对于下一个 label：

```python
stop_index = np.searchsorted(
    ring_timestamps_us,
    next_label_timestamp_us,
    side="left",
)
```

Segment 使用半开区间：

```python
segment = imu[start_index:stop_index]
```

等价于：

```text
包含：
timestamp >= current_label_timestamp

不包含：
timestamp >= next_label_timestamp
```

因此 segment 的最后一个 sample 是：

```text
下一个 label timestamp 之前的最后一个 Ring sample
```

这比：

```python
nearest_index(next_timestamp) - 1
```

更稳定，尤其是在 Ring timestamps 存在重复值时。

---

# 8. 最后一个 Label

最后一个 label 没有下一个 timestamp。

第一版定义：

```python
last_stop_index = len(ring_timestamps_us)
```

因此：

```python
last_segment = imu[last_start_index:]
```

之后同样执行：

* 不足 600 samples：尾部补零；
* 超过 600 samples：保留前 600 samples。

需要在 metadata 中记录：

```text
is_last_label = true
segment_end_source = ring_recording_end
```

不能直接丢弃最后一个 label，否则每个 timestamp 文件会无条件损失一个类别样本。

---

# 9. Segment 长度标准化

对于原始 segment：

```python
raw_segment.shape == (L, 6)
```

创建：

```python
fixed_segment = np.zeros(
    (600, 6),
    dtype=output_dtype,
)
```

## 情况 A：长度正好为 600

```python
fixed_segment[:] = raw_segment
```

## 情况 B：长度小于 600

采用尾部 zero padding：

```python
fixed_segment[:L] = raw_segment
fixed_segment[L:] = 0
```

即：

```text
[真实 IMU samples][zero padding]
```

不要在 segment 前部补零，因为 label timestamp 定义的是动作开始。

## 情况 C：长度大于 600

保留 segment 的前 600 samples：

```python
fixed_segment[:] = raw_segment[:600]
```

即保留从 label timestamp 开始的前三秒。

同时记录：

```text
original_sample_count = L
valid_sample_count = min(L, 600)
was_padded = L < 600
was_truncated = L > 600
```

---

# 10. 为什么需要 Valid Length 和 Mask

只保存：

```text
rawIMU.npy
labels.npy
```

虽然可以训练，但模型无法判断：

```text
值为 0 是真实 IMU 值
还是 padding
```

因此强烈建议同时输出：

```python
valid_lengths.shape == (N,)
valid_mask.shape == (N, 600)
```

例如，一个原始长度为 423 的 segment：

```python
valid_lengths[i] = 423
```

对应 mask：

```text
[True × 423][False × 177]
```

生成：

```python
valid_mask = np.zeros(
    (600,),
    dtype=np.bool_,
)

valid_mask[:valid_sample_count] = True
```

对于 CNN，可以在 pooling 或 loss 中使用 mask；对于 RNN/Transformer，可以使用 sequence length 或 attention mask。

---

# 11. Label 数据类型

建议第一版保存原始字符串 label：

```python
labels_array = np.asarray(
    labels,
    dtype=np.str_,
)
```

例如：

```python
labels_array.shape == (N,)
labels_array[0] == "a"
```

不要在每个 user/action 内单独把 label 转成整数，因为可能出现：

```text
user_0: a → 0, b → 1
user_1: b → 0, c → 1
```

导致类别 ID 跨 participant 不一致。

如果模型要求整数 label，应在所有数据处理完成后，建立一个全局 label map：

```json
{
  "a": 0,
  "b": 1,
  "c": 2
}
```

并保存：

```text
global_label_map.json
```

推荐同时保留原始字符串 label，便于审计。

---

# 12. 多 Dataset 合并顺序

同一个 user/action 下可能有多个 dataset。

必须使用确定顺序：

```text
dataset_id 从小到大
→ 每个 dataset 内按 timestamp.txt 行顺序
```

例如：

```text
dataset 0:
    label 0
    label 1
    ...
    label 25

dataset 1:
    label 0
    label 1
    ...
    label 25
```

最终数组顺序：

```text
dataset 0 的全部 segments
接着 dataset 1 的全部 segments
接着 dataset 2 的全部 segments
...
```

不要依赖：

```python
Path.glob()
os.listdir()
```

返回的未排序顺序。

应显式解析 dataset ID 并排序：

```python
recordings = sorted(
    recordings,
    key=lambda recording: recording.dataset_id,
)
```

---

# 13. 输出 Shape

完成一个 user/action 后：

```python
raw_imu_array = np.stack(
    all_fixed_segments,
    axis=0,
)
```

预期：

```python
raw_imu_array.shape == (
    total_segment_count,
    600,
    6,
)
```

Labels：

```python
labels_array = np.asarray(
    all_labels,
    dtype=np.str_,
)
```

预期：

```python
labels_array.shape == (
    total_segment_count,
)
```

必须验证：

```python
raw_imu_array.shape[0] == labels_array.shape[0]
raw_imu_array.shape[1:] == (600, 6)
valid_lengths.shape == (N,)
valid_mask.shape == (N, 600)
```

---

# 14. 输出 Dtype

Ring 原始文件使用：

```text
float64
```

但 ML 通常使用：

```text
float32
```

建议配置：

```python
output_dtype = np.float32
```

优点：

* 存储空间约减半；
* PyTorch、TensorFlow 默认训练通常使用 float32；
* 不需要模型加载后再次转换。

如果目标是完全保持原始数值精度，可以配置：

```python
output_dtype = np.float64
```

但必须在 metadata 中记录。

---

# 15. Segment Metadata

建议同时输出：

```text
user_0_action_0_segments.csv
```

每一行对应一个 segment：

```text
segment_index
user
action
dataset_id
label_index
label
label_timestamp_us
next_label_timestamp_us
start_sample_index
stop_sample_index_exclusive
first_sample_timestamp_us
last_sample_timestamp_us
original_sample_count
valid_sample_count
was_padded
was_truncated
is_last_label
ring_source_path
label_source_path
```

示例：

```text
0,user_0,0,0,0,a,1720476263355268,1720476264859021,1686,1984,...
```

该 manifest 用于：

* 检查 segmentation 是否正确；
* 定位异常 segment；
* 将模型预测追溯到原 recording；
* 统计 padding/truncation 比例；
* 后续可视化。

---

# 16. 建议代码结构

新增：

```text
src/writingring/segmentation.py
```

主要数据结构：

```python
@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    sampling_rate_hz: float = 200.0
    segment_duration_s: float = 3.0
    target_samples: int = 600
    channel_count: int = 6
    output_dtype: str = "float32"
    padding_value: float = 0.0
    include_last_label: bool = True
```

```python
@dataclass(frozen=True, slots=True)
class SegmentedSample:
    imu: np.ndarray
    label: str
    valid_length: int
    original_length: int
    start_sample_index: int
    stop_sample_index_exclusive: int
    label_timestamp_us: float
    next_label_timestamp_us: float | None
    was_padded: bool
    was_truncated: bool
```

主要接口：

```python
def segment_recording_by_labels(
    *,
    ring_imu: np.ndarray,
    ring_timestamps_us: np.ndarray,
    labels: Sequence[SegmentLabel],
    config: SegmentationConfig,
) -> tuple[SegmentedSample, ...]:
    ...
```

聚合接口：

```python
def segment_user_action(
    *,
    data_root: Path,
    user: str,
    action: str,
    output_root: Path,
    config: SegmentationConfig,
    overwrite: bool = False,
) -> UserActionSegmentationResult:
    ...
```

---

# 17. 核心算法伪代码

```python
all_segments = []
all_labels = []
all_lengths = []
all_masks = []
metadata_rows = []

for recording in sorted_recordings:
    ring = load_ring_0(recording.ring_path)
    imu = ring[:, :6]
    ring_timestamps = ring[:, 6]

    markers = load_timestamp_labels(
        recording.timestamp_path
    )

    for marker_index, marker in enumerate(markers):
        start_index = np.searchsorted(
            ring_timestamps,
            marker.timestamp_us,
            side="left",
        )

        if marker_index + 1 < len(markers):
            next_timestamp = (
                markers[marker_index + 1].timestamp_us
            )
            stop_index = np.searchsorted(
                ring_timestamps,
                next_timestamp,
                side="left",
            )
        else:
            next_timestamp = None
            stop_index = len(ring_timestamps)

        raw_segment = imu[start_index:stop_index]
        original_length = len(raw_segment)
        valid_length = min(original_length, 600)

        fixed_segment = np.zeros(
            (600, 6),
            dtype=np.float32,
        )

        fixed_segment[:valid_length] = (
            raw_segment[:valid_length]
        )

        mask = np.zeros(600, dtype=np.bool_)
        mask[:valid_length] = True

        all_segments.append(fixed_segment)
        all_labels.append(marker.label)
        all_lengths.append(valid_length)
        all_masks.append(mask)

        metadata_rows.append(...)
```

最终：

```python
raw_imu = np.stack(all_segments)
labels = np.asarray(all_labels, dtype=np.str_)
valid_lengths = np.asarray(
    all_lengths,
    dtype=np.int32,
)
valid_mask = np.stack(all_masks)
```

---

# 18. CLI

新增：

```text
scripts/segment_ring_imu.py
```

命令示例：

```bash
conda run --no-capture-output -n writingring-viz \
    python scripts/segment_ring_imu.py \
    --data-root data \
    --user user_0 \
    --action 0 \
    --output-root outputs/segmentedIMU \
    --sampling-rate 200 \
    --segment-seconds 3 \
    --target-samples 600
```

允许覆盖：

```bash
--overwrite
```

建议支持：

```text
--output-dtype {float32,float64}
--exclude-last-label
--no-mask
```

默认行为：

```text
使用 ring_0
包含最后一个 label
尾部 zero padding
超过 600 时截断尾部
输出 float32
输出 valid lengths、mask 和 manifest
默认不覆盖
```

---

# 19. 输出示例

对于：

```text
user_0
action 0
4 个 dataset
每个 timestamp.txt 有 26 个 labels
```

如果所有 markers 均有效：

```python
N = 4 * 26
N = 104
```

输出：

```python
user_0_action_0_rawIMU.npy.shape
# (104, 600, 6)

user_0_action_0_labels.npy.shape
# (104,)

user_0_action_0_valid_lengths.npy.shape
# (104,)

user_0_action_0_valid_mask.npy.shape
# (104, 600)
```

注意：`N=104` 只是根据当前样例文件数量推导出的示例。实际实现必须根据成功解析并保留的 segments 动态确定。

---

# 20. 数据质量检查

每个 recording 需要检查：

```text
label timestamps 严格递增
label timestamp 位于 Ring 时间范围内
start_index < len(ring)
stop_index >= start_index
segment 至少包含一个 sample
Ring timestamps 非递减
Ring IMU 全部有限
```

建议默认严格模式：

* label timestamp 早于 Ring start：报错；
* label timestamp 晚于 Ring end：报错；
* segment 长度为 0：报错；
* duplicate label timestamp：报错；
* label 数量为 0：报错；
* Ring 文件缺失：报错；
* 对应 timestamp 文件缺失：报错。

不要静默跳过异常 segment，否则 `N` 的变化不容易被发现。

如果需要批量处理容错，可增加：

```text
--skip-invalid-recordings
```

但必须在最终 summary 中列出所有被跳过的文件及原因。

---

# 21. Segmentation Verification

建议为每个 user/action 生成一个 summary：

```text
user_0_action_0_segmentation_summary.json
```

内容：

```json
{
  "user": "user_0",
  "action": "0",
  "recording_count": 4,
  "segment_count": 104,
  "target_samples": 600,
  "channel_count": 6,
  "padded_segment_count": 104,
  "truncated_segment_count": 0,
  "minimum_original_length": 351,
  "maximum_original_length": 497,
  "median_original_length": 401,
  "output_dtype": "float32"
}
```

还建议随机绘制少量 segment：

```text
outputs/segmentationVerification/
└── user_0/
    └── action_0/
        ├── segment_0000_a.png
        ├── segment_0025_z.png
        └── segment_0052_A.png
```

图片应显示：

* 六轴 IMU；
* 真实 sample 区域；
* zero-padding 起点；
* label；
* dataset ID；
* label timestamp；
* original length。

---

# 22. 测试计划

新增：

```text
tests/test_segmentation.py
tests/test_segment_ring_imu_cli.py
```

## 时间边界测试

构造：

```python
ring_timestamps = [
    100,
    200,
    300,
    400,
    500,
]
```

Labels：

```text
200 a
400 b
```

预期：

```text
a segment 包含 timestamp 200、300
b segment 包含 timestamp 400、500
```

---

## Duplicate Timestamp 测试

构造：

```python
ring_timestamps = [
    100,
    200,
    200,
    200,
    300,
]
```

如果下一个 label timestamp 是 200，则：

```python
np.searchsorted(..., 200, side="left")
```

应排除所有 timestamp 等于 200 的 samples，不允许一个 sample 同时属于两个 segments。

---

## Padding 测试

原始长度：

```text
420
```

预期：

```text
前 420 samples 等于原数据
后 180 samples 全为 0
valid_length = 420
mask 前 420 为 True
mask 后 180 为 False
```

---

## Truncation 测试

原始长度：

```text
750
```

预期：

```text
只保留前 600 samples
was_truncated = true
original_length = 750
valid_length = 600
```

---

## 最后一个 Label 测试

最后一个 marker：

```text
从 marker timestamp 到 Ring recording end
```

之后执行 pad 或 truncate。

---

## 聚合顺序测试

确保输出顺序为：

```text
dataset 0 labels
dataset 1 labels
dataset 2 labels
```

而不是文件系统返回顺序。

---

## Shape 测试

必须满足：

```python
raw_imu.ndim == 3
raw_imu.shape[1:] == (600, 6)

labels.ndim == 1
labels.shape[0] == raw_imu.shape[0]

valid_lengths.shape == labels.shape
valid_mask.shape == (
    raw_imu.shape[0],
    600,
)
```

---

## Round-Trip 测试

保存后重新加载：

```python
loaded_imu = np.load(raw_imu_path)
loaded_labels = np.load(labels_path)
```

验证：

```python
np.array_equal(
    loaded_imu,
    original_imu,
)

np.array_equal(
    loaded_labels,
    original_labels,
)
```

加载 labels 时不应要求：

```python
allow_pickle=True
```

因此 labels 应使用 NumPy Unicode dtype，而不是 Python object dtype。

---

# 23. 文件修改清单

新增：

```text
src/writingring/segmentation.py
scripts/segment_ring_imu.py
tests/test_segmentation.py
tests/test_segment_ring_imu_cli.py
docs/IMU_SEGMENTATION.md
```

修改：

```text
src/writingring/__init__.py
README.md
```

原则上不修改：

```text
data/
data_sample/
vendor/
src/writingring/event_alignment.py
src/writingring/gravity.py
```

Segmentation 使用 timestamp markers 与 Ring IMU，不需要 Board 数据，也不需要 Ring–Board offset。

---

# 24. 实施顺序

1. 固定 segment 定义为 `[current_label, next_label)`。
2. 固定输出长度为 600 samples。
3. 实现 timestamp label parser。
4. 实现 Ring input validation。
5. 实现 timestamp-to-index mapping。
6. 实现最后一个 label 处理。
7. 实现 zero padding 和 truncation。
8. 实现 valid length 和 mask。
9. 实现单 recording segmentation。
10. 实现 user/action 多 dataset 聚合。
11. 实现 `.npy` 和 CSV manifest 输出。
12. 实现 CLI。
13. 增加边界、duplicate timestamp、padding 和 truncation tests。
14. 对 `user_0/action_0` 运行样例。
15. 检查输出 shape、label 顺序和 manifest。
16. 随机绘制 segments 进行人工验证。

---

# 25. 完成标准

实现完成后必须满足：

* 使用 `{dataset_id}_timestamp.txt` 的 label timestamp 作为 segment 起点；
* 当前 segment 在下一个 label timestamp 之前结束；
* 使用半开区间 `[current_timestamp, next_timestamp)`；
* 最后一个 label 延伸至 recording 结尾；
* 每个 segment 固定输出 `(600, 6)`；
* 不足 600 samples 时在尾部补零；
* 超过 600 samples 时保留前 600 samples；
* 同一个 user/action 下所有 dataset 的 segments 合并；
* 聚合顺序确定且可复现；
* 最终输出满足：

```python
rawIMU.shape == (N, 600, 6)
labels.shape == (N,)
```

* `rawIMU[i]` 与 `labels[i]` 严格对应；
* 同时保留 valid lengths、padding mask 和 segment manifest；
* 原始 Ring binary 和 timestamp 文件不被修改；
* 所有新增测试和现有测试通过。
