# SpikeIMU Alignment 与双模式 Segmentation 修改计划

## 1. 总体目标

保留一个统一入口：

```text
scripts/segment_ring_imu.py
```

通过现有参数选择两种 segmentation 方向：

```text
--boundary-mode label
--boundary-mode aligned-board-events
```

当前 CLI 已经具备该分支结构：`label` 路径调用 `segment_user_action()`，`aligned-board-events` 路径调用 `segment_user_action_by_aligned_board_events()`。

在此基础上新增第二个独立维度，用于选择待分段的数据：

```text
--input-kind raw-ring
--input-kind spike-imu
```

推荐组合矩阵：

| `input-kind` | `boundary-mode`        | 实际边界来源                    | 是否要求 alignment |
| ------------ | ---------------------- | ------------------------- | -------------- |
| `raw-ring`   | `label`                | `<data_id>_timestamp.txt` | 否              |
| `raw-ring`   | `aligned-board-events` | 对齐后的 Board press/lift     | 是              |
| `spike-imu`  | `label`                | `<data_id>_timestamp.txt` | 否              |
| `spike-imu`  | `aligned-board-events` | 对齐后的 Board press/lift     | 是              |

关键原则：

```text
input-kind 决定“切什么数据”
boundary-mode 决定“在哪里切”
```

二者不能混为一个 CLI 选项。

---

## 2. 两种 segmentation 模式的权威定义

## 2.1 Label segmentation

调用链保持：

```text
segment_ring_imu.py
  → segment_user_action()
  → load_timestamp_labels()
  → segment_recording_by_labels()
```

边界来源始终为：

```text
<data_id>_timestamp.txt
```

对第 `i` 个 label：

```text
[label_i_timestamp, label_(i+1)_timestamp)
```

实际 sample index 为：

```python
start = np.searchsorted(
    timestamps_us,
    label_i_timestamp,
    side="left",
)

stop = np.searchsorted(
    timestamps_us,
    label_next_timestamp,
    side="left",
)
```

当前 `segment_recording_by_labels()` 已按此规则实现；最后一个 label 可根据 `include_last_label` 延伸到 recording 末尾。

因此：

```text
Label segmentation 不依赖 Ring–Board alignment。
Label segmentation 不使用 Board press/lift 生成边界。
Label segmentation 不因 Board event overlay 而改变 start/stop。
```

当前 summary 也明确记录：

```text
segment_i = [label_i_timestamp, label_(i+1)_timestamp)
```

### Label verification 中的 Board overlay

以下组合：

```text
--boundary-mode label
--write-label-verification
--overlay-aligned-board-events
--alignment-offset-root ...
```

只把对齐后的 Board events 绘制到 verification 图中。

当前 `_write_label_verifications()` 先完成 `segment_recording_by_labels()`，之后才可选地读取 alignment offset 和 Board events，并把它们传给 verification figure。函数注释也明确说明不会改变 label boundaries。

因此必须保持：

```python
segments_without_overlay == segments_with_overlay
```

包括：

```text
segment count
start_sample_index
stop_sample_index_exclusive
segment_offsets
segment_lengths
segmented feature values
```

全部完全相同。

---

## 2.2 Board-assist segmentation

调用链保持：

```text
segment_ring_imu.py
  → segment_user_action_by_aligned_board_events()
  → read alignment offset
  → align Board event tables
  → segment_recording_by_aligned_board_events()
```

该模式必须使用：

```text
--boundary-mode aligned-board-events
--alignment-offset-root PATH
```

当前 CLI 已要求 aligned mode 必须提供 alignment offset，而 label verification 相关参数不得用于 aligned mode。

最终边界来自：

```text
first aligned valid Board press - pre-press context
last aligned valid Board lift + post-lift context
```

之后执行 collision resolution，并通过 canonical Ring timestamp 转换成 sample index：

```python
start = np.searchsorted(
    timestamps_us,
    final_start_us,
    side="left",
)

stop = np.searchsorted(
    timestamps_us,
    final_end_us,
    side="left",
)
```

再使用同一 `[start:stop]` 同时切 feature values 和 Board event targets。

这里 labels 仍可用于划分字符区间、关联输出 label 和检查 Board event eligibility，但实际 segment start/stop 是 aligned Board events 生成的，不是直接采用相邻 label timestamps。

---

# 3. 最终处理流程

## 3.1 Label segmentation 路径

```text
raw Ring recording
→ gravity preprocessing
→ preprocessedIMU.npy
→ spike encoding
→ spikeIMU.npy
→ load <data_id>_timestamp.txt
→ 使用 label timestamps 计算 start/stop
→ 对 spikeIMU[start:stop, :] 切片
→ 输出 label-segmented SpikeIMU
```

不需要：

```text
Ring–Board alignment
alignment offset
Board data
transient peak detection
```

只有显式开启 Board overlay 时，才读取 alignment offset 和 Board data；这些信息只进入 verification renderer。

## 3.2 Board-assist segmentation 路径

```text
raw Ring recording
→ gravity preprocessing
→ preprocessedIMU.npy
→ spike encoding
→ spikeIMU.npy
→ 从 spikeIMU 尾部六轴计算 transient score
→ Ring–Board alignment
→ 导出带 SpikeIMU provenance 的 offset
→ 加载 aligned Board press/lift
→ 生成 Board-assist boundaries
→ 对同一个 spikeIMU[start:stop, :] 切片
→ 输出 Board-assist segmented SpikeIMU
```

---

# 4. CLI 设计

## 4.1 保留 `--boundary-mode`

不新增重复含义的 `--segmentation-mode`。

继续使用：

```text
--boundary-mode label
--boundary-mode aligned-board-events
```

因为该参数准确描述边界来源。

## 4.2 新增 `--input-kind`

```text
--input-kind raw-ring|spike-imu
```

建议默认：

```text
raw-ring
```

以保持现有行为。

### `raw-ring`

执行当前路径：

```text
load ring_0.bin
→ 根据 gravity CLI 参数 preprocessing
→ 生成 9-channel feature values
→ segmentation
```

允许：

```text
--gravity-removal-method
--sampling-rate
--low-pass-cutoff-hz
--madgwick-beta
--provisional
```

### `spike-imu`

执行：

```text
加载已经生成的 spikeIMU.npy
加载 spike metadata
加载 timestamps_us.npy
验证 schema、identity、sample count 和 hashes
→ segmentation
```

新增：

```text
--spike-root PATH
```

Spike 模式下禁止重新 preprocessing，因此下列参数若被显式设置应报错，而不是静默忽略：

```text
--gravity-removal-method
--low-pass-cutoff-hz
--madgwick-beta
--provisional
```

采样率应从 Spike metadata 读取；CLI 中显式提供的采样率只能作为一致性检查，不能覆盖 artifact metadata。

---

# 5. CLI 使用示例

## 5.1 SpikeIMU + label segmentation

```bash
python scripts/segment_ring_imu.py \
  --data-root data \
  --user user_0 \
  --action 0 \
  --input-kind spike-imu \
  --spike-root outputs/spikeEncoding/custom-wavelet \
  --boundary-mode label \
  --output-root outputs/segmentedSpikeIMU/label
```

该命令：

```text
不读取 alignment offset
不加载 Board recording
不计算 Ring–Board alignment
```

## 5.2 SpikeIMU + label verification

```bash
python scripts/segment_ring_imu.py \
  --data-root data \
  --user user_0 \
  --action 0 \
  --input-kind spike-imu \
  --spike-root outputs/spikeEncoding/custom-wavelet \
  --boundary-mode label \
  --write-label-verification \
  --output-root outputs/segmentedSpikeIMU/label
```

Verification transient score 使用：

```python
spike_imu[:, 15:21]
```

仍不需要 alignment。

## 5.3 SpikeIMU + label verification + Board overlay

```bash
python scripts/segment_ring_imu.py \
  --data-root data \
  --user user_0 \
  --action 0 \
  --input-kind spike-imu \
  --spike-root outputs/spikeEncoding/custom-wavelet \
  --boundary-mode label \
  --write-label-verification \
  --overlay-aligned-board-events \
  --alignment-offset-root outputs/alignment/spike-imu/offsets \
  --output-root outputs/segmentedSpikeIMU/label
```

Alignment 只用于绘图：

```text
Label boundaries unchanged.
Board events are diagnostic overlays only.
```

## 5.4 SpikeIMU + Board-assist segmentation

先运行 alignment：

```bash
python scripts/align_ring_board.py \
  --data-root data \
  --user user_0 \
  --action 0 \
  --dataset-id 0 \
  --input-kind spike-imu \
  --spike-root outputs/spikeEncoding/custom-wavelet \
  --offset-output-root outputs/alignment/spike-imu/offsets \
  --report-output-root outputs/alignment/spike-imu/reports \
  --verification-output-root outputs/alignment/spike-imu/verification
```

再运行 segmentation：

```bash
python scripts/segment_ring_imu.py \
  --data-root data \
  --user user_0 \
  --action 0 \
  --input-kind spike-imu \
  --spike-root outputs/spikeEncoding/custom-wavelet \
  --boundary-mode aligned-board-events \
  --alignment-offset-root outputs/alignment/spike-imu/offsets \
  --output-root outputs/segmentedSpikeIMU/aligned-board-events
```

---

# 6. 统一 SpikeIMU 输入契约

新增：

```text
src/writingring/recording_features.py
```

定义：

```python
@dataclass(frozen=True)
class RecordingFeatureInput:
    values: np.ndarray
    timestamps_us: np.ndarray

    user: str
    action: str
    dataset_id: int

    input_kind: str
    feature_schema: str
    channel_names: tuple[str, ...]
    units: tuple[str, ...]

    transient_channel_indices: tuple[int, ...]
    transient_channel_names: tuple[str, ...]

    values_path: Path
    metadata_path: Path
    timestamps_path: Path

    values_sha256: str
    metadata_sha256: str
    timestamps_sha256: str

    sampling_rate_hz: float
```

提供：

```python
load_raw_ring_features(...)
load_spike_imu_features(...)
```

`load_raw_ring_features()` 封装当前：

```text
load_ring()
→ preprocess_ring_imu()
→ raw Ring timestamps
```

`load_spike_imu_features()` 加载预生成 artifact，不再执行 gravity removal。

SpikeIMU schema：

```text
shape: (N, 21)

columns 0:15
    signed wavelet event channels

columns 15:18
    acceleration x/y/z, m/s²

columns 18:21
    gyro x/y/z, rad/s
```

验证：

```python
len(values) == len(timestamps_us)
values.shape[1] == 21
np.isfinite(values).all()
np.isfinite(timestamps_us).all()
np.all(np.diff(timestamps_us) >= 0)
```

当前 generic label input validator只接受 6 或 9 通道，因此必须改为由 feature loader 验证 schema，segmentation core 只验证二维、非空、finite 和 row count。

---

# 7. Canonical timestamp artifact

Preprocessing 阶段为每个 recording 输出：

```text
<data_id>_preprocessedIMU.npy
<data_id>_timestamps_us.npy
<data_id>_preprocessing.json
```

Spike encoding metadata 引用同一个：

```text
<data_id>_timestamps_us.npy
```

必须满足：

```python
timestamps_us[i]
↔ preprocessed_imu[i]
↔ spike_imu[i]
```

不允许：

```text
重新排序
重新采样
删除重复 timestamp
在 alignment 中用首尾点重建权威 timestamp
```

两种 segmentation 模式均使用这一个 timestamp artifact：

```text
label mode:
    label timestamp → searchsorted(canonical timestamps)

aligned-board-events mode:
    aligned Board timestamp → searchsorted(canonical timestamps)
```

因此，同一个时间轴同时支持两种边界策略。

---

# 8. Label segmentation 修改内容

## 8.1 泛化函数参数

将：

```python
segment_recording_by_labels(
    ring_imu=...,
    ring_timestamps_us=...,
)
```

逐步重命名为：

```python
segment_recording_by_labels(
    feature_values=...,
    timestamps_us=...,
)
```

为了避免一次性破坏现有调用，可先保留旧参数或增加兼容 wrapper。

核心函数不关心 feature 是 9 还是 21 通道，只执行：

```python
segment = feature_values[start:stop].copy()
```

## 8.2 `segment_user_action()` 使用统一 loader

修改为：

```python
feature_input = load_recording_features(
    recording,
    input_kind=input_kind,
    ...
)
```

然后：

```python
labels = load_timestamp_labels(recording.timestamp_path)

samples = segment_recording_by_labels(
    feature_values=feature_input.values,
    timestamps_us=feature_input.timestamps_us,
    labels=labels,
    config=config,
)
```

## 8.3 Label verification

当前 label verification 会再次：

```text
load raw Ring
→ preprocess
→ segment
→ 从 raw Ring DataFrame 计算 transient score
```

Spike 模式下应改为使用与正式 segmentation 完全相同的 `RecordingFeatureInput`。

Transient score：

```python
score = compute_transient_score_array(
    feature_input.values[
        :,
        feature_input.transient_channel_indices,
    ]
)
```

对 SpikeIMU：

```python
transient_channel_indices == (15, 16, 17, 18, 19, 20)
```

## 8.4 Overlay 隔离

建议将 label verification 分成两个阶段：

```python
label_result = segment_recording_by_labels(...)
verification_context = build_label_verification_context(...)
```

Board overlay 只能修改：

```text
verification_context.aligned_board_events
verification_context.alignment_offset_us
verification figure
```

不得修改：

```text
label_result.samples
label_result.start/stop indices
aggregate arrays
manifest boundaries
segment offsets
```

---

# 9. Board-assist segmentation 修改内容

Board-assist 方案维持上一版设计。

## 9.1 SpikeIMU alignment

修改 `align_ring_board.py` 支持：

```text
--input-kind raw-ring|spike-imu
--spike-root PATH
```

Spike 模式下：

```python
transient_score = compute_transient_score_array(
    spike_imu[:, 15:21]
)
```

前 15 个 spike event channels 不参与 alignment。

## 9.2 Alignment offset provenance

Spike alignment 输出 schema v2，记录：

```text
alignment_signal_source=spike-imu
feature_schema=signed_wavelet_events_plus_imu_v1
feature_values_sha256
feature_metadata_sha256
timestamp_sha256
timestamp_unit=microseconds
transient_channel_indices=15,16,17,18,19,20
spike_event_channels_used=false
```

## 9.3 Board-assist segmentation 的严格匹配

执行 segmentation 前验证：

```python
offset.user == feature_input.user
offset.action == feature_input.action
offset.dataset_id == feature_input.dataset_id

offset.feature_values_sha256
    == feature_input.values_sha256

offset.metadata_sha256
    == feature_input.metadata_sha256

offset.timestamp_sha256
    == feature_input.timestamps_sha256
```

任一不一致立即失败。

## 9.4 Board-assist 切片

最终：

```python
segment = spike_imu[start:stop, :]
targets = board_event_targets[start:stop]
```

必须满足：

```python
len(segment) == len(targets)
```

---

# 10. Label overlay 的 provenance 规则

虽然 overlay 不改变 segmentation 边界，但它仍然表达 Board events 在当前 Ring/Spike 时间轴上的位置。

因此，SpikeIMU label overlay 应要求：

```text
alignment offset 是由同一个 spikeIMU 产生
timestamp hash 与当前 SpikeIMU timestamp 相同
recording identity 相同
```

不匹配时应拒绝生成 overlay 图，但已经生成的 label segmentation 数据不应被删除或重算。

建议 CLI 执行顺序：

```text
1. 完成 label segmentation
2. 发布或暂存 segmentation artifacts
3. 若请求 verification，则构建 verification
4. 若 overlay provenance 失败，整个命令返回失败
```

为保持输出事务性，更稳妥的实现是先在 staging directory 完成 segmentation 和所有请求的 verification，全部成功后再统一发布。

---

# 11. 输出设计

## 11.1 Label + SpikeIMU

```text
outputs/segmentedSpikeIMU/label/
└── <user>/
    └── action_<action>/
        ├── <user>_action_<action>_spikeIMU.npy
        ├── <user>_action_<action>_labels.npy
        ├── <user>_action_<action>_segment_offsets.npy
        ├── <user>_action_<action>_segment_lengths.npy
        ├── <user>_action_<action>_segments.csv
        ├── <user>_action_<action>_segmentation_summary.json
        └── <dataset_id>_segmentation_verification.png
```

不输出：

```text
board_event_targets.npy
board_events.csv
```

除非未来专门增加纯 diagnostic sidecar；它们不属于 label segmentation 的必要输出。

## 11.2 Board-assist + SpikeIMU

```text
outputs/segmentedSpikeIMU/aligned-board-events/
└── <user>/
    └── action_<action>/
        ├── <user>_action_<action>_spikeIMU.npy
        ├── <user>_action_<action>_labels.npy
        ├── <user>_action_<action>_segment_offsets.npy
        ├── <user>_action_<action>_segment_lengths.npy
        ├── <user>_action_<action>_board_event_targets.npy
        ├── <user>_action_<action>_segments.csv
        ├── <user>_action_<action>_board_events.csv
        ├── <user>_action_<action>_segmentation_summary.json
        └── <dataset_id>_segmentation_verification.png
```

## 11.3 Summary 公共字段

```json
{
  "input_kind": "spike-imu",
  "boundary_mode": "label",
  "feature_schema": "signed_wavelet_events_plus_imu_v1",
  "channel_count": 21,
  "event_channel_slice": [0, 15],
  "acceleration_m_s2_channel_slice": [15, 18],
  "gyroscope_channel_slice": [18, 21],
  "padding_or_truncation": "disabled"
}
```

Label 模式增加：

```json
{
  "boundary_source": "timestamp_labels",
  "time_mapping": "segment_i = [label_i_timestamp, label_(i+1)_timestamp)",
  "alignment_required": false,
  "aligned_board_overlay": {
    "requested": false,
    "affects_boundaries": false
  }
}
```

Board-assist 模式增加：

```json
{
  "boundary_source": "aligned_board_events",
  "alignment_required": true,
  "alignment_input_hash_match_verified": true,
  "board_event_targets_present": true
}
```

---

# 12. CLI 参数验证

## `boundary-mode=label`

允许：

```text
--write-label-verification
--overlay-aligned-board-events
```

规则：

```text
--overlay-aligned-board-events
    requires --write-label-verification

--overlay-aligned-board-events
    requires --alignment-offset-root

--alignment-offset-root in label mode
    requires --overlay-aligned-board-events
```

这些规则与当前实现一致。

禁止 Board-assist boundary 参数：

```text
--pre-press-context-seconds
--post-lift-context-seconds
--missing-event-policy
--crossing-touch-policy
```

## `boundary-mode=aligned-board-events`

要求：

```text
--alignment-offset-root
```

禁止：

```text
--write-label-verification
--overlay-aligned-board-events
```

因为 aligned mode 自身已经生成 Board-guided verification。

## `input-kind=spike-imu`

要求：

```text
--spike-root
```

禁止显式 preprocessing 参数。

## `input-kind=raw-ring`

禁止：

```text
--spike-root
```

继续允许现有 gravity 参数。

---

# 13. 测试计划

## 13.1 CLI 模式矩阵

覆盖四种合法组合：

```text
raw-ring + label
raw-ring + aligned-board-events
spike-imu + label
spike-imu + aligned-board-events
```

覆盖非法组合：

```text
spike-imu without --spike-root
label + Board boundary options
aligned-board-events without offset root
label overlay without verification
label overlay without offset root
spike-imu + explicit gravity settings
raw-ring + --spike-root
```

## 13.2 Label 边界正确性

构造 timestamps 和 labels，验证：

```python
start_i == searchsorted(timestamps, label_i, side="left")
stop_i == searchsorted(timestamps, label_i_plus_1, side="left")
```

验证 21 通道精确切片：

```python
segment_i == spike_imu[start_i:stop_i, :]
```

验证：

```python
segment_i[:, :15]
    == spike_events[start_i:stop_i]

segment_i[:, 15:21]
    == accel_gyro[start_i:stop_i]
```

## 13.3 Overlay 不改变边界

分别运行：

```text
label segmentation without overlay
label segmentation with aligned Board overlay
```

断言：

```python
assert_array_equal(result_a.feature_values, result_b.feature_values)
assert_array_equal(result_a.labels, result_b.labels)
assert_array_equal(result_a.segment_offsets, result_b.segment_offsets)
assert_array_equal(result_a.segment_lengths, result_b.segment_lengths)
```

Manifest 中的：

```text
start_sample_index
stop_sample_index_exclusive
label_timestamp_us
next_label_timestamp_us
```

也必须完全相同。

## 13.4 Label 模式不加载 alignment

使用 mock 验证普通 label segmentation 不调用：

```text
read_recording_alignment_offset
load_board
align_board_event_tables
align_events_to_transient_peaks
```

## 13.5 Spike transient score

验证：

```python
score = compute_transient_score_array(spike_imu[:, 15:21])
```

修改前 15 个 event channels 后：

```text
transient score 不变
peak regions 不变
alignment offset 不变
```

## 13.6 Board-assist segmentation

验证：

```text
alignment 使用同一个 SpikeIMU
offset hash 匹配
timestamp hash 匹配
Board boundaries 正确
21-channel slice 精确
Board event targets 等长
```

## 13.7 Timestamp 测试

覆盖：

```text
nondecreasing timestamps
重复 timestamp
label 恰好落在重复 timestamp
aligned Board event 恰好落在重复 timestamp
label 和 Board 两种模式使用同一个 timestamp hash
```

## 13.8 向后兼容

现有以下行为必须保留：

```text
默认 input-kind=raw-ring
默认 boundary-mode=label
9-channel raw-ring label segmentation
9-channel raw-ring Board-assist segmentation
现有 output dtype 和 label filtering 规则
```

---

# 14. 代码修改范围

## 公共输入与 timestamp

```text
src/writingring/recording_features.py        新增
src/writingring/preprocessing_export.py
src/writingring/preprocessing_io.py
src/writingring/spike_encoding/io.py
src/writingring/spike_encoding/publication.py
```

## Alignment

```text
scripts/align_ring_board.py
src/writingring/event_alignment.py
src/writingring/alignment.py
src/writingring/alignment_io.py
src/writingring/alignment_verification.py
```

## 双模式 segmentation CLI

```text
scripts/segment_ring_imu.py
```

该文件负责：

```text
解析 input-kind
解析 boundary-mode
校验合法参数组合
创建统一 RecordingFeatureInput loader config
路由到 label 或 aligned-board-events implementation
```

## Label segmentation

```text
src/writingring/segmentation.py
src/writingring/segmentation_verification.py
```

## Board-assist segmentation

```text
src/writingring/board_event_segmentation.py
src/writingring/segmentation_verification.py
```

---

# 15. 文档修改

## `README.md`

加入四种 CLI 组合，并重点提供：

```text
SpikeIMU + label
SpikeIMU + aligned-board-events
```

## `docs/notes/IMU_SEGMENTATION.md`

明确：

```text
boundary-mode=label
    边界只来自 timestamp.txt

boundary-mode=aligned-board-events
    边界来自 aligned Board events

overlay-aligned-board-events
    只影响 label verification
    不影响 label segmentation
```

## `docs/notes/BOARD_EVENT_GUIDED_SEGMENTATION.md`

保留上一版 Board-assist 设计：

```text
Spike transient score
alignment provenance
Board press/lift boundaries
21-channel slicing
Board targets
```

## `docs/notes/SPIKE_ENCODING.md`

增加：

```text
SpikeIMU timestamp contract
segmentation consumer contract
transient channels 15:21
```

## `docs/notes/ALIGNMENT_OUTPUTS.md`

增加：

```text
SpikeIMU alignment
offset schema v2
input and timestamp hashes
label overlay is diagnostic only
```

## 新增

```text
docs/notes/SPIKE_SEGMENTATION_PIPELINE.md
```

用模式矩阵集中说明：

```text
input-kind × boundary-mode
```

---

# 16. 推荐 PR 拆分

## PR 1：统一 feature/timestamp 输入

内容：

```text
RecordingFeatureInput
timestamps_us artifact
SpikeIMU metadata/hash validation
raw-ring compatibility loader
```

## PR 2：SpikeIMU Ring–Board alignment

内容：

```text
array transient score
SpikeIMU alignment input
canonical timestamps
offset schema v2
alignment provenance tests
```

## PR 3：SpikeIMU label segmentation

内容：

```text
--input-kind
generic feature slicing
21-channel label outputs
Spike verification
Board overlay isolation
overlay-does-not-change-boundaries tests
```

## PR 4：SpikeIMU Board-assist segmentation

内容：

```text
same-artifact offset validation
21-channel Board-guided slicing
Board targets
transactional outputs
end-to-end tests
```

## PR 5：文档与兼容性

内容：

```text
README
notes
CLI examples
migration notes
legacy raw-ring regression tests
```

---

# 17. 最终验收条件

## Label segmentation

```python
segments = segment_by_labels(
    spike_imu,
    timestamps_us,
    labels,
)

segment_i == spike_imu[
    searchsorted(timestamps_us, label_i):
    searchsorted(timestamps_us, label_i_plus_1)
]
```

并满足：

```text
无需 alignment
无需 Board data
overlay 不改变任何 segmentation artifact
```

## Board-assist segmentation

```python
score = compute_transient_score_array(
    spike_imu[:, 15:21]
)

offset.feature_values_sha256
    == sha256(spike_imu)

offset.timestamp_sha256
    == sha256(timestamps_us)

segment_i
    == spike_imu[board_start_i:board_stop_i]
```

## CLI

一个 CLI 通过：

```text
--boundary-mode
```

明确选择：

```text
label segmentation
或
Board-assist segmentation
```

再通过：

```text
--input-kind
```

明确选择：

```text
现有 raw Ring/preprocessing 数据
或
预生成 SpikeIMU 数据
```

最终架构必须保证：

```text
boundary-mode 决定边界
input-kind 决定被切片的数据
verification overlay 永远不能反向修改边界
```
