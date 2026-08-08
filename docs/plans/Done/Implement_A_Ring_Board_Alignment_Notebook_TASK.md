# Task: Implement a Ring–Board Alignment Notebook

## Objective

新增一个 notebook：

```text
notebooks/align_ring_board.ipynb
```

用于：

1. 加载指定 WritingRing recording；
2. 从 Board 数据中检测 press event；
3. 在每个 Board press timestamp 附近提取 Ring IMU 窗口；
4. 显示六轴 IMU 信号，供人工选择对应的接触峰值；
5. 根据多组 Board press–IMU peak anchors 估计固定时间偏移；
6. 输出对齐诊断结果。

第一版只实现人工确认峰值和 fixed-offset alignment，不实现全自动 touch detection。

---

## Scope constraints

禁止：

* 修改现有接口协议；
* 重构其他组件；
* 升级依赖；
* 调整数据库；
* 修改无关代码或格式。

不得修改：

```text
vendor/
data/
data_sample/
```

不得改变现有 Ring 和 Board loader 的行为。

如发现范围外问题，只记录到 notebook 输出或报告中，不处理。

---

## 1. Load one recording

使用现有接口：

```python
recordings = discover_recordings(DATA_ROOT)

recording = select_recording(
    recordings,
    user=USER,
    action=ACTION,
    dataset_id=DATASET_ID,
)

ring_data = load_ring(recording)
board_data = load_board(recording)
```

要求：

* 只加载 `ring_0`；
* 不加载 `ring_1`；
* 保留原始 Ring 和 Board timestamp；
* 不重排、修复或删除原始数据。

---

## 2. Select the valid Board interval

提供配置：

```python
BOARD_START_TIMESTAMP = ...
BOARD_END_TIMESTAMP = ...
```

只保留满足以下条件的 Board frames 和 contacts：

```python
BOARD_START_TIMESTAMP <= frame_timestamp_raw <= BOARD_END_TIMESTAMP
```

Dataset 0 默认使用有效 Board 时间段，不使用旧的 stale tail。

如检测到 timestamp backward jump，只报告，不自动删除或修复。

---

## 3. Detect Board press events

使用 Board frame table 中的：

```text
contact_count
frame_timestamp_raw
```

定义 press：

```text
上一帧 contact_count == 0
当前帧 contact_count > 0
```

定义 lift：

```text
上一帧 contact_count > 0
当前帧 contact_count == 0
```

代码逻辑：

```python
occupied = frames["contact_count"].to_numpy() > 0
transition = np.diff(occupied.astype(np.int8), prepend=0)

press_indices = np.flatnonzero(transition == 1)
lift_indices = np.flatnonzero(transition == -1)
```

对每个 press 保存：

```text
event_index
global_frame_index
frame_timestamp_raw
chunk_index
contact_count
```

可将持续不足 3 个 Board frames 的接触标记为 transient，但不要静默删除。

---

## 4. Build a reconstructed Ring time axis

Ring raw timestamp 存在重复值，因此保留 raw timestamp，同时建立严格递增的 reconstructed timestamp。

计算 recording-specific sampling rate：

```python
ring_timestamps = ring_data.dataframe["timestamp"].to_numpy()
sample_count = len(ring_timestamps)

duration_s = (
    ring_timestamps[-1] - ring_timestamps[0]
) / 1_000_000.0

imu_sampling_rate_hz = (sample_count - 1) / duration_s
```

重建时间轴：

```python
ring_timestamp_reconstructed = (
    ring_timestamps[0]
    + np.arange(sample_count)
    * 1_000_000.0
    / imu_sampling_rate_hz
)
```

要求：

* 不覆盖原始 `timestamp`；
* 新列命名为 `ring_timestamp_reconstructed`；
* 在输出中明确该时间轴基于 microsecond interpretation 和观测采样率。

---

## 5. Locate each Board press in the IMU data

对于一个 Board press：

```python
board_press_timestamp = press_event["frame_timestamp_raw"]
```

初始使用：

```python
coarse_offset_us = 0.0
```

计算预计 IMU 时间：

```python
target_imu_timestamp = (
    board_press_timestamp
    + coarse_offset_us
)
```

找最近的 reconstructed IMU sample：

```python
center_index = int(
    np.argmin(
        np.abs(
            ring_timestamp_reconstructed
            - target_imu_timestamp
        )
    )
)
```

使用固定搜索半径：

```python
radius_samples = 2000
```

提取窗口：

```python
start = max(0, center_index - radius_samples)
stop = min(
    len(ring_data.dataframe),
    center_index + radius_samples + 1,
)

imu_window = ring_data.dataframe.iloc[start:stop].copy()
```

在约 200 Hz 下：

```text
2000 samples ≈ 10 seconds
```

因此窗口约为：

```text
Board press 预计位置前 10 秒
到
Board press 预计位置后 10 秒
```

总窗口最长约 20 秒。

---

## 6. Plot IMU candidate windows

对每个 Board press 绘制：

```text
acc_x
acc_y
acc_z
gyr_x
gyr_y
gyr_z
```

x 轴使用相对于预计 press 位置的秒数：

```python
relative_time_s = (
    ring_timestamp_reconstructed[start:stop]
    - target_imu_timestamp
) / 1_000_000.0
```

图中标记：

* 预计 Board press 位置：`0 s`；
* IMU sample index；
* Board press timestamp；
* 当前 coarse offset；
* 搜索窗口范围。

每个 event 保存一张图：

```text
outputs/alignment/candidate_windows/press_000.png
outputs/alignment/candidate_windows/press_001.png
...
```

---

## 7. Optional transient score

为辅助人工选择峰值，计算六通道 transient score。

```python
signal_columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
]

signals = ring_data.dataframe[signal_columns].to_numpy()

differences = np.diff(
    signals,
    axis=0,
    prepend=signals[[0]],
)

median = np.median(differences, axis=0)
mad = np.median(
    np.abs(differences - median),
    axis=0,
)

normalized = (
    differences - median
) / (1.4826 * mad + 1e-12)

transient_score = np.sqrt(
    np.sum(normalized ** 2, axis=1)
)
```

将 transient score 作为额外 subplot。

它只能用于推荐候选峰值，不得声称这是论文中的 touch detection 方法。

---

## 8. Manual alignment anchors

在 notebook 中提供一个手动配置 cell：

```python
MANUAL_ANCHORS = [
    {
        "board_event_index": 0,
        "imu_sample_index": 1500,
        "signal_name": "acc_z",
        "confidence": "high",
        "notes": "",
    },
]
```

每个 anchor 必须包含：

```text
board_event_index
board_timestamp_raw
imu_sample_index
imu_timestamp_reconstructed
signal_name
confidence
notes
```

验证：

* `board_event_index` 必须存在；
* `imu_sample_index` 必须在 Ring 范围内；
* 不允许重复使用同一个 Board event；
* 建议至少使用 5 个 anchors。

---

## 9. Estimate a fixed offset

对每个 anchor：

```python
offset_us = (
    imu_timestamp_reconstructed
    - board_timestamp_raw
)
```

最终固定 offset：

```python
fixed_offset_us = float(
    np.median(anchor_offsets_us)
)
```

残差：

```python
residual_us = (
    anchor_offsets_us
    - fixed_offset_us
)
```

报告：

```text
anchor_count
fixed_offset_us
median_absolute_residual_ms
maximum_absolute_residual_ms
p95_absolute_residual_ms
```

绘制：

```text
anchor elapsed time
vs.
alignment residual in milliseconds
```

第一版不实现 affine clock correction。

如残差明显随时间变化，只在报告中记录可能存在 clock drift，不处理。

---

## 10. Aligned overlay

使用：

```python
aligned_board_press_timestamp = (
    board_press_timestamp
    + fixed_offset_us
)
```

将 Board press 映射到 Ring reconstructed timestamp。

绘制一张总览图：

```text
Ring transient score
Board press mapped positions
manual IMU peak anchors
```

所有 x 轴使用 Ring reconstructed elapsed time。

---

## 11. Outputs

输出目录：

```text
outputs/alignment/
```

至少生成：

```text
board_press_events.csv
alignment_anchors.csv
alignment_report.json
anchor_residuals.png
aligned_event_overlay.png
candidate_windows/
```

`alignment_report.json` 至少包含：

```json
{
  "recording": {
    "user": "user_0",
    "action": "0",
    "dataset_id": 0
  },
  "ring_sampling_rate_hz": 0.0,
  "search_radius_samples": 2000,
  "approximate_search_radius_seconds": 10.0,
  "alignment_model": "constant_offset",
  "fixed_offset_us": 0.0,
  "anchor_count": 0,
  "median_absolute_residual_ms": 0.0,
  "maximum_absolute_residual_ms": 0.0,
  "warnings": []
}
```

---

## 12. Required checks

Notebook 最后验证：

```python
assert ring_data.source_path.name.endswith("_ring_0.bin")
assert radius_samples == 2000
assert np.all(np.diff(ring_timestamp_reconstructed) > 0)
assert all(
    0 <= anchor["imu_sample_index"] < len(ring_data.dataframe)
    for anchor in MANUAL_ANCHORS
)
```

同时检查所有输出文件存在且非空。

---

## Acceptance criteria

任务完成需满足：

1. Notebook 可选择一个 `user/action/dataset_id`；
2. 只加载 `ring_0`；
3. 能从 Board `contact_count` 检测 press；
4. 能建立严格递增的 Ring reconstructed time；
5. 每个 press 使用 `radius_samples = 2000` 提取 IMU 窗口；
6. 能绘制六轴 IMU 和 transient score；
7. 能通过 `MANUAL_ANCHORS` 指定 IMU peak；
8. 能计算 median fixed offset 和 residual；
9. 能导出 CSV、JSON 和 PNG；
10. 不修改原始数据、现有接口或其他组件；
11. 范围外问题只记录，不处理。
