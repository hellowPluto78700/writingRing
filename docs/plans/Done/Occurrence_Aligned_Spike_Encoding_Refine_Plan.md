# Occurrence-Aligned Spike Encoding 修改计划

## 1. 目标

目标数据流为：

```text
preprocessed / gravity-removed rawIMU
→ recording-level Custom Wavelet encoding
→ occurrence-index alignment
→ 生成 21-channel spike IMU
→ 共享原 rawIMU 的 timestamp、label 和 segment metadata
```

本次只修改 spike encoding 相关代码，不修改：

```text
src/writingring/segmentation.py
src/writingring/board_event_segmentation.py
现有 label boundary 计算
现有 timestamp → sample index 映射
```

关键设计约束：

1. 每个 recording 独立编码和 reset。
2. recording 前后各增加约 `0.15 s` padding。
3. padding sample 数根据 sampling rate 和 extrema window 推导。
4. detector 确认的 event 回填至对应 occurrence index。
5. 忽略并且不补偿 IIR phase/group delay。
6. 接受 recording 边缘的 synthetic-padding assumption。
7. 保留全部 `3 axes × 5 frequencies = 15` 个 spike 通道。
8. 保留 local maximum 和 local minimum 的 signed amplitude。
9. 删除输出中的三个 `acceleration_*_g` 通道。
10. 保留原始三个 `acceleration_*` 和三个 `gyro_*` 通道。
11. 最终输出长度与原 rawIMU 完全一致。
12. timestamp、label、offset 和 sample index 不平移。

---

## 2. 输入与输出 schema

### 2.1 输入 rawIMU

当前 canonical rawIMU 是：

```text
shape = (N, 9)
```

通道顺序：

```text
0  acceleration_x_g
1  acceleration_y_g
2  acceleration_z_g

3  acceleration_x       # m/s²
4  acceleration_y
5  acceleration_z

6  gyro_x               # rad/s
7  gyro_y
8  gyro_z
```

Spike encoder 使用前三列：

```python
acceleration_g = raw_imu[:, 0:3]
```

当前 loader 本身已经按照这一规则加载 `(N,9)` 输入。

### 2.2 Custom Wavelet 原始输出

默认配置：

```text
3 axes × 5 frequency bands = 15 channels
```

频率：

```text
0.5, 1, 2, 4, 8 Hz
```

通道顺序保持现有的 axis-major、frequency-minor：

```text
0   event_x_0.5_hz
1   event_x_1_hz
2   event_x_2_hz
3   event_x_4_hz
4   event_x_8_hz

5   event_y_0.5_hz
6   event_y_1_hz
7   event_y_2_hz
8   event_y_4_hz
9   event_y_8_hz

10  event_z_0.5_hz
11  event_z_1_hz
12  event_z_2_hz
13  event_z_4_hz
14  event_z_8_hz
```

现有 encoder 已经按该顺序生成 15 个 channel names，不再进行任何频带合并或降维。

### 2.3 最终输出

从 rawIMU 中删除：

```text
acceleration_x_g
acceleration_y_g
acceleration_z_g
```

在相同逻辑位置插入 15 个 spike channels。

最终输出：

```text
shape = (N, 21)
```

通道顺序：

```text
0–14   15 个 signed spike channels

15     acceleration_x       # 原 rawIMU column 3
16     acceleration_y       # 原 rawIMU column 4
17     acceleration_z       # 原 rawIMU column 5

18     gyro_x               # 原 rawIMU column 6
19     gyro_y               # 原 rawIMU column 7
20     gyro_z               # 原 rawIMU column 8
```

构造方式：

```python
spike_imu = np.column_stack(
    (
        spike_events,       # (N, 15)
        raw_imu[:, 3:9],    # (N, 6)
    )
)
```

必须满足：

```python
spike_imu.shape == (N, 21)
np.array_equal(spike_imu[:, 15:21], raw_imu[:, 3:9])
```

原 `rawIMU.npy` 保持只读，不允许原地修改。

---

## 3. Local maximum 和 local minimum 语义

现有 `_LocalExtremaDetector` 对每个：

```text
axis × frequency band
```

计算时间和邻近频带窗口内的：

```python
maximum = local maximum
minimum = local minimum
```

然后检查窗口中心：

```python
centre == maximum
centre == minimum
```

当前实现返回：

```python
events = maximum * (maximum == centre) \
       + minimum * (minimum == centre)
```

因此：

* 局部最大值被保留；
* 局部最小值被保留；
* event 保存的是 wavelet response amplitude；
* event 不是 binary `0/1`；
* 正负 polarity 被保留；
* 非 extrema 位置为零。

输出 representation 继续使用：

```text
signed_sparse_wavelet_extrema
```

不执行：

```text
absolute value
binary threshold
positive/negative channel splitting
frequency reduction
axis reduction
```

### Plateau/tie 规则

本次修改应同时明确一个边界行为：

如果窗口中心同时等于 maximum 和 minimum，例如整个局部窗口是同一个非零常数，当前求和形式可能将同一个值计算两次。

建议将 tie policy 明确为：

```text
中心同时是 max 和 min
→ 只输出一次 centre value
```

而不是：

```text
centre + centre
```

建议实现：

```python
is_maximum = centre == maximum
is_minimum = centre == minimum

events = np.where(
    is_maximum | is_minimum,
    centre,
    0.0,
)
```

这仍然保留局部最大和局部最小，同时避免 nonzero plateau 被双倍计算。

---

## 4. Padding 长度

定义：

```text
extrema window duration = 0.3 s
half-window duration    = 0.15 s
```

padding 不能写死成 30 samples。

从 encoder 的实际 odd extrema window 推导：

```python
time_window_samples = encoder.max_filter_time_samples
padding_samples = time_window_samples // 2
```

默认 200 Hz：

```text
time_window_samples = 61
padding_samples = 30
actual padding duration = 30 / 200 = 0.15 s
```

其他 sampling rate 下，padding 自动变化。

例如：

```text
100 Hz → approximately 15 samples
200 Hz → 30 samples
400 Hz → approximately 60 samples
```

实际值必须以 detector 的 half-window 为准：

```python
padding_duration_actual = padding_samples / sampling_rate_hz
```

这样 padding 长度和 detector 确认延迟始终一致，不会因为整数舍入造成一行错位。

---

## 5. Padding 模式

采用前面已选定的策略 C：

```text
接受 recording boundary padding assumption
```

默认 padding mode：

```python
mode = "reflect"
```

即：

```python
padded_acceleration = np.pad(
    acceleration_g,
    pad_width=((padding_samples, padding_samples), (0, 0)),
    mode="reflect",
)
```

数据布局：

```text
[left reflect padding | original recording | right reflect padding]
```

不生成：

```text
valid_context_mask.npy
```

不删除 recording 的首尾事件。

不对 padding 造成的 synthetic boundary effect 作进一步补偿。

Metadata 必须明确：

```json
{
  "padding_mode": "reflect",
  "padding_boundary_assumption": "accepted",
  "boundary_validity_mask_emitted": false
}
```

---

## 6. Occurrence-index 回填

设：

```text
原 recording 长度 = N
padding samples   = H
padded 长度       = N + 2H
```

原始 sample `i` 在 padded sequence 中的位置：

```text
padded occurrence index = i + H
```

由于 detector 需要再等待 `H` 个未来样本，event 在 detection-aligned 输出中出现于：

```text
padded detection index = i + 2H
```

因此，causal encoder 先生成：

```text
detected_events.shape = (N + 2H, 15)
```

随后 occurrence alignment 为：

```python
occurrence_events = detected_events[
    2 * H : 2 * H + N
]
```

这等价于：

```python
occurrence_events[i] = detected_events[i + 2 * H]
```

最终：

```text
occurrence_events.shape = (N, 15)
```

并满足：

```text
occurrence_events[i]
↔ raw_imu[i]
↔ timestamp[i]
↔ label/segment boundary 中的 sample index i
```

### 为什么是 `2H`

一个原始事件经过了两次索引偏移：

```text
原始 sample i
→ 因 left padding 移到 i + H
→ detector 等待 H samples 后在 i + 2H 确认
```

所以从 detection-aligned padded 输出中取：

```text
[2H : 2H + N]
```

即可同时完成：

* occurrence 回填；
* 删除 left padding；
* 删除 right flush；
* 恢复原始 N 行坐标系。

---

## 7. Recording reset 规则

padding 和 reset 必须按 recording 执行，而不是按 label segment 执行。

每个 recording：

```text
encoder.reset()
→ left reflect padding
→ original recording
→ right reflect padding / flush
→ occurrence alignment
→ crop to N rows
```

如果一次编码多个 recording，需要提供 recording offsets：

```text
[0, recording_1_end, recording_2_end, ...]
```

runner 对每个 recording：

```python
for start, stop in zip(recording_offsets[:-1], recording_offsets[1:]):
    encoder.reset()
    encode_one_recording(acceleration_g[start:stop])
```

禁止使用 label segment offsets 作为 encoder reset boundary，否则会重新引入：

* 每个 label 开头 IIR reset；
* 每个 label 单独 synthetic padding；
* segment 尾部 flush artifact。

Metadata：

```json
{
  "sequence_boundary_semantics": "recording",
  "state_reset_boundary": "recording"
}
```

---

## 8. IIR 延迟处理

本次只补偿 extrema detector 的固定确认延迟。

不补偿：

```text
causal IIR wavelet filter phase delay
frequency-dependent group delay
IIR startup transient
```

Metadata 必须明确：

```json
{
  "extrema_detection_latency_compensated": true,
  "event_index_semantics": "wavelet_extrema_occurrence_index",
  "iir_delay_compensated": false,
  "iir_warmup_guaranteed": false
}
```

这里的 occurrence index 指：

```text
wavelet response local-extrema 的 occurrence index
```

而不是经过 IIR delay correction 后的原始物理运动发生时间。

---

## 9. Timestamp、label 和 segmentation metadata 共享

当前 rawIMU 数组本身不包含 timestamp 或 label。它们是通过旁路文件或 manifest 与行索引关联的。

“共享 rawIMU 的 timestamp、label 等”定义为：

```text
spikeIMU row i 与 rawIMU row i 指向相同 timestamp 和 label
```

Encoder 必须保证：

```python
len(spike_imu) == len(raw_imu)
len(spike_events) == len(raw_imu)
```

Encoder 不修改：

```text
timestamps
labels
segment_offsets
segment_lengths
segments CSV
dataset/recording ownership
```

不执行：

```text
timestamp + 0.15 s
label boundary + H
segment offset + H
```

padding 只存在于 encoder 内部。

最终 spike artifact 直接复用原始 sidecars：

```text
*_labels.npy
*_segment_offsets.npy
*_segment_lengths.npy
*_segments.csv
timestamp source / manifest
```

建议 spike summary 保存这些 source references：

```json
{
  "alignment": {
    "row_aligned_with_source_raw_imu": true,
    "sample_count_preserved": true,
    "timestamps_reused_without_shift": true,
    "labels_reused_without_shift": true,
    "segment_offsets_reused_without_shift": true
  }
}
```

---

## 10. 不修改 segmentation 的接口限制

这里存在一个必须明确的约束：

当前 segmentation 输入验证只接受：

```text
(M, 6) 或 (M, 9)
```

新的 spike IMU 是：

```text
(M, 21)
```

所以不能把 21-channel spike IMU 重新传入当前 segmentation API。当前代码会拒绝该 shape。

因此，在“不修改 segmentation”的约束下，本计划采用：

```text
不让 segmentation 重新处理 21-channel array
```

而是：

```text
1. 保证 spikeIMU 和 rawIMU 行数完全相同
2. 保证 occurrence index 完全对齐
3. 直接复用 rawIMU 已有的 timestamp、label 和 segment offsets
4. 使用相同 offsets 对 spikeIMU 做视图或切片
```

例如：

```python
spike_segment_i = spike_imu[
    segment_offsets[i] : segment_offsets[i + 1]
]
```

这不会修改 segmentation，也不会重新计算 boundary。

### 重要结论

以下两项无法同时成立：

```text
A. 将 (N,21) spikeIMU 直接送入当前 segmentation API
B. segmentation 源代码完全不修改
```

本计划选择 B，因此通过共享现有 offsets/labels 实现 segmentation 语义，而不是重新调用 segmenter 处理 21-channel input。

---

### 标注说明（新增）

⚠️ **后续计划会修改 segmentation logic，使其原生支持 spikeIMU (21-channel) 输入。**

但该变更 **不属于当前 spike encoder 设计范围**，当前阶段：

* segmentation API 保持不变
* 仅通过 offsets + slicing 实现语义对齐
* 不引入任何 segmentation 逻辑修改或适配层

该标注用于区分未来 refactor 与当前 encoder-only 改动边界。

---

## 11. 输出文件

建议 canonical 输出为：

```text
{stem}_spikeIMU.npy
```

形状：

```text
(N, 21)
```

建议同时保留纯 encoder 输出：

```text
{stem}_spikeEvents.npy
```

形状：

```text
(N, 15)
```

输出集合：

```text
{stem}_spikeEvents.npy
{stem}_spikeIMU.npy
{stem}_spike_recording_offsets.npy
{stem}_spike_sequences.csv
{stem}_spike_encoding_summary.json
```

原文件保持不变：

```text
{stem}_rawIMU.npy
{stem}_labels.npy
{stem}_segment_offsets.npy
{stem}_segment_lengths.npy
```

现有 publication 已经采用 staged、atomic publish，可以在该模块中增加 `spikeIMU.npy`，继续维持原子发布和 overwrite 检查。

---

## 12. 输出 channel names

`spikeIMU.npy` 的 channel schema：

```text
event_x_0.5_hz
event_x_1_hz
event_x_2_hz
event_x_4_hz
event_x_8_hz

event_y_0.5_hz
event_y_1_hz
event_y_2_hz
event_y_4_hz
event_y_8_hz

event_z_0.5_hz
event_z_1_hz
event_z_2_hz
event_z_4_hz
event_z_8_hz

acceleration_x
acceleration_y
acceleration_z

gyro_x
gyro_y
gyro_z
```

Units：

```text
event_*         signed wavelet response amplitude
acceleration_*  m/s²
gyro_*          rad/s
```

不能再声明：

```text
前三列单位为 g
前三列 × 9.80665 = acceleration_m_s2
```

新的 schema 建议命名：

```text
signed_wavelet_events_plus_imu_v1
```

---

## 13. 计划修改的文件

### 13.1 `src/writingring/spike_encoding/encoders/custom_wavelet.py`

修改 `encode_sequence()`：

```text
validate (N,3)
→ calculate H
→ reflect pad to N+2H
→ causal encode padded sequence
→ retain all 15 channels
→ select detection rows [2H:2H+N]
→ return occurrence-aligned (N,15)
```

保持：

```text
axis-major / frequency-minor channel order
signed local maximum and local minimum amplitudes
```

增加 metadata properties：

```text
occurrence_lookahead_samples
padding_samples_each_side
padding_duration_seconds
event_index_semantics
```

处理 max/min tie，避免同一个 centre 被重复相加。

### 13.2 `src/writingring/spike_encoding/runner.py`

保持 runner 对 channel 数量通用，不硬编码为 3。

验证：

```python
encoded.shape == (
    original_sequence_length,
    len(encoder.output_channel_names),
)
```

对 Custom Wavelet 默认应为：

```text
(sequence_length, 15)
```

每个显式 sequence 仍调用：

```python
encoder.reset()
```

但文档和 metadata 明确 sequence 必须代表 recording。

### 13.3 `src/writingring/spike_encoding/publication.py`

新增：

```text
spike_imu_path
```

构造：

```python
spike_imu = np.column_stack(
    (
        output.values,
        input_data.raw_imu[:, 3:9],
    )
)
```

验证：

```python
assert spike_imu.shape == (N, 21)
assert output.values.shape == (N, 15)
assert np.array_equal(
    spike_imu[:, 15:],
    input_data.raw_imu[:, 3:],
)
```

将 `spikeEvents.npy` 和 `spikeIMU.npy` 一起 staged、verified、atomically published。

### 13.4 `src/writingring/spike_encoding/io.py`

继续要求输入 rawIMU：

```text
nonempty (N,9)
```

继续读取：

```python
acceleration_g = raw_imu[:, :3]
```

保留完整 rawIMU，用于 publication 拼接剩余六个 IMU channels。

增加可选 metadata source paths：

```text
labels path
segment offsets path
segment lengths path
timestamp/manifest path
recording offsets path
```

只验证和引用，不修改这些文件。

### 13.5 `scripts/encode_spikes.py`

CLI 增加或明确：

```text
--recording-offsets
--boundary-padding-mode reflect
--event-index-semantics occurrence
```

Custom Wavelet 默认：

```text
occurrence alignment enabled
padding duration derived from extrema half-window
all 15 channels preserved
IIR delay compensation disabled
```

输出日志：

```text
Encoded N rows into 15 signed spike channels
Published N × 21 aligned spike IMU
Padding H samples per recording
Event indices aligned to extrema occurrence
Labels/timestamps/segment offsets unchanged
```

---

## 14. 测试计划

### 14.1 15-channel preservation

验证：

```python
spike_events.shape == (N, 15)
len(output_channel_names) == 15
```

验证 axis-major、frequency-minor 顺序不变。

### 14.2 Local maximum

构造已知 wavelet response 局部峰值：

```text
..., 1, 2, 5, 2, 1, ...
```

验证中心输出：

```text
+5
```

其他位置为零。

### 14.3 Local minimum

构造：

```text
..., -1, -2, -5, -2, -1, ...
```

验证中心输出：

```text
-5
```

确认负 polarity 被完整保留。

### 14.4 Plateau tie

构造 nonzero constant local window，验证 centre 最多输出一次，不发生双倍 amplitude。

### 14.5 Padding length

参数化：

```text
fs = 100 Hz
fs = 200 Hz
fs = 400 Hz
```

验证：

```python
H == encoder.max_filter_time_samples // 2
```

并验证：

```python
actual_padding_seconds == H / fs
```

### 14.6 Occurrence alignment

事件发生于原 sample `k`，detector 在 `k+H` 后确认。

验证最终：

```python
spike_events[k] != 0
```

而不是：

```python
spike_events[k + H]
```

### 14.7 Recording 首尾 flush

构造事件位于：

```text
sample 0
sample N-1
```

验证借助前后 padding，输出仍回填到：

```text
0
N-1
```

### 14.8 Label boundary 归属

设置 boundary：

```text
B = 1000
```

事件 occurrence：

```text
B - 10
```

验证：

```python
spike_events[B - 10] != 0
```

使用原 segment offsets 后，该 event 仍属于 boundary 前的 segment。

### 14.9 21-channel schema

验证：

```python
spike_imu.shape == (N, 21)
spike_imu[:, :15] == spike_events
spike_imu[:, 15:] == raw_imu[:, 3:]
```

### 14.10 Metadata reuse

验证以下文件内容未改变：

```text
labels.npy
segment_offsets.npy
segment_lengths.npy
timestamps/manifest
```

验证 spike summary 指向同一 source metadata。

### 14.11 Multiple recordings

提供 recording offsets：

```text
[0, N1, N1+N2]
```

验证：

* 两个 recording 分别 pad；
* 两个 recording 分别 reset；
* 输出总长度为 `N1+N2`；
* recording 1 的 right padding 不进入 recording 2；
* label segment boundary 不触发 reset。

### 14.12 Atomic publication

验证：

* `spikeEvents.npy` 和 `spikeIMU.npy` 同时成功；
* 失败时不留下半套文件；
* 未指定 overwrite 时不覆盖已有结果；
* source rawIMU 不发生变化。

---

## 15. Acceptance criteria

实现完成必须满足：

1. Custom Wavelet 保留全部 15 个 spike channels。
2. 不存在 15→3 frequency reduction。
3. Local maximum 和 local minimum 都参与事件生成。
4. Spike amplitude 保留正负符号。
5. Plateau extrema 不重复累计。
6. 默认 200 Hz 时，recording 前后各 pad 30 samples。
7. padding sample 数根据 sampling rate/extrema window 推导。
8. padding mode 为 reflect。
9. 每个 recording 独立 pad、flush 和 reset。
10. Label segment boundary 不触发 encoder reset。
11. Event index 回填到 wavelet-extrema occurrence index。
12. IIR delay 不补偿，并在 metadata 中明确。
13. `spikeEvents.npy` shape 为 `(N,15)`。
14. `spikeIMU.npy` shape 为 `(N,21)`。
15. `spikeIMU[:,0:15]` 为 15-channel signed spike sequence。
16. `spikeIMU[:,15:21]` 与 rawIMU `[:,3:9]` 完全一致。
17. 三个 `acceleration_*_g` 不出现在 spikeIMU 中。
18. 输出行数与 rawIMU 完全一致。
19. Timestamp 不平移。
20. Label 不平移。
21. Segment offsets 和 lengths 不改变。
22. Padding rows 不出现在最终输出中。
23. 原 rawIMU 不被覆盖。
24. Segmentation 源代码零修改。
25. 21-channel 输出不重新传入当前只接受 6/9 channels 的 segmentation API。
26. Spike output 通过共享原始 offsets、labels 和 timestamp 获得完全一致的 segmentation 语义。

最终数据流：

```text
rawIMU (N,9)
→ 读取 acceleration_g columns 0:3
→ 按 recording 前后 reflect pad H samples
→ causal IIR wavelet bank
→ local maximum + local minimum detection
→ 15 signed event channels
→ occurrence-index alignment
→ crop back to N rows
→ 删除 rawIMU acceleration_g columns
→ 拼接原 rawIMU columns 3:9
→ spikeIMU (N,21)
→ 复用原 timestamp、labels、segment offsets
```
