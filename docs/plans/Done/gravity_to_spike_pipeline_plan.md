## 结论

**当前实现“有条件正确”。**

只要你所说的 gravity removal 输出满足以下条件，就可以直接作为当前 `custom-wavelet` spike encoder 的输入：

1. 每个 `.npy` 文件仍是一条完整 recording，目录中的 `user/action/data_id` 只用于组织文件；
2. 数组严格为 `(N, 9)`；
3. 九列语义严格为：

```text
0: acceleration_x_g
1: acceleration_y_g
2: acceleration_z_g
3: acceleration_x        # m/s²
4: acceleration_y        # m/s²
5: acceleration_z        # m/s²
6: gyroscope_x           # rad/s
7: gyroscope_y           # rad/s
8: gyroscope_z           # rad/s
```

1. 前三列确实是去重力后的 body-frame acceleration，单位为 `g`；
2. sampling rate 与 spike encoder 配置一致。

仓库里的 `preprocess_ring_imu(...).imu` 正好产生上述统一 `(N,9)` 格式，包括 Xylo、low-pass 和 Madgwick 路径。

但如果所谓 gravity removal 输出是底层：

```python
xylo_rotate_and_remove_gravity(...).linear_acceleration_g
```

那么它只有 `(N,3)`，**不能直接交给 CLI**；必须通过 `preprocess_ring_imu()` 包装为完整 `(N,9)`。

此外，README 中的 `plot_ring_linear_acceleration.py` 只是诊断绘图工具，不会导出可供 spike encoder 使用的 `(N,9)` 文件。

---

## 当前调用链验证

### 1. Gravity preprocessing 输出

`preprocess_ring_imu()` 保持输入样本数量不变，并返回完整 recording：

```text
raw recording
    ↓
gravity removal / preprocessing
    ↓
(N,9) preprocessed IMU
```

Xylo 路径内部先把 acceleration 转为 `g`，完成旋转和去重力，再同时生成：

* columns `0:3`：去重力 acceleration，单位 `g`
* columns `3:6`：同一个 acceleration，单位 `m/s²`
* columns `6:9`：gyroscope，单位 `rad/s`

因此，**它与 spike loader 目前使用的通道契约相符**。

### 2. Spike loader 实际读取的内容

`load_spike_encoding_input()`：

* 强制输入为 `(N,9)`；
* 保存完整九通道数组；
* 将 `array[:, :3]` 提取为 `acceleration_g`；
* 只把前三列送入 encoder。

所以当前真实链路是：

```text
preprocessed_imu: (N,9)
        │
        ├── [:, 0:3] → custom-wavelet encoder
        │
        └── [:, 3:9] → 最终 spikeIMU 附加通道
```

### 3. Recording 边界

对 `custom-wavelet`，CLI 明确构造：

```python
boundaries = [0, N]
```

即整个文件被视为一个 recording，没有根据目录、label 或内部数据重新切分。

Runner 在每个 boundary 开始时调用一次 `encoder.reset()`；使用 `[0,N]` 时，一整个文件只 reset 一次。

因此：

```text
user/action/data_id/file.npy
```

中的目录层次不会影响 encoder state。只要一个文件对应一个完整 recording，这部分设计是正确的。

### 4. Encoder 输出

当前 Custom Wavelet：

* 输入 `(N,3)`；
* 三个 acceleration 轴；
* 每轴五个频率；
* 输出 `(N,15)`；
* 不改变时间轴长度。

发布阶段生成：

```text
spikes.npy    : (N,15)
spikeIMU.npy  : (N,21)
```

其中：

```text
spikeIMU[:, :15]  = spike events
spikeIMU[:, 15:]  = source_imu[:, 3:9]
```

也就是附加的六列是：

```text
acceleration_x/y/z in m/s²
gyroscope_x/y/z in rad/s
```

不是原始的 `acc + gyro + magnetometer`。

### 5. Sampling rate

默认 Custom Wavelet 配置使用 `200 Hz`。

如果 gravity removal 输出不是 200 Hz：

* 当前代码不会自动 resample；
* 必须修改配置中的 `sampling_rate_hz`；
* 若同时提供 summary，summary 与配置中的采样率必须一致。

---

## 当前仍存在的问题

### P0：输入验证只验证形状，不验证通道物理语义

当前 loader 能检查：

* `(N,9)`
* numeric
* finite

但不会检查：

```python
imu[:, 3:6] ≈ imu[:, 0:3] * 9.80665
```

因此一个旧格式的 `(acc, gyro, mag)` 九通道数组，也可能被静默解释为：

```text
acceleration_g + acceleration_m_s2 + gyroscope
```

这是目前最大的 correctness 风险。

现有 CLI 测试甚至使用任意 `np.arange(...).reshape(N,9)` 构造输入，并不满足两套 acceleration 单位之间的关系，所以测试目前只证明了 shape plumbing，没有证明真实 gravity-output contract。

### P0：底层 Xylo 输出和可发布输入容易混淆

仓库内有两个不同层级的“gravity output”：

```text
xylo_rotate_and_remove_gravity()
    → (N,3), acceleration only

preprocess_ring_imu()
    → (N,9), publishable common IMU schema
```

这一区别在当前 Xylo 文档中有所体现，但没有明确写成 spike encoder 的硬性 handoff contract。

### P1：缺少正式的 gravity-preprocessing 导出入口

当前仓库有：

* gravity plotting CLI；
* segmentation CLI；
* spike encoding CLI；

但没有一个正式的：

```text
raw recording → gravity-removed (N,9) .npy + metadata JSON
```

导出脚本。现有公共逻辑在 `preprocess_ring_imu()` 中，但文件发布流程并未形成完整 CLI。仓库文件树也没有独立的 preprocessing export script。

这意味着：

* 从 Python API 手工保存 `result.imu` 可以工作；
* 仓库当前却没有完整、可重复的命令行工作流。

### P1：文档仍以 segmentation 为主要输入来源

README 和 `SPIKE_ENCODING.md` 虽然已经写明使用前三个 g-domain acceleration 通道，但仍把 spike 输入描述为：

```text
complete segmentation export
*_rawIMU.npy
segmentation summary
```

而你现在的目标是：

```text
complete gravity-preprocessed recording
```

这两者在数组层面可以相同，但 provenance、文件命名和用户心智模型不同。

### P1：Summary 没有与实际数组充分交叉验证

当前 summary validator 会检查：

* channel count；
* channel names；
* standard gravity；
* sampling rate 等字段。

但没有可靠验证：

* `sample_count == array.shape[0]`；
* summary 指向的源文件就是当前 `.npy`；
* user/action/data_id 与目录是否一致；
* 文件内容 hash；
* gravity removal method 是否真的不是 `raw`。

### P1：输出目录不支持真正的集中式 batch pipeline

当前单文件 CLI 要求 `--output-root` 等于输入文件所在目录。

因此对：

```text
gravity_output/
  user_01/
    action_a/
      data_001.npy
```

逐文件调用可以工作，但无法自然执行：

```text
--input-root gravity_output/
--output-root spike_output/
```

并自动保留相对目录层次。

### P2：命名仍带有旧数据结构语义

代码中的以下名称容易误导：

```python
raw_imu
raw_imu_path
segmentation summary
_rawIMU.npy
```

实际上当前 encoder 预期的是经过统一 preprocessing 的 IMU，不是原始 IMU；最终保留的六列也不是传统 raw IMU 的 gyro/magnetometer 结构。

---

# 修改 Plan

## Phase 1：固定并强化输入契约

### 1. 新增统一 preprocessing artifact validator

建议新增：

```text
src/writingring/preprocessing_io.py
```

提供：

```python
load_preprocessed_imu(...)
validate_preprocessed_imu(...)
load_preprocessing_summary(...)
```

验证内容：

```python
array.ndim == 2
array.shape[1] == 9
array.shape[0] > 0
np.isfinite(array).all()

np.allclose(
    array[:, 3:6],
    array[:, 0:3] * STANDARD_GRAVITY_M_S2,
    rtol=...,
    atol=...,
)
```

并要求 metadata 明确声明：

```json
{
  "schema_version": 1,
  "channel_names": [...],
  "sampling_rate_hz": 200.0,
  "acceleration_semantics": "gravity_removed_body_frame",
  "gravity_removal_method": "xylo",
  "standard_gravity_m_s2": 9.80665
}
```

对于当前目标，默认拒绝：

```json
"gravity_removal_method": "raw"
```

除非用户明确传入类似：

```text
--allow-gravity-included
```

### 2. Spike loader 改为复用 preprocessing validator

修改：

```text
src/writingring/spike_encoding/io.py
```

将当前独立的 shape-only 检查替换为统一契约检查。

同时重命名内部字段：

```python
raw_imu_path → source_imu_path
raw_imu      → preprocessed_imu
```

若需要兼容已有调用，可暂时保留 deprecated property alias。

### 3. 将 summary 改成来源无关的 preprocessing summary

重命名或泛化：

```python
load_spike_encoding_source_summary()
```

不再把 docstring 和错误信息绑定到 segmentation。

增加交叉验证：

```text
summary.sample_count == array.shape[0]
summary.channel_names == PREPROCESSED_IMU_COLUMNS
summary.sampling_rate_hz == encoder sampling rate
summary.source_file matches current file
summary.gravity_removed == true
```

最好加入文件 SHA-256，避免 summary 与 `.npy` 错配。

---

## Phase 2：建立正式 gravity → spike 工作流

### 4. 新增 gravity/preprocessing 导出 CLI

建议新增：

```text
scripts/preprocess_ring_imu.py
```

职责：

```text
raw recording
    ↓ preprocess_ring_imu()
complete (N,9) recording
    ↓
.npy + .json
```

建议输出结构：

```text
outputs/preprocessedIMU/
  <user>/
    <action>/
      <data_id>/
        <data_id>_preprocessedIMU.npy
        <data_id>_preprocessing.json
```

要求：

* 一个输入 recording 对应一个输出 recording；
* 不做 segmentation；
* 保留所有 N 个样本；
* summary 写入 method、采样率、单位、通道名和 recording identity；
* Xylo 路径必须保存 `preprocess_ring_imu(...).imu`，不能只保存底层 `(N,3)` 结果。

### 5. 为 spike CLI 增加 batch 输入模式

修改：

```text
scripts/encode_spikes.py
```

保留现有：

```text
--input-imu file.npy
```

新增：

```text
--input-root outputs/preprocessedIMU
--pattern "*_preprocessedIMU.npy"
--output-root outputs/spikeEncoding
```

行为：

```text
每个文件独立 load
每个文件独立 reset
每个文件使用 boundaries=[0,N]
保留 input-root 下的相对 user/action/data_id 目录
```

示例输出：

```text
outputs/spikeEncoding/
  custom-wavelet/
    <user>/
      <action>/
        <data_id>/
          spikes.npy
          spikeIMU.npy
          metadata.json
```

### 6. 放宽 publication 的 output-root 限制

修改：

```text
src/writingring/spike_encoding/publication.py
scripts/encode_spikes.py
```

不再要求：

```text
output_root == input_file.parent
```

改为显式传入：

```python
recording_relative_path
recording_id
```

并让输出路径与输入文件名后缀无关，而不是重点依赖 `_rawIMU.npy`。

### 7. 更新 publication metadata

将 metadata schema 升级，至少增加：

```json
{
  "source_imu_path": "...",
  "source_summary_path": "...",
  "recording": {
    "user": "...",
    "action": "...",
    "data_id": "..."
  },
  "input_channel_names": [...],
  "encoder_input_channel_names": [
    "acceleration_x_g",
    "acceleration_y_g",
    "acceleration_z_g"
  ],
  "gravity_removal_method": "xylo",
  "sampling_rate_hz": 200.0
}
```

明确记录 `spikeIMU.npy` 的 trailing channels：

```text
acceleration_x/y/z_m_s2
gyroscope_x/y/z_rad_s
```

---

## Phase 3：补齐测试

### 8. 增加真实 producer-consumer integration test

新增例如：

```text
tests/test_gravity_to_spike_pipeline.py
```

测试流程：

```python
raw recording
→ preprocess_ring_imu(...)
→ np.save(result.imu)
→ encode_spikes CLI
→ load spikes.npy / spikeIMU.npy / metadata.json
```

断言：

```python
output.shape[0] == input.shape[0]
spikes.shape == (N, 15)
spike_imu.shape == (N, 21)

spike_imu[:, 15:] == preprocessed_imu[:, 3:9]
encoder received preprocessed_imu[:, 0:3]
```

并验证一条 recording 只 reset 一次。

### 9. 为各 gravity method 参数化测试

覆盖：

```text
low-pass
madgwick
xylo
```

仓库现有 preprocessing 单元测试已经分别检查这些方法可产生统一 `(N,9)`，可以直接扩展到 spike handoff。

### 10. 增加失败用例

必须覆盖：

* `(N,3)` 底层 Xylo 结果：给出可操作错误信息；
* legacy `(acc, gyro, mag)` 九通道：因单位一致性检查失败而拒绝；
* `m/s² != g × 9.80665`；
* summary sample count 不匹配；
* summary channel names 不匹配；
* sampling rate 不匹配；
* gravity method 为 `raw`；
* 两个 recording 连续处理时 encoder state 不泄漏；
* user/action/data_id 相对目录正确保留。

### 11. 修正现有 synthetic fixtures

把当前任意 `np.arange(N*9)` 测试数据改为物理一致的 fixture：

```python
acc_g = ...
acc_m_s2 = acc_g * 9.80665
gyro = ...
imu = np.column_stack([acc_g, acc_m_s2, gyro])
```

这样测试才能真正保护 gravity-to-spike contract。现有 I/O 测试已经覆盖 shape 和前三列提取，但尚未覆盖该物理关系。

---

# 文档修改 Plan

## README.md

将主流程改为：

```text
Raw recording
→ Gravity removal / IMU preprocessing
→ Complete preprocessed recording (N,9)
→ Spike encoding
```

具体修改：

1. 不再把 spike encoder 的主要输入称为 “segmentation export”；
2. 增加从 gravity preprocessing 到 spike encoding 的完整命令示例；
3. 列出精确九通道顺序与单位；
4. 明确 encoder 只消费前三列；
5. 明确一个文件就是一个完整 recording；
6. 明确 `user/action/data_id` 只是目录组织方式；
7. 明确底层 Xylo `(N,3)` 不能直接传给 CLI；
8. 明确默认 Custom Wavelet 为 200 Hz；
9. 给出 batch 输入与输出目录示例；
10. 将 segmentation 说明移动到独立的可选/后续工作流。

## docs/notes/XYLO_GRAVITY_REMOVAL.md

新增一个明确的 **Spike Encoding Handoff** 章节：

```text
Valid handoff:
preprocess_ring_imu(...).imu → (N,9)

Invalid direct handoff:
xylo_rotate_and_remove_gravity(...).linear_acceleration_g → (N,3)
```

同时说明：

* N 不变；
* recording 不分段；
* g 与 m/s² 两组三轴是同一物理信号；
* gyroscope 被保留；
* 推荐的文件名与 summary schema；
* user/action/data_id 目录不会改变 recording 边界。

## docs/notes/SPIKE_ENCODING.md

核心术语调整：

```text
one complete segmentation export
```

改为：

```text
one complete preprocessed IMU recording
```

并增加：

* source-agnostic input contract；
* gravity-removed requirement；
* g/m/s² 一致性约束；
* summary 的必需字段；
* 全 recording `boundaries=[0,N]`；
* 每个文件只 reset 一次；
* `spikeIMU` trailing six channel semantics；
* gravity-output batch 示例；
* sampling-rate mismatch 行为；
* 与低层 `(N,3)` Xylo 输出的区别。

Segmentation sidecar、occurrence、label 等内容从核心输入说明中移出，保留链接即可。

## docs/notes/OCCURRENCE_ALIGNED_SPIKE_ENCODING.md

本阶段不修改其算法或数据流程。最多在开头补一句：

```text
This document describes an optional downstream segmentation/alignment
workflow and is not part of the complete-recording gravity-to-spike path.
```

这样可以避免它继续干扰当前主目标，同时不提前改 segmentation 设计。

---

## 建议实施顺序

1. **先固定 `(N,9)` artifact 与 metadata contract**；
2. 加强 loader 的物理语义和 provenance 验证；
3. 新增 gravity preprocessing export CLI；
4. 新增 spike batch mode与相对目录保留；
5. 补 gravity → spike 端到端测试；
6. 最后同步 README、`XYLO_GRAVITY_REMOVAL.md` 和 `SPIKE_ENCODING.md`。

最终验收标准应是：

```text
无需 segmentation 文件或 segmentation summary；
任意 user/action/data_id 下的完整 gravity-removed recording；
可以独立、确定性地生成 N 行 spike 输出；
不同 recording 之间没有 encoder state 泄漏；
metadata 能证明输入通道、单位、采样率和 gravity method。
```

本次结论来自对当前代码和测试的静态调用链核查。我未能在本地执行完整测试套件，因此“接口逻辑成立”已经验证，但“当前 main 在真实数据上端到端通过”仍需要新增上述 integration test 后才能正式确认。
