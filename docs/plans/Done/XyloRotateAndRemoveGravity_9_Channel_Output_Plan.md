# XyloRotateAndRemoveGravity 9-Channel Output Plan

## 1. 最终输出通道

所有 gravity-removal method 统一输出 9 个通道，固定顺序为：

```text
0  acceleration_x_g
1  acceleration_y_g
2  acceleration_z_g

3  acceleration_x
4  acceleration_y
5  acceleration_z

6  gyro_x
7  gyro_y
8  gyro_z
```

数组形状由：

```text
(N, 6)
```

修改为：

```text
(N, 9)
```

其中：

```text
acceleration_*_g
单位：g

acceleration_*
单位：m/s²

gyro_*
单位：保持当前 gyroscope 输出单位，预期为 rad/s
```

两组 acceleration 必须表示**同一份经过所选 method 处理后的信号**，只能有单位差异，不能一组保存原始 acceleration、另一组保存 gravity-removed acceleration。

必须始终满足：

[
a_{m/s^2}=a_g\times9.80665
]

以及：

[
a_g=\frac{a_{m/s^2}}{9.80665}
]

标准重力常量：

```python
STANDARD_GRAVITY_M_S2 = 9.80665
```

---

## 2. 各 Method 的通道语义

### Raw

Raw method 不移除重力。

处理顺序：

```text
原始 acceleration，单位 m/s²
→ 保留为 acceleration_x/y/z
→ 除以 9.80665
→ 保存为 acceleration_x/y/z_g
```

实现：

```python
acceleration_m_s2 = raw_acceleration_m_s2

acceleration_g = (
    acceleration_m_s2 / STANDARD_GRAVITY_M_S2
)
```

最终两组通道分别为：

```text
acceleration_*_g:
    原始 measured acceleration，单位 g

acceleration_*:
    原始 measured acceleration，单位 m/s²
```

---

### Low-pass

处理顺序：

```text
原始 acceleration，单位 m/s²
→ low-pass gravity estimation
→ gravity subtraction
→ linear acceleration，单位 m/s²
→ 同时保存 m/s² 和 g
```

实现：

```python
acceleration_m_s2 = (
    gravity_result.linear_acceleration_body
)

acceleration_g = (
    acceleration_m_s2 / STANDARD_GRAVITY_M_S2
)
```

最终两组 acceleration 都表示：

```text
low-pass gravity-removed acceleration
```

---

### Madgwick

处理顺序：

```text
原始 acceleration，单位 m/s²
→ Madgwick gravity estimation
→ gravity subtraction
→ linear acceleration，单位 m/s²
→ 同时保存 m/s² 和 g
```

实现：

```python
acceleration_m_s2 = (
    gravity_result.linear_acceleration_body
)

acceleration_g = (
    acceleration_m_s2 / STANDARD_GRAVITY_M_S2
)
```

最终两组 acceleration 都表示：

```text
Madgwick gravity-removed acceleration
```

现有 Madgwick calibration、gate、gravity magnitude 和 propagation 继续在 m/s² 工作域中运行。

---

### XyloRotateAndRemoveGravity

Xylo 必须先转换到 g，再进入 quantizer 和 RotationRemoval。

处理顺序：

```text
原始 acceleration，单位 m/s²
→ 除以 9.80665
→ 原始 acceleration，单位 g
→ XyloRotateAndRemoveGravity
→ gravity-removed acceleration，单位 g
→ 乘以 9.80665
→ gravity-removed acceleration，单位 m/s²
```

实现：

```python
raw_acceleration_g = (
    raw_acceleration_m_s2
    / STANDARD_GRAVITY_M_S2
)

xylo_result = xylo_rotate_and_remove_gravity(
    raw_acceleration_g,
    sampling_rate_hz=config.sampling_rate_hz,
    config=config.xylo,
)

acceleration_g = (
    xylo_result.linear_acceleration_g
)

acceleration_m_s2 = (
    acceleration_g * STANDARD_GRAVITY_M_S2
)
```

不能执行：

```text
Xylo 输出后再次除以 9.80665
```

Xylo 输出本身已经处于 g 域，m/s² 通道必须通过乘以 `9.80665` 得到。

---

## 3. 统一 Preprocessing Result

修改统一 dispatcher 的结果类型：

```python
@dataclass(frozen=True, slots=True)
class IMUPreprocessingResult:
    imu: np.ndarray

    acceleration_g: np.ndarray
    acceleration_m_s2: np.ndarray
    gyroscope_rad_s: np.ndarray

    method: str
    acceleration_semantics: str

    input_acceleration_unit: str
    acceleration_g_unit: str
    acceleration_m_s2_unit: str
    gyroscope_unit: str

    standard_gravity_m_s2: float

    gravity_result: GravityRemovalResult | None
    xylo_result: XyloGravityResult | None
```

字段要求：

```text
acceleration_g.shape == (N, 3)
acceleration_m_s2.shape == (N, 3)
gyroscope_rad_s.shape == (N, 3)
imu.shape == (N, 9)
```

最终数组构造：

```python
imu = np.column_stack(
    (
        acceleration_g,
        acceleration_m_s2,
        gyroscope_rad_s,
    )
)
```

---

## 4. Acceleration Semantics

由于通道名称统一使用：

```text
acceleration_x
acceleration_y
acceleration_z
```

必须在 summary 和 manifest 中明确该 acceleration 是否去除了重力。

建议字段：

```text
acceleration_semantics
```

不同 method 的值：

```text
raw:
    measured_acceleration_with_gravity

low-pass:
    gravity_removed_linear_acceleration

madgwick:
    gravity_removed_linear_acceleration

xylo-rotate-and-remove-gravity:
    xylo_gravity_removed_acceleration
```

这样可以避免仅根据通道名误判信号含义。

---

## 5. 输入列与输出列分离

不能继续用一个常量同时表示 Ring 原始输入和 preprocessing 输出。

改为：

```python
RING_SOURCE_IMU_COLUMNS = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
)
```

新增：

```python
PREPROCESSED_IMU_COLUMNS = (
    "acceleration_x_g",
    "acceleration_y_g",
    "acceleration_z_g",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
```

验证：

```python
len(RING_SOURCE_IMU_COLUMNS) == 6
len(PREPROCESSED_IMU_COLUMNS) == 9
```

Ring loader 仍读取原始 6 通道。

Segmentation 输出统一使用 9 通道。

---

## 6. 数值一致性检查

Dispatcher 完成后必须验证：

```python
if not np.isfinite(acceleration_g).all():
    raise IMUPreprocessingError(
        "g-domain acceleration contains non-finite values"
    )

if not np.isfinite(acceleration_m_s2).all():
    raise IMUPreprocessingError(
        "m/s^2 acceleration contains non-finite values"
    )
```

然后验证单位对应关系：

```python
expected_m_s2 = (
    acceleration_g * STANDARD_GRAVITY_M_S2
)

if not np.allclose(
    acceleration_m_s2,
    expected_m_s2,
    rtol=1e-6,
    atol=1e-7,
):
    raise IMUPreprocessingError(
        "g and m/s^2 acceleration channels are inconsistent"
    )
```

为了减少重复转换误差，每个 method 应选择一个 canonical representation。

### Raw、Low-pass、Madgwick

Canonical：

```text
m/s²
```

从 m/s² 派生 g。

### Xylo

Canonical：

```text
g
```

从 g 派生 m/s²。

---

1. Segmentation 修改（暂缓实施，后续再改）

以下两个 segmentation mode 当前保持现状不变，本节仅作为后续重构的设计说明：

label
aligned-board-events
Label-only

当前实现保持：

rawIMU.shape == (total_samples, 6)

计划中的变更（暂不实施）：

rawIMU.shape == (total_samples, 9)
Board-assisted

当前实现保持：

rawIMU.shape == (total_samples, 6)

计划中的变更（暂不实施）：

rawIMU.shape == (total_samples, 9)

Board event targets 当前保持不变：

board_event_targets.shape == (
    total_samples,
    4,
)

Segmentation boundary、offset、length、label 和 Board event target mapping 当前均不做任何修改，后续在统一 IMU schema 重构时再同步调整。

1. Fixed-Length Padding 修改（暂缓实施，后续再改）

当前 padding 实现 保持完全不变，仍然基于 6 通道假设运行。

当前行为

代码仍然假设：

raw_imu.shape[1] == 6
计划中的通用化改造（暂不实施）

未来将改为：

channel_count = raw_imu.shape[1]

并支持动态 channel 数。

未来目标输出格式（仅设计，不落地）
paddedIMU.shape == (segment_count, target_length, 9)
Padding 逻辑（当前不变）

当前仍然对 6 通道数据进行统一 padding：

padded_imu[
    segment_index,
    :valid_length,
    :,
] = source_segment
兼容性说明

未来设计中将支持：

6  legacy dataset
9  new dual-unit dataset

但当前版本：

不引入 channel_count 动态逻辑
不修改 padding 行为
不修改 summary schema
不强制任何 channel 数约束
未来 schema 记录（仅预告）

后续版本可能在 summary 中加入：

channel_count
channel_names
output_schema_version

但当前版本不新增任何强制要求，包括：

channel_count == 9

该约束将在 segmentation + preprocessing 完整重构阶段统一引入。

## 9. Summary 更新

统一增加：

```json
{
  "output_schema_version": 3,
  "channel_count": 9,
  "channel_names": [
    "acceleration_x_g",
    "acceleration_y_g",
    "acceleration_z_g",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "gyro_x",
    "gyro_y",
    "gyro_z"
  ],
  "units": {
    "acceleration_x_g": "g",
    "acceleration_y_g": "g",
    "acceleration_z_g": "g",
    "acceleration_x": "m/s^2",
    "acceleration_y": "m/s^2",
    "acceleration_z": "m/s^2",
    "gyro_x": "rad/s",
    "gyro_y": "rad/s",
    "gyro_z": "rad/s"
  },
  "standard_gravity_m_s2": 9.80665,
  "acceleration_semantics": "gravity_removed_linear_acceleration"
}
```

`output_schema_version` 应相对现有版本递增；禁止将 6 通道和 9 通道数据放在同一个训练 root 中。

---

## 10. Manifest 更新

每个 segment 增加：

```text
channel_count
channel_schema
acceleration_semantics
acceleration_g_unit
acceleration_m_s2_unit
gyroscope_unit
standard_gravity_m_s2
```

示例：

```text
channel_count = 9
channel_schema = dual_acceleration_units_v1
acceleration_g_unit = g
acceleration_m_s2_unit = m/s^2
gyroscope_unit = rad/s
standard_gravity_m_s2 = 9.80665
```

---

## 11. 文件命名

可以继续保留：

```text
*_rawIMU.npy
```

以避免大范围破坏加载路径，但文档必须明确：

```text
rawIMU 表示 segmentation 的完整 IMU feature tensor，
不代表 acceleration 未经过 preprocessing。
```

更清晰的长期命名可为：

```text
*_imu_features.npy
```

但不建议在本次修改中同时改文件名和 channel schema，以减少迁移风险。

---

## 12. 测试计划

### Channel Order

验证：

```python
assert result.imu.shape == (sample_count, 9)

np.testing.assert_array_equal(
    result.imu[:, 0:3],
    result.acceleration_g,
)

np.testing.assert_array_equal(
    result.imu[:, 3:6],
    result.acceleration_m_s2,
)

np.testing.assert_array_equal(
    result.imu[:, 6:9],
    result.gyroscope_rad_s,
)
```

### 单位关系

对所有四种 method：

```python
np.testing.assert_allclose(
    result.imu[:, 3:6],
    result.imu[:, 0:3] * 9.80665,
    rtol=1e-6,
    atol=1e-7,
)
```

### Raw

输入：

```python
acceleration = [[0.0, 0.0, 9.80665]]
```

输出：

```text
acceleration_g     = [0, 0, 1]
acceleration_m_s2  = [0, 0, 9.80665]
```

### Low-pass / Madgwick

Mock gravity-removed结果：

```python
linear_acceleration_body = [
    [9.80665, 0.0, -9.80665]
]
```

输出：

```text
g channels:
    [1.0, 0.0, -1.0]

m/s² channels:
    [9.80665, 0.0, -9.80665]
```

### Xylo

验证 Quantizer 输入：

```python
np.clip(
    raw_acceleration_m_s2
    / 9.80665
    / 2.0,
    -1.0 + epsilon,
    1.0 - epsilon,
)
```

假设 Xylo 输出：

```python
linear_acceleration_g = [
    [0.5, -0.25, 0.0]
]
```

则 m/s² 通道必须为：

```python
[
    [4.903325, -2.4516625, 0.0]
]
```

### No Double Conversion

分别检查：

```text
Raw/Low-pass/Madgwick:
    只从 m/s² 除一次得到 g

Xylo:
    输入时除一次进入 g
    输出 m/s² 时乘一次
    不再次除以 9.80665
```

### Segmentation

两个 boundary mode 均验证：

```text
rawIMU.shape[1] == 9
segment offsets 不变
segment lengths 不变
labels 不变
Board targets 不变
```

### Padding

验证：

```text
输入 (total_samples, 9)
输出 (N, T, 9)
mask 和 valid lengths 不受 channel count 影响
```

---

## 13. 修改文件

新增或继续使用：

```text
src/writingring/imu_preprocessing.py
src/writingring/xylo_gravity.py
```

修改：

```text
src/writingring/segmentation.py
src/writingring/board_event_segmentation.py
src/writingring/segment_padding.py
src/writingring/__init__.py

scripts/segment_ring_imu.py
scripts/pad_segmented_imu.py

tests/test_imu_preprocessing.py
tests/test_xylo_gravity.py
tests/test_segmentation.py
tests/test_board_event_segmentation.py
tests/test_segment_padding.py
tests/test_segment_ring_imu_cli.py

README.md
docs/notes/IMU_SEGMENTATION.md
docs/notes/BOARD_EVENT_GUIDED_SEGMENTATION.md
docs/notes/XYLO_GRAVITY_REMOVAL.md
docs/notes/SEGMENT_PADDING.md
```

---

## 14. 实施顺序

1. 定义标准重力常量 `9.80665`。
2. 定义固定 9 通道 schema。
3. 扩展 `IMUPreprocessingResult`。
4. 修改 Raw 输出，同时产生 g 和 m/s²。
5. 修改 Low-pass 输出，同时产生 g 和 m/s²。
6. 修改 Madgwick 输出，同时产生 g 和 m/s²。
7. 实现 Xylo 的 g-domain processing。
8. 将 Xylo 输出反向转换为 m/s² 辅助通道。
9. 验证两组 acceleration 的数值对应关系。
10. 将 label segmentation 改为 9 通道。
11. 将 Board-assisted segmentation 改为 9 通道。
12. 修改 padding 模块，移除 6 通道硬编码。
13. 更新 manifest、summary 和 schema version。
14. 更新所有测试 fixture 的 channel shape。
15. 更新 README 和相关文档。
16. 重新生成 segmentation 和 padded datasets。
17. 禁止旧 6 通道数据与新 9 通道数据混合训练。

---

## 15. 完成标准

实现完成后必须满足：

* 所有 method 输出固定 9 通道；
* 前三通道为处理后 acceleration，单位 g；
* 中间三通道为同一 acceleration，单位 m/s²；
* 后三通道为 gyroscope；
* `acceleration_m_s2 == acceleration_g * 9.80665`；
* Raw 两组 acceleration 都包含重力；
* Low-pass 两组 acceleration 都是 low-pass gravity-removed signal；
* Madgwick 两组 acceleration 都是 Madgwick gravity-removed signal；
* Xylo 先转换到 g，再进行 quantization 和 gravity removal；
* Xylo 的 m/s² 通道从最终 g 输出反向换算；
* 不进行 64 Hz 重采样；
* sample count 和 timestamps 不变；
* label 和 Board-assisted segmentation 均使用同一通道 schema；
* padding 支持 `(N,T,9)`；
* summary 和 manifest 清晰记录单位与 acceleration semantics；
* 旧 6 通道数据不会与新 9 通道数据混合。
