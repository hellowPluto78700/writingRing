# WritingRing 通用 Spike Encoding 与 Custom Wavelet 实施计划

## 1. 项目目标

为 WritingRing 增加一个独立、可扩展的 spike-encoding 后处理流程，对现有 gravity-removal 输出中的前三个通道进行编码：

```text
0  acceleration_x_g
1  acceleration_y_g
2  acceleration_z_g
```

适用的上游预处理方法包括：

```text
raw
low-pass
madgwick
xylo-rotate-and-remove-gravity
```

其中：

* `raw` 的前三个通道是包含重力的三轴加速度，单位为 g；
* 其他方法的前三个通道是对应方法处理后的三轴加速度，单位为 g；
* spike-encoding 层只读取已有结果，不重新进行 gravity removal；
* 第一版实现 `custom-wavelet`；
* 通用框架必须为 threshold、step-forward、BSA、Poisson 等其他方法保留接入接口。

Neuromorphic-Gravity 的 Custom Wavelet 参考流程分布在流程调度、IIR/峰谷检测、wavelet 定义和 Prony 拟合四个文件中。

---

## 2. 范围界定

### 2.1 本计划包含

* 通用 spike-encoding CLI；
* encoder 公共接口；
* encoder registry；
* 输入文件发现与验证；
* sequence 状态边界处理；
* 独立 Custom Wavelet encoder；
* 可配置的 `frequencies_hz`；
* 根据频率动态计算 wavelet widths；
* wavelet、Prony IIR、局部峰谷检测；
* 独立 event 输出；
* summary、统计和测试。

### 2.2 本计划不包含

* 修改 `preprocess_ring_imu()`；
* 修改任何 gravity-removal method；
* 修改九通道 `rawIMU.npy`；
* 修改 label segmentation；
* 修改 Board-event segmentation；
* 修改 segment boundaries；
* 修改 padding；
* 将 spike channels 拼接进九通道 IMU；
* 修改 `vendor/Neuromorphic-Gravity/**`；
* 修改 `data_sample/**`。

现有 segmentation 或 padding 输出可以作为只读输入，但本功能不改变其实现、格式或结果。

---

## 3. 总体数据流

```text
已有 gravity-removal 输出
        │
        ├── *_rawIMU.npy              必需
        ├── *_segmentation_summary.json  可选
        └── *_segment_offsets.npy        可选，只读
        │
        ▼
scripts/encode_spikes.py
        │
        ├── 加载输入
        ├── 提取 rawIMU[:, 0:3]
        ├── 从 registry 创建 encoder
        ├── 按 sequence 调用 encoder
        ├── 聚合编码结果
        └── 原子化发布输出
        │
        ▼
Encoder registry
        │
        ├── custom-wavelet
        ├── future-threshold
        ├── future-bsa
        └── future-...
                │
                ▼
独立 spike-encoding 输出
```

Custom Wavelet 的数学和状态逻辑不能写进通用 CLI，必须封装在独立 encoder 文件中。

---

## 4. 建议文件结构

```text
scripts/
└── encode_spikes.py

configs/
└── spike_encoding/
    └── custom_wavelet.json

src/writingring/spike_encoding/
├── __init__.py
├── contracts.py
├── registry.py
├── io.py
├── runner.py
└── encoders/
    ├── __init__.py
    └── custom_wavelet.py

tests/
├── test_spike_encoding_contracts.py
├── test_spike_encoding_registry.py
├── test_spike_encoding_io.py
├── test_spike_encoding_runner.py
├── test_custom_wavelet_settings.py
├── test_custom_wavelet_encoder.py
├── test_custom_wavelet_vendor_parity.py
└── test_encode_spikes_cli.py

docs/
└── SPIKE_ENCODING.md
```

职责划分：

### `scripts/encode_spikes.py`

只处理：

* CLI 参数；
* 路径解析；
* settings 加载；
* encoder 创建；
* 错误打印；
* 调用通用 runner；
* 输出路径展示。

### `registry.py`

只处理：

* encoder 注册；
* encoder 名称查询；
* encoder 实例创建。

### `runner.py`

只处理：

* 提取输入通道；
* sequence 拆分；
* encoder reset；
* 逐 sequence 编码；
* 结果聚合；
* 公共统计。

### `encoders/custom_wavelet.py`

只处理：

* Custom Wavelet settings；
* wavelet 生成；
* Prony 拟合；
* IIR filter bank；
* 局部峰谷检测；
* Custom Wavelet channel naming；
* Custom Wavelet diagnostics。

---

## 5. Encoder 公共接口

在 `contracts.py` 中定义统一协议。

```python
from typing import Protocol

import numpy as np


class SpikeEncoder(Protocol):
    name: str
    representation: str

    @property
    def output_channel_names(self) -> tuple[str, ...]:
        ...

    def reset(self) -> None:
        ...

    def encode_sequence(
        self,
        acceleration_g: np.ndarray,
    ) -> "SpikeEncodingSequenceResult":
        ...
```

公共 sequence 结果：

```python
@dataclass(frozen=True, slots=True)
class SpikeEncodingSequenceResult:
    values: np.ndarray
    channel_names: tuple[str, ...]
    representation: str
    diagnostics: dict[str, object]
```

公共整体结果：

```python
@dataclass(frozen=True, slots=True)
class SpikeEncodingOutput:
    values: np.ndarray
    encoder_name: str
    representation: str
    channel_names: tuple[str, ...]
    sequence_offsets: np.ndarray
    sequence_statistics: tuple[dict[str, object], ...]
    summary: dict[str, object]
```

公共接口只假定：

* 输入为 `(N, 3)`；
* 输出为 `(N, C)`；
* `C` 由具体 encoder 决定；
* encoder 可以是 stateful；
* encoder 必须提供 `reset()`；
* 输出 sample count 默认必须等于输入 sample count。

公共框架不能假定：

* 输出总是 15 通道；
* encoder 总是使用 wavelet；
* encoder 总是使用 frequencies；
* encoder 总是输出二值 spikes；
* encoder 总是使用 Prony 或 IIR。
* encoder 总是具有 wavelet/frequency 维度或 axis-major flatten 顺序。

通道布局属于 encoder-specific metadata。encoder 可以可选声明：

```python
output_metadata = {
    "channel_order": "...",
}
```

通用发布层只在 encoder 明确声明 `channel_order` 时写入 summary；未声明的
threshold、BSA 或其他 encoder 不得被标注为 Custom Wavelet 的顺序。

---

## 6. Encoder Registry

第一版 registry：

```python
ENCODER_REGISTRY = {
    "custom-wavelet": CustomWaveletEncoder,
}
```

公共函数：

```python
def available_encoders() -> tuple[str, ...]:
    ...


def create_encoder(
    name: str,
    *,
    settings: dict[str, object],
) -> SpikeEncoder:
    ...
```

通用脚本统一调用：

```python
encoder = create_encoder(
    args.encoder,
    settings=encoder_settings,
)
```

禁止使用：

```python
if encoder_name == "custom-wavelet":
    ...
elif encoder_name == "threshold":
    ...
elif encoder_name == "bsa":
    ...
```

新增其他方法时，只需：

1. 新增独立 encoder 文件；
2. 实现公共接口；
3. 在 registry 注册；
4. 添加该 encoder 的 settings 和测试。

---

## 7. 输入契约

### 7.1 必需输入

```text
*_rawIMU.npy
```

输入必须满足：

```text
shape == (N, 9)
N > 0
numeric dtype
所有值 finite
allow_pickle=False
```

通道顺序必须是：

```text
acceleration_x_g
acceleration_y_g
acceleration_z_g
acceleration_x
acceleration_y
acceleration_z
gyro_x
gyro_y
gyro_z
```

通用 runner 只提取：

```python
acceleration_g = raw_imu[:, 0:3]
```

不得：

* 再次除以或乘以标准重力；
* 再次去除重力；
* 重新估计姿态；
* 重新滤波 gravity；
* 修改输入数组；
* 修改输入文件。

### 7.2 可选 summary

可以读取：

```text
*_segmentation_summary.json
```

仅用于获取和验证：

```text
sampling_rate_hz
gravity_removal_method
acceleration_semantics
channel_names
standard_gravity_m_s2
```

不修改该 summary。

如果 summary 提供 gravity-removal provenance，则必须同时提供并验证：

```text
gravity removal method
acceleration semantics
```

method 只能是 `raw`、`low-pass`、`madgwick` 或
`xylo-rotate-and-remove-gravity`，且 semantics 必须与对应 method 一致。
只提供其中一个、未知 method 或不一致的 semantics 都必须失败；两个字段都
省略时仍允许 summary 只提供其他可选 metadata。

### 7.3 可选 sequence offsets

可以读取：

```text
*_segment_offsets.npy
```

只用于定义 encoder 的状态重置边界。

这不属于 segmentation 修改，只是消费已有边界信息。

---

## 8. Sequence 边界和状态

Custom Wavelet 的 IIR filter 和局部峰谷窗口均保存历史状态。参考实现也是创建一次 pipeline，并在同一序列上逐采样连续调用。

通用脚本提供：

```text
--sequence-mode offsets
--sequence-mode single-array
```

### 8.1 `offsets`

要求提供：

```text
--sequence-offsets <path>
```

对每个区间：

```python
start = offsets[i]
stop = offsets[i + 1]
```

执行：

```python
encoder.reset()
result = encoder.encode_sequence(
    acceleration_g[start:stop]
)
```

必须验证：

```text
offsets.ndim == 1
offsets[0] == 0
offsets[-1] == N
offsets 严格递增
每个 sequence 非空
```

### 8.2 `single-array`

把整个 `(N, 3)` 输入作为一条连续 sequence：

```python
encoder.reset()
result = encoder.encode_sequence(acceleration_g)
```

该模式适用于输入确实是一条连续 recording 的情况。

### 8.3 默认规则

建议默认：

```text
如果明确提供 offsets：使用 offsets
否则：使用 single-array
```

脚本不得自行猜测不同 row 之间是否连续。

输出 summary 必须记录：

```text
sequence_mode
sequence_count
state_reset_boundary
sequence_offsets_source
```

---

## 9. 通用 CLI

示例：

```bash
python scripts/encode_spikes.py \
  --input-imu outputs/user_0_action_0_rawIMU.npy \
  --encoder custom-wavelet \
  --encoder-settings configs/spike_encoding/custom_wavelet.json \
  --sequence-mode offsets \
  --sequence-offsets outputs/user_0_action_0_segment_offsets.npy \
  --output-root outputs/spike_encoding \
  --overwrite
```

公共参数：

```text
--input-imu
--input-summary
--encoder
--encoder-settings
--sequence-mode
--sequence-offsets
--output-root
--output-stem
--output-dtype
--overwrite
```

`--encoder-settings` 是 encoder 私有配置文件。通用 CLI 不为每种 encoder 增加大量专属参数。

例如未来不应添加：

```text
--custom-wavelet-frequency
--bsa-filter-order
--poisson-seed
--threshold-value
```

这些内容全部进入各自 settings 文件。

---

## 10. Custom Wavelet Settings

默认配置文件：

```json
{
  "wavelet_name": "acceleration",
  "frequencies_hz": [
    0.5,
    1.0,
    2.0,
    4.0,
    8.0
  ],
  "sampling_rate_hz": 200.0,
  "prony_denominator_order": 2,
  "prony_numerator_order": 2,
  "max_filter_time_s": 0.3,
  "max_filter_frequency_decades": 0.5,
  "output_dtype": "float32"
}
```

对应 dataclass：

```python
@dataclass(frozen=True, slots=True)
class CustomWaveletSettings:
    wavelet_name: str = "acceleration"
    frequencies_hz: tuple[float, ...] = (
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
    )
    sampling_rate_hz: float = 200.0
    prony_denominator_order: int = 2
    prony_numerator_order: int = 2
    max_filter_time_s: float = 0.3
    max_filter_frequency_decades: float = 0.5
    output_dtype: str = "float32"
```

参考路径默认采用：

```text
frequencies = [0.5, 1.0, 2.0, 4.0, 8.0]
```

三个输入轴和五个尺度对应 15 个 event 通道。

---

## 11. 可配置的 Frequencies 与 Wavelet Widths

### 11.1 核心要求

`frequencies_hz` 必须是可修改的 settings，而不是代码常量。

默认值：

```text
[0.5, 1.0, 2.0, 4.0, 8.0]
```

用户可以修改为任意合法频率列表，例如：

```json
{
  "frequencies_hz": [
    0.25,
    0.5,
    1.0,
    2.0,
    4.0
  ]
}
```

或者：

```json
{
  "frequencies_hz": [
    1.0,
    2.0,
    5.0,
    10.0
  ]
}
```

### 11.2 Wavelet widths 的计算

`wavelet_widths_samples` 不作为另一组独立 settings 输入。

它始终根据采样率和频率动态计算：

```python
wavelet_widths_samples = tuple(
    int(sampling_rate_hz / frequency_hz)
    for frequency_hz in frequencies_hz
)
```

参考实现同样使用：

```python
widths = sampleFreq / frequencies
```

生成每个 wavelet 尺度。

默认：

```text
sampling_rate_hz = 200

frequencies_hz:
[0.5, 1.0, 2.0, 4.0, 8.0]

wavelet_widths_samples:
[400, 200, 100, 50, 25]
```

自定义：

```text
sampling_rate_hz = 200

frequencies_hz:
[1.0, 2.0, 5.0, 10.0]

wavelet_widths_samples:
[200, 100, 40, 20]
```

因此：

> 用户通过修改 `frequencies_hz` 修改 wavelet widths。

不提供独立的 `wavelet_widths_samples` 配置，是为了避免 frequencies 和 widths 互相矛盾。

### 11.3 Frequency 验证

`frequencies_hz` 必须：

1. 至少包含一个值；
2. 所有值为有限数；
3. 所有值大于 0；
4. 所有值严格小于 Nyquist frequency；
5. 不包含重复值；
6. 严格按 Hz 从小到大排列。

验证：

```python
if any(
    frequencies_hz[i] >= frequencies_hz[i + 1]
    for i in range(len(frequencies_hz) - 1)
):
    raise CustomWaveletSettingsError(
        "frequencies_hz must be strictly increasing in Hz"
    )
```

程序不能静默排序。

以下配置必须失败：

```json
{
  "frequencies_hz": [4.0, 1.0, 8.0]
}
```

以下配置也必须失败：

```json
{
  "frequencies_hz": [1.0, 2.0, 2.0, 4.0]
}
```

### 11.4 Width 验证

动态生成后必须验证：

```text
每个 width 是正整数
每个 width 足以支持 Prony orders
width 数量等于 frequency 数量
没有 width 因整数截断而重复
```

最后一项很重要。例如两个很接近的高频值可能经过：

```python
int(sampling_rate_hz / frequency_hz)
```

得到同一个 width。

建议遇到重复 width 时明确失败：

```text
configured frequencies collapse to duplicate wavelet widths
```

避免两个输出通道实际使用完全相同的 wavelet kernel。

---

## 12. 可扩展 Wavelet Shape

Custom Wavelet encoder 内部维护 wavelet registry：

```python
WAVELET_REGISTRY = {
    "acceleration": acceleration_wavelet,
}
```

第一版正式支持：

```text
wavelet_name = acceleration
```

未来可增加：

```text
velocity
power
其他自定义 wavelet
```

新增 wavelet shape 时：

* 不修改通用 `encode_spikes.py`；
* 不修改 encoder registry；
* 只在 Custom Wavelet 模块内部注册新 wavelet；
* 使用同一套 `frequencies_hz` 和 width 生成规则。

每个 wavelet factory 的接口：

```python
def wavelet_factory(
    length: int,
    scale: float,
) -> np.ndarray:
    ...
```

---

## 13. Acceleration Wavelet

默认 wavelet 使用参考实现的 acceleration wavelet：

```python
x = (
    np.arange(length)
    - (length - 1) / 2
) / scale

wavelet = (
    (x > -0.5)
    * (x < 0.5)
    * 29 / 4
    * x
    * (4 * x**2 - 1)
)

output = np.sqrt(1 / scale) * wavelet
```

参考 `accelerationWavelet()` 只负责定义 wavelet 核，不负责滤波、状态、峰谷检测或保存。

建议使用 NumPy 重新实现，不在 WritingRing runtime 中直接 import vendor 的 PyTorch pipeline。

---

## 14. Prony IIR 拟合

每个 frequency 对应一个 wavelet kernel。

默认使用：

```text
denominator order = 2
numerator order   = 2
```

处理流程：

```text
frequency
    ↓
wavelet width
    ↓
离散 wavelet kernel
    ↓
Prony 拟合
    ↓
二阶 IIR coefficients
```

系数形式：

```text
b = [b0, b1, b2]
a = [1, a1, a2]
```

参考流程使用 `prony(wavelet, 2, 2)` 将长 wavelet kernel 转为短状态 IIR filter。

拟合后验证：

```text
coefficients 全部 finite
a[0] == 1
coefficient shape 正确
IIR poles 满足定义的稳定性策略
frequency、width 和 coefficients 一一对应
```

不允许：

* Prony 失败后静默跳过某个 frequency；
* 将非 finite 系数替换成 0；
* 自动改变 filter orders；
* 静默改变用户 frequencies。

---

## 15. Stateful IIR Filter Bank

对于：

```text
K = len(frequencies_hz)
```

单步输入：

```text
shape = (3,)
```

单步 wavelet response：

```text
shape = (3, K)
```

内部状态：

```text
previous inputs:
    (3, 2)

previous filter outputs:
    (3, K, 2)
```

递推：

```text
y[t]
=
b0*x[t]
+ b1*x[t-1]
+ b2*x[t-2]
- a1*y[t-1]
- a2*y[t-2]
```

参考实现为每个轴、每个尺度维护两个输入和两个输出的历史状态。

Custom Wavelet encoder 提供：

```python
class CustomWaveletEncoder:
    name = "custom-wavelet"
    representation = "signed_sparse_wavelet_extrema"

    def reset(self) -> None:
        ...

    def step(
        self,
        acceleration_g: np.ndarray,
    ) -> np.ndarray:
        ...

    def encode_sequence(
        self,
        acceleration_g: np.ndarray,
    ) -> SpikeEncodingSequenceResult:
        ...
```

---

## 16. 局部峰谷 Event Detection

默认：

```text
max_filter_time_s = 0.3
```

窗口大小：

```python
window_samples = int(
    max_filter_time_s * sampling_rate_hz
)

if window_samples % 2 == 0:
    window_samples += 1
```

例如：

```text
64 Hz  → 19 samples
200 Hz → 61 samples
```

因为使用 acceleration wavelet 而不是 power signal，需要同时检测：

* 正局部峰值；
* 负局部谷值。

输出规则：

```text
非局部峰谷 → 0
局部正峰   → 保留正响应幅值
局部负谷   → 保留负响应幅值
```

参考实现也是保留峰谷响应幅值，而不是输出简单布尔值。

因此输出 representation 应为：

```text
signed_sparse_wavelet_extrema
```

不能错误记录为：

```text
binary_spike_train
```

参考 pipeline 虽然实例化了 `snn.Leaky`，但它没有进入实际执行的 `self.model`。

---

## 17. 输出维度与通道顺序

对于：

```text
K = len(frequencies_hz)
```

内部输出：

```text
(N, 3, K)
```

flatten 后：

```text
(N, 3 × K)
```

默认五个 frequencies：

```text
(N, 15)
```

四个 frequencies：

```text
(N, 12)
```

公共框架不能硬编码 15。

通道顺序固定为：

```text
axis-major, frequency-minor
```

即：

```text
x 轴：frequencies_hz 从小到大
y 轴：frequencies_hz 从小到大
z 轴：frequencies_hz 从小到大
```

默认：

```text
event_x_0p5_hz
event_x_1_hz
event_x_2_hz
event_x_4_hz
event_x_8_hz

event_y_0p5_hz
event_y_1_hz
event_y_2_hz
event_y_4_hz
event_y_8_hz

event_z_0p5_hz
event_z_1_hz
event_z_2_hz
event_z_4_hz
event_z_8_hz
```

参考流程 flatten 时同样按照 x 的全部尺度、y 的全部尺度、z 的全部尺度排列。

---

## 18. Sampling Rate 规则

Sampling rate 的解析优先级：

1. settings 中显式提供；
2. 输入 summary 中提供；
3. 两者都提供时必须一致；
4. 两者都没有时失败。

Sampling rate 只用于：

* wavelet width 计算；
* max-filter window 计算；
* metadata。

不进行：

* 重采样；
* 插值；
* timestamp 修改；
* row count 修改。

冲突示例：

```text
settings sampling_rate_hz = 200
input summary sampling_rate_hz = 100
```

必须失败，而不是任选一个。

---

## 19. 输出文件

输出必须放在**与 rawIMU 相同的根目录（input root）下**，并按以下结构组织：

```text
<input-root>/
└── custom-wavelet/
    └── <output-stem>/
        ├── <output-stem>_spikeEvents.npy
        ├── <output-stem>_spike_sequence_offsets.npy
        ├── <output-stem>_spike_sequences.csv
        └── <output-stem>_spike_encoding_summary.json
```

其中：

* `<input-root>`：与 `rawIMU.npy` 所在的同一根目录；
* `custom-wavelet/`：固定 encoder namespace；
* `<output-stem>/`：由 user/action 或文件名派生的输出标识。

---

### `*_spikeEvents.npy`

```text
shape = (N, 3 × K)
```

要求：

* sample count 与输入相同；
* dtype 为配置的 float32 或 float64；
* 所有值 finite；
* `allow_pickle=False`；
* 不覆盖 `rawIMU.npy`。

---

### `*_spike_sequence_offsets.npy`

保存本次编码实际使用的 sequence 边界。

即使使用 `single-array` 模式，也必须显式保存：

```text
[0, N]
```

用于保证 encoding 可复现性与审计一致性。

---

### `*_spike_sequences.csv`

每个 sequence 一行：

```text
sequence_index
start_offset
stop_offset_exclusive
sample_count
nonzero_event_count
positive_event_count
negative_event_count
event_density
```

该文件仅用于 spike-encoding 层的统计与审计，不得写回或修改 segmentation manifest。

---

## 20. Summary 结构

示例：

```json
{
  "schema_version": 1,
  "encoder": {
    "name": "custom-wavelet",
    "representation": "signed_sparse_wavelet_extrema"
  },
  "source": {
    "raw_imu_path": "...",
    "summary_path": "...",
    "gravity_removal_method": "madgwick",
    "acceleration_semantics": "gravity_removed_linear_acceleration"
  },
  "input": {
    "sample_count": 100000,
    "channel_indices": [0, 1, 2],
    "channel_names": [
      "acceleration_x_g",
      "acceleration_y_g",
      "acceleration_z_g"
    ],
    "unit": "g",
    "sampling_rate_hz": 200.0
  },
  "settings": {
    "wavelet_name": "acceleration",
    "frequencies_hz": [
      0.5,
      1.0,
      2.0,
      4.0,
      8.0
    ],
    "frequency_order": "strictly_increasing_hz",
    "wavelet_widths_samples": [
      400,
      200,
      100,
      50,
      25
    ],
    "width_calculation": "int(sampling_rate_hz / frequency_hz)",
    "prony_denominator_order": 2,
    "prony_numerator_order": 2,
    "max_filter_time_s": 0.3,
    "max_filter_time_samples": 61
  },
  "sequence_processing": {
    "mode": "offsets",
    "sequence_count": 200,
    "state_reset_boundary": "sequence",
    "offsets_source": "..."
  },
  "output": {
    "sample_count": 100000,
    "channel_count": 15,
    "channel_order": "axis_major_frequency_minor",
    "dtype": "float32",
    "binary": false,
    "polarity_preserved": true,
    "amplitude_preserved": true
  },
  "statistics": {
    "nonzero_event_count": 0,
    "positive_event_count": 0,
    "negative_event_count": 0,
    "event_density": 0.0
  }
}
```

Summary 必须同时记录：

* 用户配置的 frequencies；
* 自动生成的 widths；
* width 计算公式；
* 最终 channel names；
* sequence reset 规则；
* input gravity-removal semantics。

---

## 21. 错误处理

以下错误应返回 CLI exit code `2`，且不显示 traceback：

* encoder 名称不存在；
* settings 文件不存在；
* settings JSON 无法解析；
* settings 包含未知字段；
* `frequencies_hz` 为空；
* frequency 非 finite；
* frequency 小于或等于 0；
* frequency 未严格升序；
* frequency 重复；
* frequency 超过 Nyquist；
* 不同 frequencies 生成重复 widths；
* wavelet name 未注册；
* sampling rate 缺失；
* sampling rate 冲突；
* Prony 拟合失败；
* IIR coefficients 非 finite；
* 输入不是 `(N, 9)`；
* 输入包含 NaN 或 Inf；
* offsets 非法；
* encoder 输出 sample count 改变；
* encoder 输出 channel count 与 channel names 不一致；
* 输出文件已存在且没有 `--overwrite`。

不得静默：

* 排序 frequencies；
* 去重 frequencies；
* 修改 sampling rate；
* 跳过失败的 sequence；
* 删除非 finite rows；
* 改变输出 channel order；
* 覆盖已有结果。

---

## 22. 原子化发布

所有 sequence 成功后才能发布结果。

推荐流程：

1. 验证输入；
2. 创建 encoder；
3. 完成所有 sequence 编码；
4. 验证聚合结果；
5. 写临时 NPY、CSV、JSON；
6. 重新加载 NPY 验证；
7. 原子替换到目标路径。

发布前必须满足：

```text
output rows == input rows
output columns == len(channel_names)
所有输出 finite
sequence offsets 覆盖完整输入
CSV 行数 == sequence count
summary 可序列化
```

`--overwrite` 只允许替换 spike-encoding 自己的输出。

不得删除或修改：

```text
rawIMU.npy
labels.npy
segment_offsets.npy
segment_lengths.npy
segmentation summary
padding outputs
```

---

## 23. 测试计划

### 23.1 公共框架

验证：

* registry 可创建 `custom-wavelet`；
* 未知 encoder 被拒绝；
* dummy encoder 可正常注册；
* 公共 runner 支持任意输出 channel count；
* 通用 CLI 不依赖 Custom Wavelet 私有字段；
* encoder 每个 sequence 前正确 reset。

### 23.2 Settings

验证：

* 默认 frequencies 正确；
* 自定义 frequencies 生效；
* frequency 数量可变；
* 严格升序通过；
* 乱序失败；
* 重复失败；
* 非正数失败；
* Nyquist 外频率失败；
* 重复 widths 失败；
* 未知 settings 字段失败；
* 未注册 wavelet name 失败。

### 23.3 Width 计算

验证：

```text
200 Hz + [0.5,1,2,4,8]
→ [400,200,100,50,25]

200 Hz + [1,2,5,10]
→ [200,100,40,20]
```

还需覆盖：

* 非整数除法；
* 高频导致过短 width；
* 相邻频率整数截断后产生相同 width。

### 23.4 Wavelet 和 Prony

验证：

* acceleration wavelet 与 vendor 结果一致；
* 默认五个尺度的 Prony coefficients 与 golden fixture 一致；
* 自定义 frequency 数量时 coefficient bank shape 正确；
* 所有 coefficients finite；
* filter order 生效；
* frequency、width、coefficients 映射顺序正确。

### 23.5 IIR

输入：

* impulse；
* constant；
* ramp；
* sine；
* 多频率 sine；
* 三轴不同信号；
* 固定 seed 随机信号。

验证：

* `step()` 与 `encode_sequence()` 一致；
* reset 后结果可重复；
* 三个轴互不串扰；
* 不同 frequency 通道不串位；
* sequence 之间不泄漏状态。

### 23.6 局部峰谷

验证：

* 非 extrema 输出 0；
* 正峰保留正幅值；
* 负谷保留负幅值；
* plateau 行为与 vendor 一致；
* 64 Hz 窗口为 19；
* 200 Hz 窗口为 61；
* 自定义 sampling rate 下窗口正确。

### 23.7 Vendor parity

对相同 `(N, 3)` 输入比较：

```text
vendor NeuromorphicIMUPipeline
WritingRing CustomWaveletEncoder
```

比较：

```text
wavelet responses
local-extrema events
flatten 后通道顺序
```

覆盖：

* 默认 frequencies；
* 64 Hz 默认参考配置；
* impulse；
* sine mixture；
* 正负运动；
* plateau；
* 固定随机输入。

### 23.8 I/O 和 CLI

验证：

* 只读取 `rawIMU[:, 0:3]`；
* 不修改输入；
* `single-array` 模式；
* `offsets` 模式；
* raw/low-pass/madgwick/xylo metadata；
* 自定义 frequencies；
* 动态 channel count；
* output summary；
* overwrite protection；
* atomic publication；
* 预期错误没有 traceback。

---

## 24. 实施阶段

### Phase 1：通用框架

实现：

```text
contracts.py
registry.py
io.py
runner.py
encode_spikes.py
```

使用 dummy encoder 验证框架与 Custom Wavelet 解耦。

完成条件：

* registry 可扩展；
* I/O 和 sequence runner 测试通过；
* 不涉及 segmentation 或 padding 修改。

### Phase 2：Custom Wavelet 核心

实现：

```text
CustomWaveletSettings
frequency validation
dynamic width calculation
wavelet registry
acceleration wavelet
Prony fitting
IIR filter bank
local extrema detector
channel naming
```

完成条件：

* 默认 frequencies 生效；
* 自定义 frequencies 生效；
* output channel count 动态变化；
* 核心单元测试通过。

### Phase 3：Vendor Parity

建立 vendor golden fixtures，比较：

* wavelet；
* coefficients；
* IIR response；
* extrema events；
* flatten 顺序。

完成条件：

* 默认 64 Hz 参考配置满足指定数值 tolerance；
* 差异均有明确原因和文档。

### Phase 4：输出、CLI 与文档

实现：

* NPY 输出；
* sequence offsets；
* sequence CSV；
* summary JSON；
* atomic publication；
* README 和 `docs/SPIKE_ENCODING.md`。

完成条件：

* raw、low-pass、madgwick、xylo 输出均可独立编码；
* 完整 pytest suite 通过；
* segmentation 和 padding 文件没有代码变更。

---

## 25. 预计修改文件

### 新增

```text
scripts/encode_spikes.py

configs/spike_encoding/custom_wavelet.json

src/writingring/spike_encoding/__init__.py
src/writingring/spike_encoding/contracts.py
src/writingring/spike_encoding/registry.py
src/writingring/spike_encoding/io.py
src/writingring/spike_encoding/runner.py
src/writingring/spike_encoding/encoders/__init__.py
src/writingring/spike_encoding/encoders/custom_wavelet.py

tests/test_spike_encoding_contracts.py
tests/test_spike_encoding_registry.py
tests/test_spike_encoding_io.py
tests/test_spike_encoding_runner.py
tests/test_custom_wavelet_settings.py
tests/test_custom_wavelet_encoder.py
tests/test_custom_wavelet_vendor_parity.py
tests/test_encode_spikes_cli.py

docs/SPIKE_ENCODING.md
```

### 可修改

```text
src/writingring/__init__.py
README.md
pyproject.toml
```

其中 `pyproject.toml` 仅在需要新增依赖或 CLI entry point 时修改。

### 明确不修改

```text
src/writingring/segmentation.py
src/writingring/board_event_segmentation.py
src/writingring/segment_padding.py
scripts/segment_ring_imu.py
scripts/pad_segmented_imu.py

vendor/Neuromorphic-Gravity/**
vendor/WritingRing/**
data_sample/**
```

---

## 26. 最终完成标准

实现完成后必须满足：

* 存在一个通用 spike-encoding CLI；
* 通用脚本支持通过 registry 接入其他 encoder；
* Custom Wavelet 是独立实现文件；
* 通用脚本不包含 Custom Wavelet 数学逻辑；
* 输入固定为九通道输出的前三个 g-domain acceleration channels；
* raw、low-pass、madgwick、xylo 均可作为输入；
* 不重新执行 gravity removal；
* 不修改 `rawIMU.npy`；
* 不修改 segmentation；
* 不修改 padding；
* 不重采样；
* 不改变 sample count；
* `frequencies_hz` 位于 Custom Wavelet settings；
* 默认值为 `[0.5, 1.0, 2.0, 4.0, 8.0]`；
* `frequencies_hz` 可由用户修改；
* frequencies 必须严格按 Hz 从小到大排列；
* 程序不静默排序；
* wavelet widths 根据 sampling rate 和 frequencies 动态计算；
* widths 不作为独立、可能冲突的配置；
* frequency 数量可变；
* 输出 channel count 为 `3 × len(frequencies_hz)`；
* 通道顺序固定为 axis-major、frequency-minor；
* Custom Wavelet 输出是有符号稀疏峰谷幅值；
* sequence 状态边界明确且可审计；
* 输出作为独立 NPY、CSV 和 JSON 发布；
* vendor parity 和完整 pytest suite 通过。
