# Plan: Configurable Five-Band Wavelet Encoder and Metadata-Driven Reconstruction

## 0. Plan Metadata

```text
Plan:
    configurable_wavelet_encoder_contract

Status:
    DRAFT

Workflow class:
    HIGH_RISK

Repository:
    hellowPluto78700/writingRing

Validated baseline:
    main @ 713d61ba43c3e5ba21abaa67afdf9a8beee8240e

Primary durable contracts:
    docs/notes/SPIKE_ENCODING.md
    docs/notes/GRAVITY_TO_SPIKE_PIPELINE.md
    docs/notes/SPIKE_SEGMENTATION_PIPELINE.md
    docs/notes/SEGMENTED_SPIKE_ACCEL_RECONSTRUCTION.md

User-facing pipeline surface:
    scripts/bash_script/preprocessing_pipeline/*.sh
    scripts/bash_script/preprocessing_pipeline/_common.bash
    scripts/encode_spikes.py
```

---

# 1. Purpose

当前 Custom Wavelet spike encoder 固定使用五个 wavelet frequency bands，默认：

```text
[0.5, 1, 2, 4, 8] Hz
```

本计划要把这组五个 wavelet bands 变成 pipeline-level 可配置实验参数，例如：

```text
[1, 2, 4, 8, 16] Hz
```

同时解决由此引出的四个 durable contract 问题：

1. event channel 不再使用 frequency / wavelet length 作为 channel identity，而只使用稳定的 axis + band index；
2. spike encoder 的完整、规范化配置必须保存在 metadata 中，并具有稳定 identity/fingerprint；
3. segmentation、padding、multi-action/multi-root consumers 必须验证它们读取的 SpikeIMU 使用相同 encoder configuration，不能仅依赖 `(N, 21)` shape 或通用 feature schema；
4. reconstruction 必须使用 encode 时实际采用的 wavelet widths，而不能继续硬编码 `[0.5, 1, 2, 4, 8]` 或重新从 frequency 假设整数 sample width。

最终目标是允许下面两类 dataset 都合法存在：

```text
Experiment A:
    frequencies_hz = [0.5, 1, 2, 4, 8]

Experiment B:
    frequencies_hz = [1, 2, 4, 8, 16]
```

但任何 consumer 在试图把它们作为同一个 feature space 混合训练/评估时，必须在 preflight metadata validation 阶段明确拒绝，而不是在模型训练后才暴露语义错误。

---

# 2. Why This Is HIGH_RISK

按照 `AGENTS.md`，本工作不是 FAST_FIX，因为它改变或强化以下 durable interfaces：

```text
channel semantics
pipeline CLI / bash configuration contract
persistent metadata contract
resume / stale-artifact identity
segmentation producer-consumer contract
padding producer-consumer contract
reconstruction producer-consumer contract
cross-action / multi-root dataset compatibility contract
```

因此采用：

```text
PRIMARY
  -> targeted probe only where a concrete unknown remains
  -> freeze TaskSpec
  -> luna_worker
  -> luna_verifier
```

每个 implementation task 都必须独立 verifier PASS。

Probe 不是机械必经步骤；如果 PRIMARY 已经从 docs/current code/tests 建立了足够事实，可以直接 freeze。只有存在具体 unresolved repository fact 时才使用 `luna_probe`。

`vendor/**` 与 `data_sample/**` 始终禁止修改。

---

# 3. Baseline Facts Already Established

以下事实已经从当前 docs/code 建立；开始执行时只在相关代码发生变化时重新确认，不需要重复全面调查。

## 3.1 Encoder layout

Custom Wavelet 当前仍是：

```text
3 acceleration axes × 5 wavelet bands = 15 event channels
```

SpikeIMU 仍是：

```text
0:15   wavelet event channels
15:18  acceleration x/y/z in m/s²
18:21  gyro x/y/z in rad/s
```

因此本计划保持：

```text
spikes.npy   -> (N, 15)
spikeIMU.npy -> (N, 21)
```

不改变模型读取前 15 channels 的基本 tensor contract。

## 3.2 Encoder frequency handling

当前 encoder 的 `frequencies_hz` 已经来自 settings，而不是必须写死在 encoder class 中。

当前 wavelet width 语义是：

```python
width_samples = int(sampling_rate_hz / frequency_hz)
```

所以 200 Hz sampling rate 下：

```text
[0.5, 1, 2, 4, 8]
    -> [400, 200, 100, 50, 25]

[1, 2, 4, 8, 16]
    -> [200, 100, 50, 25, 12]
```

注意：

```text
200 / 16 = 12.5
encoder 实际 width = int(12.5) = 12
```

这意味着 reconstruction 不能要求 `sampling_rate / frequency` 必须是整数，也不能自行用另一种 rounding policy 重新计算 width。

## 3.3 Current metadata

per-recording spike publication 已经有：

```text
metadata.json
    encoder
    settings
    output.channel_names
    spike_imu.channel_names
    sampling rate
    representation
    hashes / source provenance
```

因此本计划不是重建 metadata 系统，而是在现有 metadata 上增加一个稳定、规范化的 encoder identity contract。

## 3.4 Current downstream gap

当前 downstream 多数验证的是：

```text
feature_schema
channel_count
channel_names
units
sampling_rate_hz
target_length
padding_side
```

但没有把完整 spike encoder configuration 当作 compatibility identity。

如果 channel 改成 index-only naming：

```text
event_x_0 ... event_x_4
```

那么以下两个 encoding 会具有相同 shape、channel names、units、feature schema：

```text
A = [0.5, 1, 2, 4, 8]
B = [1, 2, 4, 8, 16]
```

因此必须新增显式 encoder compatibility validation。

## 3.5 Current reconstruction gap

两个 reconstruction scripts 当前仍把默认 frequencies 写死为：

```text
[0.5, 1, 2, 4, 8]
```

并存在要求 `sampling_rate / frequency` 为整数的 validation。

这与 encoder 对 16 Hz 在 200 Hz 下实际使用 width 12 的行为不兼容，必须修复。

---

# 4. Final Contract Decisions

这些是本 plan 的目标 contract。TaskSpec freeze 时可以根据 Probe 修正字段落点，但不能无声改变这些用户可见/持久化语义。

## 4.1 Five-band count remains fixed

本计划只允许改变五个 band 的 frequency values，不改变 band count。

Pipeline-level Custom Wavelet contract 保持：

```text
exactly 5 frequencies
strictly increasing
finite and > 0
all below Nyquist
```

因此：

```text
3 axes × 5 bands = 15 event channels
15 events + 6 trailing IMU = 21 SpikeIMU channels
```

仍是固定 contract。

Variable band-count encoder 不属于本计划。

---

## 4.2 User-facing frequency override

推荐并冻结为明确的 Hz 语义，不使用含糊的 `--encoderWavelet` 名称。

Pipeline wrapper：

```bash
ENCODER_FREQUENCIES_HZ="1 2 4 8 16"
export ENCODER_FREQUENCIES_HZ
```

CLI：

```bash
python scripts/encode_spikes.py \
  ... \
  --encoder-frequencies-hz 1 2 4 8 16
```

Precedence：

```text
explicit CLI / pipeline override
    > encoder settings JSON
    > encoder default
```

如果 wrapper 不提供 `ENCODER_FREQUENCIES_HZ`，继续使用 `configs/spike_encoding/custom_wavelet.json` 中的默认 frequencies。

不要采用：

```bash
--encoderWavelet [1 2 4 8 16]
```

因为 `[]` 不是 shell/argparse 的自然 list syntax，而且参数实际上表达 frequency，不是直接表达 wavelet kernel length。

---

## 4.3 Stable index-based channel names

15 个 event channels 的物理排列不改变，仍为 axis-major、band/frequency-minor：

```text
channel 0-4   -> x axis, band 0-4
channel 5-9   -> y axis, band 0-4
channel 10-14 -> z axis, band 0-4
```

新的稳定名字：

```text
event_x_0
event_x_1
event_x_2
event_x_3
event_x_4

event_y_0
...
event_y_4

event_z_0
...
event_z_4
```

channel name 不再编码 frequency 或 width。

band index 与 frequency/width 的含义必须从 metadata 读取。

---

## 4.4 Canonical spike encoder spec

每个新的 canonical SpikeIMU `metadata.json` 必须包含一个稳定的 normalized encoder specification，例如：

```json
{
  "spike_encoder": {
    "schema": "custom_wavelet_encoder_spec_v1",
    "name": "custom-wavelet",
    "wavelet_name": "acceleration",
    "sampling_rate_hz": 200.0,
    "frequency_band_count": 5,
    "frequencies_hz": [1.0, 2.0, 4.0, 8.0, 16.0],
    "wavelet_widths_samples": [200, 100, 50, 25, 12],
    "prony_denominator_order": 2,
    "prony_numerator_order": 2,
    "max_filter_time_s": 0.3,
    "max_filter_frequency_decades": 0.5,
    "boundary_padding_mode": "reflect",
    "event_index_semantics": "occurrence",
    "channel_order": "axis_major_frequency_minor",
    "event_channel_names": [
      "event_x_0", "event_x_1", "event_x_2", "event_x_3", "event_x_4",
      "event_y_0", "event_y_1", "event_y_2", "event_y_3", "event_y_4",
      "event_z_0", "event_z_1", "event_z_2", "event_z_3", "event_z_4"
    ],
    "post_encode_transform": null,
    "event_representation": "signed_sparse_wavelet_extrema"
  },
  "spike_encoder_spec_sha256": "..."
}
```

字段名的最终落点可以在 T001 Probe 后微调，但必须满足：

1. spec 只包含影响 event semantics / reconstruction / feature compatibility 的稳定字段；
2. 不包含 path、timestamp、run time、source file hash 等每次运行变化的字段；
3. numeric values 在计算 fingerprint 前规范化，避免 `1` vs `1.0` 造成无意义 identity 差异；
4. `wavelet_widths_samples` 是 encode 时实际使用的 widths，不是 downstream 推算值；
5. `post_encode_transform` 和最终 `event_representation` 必须属于 encoder identity，因为 signed 与 AbsRectify 不能混合作为同一个 event feature space。

现有 `metadata["settings"]` 保留作为完整 effective settings / backward-compatible diagnostics；新的 `spike_encoder` 是稳定 compatibility contract。

---

## 4.5 Fingerprint contract

建议 `spike_encoder_spec_sha256` 对 normalized `spike_encoder` object 做 canonical JSON SHA-256。

如果实现 fingerprint，则 hashing contract 必须 deterministic，例如：

```text
normalized types
+ sort_keys=True
+ compact separators
+ allow_nan=False
+ UTF-8 bytes
+ SHA-256
```

consumer 做 compatibility check 时优先比较 fingerprint，同时可以在报错信息中输出关键 field differences。

不要仅比较 `frequencies_hz`，因为以下变化同样影响 feature semantics：

```text
wavelet implementation/name
Prony orders
filter/extrema settings
sampling rate
post-encode transform
occurrence/index semantics
channel order
```

---

## 4.6 Feature schema remains the 21-channel layout identifier

本计划默认不进行全仓 `signed_wavelet_events_plus_imu_v1` schema rename/migration。

理由：

```text
15 event + 3 acceleration + 3 gyro
```

的 layout 没有改变。

具体 encoder identity 由：

```text
spike_encoder
spike_encoder_spec_sha256
```

表达。

如果 Probe 发现某个 consumer 把 `signed_wavelet_events_plus_imu_v1` 名称中的 `signed` 当作强制 polarity guarantee，而不是 layout identifier，则必须 REPLAN；不能在当前 plan 中默默破坏该 consumer。

---

## 4.7 Reconstruction authority

Reconstruction 的 DSP authority 必须是：

```text
wavelet_widths_samples from encoder metadata
```

而不是：

```text
hard-coded frequency list
```

也不是：

```text
recompute width using sampling_rate/frequency with its own rounding rule
```

`frequencies_hz` 保留用于解释、审计和实验 provenance；真正生成 reconstruction kernel 的 width 必须与 encoder 当时实际使用值完全相同。

因此：

```text
200 Hz + 16 Hz
encoder width = 12
reconstruction width = metadata[...]=12
```

必须合法。

---

## 4.8 Fail closed on missing encoder identity in new downstream flows

由于 channel names 变成 index-only，缺少 encoder metadata 时无法安全解释 band 0-4。

因此新 pipeline downstream 不得在 metadata 缺失时静默假设：

```text
[0.5, 1, 2, 4, 8]
```

对于旧 artifact：

- pipeline `continue` 应把缺少新 encoder identity 的 SpikeIMU 判定为 stale，并重新 encode；
- reconstruction / multi-root combine 不得静默套用 default；应明确报错并要求使用新 pipeline 重新生成 compatible artifacts；
- 不要求本计划增加 metadata-less legacy reconstruction fallback。

这是一次有意的 one-time artifact refresh，以换取之后稳定的 feature identity。

---

## 4.9 Output path policy

本计划不自动把 frequency set 或 fingerprint 插入现有 output directory hierarchy。

也就是说同一个：

```text
OUTPUT_BASE
```

下切换 encoder configuration 时，`overwrite/continue` 由 metadata identity 决定是否重建。

如果用户希望同时保留多个 encoder variants 做实验，应使用不同 `OUTPUT_BASE` / experiment root。

自动 variant-directory naming 属于后续独立需求，不在本计划范围。

---

# 5. Non-Goals

本计划不做：

- 支持 4 个、6 个或任意数量 wavelet bands；
- 改变 SpikeIMU 的 21-channel layout；
- 改变前 15 channels 的 axis-major ordering；
- 改变 gravity preprocessing；
- 改变 segmentation boundary 算法；
- 改变 padding policy；
- 改变 SNN / CNN architecture；
- 自动为每个 encoder configuration 生成新的 output path；
- 重写 wavelet mathematical definition；
- 改变 extrema detector 的 occurrence alignment contract；
- 为缺少 encoder metadata 的历史 segmentation/padding artifacts 猜测 wavelet configuration；
- 修改 `vendor/**` 或 `data_sample/**`。

---

# 6. Dependency DAG

```text
T001 Canonical configurable encoder + stable metadata identity
  |
  +---------> T002 Pipeline override + resume/stale-artifact protection
  |
  +---------> T003 Encoder identity propagation through segmentation/padding
                 |
                 +---------> T004 Metadata-driven reconstruction
                 |
                 +---------> T005 Cross-action compatibility + integration/docs closure

T002 + T003 + T004
          \
           +------> T005
```

| Task | Goal | Dependencies | Initial state |
| --- | --- | --- | --- |
| **T001** | Make the five-band encoder configurable, use index-based channels, publish canonical encoder identity | — | DRAFT |
| **T002** | Expose frequency override in pipeline/CLI and make continue/resume reject stale encoder artifacts | T001 | DRAFT |
| **T003** | Carry encoder identity through recording features, segmentation and padding; reject intra-dataset mismatch | T001 | DRAFT |
| **T004** | Reconstruct from published encoder widths/spec instead of hard-coded frequencies | T003 | DRAFT |
| **T005** | Enforce cross-action/multi-root encoder compatibility, run integration regression, update durable docs | T002-T004 | DRAFT |

Parallelism guidance：

- T002 与 T003 在 T001 verifier PASS 后可以并行，但只有 frozen write scopes 不重叠时才允许；
- T004 必须等待 T003，因为 reconstruction 需要 segmentation/padding 中可用的 encoder spec；
- T005 等待 T002-T004 全部 verifier PASS；
- 不允许两个 worker 同时修改同一个 central loader、pipeline common bash 或同一个 test file。

---

# 7. T001 — DRAFT TaskSpec: Configurable Encoder and Canonical Identity

## Goal

让 Custom Wavelet encoder 在保持 exactly-five-band / 15-event-channel contract 的情况下接受不同 frequency sets，并把 event channels 改成稳定 index-based identity，同时发布完整、可比较、可重建的 canonical spike encoder metadata。

## Concrete unresolved facts / Probe scope

T001 建议使用一次 targeted `luna_probe`，只回答以下具体问题：

1. 全仓是否存在依赖当前 frequency-bearing channel names 字符串内容的 consumer，而不仅是使用前 15 column indices；
2. 当前 `encoding_metadata` / `output_metadata` 已经包含哪些字段，可以直接用于 normalized spec；
3. 是否存在 consumer 把 feature schema 名中的 `signed` 当作强 polarity guarantee；
4. vendor parity fixtures 是否把 output channel names 也作为 parity contract，还是只比较 numerical values/order。

最小 Probe surface：

```text
src/writingring/spike_encoding/encoders/custom_wavelet.py
src/writingring/spike_encoding/publication.py
src/writingring/spike_encoding/contracts.py
src/writingring/spike_encoding/runner.py

tests/test_custom_wavelet_encoder.py
tests/test_custom_wavelet_settings.py
tests/test_custom_wavelet_vendor_parity.py
tests/test_spike_encoding_publication.py

repo-wide exact search for current frequency-based event channel name format
```

Probe 不应检查 `vendor/**`，除非 current test/reference 明确无法解释且 PRIMARY 重新授权。

## Required behavior

Freeze 后必须满足：

- Custom Wavelet pipeline 仍要求 exactly 5 frequencies；
- frequency validation 继续要求 strictly increasing、positive、finite、below Nyquist；
- default frequencies 保持 `[0.5, 1, 2, 4, 8]`；
- changing frequency set changes filter/wavelet widths and output values but not output shape/order；
- event channel names 固定为 `event_{axis}_{band_index}`；
- `channel 0..14` 到 `(axis, band_index)` 的 mapping 不随 frequency set 改变；
- canonical metadata publishes actual frequencies and actual encoder-used widths；
- canonical encoder identity includes all settings that materially affect event semantics；
- metadata publishes deterministic `spike_encoder_spec_sha256` or equivalent normalized identity；
- existing detailed `settings` metadata remains available；
- numerical encoding for the default frequency set remains unchanged except metadata/channel-name text changes。

## Preserved contracts

必须保持：

```text
spikes.npy.shape == (N, 15)
spikeIMU.npy.shape == (N, 21)
event columns remain axis-major
spikeIMU[:, 15:21] remains row-for-row source IMU trailing columns
canonical timestamps remain row-aligned and unchanged
occurrence alignment behavior remains unchanged
signed/AbsRectify mathematical behavior remains unchanged
```

Do not alter the acceleration wavelet formula or extrema detection algorithm as part of this task.

## Anticipated allowed write scope

Freeze exact write scope after Probe. Expected maximum surface：

```text
src/writingring/spike_encoding/**
configs/spike_encoding/custom_wavelet.json   # only if metadata/default representation requires it
tests/test_custom_wavelet_*.py
tests/test_spike_encoding_publication.py
```

## Forbidden writes

```text
segmentation implementation
padding implementation
reconstruction scripts
pipeline bash wrappers
SNN/CNN dataset loaders
docs/notes/**
README.md
vendor/**
data_sample/**
```

These belong to later tasks.

## Acceptance criteria

1. Default `[0.5,1,2,4,8]` still produces exactly 15 event channels and the same numerical event matrix as the pre-change encoder for the same fixture, ignoring renamed metadata/channel labels.
2. `[1,2,4,8,16]` is accepted at 200 Hz and publishes actual widths `[200,100,50,25,12]`.
3. Channel names are exactly the stable 15 index-based names in the documented order.
4. Both frequency sets produce identical channel names/order while publishing different encoder specs/fingerprints.
5. Changing only `post_encode_transform` changes encoder identity.
6. Changing any event-affecting encoder setting covered by the spec changes encoder identity.
7. Reordering or invalid frequency values is rejected before encoding.
8. Metadata has no ambiguous need to infer frequency from channel name.
9. Focused tests PASS.
10. Independent verifier PASS.

## Validation

Run cheap-to-expensive, adjusting exact filenames after Probe：

```bash
conda run -n writingring-gpu python -m pytest -q \
  tests/test_custom_wavelet_settings.py \
  tests/test_custom_wavelet_encoder.py \
  tests/test_custom_wavelet_vendor_parity.py \
  tests/test_spike_encoding_publication.py
```

Fall back to `writingring-viz` only if the GPU env is unavailable and dependencies permit.

## Replan triggers

Return REPLAN if：

- changing channel names breaks a documented external/public API that cannot be migrated inside this plan；
- current feature schema is proven to guarantee signed polarity rather than act as layout identifier；
- exact five-band output is not sufficient to preserve a current downstream public contract；
- encoder uses a different width rule than the currently established `int(fs/f)` behavior；
- correct fingerprint identity requires a broader schema migration not represented here。

---

# 8. T002 — DRAFT TaskSpec: Pipeline Override and Resume Identity

## Goal

让用户只修改 pipeline bash configuration 就能选择五个 wavelet frequencies，并确保 `PIPELINE_MODE=continue` 不会复用由不同 encoder configuration 或旧 channel contract 生成的 SpikeIMU artifact。

## Probe policy

如果 T001 已经 freeze 了 exact metadata identity location，并且 `_common.bash` 当前 resume paths 已知，则无需单独 Probe。

只有在无法确定 encode-stage resume validator 的唯一入口时才使用 targeted Probe。

## Required behavior

Pipeline 支持：

```bash
ENCODER_FREQUENCIES_HZ="1 2 4 8 16"
```

并向 encoder CLI 传递：

```text
--encoder-frequencies-hz 1 2 4 8 16
```

要求：

- unset override -> continue using settings JSON frequencies；
- explicit override -> overrides settings JSON；
- override must contain exactly five valid numeric values；
- pipeline log records the effective CLI invocation；
- new metadata records effective values, not only source JSON values；
- resume validation compares the expected encoder identity with existing artifact metadata；
- changing frequencies invalidates old encode output；
- changing post-encode transform invalidates old encode output；
- changing any encoder-spec field owned by the pipeline invalidates old encode output；
- artifact missing the new encoder identity is stale after this migration；
- stale encode output triggers the pipeline's existing safe rebuild path rather than being silently accepted。

## Preserved contracts

- Existing wrappers with no new variable preserve default numerical encoder configuration；
- `PIPELINE_MODE=overwrite` remains full rebuild behavior；
- `PIPELINE_MODE=continue` remains resumable when the existing artifact matches the requested encoder identity；
- existing preprocessing outputs must not be regenerated solely because encoder frequencies changed unless current orchestration already requires restarting from preprocess for a discovered invalid downstream state；
- no new shell list syntax using literal brackets is introduced。

## Anticipated allowed write scope

```text
scripts/encode_spikes.py
scripts/bash_script/preprocessing_pipeline/_common.bash
scripts/bash_script/preprocessing_pipeline/*.sh   # only examples/default declarations if needed
focused encode CLI / pipeline tests
```

Do not mass-edit all wrappers unless a shared `_common.bash` default is insufficient.

## Acceptance criteria

1. No override reproduces default settings behavior.
2. `ENCODER_FREQUENCIES_HZ="1 2 4 8 16"` reaches the encoder as five ordered float values.
3. CLI explicit override wins over JSON settings.
4. Invalid count/order/value fails before publication.
5. `continue` accepts an existing artifact only when encoder identity matches.
6. Default-old artifact without the new identity is treated as stale once, because channel identity contract changed.
7. Frequency mismatch produces an actionable stale/mismatch diagnostic.
8. AbsRectify vs signed mismatch remains rejected.
9. Focused CLI and pipeline tests PASS.
10. Independent verifier PASS.

## Validation

Expected focused surface：

```bash
conda run -n writingring-gpu python -m pytest -q \
  tests/test_encode_spikes_cli.py \
  tests/test_gravity_to_spike_pipeline.py \
  tests/test_action0_pipeline_scripts.py
```

Use exact existing filenames found by worker; do not invent a new test subsystem if equivalent tests already exist.

## Replan triggers

- pipeline has multiple independent encode entry points that cannot share one authoritative override contract；
- continue-mode stage invalidation cannot safely rebuild only from the correct upstream stage under current orchestration；
- a public CLI already uses an incompatible frequency override flag that must be preserved。

---

# 9. T003 — DRAFT TaskSpec: Encoder Provenance Propagation and Intra-Dataset Compatibility

## Goal

让 encoder identity 从 per-recording SpikeIMU metadata 一路传播到 segmentation 和 padding artifacts，并确保一个被聚合的数据集不能包含不同 spike encoder configurations。

## Concrete unresolved facts / Probe scope

建议 targeted Probe 只确认：

1. label segmentation 与 aligned-board segmentation 各自创建 summary 的唯一入口；
2. root-level padding summary 是如何从多个 user/action package 聚合；
3. 是否存在除这两条 segmentation route 以外的 canonical SpikeIMU segmentation producer；
4. current loaders 是否已经有可复用的 producer metadata dataclass/validation helper。

最小 surface：

```text
src/writingring/recording_features.py
src/writingring/segmentation.py
src/writingring/board_event_segmentation.py
src/writingring/segment_padding.py
focused segmentation/padding tests
```

## Required behavior

### Per-recording load

`load_spike_imu_features()` 或等价 canonical loader 必须：

- read `spike_encoder` and fingerprint from recording `metadata.json`；
- validate spec structure and hash；
- expose immutable encoder identity in the loaded feature object；
- continue validating values/timestamps/hash/channel counts as before。

### Same action / aggregation

聚合同一 user/action 的多个 recordings 时，除现有：

```text
input_kind
feature_schema
channel_count/channel_names
units
sampling rate
```

外，必须要求：

```text
spike_encoder_spec_sha256 identical
```

如果不同，明确 reject，而不是 intersect、coerce 或采用 first recording。

这条规则同时适用于：

```text
label segmentation
aligned-board-events segmentation
```

### Segmentation publication

segmentation summary 必须保存：

```text
spike_encoder
spike_encoder_spec_sha256
```

对于 raw-ring input，这些字段应为 absent/None according to current summary style，而不是伪造 SpikeIMU encoder。

### Padding validation/publication

padding consumer 必须：

- require SpikeIMU segmentation input to contain valid encoder identity；
- propagate identical encoder identity into each package summary；
- root-level `padding_dataset_summary.json` publishes one authoritative encoder identity only after proving every included package uses the same spec；
- if padding input root contains packages with different encoder specs, fail before producing a mixed root-level dataset。

## Preserved contracts

- segmentation slicing/boundaries remain identical；
- event channel values are not transformed during propagation；
- padding operates only on time dimension and does not reinterpret event channels；
- existing channel names/units/sampling-rate metadata remain published；
- label and board alignment behavior remains unchanged；
- raw-ring segmentation remains supported without a fake spike encoder spec。

## Anticipated allowed write scope

```text
src/writingring/recording_features.py
src/writingring/segmentation.py
src/writingring/board_event_segmentation.py
src/writingring/segment_padding.py
focused recording-feature / segmentation / padding tests
```

## Acceptance criteria

1. Two recordings with same normalized encoder spec aggregate successfully.
2. Two recordings with identical 21-channel shape/names but different frequencies are rejected.
3. Same frequencies but different post-encode transform are rejected.
4. Same frequencies but another event-affecting encoder field differs -> rejected.
5. Both segmentation modes publish the same encoder spec/hash they consumed.
6. Padding preserves spec/hash without mutation.
7. Padding root rejects packages with different encoder identities before publishing a combined dataset summary.
8. Raw-ring segmentation remains unaffected.
9. Existing segmentation boundaries/values tests remain unchanged except expected metadata additions/channel-name migration.
10. Independent verifier PASS.

## Validation

Focused tests should cover at least：

```text
recording feature metadata load/validation
label SpikeIMU segmentation
aligned-board SpikeIMU segmentation
segment padding
negative mixed-spec fixtures
```

Run relevant current test files only; one broader integration run is deferred to T005.

## Replan triggers

- there is no single encoder identity available at per-recording publication；
- a canonical segmentation path bypasses `recording_features` and cannot safely obtain metadata within anticipated scope；
- current padding design intentionally permits heterogeneous feature semantics in one root and changing that is a broader public dataset contract decision。

---

# 10. T004 — DRAFT TaskSpec: Metadata-Driven Reconstruction

## Goal

移除 reconstruction 对 `[0.5,1,2,4,8]` 的 hard-coded semantic dependency，使 segmented 和 padded reconstruction 使用 source SpikeIMU encoder 实际发布的 wavelet widths。

## Probe policy

当前 hard-coded frequencies 和 integer-ratio validation 已经明确存在，因此默认不需要 Probe。

只有 worker 发现 reconstruction metadata 无法从 T003 output 唯一获取 encoder widths 时才返回 `NEEDS_REPLAN`。

## Required behavior

### Source of truth

两个 reconstruction paths 都必须从 source segmentation/padding metadata 读取：

```text
spike_encoder.wavelet_widths_samples
spike_encoder.frequencies_hz
spike_encoder_spec_sha256
sampling_rate_hz
```

`wavelet_widths_samples` 是 kernel generation 的 authority。

### No hidden default

禁止：

```text
DEFAULT_FREQUENCIES_HZ = (0.5,1,2,4,8)
```

继续决定 reconstruction kernel semantics。

代码可以保留 default 常量用于明确的 test fixture / documentation helper，但 production reconstruction 不得在 source metadata 缺失时静默 fallback。

### 16 Hz behavior

200 Hz 下：

```text
frequencies_hz = [1,2,4,8,16]
wavelet_widths_samples = [200,100,50,25,12]
```

必须成功 reconstruction。

不要再以：

```text
200 / 16 = 12.5 is non-integer
```

为理由拒绝，因为 encode 时已经合法使用 width 12。

### Reconstruction metadata

reconstruction output metadata 必须记录：

```text
source_spike_encoder_spec_sha256
frequencies_hz used for provenance
wavelet_widths_samples actually used
scale_divisor
wavelet reconstruction method
```

使 reconstruction artifact 能证明它使用了与 source events 相同的 band widths。

### Segment behavior

必须保持现有 reconstruction boundary contract：

```text
reconstruct each original variable-length segment independently
zero context outside each segment
never convolve across segment boundaries
padded reconstruction reconstructs unpadded segment first, then right-pads zeros
```

## Preserved contracts

- only event channels `0:15` participate in reconstruction；
- trailing `15:21` IMU values are not used to synthesize reconstructed acceleration；
- output remains x/y/z acceleration in m/s²；
- existing scale divisor remains unchanged unless a separate validated scientific change is requested；
- AbsRectify remains lossy; reconstruction does not claim to recover lost polarity；
- padded convolution never sees synthetic pad samples。

## Anticipated allowed write scope

```text
scripts/reconstruct_segmented_spike_accel.py
scripts/reconstruct_padded_spike_accel.py
focused reconstruction tests
reusable reconstruction helper module only if worker demonstrates duplication warrants it within TaskSpec scope
```

Do not refactor unrelated acceleration evaluation code.

## Acceptance criteria

1. Default encoder metadata reconstructs with widths `[400,200,100,50,25]`.
2. `[1,2,4,8,16]` at 200 Hz reconstructs with widths `[200,100,50,25,12]` without integer-ratio rejection.
3. Reconstruction does not infer widths from channel names.
4. Reconstruction fails clearly when required encoder spec/widths are absent or corrupt.
5. Reconstruction verifies source encoder fingerprint/spec consistency before use.
6. Segmented reconstruction still cannot leak energy across segment boundaries.
7. Padded reconstruction still reconstructs first and pads second.
8. Output metadata records the exact encoder identity and widths used.
9. Existing default numerical reconstruction parity remains unchanged for the default encoder spec.
10. Independent verifier PASS.

## Validation

Run focused reconstruction tests plus current notebook/reference parity fixture tests if they are already automated.

At minimum include a synthetic/default case and a 16-Hz/non-integral-frequency-ratio case.

## Replan triggers

- encoder metadata widths do not reproduce the actual encoder kernels；
- current reconstruction intentionally uses a mathematically distinct width rule documented as scientific contract；
- required encoder provenance is lost before segmentation despite T003；
- preserving notebook/reference parity requires changing the scientific reconstruction method rather than metadata plumbing。

---

# 11. T005 — DRAFT TaskSpec: Cross-Action Compatibility, Integration, and Documentation Closure

## Goal

确保任何把两个 actions / dataset roots 组合到同一个 training/evaluation feature space 的 consumer 都验证 spike encoder identity，并完成本 feature 的 end-to-end regression 与 durable documentation。

## Concrete unresolved facts / Probe scope

T005 建议 targeted Probe，因为 exact cross-action consumers 可能继续演化。

Probe 只需要找到：

1. 当前哪些 loaders 支持一个以上 padded root / action；
2. 哪些 loaders 只检查 shape/schema/sampling/target length 而没有 encoder identity；
3. Action-specific dataset loaders 是否实际上不会跨 action，因此无需修改；
4. acceleration reconstruction evaluation multi-root compatibility 是不是当前唯一 production multi-root path。

预计至少检查：

```text
snn/accel_reconstruction_eval/datasets.py
snn/action0_dataset.py
current multi-action/multi-root training entry points
focused dataset compatibility tests
```

不要因为存在一个 loader 就修改它；只有它确实组合不同 action/root 时才进入 frozen scope。

## Required behavior

任何 consumer 如果要把多个 SpikeIMU/padded roots 视为同一个 feature space，必须要求：

```text
spike_encoder_spec_sha256 identical
```

并继续保留现有 compatibility checks，例如：

```text
feature_schema
channel_count
target_length
sampling_rate_hz
padding_side
```

典型结果：

```text
Action 0:
    [1,2,4,8,16], signed

Action 1:
    [1,2,4,8,16], signed

=> compatible
```

但：

```text
Action 0:
    [0.5,1,2,4,8]

Action 1:
    [1,2,4,8,16]

=> reject before dataset construction/training
```

同样：

```text
same frequencies + signed
vs
same frequencies + AbsRectify
```

必须 reject。

### Error diagnostics

错误信息应至少说明：

```text
which roots/actions differ
expected encoder fingerprint
actual encoder fingerprint
one or more human-readable differing encoder fields when available
```

不要只报 `metadata mismatch`。

## Integration matrix

T005 必须至少验证下面矩阵：

| Case | Expected |
| --- | --- |
| default freq set, same spec across recordings | PASS |
| `[1,2,4,8,16]`, same spec across recordings | PASS |
| same spec across Action 0 + Action 1 | PASS |
| different frequencies across actions | FAIL before training |
| same frequencies, different post-transform | FAIL before training |
| old artifact missing encoder identity in new combine path | FAIL / stale, never silently default |
| `continue` with exact same new spec | reuse valid encode stage |
| `continue` after frequency change | rebuild encode/downstream as required |
| default reconstruction | PASS, numerical parity preserved |
| 16 Hz / width 12 reconstruction | PASS |

## Durable docs updates

After implementation behavior is verified, update the minimum durable docs that describe actual contracts：

```text
docs/notes/SPIKE_ENCODING.md
docs/notes/GRAVITY_TO_SPIKE_PIPELINE.md
docs/notes/SPIKE_SEGMENTATION_PIPELINE.md
docs/notes/SEGMENTED_SPIKE_ACCEL_RECONSTRUCTION.md
README.md                    # only user-facing pipeline invocation/entry points
```

Docs must explicitly state：

- five frequencies are configurable；
- canonical bash/CLI syntax；
- event channel names are index-based；
- `band_index -> frequency/width` comes from metadata；
- encoder spec/fingerprint defines feature compatibility；
- segmentation/padding reject mixed specs；
- multi-action consumers reject mismatched specs；
- reconstruction uses published widths, not hard-coded defaults；
- 16 Hz at 200 Hz uses width 12 under current encoder rule；
- old artifacts without the new identity must be regenerated for new downstream flows。

## Preserved contracts

- models still receive first 15 event channels in the same axis-major order；
- raw acceleration baseline still uses `15:18`；
- alignment transient channels remain `15:21`；
- experiment split/training semantics are not changed by this plan；
- no model architecture migration。

## Anticipated allowed write scope

Freeze after Probe. Expected surface：

```text
multi-root/multi-action dataset compatibility loaders
focused compatibility tests
README.md
docs/notes/SPIKE_ENCODING.md
docs/notes/GRAVITY_TO_SPIKE_PIPELINE.md
docs/notes/SPIKE_SEGMENTATION_PIPELINE.md
docs/notes/SEGMENTED_SPIKE_ACCEL_RECONSTRUCTION.md
this plan file / WORKBOARD only if repository workflow uses them
```

## Acceptance criteria

1. Same encoder spec across two actions is accepted by the actual multi-root consumer.
2. Different frequencies across actions are rejected before model training/evaluation.
3. Different post-transform across actions are rejected.
4. Missing encoder identity is not silently treated as default.
5. Error identifies conflicting roots/actions and encoder identities.
6. All T001-T004 focused suites remain PASS in the integrated tree.
7. Relevant pipeline integration tests PASS.
8. One final broad regression is run because this is a HIGH_RISK cross-pipeline/channel-semantics change.
9. Durable docs match verified implementation behavior.
10. Independent verifier PASS over complete plan surface.
11. Plan status changes to DONE only after all tasks verifier PASS.

## Final validation order

Run once per final implementation state：

```text
1. focused encoder tests
2. focused CLI/resume tests
3. focused recording/segmentation/padding tests
4. focused reconstruction tests
5. focused multi-root/action compatibility tests
6. relevant pipeline integration tests
7. full pytest once, if environment/dependencies permit
8. final diff/docs consistency review
```

Do not repeatedly rerun unchanged expensive commands.

## Replan triggers

- multiple current multi-action consumers have intentionally different encoder-compatibility semantics；
- a public training protocol intentionally allows heterogeneous encoder feature spaces in one model input；
- root-level metadata cannot uniquely identify the producer encoder without a broader artifact migration；
- final integration reveals that maintaining the existing feature schema is unsafe。

---

# 12. Expected End-State Data Flow

```text
pipeline wrapper
    |
    | ENCODER_FREQUENCIES_HZ="1 2 4 8 16"
    v
scripts/encode_spikes.py
    |
    | --encoder-frequencies-hz 1 2 4 8 16
    v
CustomWaveletEncoder
    |
    | frequencies_hz
    |     [1, 2, 4, 8, 16]
    |
    | actual widths
    |     [200, 100, 50, 25, 12]
    |
    | channels
    |     event_x_0 ... event_z_4
    v
spikes.npy (N, 15)
spikeIMU.npy (N, 21)
metadata.json
    |
    +-- spike_encoder
    +-- spike_encoder_spec_sha256
    |
    v
recording_features
    |
    | validate spec + fingerprint
    v
segmentation
    |
    | require all recordings same encoder identity
    | propagate spec
    v
segmentation summary
    |
    v
padding
    |
    | require all packages same encoder identity
    | propagate spec
    v
padding_dataset_summary.json
    |
    +-----------------------------+
    |                             |
    v                             v
multi-action loader          reconstruction
    |                             |
    | compare encoder hash        | read actual widths
    |                             | never hard-code frequencies
    v                             v
compatible -> train          reconstructed accel
mismatch   -> reject         + source encoder provenance
```

---

# 13. Example Compatibility Semantics

## Compatible

```json
Action0.spike_encoder = {
  "frequencies_hz": [1,2,4,8,16],
  "wavelet_widths_samples": [200,100,50,25,12],
  "post_encode_transform": null
}

Action1.spike_encoder = {
  "frequencies_hz": [1,2,4,8,16],
  "wavelet_widths_samples": [200,100,50,25,12],
  "post_encode_transform": null
}
```

```text
same normalized spec
-> same fingerprint
-> may combine if all other existing dataset compatibility checks also pass
```

## Incompatible: frequencies

```text
Action0 = [0.5,1,2,4,8]
Action1 = [1,2,4,8,16]
```

Even though both are：

```text
21 channels
same index-based channel names
same sampling rate
same feature_schema
```

result must be：

```text
REJECT
```

## Incompatible: transform

```text
Action0 = signed
Action1 = AbsRectify
```

result：

```text
REJECT
```

## Incompatible: filter implementation settings

```text
same frequencies
but different Prony orders / relevant filter settings
```

result：

```text
REJECT
```

---

# 14. Migration / Backward Compatibility Policy

这次 channel names 与 encoder identity 是 durable artifact contract change，因此采用明确 migration policy：

## Source/config compatibility

保持兼容：

```text
existing custom_wavelet.json default frequencies
existing 15/21 tensor layout
existing sampling rate handling
existing signed/AbsRectify behavior
existing pipeline wrappers without new frequency override
```

## Artifact compatibility

新的 downstream contract 不保证继续接受旧的 metadata-less encoder identity。

特别是旧 SpikeIMU 即使 numerical data 是默认 bands：

```text
[0.5,1,2,4,8]
```

也因为旧 channel naming / 缺少 canonical spec，不能安全满足新 index-based channel contract。

因此：

```text
new pipeline + old SpikeIMU metadata
    -> stale
    -> regenerate encode/downstream artifacts
```

不要通过“如果缺失就假设 default”来掩盖迁移。

---

# 15. Scientific/Experimental Invariants

这个 feature 是为了支持 wavelet-band ablation/optimization，而不是改变实验的其他变量。

做 encoder comparison 时，实验记录至少需要保存：

```text
spike_encoder_spec_sha256
frequencies_hz
wavelet_widths_samples
post_encode_transform
sampling_rate_hz
dataset/output root
```

如果两个模型比较不同 frequency sets，应明确视为不同 input representation experiments。

不要把不同 encoder specs 的 samples 混入同一个训练集，除非未来另有明确研究设计并定义 heterogeneous representation contract。

---

# 16. Plan Completion Checklist

只有全部满足才关闭 plan：

- [ ] T001 verifier PASS
- [ ] T002 verifier PASS
- [ ] T003 verifier PASS
- [ ] T004 verifier PASS
- [ ] T005 verifier PASS
- [ ] Default encoder numerical behavior preserved
- [ ] `[1,2,4,8,16]` encode succeeds at 200 Hz
- [ ] 16 Hz reconstruct uses width 12 and succeeds
- [ ] index-based event channel names are stable
- [ ] encoder spec/fingerprint published per recording
- [ ] segmentation propagates and validates encoder identity
- [ ] padding propagates and validates encoder identity
- [ ] cross-action/multi-root mismatch is rejected
- [ ] `continue` invalidates stale encoder artifacts
- [ ] old missing-spec artifacts are never silently treated as default in new downstream flows
- [ ] docs/notes updated from verified implementation
- [ ] README user-facing invocation updated
- [ ] one final broad regression completed or a concrete environment limitation documented
- [ ] no writes under `vendor/**`
- [ ] no writes under `data_sample/**`

---

# 17. Recommended Repository Location

在开始执行前建议把本文件放到：

```text
docs/plans/TODO/09_Configurable_Wavelet_Encoder_Contract_Plan.md
```

全部 task 完成并 verifier PASS 后，按照仓库已有 convention 移到：

```text
docs/plans/Done/09_Configurable_Wavelet_Encoder_Contract_Plan.md
```

如果当前 TODO numbering 已被占用，PRIMARY 在写入仓库前只调整编号/文件名，不改变本 plan 的 task identity 或 dependency DAG。
