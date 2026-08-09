## 1. 最终目标

最终 Action-0 pipeline 中 `_common.bash` 有一个非常明确的实验开关：

```bash
POST_ENCODE_TRANSFORM="${POST_ENCODE_TRANSFORM:-none}"
```

允许：

```text
none
AbsRectify
```

你以后主要只需要改这里。

整体数据流应固定成：

```text
preprocessed acceleration
        ↓
wavelet filter bank
        ↓
local extrema detector
        ↓
occurrence alignment
        ↓
SIGNED encoded spikes
        ↓
post_encode_transform
        ├── none
        │     ↓
        │   signed spikes
        │
        └── AbsRectify
              ↓
            abs(spikes)
        ↓
spikes.npy
        ↓
spikeIMU.npy[:, 0:15]
```

这里最重要的 contract 是：

```python
rectified == np.abs(signed)
```

并且：

```python
rectified != 0
```

与：

```python
signed != 0
```

必须逐元素完全相同。

也就是说 **AbsRectify 绝不能重新决定 extrema，也不能改变 timestamp / occurrence row / channel / sparsity**。当前 encoder 是在 extrema detector 后进行 occurrence alignment，因此 transform 应放在已经裁回 `(N, C)` 的最终 encoded matrix 上，而不是 wavelet response 上。现有实现和文档明确把 extrema occurrence alignment 与 signed extrema 作为当前 contract。

---

# 2. Task DAG

根据 `AGENTS.md` 的要求，这里应该先建立 coarse DAG，而不是提前把所有下游 TaskSpec 全部冻结；只有 dependency-ready task 才 Probe，然后 Freeze。

| Task     | 目标                                                                          | Depends on |
| -------- | --------------------------------------------------------------------------- | ---------- |
| **T001** | 建立 encoder-level post-transform contract，并保证 metadata 正确表达 signed/rectified | —          |
| **T002** | 将 transform 暴露到 `encode_spikes.py`，再接入 `_common.bash`                       | T001       |
| **T003** | 让 `PIPELINE_MODE=continue` 能识别 transform 改变，禁止复用错误的旧 SpikeIMU               | T002       |
| **T004** | 集成验证、全量 pytest、durable docs、关闭 plan                                         | T001–T003  |

生命周期对每个 implementation task 都必须是：

```text
DRAFT
 ↓
PROBING
 ↓
FROZEN
 ↓
IMPLEMENTING
 ↓
VERIFYING
 ↓
DOCUMENTING
 ↓
DONE
```

Probe 是只读调查，Worker 只能修改 TaskSpec 授权路径，Verifier 必须独立检查 actual diff，不能因为 Worker 自己说测试过就算通过。

---

# 3. T001 — Encoder post-transform contract

这是第一步，也是整个 feature 最核心的一步。

### Probe 要确认

`luna_probe` 重点检查：

```text
src/writingring/spike_encoding/encoders/custom_wavelet.py
src/writingring/spike_encoding/runner.py
src/writingring/spike_encoding/publication.py

tests/test_custom_wavelet_settings.py
tests/test_custom_wavelet_encoder.py
tests/test_spike_encoding_publication.py

docs/notes/SPIKE_ENCODING.md
docs/notes/OCCURRENCE_ALIGNED_SPIKE_ENCODING.md
```

需要确认这些事实：

1. transform 可以在 occurrence alignment 以后执行，而不破坏 detector state。
2. `step()` 是否应该继续保持 signed behavior。
3. `run_spike_encoder()` 的 sequence statistics 是基于最终 returned values 计算，因此 AbsRectify 后 `negative_event_count == 0`。
4. `representation` 变化是否会影响任何 downstream consumer。
5. `spike_imu.schema = signed_wavelet_events_plus_imu_v1` 的硬编码消费者范围。
6. 是否有代码依赖 event channels 必须含负值。

当前 test suite 已经分别存在 Custom Wavelet settings、encoder 和 CLI 测试面。

### Frozen behavior

如果 Probe CONFIRMED，T001 冻结为：

```python
CustomWaveletSettings(
    post_encode_transform=None
)
```

允许：

```python
None
"AbsRectify"
```

拒绝：

```python
"abs"
"rectify"
"ABSRECTIFY"
"Binary"
True
1
```

即严格枚举，不默默猜测。

建议：

```python
post_encode_transform: str | None = None
```

validation：

```python
if self.post_encode_transform not in {None, "AbsRectify"}:
    raise CustomWaveletSettingsError(...)
```

### transform 执行位置

建议在：

```python
encoded = detected[
    2 * padding : 2 * padding + len(values)
].copy()
```

之后：

```python
encoded = _apply_post_encode_transform(
    encoded,
    self.settings.post_encode_transform,
)
```

而不是：

```python
response = np.abs(response)
```

也不是：

```python
events = np.abs(self._extrema.step(response))
```

原因是你的需求明确是：

> 已经 encode 完的 spike 再取绝对值。

这样 `_LocalExtremaDetector` 和 `step()` 可以继续作为 signed detector reference，不需要破坏当前 vendor-parity / extrema 单元测试。

### 建议抽 helper

不要直接散落：

```python
if transform == ...
```

建议：

```python
def _apply_post_encode_transform(
    values: np.ndarray,
    transform: str | None,
) -> np.ndarray:
    if transform is None:
        return values

    if transform == "AbsRectify":
        return np.abs(values)

    raise CustomWaveletSettingsError(...)
```

以后增加：

```text
Binary
Threshold
Clip
Normalize
```

时有一个唯一 extension point。

### representation

现在不能继续无条件使用：

```text
signed_sparse_wavelet_extrema
```

rectify 后应该例如：

```text
abs_rectified_sparse_wavelet_extrema
```

所以建议把 `representation` 改成由 setting 决定：

```python
@property
def representation(self) -> str:
    if self.settings.post_encode_transform == "AbsRectify":
        return "abs_rectified_sparse_wavelet_extrema"
    return "signed_sparse_wavelet_extrema"
```

这是必要的，因为 publication 当前是根据 representation 判断 polarity 是否保留；如果 values 已经全部非负，但 metadata 仍声称 signed，会形成 contract mismatch。

### encoding metadata

至少新增：

```json
{
  "post_encode_transform": "AbsRectify"
}
```

或：

```json
{
  "post_encode_transform": null
}
```

建议再明确：

```json
{
  "event_representation": "abs_rectified_sparse_wavelet_extrema"
}
```

### SpikeIMU schema

这里我建议 **本次不做全仓 schema migration**。

现在 downstream 已经把：

```text
signed_wavelet_events_plus_imu_v1
```

作为固定 SpikeIMU feature schema，改它会扩散到 recording loader、segmentation、padding、training。当前 pipeline 的消费者确实使用固定 SpikeIMU contract。

本次采用兼容方案：

```json
"spike_imu": {
    "schema": "signed_wavelet_events_plus_imu_v1",
    "event_representation": "abs_rectified_sparse_wavelet_extrema"
}
```

并在 docs 明确说明：

> `schema` 当前保留为 legacy 21-channel layout identifier；具体 event polarity 以 `event_representation` / encoder representation 为准。

这是一个需要 Probe 明确确认的 contract 决策。如果 Probe 发现已有 consumer 将 schema 名里的 `"signed"` 当作真实 polarity guarantee，而不仅是 identifier，则 T001 必须返回 `REVISE`，不能偷偷继续。

### T001 acceptance tests

最核心测试：

```python
signed = ...
rectified = ...

np.testing.assert_array_equal(
    rectified.values,
    np.abs(signed.values),
)

np.testing.assert_array_equal(
    rectified.values != 0,
    signed.values != 0,
)

assert np.all(rectified.values >= 0)
assert np.count_nonzero(rectified.values < 0) == 0
```

还必须验证：

```python
signed.channel_names == rectified.channel_names
signed.values.shape == rectified.values.shape
```

以及：

```python
signed_nonzero_count == rectified_nonzero_count
```

原来的 detector test：

```text
test_extrema_detector_preserves_signed_maxima_and_minima_once
```

**不应该修改期望值**。

Detector 本身仍应该产生 signed extrema。

---

# 4. T002 — CLI + `_common.bash`

T001 PASS 后再 Probe T002。

### 修改入口

允许写：

```text
scripts/encode_spikes.py
scripts/action0_pipeline/_common.bash

tests/test_encode_spikes_cli.py
tests/test_action0_pipeline_scripts.py
```

现在仓库已经有专门的 `test_encode_spikes_cli.py` 和 `_common.bash` smoke/structure test，所以不需要创造新的测试体系。

### `_common.bash`

在 `pipeline_init()` 与：

```bash
SAMPLING_RATE="${SAMPLING_RATE:-200}"
ENCODER="${ENCODER:-custom-wavelet}"
```

相邻的位置新增：

```bash
POST_ENCODE_TRANSFORM="${POST_ENCODE_TRANSFORM:-none}"
```

然后：

```bash
case "$POST_ENCODE_TRANSFORM" in
    none|AbsRectify) ;;
    *)
        pipeline_die \
            "POST_ENCODE_TRANSFORM must be none or AbsRectify"
        ;;
esac
```

默认必须是：

```text
none
```

这样现有行为完全不变。

### Action-0 encode command

当前 `_common.bash` 集中通过 `pipeline_encode()` 调 `scripts/encode_spikes.py`。

增加：

```bash
--post-encode-transform "$POST_ENCODE_TRANSFORM"
```

最终 log 应明确出现：

```text
--post-encode-transform none
```

或：

```text
--post-encode-transform AbsRectify
```

这对以后回溯实验结果非常重要。

### `encode_spikes.py`

新增：

```python
parser.add_argument(
    "--post-encode-transform",
    choices=("none", "AbsRectify"),
)
```

这里我建议 **default=None，而不是 `"none"`**。

区别是：

```text
CLI 没写这个参数
    → 不覆盖 encoder JSON/settings

CLI 明确传 none
    → 强制关闭任何 config 中定义的 transform

CLI 传 AbsRectify
    → 强制启用
```

也就是明确的 precedence：

```text
explicit CLI
    >
encoder settings file
    >
CustomWaveletSettings default
```

Action-0 pipeline 永远明确传：

```text
none
```

或：

```text
AbsRectify
```

所以对 Action-0 来说，**bash 是最终 authority**。

具体转换：

```python
if args.post_encode_transform is not None:
    settings["post_encode_transform"] = (
        None
        if args.post_encode_transform == "none"
        else args.post_encode_transform
    )
```

这样就算以后 JSON 里有人放：

```json
"post_encode_transform": "AbsRectify"
```

而 bash 是：

```bash
POST_ENCODE_TRANSFORM=none
```

Action-0 仍然按照 bash。

这正符合你希望“之后在 bash 里改”的使用模式。

---

# 5. T003 — continue 模式 stale artifact protection

这是我认为最不能省的一项。

如果今天运行：

```bash
POST_ENCODE_TRANSFORM=none
```

生成 signed：

```text
spikeIMU.npy
metadata.json
```

明天只改：

```bash
POST_ENCODE_TRANSFORM=AbsRectify
```

然后：

```bash
PIPELINE_MODE=continue
```

pipeline 绝不能说：

> 已有 SpikeIMU 看起来 shape 正确，因此 skip encode。

当前 `_common.bash` 已有 `pipeline_encode_outputs_valid()` / `pipeline_validate_spike_artifact()`，并会检查已有 SpikeIMU 和 metadata；这是正好应该扩展的位置。

### validation 参数

改：

```bash
pipeline_validate_spike_artifact \
    "$values_path" \
    "$metadata_path" \
    "$timestamps_path"
```

为：

```bash
pipeline_validate_spike_artifact \
    "$values_path" \
    "$metadata_path" \
    "$timestamps_path" \
    "$POST_ENCODE_TRANSFORM"
```

内部 Python：

```python
expected = sys.argv[4]

expected_transform = (
    None if expected == "none" else expected
)

settings = metadata.get("settings", {})
actual_transform = settings.get("post_encode_transform")
```

然后：

```python
if actual_transform != expected_transform:
    raise SystemExit(
        "SpikeIMU post_encode_transform mismatch: "
        f"expected={expected_transform!r}, "
        f"actual={actual_transform!r}"
    )
```

### 旧 artifact backward compatibility

这里建议一个很实用的规则：

旧 metadata 里如果完全没有：

```text
post_encode_transform
```

视作：

```python
None
```

因为旧 encoder 本来就是 signed/no-transform。

所以：

```text
旧 artifact + POST_ENCODE_TRANSFORM=none
```

可以继续复用。

而：

```text
旧 artifact + POST_ENCODE_TRANSFORM=AbsRectify
```

必须失效。

即：

| existing metadata |  requested | valid? |
| ----------------- | ---------: | -----: |
| key missing       |       none |    yes |
| null              |       none |    yes |
| AbsRectify        |       none |     no |
| missing/null      | AbsRectify |     no |
| AbsRectify        | AbsRectify |    yes |

这样不会因为 feature 上线就无条件逼所有人重新 encode。

### 不重写现有 continue policy

T003 **不负责重新设计**：

```text
pipeline_plan_continue()
```

到底从 preprocess、encode 还是其他 stage 重启。

它只负责：

> 当前 transform 和 artifact transform 不一致时，encode artifact validation 必须失败。

然后继续沿用现有 pipeline 的 invalid-output handling。

Probe 必须检查现有 stage-resume 行为，并把结果写进 TaskSpec，避免 Worker顺手重新设计整个 resume mechanism。

---

# 6. Bash 使用方式

完成后你最常用的方式就是：

```bash
POST_ENCODE_TRANSFORM="${POST_ENCODE_TRANSFORM:-none}"
```

想保持现在：

```bash
POST_ENCODE_TRANSFORM="${POST_ENCODE_TRANSFORM:-none}"
```

想默认 AbsRectify：

```bash
POST_ENCODE_TRANSFORM="${POST_ENCODE_TRANSFORM:-AbsRectify}"
```

或者完全不用编辑文件：

```bash
POST_ENCODE_TRANSFORM=AbsRectify \
PIPELINE_MODE=overwrite \
bash scripts/action0_pipeline/01_....sh
```

以后实验不同 representation 时，这个环境变量就是入口。

---

# 7. T004 — 集成验证

在 T001–T003 都通过独立 verifier 后，再做最终 integration validation。

重点验证矩阵：

| 场景                           | 期望                                |
| ---------------------------- | --------------------------------- |
| default/none                 | 与现有 signed behavior 一致            |
| AbsRectify                   | 所有 event >= 0                     |
| signed vs rectified          | 非零位置完全相同                          |
| signed vs rectified          | trailing `spikeIMU[:,15:21]` 完全相同 |
| signed vs rectified          | timestamps 完全相同                   |
| none metadata                | polarity preserved                |
| AbsRectify metadata          | polarity not preserved            |
| continue + same transform    | 可复用                               |
| continue + changed transform | encode artifact invalidated       |
| invalid transform            | bash/CLI fail fast                |
| CLI omitted                  | settings/default 生效               |
| CLI explicit                 | CLI override settings             |

特别要验证：

```python
np.testing.assert_array_equal(
    signed_spike_imu[:, 15:],
    rectified_spike_imu[:, 15:],
)
```

因为 alignment / Board transient logic 使用的是后 6 个 IMU channels，不应该被 rectification 影响。

---

# 8. 测试命令

AGENTS 有一个当前已知的环境矛盾：

`AGENTS.md` 要求 **Python 3.11 + `writingring-viz`**；但现有 Workboard 记录 `writingring-viz` 实际是 Python 3.10.20，同时用户此前授权了隔离的 `writingring-test` Python 3.11.15。

因此这个 plan 不应该假装矛盾不存在。建议验证分两层：

```bash
conda run --no-capture-output -n writingring-viz \
    python -m pytest \
    tests/test_custom_wavelet_settings.py \
    tests/test_custom_wavelet_encoder.py \
    tests/test_spike_encoding_publication.py \
    tests/test_encode_spikes_cli.py \
    tests/test_action0_pipeline_scripts.py
```

同时在已授权 Python 3.11 环境：

```bash
conda run --no-capture-output -n writingring-test \
    python -m pytest \
    tests/test_custom_wavelet_settings.py \
    tests/test_custom_wavelet_encoder.py \
    tests/test_spike_encoding_publication.py \
    tests/test_encode_spikes_cli.py \
    tests/test_action0_pipeline_scripts.py
```

最终：

```bash
conda run --no-capture-output -n writingring-viz \
    python -m pytest
```

以及：

```bash
conda run --no-capture-output -n writingring-test \
    python -m pytest
```

Worker 角色本身明确要求 code change 后运行 repository-required pytest，task-specific tests 不能替代 full pytest。

---

# 9. Durable documentation

Verifier PASS 后，由 PRIMARY—not Worker—更新文档。

建议更新：

```text
docs/notes/SPIKE_ENCODING.md
docs/notes/OCCURRENCE_ALIGNED_SPIKE_ENCODING.md
docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md

docs/plans/Done/03_Spike_Post_Encode_Transform_Plan.md
docs/plans/Done/03_Spike_Post_Encode_Transform_TASKS.md
docs/plans/WORKBOARD.md
```

`README.md` 我目前倾向于**不改**：这是 encoder 的实验性配置行为，不改变 repository-level workflow 或 entry-point 名字。

文档必须明确记录：

```text
none:
    preserve signed local-extrema amplitude

AbsRectify:
    apply absolute value after complete occurrence-aligned encoding

AbsRectify does NOT:
    alter extrema detection
    alter event occurrence rows
    alter event channel indices
    alter event density
    alter timestamps
    alter spikeIMU trailing IMU channels
```

---

# 10. Scope boundaries

这次明确 **不做**：

```text
不修改 vendor/**
不修改 data_sample/**
不改 wavelet / Prony 算法
不改 extrema detector
不改 extrema window
不改 occurrence alignment
不改 sampling rate behavior
不改 channel count
不改 channel ordering
不改 timestamps
不改 segmentation algorithm
不改 alignment transient channels
不改 Action0 network
不增加 binary/threshold transform
不做全仓 SpikeIMU schema-v2 migration
```

这能把 feature 控制在一个很清楚的范围内。

## 最后的 implementation 形态

我建议最终代码结构是：

```text
_common.bash
    POST_ENCODE_TRANSFORM=none|AbsRectify
              │
              ▼
scripts/encode_spikes.py
    --post-encode-transform
              │
              ▼
settings["post_encode_transform"]
              │
              ▼
CustomWaveletSettings
              │
              ▼
encode_sequence()
    wavelet
    extrema
    occurrence alignment
              │
              ▼
_apply_post_encode_transform()
    None        -> unchanged
    AbsRectify  -> np.abs(...)
              │
              ▼
run_spike_encoder()
    statistics on FINAL values
              │
              ▼
publication
    spikes.npy
    spikeIMU.npy
    metadata.json
              │
              ▼
_common.bash continue validation
    metadata transform == requested transform
```

## T004 execution record

The final same-input, producer-backed matrix passed on Python 3.11.15 in the
available `writingring-gpu` fallback: `AbsRectify` produced the absolute value
of signed events without changing the event mask, rows, channels, trailing
SpikeIMU values, timestamp provenance, or legacy layout schema. Metadata,
polarity flags, and final statistics reflected the chosen transform. The
focused suite passed 61 tests; the full suite passed 527 tests with one known
skip for an unavailable real SpikeIMU artifact. `writingring-viz` itself could
not be entered because Conda raised `NoWritableEnvsDirError`; this is recorded
as an environment-access limitation, distinct from implementation evidence.

## Completion

All T001--T004 are complete. A fresh independent verifier passed the final
producer-backed integration matrix, including CLI/Bash control and
continue-mode transform provenance. The plan preserves its required Python
3.11 evidence through the available `writingring-gpu` environment; the
repository-named `writingring-viz` environment remains unverified because
Conda cannot access a writable environment directory.
