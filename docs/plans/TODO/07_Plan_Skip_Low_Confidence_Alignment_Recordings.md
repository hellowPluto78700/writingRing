# Plan: Skip Low-Confidence Alignment Recordings

## 目标

修复这类 recording：

```text
Board / SpikeIMU 数据合法
valid touch pairs 足够
alignment 能正常跑完
但 event coverage 低于阈值
```

当前会：

```text
FAILED
→ Action0 整体中断
```

修改后应：

```text
standalone 默认：
FAILED / non-zero

Action0 明确允许 skip：
SKIPPED(insufficient_event_coverage)
→ 继续处理其它 recording
```

**不降低现有 alignment 阈值。**

---

## 1. `src/writingring/event_alignment.py`

### 修改

把 alignment confidence failure 从单纯：

```python
result.success = False
warnings = [...]
```

改成同时输出结构化 failure information。

在 alignment report 中加入类似：

```json
{
  "confidence_checks": {
    "minimum_valid_touch_pairs": {
      "passed": true,
      "actual": 22,
      "minimum": 5
    },
    "minimum_event_coverage_ratio": {
      "passed": false,
      "actual": 0.272727,
      "minimum": 0.4
    }
  },
  "failed_confidence_checks": [
    "minimum_event_coverage_ratio"
  ]
}
```

保留现有：

```text
minimum_valid_touch_pairs = 5
minimum_event_coverage_ratio = 0.40
```

不修改 matching、candidate ranking、peak detection。

### 目的

让 CLI 可以可靠区分：

```text
valid pairs 不足
coverage 不足
真正的 alignment error
```

不能再通过解析 warning 字符串判断。

### 验收

对于：

```text
22 valid pairs
44 valid events
12 matched events
coverage = 0.2727
```

必须得到：

```text
success = false
failed_confidence_checks =
["minimum_event_coverage_ratio"]
```

---

# 2. `src/writingring/alignment_io.py`

### 修改

扩展合法 SKIPPED reason。

现有：

```text
initial_interval_no_usable_pair
```

新增：

```text
insufficient_valid_touch_pairs
insufficient_event_coverage
```

同时给两个 reason 增加 semantic validation。

### `insufficient_valid_touch_pairs`

skip artifact 至少保存：

```text
total_valid_touch_pair_count
minimum_valid_touch_pairs
matched_event_count
total_valid_event_count
event_coverage_ratio
minimum_event_coverage_ratio
failed_confidence_checks
```

validator 必须检查：

```text
0 < total_valid_touch_pair_count
    < minimum_valid_touch_pairs
```

---

### `insufficient_event_coverage`

skip artifact 至少保存：

```text
matched_event_count
total_valid_event_count
event_coverage_ratio
minimum_event_coverage_ratio

matched_press_count
total_valid_press_count
press_coverage_ratio

matched_lift_count
total_valid_lift_count
lift_coverage_ratio

fully_matched_touch_pair_count
total_valid_touch_pair_count

best_offset_us
best_vs_second_best_nearly_tied
failed_confidence_checks
```

validator 必须检查：

```text
total_valid_touch_pair_count
    >= minimum_valid_touch_pairs
```

以及：

```text
event_coverage_ratio
    < minimum_event_coverage_ratio
```

以及：

```text
event_coverage_ratio
≈ matched_event_count / total_valid_event_count
```

### 目的

确保：

```text
SKIPPED
```

不是靠一个 reason 字符串伪造出来的，而是可以由 diagnostics 证明。

### 验收

伪造以下 artifact 必须报错：

```text
reason = insufficient_event_coverage
event_coverage_ratio = 0.50
minimum = 0.40
```

或者：

```text
matched_event_count = 40
total_valid_event_count = 44
event_coverage_ratio = 0.27
```

---

# 3. `scripts/align_ring_board.py`

### 修改

增加一个通用 CLI 参数：

```text
--unalignable-recording-policy error|skip
```

默认：

```text
error
```

处理 alignment result 时按以下规则分类。

### Case 1

```text
InitialIntervalNoUsablePairError
```

如果 policy 允许 skip：

```text
SKIPPED
reason = initial_interval_no_usable_pair
```

---

### Case 2

```text
failed_confidence_checks
包含 minimum_valid_touch_pairs
```

如果 policy 为 `skip`：

```text
SKIPPED
reason = insufficient_valid_touch_pairs
```

---

### Case 3

```text
failed_confidence_checks
仅包含 minimum_event_coverage_ratio
```

如果 policy 为 `skip`：

```text
SKIPPED
reason = insufficient_event_coverage
```

---

### Case 4

其它 exception、malformed input、provenance failure、unexpected failure：

```text
FAILED
```

不得转成 SKIPPED。

---

### 重要

禁止写：

```python
if not result.success:
    publish_skipped(...)
```

必须根据 structured failure reason 明确分类。

SKIPPED 时：

```text
可以保存 best_offset_us 到 diagnostics
```

但：

```text
不能创建 offset TXT
不能创建 verification PNG
```

### 目的

让“alignment evidence 不够”变成可跳过 recording，而不是把整个 Action0 打断。

同时保证 standalone 默认仍然严格。

---

# 4. `scripts/action0_pipeline/_common.bash`

### 修改

Action0 调用 `align_ring_board.py` 时增加：

```bash
--unalignable-recording-policy skip
```

现有：

```text
--initial-interval-policy skip
```

如果仍需兼容可以暂时保留，但最终控制 recording-level unalignable behavior 的应该是新参数。

### 目的

使 Action0 自动接受：

```text
initial_interval_no_usable_pair
insufficient_valid_touch_pairs
insufficient_event_coverage
```

这些合法 SKIPPED outcomes。

而 standalone 用户不传该参数时仍然 hard fail。

### 验收

当前：

```text
user_7 / action 0 / dataset 1
```

不再导致：

```text
conda run ... failed
```

而应输出类似：

```text
Alignment skipped: insufficient_event_coverage
```

然后继续其它 dataset。

---

# 5. `src/writingring/board_event_segmentation.py`

### 修改

不要新增 alignment classification 逻辑。

只确认现有：

```text
validate_alignment_outcome()
```

能够接受新增的两个 SKIPPED reason。

行为保持：

```text
SUCCESS
→ segmentation

SKIPPED
→ 验证 provenance + skip diagnostics
→ omit recording

FAILED / missing / malformed / stale
→ hard fail
```

### 目的

让新的 SKIPPED recording 能自然进入已经实现好的 downstream outcome contract。

### 验收

```text
dataset 0 = SUCCESS
dataset 1 = SKIPPED(insufficient_event_coverage)
```

最终：

```text
processed_recording_count = 1
skipped_recording_count = 1
```

dataset 1 不进入 segmentation。

---

# 6. `tests/test_event_alignment.py`

### 新增测试

## Test A — low coverage structured failure

构造：

```text
valid pairs >= 5
event coverage < 0.40
```

断言：

```text
success == False
failed_confidence_checks ==
["minimum_event_coverage_ratio"]
```

---

## Test B — insufficient pairs

构造：

```text
valid pairs = 3
minimum = 5
```

断言：

```text
failed_confidence_checks
包含 minimum_valid_touch_pairs
```

---

## Test C — boundary

构造：

```text
event coverage == 0.40
valid pairs >= 5
```

必须：

```text
SUCCESS
```

### 目的

锁死 confidence classification，避免未来阈值或比较符号 regression。

---

# 7. `tests/test_alignment_io.py`

### 新增测试

覆盖：

```text
insufficient_valid_touch_pairs
insufficient_event_coverage
```

合法 artifact 必须通过。

以下必须失败：

```text
coverage >= minimum 却声称 insufficient_event_coverage
```

```text
event count / total count 与 coverage 对不上
```

```text
pair count >= minimum 却声称 insufficient_valid_touch_pairs
```

```text
unknown skip reason
```

### 目的

保证新的 SKIPPED outcome 有严格 semantic validation。

---

# 8. `tests/test_align_ring_board.py`

或当前实际负责 `align_ring_board.py` 的 CLI test 文件

### 新增测试

## standalone default

low coverage：

```text
--unalignable-recording-policy error
```

预期：

```text
FAILED
non-zero
no offset
no verification
report exists
```

---

## explicit skip

low coverage：

```text
--unalignable-recording-policy skip
```

预期：

```text
exit 0
SKIPPED
reason = insufficient_event_coverage
no offset
no verification
```

---

## insufficient pairs

```text
1–4 valid pairs
policy = skip
```

预期：

```text
SKIPPED
reason = insufficient_valid_touch_pairs
```

---

## unexpected failure

模拟其它 alignment exception。

即使 policy 为：

```text
skip
```

仍必须：

```text
FAILED / non-zero
```

### 目的

证明 skip policy 只吞掉明确允许的 confidence insufficiency，不吞真正 bug。

---

# 9. `tests/test_action0_pipeline_scripts.py`

### 新增测试

确认 Action0 command 包含：

```text
--unalignable-recording-policy skip
```

增加 mixed outcome 场景：

```text
dataset 0 = SUCCESS
dataset 1 = SKIPPED(insufficient_event_coverage)
```

预期：

```text
alignment stage valid
pipeline continues to segmentation
```

### 目的

防止以后 Action0 忘记传 skip policy。

---

# 10. `tests/test_board_event_segmentation.py`

### 新增测试

构造：

```text
dataset 0:
SUCCESS

dataset 1:
SKIPPED(insufficient_event_coverage)
```

断言：

```text
dataset 0 processed
dataset 1 omitted
processed_recording_count = 1
skipped_recording_count = 1
```

再增加：

```text
all recordings = SKIPPED
```

仍必须：

```text
hard fail before aggregation
```

### 目的

确认新增 skip reason 不破坏现有 downstream contract。

---

# 11. Docs

修改：

```text
docs/notes/ALIGNMENT_OUTPUTS.md
```

明确记录三个 SKIPPED reason：

```text
initial_interval_no_usable_pair
insufficient_valid_touch_pairs
insufficient_event_coverage
```

并说明：

```text
SKIPPED 不产生 offset / verification
```

---

修改：

```text
docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md
```

写清：

```text
standalone default = strict error
Action0 = explicit unalignable-recording-policy skip
```

---

新增：

```text
docs/plans/TODO/03_Alignment_Unalignable_Recording_Skip_Plan.md
docs/plans/TODO/03_Alignment_Unalignable_Recording_Skip_TASKS.md
```

更新：

```text
docs/plans/WORKBOARD.md
```

---

# 12. 最终真实数据验证

修复完成后重新运行：

```text
user_7
action 0
dataset 1
```

当前已知数据：

```text
matched_event_count = 12
total_valid_event_count = 44
event_coverage_ratio = 0.272727

total_valid_touch_pair_count = 22
fully_matched_touch_pair_count = 0
```

预期 outcome：

```text
SKIPPED
reason = insufficient_event_coverage
```

必须保留 diagnostics：

```text
12 / 44
coverage ≈ 0.272727
minimum coverage = 0.40
valid pairs = 22
fully matched pairs = 0
```

并确认：

```text
offset TXT 不存在
verification PNG 不存在
report 存在
skip JSON 存在
```

然后跑完整 Action0。

如果还有其它 SUCCESS recordings：

```text
alignment stage PASS
segmentation 只处理 SUCCESS
padding PASS
final QA PASS
```

---

# 13. 完成标准

必须全部通过：

```text
focused tests PASS
full pytest PASS
bash -n PASS
git diff --check PASS
user_7/action0/dataset1 real-data PASS
fresh luna_verifier PASS
```

最终行为必须是：

```text
可信 alignment
→ SUCCESS

明确的 evidence insufficiency
→ Action0 SKIPPED
→ standalone 默认 error

malformed / stale / corrupt / unexpected failure
→ hard fail

SKIPPED
→ 永远没有可消费 offset
```
