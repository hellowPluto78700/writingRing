## 目标

当 Board recording 满足下面这种情况：

```text
完整 Board 数据中存在 valid press/lift pairs
+
Board timestamp 存在 backward jump
+
当前策略保留的 initial monotonic interval 中没有 valid pair
```

不要终止整个 aligned-board pipeline，而是：

```text
标记该 recording 为 skipped
→ 继续 alignment 后续 recording
→ Board segmentation 忽略该 recording
→ padding / QA 正常继续
```

其他错误继续 hard fail，包括：

```text
Board 根本没有有效 touch
SpikeIMU 缺失
provenance/hash 不一致
alignment coverage 不足
代码异常
文件损坏
```

不能使用 `except Exception: skip`。

---

## 1. `event_alignment.py`：定义一个精确的可跳过异常

文件：

```text
src/writingring/event_alignment.py
```

当前 `select_board_interval_from_presses()` 遇到第一次 timestamp backward jump 后，只使用它之前的 initial nondecreasing interval。当前这套逻辑正是这次 recording 被截到 global frame 5 的原因。

新增专门异常，例如：

```python
class UnusableInitialBoardTimestampIntervalError(EventAlignmentError):
    """Valid Board touches exist, but none lie in the initial monotonic timestamp interval."""
```

修改 interval selection：

```python
all_valid_pairs = detection.touch_pairs.loc[
    detection.touch_pairs["valid_touch"].astype(bool)
].copy()

usable_valid_pairs = all_valid_pairs.loc[
    all_valid_pairs["press_global_frame_index"] <= maximum_global_frame
].copy()
```

然后区分两种情况。

### 情况 A：完整 Board 本身就没有 valid pair

继续作为普通错误：

```python
if all_valid_pairs.empty:
    raise EventAlignmentError(
        "no valid Board press/lift pair is available for interval selection"
    )
```

这种不要自动 skip。

### 情况 B：完整 Board 有 valid pair，但 initial timestamp interval 一个都没有

只有这种抛新的可跳过异常：

```python
if usable_valid_pairs.empty and len(backward_positions):
    raise UnusableInitialBoardTimestampIntervalError(
        "valid Board press/lift pairs exist, but none are available "
        "before the first Board timestamp backward jump"
    )
```

你当前的 `user_3/action_0/dataset_2` 就属于这个分支：总共有 30 个 valid pairs，但 initial usable interval 只有 global frame 0–5，因此 usable pair 为 0。

不要修改 timestamp，不排序 Board chunk，也不要自动选择 backward jump 后的 epoch。

---

## 2. `alignment_io.py`：建立正式的 skip artifact

文件：

```text
src/writingring/alignment_io.py
```

这里已经集中维护 alignment artifact/domain 相关逻辑，因此 skip artifact 的路径和读取/校验也放这里比较合适。当前 work axis 已经统一为 canonical endpoints 重建，与这次 skip 修复无关，不要改。

新增：

```python
build_alignment_skip_path(...)
write_alignment_skip_json(...)
read_alignment_skip_json(...)
```

路径建议：

```text
alignment/offsets/
└── user_3/
    └── action_0/
        └── 2_ring_board_skip.json
```

schema：

```json
{
  "schema_version": 1,
  "status": "skipped",
  "user": "user_3",
  "action": "0",
  "dataset_id": 2,
  "reason": "unusable_initial_board_timestamp_interval",
  "message": "valid Board press/lift pairs exist, but none are available before the first Board timestamp backward jump",
  "board_timestamp_backward_jump_count": 1,
  "first_backward_global_frame_index": 6,
  "total_valid_touch_pair_count": 30,
  "usable_valid_touch_pair_count": 0
}
```

读取 skip marker 时必须校验：

```text
schema_version
status == skipped
user/action/dataset_id
reason
```

不能只看文件是否存在。

---

## 3. `align_ring_board.py`：增加 skip policy

文件：

```text
scripts/align_ring_board.py
```

当前这个 CLI 把 `EventAlignmentError` 等统一捕获后返回 exit code 2，因此任何 interval-selection error 都会终止 Bash pipeline。

增加参数：

```text
--unusable-board-policy error|skip
```

默认：

```text
error
```

保持 standalone CLI 的严格行为。

Action0 Board pipeline 显式传：

```text
--unusable-board-policy skip
```

只捕获：

```python
UnusableInitialBoardTimestampIntervalError
```

流程：

```text
select_board_interval_from_presses()
        │
        ├── 正常
        │     ↓
        │   原 alignment 流程
        │
        └── UnusableInitialBoardTimestampIntervalError
              ↓
            policy=error → 原样 exit 2
              ↓
            policy=skip
              ↓
            写 skip.json
            写 minimal alignment report
            删除 stale success artifacts
            print skipped
            return 0
```

不要捕获普通 `EventAlignmentError` 做 skip。

---

## 4. Skip 时仍然写 alignment report

即使 skip，也应该保留：

```text
alignment/reports/user_3/action_0/2_alignment_report.json
```

内容建议：

```json
{
  "recording": {
    "user": "user_3",
    "action": "0",
    "dataset_id": 2
  },
  "alignment_status": "skipped",
  "alignment_success": false,
  "skip_reason": "unusable_initial_board_timestamp_interval",
  "offset_written": false,
  "verification_written": false,
  "board_diagnostics": {
    "timestamp_backward_jump_count": 1,
    "first_backward_global_frame_index": 6,
    "total_valid_touch_pair_count": 30,
    "usable_valid_touch_pair_count": 0
  }
}
```

这样以后能明确区分：

```text
alignment 真失败
vs
pipeline 主动 skip
```

---

## 5. 明确定义 success / skip artifact contract

一个 recording 只能处于下面两种合法状态之一。

成功：

```text
offset.txt          exists
report.json         exists, status=success
verification.png    exists
skip.json           不存在
```

跳过：

```text
skip.json           exists
report.json         exists, status=skipped
offset.txt          不存在
verification.png    不存在
```

禁止：

```text
offset.txt + skip.json 同时存在
```

因为这会让下游不知道该采用哪个结果。

---

## 6. Overwrite 时清除 stale artifact

这个必须处理。

假设 dataset 2 上一次成功，有：

```text
2_ring_board_offset.txt
2_alignment_verification.png
```

这次重新运行发现应该 skip。

写 skip marker 前必须删除旧：

```text
offset
verification
旧 success report
```

然后写新的 skip report。

反过来，如果之前是 skip，现在重新执行成功：

```text
删除 skip.json
写 offset
写 verification
写 success report
```

始终保持：

```text
success XOR skipped
```

---

## 7. `_common.bash`：Board alignment 开启 skip policy

你本地 `_common.bash` 已经有新的 `continue / overwrite` 逻辑，因此实现时要基于当前 checkout 修改，不要拿 GitHub 旧版覆盖。

Board alignment command 增加：

```bash
--unusable-board-policy skip
```

即：

```bash
scripts/align_ring_board.py \
    ... \
    --unusable-board-policy skip
```

---

## 8. `_common.bash`：alignment artifact check 改成 outcome check

不能再要求每个 recording 都必须有：

```text
*_ring_board_offset.txt
```

改成：

```text
valid success artifact set
OR
valid skip artifact set
```

伪逻辑：

```bash
if valid_alignment_success; then
    completed
elif valid_alignment_skip; then
    completed
else
    invalid_or_incomplete
fi
```

在 `continue` mode：

```text
valid success → 不重跑
valid skip    → 不重跑
无结果        → 从 alignment 继续
结果冲突/损坏 → validation failure，按现有 continue 策略触发 rebuild
```

因此这个 dataset 下次运行：

```bash
PIPELINE_MODE=continue
```

不会再次尝试 alignment。

---

## 9. Board segmentation：遇到 skip recording 直接 `continue`

文件：

```text
src/writingring/board_event_segmentation.py
```

当前 `_build_aligned_recording_artifacts()` 对每个 recording 强制要求 offset；offset 不存在就直接抛错。

修改为：

```text
for recording:

    如果存在合法 skip marker:
        validate marker identity
        record skipped metadata
        continue

    如果存在合法 offset:
        正常 Board segmentation

    两者都没有:
        hard fail

    两者同时存在:
        hard fail
```

所以最终：

```text
dataset 0 → segment
dataset 1 → segment
dataset 2 → skip
dataset 3 → segment
```

dataset 2 不进入 aggregate arrays。

---

## 10. Segmentation summary 记录 recording-level skip

建议增加：

```json
{
  "source_recording_count": 4,
  "processed_recording_count": 3,
  "skipped_recording_count": 1,
  "skipped_recordings": [
    {
      "dataset_id": 2,
      "reason": "unusable_initial_board_timestamp_interval"
    }
  ]
}
```

不要把这个和现有的：

```text
label interval skipped
```

混为一谈。

两者语义不同：

```text
recording skip
    = 整个 dataset 不参与 Board segmentation

segment skip
    = recording 可用，但某个 label interval 没有完整 touch
```

当前 Board segmentation 已经支持 label-level `missing_event_policy=skip`，这个功能保持不变。

---

## 11. 如果一个 user 的所有 recording 都被 skip

不要生成看似正常的空 dataset。

例如：

```text
user_7:
dataset 0 skipped
dataset 1 skipped
dataset 2 skipped
```

应该明确 hard fail：

```text
all aligned-board recordings were skipped for user_7/action_0
```

避免后面 aggregate 使用空 artifact 集合，也避免 padding 出现没有实际数据但 pipeline 显示成功的情况。

---

## 12. `_common.bash` QA 改成 outcome accounting

旧逻辑概念上是：

```text
alignment_offset_count == ring_recording_count
```

新逻辑：

```text
alignment_success_count
+
alignment_skip_count
==
ring_recording_count
```

并写入 QA log：

```text
alignment_successful_recordings=...
alignment_skipped_recordings=...
alignment_total_outcomes=...
```

另外验证：

```text
每个 recording 恰好只有一个 outcome
```

不能只比较总文件数量，否则一个 recording 同时有 success/skip，而另一个什么都没有，也可能错误地数量相等。

---

## 13. Padding 不需要增加特殊逻辑

Padding 只消费最终 segmentation 输出。

因为 skipped recording 根本不会进入 Board segmentation aggregate：

```text
alignment skip
→ segmentation 排除 recording
→ padding 自然只看到成功 recording 的 segments
```

因此不应该让 padding 感知 alignment skip。

---

## 14. Continue / overwrite 的预期行为

### 默认 continue

第一次：

```text
dataset 2
→ 检测异常
→ skip marker
→ 继续 dataset 3
```

第二次运行：

```text
dataset 2 skip marker 校验通过
→ alignment completed
→ 不重新尝试
```

### overwrite

```bash
PIPELINE_MODE=overwrite ...
```

不读取 skip marker 作为完成证据：

```text
从 preprocess 全部重新执行
→ dataset 2 alignment 再次实际检查
→ 如果仍然异常，再重新生成 skip marker
```

符合当前 overwrite “无条件重新计算”的定义。

---

## 15. Tests

至少增加以下 regression tests：

```text
event_alignment
- overall valid pairs > 0
- first backward jump 前 valid pairs = 0
- 抛 UnusableInitialBoardTimestampIntervalError

event_alignment
- 整个 recording valid pairs = 0
- 仍抛普通 EventAlignmentError
- 不能被 skip policy 吞掉

align CLI
- policy=error → exit 2
- policy=skip → exit 0 + skip.json + skipped report

align CLI overwrite
- old offset → new skip：旧 offset/verification 被删除
- old skip → new success：旧 skip marker 被删除

board segmentation
- offset recording 正常进入 aggregate
- skip recording 不进入 aggregate
- missing offset + missing skip → hard fail
- offset + skip 同时存在 → hard fail
- 所有 recording skipped → hard fail

pipeline continue
- valid skip marker 被视为 alignment completed
- 不重复执行该 recording

pipeline overwrite
- 不因为已有 skip marker而跳过

QA
- success + skipped == discovered recordings
```

---

## 16. 明确不修改的东西

本次不要修改：

```text
alignment work axis
canonical timestamps
SpikeIMU
peak detection
alignment coverage threshold
minimum_duration_frames
label-based segmentation
Board label interval missing-event policy
padding algorithm
```

当前 work axis 已经采用 representation-invariant endpoint reconstruction，这和此次 Board skip 属于两个独立问题。

---

## 最终目标行为

针对当前：

```text
user_3/action_0/dataset_2
```

运行结果应该类似：

```text
Alignment skipped:
user_3/action_0/dataset_2
reason=unusable_initial_board_timestamp_interval
valid_pairs=30
usable_pairs_before_backward_jump=0

Skip marker:
.../2_ring_board_skip.json

Alignment report:
.../2_alignment_report.json

Continuing with next recording...
```

然后 pipeline 不退出，继续处理后面的 dataset。

最终 QA 能明确显示：

```text
successful alignment recordings: N
skipped alignment recordings: 1
```

而不是把这个 recording 当成成功 alignment，也不是把它静默丢掉。
