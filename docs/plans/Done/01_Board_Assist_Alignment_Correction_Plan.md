# Plan: Board-Assisted Alignment Skippable Outcome Correction

## Active follow-up revision — Board-Assisted Alignment Outcome Follow-up Correction

This revision supersedes the completion claim for the initial T1/T2 work. The
active implementation DAG is:

```text
T1R Event-prefix correction
        |
        v
T2R Outcome-contract correction
      /   \
     v     v
T3 Action0  T4 Board segmentation
      \   /
       v v
T5 End-to-end verification and state cleanup
```

The final contract is that a usable Board pair, and every event derived for
alignment, must be wholly inside the initial monotonic prefix; `FAILED` is not
a completed transition; malformed outcome reports always fail through the
public alignment-outcome error family; and a `SKIPPED` artifact proves its
reason with self-consistent diagnostics. T1R and T2R must each repeat the
probe → freeze → implementation → fresh-verifier lifecycle before their
respective corrected behavior may be recorded as DONE.

The revision does not redesign alignment ranking, timestamp repair/sorting,
skip reasons, per-recording Action0 resume, padding, SNN, or notebooks.

## 1. 目标

当一个 Board recording 同时满足以下条件时：

```text
完整 Board 数据存在 valid press/lift pair
Board timestamp 存在 backward jump
initial monotonic interval 内不存在完整 valid press/lift pair
```

允许该 recording 产生正式的 `SKIPPED` alignment outcome，而不是终止整个 Action0 aligned-board pipeline。

最终 contract：

```text
SUCCESS
    → 正常参与 Board segmentation

SKIPPED
    → 不参与 Board segmentation，但计为合法完成 outcome

FAILED / MISSING / CONFLICT
    → hard fail
```

除上述精确场景外，现有 alignment、provenance、数据完整性和代码错误继续 hard fail。

---

## 2. Scope

### In scope

```text
src/writingring/event_alignment.py
src/writingring/alignment_io.py
src/writingring/__init__.py
scripts/align_ring_board.py
scripts/action0_pipeline/_common.bash
src/writingring/board_event_segmentation.py
相关 tests
相关 docs/notes/**
README.md（仅在需要同步用户可见 contract 时）
```

本次需要建立：

```text
精确的 Board initial-interval unusable 判定
结构化 skip diagnostics
SUCCESS / SKIPPED / FAILED alignment outcome contract
skip artifact + input provenance
统一 Python outcome reader / validator
安全的 success ↔ skip overwrite transition
Action0 continue / QA outcome validation
Board segmentation recording-level skip
recording-level skip summary
all-recordings-skipped hard failure
```

### Out of scope

不修改：

```text
alignment work axis
canonical timestamp 语义
SpikeIMU feature schema
peak detection
alignment coverage threshold
Board chunk ordering规则
label segmentation 算法
label-level missing_event_policy
padding 算法
```

不引入：

```text
except Exception: skip
timestamp 排序或修复
自动选择 backward jump 后的 epoch
per-recording incremental Action0 resume
```

---

## 3. Task DAG

所有 implementation task 均按：

```text
PROBE
→ FROZEN TaskSpec
→ WORKER
→ VERIFIER
→ DOCUMENT
→ DONE
```

执行。

```text
T1 Board interval semantics
        ↓
T2 Alignment outcome contract + CLI
        ↓
   ┌────┴────┐
   ↓         ↓
T3 Action0   T4 Board segmentation
outcome QA   outcome consumer
   └────┬────┘
        ↓
T5 End-to-end contract verification
```

---

## 4. Tasks

### T1 — Board interval semantics

**目标：** 精确定义并实现“完整 Board 有 valid pairs，但 initial monotonic interval 内没有完整 valid pair”的唯一可 skip 条件。

**验收标准：**

* usable pair 要求 `press` 和 `lift` 都完整位于第一次 backward jump 之前。
* jump 后 frame 不得通过 timestamp overlap 泄漏回 initial interval。
* 可 skip 异常携带 backward-jump、frame boundary、total/usable pair count 等结构化 diagnostics。
* Board 全局无 valid pair、无 backward jump 等情况继续使用现有 hard-failure semantics。
* targeted regression tests PASS。

---

### T2 — Alignment outcome contract + CLI

**目标：** 建立统一、可验证、带 provenance 的 `SUCCESS / SKIPPED / FAILED` alignment outcome，并让 alignment CLI 只对 T1 定义的异常支持显式 skip policy。

**验收标准：**

* `alignment_io.py` 成为 offset、skip、report outcome validation 的唯一 Python contract 层。
* `SKIPPED` artifact 包含 recording identity、稳定 reason、结构化 diagnostics 和当前 SpikeIMU/Board input provenance。
* 每个 completed recording 必须严格满足 `SUCCESS XOR SKIPPED`；FAILED report 不计 completed outcome。
* report 使用统一的 `alignment_status = success|failed|skipped`。
* standalone CLI 默认仍为 `error`；仅显式 `skip` policy 可以产生 SKIPPED。
* success ↔ skip transition 清理 stale artifacts，并遵守 overwrite authorization。
* completion artifact 发布不会把明显的 partial write 当成合法完成状态。
* public API 如有新增同步更新 package exports。
* targeted IO / CLI / overwrite tests PASS。

---

### T3 — Action0 outcome validation and QA

**目标：** 让 Action0 pipeline 按 authoritative recording list 验证 alignment outcome，而不是继续依赖 offset 文件数量或 Bash 自行解析 artifact。

**验收标准：**

* `_common.bash` 不自行解析 offset/skip schema，而是调用统一 Python validator。
* Action0 alignment 显式启用新的 skip policy。
* 每个 recording 必须对应且只对应一个 provenance-valid `SUCCESS` 或 `SKIPPED` outcome。
* missing、malformed、stale provenance、success+skip conflict 均判定为 invalid stage。
* `continue` 保持现有 stage-level semantics：完整 stage 复用，partial/invalid stage 走现有 rebuild path，不新增 per-recording resume。
* QA 输出 success、skipped、total outcome accounting，并验证 recording identity，而非仅比较文件数量。
* targeted continue / overwrite / QA tests PASS。

---

### T4 — Board segmentation outcome consumer

**目标：** 让 aligned-board segmentation 正确消费 alignment outcome：SUCCESS 正常处理，SKIPPED 验证后排除，其他状态 hard fail。

**验收标准：**

* SKIPPED recording 在进入 segmentation 前验证 identity、reason 和当前 input provenance。
* SUCCESS recording 保持现有 offset/provenance/segmentation contract。
* skipped recording 不进入 segment arrays 或 aggregate artifacts。
* aggregate consistency checks 只针对实际 processed recordings。
* summary 明确区分 source、processed、recording-level skipped 和现有 segment-level skipped。
* 所有 recordings 都被 skip 时，在 aggregate/padding 前明确 hard fail。
* targeted mixed-success/skip、stale skip、conflict、all-skipped tests PASS。

---

### T5 — End-to-end contract verification

**目标：** 验证新的 alignment outcome contract 在 alignment → Action0 continue/overwrite → Board segmentation → QA 全链路一致，并同步 durable documentation。

**验收标准：**

* mixed SUCCESS/SKIPPED Action0 pipeline 可以完成 alignment 和 segmentation，padding 无需感知 recording-level skip。
* 非允许 skip 场景仍然 hard fail。
* stale provenance、artifact conflict、partial outcome 均无法通过 continue/QA。
* relevant pytest suite 在项目规定的 Python 3.11 环境 PASS。
* `ALIGNMENT_OUTPUTS.md`、Board segmentation、Action0 Bash/QA 等受影响的 `docs/notes/**` 与最终实现一致。
* README 仅在用户可见 workflow/contract 改变时更新。
* fresh verifier 对最终 DAG contract 给出 PASS。
