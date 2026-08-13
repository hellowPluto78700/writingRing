
# Board-Assisted Segmentation Follow-up Plan

## 目标

把新的 Board-assisted segmentation 规则完整接入 CLI、Action0 pipeline、测试和文档，并验证：

* Board label gap 可以 >5s；
* 所有最终 Board-assisted segment 必须 `<= 5s`；
* carry-in press ownership/start 规则正确；
* empty-leading-Board 首段可以使用 `(label, first Board frame)` 内的 alignment-grade strong transient peak 恢复 start；
* label-only segmentation 原有行为不受影响。

---

## Scope

**修改范围**

```text
scripts/segment_ring_imu.py
scripts/bash_script/action0_pipeline/_common.bash

tests/test_board_event_segmentation.py
tests/test_segment_ring_imu_cli.py
tests/test_action0_pipeline_scripts.py
tests/test_segmentation.py
tests/test_spike_imu_segmentation.py

docs/notes/BOARD_EVENT_GUIDED_SEGMENTATION.md
docs/notes/IMU_SEGMENTATION.md
docs/notes/SPIKE_SEGMENTATION_PIPELINE.md
docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md
```

**默认不改**

```text
src/writingring/board_event_segmentation.py   # 已完成，除非测试发现 bug
src/writingring/event_alignment.py
src/writingring/board_loader.py
src/writingring/alignment_io.py
scripts/align_ring_board.py
```

当前 CLI 负责构造 `BoardEventSegmentationConfig`，Action0 `_common.bash` 又显式固定 Board segmentation 参数，因此这两个调用层需要与新 contract 对齐。

---

## DAG

```text
T0 Core Board segmentation replacement [DONE]
                |
        +-------+-------+
        |               |
        v               v
T1 CLI / config      T2 Core regression tests
integration
        |               |
        v               |
T3 Action0 wrapper     |
integration             |
        +-------+-------+
                |
                v
        T4 Documentation
                |
                v
        T5 Final verification
```

---

## Tasks

### T1 — CLI / config integration

**目标：** 让 `scripts/segment_ring_imu.py` 明确传递新的 Board-assisted 参数和默认 contract，同时保持 label mode 不变。

**验收标准：**

* Board mode CLI 能构造新的 config；
* 默认 final segment limit = `5s`；
* carry-in lookback = `0.2s`；
* pre/post context 仍为 `0.2s`；
* label mode CLI 行为无变化。

---

### T2 — Segmentation regression tests

**目标：** 用 focused tests 覆盖三个新规则及所有 Board segment `<=5s` invariant。

**验收标准：**

* `label gap >5s + final segment <=5s` → export；
* `final segment >5s` → skip；
* carry-in ownership、正常 start、midpoint start 均有测试；
* label 前最近 event 为 lift / press 超过 0.2s → 不触发 carry-in；
* empty `board_0` 首段只搜索 `(label, first Board frame)`；
* 多峰时选择最大 prominence 的 alignment-grade peak；
* 无合格 strong peak → 不 fallback；
* 所有导出的 Board-assisted samples 均断言 duration `<=5s`；
* label-only `>5s` 原测试继续通过。

当前文档仍明确写着 Board mode 继承最大 5s **label validity** 和 press-timestamp ownership，因此测试需要锁定新的替代语义。

---

### T3 — Action0 integration

**目标：** 让 Action0 aligned-board wrapper 显式运行新的 segmentation contract。

**验收标准：**

* `_common.bash` 的 aligned-board command 显式传新参数；
* raw/low-pass/Madgwick/Xylo aligned-board wrappers 共用同一规则；
* label wrappers 不受影响；
* Action0 script tests 通过。

当前 Action0 的 segmentation command 已集中在 `_common.bash`，所以只需要改共享入口，不要逐个修改 4 个 aligned wrapper。

---

### T4 — Documentation contract update

**目标：** 把 docs 从旧的“label gap 最大 5s + press label ownership”更新成新的实际 Board-assisted contract。

**验收标准：**

* `BOARD_EVENT_GUIDED_SEGMENTATION.md` 写清三个新规则；
* `IMU_SEGMENTATION.md` 明确 `label gap >5s invalid` 只属于 label mode；
* `SPIKE_SEGMENTATION_PIPELINE.md` 写明首段 recovery 使用 SpikeIMU transient channels `15:21` 和 alignment-grade peak detector；
* `SEGMENTATIONS_BASH_SCRIPTS.md` 与 Action0 实际参数一致；
* 不修改 alignment offset/time-domain contract。

SpikeIMU 文档当前已经规定 transient scoring 使用尾部 IMU 通道而不是 `0:15` spike channels，所以首段 recovery 应继续沿用这个 contract。

---

### T5 — Final verification

**目标：** 验证新 segmentation contract 在 recording API、CLI 和 Action0 pipeline 三层一致，没有破坏原有 label/alignment 行为。

**验收标准：**

* focused Board/CLI/Action0/SpikeIMU tests 全部通过；
* full pytest 通过；
* `git diff --check` 通过；
* 所有 Board-assisted export 满足 `final_duration <= 5s`；
* label-only segmentation 仍保留原来的 5s label-interval rule；
* alignment 与 Board loader 测试无 regression；
* docs 与最终实现一致。

这样整个任务实际上就是 **4 个实现/验证 task + 1 个 final verifier**，规模和 repo 现在的 agent TaskSpec 工作流比较一致，不需要再拆得更细。
