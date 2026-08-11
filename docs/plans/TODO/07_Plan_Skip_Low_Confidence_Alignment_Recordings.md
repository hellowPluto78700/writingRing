## 目标

在**不修改现有 Board-assisted segmentation 算法、alignment 判定和 padding 算法**的前提下，引入 recording-level fault isolation：某个 recording 在 Board-assisted segmentation 中发生可归因于该 recording 的错误时，记录错误并跳过该 recording，继续处理同 user 的后续 recording 和后续 users；所有成功 recording 正常进入后续 segmentation/padding，全部 user 处理结束后输出统一 error report。

## Scope

**In scope**

* `src/writingring/board_event_segmentation.py`
* `scripts/segment_ring_imu.py`
* `scripts/action0_pipeline/_common.bash`
* Board-assisted segmentation summary / root-level run report contract
* resume / QA 对 recording error 和 failed-user terminal state 的识别
* 对应 focused / integration tests
* `docs/notes/BOARD_EVENT_GUIDED_SEGMENTATION.md`
* `docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md`

**Out of scope**

* alignment matching、threshold、SUCCESS/SKIPPED classification 和 offset 算法
* label-mode segmentation
* Board event boundary/segment 算法
* Spike encoding / preprocessing
* padding target、padding 算法
* `data_sample/**`、`vendor/**`
* 将 unexpected programmer/system failure 无条件吞掉

## DAG

```text
P0 Probe
   ↓
F0 Freeze TaskSpec
   ↓
T1 Recording-error contract
   ↓
T2 Recording-level isolation
   ├──────────────┐
   ↓              ↓
T3 Summary      T4 Action0 orchestration
   │              │
   └──────┬───────┘
          ↓
T5 Resume / QA / downstream reconciliation
          ↓
T6 Tests + durable docs
          ↓
V0 Independent verifier
```

## Tasks

**P0 — Probe**
一句话目标：枚举 Board-assisted SUCCESS recording 从加载到 verification 的所有 failure point，并明确哪些属于可 quarantine 的 recording-local error、哪些必须继续 hard fail。

**F0 — Freeze TaskSpec**
一句话目标：冻结 recording terminal-state、error classification、summary/report、all-recordings-error user、resume/QA 和 downstream exclusion 的行为 contract，不冻结具体实现方式。

**T1 — Recording-error contract**
一句话目标：增加独立于 alignment `SUCCESS/SKIPPED` 的 segmentation recording-error 表达和显式 `error|skip` policy，保证 segmentation error 不伪装成 alignment SKIPPED。

**T2 — Recording-level isolation**
一句话目标：将单个 alignment-SUCCESS recording 的 Board-assisted segmentation 封装成独立执行单元，使允许跳过的错误只淘汰当前 recording 并继续下一个 recording。

**T3 — Summary / report contract**
一句话目标：扩展 user/action summary 记录 processed、alignment-skipped 和 segmentation-error recordings，同时保持 `alignment_outcome_dependency` 对全部 alignment SUCCESS/SKIPPED recording 的完整描述。

**T4 — Action0 orchestration**
一句话目标：让 Board-assisted Action0 segmentation 在单个 recording 甚至单个 user 无可用 recording 时仍继续后续 users，并在所有 users 完成后生成统一 machine-readable report 和 error CSV。

**T5 — Resume / QA / downstream reconciliation**
一句话目标：让 continue-mode、QA、length analysis 和 padding 把成功 package 与显式 terminal error 状态区分开，避免 skipped recording 被重新处理或被误判为 partial output。

**T6 — Tests + durable docs**
一句话目标：用 focused/integration tests 锁定 mixed-success、all-error-user、hard-fail 边界、resume 和 downstream exclusion，并同步更新两个 durable notes。

**V0 — Verifier**
一句话目标：独立验证完整 TaskSpec，重点检查 silent data loss、alignment contract 漂移、stale/resume 判断和 error-report/count reconciliation。

## 验收标准

* 同一 user 内 `dataset 0=success, dataset 1=recording error, dataset 2=success` 时，dataset 1 被记录并排除，dataset 0/2 正常完成 segmentation。
* dataset 1 不出现在 exported segments、Board targets、length analysis、padding 或训练输入中。
* 一个 user 的全部 recording 都发生允许跳过的 segmentation error 时，不生成伪造的空 segmentation package，也不阻止后续 users。
* 所有 users 尝试完成后生成统一 error report，至少包含 `user / action / dataset_id / stage / error_code or type / message`。
* user/action summary 满足 `source_recordings = processed + alignment_skipped + segmentation_errors`。
* segmentation-error recording 若 alignment 原本为 SUCCESS，仍保留在 `alignment_outcome_dependency.SUCCESS`，不得转换为 alignment SKIPPED。
* provenance stale/malformed、非法 alignment outcome、aggregate invariant、publication/filesystem failure 和未分类 unexpected failure 保持 hard fail。
* standalone 默认保持 strict；Action0 Board-assisted workflow 显式启用 recording skip policy。
* `PIPELINE_MODE=continue` 能识别上次已经记录的 terminal recording/user errors，不因缺失对应 segment package 而误触发 full preprocess rebuild。
* QA 同时核对 discovered recording、alignment SUCCESS/SKIPPED、segmentation processed/error、成功 user package 和 padded package 数量。
* 现有 Board boundary、event classification、segment slicing、alignment 和 padding 数值行为无 regression。
* focused tests、相关 integration tests 通过，Verifier 对冻结 TaskSpec 返回 `PASS`。
