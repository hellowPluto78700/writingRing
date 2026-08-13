## 目标

为 A/B/C/D 实验增加**固定排除用户（excluded users）** (以user_17为例)能力，使指定用户在任何情况下都不进入 train / val / test，同时保持现有 user-disjoint split、A checkpoint 作为 B/C/D split authority、以及实验可复现性不变。

## Scope

仅修改 acceleration reconstruction experiment workflow：

* `snn/accel_reconstruction_eval/config.py`
* `snn/accel_reconstruction_eval/datasets.py`
* `scripts/run_experiment_a.py`
* `scripts/run_experiment_b.py`
* `scripts/run_experiment_c.py`
* `scripts/run_experiment_d.py`
* `notebooks/experiment_A/B/C/D*.ipynb`
* 对应 split/config/runner tests

不修改 CNN、training、embedding、metric、reconstruction producer。

## DAG

```text
T1 Config contract
      ↓
T2 Split-layer exclusion
      ↓
T3 A runner integration
      ↓
T4 A checkpoint/provenance contract
      ↓
┌─────┼──────────┐
T5 B  T6 C       T7 D
└─────┼──────────┘
      ↓
T8 Notebook controls
      ↓
T9 Tests + regression validation
```

## Tasks

**T1 — Config contract**
目标：在 experiment config 中增加标准化的 `excluded_users` 配置，并保证 user name normalization 与现有 split 逻辑一致。

**T2 — Split-layer exclusion**
目标：让 `prepare_user_disjoint_splits()` 在自动或 explicit split 之前移除 excluded users，并显式验证 excluded user 不会出现在任何 split。

**T3 — Experiment A integration**
目标：让 A runner 将 `excluded_users` 传入 split 层，并基于过滤后的 eligible cohort 生成新的 train/val/test。

**T4 — A checkpoint/provenance contract**
目标：将 excluded users 和 eligible cohort 信息写入 A checkpoint、`sample_manifest`/split artifacts 与 `provenance.json`，使后续实验能够审计实际 cohort。

**T5 — Experiment B inheritance**
目标：让 B 以 A checkpoint 的 train/val/test/excluded cohort 为唯一 authority，并拒绝当前 dataset 中与 A cohort contract 冲突的情况。

**T6 — Experiment C inheritance**
目标：让 C 默认继承 A 的 excluded cohort 和 split metadata，同时保持“只复用 split/class metadata、不复用 A weights/normalization”的协议。

**T7 — Experiment D inheritance**
目标：让 D 默认继承 A 的 excluded cohort 和 split metadata，同时保持 mixed-training protocol 不变。

**T8 — Notebook controls**
目标：在 A notebook 暴露 `EXCLUDED_USERS` 作为唯一用户级入口，B/C/D 只展示从 A checkpoint 继承的 exclusion，不允许各自静默定义不同 cohort。

**T9 — Tests + regression validation**
目标：覆盖 automatic split、explicit split、单/多用户 exclusion、非法重叠、A→B/C/D inheritance 以及空 split/缺 label 等边界条件，并验证无 exclusion 时行为与当前实现一致。

## 验收标准

* 指定 `EXCLUDED_USERS={"user_x"}` 后，该用户的任何 sample 都不出现在 train、val、test、DataLoader 或 embedding artifacts 中。
* train / val / test 仍严格 user-disjoint，且默认比例基于**排除后的 eligible users**计算。
* A checkpoint 明确保存 `excluded_users`、train/val/test users 和 class mapping。
* B/C/D 默认从同一个 A checkpoint 继承 cohort，不会重新把 excluded user 引入。
* B 仍严格 frozen A weights + A normalization；C/D 仍不复用 A weights/normalization。
* provenance 和 split artifacts 能明确回答“哪些用户被排除、哪些用户进入各 split”。
* explicit split 若包含 excluded user，必须 fail fast，而不是静默处理。
* exclusion 导致某个 split 为空、某 label 缺失或剩余用户不足时，必须给出明确错误。
* `excluded_users=()` 时，现有 A/B/C/D 行为与结果 contract 不发生回归。
* 所有新增及现有相关测试通过。
