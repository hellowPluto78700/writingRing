# Plan: Action0 SNN Training Notebook

## Goal

创建一个 Jupyter Notebook，用于现有 Action0 SNN 的：

```text
数据检查
→ 训练前可视化
→ 参数配置
→ 启动现有 SNN 训练
→ 训练结果可视化
→ 最终 test 结果汇总
```

Notebook 只作为现有训练 pipeline 的可视化和执行入口。

---

## Scope

Notebook 包含：

```text
1. 配置训练参数

2. 加载 train / val / test dataset

3. 显示 label 分布
   - label 数量
   - 每个 label 占 split 的百分比

4. 可视化一个 segment
   - 15 个 channel
   - 纵向排列
   - x = time
   - y = spike amplitude
   - 不显示 padding

5. 构建现有 SynNet

6. 启动现有训练

7. 记录每个 epoch 的 train / val metrics

8. 绘制：
   - loss
   - accuracy
   - balanced accuracy
   - macro F1
   - mean output spikes

9. 恢复 best validation model

10. 输出 train / val / test 最终 metrics 表格
```

---

## Out of Scope

本计划不修改：

```text
loss
prediction rule
SynNet implementation
spike encoding
segmentation
padding
dataset schema
training methodology
```

也不包含：

```text
自动超参搜索
cross-validation
confusion matrix
新的 performance metric
新的实验方案
```

---

# Task DAG

```text
T001 Notebook Foundation
        ↓
T002 Pre-training Visualization
        ↓
T003 SNN Training Integration
        ↓
T004 Result Visualization & Test
        ↓
T005 Final Verification
```

按照 `AGENTS.md`：

```text
DAG early.
TaskSpec late.
```

这里只定义粗粒度任务。

具体 TaskSpec 在任务变成 dependency-ready 后再：

```text
PROBE
→ FROZEN
→ IMPLEMENT
→ VERIFY
→ DONE
```

---

# T001 — Notebook Foundation

## Goal

建立 notebook 基础结构，并接入现有 Action0 dataset。

## Main Work

```text
Imports

Configuration cell

resolve segmentation_padded

load producer metadata

discover class_to_idx

build:
    train_dataset
    val_dataset
    test_dataset

display basic dataset summary
```

## Acceptance

```text
[ ] notebook 能加载真实 Action0 dataset

[ ] train / val / test 均可正常创建

[ ] 显示 segment 数量

[ ] 显示各 split label 数量

[ ] 显示 global class count

[ ] 所有训练参数集中在一个 configuration cell
```

---

# T002 — Pre-training Visualization

## Goal

完成训练开始前需要查看的两类数据图。

## Main Work

### Label distribution

分别绘制：

```text
Train
Validation
Test
```

每张图：

```text
x = label
y = percentage of split
```

并提供：

```text
Label | Train % | Val % | Test %
```

的表格。

### Segment visualization

从 dataset 中选择一个 segment。

绘制：

```text
15 rows × 1 column
```

对应：

```text
Channel 0
...
Channel 14
```

其中：

```text
x-axis = time
y-axis = spike amplitude
```

只显示：

```text
valid_mask == True
```

区域。

## Acceptance

```text
[ ] train label 图正确

[ ] val label 图正确

[ ] test label 图正确

[ ] percentage 计算正确

[ ] 一个真实 segment 可以被选择

[ ] 15 个 channel 全部显示

[ ] channel 上下排列

[ ] x-axis 是 time

[ ] y-axis 是 spike amplitude

[ ] padding 不显示
```

---

# T003 — Existing SNN Training Integration

## Goal

从 notebook 调用当前已有的 SNN training implementation。

## Reuse

必须复用当前：

```text
Action0SegmentDataset
SynNet
MaskedCrossEntropySpkReg
run_epoch
```

不重新实现新的训练方法。

## Configuration

Notebook 支持编辑：

```text
NEURONS_NETWORK

SHIFT_SYN
SHIFT_MEM

BATCH_SIZE
NUM_EPOCHS
LEARNING_RATE
SPIKE_REGULARIZATION

RANDOM_SEED
USE_GPU
```

## Training

每个 epoch：

```text
train
→ collect metrics
→ validation
→ collect metrics
→ update best validation model
```

继续使用：

```text
validation balanced_accuracy
```

选择 best model。

## Metrics

记录：

```text
loss
accuracy
balanced_accuracy
macro_f1
weighted_f1
mean_output_spikes
mean_total_spikes
zero_output_spike_fraction
```

## Acceptance

```text
[ ] notebook 可以启动现有 SynNet

[ ] output size 自动等于 class 数

[ ] train 正常运行

[ ] validation 正常运行

[ ] 每个 epoch metrics 都被保存

[ ] best validation model 被记录

[ ] 没有修改现有 loss 或 prediction behavior
```

---

# T004 — Result Visualization & Final Test

## Goal

把 T003 的训练数据画出来，并完成最终 test。

## Required Figures

```text
1. Train / Val Loss

2. Train / Val Accuracy

3. Train / Val Balanced Accuracy

4. Train / Val Macro F1

5. Train / Val Mean Output Spikes
```

## Final Evaluation

训练完成后：

```text
restore best validation model
```

然后运行：

```text
train evaluation
validation evaluation
test evaluation
```

## Final Table

输出：

| Metric                     | Train | Val | Test |
| -------------------------- | ----: | --: | ---: |
| Loss                       |       |     |      |
| Accuracy                   |       |     |      |
| Balanced Accuracy          |       |     |      |
| Macro F1                   |       |     |      |
| Weighted F1                |       |     |      |
| Mean Output Spikes         |       |     |      |
| Mean Total Spikes          |       |     |      |
| Zero Output Spike Fraction |       |     |      |

## Acceptance

```text
[ ] 5 张训练曲线正常生成

[ ] 使用真实 epoch history

[ ] best model 被恢复

[ ] test set 被独立评价

[ ] final metrics table 正常显示
```

---

# T005 — Final Verification

## Goal

确认整个 notebook 可以从头到尾执行，并且没有改变现有 SNN contract。

## Verification

检查：

```text
Configuration
→ Dataset load
→ Label plots
→ Segment plot
→ Build SynNet
→ Train
→ Training curves
→ Best model
→ Test
→ Final table
```

同时运行相关 pytest。

环境：

```text
Python 3.11
writingring-gpu
Matplotlib
```

## Acceptance

```text
[ ] notebook clean run 成功

[ ] 相关 SNN tests PASS

[ ] 没有修改 loss

[ ] 没有修改 SynNet behavior

[ ] 没有修改 producer / dataset contract

[ ] 没有加入 scope 外实验
```

---

# Definition of Done

整个 plan 完成时必须有：

```text
[ ] T001 DONE
[ ] T002 DONE
[ ] T003 DONE
[ ] T004 DONE
[ ] T005 PASS
```

最终 deliverable：

```text
1 个 Action0 SNN training notebook
```

功能仅包括：

```text
dataset inspection
label distribution
15-channel segment plot
existing SNN training
training curves
final train/val/test metrics
```
