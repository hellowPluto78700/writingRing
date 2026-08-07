
---

# Codex Plan：Action0 Label-Based SNN Training

## 0. 最终目标

在当前仓库：

```text
~/project/writingRing/
├── AGENTS.md
├── TASK.md
├── PROGRESS.md
├── README.md
├── configs/
├── data/
├── docs/
├── notebooks/
├── outputs/
├── scripts/
├── snn/
├── src/
├── tests/
├── environment.yml
└── pyproject.toml
```

建立一条新的、独立的 SNN segment classifier 训练路径。

固定任务定义：

```text
task:
    segment-level classification

dataset variant:
    lowpass
    raw
    madgwick
    xylo

boundary:
    强制 label
    不允许 aligned-board-events

dataset root example:
    outputs/action0_pipeline/low-pass/label/

training data:
    outputs/action0_pipeline/<variant_dir>/label/segmentation/

input:
    padded SpikeIMU segments

input channels:
    spikeIMU[..., 0:15]

input channel count:
    15

target:
    *_labels.npy

mask:
    *_valid_mask.npy

segment length:
    across-dataset fixed padded length

network:
    原 SynNet

hidden sizes:
    [24, 24, 24]

default:
    shift_syn = 2
    shift_mem = 1

criterion:
    masked cross entropy

baseline:
    train from scratch
    spike_regularization = 0
```

---

# 1. 首先检查现有 producer contract

Codex 开始写代码前必须先读取：

```text
AGENTS.md
README.md

scripts/action0_pipeline/_common.bash
src/writingring/segment_padding.py

snn/utils_architectures.py
snn/utils_parser.py
snn/har_snn.py
snn/utils_losses.py
snn/utils_run.py
```

重点检查：

```text
scripts/action0_pipeline/_common.bash:420 附近

src/writingring/segment_padding.py:371 附近
src/writingring/segment_padding.py:815 附近
```

确认真实输出 contract：

```text
*_spikeIMU.npy
*_labels.npy
*_segment_offsets.npy
*_segment_lengths.npy
*_valid_mask.npy
```

尤其确认：

```text
valid_mask.npy shape
padding 后 offsets 的语义
padding 后 spikeIMU 的实际布局
每个 segment 如何根据 offsets 定位
```

**实际 producer 代码是 source of truth。不要根据本 plan 猜 shape。**

---

# 2. 数据集 variant 设计

训练入口必须先选择一个 dataset variant。

CLI 使用：

```text
--dataset_variant
```

允许：

```text
lowpass
raw
madgwick
xylo
```

内部建立唯一映射：

```python
DATASET_VARIANT_DIRS = {
    "lowpass": "low-pass",
    "raw": "raw",
    "madgwick": "madgwick",
    "xylo": "xylo",
}
```

因此：

```text
--dataset_variant lowpass
```

解析为：

```text
outputs/action0_pipeline/low-pass/label/
```

而：

```text
--dataset_variant madgwick
```

解析为：

```text
outputs/action0_pipeline/madgwick/label/
```

---

# 3. boundary 不作为训练参数

**不要提供：**

```text
--boundary
```

不要允许：

```text
aligned-board-events
```

训练代码内部固定：

```python
BOUNDARY = "label"
```

dataset root：

```python
dataset_root = (
    Path(args.pipeline_root)
    / DATASET_VARIANT_DIRS[args.dataset_variant]
    / "label"
)
```

segmentation root：

```python
segmentation_root = dataset_root / "segmentation"
```

例如：

```text
outputs/action0_pipeline/low-pass/label/segmentation/
```

启动时打印：

```text
Dataset variant: lowpass
Boundary: label
Dataset root: outputs/action0_pipeline/low-pass/label
Segmentation root: ...
```

如果目录不存在，直接明确报错。

---

# 4. 新 SNN 文件结构

不要直接破坏 legacy HAR trainer。

保留原文件作为参考：

```text
snn/
├── har_snn.py
├── utils_parser.py
├── utils_datasets.py
├── utils_architectures.py
├── utils_losses.py
└── utils_run.py
```

新增 Action0-specific 路径：

```text
snn/
├── action0_dataset.py
├── action0_losses.py
├── action0_engine.py
├── action0_parser.py
└── train_action0.py
```

最终：

```text
snn/
├── __init__.py
│
├── utils_architectures.py    # existing，保留 SynNet，最小修改
│
├── action0_dataset.py        # NEW
├── action0_losses.py         # NEW
├── action0_engine.py         # NEW
├── action0_parser.py         # NEW
├── train_action0.py          # NEW
│
├── har_snn.py                # legacy
├── utils_parser.py           # legacy
├── utils_datasets.py         # legacy
├── utils_losses.py           # legacy
└── utils_run.py              # legacy
```

---

# 5. `snn/action0_dataset.py`：完全新写

这个文件只服务新的 Action0 pipeline。

不要继承旧：

```text
BaseHARDataset
CAPTURE24
PAMAP
WISDM
...
```

不要执行旧 transforms。

新 Dataset 的职责：

```text
选择 dataset variant
    ↓
固定进入 label boundary
    ↓
扫描 segmentation/
    ↓
定位 user/action
    ↓
读取 padded SpikeIMU
    ↓
根据 offsets 取得 segment
    ↓
只取 channel 0:15
    ↓
读取 segment label
    ↓
读取对应 valid_mask
    ↓
return x, label, mask
```

---

# 6. Dataset 初始化接口

建议：

```python
Action0SegmentDataset(
    segmentation_root,
    users,
    class_to_idx,
)
```

dataset variant 不一定需要传进 Dataset class。

variant → path 的转换应在：

```text
train_action0.py
```

或者一个小 utility 中完成。

这样 Dataset 本身只关心：

```text
segmentation_root
```

而不知道：

```text
lowpass/raw/madgwick/xylo
```

---

# 7. Dataset 输出接口

严格固定：

```python
x, label, valid_mask = dataset[index]
```

其中：

```text
x:
    dtype torch.float32
    shape (T_pad, 15)

label:
    scalar integer
    compatible with torch.long

valid_mask:
    dtype torch.bool
    shape (T_pad,)
```

DataLoader 自动 stack：

```text
inputs:
    (B, T_pad, 15)

labels:
    (B,)

valid_mask:
    (B, T_pad)
```

因为 upstream 已经完成 across-dataset fixed padding，所以：

**不要实现动态 `pad_sequence()`。**

---

# 8. 输入严格只取 spike channels

原始：

```text
SpikeIMU:
    0:15    wavelet spike/event channels
    15:18   acceleration
    18:21   gyroscope
```

Dataset 中明确：

```python
x = x[:, :15]
```

然后：

```python
assert x.shape[1] == 15
```

禁止：

```text
raw accel
gyro
all21
imu6
```

进入本 baseline。

不要增加 `feature_mode` 参数。

当前 trainer 永远使用：

```text
inputSize = 15
```

---

# 9. Dataset 完整性检查

Dataset 初始化必须验证 producer output。

至少：

```text
SpikeIMU.ndim == 2
SpikeIMU.shape[1] == 21

offsets.ndim == 1
offsets 单调非递减/严格递增，按实际 producer contract

labels segment count 正确
valid_mask segment count 正确

所有 padded segment T_pad 一致

每个 segment mask shape == (T_pad,)
每个 segment 至少有一个 True

输入不存在 NaN/Inf
```

如果 producer contract 保证：

```text
mask=True 的区域连续
```

则增加连续性检查。

---

# 10. train / val / test split

优先按照 user 划分。

parser：

```text
--train_users
--val_users
--test_users
```

启动时必须验证：

```python
train_users.isdisjoint(val_users)
train_users.isdisjoint(test_users)
val_users.isdisjoint(test_users)
```

如果失败：

```text
raise ValueError
```

不要自动修复。

不要默认随机 segment split。

---

# 11. labels 与 class mapping

先扫描选定 dataset variant 中的 labels。

建立全局：

```python
class_to_idx
idx_to_class
```

train / val / test 必须使用完全相同 mapping。

不要使用旧：

```python
num_outputs = {
    "Capture24": ...,
    "MHealth": ...,
}
```

新：

```python
num_outputs = len(class_to_idx)
```

启动时打印：

```text
class_to_idx
train class distribution
val class distribution
test class distribution
```

---

# 12. `shift_syn` / `shift_mem`：严格沿用原代码参数方式

这是本次 plan 的重要修改。

新 parser 中参数名称、默认值按照原代码：

```python
parser.add_argument(
    "--shift_syn",
    type=int,
    default=2,
)

parser.add_argument(
    "--shift_mem",
    type=int,
    default=1,
)
```

原 parser 本来就是：

```text
--shift_syn default=2
--shift_mem default=1
```

不要改成：

```text
tau_syn
tau_mem
alpha
beta
continuous_tau
```

作为用户训练参数。

---

# 13. `train_action0.py` 传 shift 的方式也沿用原代码

原 `har_snn.py` 的逻辑是：

```python
kwargs = {}

if "SynNet" in args.network_type:
    kwargs.update(
        [
            ("shiftSyn", args.shift_syn),
            ("shiftMem", args.shift_mem),
        ]
    )
```

然后：

```python
model = createModel(
    ...,
    **kwargs,
)
```

新代码保持同样设计。

即：

```python
model_kwargs = {
    "shiftSyn": args.shift_syn,
    "shiftMem": args.shift_mem,
}

model = createModel(
    "SynNet",
    inputSize=15,
    outputSize=num_outputs,
    hiddenSizes=args.neurons_network,
    device=device,
    sampleFreq=args.sample_rate,
    **model_kwargs,
)
```

不要在 trainer 中直接计算 alpha/beta。

---

# 14. SynNet 内部 shift 定义完全保持原样

当前 SynNet：

```python
shiftsSyn = list(
    range(
        shiftSyn,
        shiftSyn + 8,
    )
)
```

隐藏层 alpha：

```python
alphas = [
    [
        1 - 2 ** (-shiftSyn)
        for shiftSyn in shiftsSyn[:2 ** (l + 1)]
        for _ in range(
            hiddenSizes[l] // 2 ** (l + 1)
        )
    ]
    for l in range(len(hiddenSizes))
]
```

膜 decay：

```python
beta = (
    1 - 2 ** (-shiftMem)
)
```

这部分**不要重写、不要简化、不要自动转换 sample rate**。

默认：

```text
shift_syn = 2

lif1:
    shifts 2,3

lif2:
    shifts 2,3,4,5

lif3:
    shifts 2,3,4,5,6,7,8,9

shift_mem = 1

所有 hidden membrane beta:
    0.5
```

---

# 15. sample rate 不参与 shift 自动映射

新 parser 仍然需要：

```text
--sample_rate
```

或者如果希望和原风格统一：

```text
--sample_freq
```

建议直接叫：

```text
--sample_freq
```

因为 `createModel()` / `SynNet` 原来就是 `sampleFreq`。

例如：

```python
parser.add_argument(
    "--sample_freq",
    type=float,
    required=True,
)
```

传入：

```python
sampleFreq=args.sample_freq
```

但**禁止**：

```text
根据 sample_freq 自动改变 shift_syn
根据 sample_freq 自动改变 shift_mem
自动生成新的 alpha/beta
```

sample frequency 当前只是：

```text
实验 metadata
tau diagnostics
兼容原 model constructor
```

---

# 16. 保留原 tau 计算，但不要改变模型行为

原 SynNet 内部当前会计算：

```python
tausSyn = [...]
tauMem = ...
```

这段可以保持。

但是现阶段不要尝试：

```text
把 tau 重新传给 snn.Synaptic
修改 alpha/beta
让 tau learnable
```

baseline 网络动力学必须尽可能与原实现一致。

---

# 17. `utils_architectures.py` 唯一必要功能修改：mask-aware `spkTotal`

网络结构不变。

修改：

```python
def forward(self, x):
```

为：

```python
def forward(
    self,
    x,
    valid_mask=None,
):
```

时间循环不变：

```python
for step in range(x.shape[1]):
    ...
```

不要根据 mask skip timestep。

不要 freeze neuron state。

---

# 18. mask 第一阶段只控制统计，不控制 neuron dynamics

即使：

```text
valid_mask=False
```

SynNet 仍然正常计算该时间步。

mask 只用于：

```text
classification loss
final output spike count
spike statistics
spike regularization
```

这样不会因为 dataset migration 同时改变原 SNN dynamics。

---

# 19. `spkTotal` 改成只统计 valid 区域

原模型：

```python
self.spkTotal = (
    spk1_rec.sum()
    + spk2_rec.sum()
    + spk3_rec.sum()
    + spk4_rec.sum()
)
```

改为：

```python
if valid_mask is None:
    self.spkTotal = (
        spk1_rec.sum()
        + spk2_rec.sum()
        + spk3_rec.sum()
        + spk4_rec.sum()
    )
else:
    mask = valid_mask.unsqueeze(-1).to(
        spk1_rec.dtype
    )

    self.spkTotal = (
        (spk1_rec * mask).sum()
        + (spk2_rec * mask).sum()
        + (spk3_rec * mask).sum()
        + (spk4_rec * mask).sum()
    )
```

输出仍然：

```python
return spk4_rec
```

---

# 20. 不让 Rockpool 阻塞 baseline

如果 `utils_architectures.py` 顶层仍然：

```python
from rockpool_nn_networks_synnet import ...
```

尽量改成 lazy import。

要求：

```text
Action0 + local SynNet
```

不应该因为：

```text
Rockpool
Xylo deployment code
```

而无法 import。

注意：

```text
dataset_variant = xylo
```

这里只表示：

```text
读取 outputs/action0_pipeline/xylo/label/
```

**不代表网络必须使用 Rockpool SynNetRP。**

训练网络仍然是：

```text
local snnTorch SynNet
```

---

# 21. `snn/action0_losses.py`

新写：

```text
MaskedCrossEntropySpkReg
```

接口：

```python
criterion(
    output,
    labels,
    valid_mask,
    spikes=None,
)
```

shape：

```text
output:
    B,T,C

labels:
    B

valid_mask:
    B,T
```

逻辑仍然遵循原 CrossEntropy 思路：

> 一个 segment 的 label 复制到每个有效时间步。

但是只计算：

```text
valid_mask == True
```

的位置。

---

# 22. masked CE 计算

实现：

```python
B, T, C = output.shape

targets = labels[:, None].expand(
    B,
    T,
)

loss_per_step = F.cross_entropy(
    output.reshape(B * T, C),
    targets.reshape(B * T),
    reduction="none",
).reshape(B, T)

mask = valid_mask.to(
    loss_per_step.dtype
)

classification_loss = (
    loss_per_step * mask
).sum() / mask.sum().clamp_min(1)
```

---

# 23. 第一版不用 FirstWin

不要迁移：

```text
cutoff=320
```

的旧 FirstWin behavior。

当前 baseline：

```text
criterion_type = CrossEntropy
```

甚至可以第一版不提供 `criterion_type` 参数。

固定：

```python
criterion = MaskedCrossEntropySpkReg(...)
```

---

# 24. spike regularization baseline 关闭

parser 保留：

```text
--spike_regularization
```

默认：

```text
0
```

第一阶段不复制旧：

```text
10 sec × 64 Hz
```

归一化公式。

以后如果非零，则采用：

```text
valid neuron-time
```

归一化。

---

# 25. `snn/action0_engine.py`

新写：

```python
run_epoch(
    model,
    dataloader,
    criterion,
    optimizer=None,
    split="train",
    device=None,
)
```

DataLoader：

```python
for (
    inputs,
    labels,
    valid_mask,
) in dataloader:
```

---

# 26. forward

```python
inputs = inputs.to(
    device,
    dtype=torch.float32,
)

labels = labels.to(
    device,
    dtype=torch.long,
)

valid_mask = valid_mask.to(
    device,
    dtype=torch.bool,
)

output = model(
    inputs,
    valid_mask=valid_mask,
)
```

assert：

```python
assert output.ndim == 3
assert output.shape[:2] == inputs.shape[:2]
assert output.shape[2] == num_classes
```

---

# 27. loss

```python
loss = criterion(
    output,
    labels,
    valid_mask,
    spikes=model.spkTotal,
)
```

训练：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

# 28. prediction 必须 mask

不能继续：

```python
output.sum(1)
```

而应：

```python
mask = valid_mask.unsqueeze(-1).to(
    output.dtype
)

masked_output = output * mask

spike_counts = masked_output.sum(dim=1)
```

得到：

```text
(B, num_classes)
```

最终：

```python
preds = spike_counts.argmax(dim=1)
```

---

# 29. score

为了 metrics：

```python
eps = 1e-6

scores = (
    spike_counts + eps
) / (
    spike_counts.sum(
        dim=1,
        keepdim=True,
    )
    + eps * spike_counts.shape[1]
)
```

必须支持：

```text
output 全零 spike
```

而不会产生 NaN。

---

# 30. engine metrics

至少：

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

AUROC 可保留，但如果某 split 缺类：

```text
warning
return nan
```

不要 crash。

---

# 31. `snn/action0_parser.py`

建议完整参数：

```text
# Dataset
--pipeline_root
--dataset_variant

--train_users
--val_users
--test_users

--sample_freq
--num_workers

# Architecture
--neurons_network
--network_type
--shift_syn
--shift_mem

# Training
--batch_size
--num_epochs
--learning_rate
--spike_regularization

# Reproducibility
--random_seed

# Runtime
--use_gpu
--use_wandb

# Checkpoint
--model_checkpoint
--save_model
--model_dir

# Debug
--dry_run
--max_train_batches
```

---

# 32. architecture parser 尽量保留原风格

尤其保持：

```python
parser.add_argument(
    "--neurons_network",
    nargs=3,
    type=int,
    default=[24, 24, 24],
)

parser.add_argument(
    "--network_type",
    type=str,
    default="SynNet",
)

parser.add_argument(
    "--shift_syn",
    type=int,
    default=2,
)

parser.add_argument(
    "--shift_mem",
    type=int,
    default=1,
)
```

虽然当前只训练 SynNet，但这样和原训练脚本保持一致。

启动时：

```python
if args.network_type != "SynNet":
    raise NotImplementedError(
        "Action0 baseline currently supports SynNet only"
    )
```

---

# 33. 删除旧 HAR-only parser 参数

新的 Action0 parser 不包含：

```text
dataset_name
label_type
subject_splits

use_xylo
raw_data_wg
raw_data_gr

time_compress
add_gravity
compress_channels
polarity_bichannel
rectify_spikes
system_type
time_resolution
quantize_spikes

relax_taus
```

特别注意：

旧：

```text
--use_xylo
```

和新：

```text
--dataset_variant xylo
```

是完全不同概念。

新 `xylo` 只是数据目录选择。

---

# 34. `snn/train_action0.py`

执行顺序：

```text
1. parse args

2. validate dataset_variant

3. map variant to directory

4. force boundary="label"

5. build:
   outputs/action0_pipeline/<variant>/label/

6. resolve segmentation/

7. discover class mapping

8. construct train/val/test Dataset

9. print dataset statistics

10. construct DataLoaders

11. inspect first batch

12. assert input channels=15

13. num_inputs=15

14. num_outputs=len(class_to_idx)

15. create SynNet using ORIGINAL shift passing style

16. print SNN dynamics

17. create masked CE

18. create Adam

19. optional checkpoint

20. training loop

21. validation

22. save best

23. final test
```

---

# 35. 模型创建必须长这样

逻辑上尽量保持原版：

```python
kwargs = {}

if "SynNet" in args.network_type:
    kwargs.update(
        [
            (
                "shiftSyn",
                args.shift_syn,
            ),
            (
                "shiftMem",
                args.shift_mem,
            ),
        ]
    )

model = createModel(
    args.network_type,
    inputSize=15,
    outputSize=num_outputs,
    hiddenSizes=args.neurons_network,
    device=device,
    sampleFreq=args.sample_freq,
    **kwargs,
)
```

不要在这里自行计算：

```text
alpha
beta
tau
new shift
```

---

# 36. optimizer 保持简单

baseline：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
)
```

不要搬 legacy checkpoint fine-tuning 的：

```text
layer1 lr/8
layer2 lr/4
layer3 lr/2
layer4 lr
```

第一版 from scratch 不需要。

---

# 37. Dataset variant CLI 示例

low-pass：

```bash
python -m snn.train_action0 \
    --dataset_variant lowpass \
    --pipeline_root outputs/action0_pipeline \
    --sample_freq <ACTUAL_HZ> \
    --train_users ... \
    --val_users ... \
    --test_users ... \
    --shift_syn 2 \
    --shift_mem 1
```

解析路径：

```text
outputs/action0_pipeline/low-pass/label/
```

raw：

```bash
--dataset_variant raw
```

路径：

```text
outputs/action0_pipeline/raw/label/
```

madgwick：

```text
outputs/action0_pipeline/madgwick/label/
```

xylo：

```text
outputs/action0_pipeline/xylo/label/
```

---

# 38. 启动时必须打印 dynamics

打印：

```text
Network:
    SynNet

Input channels:
    15

Hidden sizes:
    [24,24,24]

Sample frequency:
    xxx Hz

shift_syn:
    2

hidden synaptic shifts:
    layer 1: 2-3
    layer 2: 2-5
    layer 3: 2-9

shift_mem:
    1

beta:
    0.5
```

tau 可以作为诊断信息计算，但必须注明：

```text
diagnostic only
does not change alpha/beta
```

---

# 39. checkpoint

保存：

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,

    "epoch": ...,
    "best_val_metric": ...,

    "dataset_variant": ...,
    "boundary": "label",

    "class_to_idx": ...,

    "input_channels": [0, 15],
    "num_inputs": 15,
    "num_outputs": ...,

    "hidden_sizes": ...,

    "shift_syn": ...,
    "shift_mem": ...,
    "sample_freq": ...,

    "train_users": ...,
    "val_users": ...,
    "test_users": ...,
}
```

特别保存：

```text
dataset_variant
boundary=label
shift_syn
shift_mem
sample_freq
```

以后才能准确复现实验。

---

# 40. `--dry_run`

必须实现：

```bash
python -m snn.train_action0 ... --dry_run
```

执行：

```text
resolve dataset
→ Dataset
→ DataLoader
→ first batch
→ create SynNet
→ forward
→ masked loss
→ backward
→ optimizer.step
→ report
→ exit
```

输出：

```text
dataset_variant
dataset_root

inputs shape
labels shape
mask shape

valid fraction
num classes

shift_syn
shift_mem
beta

output shape
loss

total valid spikes
output valid spikes
zero-output-spike fraction
```

---

# 41. Tests

新增：

```text
tests/test_snn_action0_dataset.py
tests/test_snn_action0_loss.py
tests/test_snn_action0_model.py
tests/test_snn_action0_smoke.py
```

Dataset test 要覆盖四种 variant path resolution：

```text
lowpass -> low-pass/label
raw -> raw/label
madgwick -> madgwick/label
xylo -> xylo/label
```

并验证代码永远不会寻找：

```text
aligned-board-events
```

---

# 42. Dataset path test

测试：

```python
resolve_dataset_root(
    pipeline_root,
    "lowpass",
)
```

必须等于：

```text
<pipeline_root>/low-pass/label
```

同时：

```python
resolve_dataset_root(
    pipeline_root,
    "raw",
)
```

等于：

```text
<pipeline_root>/raw/label
```

---

# 43. channel test

构造 fake SpikeIMU：

```text
(T,21)
```

其中：

```text
0:15 = known values A
15:21 = known values B
```

Dataset 返回必须：

```text
(T,15)
```

且严格等于 A。

保证 raw IMU 永远没有进入网络。

---

# 44. mask loss regression test

构造两个 output：

```text
valid region 相同
padding region 完全不同
```

要求：

```text
masked CE 完全相同
```

这样可以证明 padding 不影响 loss。

---

# 45. masked score test

同理修改 padding 区域 spikes。

要求：

```text
最终 spike_counts 不变
prediction 不变
```

---

# 46. shift regression test

这是这版 plan 新增的重点测试。

创建：

```python
model = SynNet(
    inputSize=15,
    outputSize=K,
    hiddenSizes=[24,24,24],
    shiftSyn=2,
    shiftMem=1,
)
```

验证：

```text
beta == 0.5
```

并验证 hidden alpha 分布对应：

```text
layer 1:
    shifts [2,3]

layer 2:
    shifts [2,3,4,5]

layer 3:
    shifts [2,3,4,5,6,7,8,9]
```

目的是防止迁移过程中有人把原 heterogeneous alpha 错误简化成：

```text
所有 neuron alpha=0.75
```

---

# 47. sample frequency regression test

使用不同：

```text
sample_freq=64
sample_freq=100
sample_freq=200
```

在：

```text
shift_syn=2
shift_mem=1
```

条件下验证：

```text
alpha/beta 不改变
```

即：

```text
sample frequency change
!=
shift change
```

这正是当前 baseline 的要求。

---

# 48. 不做的事情

Codex 本轮明确不要：

```text
不要允许 boundary 选择
不要加载 aligned-board-events
不要使用 board_event_targets

不要输入 channels 15:21

不要实现 all21 / imu6 feature mode

不要重新 padding
不要重新 downsample
不要重新 quantize
不要重新 spike encode

不要改变 SynNet topology

不要改变 shift_syn 定义
不要改变 shift_mem 定义

不要自动根据 sample rate remap shift

不要直接设置 alpha/beta
不要 learn_alpha
不要 learn_beta

不要改 surrogate gradient

不要实现 FirstWin

不要做 transfer learning

不要把 dataset_variant=xylo
误认为 network_type=SynNetRP
```

---

# 49. 推荐实施顺序

Codex 按以下阶段实施：

```text
Phase 1
Inspect producer + existing SNN contracts

Phase 2
Implement dataset variant path resolver
    lowpass/raw/madgwick/xylo
    boundary fixed label

Phase 3
Implement action0_dataset.py
+ dataset tests

Phase 4
Implement action0_losses.py
+ mask regression tests

Phase 5
Minimal modification to existing SynNet
    forward(valid_mask=None)
    masked spkTotal

Phase 6
Add shift regression tests
    confirm exact original behavior

Phase 7
Implement action0_engine.py
    masked score
    metrics

Phase 8
Implement action0_parser.py

Phase 9
Implement train_action0.py
    original-style shift passing

Phase 10
Implement --dry_run

Phase 11
Synthetic smoke test

Phase 12
Run one real:
    lowpass/label
    dry-run only

Phase 13
Tiny overfit experiment

Phase 14
Full training

Phase 15
Documentation
```

---

# 50. 最终验收条件

任务完成必须满足：

```text
[1]
CLI 可选择：
lowpass/raw/madgwick/xylo

[2]
lowpass 自动解析到：
outputs/action0_pipeline/low-pass/label/

[3]
boundary 永远固定为：
label

[4]
代码没有 Action0 training 的
aligned-board-events 分支

[5]
模型输入永远只有：
channels 0:15

[6]
batch shape：
(B,T_pad,15)

[7]
labels：
(B,)

[8]
valid_mask：
(B,T_pad)

[9]
padding 不参与 classification loss

[10]
padding 不参与 final spike-count prediction

[11]
padding 不参与 spkTotal/statistics

[12]
SynNet topology 保持原样

[13]
--shift_syn 参数名称和传递方式保持原代码风格

[14]
--shift_mem 参数名称和传递方式保持原代码风格

[15]
默认：
shift_syn=2
shift_mem=1

[16]
SynNet 内仍使用：
beta = 1 - 2**(-shiftMem)

[17]
SynNet 内仍使用原 heterogeneous
shiftsSyn = range(shiftSyn, shiftSyn+8)

[18]
改变 sample_freq 不自动改变 alpha/beta

[19]
network_type=SynNet

[20]
dataset_variant=xylo
仍然训练 local SynNet，
不自动切 SynNetRP

[21]
train/val/test users 无交集

[22]
没有旧机器 absolute path

[23]
不用 W&B 也能训练

[24]
不用 Rockpool 也能训练 local SynNet

[25]
--dry_run 完成：
data → forward → loss → backward → step

[26]
所有新增 pytest 通过
```

最终建议目录不变：

```text
writingRing/
├── snn/
│   ├── __init__.py
│   ├── utils_architectures.py
│   │
│   ├── action0_dataset.py
│   ├── action0_losses.py
│   ├── action0_engine.py
│   ├── action0_parser.py
│   ├── train_action0.py
│   │
│   └── legacy files...
│
├── tests/
│   ├── test_snn_action0_dataset.py
│   ├── test_snn_action0_loss.py
│   ├── test_snn_action0_model.py
│   └── test_snn_action0_smoke.py
│
└── docs/
    └── snn_action0_training.md
```

