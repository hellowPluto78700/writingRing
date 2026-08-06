## 方案结论

按仓库现有接口，8 个脚本应对应：

* 4 种 gravity removal：`raw`、`low-pass`、`madgwick`、`xylo-rotate-and-remove-gravity`
* 2 种 segmentation：`label`、`aligned-board-events`

两种 segmentation 都使用已经编码好的 `SpikeIMU`，即：

```text
(session)_ring_0.bin
  → gravity removal，生成 (N, 9) preprocessedIMU
  → custom-wavelet，生成 (N, 21) spikeIMU
  → label 或 aligned-board-events segmentation
```

仓库的预处理 CLI 正好支持上述四种方法；编码器只接收完整 recording，而不是预先切分的窗口。

## 8 个脚本

建议放在：

```text
scripts/action0_pipeline/
```

文件命名：

| 脚本                             | Gravity method                   | Segmentation           |
| ------------------------------ | -------------------------------- | ---------------------- |
| `01_raw_label.sh`              | `raw`                            | `label`                |
| `02_raw_aligned_board.sh`      | `raw`                            | `aligned-board-events` |
| `03_lowpass_label.sh`          | `low-pass`                       | `label`                |
| `04_lowpass_aligned_board.sh`  | `low-pass`                       | `aligned-board-events` |
| `05_madgwick_label.sh`         | `madgwick`                       | `label`                |
| `06_madgwick_aligned_board.sh` | `madgwick`                       | `aligned-board-events` |
| `07_xylo_label.sh`             | `xylo-rotate-and-remove-gravity` | `label`                |
| `08_xylo_aligned_board.sh`     | `xylo-rotate-and-remove-gravity` | `aligned-board-events` |

每个脚本独立完成完整 pipeline，不依赖其他脚本先执行。

## 数据发现规则

脚本不硬编码 `user_0` 到 `user_20` 或 session `0` 到 `5`，而是扫描实际存在的文件：

```bash
data/user_*/0/*_ring_0.bin
```

对每个文件：

```text
data/user_12/0/3_ring_0.bin
     └ user=user_12
                └ action=0
                  └ session/dataset-id=3
```

这里把用户所说的 `session` 映射为仓库 CLI 中的 `dataset-id`。

只匹配 `*_ring_0.bin`，因此不会处理 `ring_1`。仓库本身也把 `ring_0` 定义为 primary Ring input。Board 文件按相同 session 前缀匹配，并由仓库 loader 按数字 chunk index 加载，不应在 Bash 中逐 chunk 调用 pipeline。

建议使用：

```bash
find "$DATA_ROOT" \
  -type f \
  -path "$DATA_ROOT/user_*/0/*_ring_0.bin" \
  -print0 |
  sort -zV
```

这样 user 和 session 都按自然数字顺序运行。

## 每个脚本的公共结构

### 1. 初始化和 preflight

每个脚本开头：

```bash
set -Eeuo pipefail
```

公共配置：

```text
DATA_ROOT=data
ACTION=0
SAMPLING_RATE=200
ENCODER=custom-wavelet
ENCODER_SETTINGS=configs/spike_encoding/custom_wavelet.json
OVERWRITE=0
```

检查：

* `data/` 存在。
* 至少发现一个 `user_*/0/*_ring_0.bin`。
* session 前缀必须是整数，因为 `--dataset-id` 类型为整数。
* `configs/spike_encoding/custom_wavelet.json` 存在。
* Python 环境可以 import 项目。
* Xylo 两个脚本额外检查 Xylo dependency；仓库要求先安装 `pip install -e ".[xylo]"`。
* Board 模式下，每个 session 至少存在一个 `${session}_board_*.gz`。
* 不手动检查或重新排序 Board timestamp；交给仓库 loader 验证。

### 2. Gravity removal

对每个发现的 Ring 文件，调用一次精确 selector：

```bash
python scripts/preprocess_ring_imu.py \
  --data-root "$DATA_ROOT" \
  --output-root "$PREPROCESS_ROOT" \
  --user "$user" \
  --action 0 \
  --dataset-id "$session" \
  --gravity-removal-method "$METHOD" \
  --sampling-rate 200
```

方法相关参数：

```text
raw:
  --gravity-removal-method raw

low-pass:
  --gravity-removal-method low-pass
  --low-pass-cutoff-hz 0.2

madgwick:
  --gravity-removal-method madgwick
  --madgwick-beta 0.1

xylo:
  --gravity-removal-method xylo-rotate-and-remove-gravity
```

Madgwick 默认采用严格 calibration。可以保留环境变量开关：

```text
MADGWICK_PROVISIONAL=1 → 增加 --provisional
```

但默认不加，避免静默接受 calibration failure。

每个 session 应产生：

```text
preprocessedIMU/<user>/0/<session>/
├── <session>_preprocessedIMU.npy
├── <session>_timestamps_us.npy
└── <session>_preprocessing.json
```

### 3. Spike encoding

所有 session 完成预处理后，整棵树批量编码一次：

```bash
python scripts/encode_spikes.py \
  --input-root "$PREPROCESS_ROOT" \
  --pattern '*_preprocessedIMU.npy' \
  --output-root "$SPIKE_OUTPUT_ROOT" \
  --encoder custom-wavelet \
  --encoder-settings configs/spike_encoding/custom_wavelet.json
```

实际供下游使用的 Spike root 是：

```text
$SPIKE_OUTPUT_ROOT/custom-wavelet
```

每个 recording 独立创建 encoder、独立 reset，并保持原来的 `N` 行。最终 `spikeIMU.npy` 为 `(N, 21)`：15 个 wavelet event channel，加 6 个 acceleration/gyro channel。

`raw` 方法是唯一例外，编码命令必须额外增加：

```bash
--allow-gravity-included
```

否则 encoder 会拒绝 metadata 中声明包含重力的 measured acceleration。

### 4A. Label segmentation

对于每个实际存在的 user，只调用一次 segmentation，让 Python 一次处理该 user/action 下的所有 session：

```bash
python scripts/segment_ring_imu.py \
  --data-root "$DATA_ROOT" \
  --user "$user" \
  --action 0 \
  --input-kind spike-imu \
  --spike-root "$SPIKE_ROOT" \
  --boundary-mode label \
  --sampling-rate 200 \
  --output-root "$SEGMENT_ROOT"
```

这里不能再传：

```text
--gravity-removal-method
--low-pass-cutoff-hz
--madgwick-beta
--provisional
```

因为 `spike-imu` segmentation 直接读取已经处理好的 `(N,21)` 数据，CLI 会主动拒绝 gravity preprocessing 参数。方法信息从 SpikeIMU metadata 继承。

`label` 模式：

* 使用原始 timestamp label 文件确定边界。
* 不读取 Board 文件。
* 不需要 alignment offset。
* 保留完整 21 列 SpikeIMU。

### 4B. Aligned Board-event segmentation

Board 模式比 label 模式多一个 alignment 阶段。

#### 对每个 session 生成 SpikeIMU alignment offset

```bash
python scripts/align_ring_board.py \
  --data-root "$DATA_ROOT" \
  --user "$user" \
  --action 0 \
  --dataset-id "$session" \
  --input-kind spike-imu \
  --spike-root "$SPIKE_ROOT" \
  --offset-output-root "$OFFSET_ROOT" \
  --report-output-root "$ALIGNMENT_REPORT_ROOT" \
  --verification-output-root "$ALIGNMENT_VERIFICATION_ROOT"
```

该调用内部按 recording 加载所有匹配的 Board chunks；不需要 Bash 对 `${session}_board_0.gz`、`${session}_board_1.gz` 等逐个循环。

期望 offset：

```text
alignment/offsets/
└── user_0/
    └── action_0/
        ├── 0_ring_board_offset.txt
        ├── 1_ring_board_offset.txt
        └── ...
```

必须为每一个 session 成功产生 offset。缺失、失败、identity 不匹配或者基于旧 SpikeIMU 生成的 offset 都不能用于 Board segmentation。

#### 按 user 运行 Board segmentation

所有 offset 完成后：

```bash
python scripts/segment_ring_imu.py \
  --data-root "$DATA_ROOT" \
  --user "$user" \
  --action 0 \
  --input-kind spike-imu \
  --spike-root "$SPIKE_ROOT" \
  --boundary-mode aligned-board-events \
  --alignment-offset-root "$OFFSET_ROOT" \
  --sampling-rate 200 \
  --output-root "$SEGMENT_ROOT"
```

默认采用仓库参数：

```text
pre-press context:       0.2 秒
post-lift context:       0.2 秒
missing-event-policy:    skip
crossing-touch-policy:   accept_until_next_press
```

Board 模式除 `(N,21)` segmented SpikeIMU 外，还输出四列 target：

```text
valid_press
valid_lift
transient_press
transient_lift
```

仓库明确区分 `label` 与 `aligned-board-events` 两类 boundary source。

## 输出目录规划

为避免 8 个脚本互相覆盖，建议完全隔离：

```text
outputs/action0_pipeline/
├── raw/
│   ├── label/
│   │   ├── preprocessedIMU/
│   │   ├── spikeEncoding/custom-wavelet/
│   │   ├── segmentation/
│   │   └── logs/
│   └── aligned-board-events/
│       ├── preprocessedIMU/
│       ├── spikeEncoding/custom-wavelet/
│       ├── alignment/
│       │   ├── offsets/
│       │   ├── reports/
│       │   └── verification/
│       ├── segmentation/
│       └── logs/
├── low-pass/
├── madgwick/
└── xylo/
```

这种设计会让同一种 gravity method 的 label 和 Board 脚本各自重新执行预处理和编码，计算和磁盘开销较大，但优点是：

* 每个脚本真正独立。
* 不会复用配置不同或过期的中间产物。
* 不会因运行顺序不同产生结果差异。
* 可以直接删除一个组合的目录后完整重跑。

## 覆盖与失败策略

默认不传任何 overwrite 参数。现有结果会触发失败，防止混用旧数据。

通过：

```bash
OVERWRITE=1 ./03_lowpass_label.sh
```

统一启用对应参数：

```text
preprocess:
  --overwrite

encode:
  --overwrite

align:
  --overwrite-offset
  --overwrite-report
  --overwrite-verification

segment:
  --overwrite
  --overwrite-verification
```

脚本采用严格 fail-fast：

* 任一 Ring 预处理失败：停止。
* 任一 Spike encoding 失败：停止。
* Board 模式任一 session alignment 失败：停止，不开始 segmentation。
* 某个 user segmentation 失败：脚本返回非零。
* 不静默跳过缺失 Board chunk、缺失 label 或无效 metadata。

Alignment 失败时仓库会保留 report，但不会生成 offset；这可以用于后续定位具体 session。

## 每个脚本末尾的 QA

脚本结束前统计并核对：

1. `ring_0` recording 数量。
2. preprocessing summary 数量是否等于 recording 数量。
3. `spikeIMU.npy` 数量是否等于 recording 数量。
4. Board 模式的 offset 数量是否等于 recording 数量。
5. 每个 user 是否生成一个 `user/action0` segmentation summary。
6. `metadata.json`、timestamp sidecar 和输出矩阵是否全部存在。
7. 打印最终结果目录和失败日志目录。

日志建议：

```text
logs/
├── discovery.log
├── preprocess/
│   └── user_3_session_2.log
├── encode.log
├── alignment/
│   └── user_3_session_2.log
└── segmentation/
    └── user_3.log
```

按这个结构实现后，8 个脚本会覆盖 action `0` 下所有实际存在的 user、所有 session、唯一的 `ring_0` 输入，并在 Board segmentation 组合中使用每个 session 的全部 Board chunks。

