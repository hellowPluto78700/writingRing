# 全局 segment 长度分析与固定长度 padding

`scripts/analyze_segment_lengths.py` 和 `scripts/pad_segmented_imu.py` 只读取已完成的 variable-length SpikeIMU segmentation 输出；它们不读取任何 Ring/Board 原始数据、timestamp label、alignment offset 或 gravity-removal 配置。

输入根下每个 user/action package 必须包含：

```text
<user>_action_<action>_spikeIMU.npy          # (sample_count, 21)
<user>_action_<action>_labels.npy            # (segment_count,)
<user>_action_<action>_segment_offsets.npy   # (segment_count + 1,)
<user>_action_<action>_segment_lengths.npy   # (segment_count,)
<user>_action_<action>_segmentation_summary.json
```

summary 必须声明 `input_kind=spike-imu`、`feature_schema=signed_wavelet_events_plus_imu_v1` 和 `channel_count=21`。旧的 `rawIMU`、6 通道或 9 通道 segmentation 输出不属于此 padding loader 的输入契约。

```bash
conda run --no-capture-output -n writingring-viz \
  python scripts/analyze_segment_lengths.py \
  --input-root outputs/action0_pipeline/low-pass/label/segmentation
```

默认输出到 `input_root/padding_analysis/`，包含 JSON 机器报告、全局统计和候选长度 CSV、可追溯异常 CSV，以及 histogram/ECDF Matplotlib 图。JSON 记录每个 lengths 文件的相对路径、segment 数和 maximum，因此 padding 会拒绝使用已过期的分析报告。

报告提供三个互不混淆的 target：

- `pure_padding` 是全局 maximum（可用 `--round-to` 向上取整），不会跳过 segment。
- `p99` 是向上取整后的 P99；超过该长度的 segment 会被明确跳过。
- `balanced` 是满足 `--minimum-coverage` 的最小候选值；超过该长度的 segment 同样会跳过。

检查 `segment_length_outliers.csv` 后，选择合适的报告推荐创建独立输出根：

```bash
conda run --no-capture-output -n writingring-viz \
  python scripts/pad_segmented_imu.py \
  --input-root outputs/action0_pipeline/low-pass/label/segmentation \
  --analysis-report outputs/action0_pipeline/low-pass/label/segmentation/padding_analysis/segment_length_analysis.json \
  --recommendation pure-padding
```

默认输出为独立的 padded root。每个 user/action 会生成 `(segment_count, target_length, 21)` 的 `*_paddedSpikeIMU.npy`、对应的 retained labels、`valid_lengths`、boolean `valid_mask`、padding manifest 和 summary。Board-assisted 输入会额外生成 `(segment_count, target_length, 4)` boolean target，padding 区域恒为 `False`。Manifest 对每个输入 segment 都保留一行，`exported=false` 和 `skip_reason=length_exceeds_target` 表示被跳过；`output_segment_index` 对应 padded arrays 的索引。

Padding 固定在右侧，超过 target 的 segment 不会截断或重采样，而是明确跳过且不会修改输入。所有包先完成验证，再写入临时根目录并一次性发布；失败不会留下部分输出。

## 在 Action-0 Bash pipeline 中自动执行

`scripts/action0_pipeline/` 下的 8 个入口都通过共享的 `_common.bash`
自动执行以下顺序：完成 variable-length segmentation，扫描整棵
segmentation root，写出分析报告，再根据报告发布固定长度 padding 输出。
默认配置为满足至少 99% segment coverage 的最小候选长度：

```bash
PADDING_COVERAGE=0.99 \
PADDING_RECOMMENDATION=balanced \
scripts/action0_pipeline/03_lowpass_label.sh
```

可在 Bash 环境中覆盖：

- `PADDING_COVERAGE`：传给 `--minimum-coverage`，默认 `0.99`。
- `PADDING_RECOMMENDATION`：`balanced`、`p99` 或 `pure-padding`，默认
  `balanced`。
- `PADDING_ROUND_TO`、`PADDING_VALUE`：传给对应的分析/补齐 CLI。
- `PADDING_ANALYSIS_DIR`、`PADDING_OUTPUT_ROOT`：覆盖报告和 padded root
  的路径。

默认 `balanced` 使用 `DEFAULT_CANDIDATE_LENGTHS` 中满足 coverage 的最小
候选值；`p99` 使用经验 P99 向上取整后的长度。两种选择都不会截断超长
segment，而是把它们记录为 skipped。`OVERWRITE=1` 同时允许覆盖分析报告
和 padded 输出。

## 与 upstream 的关系

upstream `vendor/WritingRing/` 没有这类 completed-segmentation 输出解析器，因此这里没有可复用的 upstream 数据加载逻辑。实现仅验证本项目 `segmentation.py` 和 `board_event_segmentation.py` 的已发布 `.npy`/CSV schema；唯一有意的兼容性选择是将 `*_segments.csv` 视为可选 provenance 元数据，缺失时仍能以 labels 和 arrays 安全完成 padding，但 outlier/manifest 中的 `dataset_id` 将为空。
