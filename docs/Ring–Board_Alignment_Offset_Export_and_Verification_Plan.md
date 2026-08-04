# Ring–Board Alignment Offset Export and Verification Plan

## 1. 目标

扩展当前 Ring IMU 与 Board 的 alignment pipeline。每个 recording 完成同步后，自动执行以下操作：

1. 从最终 alignment result 中提取固定时间差 `best_offset_us`；
2. 将时间差写入该 recording 独立对应的 TXT 文件；
3. 使用相同的 offset 将 Board press/lift events 映射到 Ring 时间轴；
4. 读取该 recording 对应的 label 文件；
5. 创建一张 alignment verification 图片；
6. 图片由六个并排的 10 秒片段组成；
7. 每个片段同时显示：

   * Ring transient score；
   * 对齐后的 Board press events；
   * 对齐后的 Board lift events；
   * label 文件中的标签。

核心原则：

```text
alignment algorithm
        ↓
SequenceAlignmentResult.best_offset_us
        ├── offset TXT
        └── alignment verification image
```

TXT 和 verification image 必须使用同一个 `best_offset_us`，不能分别估计 offset。

---

# 2. 时间映射约定

统一定义：

[
t_{\text{Ring}}
===============

t_{\text{Board}}+\Delta t
]

其中：

```text
Δt = best_offset_us
```

程序中的映射方式：

```python
aligned_board_timestamp_us = (
    board_timestamp_raw + best_offset_us
)
```

符号含义：

* `offset_us > 0`：对应的 Ring 时间戳晚于 Board 时间戳；
* `offset_us < 0`：对应的 Ring 时间戳早于 Board 时间戳。

所有输出文件和文档中必须明确写出：

```text
ring_timestamp_us = board_timestamp_us + offset_us
```

禁止只保存一个没有方向和单位的裸数字。

原始时间戳不得被原地修改。应新增派生字段：

```text
frame_timestamp_raw
aligned_ring_timestamp_us
aligned_ring_elapsed_s
```

---

# 3. 总体处理流程

完整 pipeline：

```text
加载 Ring 数据
→ 加载 Board 数据
→ 检测 Board press/lift events
→ 计算 Ring transient score
→ 检测 Ring transient peak regions
→ 生成候选 offsets
→ sequence-level event-to-peak matching
→ 获得 SequenceAlignmentResult
→ 检查 alignment success
→ 提取 best_offset_us
→ 写出 offset TXT
→ 将 Board events 映射到 Ring 时间轴
→ 读取对应 label 文件
→ 生成 6 × 10 秒 alignment verification 图片
→ 保存 diagnostics
```

只有满足以下条件时，才能写出最终 offset 和 verification image：

```python
alignment_result.success is True
np.isfinite(alignment_result.best_offset_us)
```

Alignment 失败时：

* 不生成有效 offset TXT；
* 不生成看似成功的 verification image；
* 保留 alignment report 和失败原因；
* CLI 返回非零状态码。

---

# 4. 输出目录结构

统一输出结构：

```text
outputs/
├── alignment/
│   ├── offsets/
│   │   └── user_0/
│   │       └── action_0/
│   │           └── 0_ring_board_offset.txt
│   └── reports/
│       └── user_0/
│           └── action_0/
│               └── 0_alignment_report.json
│
└── alignmentVerification/
    └── user_0/
        └── action_0/
            └── 0_alignment_verification.png
```

对于：

```text
user = user_0
action = 0
dataset_id = 0
Ring file = 0_ring_0.bin
```

生成：

```text
outputs/alignment/offsets/user_0/action_0/
0_ring_board_offset.txt
```

以及：

```text
outputs/alignmentVerification/user_0/action_0/
0_alignment_verification.png
```

使用 `user/action/dataset` 三级身份可以避免不同 recording 文件互相覆盖。

所有输出函数必须：

* 自动创建不存在的父目录；
* 默认拒绝覆盖已有文件；
* 只有 `overwrite=True` 时允许覆盖。

---

# 5. Offset 数据结构

新增模块：

```text
src/writingring/alignment_io.py
```

定义：

```python
@dataclass(frozen=True, slots=True)
class AlignmentOffset:
    user: str
    action: str
    dataset_id: int
    ring_stream: str
    offset_us: float
    alignment_model: str
    alignment_success: bool
    event_coverage_ratio: float | None
    matched_event_count: int | None
    total_valid_event_count: int | None

    @property
    def offset_ms(self) -> float:
        return self.offset_us / 1_000.0
```

第一版固定：

```text
alignment_model = constant_offset
ring_stream = ring_0
```

---

# 6. 从 Alignment Result 提取 Offset

实现：

```python
def extract_alignment_offset(
    result: SequenceAlignmentResult,
    *,
    user: str,
    action: str,
    dataset_id: int,
    ring_stream: str = "ring_0",
) -> AlignmentOffset:
    ...
```

函数职责：

1. 验证输入是 `SequenceAlignmentResult`；
2. 检查 `result.success`；
3. 检查 `best_offset_us` 是有限值；
4. 从 `result.report` 提取：

   * event coverage；
   * matched event count；
   * total valid event count；
5. 返回结构化的 `AlignmentOffset`。

Alignment 失败时：

```python
raise AlignmentOffsetExportError(
    "alignment did not succeed; no offset file was written"
)
```

不得在失败状态下输出一个可能被后续程序误用的 offset。

---

# 7. Offset TXT 路径接口

实现：

```python
def build_alignment_offset_path(
    output_root: Path,
    *,
    user: str,
    action: str,
    dataset_id: int,
) -> Path:
    ...
```

默认：

```python
output_root = Path("outputs/alignment/offsets")
```

示例返回：

```text
outputs/alignment/offsets/
user_0/action_0/
0_ring_board_offset.txt
```

输入验证：

* `user` 必须是非空字符串；
* `action` 必须是非空字符串；
* `dataset_id >= 0`；
* 拒绝或转义 `/`、`\` 等不安全路径字符。

---

# 8. Offset TXT 格式

使用 `key=value` 格式。

示例：

```text
user=user_0
action=0
dataset_id=0
ring_stream=ring_0
alignment_model=constant_offset
timestamp_unit=microseconds
time_mapping=ring_timestamp_us = board_timestamp_us + offset_us
offset_us=-238451.750000
offset_ms=-238.451750
alignment_success=true
event_coverage_ratio=0.875000
matched_event_count=35
total_valid_event_count=40
```

核心字段：

```text
offset_us
timestamp_unit
time_mapping
alignment_success
```

后续程序始终使用 `offset_us`，`offset_ms` 只用于人工查看。

---

# 9. Offset TXT 写入和读取

实现：

```python
def write_alignment_offset_txt(
    offset: AlignmentOffset,
    *,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    ...
```

要求：

* UTF-8 编码；
* 固定小数精度；
* 自动创建 parent directory；
* 默认拒绝覆盖；
* 使用临时文件写入后原子替换；
* 返回最终输出路径。

同时实现：

```python
def read_alignment_offset_txt(
    path: Path,
) -> AlignmentOffset:
    ...
```

读取时验证：

* 必需字段存在；
* `timestamp_unit == "microseconds"`；
* `alignment_model == "constant_offset"`；
* `alignment_success == true`；
* `offset_us` 是有限值；
* user、action、dataset ID 与当前 recording 一致。

---

# 10. 统一 Offset 应用接口

实现：

```python
def apply_board_to_ring_offset(
    board_timestamps_us: np.ndarray,
    *,
    offset_us: float,
) -> np.ndarray:
    return board_timestamps_us + offset_us
```

所有后续代码必须调用该函数，避免不同模块分别使用：

```text
board + offset
```

和：

```text
board - offset
```

造成符号不一致。

---

# 11. Verification 模块

新增：

```text
src/writingring/alignment_verification.py
```

定义配置：

```python
@dataclass(frozen=True, slots=True)
class AlignmentVerificationConfig:
    segment_duration_s: float = 10.0
    segment_count: int = 6
    verification_start_s: float = 0.0
    label_time_domain: str = "shared"
    show_smoothed_transient: bool = False
    output_dpi: int = 200
    overwrite: bool = False
```

定义结果：

```python
@dataclass(frozen=True, slots=True)
class AlignmentVerificationResult:
    output_path: Path
    displayed_start_s: float
    displayed_stop_s: float
    ring_sample_count_displayed: int
    press_count_displayed: int
    lift_count_displayed: int
    label_count_displayed: int
    press_count_outside: int
    lift_count_outside: int
    label_count_outside: int
    warnings: tuple[str, ...]
```

---

# 12. Verification 图片路径接口

实现：

```python
def build_alignment_verification_path(
    output_root: Path,
    *,
    user: str,
    action: str,
    dataset_id: int,
) -> Path:
    ...
```

默认：

```python
output_root = Path("outputs/alignmentVerification")
```

示例：

```text
outputs/alignmentVerification/
user_0/action_0/
0_alignment_verification.png
```

要求：

* 自动创建 parent directory；
* 默认拒绝覆盖；
* 图片标题包含 user、action 和 dataset ID。

---

# 13. Ring 时间轴

以 Ring reconstructed timestamp 的第一个有效值为统一起点：

```python
ring_start_timestamp_us = ring_timestamps_us[0]

ring_elapsed_s = (
    ring_timestamps_us - ring_start_timestamp_us
) / 1_000_000.0
```

Board event 应先应用 offset：

```python
aligned_board_timestamp_us = (
    board_event_timestamp_us + best_offset_us
)

board_event_elapsed_s = (
    aligned_board_timestamp_us
    - ring_start_timestamp_us
) / 1_000_000.0
```

所有 transient、press、lift 和 label 必须绘制在同一个 Ring elapsed-time reference 上。

---

# 14. Verification 图片布局

最终图片固定为：

```text
1 row × 6 columns
```

每个 panel 为 10 秒：

```text
Panel 1:  0–10 s
Panel 2: 10–20 s
Panel 3: 20–30 s
Panel 4: 30–40 s
Panel 5: 40–50 s
Panel 6: 50–60 s
```

默认参数：

```python
segment_duration_s = 10.0
segment_count = 6
verification_start_s = 0.0
```

建议尺寸：

```python
figure, axes = plt.subplots(
    1,
    6,
    figsize=(30, 5),
    sharey=True,
    layout="constrained",
)
```

要求：

* 六个 panel 并排；
* 每个 panel 正好覆盖 10 秒；
* 所有 panel 使用相同 y-axis limits；
* x 轴显示从 Ring recording start 起的 elapsed seconds；
* panel 标题分别为 `0–10 s`、`10–20 s` 等；
* legend 只显示在第一个 panel。

---

# 15. Transient Score 显示

使用当前 alignment 中计算出的 transient score：

```python
transient_score = compute_transient_score(
    ring_dataframe,
    signal_columns=SIGNAL_COLUMNS,
)
```

默认显示原始 transient score。

如果 `show_smoothed_transient=True`，可以同时叠加 peak detection 使用的 smoothed score，但必须：

* 原始 score 为主要曲线；
* smoothed score 为较细辅助曲线；
* legend 明确区分。

六个 panel 使用相同 y limits。

为避免极端峰值压缩其他信号，可使用：

```python
y_max = np.quantile(transient_score, 0.995) * 1.1
```

但不能修改或截断原始数据，只限制显示范围。

---

# 16. Press/Lift Event 显示

从 Board event table 中读取：

```text
event_type = press
event_type = lift
```

先应用：

```python
aligned_event_timestamp_us = (
    frame_timestamp_raw + best_offset_us
)
```

绘图建议：

```text
Press:
    红色实线
    label = Board press

Lift:
    橙色虚线
    label = Board lift
```

Matched 和 unmatched event 可通过透明度区分：

```text
matched:
    较高 opacity
    较粗 linewidth

unmatched:
    较低 opacity
    较细 linewidth
```

但是 press/lift 的颜色和线型必须保持语义一致。

---

# 17. Label 文件解析

实现：

```python
@dataclass(frozen=True, slots=True)
class AlignmentLabel:
    timestamp_us: float
    label: str
    source_line_number: int
```

解析接口：

```python
def load_alignment_labels(
    path: Path,
) -> tuple[AlignmentLabel, ...]:
    ...
```

Label 路径解析接口：

```python
def resolve_alignment_label_path(
    recording_directory: Path,
    *,
    dataset_id: int,
    explicit_label_path: Path | None = None,
) -> Path:
    ...
```

查找顺序：

```text
1. explicit_label_path
2. {dataset_id}_label.txt
3. {dataset_id}_timestamp.txt
```

例如：

```text
0_label.txt
0_timestamp.txt
```

第一版要求 label 文件存在。若不存在：

```python
raise AlignmentVerificationError(
    "no label or timestamp marker file is available"
)
```

不生成缺少标签但看似完整的 verification image。

---

# 18. Label 文件格式

建议支持：

```text
timestamp label
```

例如：

```text
1720476263355268 A
1720476264859021 B
1720476267021453 C
```

解析规则：

1. 忽略空行；
2. 忽略以 `#` 开头的注释；
3. 第一列必须是有限数值 timestamp；
4. 其余文本作为完整 label；
5. label 不允许为空；
6. 保留 label 原始大小写；
7. 保存源文件行号；
8. malformed line 必须报告行号，不能静默忽略。

---

# 19. Label 时间域

配置：

```python
label_time_domain: Literal[
    "ring",
    "board",
    "shared",
]
```

转换规则：

## Ring 或 shared 时间域

```python
aligned_label_timestamp_us = label_timestamp_us
```

## Board 时间域

```python
aligned_label_timestamp_us = (
    label_timestamp_us + best_offset_us
)
```

最终转换为 Ring elapsed time：

```python
label_elapsed_s = (
    aligned_label_timestamp_us
    - ring_start_timestamp_us
) / 1_000_000.0
```

第一版默认：

```python
label_time_domain = "shared"
```

但该假设必须显示在：

* 图片副标题；
* alignment report；
* verification result metadata。

---

# 20. Label 标记方式

每个 label 在对应 panel 中显示为：

```text
灰色竖直点线
+
顶部旋转文字
```

示例实现：

```python
axis.axvline(
    label_elapsed_s,
    linestyle=":",
    linewidth=1.0,
)

axis.text(
    label_elapsed_s,
    label_height,
    label.label,
    rotation=90,
    transform=axis.get_xaxis_transform(),
)
```

相邻 label 过近时，在两个高度间交替：

```text
height level 1
height level 2
```

避免文字完全重叠。

Panel 边界统一使用：

```text
前五段：[start, stop)
最后一段：[50, 60]
```

保证同一个 label 不会在相邻 panel 中重复。

---

# 21. 短 Recording 和长 Recording

## Recording 少于 60 秒

仍然生成六个 panel。

例如 recording 长度为 52 秒：

```text
Panel 1–5：正常显示
Panel 6：显示 50–52 秒
52–60 秒保持为空
```

Panel 6 中注明：

```text
Recording ends at 52.0 s
```

## Recording 超过 60 秒

第一版只显示：

```text
0–60 s
```

在图片或日志中注明：

```text
Verification image shows the first 60 seconds only.
```

后续可通过：

```python
verification_start_s
```

支持显示其他区间。

---

# 22. Verification 主接口

实现：

```python
def create_alignment_verification_figure(
    *,
    ring_dataframe: pd.DataFrame,
    ring_timestamps_us: np.ndarray,
    transient_score: np.ndarray,
    board_events: pd.DataFrame,
    alignment_result: SequenceAlignmentResult,
    labels: Sequence[AlignmentLabel],
    label_source_path: Path,
    output_path: Path,
    config: AlignmentVerificationConfig,
) -> AlignmentVerificationResult:
    ...
```

职责：

1. 验证 alignment 成功；
2. 验证 `best_offset_us` 有效；
3. 验证 Ring timestamps 和 transient score 长度一致；
4. 验证 Ring timestamps 单调；
5. 将 Board events 映射到 Ring 时间；
6. 将 labels 映射到 Ring 时间；
7. 创建六个并排的 10 秒 panel；
8. 绘制 transient score；
9. 绘制 press events；
10. 绘制 lift events；
11. 绘制 label markers；
12. 保存非空 PNG；
13. 返回显示范围和图内/图外 event 统计。

该函数不得：

* 修改输入 DataFrame；
* 修改原始 timestamp；
* 重新估计 offset；
* 对 offset 重复应用两次。

---

# 23. 图片标题

主标题：

```text
Ring–Board Alignment Verification
user_0 / action 0 / dataset 0
```

副标题：

```text
ring_time = board_time + offset
offset = -238451.750 us (-238.452 ms)
label source = 0_timestamp.txt
label time domain = shared
```

轴标签：

```text
Y-axis: Transient score
X-axis: Elapsed time from Ring start (s)
```

---

# 24. Notebook 集成

修改：

```text
notebooks/align_ring_board.ipynb
```

在最终 alignment 成功后：

```python
offset_record = extract_alignment_offset(
    alignment_result,
    user=USER,
    action=ACTION,
    dataset_id=DATASET_ID,
    ring_stream="ring_0",
)

offset_path = build_alignment_offset_path(
    Path("outputs/alignment/offsets"),
    user=USER,
    action=ACTION,
    dataset_id=DATASET_ID,
)

write_alignment_offset_txt(
    offset_record,
    output_path=offset_path,
    overwrite=OVERWRITE_OUTPUTS,
)
```

然后：

```python
label_path = resolve_alignment_label_path(
    recording_directory,
    dataset_id=DATASET_ID,
    explicit_label_path=LABEL_PATH,
)

labels = load_alignment_labels(label_path)

verification_path = build_alignment_verification_path(
    Path("outputs/alignmentVerification"),
    user=USER,
    action=ACTION,
    dataset_id=DATASET_ID,
)

verification_result = create_alignment_verification_figure(
    ring_dataframe=ring_dataframe,
    ring_timestamps_us=ring_timestamp_reconstructed,
    transient_score=transient_score,
    board_events=selected_board_events,
    alignment_result=alignment_result,
    labels=labels,
    label_source_path=label_path,
    output_path=verification_path,
    config=AlignmentVerificationConfig(
        segment_duration_s=10.0,
        segment_count=6,
        verification_start_s=0.0,
        label_time_domain=LABEL_TIME_DOMAIN,
        overwrite=OVERWRITE_OUTPUTS,
    ),
)
```

Notebook 最终输出：

```text
Alignment offset:
-238451.750 us

Offset file:
outputs/alignment/offsets/user_0/action_0/
0_ring_board_offset.txt

Verification image:
outputs/alignmentVerification/user_0/action_0/
0_alignment_verification.png

Displayed:
6 × 10-second segments
Press events: 20
Lift events: 20
Labels: 26
```

---

# 25. CLI 支持

Alignment CLI 增加：

```text
--offset-output-root outputs/alignment/offsets
--verification-output-root outputs/alignmentVerification
--label-path PATH
--label-time-domain {ring,board,shared}
--verification-start-seconds 0
--overwrite-offset
--overwrite-verification
```

成功时输出 offset TXT 和 verification PNG。

失败时：

* 不创建 offset TXT；
* 不创建 verification PNG；
* 输出 alignment failure reason；
* 保留 report JSON；
* 返回非零 exit code。

---

# 26. 公共 API

修改：

```text
src/writingring/__init__.py
```

导出：

```python
AlignmentOffset
AlignmentOffsetExportError
extract_alignment_offset
build_alignment_offset_path
write_alignment_offset_txt
read_alignment_offset_txt
apply_board_to_ring_offset

AlignmentLabel
AlignmentVerificationConfig
AlignmentVerificationResult
AlignmentVerificationError
resolve_alignment_label_path
load_alignment_labels
build_alignment_verification_path
create_alignment_verification_figure
```

---

# 27. 测试计划

新增：

```text
tests/test_alignment_io.py
tests/test_alignment_verification.py
```

## Offset 测试

覆盖：

* 正 offset；
* 负 offset；
* `offset_ms == offset_us / 1000`；
* 文件名包含 user/action/dataset；
* parent directory 自动创建；
* `success=False` 时拒绝导出；
* `NaN` 和 `inf` offset 时拒绝导出；
* 默认拒绝覆盖；
* `overwrite=True` 允许覆盖；
* TXT write/read round-trip；
* recording identity mismatch 时报错；
* 符号方向正确。

测试例：

```text
board_timestamp = 1,000,000 us
offset = 250,000 us
aligned Ring timestamp = 1,250,000 us
```

---

## Verification 路径测试

覆盖：

* 自动创建 `user/action` 目录；
* dataset 0 输出：
  `0_alignment_verification.png`；
* 默认拒绝覆盖；
* `overwrite=True` 允许覆盖；
* PNG 文件存在且非空。

---

## Figure 布局测试

覆盖：

* 正好六个主要 axes；
* 每个 panel 为 10 秒；
* 总范围为 0–60 秒；
* 六个 panel 共用相同 y limits；
* recording 少于 60 秒仍生成六个 panel。

---

## Offset 应用测试

构造：

```text
Board event timestamp = 10,000,000 us
offset = 500,000 us
Ring start = 9,000,000 us
```

验证事件绘制位置：

```text
1.5 s
```

确保没有：

* 漏加 offset；
* 减错 offset；
* 重复加 offset。

---

## Press/Lift 测试

覆盖：

* press 和 lift 使用不同 linestyle；
* press/lift 出现在正确 panel；
* 60 秒外 event 不绘制；
* 图外 event 被统计；
* matched/unmatched metadata 正确保留。

---

## Label 测试

覆盖：

* label 文件成功解析；
* label 出现在正确 panel；
* panel 边界 label 只出现一次；
* `label_time_domain="board"` 时应用 offset；
* `label_time_domain="ring"` 时不应用 offset；
* malformed line 报告源文件行号；
* label 文件不存在时明确失败；
* label 文字大小写保持不变。

---

# 28. 文件修改清单

新增：

```text
src/writingring/alignment_io.py
src/writingring/alignment_verification.py
tests/test_alignment_io.py
tests/test_alignment_verification.py
docs/ALIGNMENT_OUTPUTS.md
```

修改：

```text
src/writingring/__init__.py
notebooks/align_ring_board.ipynb
README.md
相关 alignment CLI script
```

原则上不修改：

```text
vendor/
data/
data_sample/
```

核心 matching 算法：

```text
src/writingring/event_alignment.py
```

原则上不需要改变。它继续负责：

* Board event detection；
* transient peak detection；
* offset candidate generation；
* sequence matching；
* 生成 `SequenceAlignmentResult`。

新的 I/O 和 verification 模块只消费 alignment result。

---

# 29. 实施顺序

1. 固定时间映射约定：
   `ring = board + offset`。
2. 新建 `alignment_io.py`。
3. 实现 `AlignmentOffset`。
4. 实现 offset 提取函数。
5. 实现 offset TXT path/write/read。
6. 实现统一 offset application helper。
7. 新建 `alignment_verification.py`。
8. 实现 label path resolution。
9. 实现 label parser。
10. 实现 verification path builder。
11. 实现 Board events 和 labels 到 Ring 时间轴的转换。
12. 实现六个并排的 10 秒 panels。
13. 绘制 transient score。
14. 叠加 press/lift events。
15. 叠加 label markers。
16. 保存 verification PNG。
17. 在 `__init__.py` 中导出接口。
18. 集成到 alignment Notebook。
19. 集成到 CLI。
20. 增加 offset 和 verification tests。
21. 运行现有 alignment tests。
22. 对样例 recording 生成 TXT 和 PNG 并人工检查。

---

# 30. 完成标准

实现完成后必须满足：

* 每个成功同步的 recording 生成一个唯一 offset TXT；
* 每个成功同步的 recording 生成一个唯一 verification PNG；
* TXT 和 PNG 使用完全相同的 `best_offset_us`；
* 明确使用：
  `ring_timestamp = board_timestamp + offset`；
* offset 单位固定为 microseconds；
* verification image 固定包含六个并排的 10 秒 panel；
* 图片默认覆盖 Ring recording 的前 60 秒；
* 每个 panel 显示 transient score；
* 每个 panel显示同步后的 Board press/lift events；
* 图片显示对应 label 文件中的标签；
* label 时间域明确记录；
* 原始 Ring、Board 和 label timestamp 均不被修改；
* 输出目录不存在时自动创建；
* alignment 失败时不生成伪成功输出；
* dataset 0 示例输出为：

```text
outputs/alignment/offsets/user_0/action_0/
0_ring_board_offset.txt

outputs/alignmentVerification/user_0/action_0/
0_alignment_verification.png
```

* TXT 中的 offset 与图片副标题中的 offset 完全一致；
* 新增测试和现有 alignment tests 全部通过。
