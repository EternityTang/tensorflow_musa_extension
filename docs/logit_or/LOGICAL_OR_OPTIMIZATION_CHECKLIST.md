# LogicalOr 优化 Checklist 与基线记录指南

这份文档用于在真正改 `tensorflow_musa_extension` 里的 `logical_or` 之前，先把：

- 当前实现的理解收敛成一份 checklist
- 未优化版本的整网性能记录下来
- 后续每次改完后，按同一套口径回归对比

建议和下面两个文件一起使用：

- `run_or_pruned_graph_whole_model.sh`
- `MUDNN_IMPL_PROBE_GUIDE.md`

## 1. 当前实现结论

`logical_or` 当前在 `tensorflow_musa_extension` 中的实现特点：

- TensorFlow kernel 侧逻辑很薄
- 核心流程是 `BCast -> allocate_output -> CreateMTensor -> muDNN Binary(LOGICAL_OR).Run`
- 真正计算下沉给 muDNN `Binary`
- 当前能看到 muDNN 公开接口，但看不到 `BinaryImpl/_BinaryRun` 的内部实现

这意味着：

- 第一波优化重点应该放在插件侧固定开销
- 不要一开始就把希望全部压在 muDNN 黑盒内部

## 2. 推荐优化优先级

### P0：先做的

- 先记录未优化基线
- 给 `logical_or` 补 `IsExpensive() = false`
- 加 same-shape fast path，shape 一样时跳过 `BCast`
- 补 timing/trace，尽量分清 host 包装时间和 `Binary::Run` 时间

### P1：第二批评估

- 评估 `CreateMTensor(dim, stride)` 路径
- 评估 broadcast-view 表达方式
- 对比 same-shape 和 broadcast case 的收益差异

### P2：谨慎评估

- `forward_input_or_allocate_output`
- output 复用 input buffer

这部分必须满足两个前提：

- TensorFlow 框架层允许 forward
- muDNN `Binary::Run` 自身支持 `out == in0` 或 `out == in1`

如果只满足前者，不满足后者，结果仍然可能错误。

### P3：更大改动

- `Equal + LogicalOr` 链路联动优化
- OR 链树形归约
- pattern fusion / custom op

这类收益可能更大，但改动范围也更大，不建议作为第一刀。

## 3. 详细 Checklist

### 3.1 改代码前

- 确认 `tensorflow_musa_extension` 当前分支和 commit
- 确认 `tf_test_model` 当前分支和 commit
- 确认 docker 容器、conda 环境、插件路径固定
- 用同一组 `WARMUP_ROUNDS / INFERENCE_ROUNDS / PROFILE_WARMUP_ROUNDS`
- 先记录一次未优化 baseline

### 3.2 第一轮改动建议

- `logical_or` 增加 `IsExpensive() override { return false; }`
- 为 same-shape case 加快速路径
- 避免所有 case 都无条件构造 `BCast`
- 不改语义，不碰 output forward

### 3.3 第一轮验证项

- 单测正确性通过
- whole-model accuracy compare 通过
- whole-model inference-only 稳定
- `profile-ops` 中 `LogicalOr` 的 count 不变
- `profile-ops` 中 `LogicalOr` 的 total/avg duration 下降
- 观察 `Equal`、整网总耗时是否有联动变化

### 3.4 第二轮改动建议

- 评估 `stride` 版本的 `SetNdInfo`
- 评估 broadcast-view 是否能减少额外 reshape/broadcast 包装开销
- 对 same-shape / broadcast case 分开测

### 3.5 第二轮验证项

- 与第一轮相同
- 额外关注 broadcast case 是否稳定
- 额外关注 trace 中 `LogicalOr` 的 device time 是否真的下降

### 3.6 output forward 专项前置条件

- 明确 TensorFlow `forward_input_or_allocate_output` 的条件
- 明确 muDNN `Binary::Run` 是否支持 alias/in-place
- 设计专项 case：
  - `out == in0`
  - `out == in1`
  - same-shape
  - broadcast
  - 多 consumer
  - 连续链式调用

如果内部实现不可见，建议把 output forward 放后面，不要作为第一轮优化。

## 4. 先测未优化基线

建议在 docker 容器内、`tf26` 环境中执行。

```bash
docker exec -it zhaoye bash
conda activate tf26
cd /workspace
```

### 4.1 可直接运行的 baseline 命令

下面这段命令会：

- 记录当前两个 repo 的 branch/commit
- 运行 `run_or_pruned_graph_whole_model.sh`
- 把控制台输出写进 log
- 自动生成一份 markdown 基线记录
- 自动把最新 trace 里的 `LogicalOr` / `Equal` 摘要写进 markdown

```bash
cd /workspace/tf_test_model

mkdir -p logs/logical_or_baseline
ts="$(date +%Y-%m-%d-%H.%M.%S)"
BASE_DIR="logs/logical_or_baseline/${ts}"
mkdir -p "${BASE_DIR}"

REPORT_MD="${BASE_DIR}/baseline_report.md"
CONSOLE_LOG="${BASE_DIR}/baseline_console.log"
PLUGIN_PATH="/workspace/tensorflow_musa_extension/build/libmusa_plugin.so"

{
  echo "# LogicalOr Baseline Report"
  echo
  echo "- Date: ${ts}"
  echo "- tf_test_model branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "- tf_test_model commit: $(git rev-parse --short HEAD)"
  echo "- tensorflow_musa_extension branch: $(git -C /workspace/tensorflow_musa_extension rev-parse --abbrev-ref HEAD)"
  echo "- tensorflow_musa_extension commit: $(git -C /workspace/tensorflow_musa_extension rev-parse --short HEAD)"
  echo "- Plugin path: ${PLUGIN_PATH}"
  echo "- Warmup rounds: 5"
  echo "- Inference rounds: 500"
  echo "- Profile warmup rounds: 5"
  echo "- Focus op: LogicalOr"
  echo
  echo "## Commands"
  echo
  echo '```bash'
  echo 'RUN_ACCURACY_COMPARE=1 WARMUP_ROUNDS=5 INFERENCE_ROUNDS=500 PROFILE_WARMUP_ROUNDS=5 \\'
  echo 'FOCUS_OP_TYPE=LogicalOr ./run_or_pruned_graph_whole_model.sh \\'
  echo '  /workspace/tensorflow_musa_extension/build/libmusa_plugin.so'
  echo '```'
  echo
  echo "## Console Log"
  echo
  echo "- ${CONSOLE_LOG}"
  echo
} > "${REPORT_MD}"

RUN_ACCURACY_COMPARE=1 \
WARMUP_ROUNDS=5 \
INFERENCE_ROUNDS=500 \
PROFILE_WARMUP_ROUNDS=5 \
FOCUS_OP_TYPE=LogicalOr \
./run_or_pruned_graph_whole_model.sh "${PLUGIN_PATH}" | tee "${CONSOLE_LOG}"

TRACE_DIR="$(find logs/graph_inference prunedGraph/logs/graph_inference -maxdepth 1 -type d -name '*_trace' 2>/dev/null | sort | tail -n 1)"

python3 - "${TRACE_DIR}" "${REPORT_MD}" <<'PY'
import json
import sys
from pathlib import Path

trace_dir = Path(sys.argv[1]) if sys.argv[1] else None
report_md = Path(sys.argv[2])

lines = []
lines.append("## Trace Summary")
lines.append("")
lines.append(f"- Trace dir: {trace_dir}" if trace_dir else "- Trace dir: <not found>")
lines.append("")

if trace_dir and trace_dir.exists():
    files = sorted(trace_dir.glob("op_stats_MUSA_*.json"))
    if files:
        path = files[-1]
        data = json.loads(path.read_text())
        ops = data.get("operators", [])
        for focus in ("LogicalOr", "Equal"):
            focus_ops = [op for op in ops if op.get("op_type") == focus]
            durations = [float(op.get("duration_ms", 0.0)) for op in focus_ops]
            total = sum(durations)
            avg = total / len(durations) if durations else 0.0
            lines.append(f"### {focus}")
            lines.append("")
            lines.append(f"- Count: {len(focus_ops)}")
            lines.append(f"- Total duration (ms): {total:.6f}")
            lines.append(f"- Avg duration (ms): {avg:.6f}")
            lines.append("")
        lines.append("### op_stats file")
        lines.append("")
        lines.append(f"- {path}")
        lines.append("")
    else:
        lines.append("- No `op_stats_MUSA_*.json` found.")
        lines.append("")
else:
    lines.append("- Trace dir missing.")
    lines.append("")

report_md.write_text(report_md.read_text() + "\n".join(lines))
PY

echo
echo "Baseline report: ${REPORT_MD}"
echo "Baseline console log: ${CONSOLE_LOG}"
echo "Latest trace dir: ${TRACE_DIR}"
```

## 5. 推荐记录内容

每次做优化前后，至少保留这些信息：

- `tf_test_model` branch/commit
- `tensorflow_musa_extension` branch/commit
- plugin 路径
- batch size
- warmup rounds
- inference rounds
- profile warmup rounds
- `LogicalOr` count
- `LogicalOr total duration`
- `LogicalOr avg duration`
- `Equal total duration`
- inference-only 总体吞吐或单轮耗时
- accuracy compare 是否通过

## 6. 单测命令

改完 `tensorflow_musa_extension` 后，建议先在 container 里跑逻辑算子单测，再跑整网。

### 6.1 逻辑二元算子快捷脚本

```bash
cd /workspace/tensorflow_musa_extension
bash test/run_logical_binary_tests.sh
```

### 6.2 分别跑单个测试文件

```bash
cd /workspace/tensorflow_musa_extension
PYTHONPATH=test python3 test/ops/logicalor_op_test.py
PYTHONPATH=test python3 test/ops/logicaland_op_test.py
```

## 7. 结果怎么比较

建议对比三层结果：

### 7.1 先看 `LogicalOr`

- count 是否一致
- total duration 是否下降
- avg duration 是否下降

### 7.2 再看相关链路

- `Equal` 是否一起下降
- 整条 `Equal + LogicalOr` 链是否有联动收益

### 7.3 最后看整网

- inference-only 平均耗时是否下降
- 波动是否可接受
- accuracy compare 是否保持一致

## 8. 我当前最推荐你先试的 3 个改动

按优先级排序：

1. `IsExpensive() = false`
2. same-shape fast path，跳过 `BCast`
3. 补 timing/trace，把 host 包装和 `Binary::Run` 时间拆开

原因：

- 风险低
- 改动小
- 最容易快速判断值不值得继续深挖

## 9. 暂时不建议作为第一刀的改动

- output forward
- input/output buffer alias
- 自定义 `.mu` kernel
- 整图级 fusion

不是说这些方向没价值，而是：

- 当前证据还不足
- 改动成本更高
- debug 成本更高

先把第一轮低风险优化做完，通常更稳。
