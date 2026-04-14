# MUSA 算子优化流程

本文档总结基于真实场景的算子优化方法论，以 Abs/AddV2/LogicalOr 为例。

---

## 一、核心原则

```
┌─────────────────────────────────────────────────────────────────┐
│  以真实场景为导向                                                 │
│  - 优先用 prunedGraph 等真实网络作为优化目标                       │
│  - wukong 等通用测试集作为补充回归                                 │
│  - 围绕真实出现的 shape 做定向优化                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  一次只优化一个算子                                               │
│  - 多算子同时改动难以归因                                         │
│  - 每个算子独立走完整闭环                                         │
│  - 确保每个改动都有明确收益                                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  粗测先行，精测跟进                                               │
│  - 时间宏粗测：快速判断方向性收益                                  │
│  - Profiler 精测：量化收益来源                                    │
│  - 回归验证：确保无副作用                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、整体流程图

```
┌─────────────────────────────────────────────────────────────────┐
│  阶段 1：建立基线                                                 │
│  - 固定环境（机器/驱动/batch/seed/build）                         │
│  - 记录修改前：单测、精度、整网时间、profile                        │
│  - 基线文件统一留档                                               │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 2：Shape 盘点                                               │
│  - 以真实网络为准，列出目标 op 清单                                │
│  - 在 Compute() 临时打印真实 shape                                │
│  - 形成 Shape 归一表：op/node/次数/dtype/shape/broadcast          │
│  - 筛选热点 shape（按次数和耗时）                                  │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 3：构造针对性测试                                            │
│  - 基于真实 shape 扩展单测                                        │
│  - 补充边界 case（空tensor/标量/broadcast/特殊输入）              │
│  - 验收标准：CPU vs MUSA 一致                                     │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 4：粗测 → 精测                                              │
│  - 粗测：时间宏，快速判断趋势                                      │
│  - 精测：Profiler（系统级 → 算子级）                               │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 5：单算子优化闭环                                            │
│  - 每个算子独立走完整闭环                                          │
│  - 顺序：Abs → Add → LogicalOr                                    │
│  - 记录 before/after                                              │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 6：整网验证                                                  │
│  - 主入口：prunedGraph                                            │
│  - 补充回归：wukong                                               │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 7：合入标准检查                                              │
│  - 单测全通过                                                     │
│  - 精度对比通过                                                   │
│  - 性能无回退                                                     │
│  - PR 附完整报告                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、阶段详解

### 阶段 1：建立基线

**目标**：记录"修改前"的所有关键数据，作为后续对比基准。

**固定环境要求**：

| 项目 | 要求 |
|------|------|
| 机器 | 同一台 |
| 驱动 | 同一版本 |
| Batch Size | 固定 |
| Seed | 固定 |
| Build 模式 | 不混用 debug/release |

**基线记录内容**：

```
├── 单测结果
│   ├── Abs 单测
│   ├── Add 单测
│   └── LogicalOr 单测
│
├── 精度结果
│   └── prunedGraph --compare-accuracy
│
├── 整网时间
│   └── prunedGraph 总耗时
│
├── 算子 Profile
│   └── prunedGraph --profile-ops
│
└── 补充回归（必要时）
    └── wukong 相关结果
```

**文件留档**：

```
docs/baseline/
├── single_test_before.json      # 单测结果
├── accuracy_before.json         # 精度对比
├── performance_before.json      # 整网时间
├── ops_profile_before.csv       # 算子 profile
└── system_profile_before/       # 系统级 profile
```

---

### 阶段 2：Shape 盘点

**目标**：形成目标算子的真实 Shape 表，筛选热点 shape。

**输出表格格式**：

| op_type | node_name | 出现次数 | 输入来源 | broadcast | dtype | 运行 shape |
|---------|-----------|---------|---------|-----------|-------|-----------|
| Abs | xxx | 2 | xxx | - | float32 | [384, 1] |
| AddV2 | xxx | 384 | xxx | yes | float32 | [384, 256] × [1, 256] → [384, 256] |
| LogicalOr | xxx | 120 | xxx | no | bool | [120, 64] × [120, 64] |

**Shape 归一表模板**：

```
Abs:
├── shape_1: input_shape=[xxx]  出现次数=N
├── shape_2: input_shape=[xxx]  出现次数=M
└── ...

AddV2:
├── shape_1: lhs=[xxx] rhs=[xxx] out=[xxx] broadcast=xxx  出现次数=N
├── shape_2: lhs=[xxx] rhs=[xxx] out=[xxx] broadcast=xxx  出现次数=M
└── ...

LogicalOr:
├── shape_1: lhs=[xxx] rhs=[xxx] out=[xxx] broadcast=xxx  出现次数=N
├── shape_2: lhs=[xxx] rhs=[xxx] out=[xxx] broadcast=xxx  出现次数=M
└── ...
```

**热点 Shape 筛选标准**：

1. 出现次数高
2. 总耗时占比大
3. 典型 broadcast 模式

**临时打印方法**：

```cpp
// 在 Compute() 中临时添加
void Compute(OpKernelContext* ctx) override {
    // ... 获取输入后
    std::cout << "Op: " << name()
              << " Input shape: " << input_t.shape().DebugString()
              << std::endl;
    // ...
}
```

---

### 阶段 3：构造针对性测试

**测试分类**：

```
├── 真实 Shape Case（优先）
│   ├── 从 Shape 表中选取热点 shape
│   ├── 直接复现图中的真实场景
│   └── 精度要求：CPU vs MUSA 完全一致
│
└── 边界 Case（补充）
    ├── 空 tensor（dim_size=0）
    ├── 标量（shape=[]）
    ├── 单维 broadcast
    ├── bool 特殊输入（全True/全False/混合）
    └── 极大/极小值
```

**验收标准**：

```python
# CPU vs MUSA 对比
cpu_result = run_on_cpu(input)
musa_result = run_on_musa(input)
assert np.allclose(cpu_result, musa_result, rtol=1e-5, atol=1e-5)
```

---

### 阶段 4：粗测 → 精测

#### 4.1 粗测（时间宏）

**目的**：快速判断改动是否有方向性收益。

**方法**：

```cpp
// Debug 构建 + 临时 timing 宏
#define TIMING_START(name) auto start = std::chrono::high_resolution_clock::now();
#define TIMING_END(name) auto end = std::chrono::high_resolution_clock::now(); \
                         std::cout << name << ": " << (end-start).count() << "ns" << std::endl;
```

**观察点**：
- 热点 shape 的耗时趋势
- prunedGraph 整网耗时趋势
- 只看方向，不作为最终结论

#### 4.2 精测（Profiler）

**层次化 Profile 策略**：

```
系统级 Profile（先）
├── 工具：Nsight Systems / msys
├── 关注点：
│   ├── GPU 是否空闲
│   ├── kernel 是否碎片化
│   ├── 是否有不必要的数据搬运/同步
│   └── 整体时间线分布
│
└── 输出：时间线图、瓶颈定位

算子级 Profile（后）
├── 工具：Nsight Compute / mcu
├── 关注点：
│   ├── Duration
│   ├── Compute Throughput
│   ├── Memory Throughput
│   ├── Cache hit rate
│   └── 指令分布
│
└── 输出：详细性能指标、优化建议
```

**Profile 命令示例**：

```bash
# 系统级
msys profile --trace=cuda,musa ./run_pruned_graph.sh

# 算子级
mcu profile --metrics=duration,throughput ./run_kernel.sh
```

---

### 阶段 5：单算子优化闭环

**严格执行顺序**：

```
Abs → Add → LogicalOr
（每个算子独立走完整闭环，不并行）
```

**每个算子的闭环流程**：

```
┌─────────────────────────────────────────────────────────────────┐
│  1. 选热点 Shape                                                  │
│     - 选 1-3 个高频/高耗 shape                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. 记录 Before Baseline                                          │
│     - shape 单测耗时                                              │
│     - profiler 数据                                               │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. 做最小改动                                                    │
│     - 只改一处，便于归因                                          │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. 验证改动                                                      │
│     ├── shape 单测（新增 case）                                    │
│     ├── 原有单测（回归）                                          │
│     └── prunedGraph --compare-accuracy                            │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. 测量收益                                                      │
│     ├── 时间宏粗测                                                │
│     └── Profiler 精测                                             │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. 记录 After                                                    │
│     - 与 Before 同格式                                            │
│     - 存档便于对比                                                │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. 决策                                                          │
│     ├── 有收益 → 继续下一个优化点                                  │
│     ├── 无收益/回退 → 回退改动                                    │
│     └── 有副作用 → 分析原因再决定                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 阶段 6：整网验证

**主入口**：prunedGraph

```bash
# 三种模式验证
./run_pruned_graph.sh --compare-accuracy   # 精度对比
./run_pruned_graph.sh --inference-only     # 推理性能
./run_pruned_graph.sh --profile-ops        # 算子 profile
```

**补充回归**：wukong

```bash
# 不作为主收益判据，但需通过
./run_wukong.sh --regression
```

---

### 阶段 7：合入标准

**必须满足**：

| 检查项 | 要求 |
|--------|------|
| 单测 | Abs/Add/LogicalOr 相关单测全通过 |
| Shape Case | 基于真实 shape 的新增 case 全通过 |
| 精度 | prunedGraph 精度对比通过 |
| 性能 | 整网时间无回退，目标算子有正收益 |
| 归因 | Profiler 能解释收益来源 |

**PR 必附材料**：

```
PR 附带文件/
├── before/
│   ├── single_test.json
│   ├── accuracy.json
│   ├── performance.json
│   ├── ops_profile.csv
│   ├── system_profile/
│   └── kernel_profile/
│
├── after/
│   ├── single_test.json
│   ├── accuracy.json
│   ├── performance.json
│   ├── ops_profile.csv
│   ├── system_profile/
│   └── kernel_profile/
│
└── conclusion.md
```

**结论模板**：

```markdown
## 优化总结

### 改了什么
- Abs: xxx
- Add: xxx
- LogicalOr: xxx

### 哪些 Shape 收益最大
- Abs: shape=[xxx] 收益 xx%
- Add: shape=[xxx] 收益 xx%
- LogicalOr: shape=[xxx] 收益 xx%

### 整网是否受益
- prunedGraph 整网耗时: before xxms → after xxms (xx%)
- 主要瓶颈转移至: xxx

### 是否有副作用
- 无 / 有（说明情况）

### Profiler 分析
- 收益来源: xxx
- 关键指标变化: xxx
```

---

## 四、最终产物清单

完成一轮优化后，应产出：

```
├── 1. prunedGraph 目标 Op Shape 表
│   └── docs/optimization/shape_table.md
│
├── 2. 对应 Shape 的单测 Case
│   └── test/ops/abs_hot_shape_test.py
│   └── test/ops/add_hot_shape_test.py
│   └── test/ops/logical_or_hot_shape_test.py
│
├── 3. 每个算子的 Before/After 粗测结果
│   └── docs/optimization/abs_timing.csv
│   └── docs/optimization/add_timing.csv
│   └── docs/optimization/logical_or_timing.csv
│
├── 4. prunedGraph Before/After
│   ├── 精度 JSON
│   ├── 整网性能 JSON
│   ├── 算子 Timing JSON
│   ├── 系统级 Profile
│   └── 算子级 Profile
│
└── 5. PR 说明结论页
    └── PR description 中的 summary
```

---

## 五、常见问题

### Q: 为什么优先用 prunedGraph 而不是 wukong？

prunedGraph 是真实业务场景的子图，其中的 shape 和算子频次有实际意义。wukong 是通用测试集，shape 覆盖广但不一定命中热点。

### Q: 为什么一次只优化一个算子？

多算子同时改动时，如果性能变化，难以判断是哪个改动带来的收益或回退。单算子闭环便于精确归因。

### Q: 粗测和精测有什么区别？

| 类型 | 目的 | 工具 | 结论 |
|------|------|------|------|
| 粗测 | 快速判断方向 | 时间宏 | 只看趋势 |
| 精测 | 量化收益来源 | Profiler | 作为依据 |

### Q: Shape 盘点为什么要临时打印？

TensorFlow 的 shape 推断不一定反映真实运行时的 shape，特别是动态 shape 场景。在 Compute() 中打印才能看到真实值。

---

## 六、总结

```
算子优化流程核心：
├── 1. 真实场景驱动：以 prunedGraph 为准
├── 2. 基线先行：修改前必须留档
├── 3. Shape 盘点：找出热点 shape
├── 4. 针对性测试：围绕真实 shape 构造 case
├── 5. 粗测精测：时间宏看趋势，Profiler 找原因
├── 6. 单算子闭环：一次一个，完整验证
├── 7. 整网回归：精度+性能双重检查
└── 8. 规范合入：材料齐全，结论清晰
```