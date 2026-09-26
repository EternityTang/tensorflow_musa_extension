# MusaApplyAdagrad 与 InTopKV2 算子优化总结

## 1. 工作概览

本阶段完成了两个算子的性能优化：

- `MusaApplyAdagrad`：面向训练更新路径，重点解决原有实现存在的“假融合”、多次 Kernel/框架调用以及访存效率不足问题；
- `MusaInTopKV2`：面向分类评估路径，重点解决原始 Kernel 并行度不足、每个样本由单线程串行扫描全部类别的问题。

两项工作的优化方向不同：

```text
MusaApplyAdagrad：训练更新算子融合、减少调度和中间开销
InTopKV2：提高单行类别扫描的 GPU 并行度和带宽利用率
```

---

## 2. MusaApplyAdagrad 优化

### 2.1 提交信息

```text
cf9b487 opt: optimize musa_applyadagrad_op (#262)
```

提交人：

```text
awexxxx <106058408+awexxxx@users.noreply.github.com>
```

提交时间：2026 年 5 月 21 日。

主要修改文件：

- `musa_ext/kernels/training/musa_applyadagrad_kernel.mu`
- `musa_ext/kernels/training/musa_applyadagrad_op.cc`
- `test/ops/apply_adagrad_op_test.py`

改动统计：

```text
3 个文件
497 行新增，346 行删除
```

### 2.2 优化背景

Adagrad 更新的核心计算为：

```text
accum_new = accum + grad * grad
var_new = var - lr * grad / (sqrt(accum_new) + epsilon)
```

原有实现存在以下问题：

- 更新流程中存在不必要的框架层或 muDNN 调用开销；
- 部分路径属于逻辑上的“假融合”，算子表面上合并，但底层仍然产生额外 Kernel 或中间处理；
- Kernel Launch 时间较长；
- 参数、临时 Tensor 和状态更新路径较为复杂；
- 没有针对不同规模和数据对齐情况选择合适的执行策略。

### 2.3 核心优化方案

#### 2.3.1 新增真正的融合 Kernel

新增：

```text
musa_ext/kernels/training/musa_applyadagrad_kernel.mu
```

核心 Kernel：

```cpp
template <typename T, bool UpdateSlots>
__global__ void FusedAdagradV2Kernel(...)
```

将以下操作在同一个 Kernel 中完成：

```text
grad 读取
accum 更新
sqrt
var 更新
accum 写回
var 写回
```

通过 `UpdateSlots` 模板参数区分是否更新累积槽位，避免运行时分支，并覆盖不同的资源更新语义。

#### 2.3.2 支持多种数据类型

Kernel 中针对不同数据类型提供了统一的 Load/Store 抽象：

- `float`
- `double`
- `Eigen::half`
- `bfloat16`

半精度和 BF16 数据在计算时转换为 `float`，完成计算后再写回对应类型，以兼顾计算精度和存储格式。

#### 2.3.3 按数据规模选择线程配置

根据元素数量选择不同的启动策略：

```text
n <= 1024       → 1 个 block，128 threads
较大 Tensor      → 最多 4096 blocks，256 threads
```

同时通过 `BlocksFor` 限制最大 Block 数量，避免超大 Tensor 造成过多调度开销；Kernel 内部使用 grid-stride loop 覆盖完整数据：

```cpp
for (int64_t i = tid; i < n; i += stride)
```

#### 2.3.4 float4 向量化路径

对于满足以下条件的 float Tensor：

- 元素数量不小于 4096；
- 元素数量是 4 的倍数；
- `var`、`accum`、`grad` 地址均满足 16 字节对齐；

使用：

```text
FusedAdagradV2Float4Kernel
```

以 `float4` 为单位加载和存储，每个线程处理 4 个 float 元素，从而减少访存指令数量并提升内存吞吐。

不满足条件时自动回退到标量融合 Kernel，保证通用性和正确性。

#### 2.3.5 简化 OpKernel 执行路径

`musa_applyadagrad_op.cc` 中重新组织了：

- 标量属性读取；
- 临时 Tensor 分配；
- MUSA Stream 获取；
- 融合 Kernel Launch；
- Kernel 错误检查；
- TensorFlow 2.15.1 兼容处理。

优化后的执行链路更直接，减少了原有路径中的冗余处理。

### 2.4 性能收益

提交说明中给出的结果为：

- 优化掉假融合问题后，Kernel Launch 时间缩短约 **3.78 倍**；
- 新 Kernel 的运行时间约为原始 muDNN 总耗时的 **1/1.98**；
- 峰值带宽提升约 **2 倍**。

这些数据来自提交说明，建议答辩时结合实际运行环境和 benchmark 结果进一步确认测试条件。

### 2.5 测试

新增或补充：

```text
test/ops/apply_adagrad_op_test.py
```

测试重点包括：

- ResourceApplyAdagradV2 的正确性；
- 融合大 Tensor 路径；
- 变量、累积槽位和梯度更新结果；
- MUSA 与参考实现的结果对比；
- TensorFlow 2.15.1 兼容性。

---

## 3. InTopKV2 优化

### 3.1 提交信息

InTopKV2 相关优化包含以下提交：

```text
21533b7 opt:optimize musa_intopkv2
b9f1b27 opt:optimize musa_intopkv2
e9043e7 opt:optimize intopkv2 op (#276)
```

主要作者：

- `21533b7`：eilan `<tang007237@gmail.com>`；
- `b9f1b27`：eilan `<tang007237@gmail.com>`；
- `e9043e7`：EternityTang `<126361902+EternityTang@users.noreply.github.com>`。

其中 `b9f1b27` 与 `e9043e7` 的代码改动内容基本对应，后者是带 PR 的提交记录。

相关文档提交：

```text
1d1ee3b add intopkv2 docs
```

该提交新增：

- `docs/intopkv2_optimization_report.md`
- `docs/run_intopkv2_msys_profile.sh`
- `docs/tensordot_fusion.md`

### 3.2 算子功能

`InTopKV2` 用于判断每个样本的目标类别是否位于预测结果的 Top-K 中。

对于每个 batch row：

```text
读取 target 对应的预测分数 target_score
统计整行中严格大于 target_score 的类别数量
若 count_higher < k，则 target 位于 Top-K
```

实现使用严格比较：

```text
score > target_score
```

而不是 `>=`，从而保持 TensorFlow 在分数相等场景下的语义。

### 3.3 原始实现的问题

原始 Kernel 使用：

```cpp
const int row = blockIdx.x * blockDim.x + threadIdx.x;
```

即：

```text
一个线程处理一个 batch row
一个线程串行扫描该 row 的全部 num_classes
```

当：

```text
batch_size = 1024
num_classes = 10000
```

只有大约 1024 个线程有效工作，每个线程都需要串行读取和比较 10000 个类别分数，导致：

- GPU 并行度不足；
- 内存访问无法充分展开；
- 带宽利用率低；
- 算力利用率低；
- 大类别数场景下 Kernel 执行时间较长。

优化前 profiling 中，典型场景带宽利用率约为 **1.59%**，算力利用率约为 **0.0008%～0.0016%**，主要瓶颈是并行度不足而非计算量不足。

### 3.4 核心优化方案

#### 3.4.1 从 one-thread-per-row 改为 one-block-per-row

优化后：

```text
一个 block 处理一个 batch row
一个 block 内 256 个线程协作扫描 num_classes
```

线程以 Block Size 为步长遍历类别：

```cpp
for (int i = threadIdx.x; i < num_classes; i += BLOCK_SIZE) {
  const float score = LoadAsFloat(&row_predictions[i]);
  count_higher += score > target_score;
}
```

每个线程先统计局部结果，再通过 Block Reduction 汇总。

#### 3.4.2 Warp Shuffle + Shared Memory Reduction

新增：

```cpp
BlockReduceSum
```

归约过程分两级：

1. Warp 内使用 `__shfl_xor_sync` 完成快速归约；
2. 每个 Warp 的 lane 0 将结果写入 shared memory；
3. 第一个 Warp 再对各 Warp 结果进行归约；
4. thread 0 写入最终 `output[row]`。

这样可以在保持较少同步开销的情况下完成 block 级计数。

#### 3.4.3 边界值快速路径

针对特殊 K 值增加直接填充路径：

```text
k == 0           → 所有输出为 false
k == num_classes → 所有输出为 true
```

通过 `SetBoolKernel` 一次性设置整个输出，避免扫描预测矩阵。

#### 3.4.4 支持 int32 和 int64 targets

保留并支持两种 target 类型：

- `int32`
- `int64`

通过模板化 Kernel Launcher 复用相同的并行计算逻辑。

### 3.5 性能数据

文档记录的第一轮优化结果如下：

| 场景 | 优化前 wall time | 优化后 wall time | wall speedup |
|---|---:|---:|---:|
| batch=1024, classes=10000 | 约 1.9134 ms | 约 0.3666 ms | 约 5.22x |
| batch=8192, classes=10000 | 约 2.0403 ms | 约 0.6330 ms | 约 3.22x |
| batch=32768, classes=10000 | 约 9.0828 ms | 约 1.6240 ms | 约 5.59x |

其中 batch=8192 的 Kernel 级加速约为 **4.93x**。

后续第二轮优化中，batch=8192 场景的 Kernel 时间进一步从：

```text
1.724354 ms → 0.243652 ms
```

总 Kernel speedup 约为：

```text
7.08x
```

第二轮 benchmark 的 wall time：

| batch | 第一轮 avg | 第二轮 avg | 第二轮相对第一轮 |
|---|---:|---:|---:|
| 1024 | 0.3666 ms | 0.3177 ms | 1.15x |
| 8192 | 0.6330 ms | 0.5235 ms | 1.21x |
| 32768 | 1.6240 ms | 1.1924 ms | 1.36x |

优化后 batch=8192 的等效带宽约为 **1345 GB/s**，说明算子瓶颈已经从“并行度不足”转向更接近带宽和访存效率。

### 3.6 测试和 profiling

正确性测试覆盖：

```text
testInTopKV2BasicInt32
testInTopKV2BasicInt64
testInTopKV2ExactMatch
testInTopKV2KEquals1
testInTopKV2KEqualsAll
testInTopKV2KZero
testInTopKV2LargeBatch
testInTopKV2LargeClasses
testInTopKV2MixedResults
testInTopKV2RandomData
testInTopKV2SmallBatch
```

同时补充了：

- `test/ops/intopkv2_benchmark.py`；
- MUSA profiling 脚本；
- MSYS profile 结果分析；
- 不同 batch size、num_classes 和 K 值的 benchmark。

---

## 4. 两个算子优化的区别

| 对比维度 | MusaApplyAdagrad | InTopKV2 |
|---|---|---|
| 应用场景 | 训练参数更新 | 分类 Top-K 评估/判断 |
| 主要作者 | awexxxx | eilan、EternityTang |
| 原始瓶颈 | 假融合、额外调度和更新路径开销 | 一个线程串行扫描一个 row，并行度不足 |
| 优化层次 | OpKernel + MUSA Kernel + 数据路径 | 主要是 MUSA Kernel 并行策略 |
| 核心方案 | 真正融合 var/accum/grad 更新 | one-block-per-row + block reduction |
| 关键技术 | 模板化 UpdateSlots、grid-stride、float4 向量化、对齐判断 | Warp shuffle、shared memory reduction、边界快速路径 |
| 特殊路径 | float4 向量化和小 Tensor 线程配置 | k=0、k=num_classes 直接填充 |
| 数据类型 | float、double、half、bfloat16 | predictions 支持多类型，target 支持 int32/int64 |
| 主要收益 | Launch 时间约缩短 3.78x，峰值带宽约提升 2x | 典型 wall speedup 约 3.22x～5.59x，Kernel speedup 最高约 7.08x |
| 验证方式 | Adagrad 正确性测试和 TF2.15.1 兼容性 | 11 项正确性测试、benchmark 和 MSYS profiling |

---

## 5. 转正答辩表述

### 5.1 MusaApplyAdagrad

> 在 `MusaApplyAdagrad` 优化方面，我针对原有实现存在的假融合和额外调度开销，重新设计了 MUSA 融合 Kernel，将梯度读取、累积量更新和变量更新放到同一个 Kernel 中完成。实现中使用 `UpdateSlots` 模板参数区分不同更新路径，并根据 Tensor 规模选择线程配置；对于满足 16 字节对齐、元素数量较大的 float Tensor，进一步使用 `float4` 向量化加载和存储，提升访存吞吐。同时保留标量路径以覆盖不满足向量化条件的场景，并补充了多数据类型和 TensorFlow 2.15.1 兼容性测试。根据提交中的 profiling 数据，Kernel Launch 时间缩短约 3.78 倍，峰值带宽提升约 2 倍。

### 5.2 InTopKV2

> 在 `InTopKV2` 优化方面，我针对原始实现一个线程串行扫描一个样本全部类别、GPU 并行度不足的问题，将执行模型从 one-thread-per-row 改为 one-block-per-row。每个 row 由 256 个线程协作扫描类别分数，线程先统计局部高于 target score 的类别数量，再通过 warp shuffle 和 shared memory 完成 block reduction。针对 `k=0` 和 `k=num_classes` 增加了直接填充输出的快速路径，并保持 int32/int64 target 和 TensorFlow 严格大于比较语义。优化后典型场景 wall time 加速约 3～5 倍，后续 profiling 中 Kernel 时间最高获得约 7.08 倍加速。

---

## 6. 说明

以上性能数据来自仓库中的提交说明和 `docs/intopkv2_optimization_report.md`。其中：

- Adagrad 的性能数据主要来自 `cf9b487` 提交说明；
- InTopKV2 的详细 benchmark 和 profiling 数据来自 `1d1ee3b` 文档；
- 实际答辩时应注明测试硬件、Tensor 规模、数据类型、迭代次数和统计口径；
- 如果需要将这些内容作为正式业绩材料，建议补充统一环境下的优化前后复测数据。
