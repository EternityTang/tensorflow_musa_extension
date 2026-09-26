# awexxxx 的 StridedSliceGrad 工作总结

## 1. 工作概述

awexxxx 主要负责在已有 MUSA `StridedSliceGrad` 基础实现上，进一步完善算子功能、优化底层 Kernel 执行效率，并补充正确性测试。

其工作重点不是从零搭建算子框架，而是围绕已有实现解决以下问题：

- 不同 Rank 的 Tensor 使用统一通用路径，索引计算开销较大；
- 连续切片场景仍然使用通用 Scatter Kernel，效率不足；
- 大 Tensor 场景固定使用 `int64_t` 索引，存在额外开销；
- 缺少对 Rank、连续切片、Strided Slice 和 BF16 场景的系统测试。

通过增加 Rank 专用 Kernel、连续区域快速路径、Dense Copy 优化和索引类型模板化，提升了 `StridedSliceGrad` 的通用性和执行效率。

> Git 中实际提交用户名为 **awexxxx**，本总结沿用该提交身份；用户描述中的“awexxx”对应同一提交人。

## 2. 相关提交

### 2.1 增加并完善 StridedSliceGrad 算子

```text
02ef7ff feat: add stride_slice_grad op (#243)
```

提交人：

```text
awexxxx <106058408+awexxxx@users.noreply.github.com>
```

改动统计：

```text
3 个文件
193 行新增，18 行删除
```

主要修改文件：

- `musa_ext/kernels/array/musa_strided_slice_grad_kernel.mu`
- `musa_ext/kernels/array/musa_strided_slice_grad_op.cc`
- `test/ops/strided_slice_op_test.py`

### 2.2 优化 StridedSliceGrad Kernel

```text
a14b19c opt: optimize strided_slice_grad (#247)
```

提交人：

```text
awexxxx <106058408+awexxxx@users.noreply.github.com>
```

改动统计：

```text
2 个文件
299 行新增，66 行删除
```

主要修改文件：

- `musa_ext/kernels/array/musa_strided_slice_grad_kernel.mu`
- `musa_ext/kernels/array/musa_strided_slice_grad_op.cc`

## 3. 主要实现内容

### 3.1 增加 Rank 专用 Kernel

针对不同输入 Rank 增加了专用执行路径：

```text
StridedSliceGradRank1Kernel
StridedSliceGradRank2Kernel
StridedSliceGradRank3Kernel
StridedSliceGradRank4Kernel
```

Kernel 根据输出梯度中的线性索引，计算对应的多维坐标，再映射到原始 Tensor 的输出位置：

```text
dy 的线性索引
      ↓
还原多维坐标
      ↓
根据 begin 和 strides 计算原始坐标
      ↓
写回 output
```

对于 Rank 1 到 Rank 4，使用显式的坐标展开逻辑，减少通用循环和动态索引计算开销；更高 Rank 或不适合专用路径的场景继续使用通用 Scatter Kernel。

### 3.2 连续内层区域优化

后续优化中增加了连续内层切片专用 Kernel：

```text
StridedSliceGradInnerContiguousKernel
StridedSliceGradInnerContiguousRank1Kernel
StridedSliceGradInnerContiguousRank2Kernel
StridedSliceGradInnerContiguousRank3Kernel
StridedSliceGradInnerContiguousRank4Kernel
```

当切片结果的内层区域连续时，不再对每一个元素完整计算所有维度坐标，而是将数据拆分为：

```text
外部坐标 + 连续内层偏移
```

这样可以复用连续内层区域的地址计算，减少每个线程的索引运算。

优化后的执行选择逻辑大致为：

```text
inner_size > 1
    ├── Rank 1 专用连续 Kernel
    ├── Rank 2 专用连续 Kernel
    ├── Rank 3 专用连续 Kernel
    └── Rank 4 专用连续 Kernel

否则
    └── 普通 Rank/通用 Scatter Kernel
```

### 3.3 Dense Copy 快速路径

对于以下场景：

- 输出 Tensor 和处理区域元素数量相同；
- 所有 `begin` 均为 0；
- 所有 `strides` 均为 1；
- 输入梯度和输出区域构成连续完整拷贝；

通过 `CanUseDenseGradCopy` 判断后，直接使用：

```text
musaMemcpyAsync(..., musaMemcpyDeviceToDevice, stream)
```

完成设备到设备的异步复制，避免启动通用 Scatter Kernel。

该优化适用于连续切片或近似完整拷贝场景，可以显著降低简单 SliceGrad 操作的执行开销。

### 3.4 索引类型模板化

优化前，Kernel 主要统一使用 `int64_t` 进行索引计算。awexxxx 在后续优化中引入了模板化索引类型：

```cpp
template <typename T, typename Index>
```

并增加：

```cpp
StridedSliceGradIndexParams<Index>
```

根据实际数据规模选择合适的 Index 类型，避免所有场景都使用 64 位索引，从而降低：

- 索引参数传递开销；
- 整数除法和取模开销；
- Kernel 寄存器和局部计算压力。

同时保留对大 Tensor 的索引支持，兼顾小规模场景性能和大规模场景安全性。

### 3.5 保留通用 Scatter 路径

在增加多种专用路径的同时，没有删除通用实现，而是根据场景进行分发：

```text
简单完整拷贝
    → musaMemcpyAsync

连续内层切片
    → InnerContiguous 专用 Kernel

Rank 1~4 普通切片
    → Rank 专用 Kernel

其他复杂或高 Rank 场景
    → 通用 Scatter Kernel
```

这种设计能够在保证算子覆盖范围的同时，为常见场景选择更高效的实现。

## 4. 测试补充

awexxxx 在 `test/ops/strided_slice_op_test.py` 中补充了 StridedSliceGrad 测试，主要覆盖：

- Rank 1 连续切片；
- Rank 2 内部窗口切片；
- Rank 4 Tensor 切片；
- Strided Slice；
- BF16 梯度路径；
- CCPM 类似的 Bias 扩维切片场景。

测试通过 CPU 和 MUSA 结果对比验证算子正确性，并针对 BF16 设置相应的误差容忍度。

典型测试包括：

```text
testRank1ContiguousSlice
testRank2InnerSlice
testRank4CCPMLikeTensorSlice
testRank4StridedTensorSlice
testBfloat16Grad
testBiasNewAxisLikeCCPM
```

这部分工作使得 Kernel 优化不只停留在实现层面，也覆盖了不同 Rank、不同 Slice 方式和不同数据类型的功能验证。

## 5. 与 welo 工作的差异

这里的 welo 相关提交包括：

- `40fbaf0`，提交人 `welo516`：初始增加 MUSA `StridedSliceGrad` 算子；
- `80bd59b`，提交人 `weloMThreads`：安全启用 AddN 尾部 SliceGrad 融合；
- `1d2f954`，提交人 `weloMThreads`：OneTrans BF16 训练综合性能优化，其中包含 StridedSliceGrad 相关修改。

目前无法仅凭 Git 作者名确认 `welo516` 和 `weloMThreads` 是否为同一位人员，以下按 Git 提交身份分别描述。

| 对比维度 | awexxxx | welo |
|---|---|---|
| 工作定位 | 算子功能增强和底层 Kernel 优化 | 基础算子建设、图融合和模型场景落地 |
| 初始实现 | 基于已有实现继续开发 | `welo516` 完成初始 MUSA StridedSliceGrad 算子 |
| Kernel 路径 | Rank 专用、连续内层、通用 Scatter 多路径分发 | 初始通用 Kernel，并参与后续模型场景修改 |
| 内存优化 | 增加 Dense Copy，使用 Device-to-Device 异步拷贝 | 当前相关提交重点不在 Dense Copy 专用路径 |
| 索引优化 | 引入模板化 Index 类型，支持不同规模索引 | 当前相关提交重点不在 Index 类型模板化 |
| 图融合 | 没有发现其独立的 SliceGrad 图融合提交 | 修复 AddN 尾部 SliceGrad 融合安全启用逻辑 |
| 模型适配 | 主要关注算子本身的通用性和效率 | 参与 OneTrans BF16 训练场景综合优化 |
| 测试 | 新增并补充 StridedSliceGrad 多场景测试 | 初始算子提交主要未单独增加对应测试文件 |

## 6. 工作链路对比

可以将两者的工作串联为：

```text
welo516
  ↓
完成初始 MUSA StridedSliceGrad 算子

awexxxx
  ↓
补充 Rank 专用 Kernel、连续切片优化、Dense Copy、Index 优化和测试

weloMThreads
  ↓
修复 AddN 尾部 SliceGrad 融合启用逻辑，并参与 OneTrans BF16 综合优化
```

两类工作并非简单重复，而是处于不同层次：

- welo 更偏向 **算子从无到有、图融合和模型集成**；
- awexxxx 更偏向 **已有算子的执行路径细化和 Kernel 性能优化**。

## 7. 工作价值

awexxxx 的实现主要带来以下价值：

1. 提升 `StridedSliceGrad` 对不同 Rank 的覆盖和适配能力；
2. 通过 Rank 专用 Kernel 降低常见维度下的索引计算开销；
3. 通过连续内层 Kernel 提高连续区域写回效率；
4. 通过 Dense Copy 路径避免简单场景下不必要的 Kernel Launch；
5. 通过 Index 模板化兼顾小规模性能和大规模 Tensor 支持；
6. 通过 CPU/MUSA 对比测试验证多种切片模式和 BF16 路径；
7. 为后续模型级 SliceGrad 融合和训练性能优化提供更稳定的算子基础。

## 8. 转正答辩表述

> 在 StridedSliceGrad 方面，我主要负责在已有 MUSA 算子基础上进行功能和性能增强。首先针对不同 Tensor Rank 增加了 Rank 1 到 Rank 4 的专用 Kernel，减少通用 Scatter 路径中的多维索引开销；随后针对连续内层切片增加了专用执行路径，将索引计算拆分为外部坐标和连续内层偏移，提升连续区域写回效率。对于完整连续拷贝场景，我增加了 Dense Copy 快速路径，直接使用 Device-to-Device 异步拷贝，避免额外 Kernel Launch。同时通过模板化 Index 类型降低索引计算开销，并保留通用 Scatter 路径以覆盖复杂和高 Rank 场景。最后补充了 Rank 1、Rank 2、Rank 4、Strided Slice、BF16 以及 CCPM 类似场景的测试，完成了从 Kernel 优化到正确性验证的闭环。

与 welo 的工作相比，welo 主要完成了 StridedSliceGrad 的初始 MUSA 适配、AddN 尾部 SliceGrad 融合安全修复以及 OneTrans BF16 模型场景落地；我的工作则主要聚焦于已有算子的底层执行效率、通用性和多路径 Kernel 优化。

## 9. 说明

当前 Git 记录能够确认上述代码改动和提交归属，但未包含统一、独立的性能基准数据。因此，如果答辩需要量化收益，建议进一步补充：

- 优化前后 Kernel 执行耗时；
- 不同 Rank 和切片模式下的加速比；
- Dense Copy 与通用 Scatter 路径的耗时对比；
- Kernel Launch 数量变化；
- OneTrans BF16 训练中的端到端收益。
