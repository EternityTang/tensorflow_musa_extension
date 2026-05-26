# MUSA 融合算子面试指南

> 本文档整理了 TensorDot、LinearActivation、LayerNorm 三类融合算子的原理、融合规则、内核实现及其相关变体。

---

## 目录

1. [TensorDot 融合](#1-tensordot-融合)
2. [TensorDotBias 融合](#2-tensordotbias-融合)
3. [LinearActivation 融合](#3-linearactivation-融合)
4. [LayerNorm 融合](#4-layernorm-融合)
5. [Normalize 融合](#5-normalize-融合)
6. [融合框架总览](#6-融合框架总览)
7. [附录：优先级与依赖关系](#7-附录优先级与依赖关系)

---

## 1. TensorDot 融合

### 1.1 什么是 TensorDot

TensorDot（张量缩并）是 `np.tensordot` 的 TensorFlow 等价实现，用于对两个张量在指定轴上做缩并运算（einsum 的一种特例）。它本质上是对高维张量做 **Transpose + Reshape + MatMul + Reshape** 的组合操作。

典型应用场景：Transformer 中的注意力计算、Einsum 展开后的矩阵乘法等。

### 1.2 融合前的子图结构

TensorFlow 的 `tf.tensordot(a, b, axes)` 会被展开为一个由 8-10 个算子组成的子图：

```
Shape_1 ──┬── GatherV2_1 ── Prod_1 ──┐
          │                           ├── Pack ── Reshape_1 ──┐
          └── GatherV2_2 ── Prod_2 ──┘                        │
                                                               ├── MatMul ── Reshape_2 (输出)
Transpose ─────────────────────────────────────────────────────┘          │
                                                            ConcatV2 ────┘
```

各节点作用：
- **Shape_1**: 获取输入张量的形状信息
- **GatherV2_1/2**: 从形状中提取自由轴和收缩轴的维度
- **Prod_1/2**: 计算各轴维度的乘积，得到 reshape 后的维度
- **ConcatV2**: 拼接 reshape 后的输出形状（保留未收缩维度 + 权重的自由维度）
- **Pack**: 将两个 Prod 结果打包为 reshape 目标形状
- **Reshape_1**: 将输入 A reshape 为 2D 矩阵 `[free_dims, contract_dims]`
- **Transpose**: 对输入 A 做转置，使得收缩轴移到最后
- **MatMul**: 执行矩阵乘法 `[free_dims, contract_dims] x [contract_dims, b_free_dims]`
- **Reshape_2**: 将 MatMul 的 2D 输出 reshape 回原始的高维形状

### 1.3 融合匹配规则

**融合模式类**: `MusaTensorDotFusion`（优先级 100）

匹配算法从终点 `Reshape_2` 开始，**自底向上**逐层验证：

1. **起点检查**: 当前节点必须是 `Reshape` 类型，且节点名包含 `/Tensordot` 或 `/tensordot` 后缀（提取前缀用于归属判断）
2. **Step 1**: Reshape_2 的两个输入必须分别是 `MatMul` 和 `ConcatV2`，且都属于同一 Tensordot 前缀
3. **Step 2**: MatMul 的输入必须是 `Reshape_1`（属于前缀） + 权重节点（`Identity/ReadVariableOp/Const/Reshape`）
4. **Step 3**: Reshape_1 的两个输入中，一个是 `Pack`（属于前缀），另一个是数据输入（可能是 `Transpose` 或外部节点）
5. **Step 4**: Pack 的两个输入都是 `Prod`，每个 Prod 的输入是 `GatherV2`，两个 GatherV2 共享同一个 `Shape_1`
6. **Step 5**: ConcatV2 至少包含一个 GatherV2 的输出，其余为 Const，axis 也是 Const

**归属判断规则**: 所有子图内部节点必须属于同一 Tensordot 前缀（`node_name == prefix` 或 `node_name.startswith(prefix + "/")`）。

**融合后生成**: `MusaTensorDot` 算子，输入为原始数据（input_a）和权重（input_b），属性中携带 `axes_a` 和 `axes_b`。

### 1.4 融合后调用的内核

**Op 注册**: `MusaTensorDot`（`musa_ext/kernels/fusion/musa_tensordot_op.cc`）

**内核实现**: `MusaTensorDotOp<T>`（注册 float/double/half/bfloat16）

执行流程：
1. **ComputeTensorDims**: 计算转置排列和 reshape 维度
2. **PrepareTensor(A)**: Transpose + Reshape 为 2D `[batch, contract]`
3. **PrepareTensor(B)**: Transpose + Reshape 为 2D `[contract, batch]`
4. **DoMatMul**: 调用 muDNN 的 `mMatMul::Run()` 执行矩阵乘法
5. 输出直接写入已分配的 output buffer（零拷贝 reshape）

辅助 kernel（`musa_tensordot_kernel.mu`）：提供了一个 `TensorDotDirectKernel` 用于小规模 tensordot 的直接计算，避免 transpose 开销。

**关键优化**:
- 支持 TF32 加速（通过 `MUSA_ENABLE_TF32` 环境变量控制，默认开启）
- 当 permutation 为 identity 时跳过 transpose，直接 reshape

### 1.5 强相关融合算子

| 算子 | 关系 | 说明 |
|------|------|------|
| **TensorDotBias** | 下游融合 | TensorDot + BiasAdd，优先级 105（更高） |
| **MatMulBiasAdd** | 同级替代 | 简单 MatMul + BiasAdd 融合（优先级 98） |
| **ConcatMatMul** | 同级模式 | Concat + MatMul 融合 |
| **ReshapeMatMul** | 同级模式 | Reshape + MatMul 融合 |

---

## 2. TensorDotBias 融合

### 2.1 什么是 TensorDotBias

TensorDotBias 是 TensorDot + BiasAdd 的融合，将张量缩并与偏置加法合并为单个算子。这在 Transformer 的线性投影层中非常常见：`output = tensordot(x, weight, axes) + bias`。

### 2.2 融合前的子图结构

TensorDotBias 的融合依赖 TensorDot 融合先执行。匹配时的图结构：

```
MusaTensorDot (已被 TensorDot 融合生成)
      │
  BiasAdd ──── bias (权重节点: Identity/ReadVariableOp/Const)
      │
  (消费者)
```

### 2.3 融合匹配规则

**融合模式类**: `MusaTensorDotBiasFusion`（优先级 105）

匹配算法从 `BiasAdd` 节点开始：

1. **起点检查**: 当前节点必须是 `BiasAdd` 类型
2. **Step 1**: BiasAdd 的 `input[0]` 必须是 `MusaTensorDot` 算子（已被上游融合生成）
3. **Step 2**: BiasAdd 的 `input[1]` 必须是合法的权重节点（`Identity/ReadVariableOp/Const/Reshape`）
4. **Step 3**: 从 MusaTensorDot 节点的属性中提取 `axes_a` 和 `axes_b`

**优先级设计**: TensorDotBias 优先级 105 > TensorDot 优先级 100。这确保了：
- 如果存在 TensorDot + BiasAdd 的完整链，BiasAdd 先被 TensorDotBias 匹配
- 如果 BiasAdd 的输入还不是 MusaTensorDot，TensorDotBias 匹配失败，退化为 TensorDot 单独融合

**融合后生成**: `MusaTensorDotBias` 算子，输入为 input_a、input_b 和 bias。

### 2.4 融合后调用的内核

**Op 注册**: `MusaTensorDotBias`（`musa_ext/kernels/fusion/musa_tensordot_bias_op.cc`）

**内核实现**: `MusaTensorDotBiasOp<T>`

执行流程与 TensorDot 类似，但关键区别在 MatMul 步骤：
- 使用 `mMatMul::RunWithBiasAdd()` 替代 `mMatMul::Run()`
- muDNN 在 GEMM 计算完成后直接加上 bias（epilogue 融合），无需额外的 kernel launch

```
[batch, contract] x [contract, batch] + bias → [batch, batch]
```

### 2.5 强相关融合算子

| 算子 | 关系 | 说明 |
|------|------|------|
| **TensorDot** | 上游依赖 | TensorDotBias 的输入必须先被 TensorDot 融合 |
| **LinearActivation** | 同族扩展 | MatMul + BiasAdd + Relu 融合（优先级 120） |
| **MatMulBiasAdd** | 简化版 | 只融合 MatMul + BiasAdd（优先级 98） |

---

## 3. LinearActivation 融合

### 3.1 什么是 LinearActivation

LinearActivation 将 **MatMul + BiasAdd + Relu** 三个算子融合为一个。这是深度学习中最常见的计算模式，尤其在 Transformer 的 FFN 层和传统全连接层中：

```
output = Relu(MatMul(x, weight) + bias)
```

### 3.2 融合前的子图结构

```
MatMul ──── BiasAdd/Add/AddV2 ──── Relu
  │              │                    │
  A, B          bias               (消费者)
```

### 3.3 融合匹配规则

**融合模式类**: `LinearActivationFusion`（优先级 120）

匹配算法从 `Relu` 节点开始，**自底向上**：

1. **起点检查**: 当前节点必须是 `Relu`，且不能有 `_original` 后缀（避免重复融合）
2. **Step 1**: Relu 的 `input[0]` 的 producer 必须是 `BiasAdd`、`Add` 或 `AddV2`
3. **Step 2**: BiasAdd/Add/AddV2 的两个输入中，必须有一个是 `MatMul`，另一个是 bias 节点
4. **捕获节点**:
   - `output`: Relu 节点
   - `bias_add`: BiasAdd/Add/AddV2 节点
   - `matmul`: MatMul 节点
   - `bias`: 偏置张量节点

**融合后图变换**:
1. 将原始 Relu 重命名为 `<name>_original`
2. 创建新的 `MusaLinearActivation` 节点，输入为 MatMul 的两个输入 + bias
3. 删除未使用的原始 MatMul、BiasAdd、Relu 节点

### 3.4 融合后调用的内核

**Op 注册**: `MusaLinearActivation`（`musa_ext/kernels/fusion/musa_linear_relu_op.cc`）

**内核实现**: `MusaLinearActivationOp<T>`（注册 float/half/bfloat16/double）

执行路径（由 `MUSA_ENABLE_TF32` 环境变量控制，默认开启 TF32）：

**TF32 模式（默认）**:
```cpp
// 单次 GEMM 调用，BiasAdd + Relu 作为 epilogue 融合
mBatchMatMul op;
MatMulLtParam param;
param.SetEpilogue(MATMULLT_EPILOGUE_RELU_BIAS);  // 关键：epilogue 融合
op.RunLt(handle, mt_out, mt_a, mt_b, mt_out, mt_bias, param);
```

**非 TF32 模式**:
```cpp
// 分两步：先 GEMM+Bias，再 Relu
mBatchMatMul op;
op.RunWithBiasAdd(handle, mt_out, mt_a, mt_b, mt_bias);  // MatMul + BiasAdd
mUnary unary_op;
unary_op.SetMode(mUnary::Mode::RELU);
unary_op.Run(handle, mt_out, mt_out);                     // Relu
```

**关键优化**:
- TF32 模式下，BiasAdd + Relu 作为 GEMM 的 epilogue 在同一次 kernel launch 中完成
- 张量会被 reshape 为 3D 以支持 batch 维度（`ReshapeTo3D`）
- epilogue 融合避免了中间结果写回全局内存

### 3.5 强相关融合算子

| 算子 | 关系 | 说明 |
|------|------|------|
| **MatMulBiasAdd** | 子集 | 只融合 MatMul + BiasAdd（无 Relu），优先级 98 |
| **TensorDotBias** | 同族 | TensorDot + BiasAdd 融合，优先级 105 |
| **GeluFusion** | 下游扩展 | 独立的 Gelu 激活融合（当前未与 MatMul+Bias 合并） |

**优先级竞争**: LinearActivation（120）> TensorDotBias（105）> TensorDot（100）> MatMulBiasAdd（98）。高优先级的模式先匹配，避免低优先级模式截断更优的融合机会。

---

## 4. LayerNorm 融合

### 4.1 什么是 LayerNorm

Layer Normalization（层归一化）是 Transformer 中的核心算子，对每个样本的最后一维做归一化：

```
y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
```

TensorFlow 中 `tf.nn.layer_norm` 或手动实现会展开为 10+ 个基础算子。

### 4.2 融合前的子图结构

LayerNorm 融合是**两阶段**过程，由两个不同的融合模式协作完成：

**第一阶段：Normalize 融合**（匹配 9 个算子）

```
x ──┬── Mean_1 ── ExpandDims_1 ──┐
    │                              ├── Sub ──┬── Square ── Mean_2 ── ExpandDims_2 ── Sqrt ── MusaClip ── RealDiv
    │         (x - mean)  ────────┘          │                                              (clamp)      │
    └────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**第二阶段：LayerNorm 融合**（匹配 Normalize + Mul + AddV2）

```
MusaNormalize (第一阶段输出)
      │
  Mul(gamma) ── AddV2(beta)
      │
  (消费者)
```

### 4.3 融合匹配规则

#### 第一阶段：`MusaNormalizeFusion`（优先级 105）

从 `RealDiv` 节点开始，**自底向上**匹配 9 层：

1. **Layer 9（终点）**: `RealDiv` — `(x - mean) / clipped_std`
2. **Layer 8**: `MusaClip` — 对 sqrt(variance) 做 clamp `[epsilon, max_std]`
3. **Layer 7**: `Sqrt` — 计算标准差
4. **Layer 6**: `ExpandDims_2` — 扩展 variance 维度用于广播
5. **Layer 5**: `Mean_2` — 计算方差 `E[(x - mean)^2]`
6. **Layer 4**: `Square` — 计算 `(x - mean)^2`
7. **Layer 3**: `Sub` — 计算 `x - mean`
8. **Layer 2**: `ExpandDims_1` — 扩展 mean 维度用于广播
9. **Layer 1（起点）**: `Mean_1` — 计算均值

**归属判断**: 所有节点必须共享同一名称前缀（最后一段 `/` 之前的部分）。

**提取参数**:
- `reduction_indices`: 从 Mean 节点的属性中获取
- `epsilon`: 从 MusaClip 的 clip_min 输入获取
- `max_std`: 从 MusaClip 的 clip_max 输入获取

**输出**: `MusaNormalize` 算子，gamma=1.0, beta=0.0（仅为接口兼容）

#### 第二阶段：`MusaLayerNormFusion`（优先级 1）

从 `AddV2` 节点开始，匹配 3 层：

1. **Layer 3（终点）**: `AddV2` — 加 beta 偏置
2. **Layer 2**: `Mul` — 乘 gamma 缩放因子
3. **Layer 1（起点）**: `MusaNormalize` — 第一阶段融合的输出

**依赖关系**: 必须在 `MusaNormalizeFusion` 之后执行（优先级 1 < 105）。

**额外检查**:
- gamma 和 beta 必须是 Const（可选通过 Identity/ExpandDims 链）
- 中间节点不能被外部引用（fork 检测）
- 所有节点共享同一名称前缀

**输出**: `MusaLayerNorm` 算子，输入为 x、gamma、beta，属性携带 epsilon

#### 变体：`MusaFuseLayerNormV2Fusion`（优先级 110）

匹配基于 `FusedBatchNormV3` 的 LayerNorm 实现（某些模型框架会用 BN 来实现 LN）：

```
x -> Reshape([1, d0*d1, d2, 1])
  -> FusedBatchNormV3(scale=Fill(1.0), offset=Fill(0.0), data_format=NCHW)
  -> Reshape(Shape(x))
  -> Mul(gamma) -> Add(beta)
```

**验证条件**: BN 的 scale 必须是全 1.0，offset 必须是全 0.0（即 BN 仅用于归一化，不含学习参数）。

### 4.4 融合后调用的内核

#### MusaLayerNorm 内核

**Op 注册**: `MusaLayerNorm`（`musa_ext/kernels/fusion/musa_layernorm_op.cc`）

**内核实现**: `MusaLayerNormOp<T>`（注册 float/half/bfloat16）

```cpp
::musa::dnn::LayerNorm ln;
ln.SetEpsilon(epsilon_);
ln.SetAxis(axis_vec.size(), axis_vec.data());  // 对最后一维归一化
ln.Run(handle, mt_y, mt_mean, mt_inv_var, mt_x, mt_gamma, mt_beta, mm);
```

- 使用 muDNN 的 `LayerNorm` 原语
- 轴始终是最后一维（`x.dims() - 1`）
- 临时分配 `mean` 和 `inv_var` 张量（形状为除最后一维外的所有维度）
- 反向传播使用 `ln.RunBwd()`，会重新计算 forward 以获取 mean/inv_var

#### MusaNormalize 内核

**Op 注册**: `MusaNormalize`（`musa_ext/kernels/fusion/musa_normalize_fusion_op.cc`）

**内核实现**: 自定义 MUSA kernel（`musa_normalize_kernel.mu`），**不使用 muDNN**

```cpp
// 两遍算法（per row）：
// Pass 1: mean = warp_reduce_sum(x) / row_size
// Pass 2: var = warp_reduce_sum((x - mean)^2) / row_size
// Pass 3: output = (x - mean) / clamp(sqrt(var), epsilon, max_std)
```

两种 kernel 变体：
- `NormalizeKernel<T, 256>`: 通用版本，每行 256 threads，适用于 row_size > 32
- `NormalizeKernelSmall<T, ROW_SIZE>`: 优化版本，每行 32 threads（一个 warp），编译期模板特化处理 row_size 1-32

**数学差异**:
- LayerNorm (muDNN): `output = (x - mean) / sqrt(var + epsilon)`
- Normalize (自定义): `output = (x - mean) / clamp(sqrt(var), epsilon, max_std)`

当 `var >> epsilon^2` 时两者近似等价。Normalize 的 clamp 方式额外提供了上界限制（`max_std`）。

### 4.5 强相关融合算子

| 算子 | 关系 | 说明 |
|------|------|------|
| **Normalize** | 上游依赖 | LayerNorm 的核心归一化部分，优先级 105 |
| **FuseLayerNormV2** | 变体 | 基于 FusedBatchNormV3 的 LayerNorm，优先级 110 |
| **RMSNorm** | 同族 | 不同的归一化方式（无 mean 中心化），由优化器内联优化 |
| **GeluFusion** | 下游常见 | LayerNorm 后常接 Gelu 激活（当前独立融合） |

**RMSNorm** 不走融合模式框架，而是在 `MusaGraphOptimizer` 中直接作为内联优化 pass 处理（`OptimizeForwardRmsNorm`），重写 `Mul(x, Rsqrt(Add(Mean(Square(x)), eps))) * gamma` 为 `MusaRmsNorm`。

---

## 5. Normalize 融合

### 5.1 什么是 Normalize

Normalize 是 LayerNorm 的核心归一化部分（不含 gamma/beta 仿射变换）。数学公式：

```
output = (x - mean) / clamp(sqrt(variance), epsilon, max_std)
```

在某些模型中（如 BERT 的某些变体），归一化和仿射变换是分开的，此时只有 Normalize 部分被融合。

### 5.2 融合匹配规则

见 [4.3 第一阶段](#第一阶段musanormalizefusion优先级-105)。

**注意**: Normalize 融合后会生成 gamma=1.0, beta=0.0 的占位输入，仅为接口兼容 LayerNorm。实际计算中这些值被忽略。

### 5.3 融合后调用的内核

见 [4.4 MusaNormalize 内核](#musanormalize-内核)。

### 5.4 强相关融合算子

| 算子 | 关系 | 说明 |
|------|------|------|
| **LayerNorm** | 下游融合 | Normalize + Mul(gamma) + AddV2(beta)，优先级 1 |
| **ClipFusion** | 上游依赖 | Normalize 模式中的 MusaClip 来自 Clip 融合 |

---

## 6. 融合框架总览

### 6.1 融合执行流程

```
MusaGraphOptimizer::OptimizeFusion()
    │
    ├── 1. 获取所有注册的融合模式（FusionPatternManager 单例）
    ├── 2. 按优先级降序排列并分组
    ├── 3. 对每个优先级组：
    │       ├── 正向扫描图中每个节点
    │       │       └── pattern->Match(graph, node_idx)
    │       │               └── 如果匹配 → pattern->Apply(graph, result)
    │       └── 反向扫描图中每个节点
    │               └── 同上
    └── 4. 重复步骤 3 直到无新融合（最多 50 轮）
```

### 6.2 融合模式注册机制

```cpp
// 基类
class FusionPattern {
    virtual FusionMatchResult Match(const GraphDef& graph, int start_node_idx) const = 0;
    virtual Status Apply(GraphDef* graph, const FusionMatchResult& result) const = 0;
    virtual int GetPriority() const = 0;
    virtual std::string GetName() const = 0;
};

// 注册宏
REGISTER_FUSION_PATTERN(MusaTensorDotFusion);
REGISTER_FUSION_KERNEL(MusaTensorDotFusion, []() { return true; });
```

`FusionPatternManager` 是单例，在静态初始化阶段收集所有注册的模式。`OptimizeFusion()` 从 manager 获取排序后的模式列表。

### 6.3 匹配结果结构

```cpp
struct FusionMatchResult {
    bool matched;
    std::vector<const NodeDef*> matched_nodes;           // 所有匹配到的节点
    std::map<std::string, const NodeDef*> captured_nodes; // 语义化捕获（如 "output", "matmul", "bias"）
    std::map<std::string, std::string> captured_attrs;    // 提取的属性值（如 axes, epsilon）
};
```

---

## 7. 附录：优先级与依赖关系

### 7.1 完整融合模式优先级表

| 优先级 | 融合模式 | 输入模式 | 输出算子 |
|--------|----------|----------|----------|
| 120 | LinearActivationFusion | MatMul + BiasAdd + Relu | MusaLinearActivation |
| 110 | FuseLayerNormV2Fusion | FusedBatchNormV3-based LN | MusaLayerNorm |
| 105 | TensorDotBiasFusion | MusaTensorDot + BiasAdd | MusaTensorDotBias |
| 105 | NormalizeFusion | 9-op normalize subgraph | MusaNormalize |
| 100 | TensorDotFusion | 8-op tensordot subgraph | MusaTensorDot |
| 98 | MatMulBiasAddFusion | MatMul + BiasAdd | MusaMatMulBiasAdd |
| 1 | LayerNormFusion | MusaNormalize + Mul + AddV2 | MusaLayerNorm |

### 7.2 融合依赖链

```
MusaNormalizeFusion (优先级 105)
        │ 生成 MusaNormalize 节点
        ▼
MusaLayerNormFusion (优先级 1)  ←── 依赖上游先执行
        │ 生成 MusaLayerNorm 节点


MusaTensorDotFusion (优先级 100)
        │ 生成 MusaTensorDot 节点
        ▼
MusaTensorDotBiasFusion (优先级 105)  ←── 依赖上游先执行
        │ 生成 MusaTensorDotBias 节点
```

### 7.3 面试常见问题

**Q: 为什么 TensorDotBias 优先级（105）比 TensorDot（100）高？**
A: 因为 TensorDotBias 需要在 BiasAdd 节点处匹配。如果 TensorDot 先执行，BiasAdd 的输入已经变成了 MusaTensorDot，此时 TensorDotBias 反而匹配不到了。所以 TensorDotBias 必须先匹配。但实际上两者在同一轮扫描中都会尝试，优先级只决定同组内的匹配顺序。

**Q: LayerNorm 为什么分两阶段融合？**
A: 因为 Normalize（核心归一化）和 gamma/beta 仿射变换在图中可能不在连续位置，中间可能有其他算子。分两阶段可以灵活处理各种变体。优先级 105 先融合 Normalize，优先级 1 再将 Normalize + Mul + AddV2 合并为 LayerNorm。

**Q: Normalize 和 LayerNorm 的数学区别？**
A: Normalize 使用 `clamp(sqrt(var), epsilon, max_std)`，LayerNorm (muDNN) 使用 `sqrt(var + epsilon)`。当方差远大于 epsilon^2 时两者近似等价。Normalize 额外提供了 max_std 上界限制。

**Q: 为什么 LinearActivation 优先级最高（120）？**
A: 因为 MatMul + BiasAdd + Relu 是最常见的模式。如果 MatMulBiasAdd（98）先匹配，会把 MatMul + BiasAdd 融合掉，导致 Relu 无法一起融合。高优先级确保三算子融合优先于两算子融合。

**Q: 融合后内核调用的关键优化是什么？**
A: 核心是 **epilogue 融合**。以 LinearActivation 为例，muDNN 的 `RunLt` API 支持将 BiasAdd + Relu 作为 GEMM 的 epilogue，在同一次 kernel launch 中完成。这避免了中间结果写回全局内存，将 3 次 kernel launch 减少为 1 次。
