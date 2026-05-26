# TensorDot 与 TFMUSA 融合实现说明

本文介绍 `tf.tensordot` 的基本语义、TensorFlow 中常见的展开形式，以及 TFMUSA 中 `MusaTensorDot` / `MusaTensorDotBias` 的图融合和 kernel 实现方式。

相关代码主要位于：

- `musa_ext/graph/fusion/tensordot_fusion.cc`
- `musa_ext/graph/fusion/tensordot_fusion.h`
- `musa_ext/graph/fusion/tensordot_bias_fusion.cc`
- `musa_ext/graph/fusion/tensordot_bias_fusion.h`
- `musa_ext/kernels/fusion/musa_tensordot_op.cc`
- `musa_ext/kernels/fusion/musa_tensordot_bias_op.cc`

## 1. TensorDot 是什么

`tensordot` 可以理解成高维张量版本的矩阵乘法。它指定 A 的若干轴和 B 的若干轴做收缩求和，剩下的轴拼接成输出 shape。

例如：

```text
A shape = [a0, a1, a2, a3]
B shape = [b0, b1, b2]
axes = ([2, 3], [0, 1])
```

要求：

```text
A[2] == B[0]
A[3] == B[1]
```

输出 shape 是：

```text
A 的非收缩轴 + B 的非收缩轴
= [a0, a1] + [b2]
= [a0, a1, b2]
```

计算公式可以写成：

```text
C[i0, i1, j2] = sum(k0, k1) A[i0, i1, k0, k1] * B[k0, k1, j2]
```

本质上，`tensordot` 可以被转成一次二维矩阵乘法：

```text
A -> [A_free_product, K]
B -> [K, B_free_product]
MatMul -> [A_free_product, B_free_product]
Output -> A_free_shape + B_free_shape
```

其中：

```text
K = A 收缩轴尺寸乘积 = B 收缩轴尺寸乘积
```

## 2. TensorFlow 中 tensordot 的常见展开形式

TensorFlow Python 层的 `tf.tensordot` 通常会被展开成一串基础 op，而不是保留成单个 op。

典型子图包括：

```text
Shape
  ↓
GatherV2
  ↓
Prod
  ↓
Pack
  ↓
Reshape
  ↓
MatMul
  ↓
ConcatV2
  ↓
Reshape
```

如果 A 或 B 的收缩轴不在适合 matmul 的位置，还会插入：

```text
Transpose
```

整体结构可以理解为：

```text
A
│
├─ Shape
│   ├─ GatherV2 -> Prod ┐
│   └─ GatherV2 -> Prod ├─ Pack -> Reshape_1
│                       │
├─ Transpose? ----------┘
│
B / weight
│
└────────────── MatMul
                  │
Shape pieces ── ConcatV2
                  │
               Reshape_2
                  │
               output
```

其中：

- `Shape`：取输入张量 shape；
- `GatherV2`：从 shape 中取自由轴或收缩轴的维度；
- `Prod`：把多个维度乘成一个维度；
- `Pack`：构造 reshape 到二维矩阵的目标 shape；
- `Reshape_1`：把 A 转成 `[M, K]`；
- `MatMul`：执行核心矩阵乘法；
- `ConcatV2`：构造最终高维输出 shape；
- `Reshape_2`：把 `[M, N]` 还原成高维输出。

## 3. 为什么需要融合

如果不融合，TensorFlow runtime 会调度多个小 op：

```text
Shape -> GatherV2 -> Prod -> Pack -> Reshape -> MatMul -> ConcatV2 -> Reshape
```

这会带来几个问题：

1. 图节点多，调度开销大；
2. 后端看到的是拆散后的基础 op，不容易恢复出 `tensordot` 语义；
3. MUSA 后端无法直接使用更合适的 `mMatMul` / `RunWithBiasAdd` 路径；
4. 后续 `BiasAdd` 不容易和 matmul 合并；
5. 优化逻辑分散在多个 op 上，不利于统一处理 dtype、TF32、workspace 和 layout。

融合后的目标是把子图替换为：

```text
MusaTensorDot
```

如果后面接 `BiasAdd`，进一步替换为：

```text
MusaTensorDotBias
```

## 4. TFMUSA 中的融合入口

普通 tensordot 融合由 `MusaTensorDotFusion` 实现：

```text
musa_ext/graph/fusion/tensordot_fusion.cc
musa_ext/graph/fusion/tensordot_fusion.h
```

融合后的 op：

```text
MusaTensorDot
```

TensorDot + BiasAdd 融合由 `MusaTensorDotBiasFusion` 实现：

```text
musa_ext/graph/fusion/tensordot_bias_fusion.cc
musa_ext/graph/fusion/tensordot_bias_fusion.h
```

融合后的 op：

```text
MusaTensorDotBias
```

两个 fusion pattern 的优先级分别是：

```text
MusaTensorDotFusion     priority = 100
MusaTensorDotBiasFusion priority = 105
```

## 5. MusaTensorDotFusion 如何匹配子图

`MusaTensorDotFusion` 从最终输出的 `Reshape` 节点开始反向匹配。

入口逻辑位于：

```text
musa_ext/graph/fusion/tensordot_fusion.cc:263
```

它只从 `Reshape` 开始匹配：

```text
Reshape_2
```

这是因为 TensorFlow 展开的 tensordot 最后通常用 `Reshape_2` 把二维 matmul 结果还原成高维输出。

从最终输出节点反向匹配有一个好处：融合后的新节点可以继承最终输出节点的名字，因此外部消费者不需要重定向。

### 5.1 Tensordot 前缀检查

代码会从节点名中提取 tensordot 前缀，例如：

```text
.../Tensordot
.../tensordot
```

内部节点通常是：

```text
prefix/xxx
```

匹配逻辑要求关键节点属于同一个 tensordot 前缀，避免把普通的 `Reshape + MatMul + Reshape` 误判成 tensordot。

### 5.2 匹配最终 Reshape

最终结构要求：

```text
Reshape_2
  input[0] = MatMul
  input[1] = ConcatV2
```

含义是：

```text
MatMul 产生二维结果 [M, N]
ConcatV2 产生最终输出 shape
Reshape_2 把 [M, N] 还原成高维输出
```

图形：

```text
MatMul ─────┐
            ├─ Reshape_2
ConcatV2 ───┘
```

如果 `input[0]` 不是 `MatMul`，或者 `input[1]` 不是 `ConcatV2`，匹配失败。

### 5.3 匹配 MatMul 输入

接着匹配：

```text
MatMul
  input[0] = Reshape_1
  input[1] = weight
```

图形：

```text
Reshape_1 ─┐
           ├─ MatMul
weight ────┘
```

当前实现中，`weight` 允许的 op 类型包括：

```text
Identity
ReadVariableOp
Const
Reshape
```

这说明当前融合主要面向模型权重场景，例如：

```python
tf.tensordot(x, weight, axes=...)
```

尤其是 dense / projection / linear 类模式。

### 5.4 匹配 A 侧 Reshape

`Reshape_1` 负责把 A 转成二维矩阵。

要求结构为：

```text
Reshape_1
  input[0] = A 或 Transpose(A)
  input[1] = Pack
```

图形：

```text
A / Transpose(A) ─┐
                  ├─ Reshape_1
Pack([M, K]) ─────┘
```

如果 A 的自由轴已经在前、收缩轴已经在后，则不需要 transpose，可以直接 reshape。

例如：

```text
A shape = [2, 3, 4]
axes_a = [2]
```

A 已经满足：

```text
free axes = [0, 1]
contract axes = [2]
```

可以直接：

```text
[2, 3, 4] -> [2*3, 4]
```

如果收缩轴在中间，则需要 transpose。

例如：

```text
A shape = [2, 3, 4]
axes_a = [1]
```

需要把：

```text
[free, contract, free]
```

变成：

```text
[free, free, contract]
```

即：

```text
Transpose(A, [0, 2, 1]) -> [2, 4, 3]
Reshape -> [8, 3]
```

### 5.5 匹配 Pack、Prod、GatherV2、Shape

这一段用于识别 reshape 到二维矩阵时的 `[M, K]` 是如何构造出来的。

结构是：

```text
Pack
  input[0] = Prod_1
  input[1] = Prod_2
```

其中：

```text
Prod_1 = A 自由轴尺寸乘积 = M
Prod_2 = A 收缩轴尺寸乘积 = K
```

每个 `Prod` 的输入来自 `GatherV2`：

```text
Shape(A) -> GatherV2 -> Prod
```

例如：

```text
A shape = [2, 3, 4, 5]
axes_a = [2, 3]
```

则：

```text
A 自由轴 = [0, 1] -> shape [2, 3]
A 收缩轴 = [2, 3] -> shape [4, 5]
```

对应：

```text
GatherV2_1(Shape(A), [0, 1]) -> [2, 3]
Prod_1 -> 6

GatherV2_2(Shape(A), [2, 3]) -> [4, 5]
Prod_2 -> 20

Pack -> [6, 20]
Reshape_1 -> A_2d [6, 20]
```

这里 `GatherV2` 和 `Prod` 处理的是 shape，不是 tensor 数据本身。

### 5.6 匹配 ConcatV2 输出 shape

最终输出 shape 是：

```text
A 的自由轴 shape + B 的自由轴 shape
```

例如：

```text
A shape = [2, 3, 4, 5]
B shape = [4, 5, 7]
axes = ([2, 3], [0, 1])
```

输出 shape 为：

```text
[2, 3, 7]
```

TensorFlow 展开图中，最终 shape 通常通过 `ConcatV2` 构造。

`MusaTensorDotFusion` 会检查 `ConcatV2` 至少包含 A 自由轴相关的 `GatherV2`，并且 concat axis 是 const-like 节点。

## 6. axes 如何提取

融合后，原始 `GatherV2`、`Prod`、`Pack` 等节点会被删除，所以 fused op 必须保存 tensordot 的关键属性：

```text
axes_a
axes_b
```

当前实现中：

```text
axes_a = 从 gather_2 的 indices 常量中提取
axes_b = 默认 [0]
```

`gather_2` 对应 A 的收缩轴。

例如：

```text
Shape(A) = [2, 3, 4, 5]
gather_2 indices = [2, 3]
```

则：

```text
axes_a = [2, 3]
```

当前 `axes_b` 默认是：

```text
[0]
```

因此这个融合更偏向常见 dense-like 模式：

```text
[..., K] x [K, N] -> [..., N]
```

也就是：

```text
A 的某些轴 contract B 的第 0 维
```

kernel 层的 `MusaTensorDot` 支持 `axes_b` 属性，但 graph fusion 当前没有完整解析任意 B 侧收缩轴。

## 7. Apply 阶段如何改图

匹配成功后，`Apply()` 会把原始 tensordot 子图替换成一个 `MusaTensorDot` 节点。

核心步骤如下：

1. 获取原始输入 A；
2. 获取 weight / B 输入；
3. 获取 dtype；
4. 提取 `axes_a` 和 `axes_b`；
5. 收集需要删除的 tensordot 内部节点；
6. 保留外部仍然引用的 shared node；
7. 删除可融合子图节点；
8. 清理无消费者的孤立 producer；
9. 新增 `MusaTensorDot` 节点。

改写前：

```text
A -> Transpose?/Reshape -> MatMul -> Reshape -> output
B --------------------------┘
```

改写后：

```text
A ─┐
   ├─ MusaTensorDot -> output
B ─┘
```

新节点继承原最终输出节点的名字：

```text
fused_node->set_name(output_name)
```

这样外部消费者仍然引用同一个节点名，不需要额外重定向。

新节点输入：

```text
input[0] = A
input[1] = B / weight
```

新节点属性：

```text
T
axes_a
axes_b
```

## 8. MusaTensorDotBiasFusion

`MusaTensorDotBiasFusion` 是第二阶段融合，用来把：

```text
MusaTensorDot -> BiasAdd
```

融合成：

```text
MusaTensorDotBias
```

匹配结构：

```text
BiasAdd
  input[0] = MusaTensorDot
  input[1] = bias
```

改写前：

```text
A ─┐
   ├─ MusaTensorDot ── BiasAdd -> output
B ─┘                    ↑
                       bias
```

改写后：

```text
A ────┐
B ────┼─ MusaTensorDotBias -> output
bias ─┘
```

`MusaTensorDotBias` 有三个输入：

```text
input[0] = A
input[1] = B / weight
input[2] = bias
```

属性同样包括：

```text
T
axes_a
axes_b
```

融合 bias 的主要收益是可以让后端 matmul 库使用 `RunWithBiasAdd` 路径，避免额外一次 `BiasAdd` op 调度和输出读写。

## 9. MusaTensorDot op 注册

`MusaTensorDot` 注册在：

```text
musa_ext/kernels/fusion/musa_tensordot_op.cc
```

op 定义：

```text
Input:
  a: T
  b: T

Output:
  output: T

Attr:
  T: {float, double, half, bfloat16}
  axes_a: list(int)
  axes_b: list(int)
```

它的语义等价于：

```python
tf.tensordot(a, b, axes=(axes_a, axes_b))
```

shape function 的规则是：

```text
输出 shape = A 的非收缩轴 + B 的非收缩轴
```

例如：

```text
A shape = [2, 3, 4, 5]
B shape = [4, 5, 7]
axes_a = [2, 3]
axes_b = [0, 1]
```

输出：

```text
[2, 3, 7]
```

## 10. MusaTensorDot kernel 执行流程

`MusaTensorDotOp::Compute()` 的整体流程是：

```text
1. 读取输入 a 和 b
2. ComputeTensorDotDims 计算维度信息
3. 分配 output
4. 设置 MUDNN handle 和 TF32
5. 调用 DoTensorDot
```

### 10.1 TensorDotDims

kernel 内部会计算一个 `TensorDotDims` 结构：

```text
a_batch_size
a_contract_size
b_contract_size
b_batch_size
output_dims
a_perm
b_perm
```

含义如下：

```text
a_batch_size     = A 所有自由轴尺寸乘积，也就是 M
a_contract_size  = A 所有收缩轴尺寸乘积，也就是 K
b_contract_size  = B 所有收缩轴尺寸乘积，也就是 K
b_batch_size     = B 所有自由轴尺寸乘积，也就是 N
output_dims      = A 自由轴 shape + B 自由轴 shape
a_perm           = A 转成 [free..., contract...] 的 permutation
b_perm           = B 转成 [contract..., free...] 的 permutation
```

最终目标是：

```text
A -> [M, K]
B -> [K, N]
MatMul -> [M, N]
Output -> output_dims
```

### 10.2 维度计算示例

例如：

```text
A shape = [2, 3, 4, 5]
B shape = [4, 5, 7]
axes_a = [2, 3]
axes_b = [0, 1]
```

A：

```text
自由轴 = [0, 1] -> shape [2, 3]
收缩轴 = [2, 3] -> shape [4, 5]
```

B：

```text
收缩轴 = [0, 1] -> shape [4, 5]
自由轴 = [2]    -> shape [7]
```

所以：

```text
a_batch_size = 2 * 3 = 6
a_contract_size = 4 * 5 = 20
b_contract_size = 4 * 5 = 20
b_batch_size = 7
```

转成 matmul：

```text
A_2d = [6, 20]
B_2d = [20, 7]
A_2d @ B_2d = [6, 7]
```

最终输出：

```text
[6, 7] -> [2, 3, 7]
```

### 10.3 PrepareTensor

`PrepareTensor()` 负责把输入张量变成二维 matmul 输入。

逻辑：

```text
if need_transpose:
    allocate temporary tensor
    TransposeFunctor::Compute
    reshape view to [dim0, dim1]
else:
    reshape view to [dim0, dim1]
```

其中：

```text
Transpose 是真实数据重排
Reshape/CopyFrom 通常只是 view 变化
```

因此性能开销主要来自需要 transpose 的情况。

A 侧需要变成：

```text
[free..., contract...] -> [M, K]
```

B 侧需要变成：

```text
[contract..., free...] -> [K, N]
```

### 10.4 调用 MUDNN MatMul

准备好二维输入后，kernel 调用 MUDNN 的 `mMatMul`：

```text
mMatMul op
op.SetTranspose(false, false)
op.SetAlpha(1.0)
op.SetBeta(0.0)
op.Run(handle, mt_out, mt_a, mt_b, mm)
```

因此 `MusaTensorDot` 的核心计算不是手写乘法 kernel，而是：

```text
transpose/reshape + mMatMul
```

### 10.5 输出 reshape

MatMul 输出是二维：

```text
[M, N]
```

最终输出是高维：

```text
output_dims
```

二者元素数量相同。

kernel 会尝试把已经分配好的 output buffer 临时看成二维 view：

```text
output as [M, N]
```

这样 `mMatMul` 可以直接写入 output 的同一块内存。用户最终看到的 shape 仍然是高维 `output_dims`。

如果 view 不成功，则使用临时 tensor：

```text
matmul_temp [M, N]
```

计算后再 reshape/copy 到 output。

## 11. MusaTensorDotBias kernel 执行流程

`MusaTensorDotBias` 注册在：

```text
musa_ext/kernels/fusion/musa_tensordot_bias_op.cc
```

输入：

```text
a: T
b: T
bias: T
```

执行流程和 `MusaTensorDot` 基本相同：

```text
1. ComputeTensorDotDims
2. PrepareTensor(A)
3. PrepareTensor(B)
4. PrepareBiasTensor
5. DoMatMulWithBias
```

区别在于最终调用：

```text
mMatMul::RunWithBiasAdd
```

也就是：

```text
A_2d @ B_2d + bias
```

bias 主要按 matmul 的 N 维处理：

```text
A_2d = [M, K]
B_2d = [K, N]
output_2d = [M, N]
bias = [N]
```

最自然、最稳的 bias shape 是：

```text
[N]
```

当前实现中，标量 broadcast 和部分 2D bias 处理分支仍有 TODO，因此使用 `MusaTensorDotBias` 时应优先保证 bias 可以直接作为 `[N]` 使用。

## 12. Dense-like 示例

假设：

```text
A shape = [batch, seq, hidden] = [2, 3, 4]
B shape = [hidden, out]        = [4, 5]
axes_a = [-1]
axes_b = [0]
```

语义：

```python
output = tf.tensordot(A, B, axes=([-1], [0]))
```

输出 shape：

```text
A 非收缩轴 = [2, 3]
B 非收缩轴 = [5]
output = [2, 3, 5]
```

TensorFlow 展开后大致为：

```text
Shape(A) = [2, 3, 4]
GatherV2 free axes [0, 1] -> [2, 3]
Prod -> 6
GatherV2 contract axis [2] -> [4]
Prod -> 4
Pack -> [6, 4]
Reshape(A) -> [6, 4]
MatMul([6, 4], [4, 5]) -> [6, 5]
Reshape -> [2, 3, 5]
```

TFMUSA 融合后：

```text
MusaTensorDot(A, B, axes_a=[-1], axes_b=[0])
```

kernel 内部：

```text
A -> [6, 4]
B -> [4, 5]
mMatMul -> [6, 5]
view/reshape -> [2, 3, 5]
```

如果还有 bias：

```text
bias shape = [5]
```

则进一步融合为：

```text
MusaTensorDotBias(A, B, bias)
```

内部调用：

```text
mMatMul::RunWithBiasAdd
```

## 13. 当前实现的主要限制

### 13.1 axes_b 在 graph fusion 中默认是 [0]

当前 `MusaTensorDotFusion` 从 A 侧 `GatherV2` 提取 `axes_a`，但 `axes_b` 默认设置为：

```text
[0]
```

因此当前 graph fusion 更适合：

```text
[..., K] x [K, N]
```

这种权重矩阵第 0 维作为收缩轴的场景。

### 13.2 B 侧主要被当作 weight

匹配时 B 侧要求是 weight-like 节点：

```text
Identity
ReadVariableOp
Const
Reshape
```

这说明当前优化主要服务模型权重场景，而不是完全动态的任意 B 张量。

### 13.3 Bias broadcast 不完整

`MusaTensorDotBias` 中对标量 bias broadcast 和部分 2D bias 提取逻辑仍有 TODO。

因此推荐 bias 使用：

```text
[N]
```

其中：

```text
N = B 自由轴尺寸乘积 = b_batch_size
```

## 14. 总结

TFMUSA 的 tensordot 融合可以概括为：

```text
识别 TensorFlow 展开的 tensordot 子图
  ↓
提取原始输入、权重、axes 和 dtype
  ↓
删除 Shape/GatherV2/Prod/Pack/Reshape/MatMul/ConcatV2 等中间节点
  ↓
替换成 MusaTensorDot
  ↓
如果后面接 BiasAdd，进一步替换成 MusaTensorDotBias
  ↓
kernel 内部将高维 tensordot 降成二维 matmul
  ↓
调用 MUDNN mMatMul 或 RunWithBiasAdd
  ↓
将结果 view/reshape 成原始高维输出
```

它不是把所有小 op 直接生成一个新的 device kernel，而是恢复 tensordot 的高级语义，然后走 MUSA 后端更直接、更高效的 matmul 路径。
