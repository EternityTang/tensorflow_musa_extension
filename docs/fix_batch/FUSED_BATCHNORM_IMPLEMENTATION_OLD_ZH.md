# `musa_fused_batchnorm_op.cc` 详细实现说明（旧版保留稿）

这份文档保留的是之前那版“按实现细节展开”的说明风格，重点解释：

- `FusedBatchNorm` / `FusedBatchNormGrad` 在 MUSA 插件里是怎么实现的
- 为什么一个 BatchNorm 还需要 `Permute`
- 当前实现和旧实现相比，核心变化在哪里
- 为什么这个文件是 `.cc` 而不是 `.mu`

对应代码文件：

- `tensorflow_musa_extension/musa_ext/kernels/nn/musa_fused_batchnorm_op.cc`

---

## 1. 先给结论

这份实现不是手写 `.mu` kernel，而是：

- TensorFlow 的 Op 封装层
- 负责属性读取、输入输出张量分配、layout 处理
- 把 TensorFlow `Tensor` 包装成 muDNN 的 `mTensor`
- 再调用 muDNN 的 `mBatchNorm` 和 `mPermute`

也就是说，它本质上是“框架适配层 + 调库层”，不是“设备 kernel 本体”。

所以它是 `.cc` 文件，而不是 `.mu` 文件。

---

## 2. 为什么 BatchNorm 还需要 `Permute`

这是这份实现里最关键的部分。

TensorFlow 里常见的 4D 特征图布局有两种：

- `NHWC`：`[N, H, W, C]`
- `NCHW`：`[N, C, H, W]`

BatchNorm 的归一化语义本质上是“按 channel 做归一化”。所以如果把：

- `[N, C, H, W]`

转换成：

- `[N, H, W, C]`

那么 channel 的语义并没有变，只是 channel 维的位置换了。

在当前这套 MUSA + muDNN 环境里，直接走原生 `NCHW` BatchNorm 路径不够稳定；但 `NHWC` 路径是稳定的。所以当前实现采用的策略是：

1. 如果输入本来就是 `NHWC`，直接算
2. 如果输入是 `NCHW`，先转成 `NHWC`
3. 在 `NHWC` 上调用 muDNN BatchNorm
4. 再把输出转回 `NCHW`

这就是为什么一个 BatchNorm 算子里会出现 `Permute`。

需要注意的是：

- `scale`
- `offset`
- `mean`
- `var`

这些参数本来就是一维 `[C]`，只和 channel 数有关，不带空间维，所以它们不需要做 `Permute`。

---

## 3. 文件整体结构

`musa_fused_batchnorm_op.cc` 可以分成 5 块：

1. 头文件和工具函数
2. `PermuteTensorOnMusa` 辅助函数
3. 前向算子 `MusaFusedBatchNormOp`
4. 反向算子 `MusaFusedBatchNormGradOp`
5. TensorFlow kernel 注册

真正影响行为的核心是：

- `PermuteTensorOnMusa`
- 前向 `Compute()`
- 反向 `Compute()`

---

## 4. `PermuteTensorOnMusa` 在做什么

文件起始位置定义了：

- `PermuteTensorOnMusa(OpKernelContext* ctx, const Tensor& input, Tensor* output, const std::vector<int64_t>& perm)`

它的工作很纯粹：

1. 检查输入 rank 和 permutation 长度是否一致
2. 从 `ctx` 里拿 muDNN handle
3. 把 TensorFlow `Tensor` 包成 muDNN 的 `mTensor`
4. 用 `mPermute` 配置 stride / 维度变换
5. 在 MUSA 上执行 `Permute::Run`

同时，文件里还定义了两组固定 permutation：

- `kPermNchwToNhwc = {0, 2, 3, 1}`
- `kPermNhwcToNchw = {0, 3, 1, 2}`

对应含义分别是：

- `[N, C, H, W] -> [N, H, W, C]`
- `[N, H, W, C] -> [N, C, H, W]`

这部分代码本身不做 BatchNorm 计算，只负责 layout 转换。

---

## 5. 前向实现怎么工作

前向类是：

- `MusaFusedBatchNormOp`

它在构造函数里会先读取几个关键属性：

- `epsilon`
- `is_training`
- `exponential_avg_factor`
- `data_format`

其中 `data_format` 会被压成一个布尔量：

- `is_nhwc_ = (data_format_str == "NHWC")`

这样在真正执行时就能快速判断当前输入布局。

### 5.1 前向执行流程

前向 `Compute()` 的主流程可以概括成下面几步：

1. 读取输入张量：
   - `x`
   - `scale`
   - `offset`
   - `estimated_mean`
   - `estimated_variance`
2. 分配输出张量：
   - `y`
   - `batch_mean`
   - `batch_var`
   - `saved_mean`
   - `saved_var`
3. 获取 muDNN handle，并显式关闭 TF32：
   - `handle.SetAllowTF32(false)`
4. 如果当前是 `NCHW`：
   - 分配 `x_nhwc`
   - 做 `NCHW -> NHWC`
   - 如果输出也需要中间缓冲，则分配 `y_nhwc`
5. 把输入输出和参数包装成 `mTensor`
6. 调用 BatchNorm：
   - 训练态走 `RunComposite`
   - 推理态走 `RunPure`
7. 如果原始布局是 `NCHW`：
   - 再把 `y_nhwc` 转回 `y`

### 5.2 为什么前向里要显式关闭 TF32

这里有一行很关键：

- `handle.SetAllowTF32(false)`

它的目的不是为了“让算子更快”，而是为了优先保证精度稳定。

BatchNorm 属于数值比较敏感的算子，如果让它走 TF32，精度风险会放大。所以当前实现里对这条路径做了保守处理：先关掉 TF32，优先保证正确性。

---

## 6. 反向实现怎么工作

反向类是：

- `MusaFusedBatchNormGradOp`

它和前向的思路基本一致，只是输入输出换成了梯度相关张量。

### 6.1 反向输入输出

反向里常见的输入包括：

- `y_backprop` / `dy`
- `x`
- `scale`
- `saved_mean`
- `saved_var`

输出包括：

- `x_backprop` / `dx`
- `scale_backprop`
- `offset_backprop`
- 以及一些兼容 TensorFlow 接口的占位输出

### 6.2 反向执行流程

反向 `Compute()` 主要做这些事：

1. 读取输入和分配输出
2. 获取 handle，并显式关闭 TF32
3. 如果输入布局是 `NCHW`：
   - 把 `x` 转成 `x_nhwc`
   - 把 `dy` 转成 `dy_nhwc`
4. 把 TensorFlow `Tensor` 包装成 `mTensor`
5. 调用 muDNN 的：
   - `RunBwd`
6. 如果原始布局是 `NCHW`：
   - 把 `dx_nhwc` 再转回 `dx`

这样前向和反向都统一落在 `NHWC` 稳定路径上。

---

## 7. 当前实现相对于旧实现的核心变化

旧实现的问题不是 BatchNorm 数学公式写错了，而是：

- 直接走 `NCHW` 路径时，后端实现不够稳定

所以当前实现的核心变化不是“重写了 BatchNorm 算法”，而是：

- 把 `NCHW` 路径统一改成 `NCHW -> NHWC -> BatchNorm -> NHWC -> NCHW`

而且这套处理不只做在前向，也同步做在反向里。

这样做的好处是：

- 前向和反向行为一致
- 可直接复用已经验证过的 `NHWC` 路径
- 精度问题更容易收敛

代价是：

- 多一次输入 `Permute`
- 多一次输出 `Permute`
- 中间临时张量更多
- 大 shape 时会多一些显存和 workspace 压力

这是一个典型的工程取舍：

- 先用 layout fallback 换稳定性
- 再看后面是否有必要继续优化性能

---

## 8. 为什么它不是 `.mu`

这个问题很容易混淆。

这个仓库里：

- `.mu` 文件通常用于手写 MUSA device kernel
- `.cc` 文件通常用于 TensorFlow Op 封装、参数处理、张量转换、调用 muDNN/runtime API

`musa_fused_batchnorm_op.cc` 属于第二种。

它没有自己实现底层逐元素或 block 级别的设备内核，而是把工作交给：

- `mBatchNorm`
- `mPermute`

所以它没有对应的 `.mu` 文件是正常的。

---

## 9. 关于 `mudnn.h` 报错

之前构建里出现过：

```cpp
fatal error: mudnn.h: No such file or directory
```

这个问题不是“代码里 include 写错了”，而是构建阶段的头文件搜索路径没有覆盖到实际安装目录。

实际环境里，muDNN 头文件通常位于类似这样的目录：

- `/usr/local/musa/include/mudnnc/`
- `/usr/local/musa/include/mudnncxx/`

如果 CMake 只加了：

- `/usr/local/musa/include`

而没有把具体子目录加进去，就可能出现 `<mudnn.h>` 找不到的问题。

所以这类报错本质上是构建系统的 include path 问题，不是 BatchNorm 算子实现本身的逻辑错误。

---

## 10. 小结

这份文件的本质可以总结成一句话：

- 用 TensorFlow C++ Op 包装层，把 BatchNorm 前向和反向统一导向 muDNN 的稳定 `NHWC` 路径，并在必要时通过 `Permute` 兼容 `NCHW` 输入输出。

也正因为如此：

- 它是 `.cc`
- 它需要 `Permute`
- 它的修复重点是 layout fallback，而不是重写 BatchNorm 数学公式

如果后面继续深入优化，这条线最值得关注的方向通常是：

- 是否能在后端提供稳定的原生 `NCHW` 路径
- 是否能减少 `Permute` 带来的中间内存和额外开销
