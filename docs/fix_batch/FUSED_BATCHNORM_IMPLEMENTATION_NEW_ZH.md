# FusedBatchNorm 与 Conv2D 修复记录

本文把这次排查过的两条算子线整理到同一份文档里，统一按下面的顺序展开：

1. 问题出现
2. 问题定位
3. 代码修改
4. 修改之后的测试
5. 最终结果

本文覆盖的代码文件如下：

- `musa_ext/kernels/nn/musa_fused_batchnorm_op.cc`
- `musa_ext/kernels/math/musa_conv2d_op.cc`
- `musa_ext/kernels/math/musa_conv2d_backward_op.cc`

所有相关日志已经统一收拢到：

- `tensorflow_musa_extension/logs/operator_fix_2026-03-26`

其中：

- `batchnorm/` 保存 FusedBatchNorm 的单测、整网精度、整网推理、整网 profile 日志
- `conv2d/` 保存 Conv2DBackpropInput 单测和全量 ops 测试日志

需要先说明两点：

- 本轮修复中，FusedBatchNorm 和 Conv2D 都优先解决了正确性问题。
- 只有 FusedBatchNorm 留下了可信的修复后整网性能数据；没有同条件的修复前基线，因此不能严谨地写出“性能提升了多少百分比”。

---

## 一、FusedBatchNorm

### 1. 问题出现

`FusedBatchNormV3` 在这套 MUSA 环境里，`NHWC` 路径相对稳定，但 `NCHW` 路径不稳定。

用户侧观察到的问题主要有两类：

- 单测中 `NCHW` 场景更容易出现数值不一致。
- 整网测试里如果图中有 BatchNorm，虽然不一定每次都直接 crash，但精度风险集中在 `NCHW` 路径。

这类问题表面上看像是“BatchNorm 算子本身有错”，但进一步看后会发现，真正脆弱的是“数据布局 + 后端实现路径”的组合，而不是 BatchNorm 的数学公式本身。

---

### 2. 问题定位

问题定位的结论是：

- `NHWC` 路径可以稳定复用 muDNN 的 BatchNorm 实现。
- `NCHW` 直接喂给当前后端路径时，不够稳定。
- 因此需要把 `NCHW` 输入转换成 `NHWC`，沿用已经稳定的 `NHWC` 路径，再把结果转回去。

这里最关键的点是：为什么一个 BatchNorm 还需要 `Permute`？

原因并不复杂。TensorFlow 常见的 4D 特征图布局有两种：

- `NHWC`：`[N, H, W, C]`
- `NCHW`：`[N, C, H, W]`

BatchNorm 的归一化语义本质上是“按 channel 做归一化”。如果把：

- `[N, C, H, W]`

转换成：

- `[N, H, W, C]`

那么 channel 仍然还是同一个 channel，只是维度位置从第 2 维挪到了最后一维。也就是说：

- 数学语义没有变
- 只是 layout 变了

所以这次修复采用的是工程上更稳的策略：

1. `NCHW -> NHWC`
2. 在 `NHWC` 上执行 muDNN BatchNorm
3. `NHWC -> NCHW`

此外，`scale / offset / mean / var` 这些参数本来就是一维 `[C]`，不带空间维，所以它们不需要做 `Permute`。

---

### 3. 代码修改

核心修改全部在：

- `musa_ext/kernels/nn/musa_fused_batchnorm_op.cc`

关键点如下。

#### 3.1 增加统一的 Permute 辅助函数

文件最前面提供了：

- `PermuteTensorOnMusa(...)`

以及两组固定 permutation：

- `kPermNchwToNhwc = {0, 2, 3, 1}`
- `kPermNhwcToNchw = {0, 3, 1, 2}`

对应代码位置：

- `PermuteTensorOnMusa`：第 18 行
- `kPermNchwToNhwc`：第 46 行
- `kPermNhwcToNchw`：第 47 行

这部分的作用很单纯，就是在 MUSA 上做张量布局转换，不负责 BatchNorm 计算本身。

#### 3.2 前向路径改成“必要时转 NHWC 再算”

前向算子的构造函数会先读 `data_format`：

- `is_nhwc_ = (data_format_str == "NHWC")`

位置：

- 第 61 行

在 `Compute()` 里，前向算子做了这几件事：

1. 显式关闭 TF32
2. 如果输入本来就是 `NHWC`，直接创建 `mTensor` 并调用 BatchNorm
3. 如果输入是 `NCHW`，先分配一个 `x_nhwc`
4. 调 `PermuteTensorOnMusa(ctx, x, &x_nhwc, kPermNchwToNhwc)`
5. 在 `NHWC` 上调用：
   - 训练态：`RunComposite`
   - 推理态：`RunPure`
6. 如果原始布局是 `NCHW`，再把输出 `y_nhwc` 转回 `y`

关键位置：

- `handle.SetAllowTF32(false)`：第 94 行
- `if (!is_nhwc_)`：第 112 行
- 输入转置：第 125 行
- 训练态 `RunComposite`：第 172 行
- 推理态 `RunPure`：第 195 行
- 输出转回 `NCHW`：第 203 行

#### 3.3 反向路径也同步改成“必要时转 NHWC 再算”

反向逻辑同样不能只修一半，否则前向和反向会走不同的 layout 策略，训练时仍然会出问题。

因此在反向算子里也做了相同思路：

1. 关闭 TF32
2. 如果是 `NCHW`，把 `x` 和 `dy` 都转成 `NHWC`
3. 调 `RunBwd`
4. 如果原始布局是 `NCHW`，把 `dx` 从 `NHWC` 转回去

关键位置：

- 反向 `is_nhwc_` 读取：第 227 行
- `handle.SetAllowTF32(false)`：第 259 行
- `if (!is_nhwc_)`：第 287 行
- `x -> x_nhwc`：第 301 行
- `dy -> dy_nhwc`：第 303 行
- `RunBwd`：第 336 行
- `dx_nhwc -> dx`：第 344 行

### 4. 修改之后的测试

#### 4.1 单算子测试

单测入口：

- `test/fusion/fused_batchnorm_test.py`

日志：

- `tensorflow_musa_extension/logs/operator_fix_2026-03-26/batchnorm/fused_batchnorm_test.log`

结果：

- `Ran 8 tests in 10.156s`
- `OK (skipped=2)`

也就是说：

- 有效执行的 6 个 case 全部通过
- 没有失败项
- 没有 error

#### 4.2 整网精度测试

整网日志：

- `batchnorm/whole_model_accuracy_compare.log`

关键结果如下：

- `结果: PASSED`
- `最大绝对误差 = 2.47e-06`
- `平均绝对误差 = 4.65e-08`
- `余弦相似度 = 1.0000000000`
- `不匹配元素数 = 0 / 3200`

这说明在当前 prunedGraph 场景下，修复后的 BatchNorm 路径已经满足精度要求。

#### 4.3 整网推理与 profile

整网推理日志：

- `batchnorm/whole_model_inference_only.log`

整网 profile 日志：

- `batchnorm/whole_model_profile_ops.log`

从 profile 结果看：

- `FusedBatchNormV3` 共出现 `4` 次
- 总耗时 `0.462 ms`
- 平均耗时 `0.115 ms`
- 总占比 `1.89%`

单次节点 profile 中，4 个 BatchNorm 节点耗时分别大约是：

- `0.134 ms`
- `0.117 ms`
- `0.110 ms`
- `0.101 ms`

这说明修复后的 BatchNorm 不仅通过了整网精度，整网 trace 中也能稳定看到实际执行。

---

### 5. 最终结果

FusedBatchNorm 这条线的最终结论是：

- `NCHW` 不再直接走不稳定路径，而是转到 `NHWC` 再调用稳定实现
- 前向和反向都做了同样的 fallback 处理
- 单算子测试通过
- 整网精度测试通过
- 整网 profile 中 BatchNorm 节点能正常执行并统计到耗时

性能方面：

- 当前保留了修复后的整网推理与 profile 数据
- 但没有采集“修复前、同条件、同模型、同卡、同轮数”的可信基线
- 因此不能严谨地写出“性能提升了 X%”

换句话说，这一轮对 BatchNorm 的收益是：

- 正确性已经闭环
- 整网可用性已经闭环
- 性能数据只有修复后的绝对值，没有可信的前后百分比对比

---

## 二、Conv2D / Conv2DBackpropInput

### 1. 问题出现

在全量 ops 测试里，最早暴露出来的是 `conv2d_backward_op_test.py` 相关失败。

当时失败的 4 个核心 case 是：

- `testConv2DBackpropInputValidNCHW`
- `testConv2DBackpropInputValidNHWC`
- `testConv2DGradientNCHW`
- `testConv2DGradientNHWC`

这组失败有一个很重要的信号：

- 既有 `NCHW`
- 也有 `NHWC`

所以问题不可能只解释成“`NCHW` 布局本身有问题”。`NCHW` 确实更脆弱，但这次根因比单纯的 layout 更深。

---

### 2. 问题定位

定位结果分成两层。

#### 2.1 `NCHW` 路径确实更脆弱

在当前实现里，`Conv2D` 和 `Conv2DBackpropInput` 的 `NCHW` 并不是完全独立的一套路，而是借助：

- `NCHW -> NHWC`
- 计算
- `NHWC -> NCHW`

来做 fallback。

相关位置：

- `musa_conv2d_op.cc` 中的前向 fallback：第 354 到 368 行附近
- `musa_conv2d_backward_op.cc` 中的 backward-input fallback：第 494 到 500 行附近

这意味着：

- `NCHW` 会多一次或两次 `Permute`
- 临时张量更多
- workspace 和显存压力更高

因此从工程经验上说，`NCHW` 路径会比 `NHWC` 更容易放大问题。

#### 2.2 但这次真正的根因不只是 NCHW

继续排查之后发现，即使把场景收缩到：

- `NHWC`
- `float32`
- `Conv2DBackpropInput`

依然能复现数值严重错误。

定位结论是：

- `GetRecommendBackwardDataAlgorithm(...)` 给 `float32/double` 选出的 backward-data 算法不稳定
- `GetRecommendForwardAlgorithm(...)` 给 `float32/double` 选出的 forward 算法也不理想

它们的坏表现有两种：

1. 数值错误
2. workspace 申请异常大，导致 OOM

定位阶段的一个最小复现里，修复前：

- `Conv2DBackpropInput` 的 `max_diff` 约为 `28451.2`

这不是普通的浮点抖动，而是算法路径本身就不对。

另外，在 `GradientTape` 路径下，前向 `Conv2D` 还会出现 workspace 申请离谱的问题，曾经观察到几十 GB 级别的 workspace 请求，这会把当前 TensorFlow 进程的显存分配直接打爆。

所以这一轮 Conv2D 的核心问题不是：

- padding 算错
- output shape 算错
- filter 布局算错

而是：

- 后端“推荐算法”在当前 dtype / shape / layout 组合上选到了不可靠的路径

---

### 3. 代码修改

Conv2D 这一轮只改了“算法选择策略”，没有动 shape、padding 或反向公式。

#### 3.1 固定 backward-data 算法

修改文件：

- `musa_ext/kernels/math/musa_conv2d_backward_op.cc`

关键位置：

- `GetRecommendBackwardDataAlgorithm(...)`：第 148 行
- 强制改成 `IMPLICIT_GEMM`：第 158 行

修改思路：

- 先仍然调用 muDNN 的推荐算法接口
- 但当输入类型是 `DT_FLOAT` 或 `DT_DOUBLE` 时，不再盲信推荐值
- 直接把 backward-data 算法固定为：
  - `mConvolution::AlgorithmBwdData::IMPLICIT_GEMM`

这一步修掉的是：

- `Conv2DBackpropInputValidNHWC`
- `Conv2DBackpropInputValidNCHW`

对应的 `dx` 数值错误问题。

#### 3.2 固定 forward 算法

修改文件：

- `musa_ext/kernels/math/musa_conv2d_op.cc`

关键位置：

- `GetRecommendForwardAlgorithm(...)`：第 158 行
- 强制改成 `IMPLICIT_GEMM`：第 168 行

修改思路和 backward 一样：

- 保留推荐算法接口调用
- 但对 `DT_FLOAT` / `DT_DOUBLE` 不直接采用推荐值
- 强制使用：
  - `mConvolution::Algorithm::IMPLICIT_GEMM`

之所以连前向也要改，是因为：

- `Conv2DGradient*` 测试并不只依赖 backward
- 它们还会先走前向路径
- 如果前向还在选择异常 workspace 的算法，梯度测试仍然会因为 OOM 或错误路径失败

#### 3.3 这轮没有改动的部分

为了避免误解，这里也明确写出“没改什么”。

这轮没有改动：

- padding 计算逻辑
- dilation / stride 逻辑
- filter backward 路径
- `NCHW/NHWC` 的 shape 推导
- `Permute` 的基本结构

也就是说，这轮是一次非常聚焦的修复：

- 只修正 float32/double 的前向与 backward-data 算法选择

---

### 4. 修改之后的测试

#### 4.1 定位阶段的最小复现

在最小复现里，修复前观察到：

- `max_diff ≈ 28451.2`

修复后同类复现观察到：

- `max_diff = 9.5367431640625e-07`
- `mean_diff = 1.15629518404603e-07`

这说明问题不是“阈值太严”，而是算法路径修正之后，结果回到了正常浮点误差量级。

#### 4.2 `conv2d_backward_op_test.py`

日志：

- `tensorflow_musa_extension/logs/operator_fix_2026-03-26/conv2d/conv2d_backward_op_test.log`

结果：

- `Ran 19 tests in 43.993s`
- `OK (skipped=6)`

也就是说：

- 有效执行的 13 个 case 全部通过
- 原来集中失败的 backward / gradient case 已经清掉

#### 4.3 全量 ops 测试

日志：

- `tensorflow_musa_extension/logs/operator_fix_2026-03-26/conv2d/ops_full_test.log`

结果：

- `Ran 942 tests in 55.133s`
- `OK (skipped=229)`

这一步很关键，因为它说明：

- 之前全量测试里由 Conv2D backward 触发的 4 个失败已经全部消失
- 修复没有在全量回归里引入新的失败

#### 4.4 为什么测试固定在 2 号卡，并且串行跑

这次所有关键 GPU 回归最终都固定在：

- `MTHREADS_VISIBLE_DEVICES=2`

并且按单进程、串行方式执行。

原因不是“并行一定更差”，而是当前约束是：

- 只分到同一张 2 号卡

如果在同一张 GPU 上并行拉多个 TensorFlow 进程，会出现：

- workspace 竞争
- BFC allocator 竞争
- 显存碎片化
- 假性 OOM

所以这次串行跑的目标是：

- 先拿到可信的正确性结果

如果未来有多张独占卡可用，那么“一进程一张卡”的并行方案当然会更优；但“多进程挤同一张卡”并不是这类回归测试的最优解。

---

### 5. 最终结果

Conv2D / Conv2DBackpropInput 这条线的最终结论是：

- 问题表面上在 `NCHW` 更明显，但根因并不只属于 `NCHW`
- 真正的问题是 `float32/double` 的 forward / backward-data 推荐算法不稳定
- 通过把算法固定为 `IMPLICIT_GEMM`，数值错误和异常 workspace 问题同时得到控制
- `conv2d_backward_op_test.py` 已通过
- 全量 `ops` 测试已通过

性能方面：

- 这轮没有专门做 Conv2D 修复前后的同条件 benchmark
- 因此不能严谨地写出“性能提升 X%”
- 当前可以确认的是：修复后 correctness 已经闭环，全量测试通过

---

## 三、日志目录整理结果

本次修复相关日志统一放在：

- `tensorflow_musa_extension/logs/operator_fix_2026-03-26`

当前目录结构如下：

- `batchnorm/fused_batchnorm_test.log`
- `batchnorm/whole_model_accuracy_compare.log`
- `batchnorm/whole_model_inference_only.log`
- `batchnorm/whole_model_profile_ops.log`
- `batchnorm/whole_model_traces/`
- `conv2d/conv2d_backward_op_test.log`
- `conv2d/ops_full_test.log`

这份目录的用途是：

- BatchNorm：保留单算子 + 整网精度 + 整网推理 + 整网 profile
- Conv2D：保留关键回归单测 + 全量回归结果

如果后面还要继续补别的算子，可以继续沿用这个结构，把不同算子的日志按子目录继续收进去。
