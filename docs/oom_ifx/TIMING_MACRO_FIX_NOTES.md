Timing 宏 / Shape 链路问题定位说明
日期：2026-03-30

1. 当前结论

这次长跑随机崩溃，主线问题不在 `Unpack`，而在 `Shape` 相关的 shape tensor 链路。

当前最可疑的实际数据流是：

`Shape -> StridedSlice -> Mul -> Pack -> Reshape`

并且同一段 shape 值还会喂给：

`Pack -> Fill`

所以当 shape 值被污染时，可能先报：

- `musa_fill_op.cc:83 : Dimension ... must be >= 0`
- `RankMixerBlock_0/RankMixerBlock_0_LN_1/Reshape : Dimension size must be non-negative`

这两个报错是同一条 shape 链路上的不同出口，不应当分开看。

2. 为什么暂时不把重点放在 Unpack

已经解析过当前 `graph_def.pb`，这次出问题的 `fwffm_pbp_mlp` 子图里，和崩溃节点直接相关的 shape 输入不是从 `Unpack` 来的。

当前这次问题的主线节点是：

- `Shape`
- `StridedSlice`
- `Mul`
- `Pack`
- `Reshape`
- `Fill`

因此，`Unpack` 现在最多算次要排查对象，不是这次修复的第一优先级。

3. 为什么 `ValidateStridedSliceOp(...)` 不是第一嫌疑

`ValidateStridedSliceOp(...)` 主要负责：

- 校验 `begin / end / strides`
- 规范化 slicing 参数
- 推导 `processing_shape / final_shape`

它处理的是“切片规则”和“输入张量的 shape 元信息”，不是最终要送进 `Reshape / Fill` 的业务 shape 值本身。

这次图里的 slice 也很简单：

- `strided_slice` 是 `[0:1]`
- `strided_slice_1` 是 `[1:2]`
- `strided_slice_2` 是 `[2:3]`

这种情况下，如果最后出现 `-1688747100` 这种随机大负数，更像是内存语义/可见性问题，不像 validate 本身算错。

4. 当前最强怀疑点

当前最需要盯的是 `StridedSlice<int32>` 的实现路径。

原因是：

- `Shape` 的输出被注册成了 `HostMemory("output")`
- 但当前 MUSA 的 `StridedSlice` 只把 `begin / end / strides` 标成了 host memory
- 没有把 `input / output` 在 `int32` shape-tensor 场景下单独处理

而当前实现里，后续会直接执行：

- `auto in_mt = CreateMTensor(input);`
- `auto out_mt = CreateMTensor(*result);`
- `op.ConfigDimStrideForSlice(...)`
- `op.Run(...)`

`CreateMTensor(...)` 当前只是把 Tensor 原始地址直接交给 muDNN，并不会区分这是 host tensor 还是 device tensor。

这对普通 device tensor 没问题，但对 `Shape` 产出的 host-visible int32 shape tensor 是高风险的。

4.1 为什么不是所有 Tensor 都应该在 device memory

这里需要先区分两个概念：

- `HostMemory`：张量数据存放在 CPU 主存里，由 host 直接读取
- `DeviceMemory`：张量数据存放在 MUSA 显存里，由 device kernel 直接读取

并不是所有 Tensor 都应该放在 `DeviceMemory`。

真正参与大规模数值计算的大块数据，通常应该放在 `DeviceMemory`，例如：

- feature / activation
- weight
- gradient

但像下面这类“描述怎么计算”的小张量，天然更适合放在 `HostMemory`：

- `shape`
- `dims`
- `begin / end / strides`
- 各种标量和控制类小 tensor

原因很简单：

- 它们数据量很小，放到显存没有明显收益
- 它们更多是被框架和算子实现当作“元信息”读取，而不是拿来做大规模并行计算
- 很多下游算子本来就要求这类输入必须是 host 可见的

这次问题里最关键的一句话就是：

**本来应该一直待在 host 侧的 shape tensor，中途被错误地当成了 device tensor 来包和传。**

这不是说 `HostMemory` 很奇怪，恰恰相反，这里用 `HostMemory` 才是 TensorFlow 语义上的正常做法。

当前这条链路里，`HostMemory` 本来就是明确设计的一部分：

- `Shape` 的输出是 `HostMemory("output")`
- `Reshape` 的第二个输入 `shape` 是 `HostMemory("shape")`
- `Fill` 的 `dims` 是 `HostMemory("dims")`

因此，真正的问题不是“为什么 shape 不在显存里”，而是：

- 上游已经把它当 `HostMemory`
- 中间的 `StridedSlice<int32>` / `Pack<int32>` 却没有延续这层语义
- 反而把它按普通 device tensor 路径交给了 `CreateMTensor(...)` / muDNN

这才导致了 shape 值在长跑中被污染。

4.2 这次真正出问题的简图

```text
正常语义应该是：

  大张量数据
    ↓
  Shape
    ↓
  shape tensor（HostMemory）
    ↓
  StridedSlice<int32>（应该继续走 HostMemory）
    ↓
  Pack<int32>（应该继续走 HostMemory）
    ↓
  Reshape(shape) / Fill(dims)


旧实现实际发生的是：

  Shape
    ↓
  shape tensor（HostMemory）
    ↓
  StridedSlice<int32>
    ↓
  CreateMTensor(input) / muDNN Permute
    ↓
  被当成普通 Device Tensor 处理
    ↓
  Pack<int32>
    ↓
  继续走普通 muDNN Concat 路径
    ↓
  Reshape(shape) / Fill(dims) 读到被污染的 shape 值
    ↓
  报错：负维度 / 超大维度 / 随机长跑崩溃


一句话总结：

  正常情况：shape tensor 应该一直待在 host 侧流转
  实际问题：shape tensor 中途被错误带进了 device 路径
```

5. 目前判断哪些文件需要改

下面按优先级整理。

5.1 高优先级需要改

1. `tensorflow_musa_extension/musa_ext/kernels/array/musa_strided_slice_op.cc`

原因：

- 这是当前最强嫌疑点。
- `Shape(host output) -> StridedSlice<int32>` 这一步很可能存在 host/device 语义不匹配。
- 后续修复大概率要从这个文件下手。

重点可能落在：

- `int32` 的 kernel 注册方式
- `Compute()` 里的 `CreateMTensor(input)` / `CreateMTensor(*result)` 路径
- 是否需要对 shape-tensor 走 host-special path

2. `tensorflow_musa_extension/musa_ext/kernels/array/musa_pack_op.cc`

原因：

- `Pack<int32>` 是这条链上的下一站。
- 即使 `StridedSlice` 修正了，`Pack` 仍然要确认对 shape-tensor 的处理是否一致。
- 当前已经能看到 `Reshape_5/Reshape_6` 和 `RankMixerBlock_0_LN_1/Reshape` 都在消费 `Pack<int32>` 的输出。

重点可能落在：

- `Pack<int32/int64>` 是否要区分 shape-tensor 场景
- 当前同步是否只是绕过问题，还是语义上正确

5.2 可能需要联动改

3. `tensorflow_musa_extension/musa_ext/kernels/utils_op.cc`

原因：

- `CreateMTensor(...)` 的实现就在这里。
- 如果最终结论是“shape tensor 不能直接按普通 device tensor 包装成 mTensor”，那这里很可能要补通用辅助逻辑。

4. `tensorflow_musa_extension/musa_ext/kernels/utils_op.h`

原因：

- 如果要新增 host/device 区分 helper，或者抽公共的 shape-tensor 判定/拷贝逻辑，通常会放到这里。

5.3 验证时建议补测试

5. `tensorflow_musa_extension/test/ops/strided_slice_op_test.py`

原因：

- 现有测试主要是普通 slice 场景。
- 缺少 `Shape -> StridedSlice(int32)` 这类贴近当前问题的测试。

6. `tensorflow_musa_extension/test/ops/pack_op_test.py`

原因：

- 缺少 `Pack<int32>` 被当成 shape tensor 使用的链路测试。

7. `tensorflow_musa_extension/test/ops/reshape_op_test.py`

原因：

- 可以补 `Shape -> StridedSlice -> Pack -> Reshape` 的组合测试。

8. `tensorflow_musa_extension/test/ops/fill_op_test.py`

原因：

- 可以补 `Shape -> StridedSlice -> Pack -> Fill` 的组合测试。

6. 当前暂时不建议优先改的文件

1. `tensorflow_musa_extension/musa_ext/kernels/array/musa_reshape_op.cc`

原因：

- `Reshape` 当前更像“报错出口”，不是 shape 脏值的上游生产者。
- 现在先改它，容易变成在下游兜底，不容易抓住根因。

2. `tensorflow_musa_extension/musa_ext/kernels/array/musa_fill_op.cc`

原因：

- `Fill` 当前也是 shape 消费者，不是根因制造者。
- 先改它同样容易变成掩盖问题。

3. `tensorflow_musa_extension/musa_ext/kernels/array/musa_shape_op.cc`

原因：

- `Shape` 本身逻辑非常简单，直接从 `input.shape()` 取维度。
- 当前看不出它是第一个出错点。
- 它更像是链路起点，而不是 bug 本体。

4. `tensorflow_musa_extension/musa_ext/kernels/array/musa_pack_kernel.mu`

原因：

- 这是 `Unpack` 的 kernel 实现文件。
- 当前这次问题主线不是 `Unpack`，所以不建议先动它。

7. 当前推荐的改动顺序

如果下一步开始真正修代码，建议顺序是：

1. 先改 `musa_strided_slice_op.cc`
2. 再看 `musa_pack_op.cc`
3. 如果需要抽公共逻辑，再改 `utils_op.cc / utils_op.h`
4. 最后补 `strided_slice / pack / reshape / fill` 的组合测试

8. 当前代码基线说明

当前这轮已经不再保留“为了复现而注释同步”的状态，代码基线已经切回正式修复方向。

- `tensorflow_musa_extension/musa_ext/kernels/array/musa_pack_op.cc`
- `tensorflow_musa_extension/musa_ext/kernels/array/musa_strided_slice_op.cc`

另外：

- `musa_reshape_op.cc` 当前没有额外同步逻辑需要处理
- `musa_fill_op.cc` 当前也没有额外同步逻辑需要处理

9. 本轮已实施的修复（2026-03-30）

9.1 `musa_strided_slice_op.cc`

本轮已将 `StridedSlice<int32>` 改成 host-memory special path。

具体做法：

- `int32` 不再走原先普通的 MUSA device tensor 路径
- 注册改为：
  - `HostMemory("input")`
  - `HostMemory("begin")`
  - `HostMemory("end")`
  - `HostMemory("strides")`
  - `HostMemory("output")`
- 在实现中，对 `int32` 直接走 host 侧处理，不再把 `Shape` 产出的 host shape tensor 交给 `CreateMTensor(input)` / muDNN `Permute`
- host 路径改成插件内部自带的切片实现，直接按 `ValidateStridedSliceOp(...)` 已经规范化好的 `begin / end / strides` 做 host 侧拷贝
- `int32` host path 的分发方式改成模板特化，而不是 `if constexpr`，避免受当前编译标准影响

这里有一个重要的中间结论：

- 之前尝试直接复用 TensorFlow 内部 `HandleStridedSliceCase<CPUDevice, int32, NDIM>` 的方案，`./build.sh debug` 可以编译成功，也能生成 `libmusa_plugin.so`
- 但运行时 `tf.load_op_library(...)` 会失败，日志中报：
  - `undefined symbol: ... HandleStridedSliceCase ...`
- 本质上是“共享库文件生成成功”与“共享库运行时可被 TensorFlow 成功加载”是两件不同的事
- 因此当前保留的修法，已经不再依赖这个 TensorFlow 内部符号，而改为插件内自包含实现

这样做的目的，是把：

`Shape(host) -> StridedSlice<int32>`

这一步的 memory contract 改正确，避免 host shape tensor 在这里被误当成普通 device tensor 使用。

同时，原先为了复现问题而注释掉的同步也已经恢复。

9.2 `musa_pack_op.cc`

本轮已将 `Pack<int32>` 改成 host-memory special path。

具体做法：

- `int32` 的 `Pack` 注册改为：
  - `HostMemory("values")`
  - `HostMemory("output")`
- 在实现中新增 host 侧 `Pack` 拷贝逻辑，直接在 CPU/host 内存上完成 `Pack<int32>`
- 不再让这条 shape tensor 链继续走普通 muDNN Concat 路径

这样做的目的，是把：

`StridedSlice<int32> -> Pack<int32> -> Reshape/Fill`

这段链路也完整留在 host 侧，避免 `Reshape(shape)` 和 `Fill(dims)` 读到被污染的 shape 值。

同时，原先为了复现问题而注释掉的同步也已经恢复。

9.3 `musa_multiply_op.cc`

在 `StridedSlice<int32>` 和 `Pack<int32>` 改完之后，长跑稳定性已经明显提升，但在更长轮次下仍然可能在：

- `musa_fill_op.cc:83`
- `RankMixerBlock_0/RankMixerBlock_0_LN_1/Reshape`

再次报出负维度。

这说明：

- `StridedSlice<int32>` / `Pack<int32>` 确实修掉了主问题的一大部分
- 但 shape 链里仍然还有剩余节点在把 host-visible shape tensor 当作普通 MUSA tensor 处理

继续沿图往上游看，`Reshape` 的 shape 输入并不是 `StridedSlice -> Pack` 直接相连，中间还夹着 `Mul` 这类 shape 算术节点。

而当前 `Mul<int32>` 原本的实现是：

- 直接 `CreateMTensor(in0)`
- 直接 `CreateMTensor(in1)`
- 直接 `CreateMTensor(*output)`
- 再交给 muDNN `Binary::MUL`

这和之前 `StridedSlice<int32>` / `Pack<int32>` 的问题本质一样：

- shape tensor 本来应该继续走 host 语义
- 但 `Mul<int32>` 把它重新带回了普通 device 路径

因此本轮继续将 `Mul<int32>` 改成 host-memory special path：

- 注册改为：
  - `HostMemory("x")`
  - `HostMemory("y")`
  - `HostMemory("z")`
- 在实现中新增 host 侧 `int32` 乘法逻辑
- 支持按 TensorFlow `BCast` 规则做通用广播，不只是标量乘法

这样做的目的，是把：

`Shape(host) -> StridedSlice<int32> -> Mul<int32> -> Pack<int32> -> Reshape/Fill`

整条 shape 算术链都完整留在 host 侧，避免 `Mul<int32>` 成为新的污染源。

9.4 本轮没有改动的范围

本轮没有去改：

- `test.py` / 单算子测试文件
- `musa_reshape_op.cc`
- `musa_fill_op.cc`
- `musa_shape_op.cc`

也没有继续保留之前对整网入口脚本的试验性改法；整网脚本已经回退到更干净的状态。
