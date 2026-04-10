# 推理优化场景下的 OOM 与长跑稳定性问题复盘

日期：2026-03-31

## 1. 一句话结论

这次推理优化场景下的 OOM / 长跑随机崩溃问题，根因不是 `Reshape` 或 `Fill` 本身有逻辑错误，而是：

**本来应该一直待在 host 侧的 shape tensor，中途被错误地当成了 device tensor 来包和传。**

最终表现为：

- `musa_fill_op.cc:83 : Dimension ... must be >= 0`
- `RankMixerBlock_0/.../Reshape : Dimension size must be non-negative`

这两个报错不是两个独立问题，而是同一条 shape 链路在不同消费点上的两个出口。

---

## 2. 背景与现象

问题最初表现为：

- `build.sh release` 更容易长时间跑通
- `build.sh debug` 更容易在长跑时随机崩溃
- 崩溃轮次不固定，从几百轮到几十万轮都有可能
- 表面上最先报错的是 `Reshape` 或 `Fill`

一开始很容易误判成：

- `Reshape` 自身计算逻辑有 bug
- `Fill` 的 shape 校验不稳定
- 或者 timing 宏直接把某个算子“改坏了”

但从多轮日志和图结构分析看，真正的问题不是下游消费者，而是上游 shape 值已经被污染。

---

## 3. 最终根因

### 3.1 根因概述

本次问题的主线是 shape tensor 链路，而不是普通数值计算链路。

最关键的数据流是：

```text
Shape -> StridedSlice -> Mul -> Pack -> Reshape
                           └──────────────-> Fill
```

其中：

- `Shape` 产出的是 shape 元信息，不是业务数据本身
- `Reshape(shape)` 和 `Fill(dims)` 最终都要求输入是 host 可见的 shape/dims
- 旧实现中间的 `StridedSlice<int32>` / `Pack<int32>` 走了普通 MUSA device 路径
- 结果是 host-visible shape tensor 在中途被错误地按 device tensor 处理

这就是长跑下随机出现负维度、大维度、甚至不同轮次报错点变化的根本原因。

### 3.2 为什么不是所有 Tensor 都应该在 device memory

这里必须先把语义说清楚：

- `HostMemory`：张量数据放在 CPU 主存里，由 host 直接读取
- `DeviceMemory`：张量数据放在 MUSA 显存里，由 device kernel 直接读取

并不是所有 Tensor 都应该在 `DeviceMemory`。

真正适合常驻 device 的，一般是大块数值数据，例如：

- feature / activation
- weight
- gradient

而像下面这类“描述怎么计算”的小张量，天然更适合留在 host 侧：

- `shape`
- `dims`
- `begin / end / strides`
- 各种标量和控制类小 tensor

原因是：

- 它们数据量很小，放到显存没有明显收益
- 它们更多是被框架和算子实现当作“元信息”读取
- 很多下游算子本来就要求这类输入必须是 host 可见的

这次问题里，`HostMemory` 本来就是明确设计的一部分：

- `Shape` 的输出是 `HostMemory("output")`
- `Reshape` 的 `shape` 是 `HostMemory("shape")`
- `Fill` 的 `dims` 是 `HostMemory("dims")`

真正的错误不是“为什么 shape 不在显存里”，而是：

- 上游已经把它当成 `HostMemory`
- 中间却没有延续这层语义
- 反而把它按普通 device tensor 交给 `CreateMTensor(...)` / muDNN

### 3.3 这次问题的简图

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
```

---

## 4. 为什么 release 更容易过、debug 更容易炸

这次不是简单的“release 对、debug 错”。

更准确地说：

- release 没那么容易把问题暴露出来
- debug timing / instrumentation 改变了执行节奏
- 原本隐藏的 host/device contract 问题在 debug 下更容易被放大

因此现象上表现为：

- release 长跑更容易通过
- debug 更容易在不同轮次随机炸掉

但这并不意味着 release 本身逻辑一定正确，只能说明 timing 路径让潜在问题更容易暴露。

---

## 5. 排查路径与关键判断

### 5.1 为什么暂时不把重点放在 Unpack

解析过实际 `graph_def.pb` 后，可以确认本次出问题的 shape 输入主线并不是从 `Unpack` 来的。

和崩溃直接相关的节点更接近：

- `Shape`
- `StridedSlice`
- `Mul`
- `Pack`
- `Reshape`
- `Fill`

因此 `Unpack` 最多算次要排查对象，不是本轮修复优先级。

### 5.2 为什么 `ValidateStridedSliceOp(...)` 不是第一嫌疑

`ValidateStridedSliceOp(...)` 主要负责：

- 校验 `begin / end / strides`
- 规范化切片参数
- 推导输出 shape

它处理的是“切片规则”和“输入张量元信息”，不是最终送进 `Reshape / Fill` 的 shape 值本身。

这次图里的 slice 也都很简单：

- `[0:1]`
- `[1:2]`
- `[2:3]`

这种情况下，如果最后出现随机大负数，更像是：

- memory contract 错误
- host/device 语义错误
- 或 shape 可见性被污染

而不像 validate 本身算错。

### 5.3 为什么重点收敛到 `Shape -> StridedSlice -> Pack -> Reshape/Fill`

因为这条链同时满足几个特征：

- 上游明确有 `HostMemory` 语义
- 中间节点是 `int32` shape 算术与拼接
- 下游消费者明确要求 host-visible shape/dims
- 报错值像随机污染，不像固定公式错误

这使它比 `Reshape` / `Fill` 自身更像根因制造者。

---

## 6. 修复方案

### 6.1 `StridedSlice<int32>` 改成 host-memory special path

修复要点：

- 不再让 `Shape` 产出的 host-visible shape tensor 走普通 muDNN device 路径
- `int32` 场景单独走 host-memory special path
- 直接在 host 侧完成切片，而不是先 `CreateMTensor(input)` 再交给 muDNN

目标是把：

```text
Shape(host) -> StridedSlice<int32>
```

这一步的 memory contract 改正确。

### 6.2 `Pack<int32>` 改成 host-memory special path

修复要点：

- `Pack<int32>` 不再继续走普通 muDNN Concat 路径
- 直接在 host 侧完成 shape tensor 的打包

目标是把：

```text
StridedSlice<int32> -> Pack<int32> -> Reshape/Fill
```

也完整留在 host 侧。

### 6.3 中间失败尝试：依赖 TensorFlow 内部符号

中途尝试过复用 TensorFlow 内部 `HandleStridedSliceCase`。

现象是：

- `./build.sh debug` 可以编译通过
- 也能生成 `libmusa_plugin.so`
- 但运行时 `tf.load_op_library(...)` 失败
- 日志报 `undefined symbol`

这个过程给了一个非常重要的经验：

**“共享库文件能生成”与“共享库运行时能被 TensorFlow 正确加载”是两件不同的事。**

因此最终保留方案改成插件内部自包含实现，不再依赖未导出的 TensorFlow 内部符号。

### 6.4 为什么没有保留 `Mul<int32>` 全局 host 化

在 `StridedSlice<int32>` / `Pack<int32>` 修完后，曾进一步把 `Mul<int32>` 也全局拉回 host。

这个方案在正确性上更保守，但问题是：

- 会明显拖慢整网性能
- GPU 利用率下降
- CPU/host 侧等待和调度开销变大

所以它适合作为“验证 shape 链判断是否正确”的实验性手段，不适合作为最终常驻修法。

当前保留的经验是：

- correctness-first 可以先用保守方案验证方向
- 但性能回收阶段不能把所有 `int32` 算子都一刀切 host 化

---

## 7. 经验总结

### 7.1 真正根因通常不在第一个报错算子

这次最先报错的是 `Reshape` / `Fill`，但它们只是 shape 消费者，不是第一个制造脏值的节点。

### 7.2 先看 memory contract，再看数学逻辑

遇到长跑随机崩溃、负维度、超大维度这类问题时，不要只盯公式，要优先检查：

- 这个 tensor 到底应该在 host 还是 device
- 上下游对它的语义是否一致
- 中间算子有没有把它错带进普通 device 路径

### 7.3 不是所有 tensor 都应该在 device memory

这是本轮最值得沉淀的经验。

shape / dims / begin / end / strides 这类元信息 tensor，不应默认按普通数值张量处理。否则很容易在插件实现里出现“编译正常、短跑正常、长跑随机崩”的问题。

### 7.4 “能编过”不等于“插件能被运行时正确加载”

PluggableDevice / 自定义插件场景里，编译成功只是第一关。

还必须确认：

- `.so` 运行时能被正确加载
- 没有未解析符号
- 没有隐式依赖 TensorFlow 未导出的内部实现

### 7.5 correctness-first 和 performance-first 要分阶段做

这次正确性修复和性能修复不能混在一起。

更合理的顺序是：

1. 先把 memory contract 修正确
2. 先让长跑稳定
3. 再回收性能

---

## 8. 后续建议

如果后续要继续做性能回收，建议方向是：

- 只对真正的 shape 子图做 host special path
- 不要把所有 `int32` 算子都一刀切改成 host 路径

更具体地说，后续优化应该收窄到类似这样的场景：

```text
Shape -> StridedSlice -> Mul -> Pack -> Reshape/Fill
```

而不是：

- 所有 `StridedSlice<int32>`
- 所有 `Pack<int32>`
- 所有 `Mul<int32>`

都统一 host 化。

这样才能在保持正确性的同时，把 GPU 利用率和整网吞吐再拉回来。

---

## 9. 给后续接手同学的检查清单

如果后面又遇到类似“长跑随机 shape 崩溃”的问题，优先按下面顺序排查：

1. 先确认报错节点是不是 `Reshape` / `Fill` 这类 shape 消费者
2. 反查它的 shape 输入来自哪些节点
3. 看链路里是否有：
   - `Shape`
   - `StridedSlice`
   - `Pack`
   - `Mul`
4. 确认这些节点的 `HostMemory` / `DeviceMemory` 语义是否一致
5. 不要只看编译是否通过，还要验证插件运行时是否能正常加载
6. 如果为了正确性先做保守 host 化，要同步评估性能代价
