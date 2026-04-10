# 一次推理优化 OOM 与随机崩溃问题的定位与修复

## 1. 事情是怎么开始的

这次排障一开始非常像一个典型的“推理优化长跑 bug”：

- 模型短时间运行正常
- 但跑到几百轮、几千轮，甚至几十万轮之后会随机崩溃
- 最先报错的节点通常是 `Reshape` 或 `Fill`
- 报错信息长这样：
  - `Dimension ... must be >= 0`
  - `Dimension size must be non-negative`

更麻烦的是，这个问题还有一个很容易把人带偏的现象：

- `release` 更容易跑通
- `debug` 更容易炸

如果只看表面，很容易得出几个错误判断：

- 是不是 `Reshape` 本身实现错了？
- 是不是 `Fill` 在某些场景下算错了？
- 是不是 debug 宏把某个算子搞坏了？

但真正往图里走一层，结论完全不是这样。

---

## 2. 真正的问题，不在消费者，而在上游 shape 链

这次真正有问题的不是 `Reshape` 本身，而是它上游那条 shape 计算链。

主线大概是：

```text
Shape -> StridedSlice -> Mul -> Pack -> Reshape
                           └──────────────-> Fill
```

也就是说：

- `Reshape` 用到的 `shape`
- `Fill` 用到的 `dims`

其实都来自同一条 shape tensor 链。

所以你看到：

- 有时先炸在 `Fill`
- 有时先炸在 `Reshape`

这不是两个独立 bug，而是同一条链在不同出口暴露出来。

---

## 3. 最关键的认识：不是所有 Tensor 都应该在 device memory

这次排障里最重要的一句话，是下面这句：

**本来应该一直待在 host 侧的 shape tensor，中途被错误地当成了 device tensor 来包和传。**

要理解这句话，先得分清两个概念：

- `HostMemory`：张量数据放在 CPU 主存里，由 host 直接读取
- `DeviceMemory`：张量数据放在 GPU/MUSA 显存里，由 device kernel 直接读取

很多人第一次写插件时会有一个自然想法：

“既然是 GPU/MUSA 插件，那 tensor 不就应该都在 device memory 里吗？”

其实不是。

### 3.1 什么东西适合放在 device memory

真正适合长期待在 device 上的，一般是这些大块数值数据：

- feature
- activation
- weight
- gradient

这类数据体积大、并行度高，当然适合在显存里算。

### 3.2 什么东西更适合放在 host memory

但像下面这类 tensor，更多是在描述“怎么计算”，而不是“真正被计算的内容”：

- `shape`
- `dims`
- `begin / end / strides`
- 标量控制参数

它们往往数据很小，而且很多下游算子本来就要求它们必须是 host 可见的。

这次问题里就是这样：

- `Shape` 产出的其实是 shape 元信息
- `Reshape(shape)` 需要读 shape
- `Fill(dims)` 需要读 dims

这些东西从语义上就更适合一直留在 host 侧。

---

## 4. 这次为什么会炸

问题的根因不是 `Reshape` 算错了，而是：

- 上游 `Shape` 已经把 shape tensor 放在正确的 host 语义上
- 但中间的 `StridedSlice<int32>` / `Pack<int32>` 却没有延续这层语义
- 它们把这类 shape tensor 又按普通 device tensor 路径处理了

简单说，就是：

```text
本该留在 host 的 shape tensor
        ↓
中途被错误带进了 device 路径
        ↓
下游 Reshape / Fill 读到被污染的 shape 值
        ↓
报负维度、大维度、长跑随机崩溃
```

这个问题最麻烦的地方就在于：

- 它不是每次都立刻炸
- 它不一定在同一轮炸
- 它不一定在同一个节点先炸

这就是为什么一开始特别像“玄学 bug”。

---

## 5. 一张图看懂：正常路径和错误路径

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

从这张图你会发现，真正的问题不在最后那个 `Reshape`，而在中间 shape tensor 走错了路。

---

## 6. 为什么一开始没有盯住它

这次排障里有几个特别容易误导人的点。

### 6.1 `Reshape` / `Fill` 是第一个报错点，但不是第一个出错点

这是很多长跑问题的常见陷阱。

谁最先报错，不代表谁是根因。

`Reshape` 和 `Fill` 更像是“第一个认真检查 shape 值是否合法的人”，所以 shape 一旦脏了，它们就最先跳出来报警。

### 6.2 `ValidateStridedSliceOp(...)` 很可疑，但不是真正主因

一开始看到 `StridedSlice` 时，很容易怀疑是切片规则算错了。

但后来发现：

- 这次图里的切片都很简单
- `ValidateStridedSliceOp(...)` 更像是在做参数规范化和 shape 推导
- 真正高风险的不是 validate，而是 validate 之后把 host-visible tensor 继续按 device tensor 走的那段路径

### 6.3 `release` 正常、`debug` 更容易炸，不代表 debug 自己是根因

更准确的理解应该是：

- debug 改变了执行节奏
- timing / instrumentation 更容易把已有的 host/device contract 问题放大出来

也就是说，debug 更像“帮你把隐藏 bug 提前炸出来”。

---

## 7. 最终怎么修

这次最后保留的修法，核心是两步。

### 7.1 `StridedSlice<int32>` 改成 host path

思路很直接：

- 既然这个 `int32` tensor 本质上是 shape 元信息
- 那它就不应该再被塞进普通 device 路径

所以 `StridedSlice<int32>` 最终改成了 host-memory special path。

### 7.2 `Pack<int32>` 也改成 host path

只改 `StridedSlice` 还不够，因为 shape tensor 后面还会继续被 `Pack<int32>` 拼起来，再送进 `Reshape` / `Fill`。

所以 `Pack<int32>` 也要一起改成 host-memory special path，保证整段 shape 链保持一致语义。

---

## 8. 中途踩过的坑

### 8.1 运行时加载失败和编译失败不是一回事

中间尝试过复用 TensorFlow 内部某个现成实现。

现象特别容易让人误判：

- `./build.sh debug` 能过
- `.so` 也能生成
- 但运行时 `tf.load_op_library(...)` 直接失败

这次踩坑后的经验非常重要：

**插件“能编译出来”不等于“运行时能被正确加载”。**

共享库里有没有未解析符号，很多时候要到运行时才真正暴露。

### 8.2 correctness-first 的保守修法，可能会拖慢性能

后面为了继续验证 shape 链判断，还尝试过把 `Mul<int32>` 也一起拉回 host。

这个思路在正确性上更保守，但问题也很明显：

- CPU 参与更多
- GPU 更容易等待 host 侧 shape 计算
- 整网速度明显下降

所以这一步给出的经验是：

- 正确性和性能要分阶段修
- 先证明判断对不对
- 再收缩修法，把 host special path 只限定在真正的 shape 子图上

---

## 9. TensorFlow 和 PyTorch 都会有这种问题吗

不是 TensorFlow 独有，但 TensorFlow 会把这件事表达得更显式。

在 TensorFlow 里：

- 很多算子会直接声明某个输入/输出是 `HostMemory`
- 所以 host/device contract 非常清楚
- 一旦插件实现没跟上，就容易直接出问题

PyTorch 里也有类似概念：

- `sizes`
- `strides`
- metadata
- shape 推导相关信息

这些东西本质上也不是普通数值 tensor。

只是 PyTorch 更多是通过 runtime、dispatcher 和 metadata 系统来承载，而不是像 TensorFlow 这样常常直接把 `HostMemory` 写在 kernel 注册语义里。

所以两边本质是一样的：

- 大块数值数据适合放 device
- shape/metadata/control tensor 不该被粗暴当成普通 device data 处理

---

## 10. 这次排障最值得带走的 3 个经验

### 经验 1：先看 contract，再看公式

遇到长跑随机崩溃、负维度、大维度时，不要只盯算术公式，先看：

- 这个 tensor 到底应该在 host 还是 device
- 上下游对它的内存语义是否一致

### 经验 2：第一个报错的节点，往往不是根因

`Reshape` / `Fill` 很可能只是最先发现 shape 脏了，而不是最先把 shape 搞脏的人。

### 经验 3：先把正确性打住，再回收性能

如果已经判断是 memory contract 问题，先用保守修法把正确性稳定住是值得的。  
但最终版本一定要继续收窄，不然性能会掉得很明显。

---

## 11. 一份可复用的排障 checklist

如果你后面也遇到类似“长跑随机 shape 崩溃”的问题，可以优先按这个顺序看：

1. 先确认最先报错的是不是 shape 消费者，例如 `Reshape` / `Fill`
2. 反查它的 `shape` / `dims` 输入来自哪里
3. 看链路里有没有：
   - `Shape`
   - `StridedSlice`
   - `Pack`
   - `Mul`
4. 确认这些中间节点有没有把本该留在 host 的小 tensor 重新带进 device 路径
5. 不要只看“编译能不能过”，还要看插件运行时是否能成功加载
6. 如果为了稳定性先做了保守修法，要单独评估性能退化，不要把两件事混为一谈

---

## 12. 结尾

这次问题最有意思的地方，不是某个公式写错了，而是一个非常典型、但又很容易被忽视的工程问题：

**shape tensor 不是普通数据 tensor。**

一旦把这层语义看错，插件就可能出现：

- 短跑正常
- 编译正常
- 甚至部分模型正常
- 但长跑随机炸掉

而一旦把这层语义理顺，很多“看起来很玄学”的问题，反而就会一下子变得很具体。
