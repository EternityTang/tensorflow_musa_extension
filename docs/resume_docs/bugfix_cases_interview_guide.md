# MUSA 问题定位与修复面试指南

> 本文档整理了 3 个典型 bug 修复案例，涵盖问题现象、根因分析、定位过程和解决方案。

---

## 目录

1. [MUSA_ERROR_ILLEGAL_ADDRESS — H2D 跨 Stream 同步失效](#1-illegal_address-修复)
2. [OOM 与长跑随机崩溃 — Shape Tensor Host/Device 语义污染](#2-oom-修复)
3. [H2D 尾延迟优化 — 推理场景 Feed 瓶颈](#3-h2d-尾延迟优化)

---

## 1. MUSA_ERROR_ILLEGAL_ADDRESS — H2D 跨 Stream 同步失效

**Commit**: `2e08a25d` — `fix: bug fix for H2D Stream Synchronize (#177)`
**修改文件**: `musa_ext/mu/device/musa_device.cc`, `musa_resourcegather_op.cc`, `musa_unique_op.cc`

### 1.1 问题现象

运行推理脚本 `musa_run_pb_graph.py --spec meta_graph_2.spec --bs 32` 时触发 `MUSA_ERROR_ILLEGAL_ADDRESS`，GPU dump 显示：

```
kernel: gather_abc_kernel<int, long, FastDivmodU32, int>
stream: 1
MMU: Fault (Page Directory) — 访问地址 0x003ece8fcd80，远小于 GPU 地址空间
伴随: ReduceColumnShflKernel 级联报错
首个错误后 10ms: MUSA_ERROR_NOT_PERMITTED in Command Memcpy
```

**关键特征**：
- bs=1 成功，bs=32 失败 — 与 batch size 相关
- CPU 上相同数据 bs=32 成功 — 排除数据/图逻辑问题
- 崩溃点在 `gather_kernel`，但 gather 的 indices 来自动态计算 — 不是 indices 越界

### 1.2 问题根因

**MUSA runtime 的 `musaStreamWaitEvent` GPU 侧异步等待在 TensorFlow 复杂环境下不可靠。**

TensorFlow 的 H2D 拷贝流程：
1. `CopyCPUTensorToDevice` 在 `h2d_stream_` 上执行 `musaMemcpyAsync`
2. 通过 `musaEventRecord` + `musaStreamWaitEvent` 通知 compute stream 等待
3. API 返回 `musaSuccess`，但 compute stream 并未真正等待 H2D 完成
4. Compute kernel 在数据到达 GPU 前就开始执行，读到未映射的 GPU 内存

**时序图**：
```
旧方案（失败）：
  h2d_stream_:   [memcpy_async] → event_record
  stream_handle_: stream_wait_event(不可靠) → [compute kernel] ← 数据可能未就绪
  host:           done() 立即调用 → TF 调度 compute kernel
```

纯 MUSA runtime 的独立测试中 `musaStreamWaitEvent` 正常工作，问题仅在 TF 环境下复现，与大量并发、muDNN 算子、BFCAllocator、EventMgr 线程等 TF 特有因素有关。

### 1.3 定位过程

**阶段一：排除数据和图逻辑问题**
- CPU 对比测试：相同数据在 CPU 上成功 → 排除数据问题
- Batch size 对比：bs=1 成功，bs=32 失败 → 问题与内存/时序相关

**阶段二：排除迭代间累积**
- 每次 `sess.run()` 后调用 `musaDeviceSynchronize()`
- bs=8 iter=2000 仍然失败 → 问题在**单次 `sess.run()` 内部**

**阶段三：锁定 H2D 异步路径（关键突破）**
- 将所有 `musaMemcpyAsync` 改为同步 `musaMemcpy`
- bs=32 iter=2000 稳定运行 → **问题在 H2D 异步路径**

**阶段四：逐步二分定位**
- 恢复 pinned 路径 async → 仍然成功（但后续发现 pinned 路径从未被触发）
- 恢复 pageable bounce buffer 路径 async → **立即复现**
- 补 event/wait 跨 stream 同步 → **仍然失败**
- 排除内存池问题（改用 `musaMallocHost`）→ **仍然失败**

**阶段五：确认跨 stream 可见性**
- `musaStreamSynchronize(h2d_stream_)` → 只保证 h2d stream 完成，compute stream 不受影响
- `musaDeviceSynchronize()` → **成功**（全局同步）
- 将 H2D 改到 compute stream（同 stream）→ **成功**

**阶段六：精确验证 `musaStreamWaitEvent` 无效**
- 添加诊断日志：`sync_dst_compute` 始终为 true，event/wait 确实执行了，API 返回成功
- `musaEventSynchronize`（host 阻塞等待）→ **成功**
- 去掉 `musaStreamWaitEvent`，只留 `musaEventSynchronize` → **仍然成功**
- **结论：`musaStreamWaitEvent` 完全无效**

### 1.4 解决方案

**实际采用方案：`ThenExecute` 回调（非阻塞 host 侧完成通知）**

```cpp
// 1. sync_dst_compute: compute→H2D 方向同步（可靠方向）
if (sync_dst_compute) {
    musaEventRecord(sync_event, stream_handle_);     // compute 完成
    musaStreamWaitEvent(h2d_stream_, sync_event, 0);  // h2d 等 compute
    event_mgr_->ThenExecute(h2d_stream_, [sync_event]() {
        musaEventDestroy(sync_event);
    });
}

// 2. H2D 异步拷贝
musaMemcpyAsync(dst, bounce_buffer, bytes, ..., h2d_stream_);
pool->FreeAsync(bounce_buffer, h2d_stream_);

// 3. 完成通知：host 侧 event polling
event_mgr_->ThenExecute(h2d_stream_, [done]() {
    done(Status::OK());  // H2D GPU 完成后才通知 TF
});
```

**关键变更**：
| 变更 | 说明 |
|------|------|
| 移除 H2D→compute 的 `musaStreamWaitEvent` | GPU 侧异步等待不可靠 |
| `sync_dst_compute` 反转为 compute→H2D | 正确语义 + 可靠方向 |
| H2D 完成通知改为 `ThenExecute` 回调 | host 侧完成通知，不阻塞 host |
| D2H 添加 `TensorReference` | 防止异步 D2H 期间 tensor 被释放 |

### 1.5 最终效果

- bs=32 稳定运行，不再触发 `MUSA_ERROR_ILLEGAL_ADDRESS`
- `ThenExecute` 方案不阻塞 host 线程，吞吐优于 `musaEventSynchronize`
- 已向 Moore Threads 反馈 `musaStreamWaitEvent` 的 bug

### 1.6 测试方法

**复现测试**：运行 `musa_run_pb_graph.py --spec meta_graph_2.spec --bs 32`，修复前必定崩溃

**验证测试**：
```cpp
// 独立 MUSA 程序验证跨 stream 行为
// streamA: memcpy_async → event_record
// streamB: stream_wait_event → verify_kernel
// 结果：纯 MUSA 下正常，TF 环境下不可靠
```

---

## 2. OOM 与长跑随机崩溃 — Shape Tensor Host/Device 语义污染

**相关 Commit**: `9d5b6aca` 中的 `musa_strided_slice_op.cc`、`musa_pack_op.cc` 修改
**修改文件**: `musa_strided_slice_op.cc`, `musa_pack_op.cc`, `musa_multiply_op.cc`

### 2.1 问题现象

推理优化场景下的长跑随机崩溃：
- `build.sh release` 更容易跑通，`build.sh debug` 更容易炸
- 崩溃轮次不固定，从几百轮到几十万轮都有可能
- 表面报错：
  - `musa_fill_op.cc:83 : Dimension ... must be >= 0`
  - `RankMixerBlock_0/RankMixerBlock_0_LN_1/Reshape : Dimension size must be non-negative`
- 有时先炸在 `Fill`，有时先炸在 `Reshape` — 看起来像两个独立 bug

### 2.2 问题根因

**本来应该一直待在 host 侧的 shape tensor，中途被错误地当成了 device tensor 来包和传。**

TensorFlow 中 `Shape` 的输出被注册为 `HostMemory("output")`，`Reshape` 的 shape 输入和 `Fill` 的 dims 输入也要求 host 可见。但中间的 `StridedSlice<int32>`、`Pack<int32>`、`Mul<int32>` 没有延续这层语义，把 shape tensor 按普通 device tensor 路径处理了。

**错误路径**：
```
Shape(host) → StridedSlice<int32>(错误走 device) → Mul<int32>(device)
  → Pack<int32>(device) → Reshape/Fill 读到被污染的 shape 值
  → 报错：负维度 / 超大维度 / 随机崩溃
```

**为什么 `debug` 更容易炸**：
- debug 改变了执行节奏，timing/instrumentation 更容易把已有的 host/device contract 问题放大出来
- debug 不是根因，而是"帮你把隐藏 bug 提前炸出来"

### 2.3 定位过程

**阶段一：分析崩溃点**
- Dump 分析发现崩溃在 `gather_kernel`，MMU Page Directory Fault
- 图结构分析：23 个 GatherV2 中有 4 个的 indices 来自动态计算

**阶段二：排除数据问题**
- CPU 对比测试成功 → 排除数据/图逻辑问题
- Batch size 对比：bs=1 成功，bs=32 失败 → 内存/时序方向

**阶段三：分析 shape tensor 链路（关键突破）**
- 发现 `Reshape` 和 `Fill` 的 shape/dims 输入来自同一条链：
  ```
  Shape → StridedSlice → Mul → Pack → Reshape/Fill
  ```
- 这两个报错不是独立 bug，而是同一条链在不同出口暴露

**阶段四：确认 host/device 语义问题**
- `Shape` 产出的是 shape 元信息，注册为 `HostMemory`
- `StridedSlice<int32>` 没有区分 host/device，直接 `CreateMTensor(input)` 交给 muDNN
- `CreateMTensor(...)` 只是把 Tensor 原始地址交给 muDNN，不区分 host/device
- Shape tensor 在中途被错误地按 device tensor 处理，值被污染

**阶段五：逐步修复验证**
1. 改 `StridedSlice<int32>` 为 host path → 长跑稳定性明显提升
2. 改 `Pack<int32>` 为 host path → 进一步稳定
3. 改 `Mul<int32>` 为 host path → 完全稳定

### 2.4 解决方案

**核心思路**：将 shape tensor 链路上的所有 `int32` 算子改为 host-memory special path。

**1. `StridedSlice<int32>` 改为 host path**：
```cpp
// 注册改为 HostMemory
HostMemory("input")
HostMemory("begin")
HostMemory("end")
HostMemory("strides")
HostMemory("output")

// int32 走 host 侧切片实现，不再 CreateMTensor + muDNN Permute
```

**2. `Pack<int32>` 改为 host path**：
```cpp
// 注册改为 HostMemory
HostMemory("values")
HostMemory("output")

// 新增 host 侧 Pack 拷贝逻辑，不再走 muDNN Concat
```

**3. `Mul<int32>` 改为 host path**：
```cpp
// 注册改为 HostMemory
HostMemory("x")
HostMemory("y")
HostMemory("z")

// 新增 host 侧 int32 乘法，支持 BCast 广播
```

**踩过的坑**：尝试复用 TensorFlow 内部 `HandleStridedSliceCase<CPUDevice, int32, NDIM>`，编译成功但运行时 `tf.load_op_library(...)` 失败（undefined symbol）。结论：**插件能编译出来不等于运行时能被正确加载**。

### 2.5 最终效果

- 整条 shape tensor 链路保持 host 语义一致性
- 长跑随机崩溃问题完全消除
- `Reshape` 和 `Fill` 不再报负维度错误

### 2.6 测试方法

**功能测试**：
```python
# test/ops/strided_slice_op_test.py — 补充 Shape → StridedSlice(int32) 测试
# test/ops/pack_op_test.py — 补充 Pack<int32> 作为 shape tensor 的测试
# test/ops/reshape_op_test.py — 补充 Shape → StridedSlice → Pack → Reshape 组合测试
# test/ops/fill_op_test.py — 补充 Shape → StridedSlice → Pack → Fill 组合测试
```

**稳定性测试**：长跑推理测试，bs=32，跑 10000+ 轮不再崩溃

---

## 3. H2D 尾延迟优化 — 推理场景 Feed 瓶颈

**Commit**: `9d5b6aca` — `feat: add reshape matmul fusion and refine H2D logic (#188)`
**修改文件**: `musa_ext/mu/device/musa_device.cc`

### 3.1 问题现象

推理场景下，模型有数百个 feed tensor（输入张量）。每个 feed tensor 的 H2D 拷贝都需要：
1. 在 `h2d_stream_` 上执行 `musaMemcpyAsync`
2. `musaEventRecord` 记录事件
3. `musaStreamWaitEvent` 让 compute stream 等待
4. `ThenExecute` 回调销毁 event

**问题**：数百个 feed tensor 意味着数百次跨 stream event/wait 操作，形成"event/wait 风暴"，导致 H2D 阶段尾延迟显著。

### 3.2 问题根因

默认的 H2D 路径使用独立的 `h2d_stream_`，每个 feed tensor 都需要：
- 1 次 `musaEventRecord`
- 1 次 `musaStreamWaitEvent`
- 1 次 `ThenExecute` 回调

对于 200+ 个 feed tensor 的推理图，这意味着：
- 200+ 次 event 创建/销毁
- 200+ 次跨 stream 同步命令排队
- GPU 命令队列被大量同步命令填满，有效计算被延迟

### 3.3 定位过程

**分析 H2D 路径**：
- 观察到推理图有数百个 feed tensor
- 每个 feed 都走 `h2d_stream_` + event/wait 路径
- Profiling 显示 H2D 阶段耗时远超预期

**对比实验**：
- 将 H2D 拷贝直接放到 compute stream 上 → H2D 阶段耗时大幅下降
- 但需要确保不破坏 compute stream 上的其他操作顺序

### 3.4 解决方案

**新增两个环境变量控制 H2D 路径选择**：

```cpp
// 环境变量控制
MUSA_PAGEABLE_H2D_ON_COMPUTE_STREAM=1  // pageable 内存 H2D 走 compute stream
MUSA_PINNED_H2D_ON_COMPUTE_STREAM=1    // pinned 内存 H2D 走 compute stream
```

**Pageable H2D 走 compute stream 的实现**：
```cpp
if (sync_dst_compute && EnablePageableH2DOnComputeStream()) {
    // 直接在 compute stream 上做 H2D，天然有序，无需跨 stream 同步
    musaMemcpyAsync(dst, bounce_buffer, bytes, musaMemcpyHostToDevice,
                    stream_handle_);  // 注意：用 stream_handle_ 而非 h2d_stream_
    pool->FreeAsync(bounce_buffer, stream_handle_);
    done(Status::OK());
    return;
}
```

**Pinned H2D 走 compute stream 的实现**：
```cpp
if (sync_dst_compute && EnablePinnedH2DOnComputeStream()) {
    musaMemcpyAsync(dst, src, bytes, musaMemcpyHostToDevice, stream_handle_);
    done(Status::OK());
    return;
}
```

**同步逻辑重构**：
- 将 `sync_dst_compute` 的同步逻辑封装为 `wait_h2d_stream_for_compute` lambda
- 同样重构 D2H 路径的 `wait_d2h_stream_for_compute` lambda
- 统一错误处理，增加细粒度的错误检查

### 3.5 最终效果

| 维度 | 优化前 | 优化后 |
|------|--------|--------|
| Event/Wait 次数 | 200+ 次/推理 | **0 次**（走 compute stream） |
| H2D 尾延迟 | 显著（event/wait 风暴） | **大幅降低** |
| 适用场景 | 通用 | 推理场景（大量 feed tensor） |

**使用方式**：
```bash
# 推理场景启用优化
export MUSA_PAGEABLE_H2D_ON_COMPUTE_STREAM=1
export MUSA_PINNED_H2D_ON_COMPUTE_STREAM=1
```

**注意事项**：
- 训练场景通常不需要此优化（feed tensor 较少）
- 同 stream H2D 会阻塞 compute stream 上的后续 kernel，但对于推理场景，feed 本身就是瓶颈
- 默认关闭，通过环境变量显式启用

### 3.6 测试方法

**功能测试**：
```bash
# 推理场景测试
export MUSA_PAGEABLE_H2D_ON_COMPUTE_STREAM=1
python musa_run_pb_graph.py --spec meta_graph_2.spec --bs 32
# 验证：结果正确，H2D 阶段耗时降低
```

**性能测试**：
```python
# 对比开启前后的 H2D 阶段耗时
# 使用 MUSA_TELEMETRY 记录每个 H2D memcpy 的时间戳
export MUSA_TELEMETRY_ENABLED=1
export MUSA_TELEMETRY_LOG_PATH=/tmp/telemetry.json
# 分析 telemetry 中 H2D 事件的总耗时和尾延迟
```

---

## 附录：三个问题的对比总结

| 维度 | Illegal Address | OOM/长跑崩溃 | H2D 尾延迟 |
|------|----------------|-------------|-----------|
| **现象** | GPU MMU Fault | 随机负维度崩溃 | H2D 阶段耗时长 |
| **根因** | `musaStreamWaitEvent` 不可靠 | Shape tensor host/device 语义污染 | 大量 feed 的 event/wait 风暴 |
| **定位方法** | 逐步二分：同步→异步→event/wait | 图分析：shape tensor 链路追踪 | Profiling：H2D 耗时分析 |
| **关键实验** | 同 stream H2D 成功 | int32 改 host path 成功 | compute stream H2D 成功 |
| **解决方案** | `ThenExecute` 回调 | 全链路 host-memory special path | 环境变量控制走 compute stream |
| **效果** | 不再崩溃 | 长跑稳定 | H2D 尾延迟大幅降低 |

### 面试常见问题

**Q: 如何区分"编译成功"和"运行时可用"？**
A: 插件 `.so` 编译成功不代表运行时能被 TensorFlow 正确加载。共享库中的未解析符号（如 TF 内部函数）只在 `tf.load_op_library()` 时才暴露。解决方案是插件内自包含实现，不依赖 TF 内部符号。

**Q: 为什么 `debug` 比 `release` 更容易崩溃？**
A: debug 模式改变了执行节奏（timing/instrumentation），更容易把已有的时序 bug 放大出来。debug 不是根因，而是"帮你把隐藏 bug 提前炸出来"。

**Q: `musaStreamWaitEvent` 为什么不可靠？**
A: 在 TF 复杂环境下（大量并发、muDNN 算子、BFCAllocator、EventMgr 线程），GPU 侧异步等待的语义可能被破坏。纯 MUSA runtime 的简单场景下正常。已向 Moore Threads 反馈此 bug。

**Q: Shape tensor 为什么不能走 device path？**
A: `Shape` 产出的是 shape 元信息（每个维度的大小），`Reshape`/`Fill` 等下游算子需要在 host 侧读取这些值来决定输出形状。如果 shape tensor 在 device 上被 muDNN 算子处理后写回，host 侧读到的可能是未同步的脏值。

**Q: 推理场景的 H2D 优化为什么默认关闭？**
A: 同 stream H2D 会阻塞 compute stream 上的后续 kernel。训练场景中 feed tensor 较少，跨 stream event/wait 的开销可以接受，且并行度更重要。推理场景 feed tensor 多，H2D 本身就是瓶颈，此时同 stream 的有序性反而更优。
