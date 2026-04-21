# MUSA_ERROR_ILLEGAL_ADDRESS 调试报告

## 1. 问题概述

| 项目 | 信息 |
|-----|------|
| **错误类型** | `MUSA_ERROR_ILLEGAL_ADDRESS` |
| **设备** | MTT S5000 (Driver 3.3.1-server) |
| **崩溃点** | `musa::dnn::gather_abc_kernel` / `ReduceColumnShflKernel` |
| **根因** | MUSA `musaStreamWaitEvent` 跨 stream 同步不可靠 |
| **影响文件** | `tensorflow_musa_extension/musa_ext/mu/device/musa_device.cc` |
| **状态** | 已修复（commit 2e08a25） |

## 2. 根因

**MUSA runtime 的 `musaStreamWaitEvent` GPU 侧异步等待在 TF 复杂环境下不可靠。**

`CopyCPUTensorToDevice` 在 `h2d_stream_` 上执行 `musaMemcpyAsync`，通过 `musaEventRecord + musaStreamWaitEvent` 通知 compute stream (`stream_handle_`) 等待。API 返回 `musaSuccess`，但 compute stream 并未真正等待 H2D 完成，导致 compute kernel 在数据到达 GPU 前就开始执行，读到未映射的 GPU 内存，触发 MMU Page Directory Fault。

纯 MUSA runtime 的独立测试中 `musaStreamWaitEvent` 正常工作，问题仅在 TF 环境下复现，可能与大量并发、muDNN 算子、BFCAllocator、EventMgr 线程等 TF 特有因素有关。

**关键证据：**
- `sync_dst_compute` 始终为 true（event/wait 代码未被跳过）
- event/wait API 调用均返回成功
- 但 compute kernel 仍然读到未就绪数据
- `musaEventSynchronize`（host 侧阻塞等单个 event）可有效替代
- `musaDeviceSynchronize()`（全局同步）或同 stream memcpy 均可解决
- 独立 MUSA 程序中 `musaStreamWaitEvent` 正常（无法脱离 TF 复现）

**`sync_dst_compute` 来源：** TF 的 `BaseGPUDevice::MaybeCopyTensorToGPU` 传入 `!timestamped_allocator_`，后者默认为 false，因此 `sync_dst_compute` 始终为 true。

## 3. 证据链

| 步骤 | 改动 | 结果 | 结论 |
|------|------|------|------|
| 3.7 | H2D 全部改同步 `musaMemcpy` | ✅ | 问题在异步路径 |
| 3.8-3.10 | 恢复 pinned 路径 async（逐步） | ✅ | 无效对比（pinned 路径从未被触发） |
| 3.11 | 恢复 pageable async H2D（h2d_stream_） | ❌ | 问题在 pageable async H2D |
| 3.12 | + 补 event/wait 跨 stream 同步 | ❌ | event/wait API 返回成功但同步无效 |
| 3.13 | + musaMallocHost 替代内存池 | ❌ | 与内存池无关 |
| 3.14 | `musaStreamSynchronize` → `musaDeviceSynchronize` | ✅ | 全局同步有效 |
| 3.15 | 诊断日志确认参数 | — | sync=1, pinned=0（event/wait 执行了但不可靠） |
| **3.16** | **H2D 改到 compute stream** | **✅** | **同 stream async 正常，确认跨 stream wait 是根因** |
| 3.17 | 独立 MUSA 程序 event/wait | ✅ | 纯 MUSA runtime 下 event/wait 正常 |
| **3.18** | **`musaEventSynchronize` 替代 `musaStreamWaitEvent`** | **✅** | **host 侧阻塞等待有效，GPU 侧异步等待不可靠** |
| **3.19** | **Test A：只用 `musaEventSynchronize`，不调 `musaStreamWaitEvent`** | **✅** | **`musaStreamWaitEvent` 完全无效，修复只需 host 阻塞** |

## 4. 调试过程

### 阶段一：定位崩溃点（Step 1-2）

**初始现象：** 运行 `musa_run_pb_graph.py --spec meta_graph_2.spec --bs 32` 触发 `MUSA_ERROR_ILLEGAL_ADDRESS`。

Dump 分析：
```
kernel: gather_abc_kernel<int, long, FastDivmodU32, int>
stream: 1
MMU: Fault (Page Directory) — 访问地址 0x003ece8fcd80，远小于 GPU 地址空间 0x00000100...
伴随: ReduceColumnShflKernel 级联报错
首个错误后 10ms: MUSA_ERROR_NOT_PERMITTED in Command Memcpy
```

初期有四个假设：
1. Gather 算子 indices 越界
2. GPU 内存分配/映射问题
3. MUSA 驱动/库 bug
4. 输入数据问题

**Step 1 — 图结构分析：** 分析 frozen_graph_2.pb，发现 23 个 GatherV2 中有 4 个的 indices 来自动态计算（BOOL/FLOAT 输入 → Equal/NotEqual → Where/Select → indices）。这些 indices 可能越界。

**Step 2 — CPU 对比测试：** 相同数据在 CPU 上 bs=32 成功运行 → 排除数据/图逻辑问题。

**Step 2.5 — Batch Size 对比（关键发现）：**

| 设备 | bs | 结果 |
|------|-----|------|
| MUSA | 1 | ✅ 成功 |
| CPU | 32 | ✅ 成功 |
| MUSA | 32 | ❌ 失败 |

同一 MUSA 设备 bs=1 成功，说明不是 gather kernel 逻辑问题。问题与 batch size 相关 → 内存管理/时序方向。

### 阶段二：排除迭代间累积（Step 3.6）

**假设：** 迭代间异步操作累积导致资源耗尽。

**方法：** 在 Python 层每次 `sess.run()` 后调用 `musaDeviceSynchronize()`，确保每次迭代 GPU 完全完成。

**结果：** bs=8 iter=2000 仍然失败，dump 模式完全一致。

**结论：** ❌ 排除迭代间累积。问题在**单次 `sess.run()` 内部**就已经发生。

### 阶段三：锁定 H2D 异步路径（Step 3.7）

**假设：** H2D 异步拷贝与 compute stream 之间的同步链有问题。

**方法：** 将 `CopyCPUTensorToDevice` 中所有 `musaMemcpyAsync` 改为同步 `musaMemcpy`，包括：
- Pinned 路径：`musaMemcpyAsync` → `musaMemcpy`，移除 event/wait 和 ThenExecute 回调
- Pageable bounce buffer 路径：同样改为 `musaMemcpy`，移除 event/wait

**结果：** bs=32 iter=2000 稳定运行。

**结论：** ⭐ **问题在 H2D 异步路径。** 同步化后错误完全消失，说明 kernel 读到的数据在异步场景下不可用。

### 阶段四：逐步恢复 pinned 路径（Step 3.8-3.10）

> **注意：** 后续 Step 3.15 诊断发现 pinned 路径从未被触发（所有输入是 pageable NumPy 数组），以下三步的结论实际无效。保留记录以展示完整过程。

**方法：** 只恢复 pinned host memory 路径，逐步加回异步机制：

| 步骤 | 恢复内容 | 结果 |
|------|---------|------|
| 3.8 | `musaMemcpyAsync` + `musaStreamSynchronize(h2d_stream_)` | ✅ |
| 3.9 | + `musaEventRecord` + `musaStreamWaitEvent` | ✅ |
| 3.10 | + `event_mgr_->ThenExecute` 回调 | ✅ |

Pinned 路径逐步恢复到原始异步设计仍然稳定。但后续诊断证实此路径从未执行，故这些对比无意义。

### 阶段五：锁定 pageable bounce buffer 路径（Step 3.11）

**方法：** 保持 pinned 路径不变，只恢复 pageable 大拷贝路径的 `musaMemcpyAsync`（`h2d_stream_`），但保留 `musaStreamSynchronize(h2d_stream_)` 保守同步。**不补 event/wait**。

**结果：** ❌ 立即复现问题。Dump 模式完全一致（gather_kernel + ReduceColumnShflKernel，stream 1，Page Directory Fault）。

**结论：** 前面 3 步都只碰了 pinned 路径（未触发），第一次碰 pageable async H2D 就复现。问题锁定在 **pageable + bounce buffer + async H2D** 路径。

**此时最可疑的缺口：** 恢复了 async H2D 但没补 event/wait，即 compute stream 没有被通知等待 H2D 完成。

### 阶段六：补 event/wait 仍失败（Step 3.12）

**方法：** 在 pageable async H2D 路径补上与 pinned 路径相同的跨 stream 同步链：
- `musaEventRecord(copy_done_event, h2d_stream_)`
- `musaStreamWaitEvent(stream_handle_, copy_done_event, 0)`
- 保留 `musaStreamSynchronize(h2d_stream_)` 保守同步

**结果：** ❌ 仍然失败。

**结论：** ❌ 排除"缺少 event/wait 是唯一问题"。即使补上了跨 stream 同步链（API 返回成功），问题仍然出现。

### 阶段七：排除内存池（Step 3.13）

**假设：** `GPUPinnedMemoryPool` 的分配复用可能导致 bounce buffer 被过早重用。

**方法：** 将 `pinned_memory_pool()->Allocate(bytes)` 替换为 `musaMallocHost`（每次分配全新内存），`FreeAsync` 替换为 `musaFreeHost`。其他逻辑不变。

**结果：** ❌ 仍然失败，dump 模式一致。

**结论：** ❌ 排除内存池问题。即使用全新 pinned memory 仍然失败。

**此时对比表：**

| 路径 | `musaMemcpy` (sync) | `musaMemcpyAsync` (async) |
|------|---------------------|--------------------------|
| Pinned 直接传 | ✅ | ✅（但从未被触发） |
| Pageable + bounce buffer | ✅ | ❌ |

唯一失败的组合：**pageable → bounce buffer → `musaMemcpyAsync` → device**。

### 阶段八：确认跨 stream 可见性（Step 3.14）

**假设：** `musaStreamSynchronize(h2d_stream_)` 只保证 h2d stream 完成，不保证 compute stream 可见。同步 `musaMemcpy` 能工作是因为它隐含全局设备同步。

**方法：** 在 Step 3.13 基础上，仅将 `musaStreamSynchronize(h2d_stream_)` 替换为 `musaDeviceSynchronize()`。

**结果：** ✅ 成功。bs=32 稳定运行。

**结论：** ⭐ **根因确认：跨 stream 可见性。** `musaStreamSynchronize(h2d_stream_)` 只阻塞 CPU 直到 h2d_stream_ 完成，但 compute stream 不受影响。当 TF 在 `done()` 后立即启动 compute kernel 时，H2D 数据对 compute stream 尚不可见。

**此时仍有关键疑问：** 代码中已有 `musaEventRecord + musaStreamWaitEvent` 做跨 stream 同步，为什么没生效？两种可能：
1. `sync_dst_compute` 为 false，event/wait 被跳过
2. `musaStreamWaitEvent` 本身不可靠

### 阶段九：确认 event/wait 确实执行了（Step 3.15）

**方法：** 在 `CopyCPUTensorToDevice` 入口和 `is_pinned` 判断处添加 LOG 诊断。

**发现：**
- `sync_dst_compute` 始终为 `1`（true）→ event/wait 代码路径被进入
- `is_pinned` 始终为 `0`（false）→ 所有 H2D 走 pageable bounce buffer 路径，pinned 路径从未被触发
- event/wait API 均返回 `musaSuccess`，有错误检查，失败会 early return

**结论：** 排除"代码被跳过"。event/wait 确实执行了，API 返回成功，但同步语义未生效。

**附带发现：** Steps 3.8-3.10 中 pinned 路径的测试结论无效（该路径从未执行），整个调试过程中真正有效的对比是 Steps 3.11-3.16。

### 阶段十：确认跨 stream wait 是根因（Step 3.16）

**假设：** 如果问题 100% 在跨 stream `musaStreamWaitEvent`，那么消除跨 stream 因素后 async H2D 应该正常。

**方法：** 将 `musaMemcpyAsync` 从 `h2d_stream_` 改到 `stream_handle_`（compute stream），移除所有跨 stream 同步链（event/wait + musaDeviceSynchronize），只用 `musaStreamSynchronize(stream_handle_)` 确保 memcpy 完成后释放 bounce buffer。

```
之前：h2d_stream_ 做 memcpy → event/wait → stream_handle_ 等待
现在：stream_handle_ 做 memcpy → 同 stream 天然有序，无需跨 stream 同步
```

**结果：** ✅ 成功。

**结论：** ⭐ **最终确认。** async H2D 传输本身完全正常，bounce buffer 机制正常，同 stream 的 `musaStreamSynchronize` 正常。唯一的问题是 MUSA 的 `musaEventRecord + musaStreamWaitEvent` 跨 stream 同步不可靠。

### 阶段十一：独立 MUSA 程序验证（Step 3.17）

**方法：** 编写独立 MUSA C++ 测试程序（不依赖 TF），模拟跨 stream event/wait 模式：streamA 做 H2D memcpy → event record → streamB wait event → streamB 启动 verify kernel → 检查输出。

**测试内容：**
1. 同 stream 对照组（1000 iter）
2. 跨 stream event/wait（1000 iter，128KB）
3. 模拟 TF 模式：20 个输入 × 500 runs，快速交替 event/wait

**结果：** 全部通过，0 次不一致。

**无同步对照组**（不加 event/wait）：100 次中 2 次数据不一致，证实跨 stream 竞争确实存在但概率很低（~2%）。

**结论：** 纯 MUSA runtime 的 `musaStreamWaitEvent` 在简单场景下工作正常。TF 环境中的某些因素（大量并发、muDNN 算子、BFCAllocator、EventMgr 线程等）可能导致 event/wait 不可靠。独立测试无法复现 bug。

### 阶段十二：`musaEventSynchronize` vs `musaStreamWaitEvent`（Step 3.18）

**假设：** 问题出在 `musaStreamWaitEvent` 的 GPU 侧异步等待语义。如果改用 host 侧阻塞等待（`musaEventSynchronize`），可能有效。

**方法：** 在 TF MUSA extension 的实际代码中恢复 h2d_stream_ + event/wait 路径，在 `musaStreamWaitEvent` 之后添加 `musaEventSynchronize(copy_done_event)` 阻塞 host 等待 event 完成。同时添加细粒度诊断日志记录每个阶段 stream 和 event 的状态。

**诊断日志关键发现：**

```
after stream wait event: event_query=device not ready     ← H2D 还没完成
                          compute_stream_query=device not ready

event synchronize result: no error                         ← host 阻塞等待成功
after event sync: event_query=no error                     ← event 完成了
```

**结果：** ✅ 成功，不再崩溃。

**结论：** ⭐ **精确定位问题机制。**

| 同步方式 | 作用位置 | 结果 |
|---------|---------|------|
| `musaStreamWaitEvent` | GPU 侧异步，host 立即返回 | ❌ 不可靠 |
| `musaEventSynchronize` | host 侧阻塞，等 event 完成 | ✅ 可靠 |
| `musaDeviceSynchronize` | host 侧阻塞，等所有 stream | ✅ 可靠 |
| 同 stream memcpy | 天然有序，无需跨 stream | ✅ 可靠 |

**问题机制确认：** `musaStreamWaitEvent` 的 GPU 侧异步等待在 TF 复杂环境下不能保证 compute stream 在 H2D 完成后才执行后续 kernel。host 侧阻塞等待（`musaEventSynchronize`）可以绕过此问题，因为它确保 host 在 event 完成后才调用 `done()`，TF 才会调度 compute kernel。

### 阶段十三：Test A — 确认 `musaStreamWaitEvent` 完全无效（Step 3.19）

**方法：** 在 Step 3.18 基础上去掉 `musaStreamWaitEvent`，只保留 `musaEventSynchronize`。如果成功，说明 GPU 侧异步等待完全不起作用，修复只需 host 阻塞。

**改动：**
```cpp
// 之前（Step 3.18）：同时有 stream wait + event sync
musaEventRecord(event, h2d_stream_);
musaStreamWaitEvent(stream_handle_, event, 0);  // GPU 侧
musaEventSynchronize(event);                     // host 侧

// Test A：去掉 stream wait，只留 host 阻塞
musaEventRecord(event, h2d_stream_);
// 不调 musaStreamWaitEvent
musaEventSynchronize(event);  // 只靠 host 阻塞
```

**结果：** ✅ 成功，bs=32 稳定运行，average_time ≈ 70ms。

**结论：** ⭐ **`musaStreamWaitEvent` 完全无效。** 有没有它结果一样，真正起作用的只是 `musaEventSynchronize` 阻塞 host。

**问题本质更新：**
- 之前认为是"跨 stream 同步不生效"
- 现在确认更精确：`musaStreamWaitEvent` 的 GPU 侧异步等待在当前 MUSA 环境下完全不起作用，compute stream 上的 kernel 不会因此等待
- 唯一可靠的机制是 host 侧阻塞：`musaEventSynchronize` 等事件完成后，host 才调 `done()` 通知 TF 调度 compute kernel，此时数据已到 GPU

### 阶段十四：最终修复落地（commit 2e08a25）

**实际采用方案：** `ThenExecute` 回调（非阻塞 host 侧完成通知），而非 debug 过程中验证的 `musaEventSynchronize`（阻塞 host）。

**方案选择理由：** `ThenExecute` 通过 event polling 在 `h2d_stream_` GPU 操作真正完成后才回调 `done()`，功能等价于 `musaEventSynchronize`，但不阻塞 host 线程。

**改动一：移除 H2D→compute 方向的 `musaStreamWaitEvent`**

删除了 pinned 路径和 pageable bounce buffer 路径中各一处 H2D→compute 跨 stream 同步代码块：

```cpp
// 删除 ❌ — H2D→compute 方向，musaStreamWaitEvent 不可靠
if (sync_dst_compute) {
  musaEventRecord(copy_done_event, h2d_stream_);
  musaStreamWaitEvent(stream_handle_, copy_done_event, 0);  // GPU 侧异步等待无效
  event_mgr_->ThenExecute(stream_handle_, [copy_done_event, device_id]() {
    musaEventDestroy(copy_done_event);
  });
}
```

H2D 完成通知改为依赖 `ThenExecute(h2d_stream_, done())`：当 `h2d_stream_` 上所有 GPU 操作完成后，`done()` 被回调，TF 才调度 compute kernel 到 `stream_handle_`，此时数据已在 GPU 上。

**改动二：`sync_dst_compute` 语义修正**

将 `sync_dst_compute` 的同步方向从 H2D→compute 反转为 compute→H2D，并移到拷贝操作之前：

```cpp
// 旧：sync_dst_compute=false 时打印警告，拷贝后做 H2D→compute 同步（不可靠）
// 新：sync_dst_compute=true 时做 compute→H2D 同步（可靠方向）
if (sync_dst_compute) {
  musaEventRecord(sync_event, stream_handle_);        // compute 完成
  musaStreamWaitEvent(h2d_stream_, sync_event, 0);     // h2d 等 compute ← 可靠
  event_mgr_->ThenExecute(h2d_stream_, [sync_event, device_id]() {
    musaEventDestroy(sync_event);                      // 延迟销毁，防止 wait 未执行就销毁
  });
}
// 然后执行 H2D 拷贝...
```

此方向与 D2H 路径中 `d2h_stream_` 等待 `stream_handle_` 同向，独立 MUSA 测试证实可靠。

**改动三：D2H 统一 compute→D2H 同步**

将 `CopyDeviceTensorToCPU` 中 pinned 和 pageable 两分支各自重复的 compute→D2H event/wait 代码提到分支之前，共享一份。

**改动四：D2H 添加 `TensorReference` 防止提前释放**

```cpp
// 新增：异步 D2H 期间保持 device tensor 引用
TensorReference input_ref(*device_tensor);  // 引用计数 +1
event_mgr_->ThenExecute(d2h_stream_, [..., input_ref]() {
  input_ref.Unref();  // D2H 完成后才释放
  done(Status::OK());
});
```

防止 `CopyDeviceTensorToCPU` 返回后 TF 释放 `device_tensor` 的 GPU 内存，而 D2H async copy 尚未完成导致读到已释放内存。

**改动五：D2H 增加 `src == nullptr` 检查**

```cpp
if (bytes == 0 or src == nullptr)  // 新增 src 空指针检查
```

**最终时序对比：**

```
旧方案（失败）：
  h2d_stream_:  [memcpy_async] → event_record
  stream_handle_: stream_wait_event(不可靠) → [compute kernel] ← 数据可能未就绪
  host: done() 立即调用

新方案（成功）：
  stream_handle_: [compute kernel] → event_record  ← sync_dst_compute 同步
  h2d_stream_:  wait(compute完成) → [memcpy_async]
  host: ThenExecute 检测到 h2d_stream_ 完成 → done()
  TF: 调度下一轮 compute kernel ← 数据已就绪
```

**修复后 H2D 数据流：**

```
CopyCPUTensorToDevice(src_cpu, dst_gpu)
├── sync_dst_compute == true (始终)
│   └── stream_handle_ record event → h2d_stream_ wait (compute→H2D，可靠方向)
├── is_pinned == true
│   └── musaMemcpyAsync(dst, src, ..., h2d_stream_)
│       └── ThenExecute(h2d_stream_, done())  ← host 侧完成通知
└── is_pinned == false
    ├── bytes <= 1KB → musaMemcpy (同步)
    └── bytes > 1KB
        └── bounce_buffer = pool->Allocate()
        └── std::memcpy(bounce_buffer, src)          // CPU → pinned
        └── musaMemcpyAsync(dst, bounce_buffer, ..., h2d_stream_)  // pinned → GPU
        └── pool->FreeAsync(bounce_buffer, h2d_stream_)
        └── ThenExecute(h2d_stream_, done())         ← host 侧完成通知
```

## 5. 架构背景

### H2D 拷贝的两条路径（修复后）

```
CopyCPUTensorToDevice(src_cpu, dst_gpu)
├── sync_dst_compute == true (始终)
│   └── stream_handle_ record → h2d_stream_ wait (compute→H2D，可靠方向)
├── is_pinned == true
│   └── musaMemcpyAsync(dst, src, ..., h2d_stream_)
│       └── ThenExecute(h2d_stream_, done())
└── is_pinned == false (所有输入走此路径)
    ├── bytes <= 1KB → musaMemcpy (同步)
    └── bytes > 1KB
        └── bounce_buffer = pool->Allocate()
        └── std::memcpy(bounce_buffer, src)                         // CPU → pinned
        └── musaMemcpyAsync(dst, bounce_buffer, ..., h2d_stream_)  // pinned → GPU
        └── pool->FreeAsync(bounce_buffer, h2d_stream_)
        └── ThenExecute(h2d_stream_, done())                        // host 侧完成通知
```

### sync_dst_compute 调用链

```
TF BaseGPUDevice::MaybeCopyTensorToGPU
  → device_context_->CopyCPUTensorToDevice(
      &from, this, copy, done,
      !timestamped_allocator_  // = !false = true
    )
```

## 6. 修复方案

### 已采用方案（commit 2e08a25）

**实际方案：** `ThenExecute` 回调（非阻塞 host 侧完成通知）。

Debug 过程中验证了 `musaEventSynchronize`（阻塞 host）可行，最终采用了功能等价但不阻塞 host 线程的 `ThenExecute` event polling 方案。

**方案选择理由：** `ThenExecute` 通过 event polling 在 `h2d_stream_` GPU 操作真正完成后才回调 `done()`，功能等价于 `musaEventSynchronize`，但不阻塞 host 线程，吞吐更好。

**修复后的 H2D 完整流程：**

```cpp
// 1. sync_dst_compute: compute→H2D 方向同步（可靠）
if (sync_dst_compute) {
  musaEventRecord(sync_event, stream_handle_);     // compute 完成
  musaStreamWaitEvent(h2d_stream_, sync_event, 0);  // h2d 等 compute
  event_mgr_->ThenExecute(h2d_stream_, [sync_event]() {
    musaEventDestroy(sync_event);                   // 延迟销毁
  });
}

// 2. H2D 异步拷贝（h2d_stream_）
musaMemcpyAsync(dst, bounce_buffer, bytes, ..., h2d_stream_);
pool->FreeAsync(bounce_buffer, h2d_stream_);

// 3. 完成通知：host 侧 event polling
event_mgr_->ThenExecute(h2d_stream_, [done]() {
  done(Status::OK());  // H2D GPU 完成后才通知 TF
});
```

**关键变更汇总：**

| 变更 | 说明 |
|------|------|
| 移除 H2D→compute 的 `musaStreamWaitEvent` | GPU 侧异步等待不可靠（Step 3.19 证实） |
| `sync_dst_compute` 反转为 compute→H2D | 正确语义 + 可靠方向 |
| H2D 完成通知改为 `ThenExecute` 回调 | host 侧完成通知，不阻塞 host 线程 |
| D2H 统一 compute→D2H event/wait | 消除重复代码 |
| D2H 添加 `TensorReference` | 防止异步 D2H 期间 tensor 被释放 |
| D2H `src == nullptr` 检查 | 防御性编程 |

### 方案对比（最终）

| 方案 | 性能 | 可靠性 | 是否采用 |
|------|------|--------|---------|
| `ThenExecute` 回调（host 侧 event polling） | 高 | ✅ | **已采用** |
| `musaEventSynchronize`（host 阻塞等单个 event） | 中高 | ✅ | Debug 验证方案 |
| 同 stream async + `musaStreamSynchronize` | 中 | ✅ | Step 3.16 备选 |
| `musaDeviceSynchronize()` | 低 | ✅ | 过于保守 |
| `musaMemcpy`（同步） | 低 | ✅ | 过于保守 |
| 跨 stream `musaStreamWaitEvent`（原始方案） | 高 | ❌ | 根因，已移除 |

### 待清理项

1. ~~移除诊断 LOG（`[H2D-DIAG]`、`[SYNC-DIAG]`）~~ 已在 commit 中清理
2. ~~评估是否恢复 `GPUPinnedMemoryPool` 替代 `musaMallocHost`~~ 已恢复使用 `GPUPinnedMemoryPool`
3. 向 Moore Threads 反馈 `musaStreamWaitEvent` 在 TF 环境下 GPU 侧异步等待不可靠的 bug

## 7. 修复提交

| 项目 | 信息 |
|------|------|
| **Commit** | `2e08a25` |
| **PR** | #177 |
| **标题** | fix: bug fix for H2D Stream Synchronize |
| **修改文件** | `musa_device.cc`, `musa_resourcegather_op.cc`, `musa_unique_op.cc` |

## 8. 相关 dump 文件

| 文件 | 说明 |
|------|------|
| `core_2026-04-15_11:24:43.952_dev_417425.mudmp` | 首次 dump |
| `core_2026-04-16_14:33:31.480_dev_14464.mudmp` | 迭代间 sync 仍失败 |
| `core_2026-04-16_14:41:04.994_dev_14810.mudmp` | 同上 |
| `core_2026-04-16_15:00:47.606_dev_16393.mudmp` | gather + ReduceColumn 级联错误 |
| `core_2026-04-16_17:13:10.829_dev_56564.mudmp` | 恢复 pageable async H2D 复现 |
| `core_2026-04-16_17:25:15.968_dev_64127.mudmp` | 补 event/wait 仍失败 |
| `core_2026-04-17_10:30:59.850_dev_69180.mudmp` | musaMallocHost 替代 pool 仍失败 |
