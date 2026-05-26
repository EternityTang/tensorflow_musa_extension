# 显存子系统与 Event 同步详解

## 一、三层 H2D 拷贝策略

文件：`musa_ext/mu/device/musa_device.cc:49-198`

### 问题背景

Host-to-Device 拷贝的性能取决于源内存类型：

| 内存类型 | 能否异步拷贝 | 需要 bounce buffer |
|---------|------------|-------------------|
| Pinned（锁页） | 可以，`musaMemcpyAsync` 直接执行 | 不需要 |
| Pageable（可分页） | **不能直接异步**，驱动需要先锁页 | 需要 |

### 决策树

```
CopyCPUTensorToDevice(cpu_tensor, device_tensor)
  │
  ├─ sync_dst_compute=true？
  │    └─ 在 H2D stream 上记录 event + wait compute stream
  │       等待计算完成再拷贝（保证数据一致性）
  │       event 销毁延迟到 wait 执行完毕后
  │
  ├─ musaPointerGetAttributes(src) → 是 pinned？
  │    └─ 【快路径】musaMemcpyAsync(src→dst, H2D stream)
  │       不需要 bounce buffer
  │
  ├─ bytes ≤ 1KB？
  │    └─ 【同步小拷贝】musaMemcpy(src→dst, sync)
  │       小数据异步开销大于收益，同步更稳定
  │
  └─ 【Bounce Buffer 路径】
        Stage 1: memcpy(pageable → pinned)  // CPU 侧同步
        Stage 2: musaMemcpyAsync(pinned → device, H2D stream)  // GPU 异步
        pinned 内存通过 FreeAsync 在有 event 保证完成后回收
```

### 为什么 1KB 以下走同步

注释（`:137-139`）：

> Small copies (<1KB) use sync path to avoid async overhead and potential driver instability with small async transfers

MUSA 驱动的异步小拷贝有稳定性问题，而且创建 event、排队、回调的开销远超 1KB 数据的拷贝时间。同步执行反而更快更稳定。

---

## 二、Bounce Buffer 与 Pinned Memory Pool

文件：`musa_ext/mu/device/pinned_memory_pool.cc`

### 核心问题

Pageable 内存不能直接做异步 GPU 拷贝。标准做法是用 pinned 内存做中转：

```
pageable src  ──memcpy──→  pinned bounce  ──musaMemcpyAsync──→  GPU dst
```

但如果 bounce buffer 用后立即释放，地址可能被新分配复用，而 GPU 的 `musaMemcpyAsync` 可能还在读旧数据。BFCAllocator 无法感知 GPU 异步操作。

### 解决方案：Event 驱动的延迟回收

```
Allocate(bytes)
  ├─ 检查 free_list_ 有无 ≥ bytes 的块（best-fit）
  │   有 → 复用
  └─ 无 → musaHostAlloc 新分配

FreeAsync(ptr, bytes, stream)
  ├─ stream=nullptr → musaDeviceSynchronize + musaFreeHost（同步释放）
  └─ 在 stream 上记录 event，块移入 pending_frees_

PollLoop() [独立线程，100μs 间隔]
  └─ PollPendingFrees()
       ├─ musaEventQuery(event) == Success → 块移入 free_list_（可复用）
       ├─ musaEventQuery(event) == NotReady → 跳过
       └─ musaEventQuery(event) == Error   → 移入 free_list_（防御性，避免泄漏）
```

三个阶段的生命周期：

```
musaHostAlloc ──→ [free_list_] ──→ Allocate() 取走
                                       │
                                       │ FreeAsync()
                                       ↓
                                  [pending_frees_] ──→ Poll: event 就绪 ──→ [free_list_]
                                       │
                                       │ 析构时
                                       ↓
                                  musaFreeHost（最终释放）
```

### Best-fit 分配（`:88-97`）

```cpp
for (size_t i = 0; i < free_list_.size(); ++i) {
    if (block.size >= alloc_size) {
        size_t waste = block.size - alloc_size;
        if (waste < best_waste) {
            best_waste = waste;
            best_idx = i;
            if (waste == 0) break;  // 完美匹配，直接停止
        }
    }
}
```

最小分配大小 256 字节，减少碎片。

---

## 三、内存染色与 UAF 检测

文件：`musa_ext/mu/device/musa_allocator.h`

### 核心原理

分配后填充 `0xAB`，释放后填充 `0xCD`：

```
分配: ptr[0..n] = 0xABABABAB...
释放: ptr[0..n] = 0xCDCDCDCD...
```

检测逻辑：
- **分配时**：如果新分配的内存里残留 `0xCDCDCDCD`，说明这块内存被释放后又被人写入——use-after-free
- **释放时**：从 device 读回前 4KB 采样，如果找到了 `0xCDCDCDCD`（上一轮释放的印记），说明在分配期间有人提前写入了释放标记——double-free 或 UAF

### 染色流程

**Alloc 路径（`:314-371`）：**

```
musaMalloc(ptr, size)
  → ApplyMemoryColoring(ptr, size)     // musaMemset(0xAB)
  → RecordAllocation(ptr, size, ...)   // 记录时间戳、alloc_id、stream_id
  → 遥测事件
```

**Free 路径（`:373-416`）：**

```
IsAddressAllocated(ptr)？  // 不在 active 表里 → 疑似 double-free
  → VerifyMemoryColoring(ptr, size)   // D2H 读取前 4KB，扫描 0xCD 残留
  → ApplyFreePattern(ptr, size)       // musaMemset(0xCD)
  → RecordFree(ptr, size, ...)        // 记录释放时间戳
  → 遥测事件
  → musaFree(ptr)
```

### 验证采样的效率考量（`:442-468`）

```cpp
// 只验证前 4KB 采样，不全量检查
const size_t sample_size = std::min(size, static_cast<size_t>(4096));
musaMemcpy(host_buffer, ptr, sample_size, musaMemcpyDeviceToHost);
// 扫描 host_buffer 中的 kMusaFreeMagic
```

全量检查需要把整个张量从 device 读回 host，对于大张量不可接受。4KB 采样在检测能力和性能之间取平衡。

### MemoryForensicsTracker（`:121-297`）

全局单例，维护两层映射：

| 数据结构 | 用途 |
|---------|------|
| `active_allocations_[ptr]` | 当前存活的分配，O(1) 查 UAF |
| `allocation_history_[ptr]` | 每个地址的完整 alloc/free 历史（含时间戳） |

`GenerateReport(ptr)` 输出 Markdown 表格，包含指定地址的所有分配/释放记录。

### 运行时控制

```cpp
EnableMemoryColoring(true);       // 开关染色
EnableMemoryHistoryTracking(true); // 开关历史追踪
EnableMemoryVerification(true);   // 开关释放时采样验证
```

编译期通过 `-DTF_MUSA_MEMORY_COLORING=1` 默认开启，也可运行时通过原子变量开关。

---

## 四、三流架构

文件：`musa_ext/mu/device/musa_device.h` + `musa_device.cc:369-468`

### 为什么需要三个 Stream

GPU 的 compute 和 memory copy 有独立的硬件引擎，可以 overlap：

```
时间线:
  compute stream:  [Kernel A][  Kernel B  ][Kernel C]
  H2D stream:      [CopyIn_B][CopyIn_C]
  D2H stream:                      [CopyOut_A]
```

三个 stream 并发执行，kernel 执行期间可以同时做下一次的输入拷贝和上一次的输出拷贝。

### 构造顺序

```cpp
musaStreamCreate(&stream_)        // compute stream
musaStreamCreate(&h2d_stream_)    // host → device
musaStreamCreate(&d2h_stream_)    // device → host
mudnn_handle_->SetStream(stream_)  // muDNN 绑定 compute stream
mublasSetStream(handle, stream_)   // muBLAS 绑定 compute stream
event_mgr_ = new MusaEventMgr()    // 回调调度器
device_context_ = new MusaDeviceContext(stream_, h2d_stream_, d2h_stream_)
```

关键：**在创建 stream/handle 之前先调 `musaMemGetInfo(&free_memory, &total_memory)`**（`:354-355`），因为 muDNN/muBLAS 初始化会消耗显存，必须先捕获可用内存量。

### 析构顺序（`:471-535`）

```cpp
// 7 步，每步之间都有依赖关系
1. device_context_->Unref()    // 等待所有 stream 操作完成
2. delete event_mgr_           // 处理剩余 callbacks
3. mublasDestroy               // BLAS handle
4. delete pinned_memory_pool_  // 在 event_mgr_ 之后，确保无 callback 引用
5. delete musa_host_allocator_
6. delete musa_allocator_      // BFC allocator
7. musaStreamDestroy(三个)     // stream 此时已空闲
```

顺序的关键约束：
- `device_context_` 必须先释放——它持有 stream 引用且析构函数会 `BlockHostUntilDone()`
- `event_mgr_` 要在 `pinned_memory_pool_` 之前释放——event 回调里面可能引用 pool
- stream 必须在 BFC allocator 之后销毁——allocator 析构可能触发 stream 同步

---

## 五、Event 同步系统

文件：`musa_ext/mu/device/musa_event_mgr.cc`

### 问题 1：musaStreamWaitEvent 是异步的

场景（`:71-93`）：

```cpp
musaEventRecord(sync_event, compute_stream);    // 在 compute stream 上打标记
musaStreamWaitEvent(h2d_stream, sync_event);     // H2D stream 排队等待这个标记
// ⚠️ Wait 只是入队了，此时可能还没执行！
musaEventDestroy(sync_event);                    // ❌ 如果立刻销毁...
//    驱动可能还没处理 wait → wait 被忽略 → compute 没完成就开始 H2D 拷贝
//    → 读到未初始化的 compute 输出 → 脏数据
```

**修复**：把 event 销毁操作也作为回调排队到 `h2d_stream` 上——确保 wait 执行完才销毁：

```cpp
event_mgr_->ThenExecute(h2d_stream_, [sync_event, device_id]() {
    musaSetDevice(device_id);
    musaEventDestroy(sync_event);  // 此时 wait 已执行完毕
});
```

### 问题 2：队头阻塞（Head-of-Line Blocking）

如果用 FIFO 队列（如 `std::deque`），一个慢 event 会卡住后面所有已完成 event 的回调：

```
deque: [Event_A(慢)] → [Event_B(完)] → [Event_C(完)]
                      ↑ B 和 C 都在等 A
```

**修复**（`:161-181`）：用 `std::list` + 迭代器遍历，乱序处理：

```cpp
auto it = used_events_.begin();
while (it != used_events_.end()) {
    musaError_t err = musaEventQuery(event);
    if (err == musaErrorNotReady) {
        ++it;           // 跳过没完成的
    } else {
        to_free->push_back(std::move(iu));
        it = used_events_.erase(it);  // 移除完成的
    }
}
```

### 问题 3：线程饥饿

如果把 poll loop 也放到 threadpool，而所有 threadpool 线程都在处理慢回调，没人去 poll——新完成的 event 的回调永远不会被调度。

**修复**（`:293-299`）：poll 线程是独立的 `std::thread`：

```cpp
polling_thread_ = std::thread(&MusaEventMgr::PollLoop, this);
```

回调本身仍通过 8 线程 threadpool 执行，但 poll 不受影响。

### 问题 4：设备上下文丢失

Threadpool 的线程不持有 MUSA device context。回调中调 `musaSetDevice` 前，MUSA API 调用可能操作错误的 device。

**修复**（`:209-212`）：

```cpp
threadpool_.Schedule([user_func, device_id]() {
    musaSetDevice(device_id);   // 注入 device context
    user_func();
});
```

### Event 复用池

频繁创建/销毁 event 有驱动开销。维护一个 `free_events_` 池：

```cpp
QueueInUse:
    if (!free_events_.empty()) event = free_events_.back(), free_events_.pop_back();
    else event = CreateEvent();  // 池空了才新建

FreeMemory:
    for (const auto& iu : to_free)
        free_events_.push_back(iu.event);  // 用完回收
```

### 关闭流程

析构时（`:33-81`）：
1. `shutting_down_ = true` → 阻止新回调入队
2. `stop_polling_ = true` → poll 循环退出
3. 遍历 `used_events_`：**不 query**（底层 resource 可能已销毁），直接执行回调 + 销毁 event
4. 清理 `free_events_` 中所有回收的 event
5. 处理 `musaErrorInvalidResourceHandle`——关闭时 event 可能已被驱动释放，忽略这个错误
