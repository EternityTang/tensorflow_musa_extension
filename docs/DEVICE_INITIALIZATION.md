# 设备初始化、流管理与内存管理完整流程

本文档详细说明 MUSA 设备从初始化到运行的完整流程，串联设备初始化、流管理和内存管理三个核心机制。

---

## 一、初始化入口（device_register.cc）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TensorFlow 请求创建设备                                  │
│                                                                             │
│   Session 创建时，TensorFlow 调用所有注册的 DeviceFactory                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              MusaDeviceFactory::CreateDevices()                             │
│              (device_register.cc:36-79)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   for (int i = 0; i < musaGetDeviceCount(); ++i) {                          │
│                                                                             │
│       // 1. 获取显存信息                                                    │
│       musaSetDevice(i);                                                     │
│       musaMemGetInfo(&free_memory, &total_memory);                          │
│       size_t memory_limit = free_memory * 0.9;                              │
│                                                                             │
│       // 2. 创建 DeviceAttributes                                           │
│       DeviceAttributes attr;                                                │
│       attr.set_name("/device:MUSA:i");                                      │
│       attr.set_device_type("MUSA");                                         │
│       attr.set_memory_limit(memory_limit);  // ← 传给 MusaDevice            │
│                                                                             │
│       // 3. 创建 MusaDevice                                                 │
│       devices.push_back(new MusaDevice(env, attr, i, executor));            │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
```

---

## 二、MusaDevice 构造函数（musa_device.cc:395-519）

### 阶段 1: 获取显存信息

```cpp
musaSetDevice(device_id_);
musaMemGetInfo(&free_memory, &total_memory);
bfc_memory_limit = free_memory * 0.9;  // 例如 72GB
```

### 阶段 2: 创建 Stream（流管理）

```cpp
musaStreamCreate(&stream_);       // 主计算流
musaStreamCreate(&h2d_stream_);   // Host→Device 传输流
musaStreamCreate(&d2h_stream_);   // Device→Host 传输流
```

### 阶段 3: 创建库 Handle

```cpp
mudnn_handle_ = new ::musa::dnn::Handle();
mudnn_handle_->SetStream(stream_);  // 绑定到计算流

mublasCreate(&mublas_handle_);
mublasSetStream(mublas_handle_, stream_);  // 绑定到计算流
```

### 阶段 4: 创建事件管理器

```cpp
event_mgr_ = new MusaEventMgr(device_id_);
// 内部启动后台轮询线程，处理异步回调
```

### 阶段 5: 创建 DeviceContext

```cpp
device_context_ = new MusaDeviceContext(
    stream_, h2d_stream_, d2h_stream_, executor, event_mgr_
);
// DeviceContext 持有所有 Stream 引用，负责数据传输
```

### 阶段 6: 创建内存分配器（内存管理）

```cpp
// GPU 显存分配器
musa_allocator_ = new BFCAllocator(
    new MusaSubAllocator(device_id_, {}, {}),
    bfc_memory_limit,        // 72GB 上限
    false,                   // allow_growth=false → 预分配模式
    "Musa_BFC_Allocator",
    true                     // garbage_collection
);

// Host 锁页内存分配器
musa_host_allocator_ = new BFCAllocator(
    new MusaHostSubAllocator({}, {}),
    256MB,
    true, "Musa_Host_BFC_Allocator", true
);

// Bounce Buffer 内存池
pinned_memory_pool_ = new GPUPinnedMemoryPool(device_id_);
```

### 阶段 7: 注册到 TensorFlow

```cpp
gpu_device_info_.stream = device_context_->stream();
gpu_device_info_.default_context = device_context_;
gpu_device_info_.gpu_id = device_id_;
set_tensorflow_gpu_device_info(&gpu_device_info_);
```

---

## 三、资源持有关系图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MusaDevice                                         │
│                      (代表一个 MUSA GPU)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        流管理                                        │   │
│  │                                                                     │   │
│  │   stream_ ───────────────► 主计算流                                 │   │
│  │                              │ Kernel 执行                          │   │
│  │                              │ MuDNN/MuBLAS 绑定                    │   │
│  │                              │                                     │   │
│  │   h2d_stream_ ──────────► Host→Device 传输流                        │   │
│  │                              │ 异步数据上传                         │   │
│  │                              │                                     │   │
│  │   d2h_stream_ ──────────► Device→Host 传输流                        │   │
│  │                              │ 异步数据下载                         │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        库 Handle                                     │   │
│  │                                                                     │   │
│  │   mudnn_handle_ ───────► MuDNN 句柄                                 │   │
│  │                              │ 绑定到 stream_                       │   │
│  │                              │                                     │   │
│  │   mublas_handle_ ───────► MuBLAS 句柄                               │   │
│  │                              │ 绑定到 stream_                       │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        内存管理                                      │   │
│  │                                                                     │   │
│  │   musa_allocator_ ──────► BFCAllocator                              │   │
│  │                              │                                      │   │
│  │                              ├──► MusaSubAllocator                  │   │
│  │                              │         │                            │   │
│  │                              │         └──► musaMalloc()           │   │
│  │                              │                                      │   │
│  │                              └──► 内存池管理                        │   │
│  │                                                                     │   │
│  │   musa_host_allocator_ ─► BFCAllocator                              │   │
│  │                              │                                      │   │
│  │                              └──► MusaHostSubAllocator              │   │
│  │                                        │                            │   │
│  │                                        └──► musaHostAlloc()        │   │
│  │                                                                     │   │
│  │   pinned_memory_pool_ ──► GPUPinnedMemoryPool                       │   │
│  │                              │                                      │   │
│  │                              ├──► free_list_ (可复用)              │   │
│  │                              │                                      │   │
│  │                              └──► pending_frees_ (等待GPU完成)      │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        异步管理                                      │   │
│  │                                                                     │   │
│  │   event_mgr_ ───────────► MusaEventMgr                              │   │
│  │                              │                                      │   │
│  │                              ├──► polling_thread_ (轮询线程)        │   │
│  │                              │                                      │   │
│  │                              ├──► used_events_ (待处理事件)         │   │
│  │                              │                                      │   │
│  │                              └──► threadpool_ (回调线程池)          │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        DeviceContext                                │   │
│  │                                                                     │   │
│  │   device_context_ ──────► MusaDeviceContext                         │   │
│  │                              │                                      │   │
│  │                              ├──► 持有 stream_ 引用                 │   │
│  │                              │                                      │   │
│  │                              ├──► 持有 h2d_stream_ 引用             │   │
│  │                              │                                      │   │
│  │                              ├──► 持有 d2h_stream_ 引用             │   │
│  │                              │                                      │   │
│  │                              ├──► CopyCPUTensorToDevice()           │   │
│  │                              │                                      │   │
│  │                              └──► CopyDeviceTensorToCPU()           │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、运行时协作流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     运行时：session.run(matmul)                              │
└─────────────────────────────────────────────────────────────────────────────┘

第 1 步: TensorFlow Executor 获取设备
──────────────────────────────────────────────────────────────────────────────
    Device* device = GetDevice("/device:MUSA:0");
    // 返回 MusaDevice 实例


第 2 步: 分配输入 Tensor（内存管理）
──────────────────────────────────────────────────────────────────────────────
    ctx->allocate_output(0, shape, &input)
        │
        ├── device->GetAllocator(attr)
        │       │
        │       └── MusaDevice::GetAllocator()
        │               │
        │               └── return musa_allocator_;  // BFCAllocator
        │
        └── BFCAllocator::AllocateRaw(size)
                │
                ├── 内存池有空间？→ 直接返回
                │
                └── 内存池无空间？
                        │
                        └── MusaSubAllocator::Alloc()
                                │
                                └── musaMalloc(72GB) ← 第一次分配时预分配


第 3 步: 数据上传（流管理 + 内存管理）
──────────────────────────────────────────────────────────────────────────────
    DeviceContext* ctx = device->TryGetDeviceContext();
    // 返回 MusaDeviceContext

    ctx->CopyCPUTensorToDevice(cpu_tensor, device, gpu_tensor, done)
        │
        ├── 检查源内存类型（pinned vs pageable）
        │
        ├── [pageable 路径]
        │       │
        │       ├── pinned_memory_pool_->Allocate(bytes)  // 内存管理
        │       │
        │       ├── std::memcpy(bounce, src, bytes)
        │       │
        │       └── musaMemcpyAsync(dst, bounce, bytes, h2d_stream_)  // 流管理
        │
        ├── [pinned 路径]
        │       │
        │       └── musaMemcpyAsync(dst, src, bytes, h2d_stream_)  // 流管理
        │
        └── 跨流同步
                │
                ├── musaEventRecord(event, h2d_stream_)
                │
                └── musaStreamWaitEvent(stream_, event)  // compute 等待 H2D


第 4 步: Kernel 执行（流管理）
──────────────────────────────────────────────────────────────────────────────
    MatMulOp::Compute(ctx)
        │
        ├── auto& handle = GetHandleByCtx(ctx)
        │       │
        │       └── MusaDevice::mudnn_handle()  // 绑定到 stream_
        │
        └── mMatMul::Run(handle, output, input)  // 在 stream_ 上执行


第 5 步: 数据下载（流管理 + 内存管理）
──────────────────────────────────────────────────────────────────────────────
    ctx->CopyDeviceTensorToCPU(gpu_tensor, device, cpu_tensor, done)
        │
        ├── musaEventRecord(event, stream_)  // 记录 compute 完成
        │
        ├── musaStreamWaitEvent(d2h_stream_, event)  // D2H 等待 compute
        │
        ├── musaMemcpyAsync(dst, src, bytes, d2h_stream_)  // 异步下载
        │
        └── event_mgr_->ThenExecute(d2h_stream_, done)  // 注册回调
                │
                └── 后台轮询线程检测 event 完成 → 执行 done()


第 6 步: 内存释放（内存管理）
──────────────────────────────────────────────────────────────────────────────
    Tensor 析构
        │
        └── BFCAllocator::DeallocateRaw(ptr)
                │
                └── 标记为空闲，加入内存池
                        │
                        └── (garbage_collection=true 时可能调用 musaFree)
```

---

## 五、完整时序图

```
┌─────────┐     ┌──────────────┐     ┌─────────┐     ┌──────────┐     ┌─────────┐
│TensorFlow│     │ MusaDevice   │     │BFCAlloc │     │MusaSubAlloc│    │ MUSA API│
└────┬────┘     └──────┬───────┘     └────┬────┘     └─────┬────┘     └────┬────┘
     │                 │                  │                │               │
     │ ══════════════ 初始化阶段 ═════════════════════════════════════════│
     │                 │                  │                │               │
     │ CreateDevices() │                  │                │               │
     │────────────────>│                  │                │               │
     │                 │                  │                │               │
     │                 │ musaMemGetInfo() │                │               │
     │                 │──────────────────────────────────────────────────>│
     │                 │                  │                │               │
     │                 │ musaStreamCreate()×3              │               │
     │                 │──────────────────────────────────────────────────>│
     │                 │                  │                │               │
     │                 │ new BFCAllocator │                │               │
     │                 │─────────────────>│                │               │
     │                 │                  │                │               │
     │ ══════════════ 运行时阶段 ═════════════════════════════════════════│
     │                 │                  │                │               │
     │ allocate_output │                  │                │               │
     │────────────────>│                  │                │               │
     │                 │ GetAllocator()   │                │               │
     │                 │─────────────────>│                │               │
     │                 │                  │                │               │
     │                 │                  │ AllocateRaw()  │               │
     │                 │                  │ (池空，第一次) │               │
     │                 │                  │───────────────>│               │
     │                 │                  │                │               │
     │                 │                  │                │ musaMalloc()  │
     │                 │                  │                │──────────────>│ 72GB
     │                 │                  │                │               │
     │                 │                  │<───────────────│               │
     │<────────────────│<─────────────────│                │               │
     │                 │                  │                │               │
     │ ══════════════ 后续分配 ═══════════════════════════════════════════│
     │                 │                  │                │               │
     │ allocate_output │                  │                │               │
     │────────────────>│                  │                │               │
     │                 │ GetAllocator()   │                │               │
     │                 │─────────────────>│                │               │
     │                 │                  │                │               │
     │                 │                  │ AllocateRaw()  │               │
     │                 │                  │ (池中有空间)   │               │
     │                 │                  │──────┐         │               │
     │                 │                  │      │ 从池切分 │               │
     │                 │                  │<─────┘         │               │
     │<────────────────│<─────────────────│                │               │
     │                 │                  │                │               │
```

---

## 六、MusaDevice 和 MusaDeviceContext 的函数作用

### MusaDevice（继承自 TensorFlow Device）

**必须实现的接口函数：**

| 函数 | 作用 | 何时被调用 |
|------|------|-----------|
| `GetAllocator(attr)` | 返回内存分配器 | OpKernel 分配 Tensor 时 |
| `TryGetDeviceContext()` | 返回 DeviceContext | TensorFlow 需要数据传输时 |
| `Sync()` | 同步设备 | 显式调用 `device.sync()` |

**本项目添加的辅助函数：**

| 函数 | 作用 | 调用者 |
|------|------|--------|
| `GetStream()` | 返回 compute stream | Kernel 获取执行流 |
| `get_device_id()` | 返回 GPU 设备 ID | 各处需要设备 ID 时 |
| `mudnn_handle()` | 返回 MuDNN 句柄 | Kernel 调用 MuDNN API |
| `mublas_handle()` | 返回 MuBLAS 句柄 | Kernel 调用 MuBLAS API |
| `event_mgr()` | 返回事件管理器 | 异步回调注册 |
| `pinned_memory_pool()` | 返回锁页内存池 | 数据传输时 |
| `musa_host_allocator()` | 返回 Host 内存分配器 | 分配 CPU 锁页内存 |

### MusaDeviceContext（继承自 TensorFlow DeviceContext）

| 函数 | 作用 | 何时被调用 |
|------|------|-----------|
| `CopyCPUTensorToDevice()` | CPU → GPU 数据传输 | Feed 数据到 GPU |
| `CopyDeviceTensorToCPU()` | GPU → CPU 数据传输 | 从 GPU 取结果 |
| `stream()` | 返回 Stream 对象 | TensorFlow Executor 调度 |
| `ThenExecute()` | 在 stream 完成后执行回调 | 异步操作完成时 |

---

## 七、总结

**MusaDevice 构造函数是枢纽：它按顺序创建 Stream（流管理）、库 Handle、EventMgr、内存分配器（内存管理），并将它们通过 DeviceContext 暴露给 TensorFlow，形成完整的设备抽象。**

| 类 | 职责 |
|------|------|
| **MusaDevice** | 代表一个 MUSA GPU 设备，持有所有资源（Stream、Allocator、Handle），实现 TensorFlow Device 接口 |
| **MusaDeviceContext** | 处理 CPU↔GPU 数据传输，管理 Stream 相关操作，实现 TensorFlow DeviceContext 接口 |
| **BFCAllocator** | TensorFlow 提供的内存池，管理 GPU 显存分配和回收 |
| **MusaSubAllocator** | 本项目实现，封装 musaMalloc/musaFree |
| **GPUPinnedMemoryPool** | 本项目实现，管理 Bounce Buffer |
| **MusaEventMgr** | 本项目实现，异步事件轮询和回调管理 |

---

**文档版本**: 2026-04-10
**适用版本**: TensorFlow MUSA Extension v1.0+