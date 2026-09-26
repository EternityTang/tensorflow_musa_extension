# Illegal Memory Access 与 H2D 尾延迟优化工作总结

## 1. 工作概览

本阶段围绕 MUSA 后端设备执行和数据拷贝链路，完成了两类基础设施问题的定位与优化：

1. **Illegal Memory Access 问题定位与修复**：针对训练和图执行过程中出现的 `MUSA_ERROR_ILLEGAL_ADDRESS`，排查 H2D 异步拷贝、pageable host memory、pinned bounce buffer 以及跨 Stream Event 同步链路，定位并修复数据尚未对计算 Stream 可见就启动后续 Kernel 的问题。
2. **H2D 尾延迟优化**：针对 TensorFlow Feed 场景下 pageable host memory 拷贝延迟抖动、pinned bounce buffer 重复申请释放以及 Event 管理开销，优化 H2D 拷贝路径和 pinned memory pool，降低 Host-to-Device 拷贝的尾延迟并提升执行稳定性。

两项工作相关，但侧重点不同：

```text
Illegal Memory Access：优先解决异步执行中的正确性和数据可见性问题
H2D 尾延迟优化：在保证正确性的基础上，降低数据拷贝延迟和运行时抖动
```

---

## 2. Illegal Memory Access 问题

### 2.1 相关提交和贡献者

Illegal Memory Access 相关工作包含以下提交：

| 提交 | 提交人 | 主要内容 |
|---|---|---|
| `8157a38` | eilan | 增加 Illegal Memory Access 调试报告，系统定位问题根因 |
| `2e08a25` | VisionaryWind，Signed-off-by Mochi | 修复 H2D Stream Synchronize 和异步资源管理问题 |
| `fb8c3ab` | greenhandF | 修复训练路径中的 Illegal Memory Access 问题 |

需要说明的是，这几笔提交解决的是相关但不完全相同的工作内容：

- eilan 主要负责问题复现、实验排查、根因定位和调试记录；
- VisionaryWind 负责 H2D Stream Synchronize 方向的实际代码修复；
- greenhandF 负责训练执行路径中的另一组非法显存访问问题修复。

### 2.2 问题现象

典型错误表现为：

```text
MUSA_ERROR_ILLEGAL_ADDRESS
Page Directory Fault
```

问题通常发生在复杂 TensorFlow 图执行过程中，表现为某些 MUSA Kernel 读取了尚未准备好的数据，或者访问了无效的设备地址。相关场景具有以下特点：

- CPU 执行结果正常；
- 简单 MUSA 程序中不容易稳定复现；
- TensorFlow 复杂图执行和多 Stream 并发时更容易触发；
- 错误可能表现为 Gather 或其他后续计算 Kernel 的非法地址访问；
- 单纯在每次迭代结束后执行全局同步，不能从根本上解决问题。

### 2.3 eilan 的调试和根因定位

提交：

```text
8157a38 add illegal memory
```

该提交新增：

```text
docs/ilegal_memory/debug_log.md
```

调试过程主要包括：

1. 定位触发 `MUSA_ERROR_ILLEGAL_ADDRESS` 的 Kernel 和执行 Stream；
2. 检查动态计算得到的 `GatherV2` indices，排除普通数据越界是唯一根因的可能性；
3. 通过 CPU 对比测试，确认输入图和数据本身基本正常；
4. 通过增加迭代间同步，排除跨迭代资源累积导致的问题；
5. 将 H2D 异步拷贝改为同步拷贝，确认错误可以消失；
6. 分别测试 pinned host memory、pageable host memory、bounce buffer 和内存池；
7. 对比 `musaStreamSynchronize`、`musaDeviceSynchronize`、`musaStreamWaitEvent` 和 `musaEventSynchronize` 的行为；
8. 最终将问题收敛到 pageable H2D 异步拷贝与 compute Stream 之间的同步和数据可见性。

### 2.4 根因分析

问题涉及如下数据流：

```text
pageable host memory
        ↓ CPU memcpy
pinned bounce buffer
        ↓ H2D async copy
H2D stream
        ↓ Event / Stream wait
compute stream
        ↓
后续计算 Kernel
```

在 TensorFlow 复杂运行环境中，原有的：

```cpp
musaEventRecord(...)
musaStreamWaitEvent(...)
```

虽然 API 调用可以返回成功，但 GPU 侧的异步等待没有可靠地保证 compute Stream 等待 H2D 拷贝真正完成。于是 TensorFlow 可能在数据对 compute Stream 完全可见之前就调度后续 Kernel，导致 Kernel 读取未准备好的数据，最终触发非法显存访问。

调试结果表明：

- `musaStreamSynchronize(h2d_stream)` 只能保证 H2D Stream 完成，不一定能保证 compute Stream 的可见性；
- `musaDeviceSynchronize()` 可以解决问题，但同步范围过大，会损失并发和性能；
- 将 H2D 拷贝放到 compute Stream 可以规避跨 Stream 同步问题，但会削弱拷贝和计算的并行性；
- `musaEventSynchronize()` 这种 host 侧等待方式可以确保 H2D 完成后再通知 TensorFlow；
- 在独立 MUSA 程序中，`musaStreamWaitEvent` 基本工作正常，但在 TensorFlow 的复杂环境下，受到多 Stream、muDNN、BFCAllocator 和 Event Manager 等因素影响，问题更容易暴露。

因此，问题本质可以概括为：

> H2D 异步拷贝已经提交，但原有跨 Stream GPU 异步等待在 TensorFlow 复杂执行环境下不能可靠地建立“拷贝完成 → 后续计算”的先行关系，导致后续 Kernel 读取未完成的数据。

### 2.5 修复方案：`2e08a25`

提交：

```text
2e08a25 fix: bug fix for H2D Stream Synchronize (#177)
```

主要修改文件：

- `musa_ext/kernels/array/musa_resourcegather_op.cc`
- `musa_ext/kernels/array/musa_unique_op.cc`
- `musa_ext/mu/device/musa_device.cc`

核心修复包括：

#### 1. 调整 H2D 完成通知机制

移除对 H2D → compute 方向 `musaStreamWaitEvent` 的单一依赖，改为使用 Event Manager 的 `ThenExecute` 机制：

```text
H2D async copy
        ↓
Event Manager 轮询拷贝完成状态
        ↓
拷贝完成后执行 done()
        ↓
TensorFlow 调度后续计算 Kernel
```

这样既能保证 TensorFlow 不会过早调度后续计算，又避免直接使用 `musaEventSynchronize` 长时间阻塞 host 线程。

#### 2. 修复 `sync_dst_compute` 语义

重新梳理 H2D 和 D2H 拷贝完成后的同步通知关系，确保 TensorFlow 收到完成回调时，相关设备数据已经满足后续使用条件。

#### 3. 统一 D2H 同步方向

对 compute → D2H 的依赖关系进行整理，避免不同路径使用不一致的同步方式。

#### 4. 延长异步 Tensor 生命周期

通过 `TensorReference` 保持异步 D2H 操作期间源 Tensor 有效，防止 Tensor 在异步拷贝完成之前被提前释放。

#### 5. 增加安全检查

对 D2H 路径中的空指针等异常情况增加检查，减少错误状态继续向后传播。

### 2.6 训练路径中的相关修复：`fb8c3ab`

提交：

```text
fb8c3ab fix: fix illegal memory access problem in training
```

提交人：

```text
greenhandF <fanbohao@stu.pku.edu.cn>
```

主要修改文件：

- `musa_ext/kernels/training/musa_applyadam_op.cc`
- `musa_ext/mu/device/musa_device.cc`
- `musa_ext/mu/device_register.cc`
- `musa_ext/mu/optimizer/musa_graph_optimizer.cc`

该提交主要面向训练执行路径，涉及 `ApplyAdam`、MUSA Device、设备注册和图优化器等部分。其工作重点是修复训练场景下的异步执行、设备状态或资源管理问题，避免训练过程中触发非法显存访问。

因此，答辩时建议将其表述为训练路径相关的独立修复，不要与 eilan 的调试报告或 `2e08a25` 的 H2D Stream Synchronize 修复混为同一笔提交。

---

## 3. H2D 尾延迟优化

### 3.1 提交信息

提交：

```text
4741a27 opt: stabilize H2D feed copy, reduce tail latency caused by pageable host memory and event management (#258)
```

主要提交人：

```text
Tang Chien <45093363+tngchien@users.noreply.github.com>
```

共同作者：

```text
timo <timo@mthreads.com>
```

提交时间：

```text
2026-05-21
```

主要修改文件：

- `musa_ext/mu/device/pinned_memory_pool.cc`
- `musa_ext/mu/device/pinned_memory_pool.h`
- `musa_ext/mu/musa_se_plugin.cc`
- `musa_ext/mu/musa_plugin_env.h`
- `CMakeLists.txt`
- `build.sh`
- `setup.py`

改动统计：

```text
7 个文件
217 行新增
103 行删除
```

### 3.2 H2D 的基本背景

H2D 是 Host to Device，即：

```text
CPU 内存 → MUSA 显存
```

TensorFlow Feed 数据不一定位于 pinned host memory，很多情况下使用的是普通 pageable host memory。普通 pageable memory 不适合直接进行高效异步 DMA 拷贝，因此需要先通过 bounce buffer 转换：

```text
pageable host memory
        ↓ CPU memcpy
pinned bounce buffer
        ↓ musaMemcpyAsync
MUSA device memory
```

这种路径虽然可以实现异步 H2D，但会引入额外的：

- CPU 内存复制；
- pinned host memory 分配和释放；
- Event 创建、记录、轮询和销毁；
- 异步 buffer 生命周期管理；
- 多线程锁和内存池管理开销。

当这些开销在高并发或大规模 Feed 场景中叠加时，平均延迟可能并不明显增加，但 P99/P999 等尾延迟会出现较大抖动。

### 3.3 原有实现的问题

原有 H2D 路径主要存在以下问题：

1. pageable host memory 需要使用 pinned bounce buffer，但 buffer 的申请和回收成本较高；
2. 异步拷贝完成后可能频繁创建和销毁 Event；
3. Event 管理和内存释放路径增加了尾部调度开销；
4. pinned、pageable 和 device memory 的处理逻辑不够清晰；
5. 部分同步和异步路径缺少统一的策略控制；
6. 不同硬件和运行场景下，pageable H2D 的稳定性和延迟表现不一致。

因此，该优化的目标不是单纯把 `musaMemcpy` 改成 `musaMemcpyAsync`，而是在保证内存生命周期和拷贝正确性的前提下，减少重复分配、重复创建 Event 以及不必要的同步。

### 3.4 核心优化方案

#### 3.4.1 增加 pinned memory pool 复用

优化后，pageable H2D 使用的 pinned bounce buffer 由内存池统一管理：

```text
申请 pinned bounce buffer
        ↓
将 pageable 数据复制到 bounce buffer
        ↓
执行异步 H2D
        ↓
确认拷贝完成
        ↓
将 buffer 放回 free list
```

内存池复用可以减少：

- `musaMallocHost` 调用；
- `musaFreeHost` 调用；
- 主机 pinned memory 分配器压力；
- 大块内存反复申请释放带来的延迟抖动。

#### 3.4.2 增加 Event 复用

`GPUPinnedMemoryPool` 增加了 Event 空闲列表和统计信息：

```text
free_events_
event_reuse_hits_
event_allocs_
```

当异步拷贝完成时，不再立即销毁 Event，而是回收到空闲 Event 列表；下一次异步释放可以直接复用：

```text
Event 完成
        ↓
回收到 free_events_
        ↓
下次 FreeAsync 时复用
```

这减少了 Event 创建和销毁频率，有助于降低 H2D 路径的尾延迟。

#### 3.4.3 区分不同源内存类型

新增 host pointer 属性识别逻辑，区分：

- pinned host memory；
- pageable host memory；
- device memory。

对应路径为：

```text
Device → Device
    → D2D copy

Pinned Host → Device
    → 直接 H2D copy，可使用异步路径

Pageable Host → Device
    → pinned bounce buffer 或同步 fallback
```

这样避免所有输入都走同一种路径，减少不必要的内存搬运和同步。

#### 3.4.4 优化 pageable bounce buffer 路径

对于 pageable host memory：

1. 从 pinned memory pool 获取 bounce buffer；
2. 使用 CPU `memcpy` 将 pageable 数据复制到 pinned buffer；
3. 使用 `musaMemcpyAsync` 将 pinned buffer 拷贝到设备；
4. 根据同步/异步场景选择合适的完成和回收方式；
5. H2D 完成后再把 bounce buffer 归还内存池。

这样既保持了 pageable memory 的兼容性，又避免了每次调用都创建新的 pinned host allocation。

#### 3.4.5 增加环境变量控制

新增可配置开关，例如：

```text
MUSA_SE_DISABLE_PAGEABLE_H2D_BOUNCE
MUSA_SE_DISABLE_SYNC_H2D_BOUNCE
```

这些开关用于：

- 在出现兼容性问题时快速回退；
- 对比 bounce buffer 和直接同步拷贝的性能；
- 支持不同硬件和负载场景下的路径选择；
- 方便定位 H2D 性能和稳定性问题。

### 3.5 优化价值

这项工作主要改善数据输入和设备调度链路，而不是某个具体计算算子。其价值包括：

- 降低 H2D Feed 拷贝的 P99/P999 延迟；
- 减少 pageable host memory 带来的延迟抖动；
- 减少 pinned memory 反复申请和释放；
- 减少 Event 创建、销毁和管理开销；
- 提高 Host-to-Device 拷贝路径的稳定性；
- 改善 CPU Feed、H2D 拷贝和 GPU 计算之间的流水线效率；
- 为后续计算 Kernel 提供更稳定的数据准备时间。

需要注意的是，当前提交标题明确说明目标是降低 pageable host memory 和 Event management 导致的尾延迟，但提交记录中没有给出统一硬件、统一数据规模下的量化 benchmark。因此，正式答辩材料中应将其表述为路径和机制优化，具体加速比例应以实际复测数据为准。

---

## 4. 两项工作的贡献边界

| 人员 | 主要贡献 |
|---|---|
| eilan | Illegal Memory Access 问题复现、实验排查、H2D/Stream/Event 根因定位和调试报告，提交 `8157a38` |
| VisionaryWind | H2D Stream Synchronize 代码修复，调整 Event 完成通知和异步资源生命周期，提交 `2e08a25` |
| Mochi | `2e08a25` 的 Signed-off-by，不等同于 Git Author |
| greenhandF | 训练执行路径中的 Illegal Memory Access 修复，提交 `fb8c3ab` |
| Tang Chien | H2D Feed 拷贝稳定性和尾延迟优化，提交 `4741a27` |
| timo | `4741a27` 共同作者，参与 H2D 路径优化 |

需要特别区分：

- `8157a38` 是调试报告提交，不是完整的代码修复提交；
- `2e08a25` 是 H2D Stream Synchronize 方向的实际修复；
- `4741a27` 是后续面向 H2D 性能稳定性和尾延迟的优化；
- `fb8c3ab` 主要针对训练路径中的非法显存访问问题。

---

## 5. 两项工作的技术区别

| 对比维度 | Illegal Memory Access | H2D 尾延迟优化 |
|---|---|---|
| 主要目标 | 消除非法显存访问，保证数据可见性和执行正确性 | 降低 H2D 拷贝尾延迟，提升稳定性 |
| 触发问题 | 后续 Kernel 读取尚未完成或不可见的数据 | pageable memory、bounce buffer 和 Event 管理造成延迟抖动 |
| 主要组件 | H2D Stream、compute Stream、Event、ThenExecute | Pinned memory pool、bounce buffer、Event pool |
| 关键机制 | Event polling、异步完成通知、Tensor 生命周期管理 | 内存复用、Event 复用、内存类型分流 |
| 是否主要是性能问题 | 首先是正确性问题，也会影响稳定性 | 主要是性能和尾延迟问题 |
| 典型解决方案 | H2D 完成后再回调 `done()` | 复用 pinned buffer 和 Event，减少分配释放 |
| 代表提交 | `8157a38`、`2e08a25`、`fb8c3ab` | `4741a27` |
| 主要提交人 | eilan、VisionaryWind、greenhandF | Tang Chien，timo 共同参与 |

简而言之：

> Illegal Memory Access 解决的是“数据还没有真正准备好，计算 Kernel 就开始读取”的正确性问题；H2D 尾延迟优化解决的是“数据拷贝虽然可以完成，但 pageable memory、bounce buffer 和 Event 管理导致延迟不稳定”的性能问题。

---

## 6. 转正答辩表述

### 6.1 Illegal Memory Access

> 在 Illegal Memory Access 问题上，我参与了从问题复现、执行链路分析到根因定位的完整排查。通过对比 CPU 和 MUSA 执行结果、分析 Gather 等后续 Kernel、逐步切换同步和异步 H2D 路径，并对 pinned memory、pageable memory、bounce buffer 以及跨 Stream Event 机制进行实验，最终定位到 TensorFlow 复杂执行环境下 H2D Stream 与 compute Stream 之间的异步等待不可靠：H2D 数据尚未对 compute Stream 完全可见时，后续 Kernel 已经开始执行，从而触发 `MUSA_ERROR_ILLEGAL_ADDRESS`。在修复方案上，通过 Event Manager 的异步完成回调，在 H2D 拷贝真正完成后再通知 TensorFlow 调度后续计算，同时完善 D2H 同步和异步 Tensor 生命周期管理，消除了错误的触发条件。

如果需要准确区分个人贡献，可以补充：

> 其中，eilan 主要完成了问题定位和调试报告；VisionaryWind 完成了 H2D Stream Synchronize 方向的代码修复；greenhandF 针对训练执行路径中的非法显存访问进行了独立修复。

### 6.2 H2D 尾延迟优化

> 在 H2D Feed 拷贝优化方面，主要针对 pageable host memory 需要经过 pinned bounce buffer，以及异步拷贝过程中频繁创建和销毁 Event 导致的尾延迟问题进行优化。通过引入 pinned memory pool 和 Event 复用机制，减少 host pinned memory 和 Event 的重复分配释放；同时识别 pinned、pageable 和 device memory，针对不同源内存选择直接异步拷贝、bounce buffer 或同步 fallback 路径，并增加环境变量控制。这样在保证异步拷贝正确性和内存生命周期安全的基础上，降低了 H2D 拷贝的 P99/P999 延迟，提高了 Feed、数据拷贝和 GPU 计算流水线的稳定性。该项优化对应提交 `4741a27`，主要提交人为 Tang Chien，timo 为共同作者。

---

## 7. 说明

以上提交归属和技术内容基于仓库 Git 历史及相关代码变更整理：

- Illegal Memory Access 调试报告：`8157a38`；
- H2D Stream Synchronize 修复：`2e08a25`；
- 训练路径 Illegal Memory Access 修复：`fb8c3ab`；
- H2D Feed 稳定性和尾延迟优化：`4741a27`。

如果将本文用于正式转正材料，建议进一步补充：

- Illegal Memory Access 修复前后的复现率；
- 不同 batch size 和迭代次数下的稳定性数据；
- H2D 优化前后的 P50、P95、P99 和 P999 延迟；
- pageable 与 pinned host memory 的对比数据；
- pinned memory pool 和 Event reuse 的命中率；
- H2D 与 compute overlap 的变化；
- 典型模型或训练任务的端到端收益。
