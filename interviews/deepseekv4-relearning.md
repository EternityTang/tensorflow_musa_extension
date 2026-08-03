# DeepSeek V4 FP8 MUSA 实习复习手册：第 3～10 章

> 更新时间：2026-08-01  
> 来源：`docs/resume_docs/deepseekv4_fp8_prefill_optimizations.md`、`docs/resume_docs/resume_final.md` 及本轮逐章梳理。  
> 使用原则：严格区分官方能力、MUSA 移植、S5000 针对性调优、本地 Kernel/系统设计；性能数字没有完整复现条件时统一标成“项目口径”。

## 0. 总体主线

```text
官方算法与框架语义
  DeepSeek V4 / SGLang / DeepGEMM / FlashMLA
                     ↓
MUSA 平台适配
  device / dtype / layout / JIT / MCCL / graph
                     ↓
S5000 性能重构
  消除中间物化 / Shape Dispatch / Prefill-Decode 分流
                     ↓
Serving 状态闭环
  Paged Cache / MTP / PP-PD / fallback / capture-replay
```

一句话定位：这不是重写 DeepSeek 算法，而是在保持官方数学与接口语义的前提下，为 S5000 重建高效、正确、可验证的 MUSA 执行路径。

### 归属标记

| 标记 | 含义 |
| --- | --- |
| `O` | 官方已有算法、接口或系统语义 |
| `P` | CUDA/NV 路径到 MUSA 的平台移植与接入 |
| `T` | 针对 S5000 Shape、带宽或 Launch 特征的调优 |
| `R` | 本地重写或新增的 MUSA Kernel/Pipeline |
| `S` | Graph、Cache、Metadata、通信和状态生命周期闭环 |

---

## 3. MoE FP8 Prefill：基于官方 Expert-Contiguous 的 S5000 Fixed-Bucket Compact Pipeline

### 3.1 这一章做了什么

官方已经提供 Expert-Contiguous 布局、`src2dst`、`m_indices` 和 DeepGEMM Contiguous Grouped GEMM。本地工作不是重新设计 MoE，而是围绕它们构造 S5000 大 Prefill 数据通路：

```text
hidden [T,H] + topk [T,k]
  → route count / prefix / src2dst
  → fused quant-scatter
  → compact_input [M,H] FP8 + scale + m_indices
  → grouped gate/up GEMM
  → masked SwiGLU + quant
  → grouped down GEMM
  → reorder/combine [T,H]
```

核心增量：

1. 将 FP8 Quant 与 Scatter 融合，直接写 Expert-Contiguous Buffer。
2. 用 Fixed-Bucket/Static-Cap 固定 Workspace Shape，支持 Graph Capture。
3. 区分 valid、padded、allocated、worst-case 四种行数口径。
4. 热 Expert 超容时置 Overflow Flag 并回退，不能截断 Route。
5. 只让大 Prefill 进入 Compact 路径，保护 Decode/Small-M。

### 3.2 官方基线

官方已经有：

- Expert-Contiguous 数据布局。
- `ep_scatter` / `ep_gather` 及路由 Metadata。
- `src2dst`：`(token, top-k slot) → compact row`。
- `m_indices`：`compact row → expert`。
- 两段 `grouped_gemm_nt_f8f8bf16_contig`。
- Fused SwiGLU/SiLU + FP8 Quant。

因此不能说“设计了 Expert-Contiguous”或“实现了 DeepGEMM”。

### 3.3 为什么 S5000 的重点不在 GEMM

以文档假设 Shape 为例：

```text
T=8192, H=4096, E=256, top-k=8, I=2048
M=T×top-k=65536
```

Gate/Up 和 Down GEMM 的算术强度很高，偏 Compute-bound；Scatter、SwiGLU、Quant、Reorder 的算术强度只有个位数，深度 Memory-bound。

所以优化优先级是：

```text
减少 GEMM 周围的 HBM 往返与 Launch
而不是继续调整已经高效的 DeepGEMM 主体
```

### 3.4 Fused Quant-Scatter

原路径：

```text
hidden [T,H] BF16
→ per-token group quant
→ x_q [T,H] FP8 + scale
→ ep_scatter
→ compact_input [M,H] FP8 + compact_scale
```

问题是 `x_q [T,H]` 只是一个短生命周期中间 Tensor，需要完整写回，再按 `src2dst` 散读。

融合路径：

```text
读取源 Token 的 hidden
→ 每 128 元素求 amax/scale
→ FP8 cast/pack
→ 按 src2dst 直接写 compact row 和 compact scale
```

Kernel 映射：

```text
group_linear = block_id * groups_per_cta + warp_id
token_id     = group_linear // (H / 128)
scale_group  = group_linear %  (H / 128)
```

组内完成：

```text
load → abs-max reduce → scale → FP8 cast/pack → compact store
```

量化必须以源 Token 行为并行方向，因为同一个 Token 被路由到 k 个 Expert 时，k 份 Compact Row 的量化字节和 Scale 相同；如果按目标 Compact Row 重复读取 BF16 并量化，会把源数据重复读取 k 次。

文档假设下流量推导：

```text
Split 路径约 655 MB/层
Fused 路径约 344 MB/层
差值约 311 MB/层
```

这个 311 MB 是按读写字节公式推导，不是实机测量。

### 3.5 Fixed-Bucket 容量契约

MoE 路由分布运行时动态，但 Graph Capture 希望 Buffer Shape 和地址稳定。因此预分配：

```text
allocated_rows = E × cap_per_expert
```

四种行数必须分清：

| 字段 | 含义 |
| --- | --- |
| `valid_routes` | 实际命中本地 Expert 的 Route 数 |
| `padded_valid_rows` | 每个 Expert 对齐到 GEMM Block 后的总行数 |
| `allocated_rows` | 实际预分配的 Compact Buffer 行数 |
| `worst_case_rows` | 极端不均衡路由下的审计上界 |

Static-Cap 的取舍：

| | 优点 | 代价 |
| --- | --- | --- |
| Static Cap | Shape/地址固定，Graph 友好 | Expert 不均衡时 Padding 浪费 |
| Exact Dynamic | 几乎没有容量浪费 | Shape 动态，Capture 困难 |

如果热 Expert 超过 `cap_per_expert`：

```text
设置 overflow flag
→ 显式回退动态/原路径
```

不能截断 Route，因为那会静默改变模型输出。

### 3.6 Dispatch 与完整执行链

主要 Guard：

```text
MUSA device
实验开关开启
FP8/DeepGEMM contiguous runner 可用
二维 hidden
Prefill/Extend 语义
token >= 8192
H % 128 == 0
group_size == 128
dtype/layout/scale shape 合法
```

完整路径：

```text
Route count/prefix/src2dst
→ Quant-Scatter
→ Gate/Up Grouped GEMM
→ Masked SwiGLU + FP8 Quant
→ Down Grouped GEMM
→ Reorder/Combine
```

注意：Gate/Up、SwiGLU+Quant、Down GEMM 仍是三个阶段。Quant-Scatter 只融合 Quant 和 Scatter，不能说成 MoE 端到端全融合。

`token >= 8192` 和 SwiGLU Quant 的 `rows >= 1024` 是不同 Guard。目前没有充分 Sweep 证据，只能称 Shape 分档。

### 3.7 归属表

| 工作 | 分类 |
| --- | --- |
| Expert-Contiguous、`src2dst`、`m_indices` | 官方已有 |
| DeepGEMM 两段 Grouped GEMM | 官方已有 |
| Fused SwiGLU + FP8 Quant | 官方已有，本地接线 |
| MUSA DeepGEMM 接入与 Layout 对齐 | MUSA 移植 |
| Fused Quant-Scatter MUSA Kernel | 本地新增 Kernel |
| Fixed-Bucket、四口径分账、Overflow 回退 | 本地系统设计 |
| Token/Rows/Hidden Shape Dispatch | S5000 调优 |

### 3.8 面试表述与证据边界

准确说法：

> DeepGEMM 和 Expert-Contiguous 布局是上游的。我在 MUSA 上接通这条链路，并针对 S5000 的低带宽特征实现 Fused Quant-Scatter，消除 `[T,H]` FP8 中间张量；同时用 Fixed-Bucket 和 Overflow Fallback 让动态 MoE 路由满足 Graph Capture，并通过大 Prefill Shape Dispatch 保护 Decode。

项目口径 MoE 算子约 `1.6×`、端到端 Prefill 约 `+8%`，但缺完整复现实验上下文，不能表述为本人独立复测。

---

## 4. TopK、Routing 与 C4 Indexer：从官方语义到 S5000 Hot-Shape Kernel

### 4.1 这一章做了什么

这章覆盖两类不同工作：

1. MoE/Sparse 路径中的 TopK、Routing 和 Page Index Transform。
2. C4 Indexer 后处理中的 RoPE/Hadamard Fusion。

官方提供 TopK/Routing 的语义和输出契约；本地工作是在 MUSA 上实现热点 Shape Kernel、Exact/Fallback 机制和 C4 Guarded Fusion。

### 4.2 TopK-512 不是普通 MoE TopK

输入：

```text
scores          [rows, max_seq_len] FP32
seq_lens        [rows] INT32
page_tables     [rows, num_pages] INT32
out_page_indices[rows, topk] INT32
```

选中 Raw Position `r` 后还要转换成 Paged Cache 物理索引：

```text
logical_page  = r >> log2(page_size)
offset        = r & (page_size - 1)
physical_page = page_tables[row, logical_page]
page_index    = (physical_page << log2(page_size)) | offset
invalid       = -1
```

因此它是“Select + Page Index Transform”，不是普通 Expert ID TopK。

### 4.3 两类生产 Shape

| 场景 | Scores | 输出 | 适合路径 |
| --- | --- | --- | --- |
| Prefill Flattened | `[2048,64]` | `[2048,512]` | 行多而短，按有效长度搬运/映射 |
| 长上下文 Paged Load | `[16,262208]` | `[16,512]` | 行少而极长，需要真正 Select/Radix |

“TopK512”不能固定对应一个 Kernel，因为两种 Shape 的并行度来源完全不同。

### 4.4 Exact Radix 与 Candidate

大 Shape 的 Exact 路径使用四轮 Byte-Radix：

```text
count/histogram
→ 定位第 512 名所在 Bucket
→ 逐轮缩小阈值范围
→ gather/prefix/write
```

只需要确定第 512 名阈值，不需要完整排序。

Candidate 路径先收集有界候选再精选，可能更快，但候选容量可能溢出。真实热点 `[16,262208] → [16,512]` 中，`candidate_capacity=1024` 曾无法容纳正确候选，因此不能作为默认 Exact 路径。

### 4.5 Graph-Safe Exact/Candidate 机制

Candidate 是否 Overflow 通常需要读取 Flag：

```text
device flag → host read → host branch
```

这会破坏 Graph Capture。安全机制是：

```text
先运行 Exact，保证结果正确
→ 再运行 Bounded Candidate
→ Candidate 无 Overflow 时在 Device 侧覆盖 Exact 输出
```

代价是两条路径都执行；收益是不需要 Host 同步且始终有 Exact 结果。

### 4.6 Dispatch 和负面实验

典型顺序：

```text
Page Size 合法性检查
→ MUSA/MATE Opt-in
→ Sequential TileLang
→ Small-N
→ Two-Phase
→ 显式实验 Variant
→ TopK512 Long Select
→ Graph 中仍 Miss：Fail-Closed
→ 非 Graph：Torch Reference Fallback
```

多个候选没有进入默认路径的原因包括：

- Split-Radix：Launch 和长行扫描次数过多。
- Split-Hist：多 Kernel、同步和原子开销高。
- Chunk-Merge：Chunk 内仍做大量串行 Select。
- Direct Gather：全局原子争用严重。
- Warp-Hist：同 Bucket 冲突不足以摊销额外指令。

所谓 `topk_transform_512_v2_musa` 在材料对应版本中只是验证 Metadata 后转调 V1，没有实现 CUDA V2 的 Cluster-Persistent Scheduling。因此不能写“实现了 TopK V2”。

### 4.7 C4 Indexer Fusion

原链：

```text
Index Selection/Write
→ RoPE
→ Hadamard
```

满足 H64、Dtype、Layout 等 Guard 时进入：

```text
Fused C4 Indexer + RoPE + Hadamard
```

融合消除阶段间 Tensor Materialization 和 Launch；TopK Select 本身仍由前面的 Kernel 完成。

### 4.8 归属表

| 工作 | 分类 |
| --- | --- |
| TopK/Routing 数学和输出格式 | 官方已有 |
| Page Table/Indexer 语义 | 官方已有 |
| MUSA TopK V1/TopK512 Kernel | MUSA 重实现/热点 Kernel |
| Shape Mapping 和 Exact Stable Path | S5000 调优 |
| Exact/Candidate Graph-Safe 双跑 | 本地系统机制 |
| Non-Exact Opt-in Gate | 正确性治理 |
| C4 Indexer/RoPE/Hadamard Fusion | MUSA Fusion；当前官方也有同类能力，不能泛称原创 |

### 4.9 面试表述

> 我保留官方 TopK 和 Routing 语义，在 MUSA 上针对短行多 Row 与超长行少 Row 两类热点分别实现和选择 Kernel。长行 TopK512 使用 Exact Radix Select，并用 Exact/Candidate 双跑避免 Overflow 判断引入 Host 同步；另外将 Page Index Transform 合入输出，并为 C4 Indexer 的 RoPE/Hadamard 增加受 Guard 保护的融合路径。

没有统一实测倍数，只能强调 Correctness、Shape Coverage、Graph Safety 和 Dispatch 机制。

---

## 5. FP8 Activation Quant 与 C4/C128 Compress：围绕 S5000 带宽重写

### 5.1 总体目标

```text
BF16/FP16 Activation
  ├─ FP8 Quant → FP8 + Scale → FP8 GEMM
  └─ C4/C128 Compress → 压缩状态/Cache → Attention
```

这两类操作计算量低、读写量高。目标不是改变数学，而是减少显存访问、提高写回宽度、批量处理更多 Group，并隔离 Decode Tiny-M。

### 5.2 官方 FP8 Quant 语义

Group Size 为 128：

```text
amax  = max(abs(x[0:128]))
scale = max(amax / 448, eps)
x_fp8 = cast(x / scale)
```

输出：

```text
x_q [rows, hidden]       FP8 E4M3
x_s [rows, hidden / 128] FP32 Row-Major
```

量化公式、Scale Layout、输出 Dtype，以及 Fused SiLU/Clamp + FP8 Quant 都是官方能力。

### 5.3 通用实现的问题

通用路径通常一个 Program 处理一个 Group：

```text
读 128 值 → amax → scale → cast → 写 FP8/scale
```

在 S5000 大 Prefill 上：

- 每个 Group 独立调度，Program 过碎。
- Scale/Store 粒度小。
- Generic TileKernels Cast 在大 CP Prefill 上可能变成慢路径。
- Small-M 又不适合为大批量 Kernel 支付固定成本。

### 5.4 Half-Warp Quant MUSA 重实现

```text
16 lanes × 每 lane 8 个连续值 = 128/group
一个 Warp 同时处理 2 groups
一个 CTA 处理多个 groups
```

步骤：

1. 每 lane 向量加载 8 个值并转 FP32。
2. 计算局部 Abs-Max。
3. Half-Warp Shuffle Reduce 得到 Amax。
4. 计算 `scale=max(amax/448,eps)`。
5. 转换并打包 8 个 FP8。
6. 连续写 `x_q`，一个 Lane 写 `x_s`。

没有单独 Amax Tensor，也没有第二个 Cast Kernel。

这套 `16×8` 结构在上游 `sgl-kernel` 中已存在，所以准确说法是“MUSA 重实现”，不能说自己设计了 Half-Warp 量化算法。

### 5.5 S5000 Shape Dispatch

```text
total_groups = rows × hidden / 128
```

```text
total_groups < 16K
→ Small/Medium 通用路径

total_groups >= 16K
→ TileLang Half-Warp
→ 或 Triton Multi-Group Fallback
```

Launch 分档：

| 条件 | `groups_per_cta` | `num_warps` |
| --- | ---: | ---: |
| `hidden <= 2048` | 8 | 4 |
| `hidden > 2048 && rows <= 4096` | 4 | 2 |
| 其他大 Shape | 8 | 2 |

重要的“负向 Dispatch”：当 `total_groups>=16K` 时，主动阻止大 CP Prefill 落回已知较慢的 Generic Cast Helper。

16K 和三组配置没有完整 Sweep 证据，只能称 Shape-Based Selection。

### 5.6 C4/C128 Compress

C4 是短归约：

```text
短 Reduce → 向量写回
```

重点是避免复杂归约固定成本，并在 Decode 下用 Paged C4 增量更新，而不是重新压缩整页。

C128 是长归约：

```text
多个线程处理 128 个元素
→ 并行 Reduce
→ 合并结果
→ 向量写回
```

目的是避免单线程或单 Warp 的长串行依赖链。

Paged C4 Decode 的关键索引：

```text
block_id  = indices[token]
write_pos = (seq_len + 3) % 4
```

`(seq_len+3)%4` 等价于 `(seq_len-1)%4`，因为 `seq_len` 已包含当前新增 Token。

### 5.7 归属表

| 工作 | 分类 |
| --- | --- |
| FP8 Group Quant 数学、Scale、Dtype | 官方已有 |
| Fused SiLU/Clamp + Quant | 官方已有 |
| C4/C128 压缩语义 | 官方已有 |
| TileKernels MUSA Bring-Up 与 Guard | MUSA 移植 |
| Half-Warp Quant 结构 | 上游标准结构的 MUSA 重实现 |
| 16K 分流、三组 Launch 配置 | S5000 调优 |
| 大 Prefill 阻断 Generic Cast | 本地 Dispatch 发现 |
| Triton Multi-Group | S5000 Kernel 重写 |
| C4 向量写回/C128 并行 Reduce | S5000 Kernel 重写，算法范式非原创 |
| Paged C4 Decode 增量 Kernel | 本地 MUSA Kernel |

### 5.8 面试表述

> 我保留官方 FP8 Quant 和 C4/C128 Compress 语义，将上游 Half-Warp Group Quant 结构重新实现到 MUSA，并针对 S5000 大 Prefill 增加 16K Group 分流、多 Group CTA、Shape 分档和慢路径隔离；同时为 Decode 实现 Paged C4 增量压缩 Kernel。

“Quant/Compress 前处理约 1.8～2×”是混合项目口径，不能当成单算子独立复测。

---

## 6. FlashMLA Sparse Prefill：从官方 Kernel 接口到 S5000 Seq-Pack

### 6.1 这一章做了什么

FlashMLA Sparse Kernel 消费：

```text
Query
连续 KV/Compressed Workspace
每 Query 的 Sparse Indices
每 Query 的 Valid Length
```

Serving 中不同 Request 的 Cache 是分散的，调用前要 Gather Compressed Region/SWA、构造 Metadata、合并索引和 Padding。Seq-Pack 优化的是主 Attention Kernel 之前的供数链路，不是 Sparse Attention 数学。

### 6.2 Seq-Pack

```text
各 Request Compressed Region + SWA Region
→ 一次 Pack 到 Flat BF16 Workspace
→ TopK/SWA Index Rebase 到 Workspace 坐标
→ -1 Padding + Combined Length
→ flash_mla_sparse_fwd
```

例子：

```text
request1 compressed_base=80, swa_base=116
local topk=[2,9,-1]
local swa =[0,1,2,3]

combined=[82,89,116,117,118,119,-1,...]
combined_len=6
```

`-1` 是 Sentinel；`combined_len` 才是有效前缀长度。

### 6.3 Chunk-Level Cache

一个 Prefill Chunk 会经过约 61 层。跨层可复用：

```text
query_start_loc / seq_lens
SWA token ids / offsets
C0 workspace/indices
C128 workspace/indices
C4 workspace buffer/base
```

生命周期：

| 模式 | 跨层行为 |
| --- | --- |
| C0 | Workspace/Indices 可预计算 |
| C128 | 固定位置结构，可预计算 |
| C4 | Buffer/Base 复用，TopK 有效前缀逐层更新 |

### 6.4 Pack8

Pack8 将 8 个 Query 行组合为一次调用：

```text
q0...q7 → flash_mla_sparse_fwd_pack8
```

正确性约束：

- Row Mask：Padding Query 不贡献 Attention。
- Pack Lens：每个 Query 只读自己的有效 Prefix。
- `q.shape[0] % 8 == 0`。
- 每个 Request 的 `extend_seq_len % 8 == 0`。
- Pack 不能跨 Request 边界。

Request 尾部不足时通过 Request Padding 规则化，同时维护 Mask 和 Valid Length；不能只在整个 Batch 尾部补零。

### 6.5 CP Round-Robin

CP Round-Robin 下 Local Query 不是 Global Query 的连续切片：

```text
GPU0: q0,q2,q4...
GPU1: q1,q3,q5...
```

因此必须重建：

```text
local query_start_loc
local extend_seq_lens
local req_pool_indices
local seq_lens
```

这是正确性适配，不是性能发明。

### 6.6 归属表

| 工作 | 分类 |
| --- | --- |
| Sparse Attention 数学和 `flash_mla_sparse_fwd` | 官方已有 |
| Compressed TopK + SWA 输入语义 | 官方已有 |
| Seq-Pack/Flat Workspace/Index Rebase | 当前官方已有 |
| `SparsePrefillChunkCache` | 当前官方已有 |
| MUSA 接口、Dtype/Layout/Workspace 接入 | MUSA 移植 |
| CP Round-Robin Metadata 重建 | 平台正确性适配，官方也有相关机制 |
| `flash_mla_sparse_fwd_pack8` | 本地新增变体 |
| 两类 Pack8 Metadata、Row Mask、Pack Lens | 本地新增 |
| Request-Level Alignment/Padding | Pack8 配套设计 |

### 6.7 面试表述

> 我接入并适配官方 FlashMLA Sparse Prefill/Seq-Pack 数据通路，使连续 Workspace、索引重定位、Chunk Cache 和 CP Metadata 在 MUSA 上闭环；在此基础上实现官方没有的 Pack8 Kernel 变体及其 Row Mask、Pack Lens 和 Request-Level Alignment。

不能说“设计了 Seq-Pack/FlashMLA Sparse Kernel”。当前官方已有这部分主体；本地最明确的独有增量是 Pack8。

“Sparse 前处理及 Metadata 构建约 2×”只是局部项目口径，不等于 Sparse Attention 或 TTFT 端到端 2×。

---

## 7. Cache、Norm、RoPE 与 MHC：Workload 分流与 Materialization 消除

### 7.1 两条主线

1. Workload 分流：Decode Small-M 与 Prefill Large-M 使用不同 Kernel。
2. Materialization 消除：Norm、RoPE、Quant、Pack 能连续完成时不写中间 Tensor。

| 子路径 | 官方语义 | S5000 决策 |
| --- | --- | --- |
| Cache Store | 写 KV/Scale/RoPE | Decode/Prefill 分流 |
| Norm/RoPE | 独立或部分融合 | 消除 Materialization、融合 Cache |
| Compress Norm/RoPE | 前处理链 | Guarded Fusion |
| MHC | Pre/Post/Prenorm | Shape Split、Mixed Handoff、Prewarm |

### 7.2 FlashMLA Cache Store 双路径

Cache 物理布局：

```text
NoPE payload : 448 FP8 bytes/token
RoPE payload : 64 BF16 = 128 bytes/token
Scale        : 7 个 UE8M0/byte groups
Index        : INT32 physical slot
```

Decode 与 Prefill：

| | Decode | Prefill |
| --- | --- | --- |
| Token 数 | 1～几十 | 几千～几万 |
| 瓶颈 | Launch/尾延迟 | HBM 带宽 |
| 策略 | Block-per-token | 多 Token/CTA、宽写回 |

Dispatch：

```text
Decode/small rows
→ decode_x4 / decode_x4_fp32 / vec2

Prefill/T>=128
→ prefill_subwarp16 / prefill_tile_parallel
```

Prefill 将 7 个 NoPE Quant Tile 分成连续 `4+3`，由多个 Half-Warp 完成 Amax、Scale Byte、FP8 Pack 和 RoPE Vector Store。`full_tiles` 只在 `T % tokens_per_cta == 0` 时启用，尾块保留 Bounds Guard。

`T>=128` 没有充分 Sweep 证据，只能称经验分档；Decode 三变体和 `4+3` 的精确选择依据当前仍需回代码确认。

### 7.3 Fused Norm-RoPE-Cache

分解链路：

```text
x --RMSNorm--> norm_out
norm_out --RoPE--> q/k_rope
k_nope --Quant--> fp8 + scale
fp8/scale/rope --Pack--> Paged Cache
```

`norm_out`、`q/k_rope`、`fp8+scale` 都是短生命周期中间 Tensor，却要完整写回并读回 HBM。

融合 Kernel：

```text
读取 x/residual/weight/freq
→ RMSNorm
→ RoPE
→ NoPE amax/scale/FP8 cast
→ 直接按 Page Layout 写 NoPE/Scale/RoPE
```

重点不是把几个函数写进同一个文件，而是中间值保留在寄存器/线程块内部，最终直接写到 Cache 物理地址。

验证：

| 测试 | 作用 |
| --- | --- |
| 分解路径等价性 | Fused 与 Norm→RoPE→Quant→Store 数值一致 |
| Strided-Input | 证明 Kernel 直接支持 Stride，没有在 Wrapper 偷加 `.contiguous()` |

Strided 测试是“反作弊测试”：普通等价测试无法发现 Wrapper 通过全量 Copy 伪装成 No-Materialization。

### 7.4 Compress Guarded Fusion

```text
Hidden/Rope Dim/Dtype/Layout/Inplace 满足契约
→ Fused Norm/RoPE

不满足
→ 稳定分解实现
```

这是受 Guard 保护的融合，不是所有 Shape 强制共用一个 Kernel。

### 7.5 MHC 的 S5000 适配

官方已有 MHC 数学、DeepGEMM、Fused Norm 和 HC Head 基线。本地补齐：

```text
MHC Prenorm Backend
Prefill/Decode Shape Split
Mixed Prefill Handoff
Token-Count Prewarm
HC Head/Cache MUSA Kernel
```

大 Prefill 可以走 Big-Fuse；Decode/Mixed 不能错误复用相同 Workspace 和 Shape 假设。Prewarm 提前编译常用 `(tokens,hidden,split)`，减少首次 JIT 抖动。

HC Head Decode 的小 M、大 K 使用 Split-K：

```text
Stage0：K 维分割并产生 Partial
Stage1：Reduce Partial
```

Split-K 是通用范式；MUSA Kernel 与经 Sweep 选择的 Split 配置属于本地实现/调优。

### 7.6 归属表

| 工作 | 分类 |
| --- | --- |
| Cache/Norm/RoPE/MHC 数学语义 | 官方已有 |
| DeepGEMM/Fused Norm/HC Head 基线 | 官方已有 |
| MUSA Backend、Dtype/Layout Guard | MUSA 移植 |
| Cache Store Decode/Prefill Kernel 族 | 本地 MUSA Kernel |
| `T>=128`、Full-Tile 等 Shape Dispatch | S5000 调优 |
| Fused Norm-RoPE-Cache | 本地手写 MUSA Kernel，强证据 |
| Stride 直接支持 | S5000 No-Materialization 优化 |
| Compress Guarded Fusion | MUSA Fusion/Dispatch |
| MHC Prenorm、Shape Split、Prewarm | MUSA适配与S5000调优 |
| Mixed Handoff | 正确性/状态适配 |
| HC Head Split-K | 本地 Kernel；算法范式非原创 |

### 7.7 面试表述

> Cache、Norm、RoPE 和 MHC 数学来自上游。我针对 Decode 小 M 与 Prefill 大 M 重写并分流 MUSA Cache Store Kernel；同时实现 Fused Norm-RoPE-Cache，在一次调用中完成 Norm、旋转、量化并直接写 Paged Cache，消除三个中间 Tensor 的 HBM 往返，并用等价性和 Strided-Input 测试证明没有隐藏 Materialization。

Prefill Cache Store 约 `1.8×` 是项目口径；Norm/RoPE/MHC 没有统一实测倍数。

---

## 8. Decode、Paged Cache、MTP/EAGLE 与 Sampling：MUSA 闭环和 Small-M 专用路径

### 8.1 为什么 Decode 需要专用路径

Decode 每步 M 很小、KV 很长，瓶颈是：

```text
Kernel Launch
KV Read/Write
Page/Prefix-Tail Dispatch
Graph Replay
Host Synchronization
Sampling
```

Prefill 大 Tile 用于 Decode 会造成有效 Block 不足、Tail/Padding 增多、Shared Memory 和寄存器成为固定成本。

### 8.2 Unified Decode Attention 扩展

```text
Queue4      → 将多个小 Query/Group 排进可复用计划
Prefix-Tail → Full Tile 处理规则 Prefix，Tail Kernel 处理末段
Page32      → 将物理 Page32 解释成 Kernel Logical K/V View
Head256     → 调整 Head 方向 Block 划分
```

Queue4 控制流：

```text
Unified Wrapper
→ 选择 Queue Variant
→ 获取缓存 Plan/Workspace/Kernel
→ RunFullTile / RunTailTile / RunGroup / RunQueue
```

Page32 不是放宽一个 Assert，还涉及 Logical View、Page Offset、Split-Reduce 段数、Aligned Full-Tile 与 Graph Workspace。Page32 和 Head256 是正交维度，应分别测试。

### 8.3 Small-M Cache 与 HC-Head

Decode Cache Kernel：

```text
decode_x4 / decode_x4_fp32 / vec2
```

以 Token 为主要并行单位，一趟完成：

```text
amax/scale → FP8 Quant → NoPE/RoPE Pack → Paged Store
```

HC Head 使用 Split-K 提升小 M、大 K 的 Block 数，但会增加 Partial Workspace 和第二次 Reduce Launch，所以 Split 数需要 Sweep。

### 8.4 Paged Cache 必须 Prefill/Decode 成对闭环

`is_paged=True` 会改变 Metadata：

```text
非 Paged/Page4：extra_data[batch] 多列
Paged：extra_data[token] 单列，按 Physical Block 解释
```

如果只修改 Prefill 写入而 Decode 仍按旧列数读取，会产生错误地址解释。

Paged C4 Decode：

```text
block_id  = indices[token]
write_pos = (seq_len + 3) % 4
写当前 kv_score
→ 遍历有效 slot
→ exp-weighted numerator/denominator
```

### 8.5 MTP/EAGLE Graph Metadata

Graph 固定 Tensor 地址，但每轮 Replay 的值会变：

```text
draft token / seq_len / cache location
accept index / expert remap / extend metadata
```

Replay 前必须更新 Capture Buffer 的内容，不能替换对象或继续使用上一轮值。

Accept Index 防护：

```text
flatten → sanitize → safe index select
→ evict 正确 slot → sanitize draft count
```

它既防止 Token 错误，也避免 Cache Allocator 重复释放 Slot。Expert Remap 和 Extend Metadata 也必须保证 Draft/Target/Replay 使用同一映射。

### 8.6 Sampling

```text
Greedy
Seeded Multinomial
No-Seed Multinomial
```

MUSA No-Seed 路径从有问题的 `torch.multinomial` Fallback 切换到可用的 `sgl_kernel` Backend；Seeded 路径仍需保留 Deterministic 契约，不能因为 No-Seed Backend 更快而统一替换。

### 8.7 归属表

| 工作 | 分类 |
| --- | --- |
| Decode Attention、Paged Cache、MTP/EAGLE、Sampling 语义 | 官方已有 |
| MUSA Unified Attention 接入 | MUSA 移植 |
| Queue4/Prefix-Tail/Page32/Head256 | MUSA Shape/Layout 扩展 |
| Decode `x4/x4_fp32/vec2` | 本地 Small-M Kernel |
| HC Head Split-K | 本地 MUSA Kernel，范式非原创 |
| Paged Prefill/Decode 配套 | 平台正确性闭环 |
| Paged C4 Decode Kernel | 本地 MUSA Kernel |
| Draft Graph/D2H `seq_len` | MUSA Graph 适配 |
| Accept Index/Expert Remap 防护 | 系统正确性修复 |
| No-Seed Sampling Backend 替换 | 平台稳定性适配 |

### 8.8 面试表述

> 官方提供 Unified Decode Attention、Paged Cache、MTP/EAGLE 和 Sampling 框架。我在 MUSA 上补齐执行闭环，并针对 S5000 Decode 小 M、长 KV 实现 Cache Store 和 HC-Head 专用 Kernel；同时补齐 Paged C4 Decode、Graph Replay Metadata、Accept Index/Expert Remap 防护和 MUSA No-Seed Sampling Backend。

TPOT `7.3→6.6 ms/token` 是项目口径，硬件、模型、并发和 MTP 配置待补。

---

## 9. Online C128、PP/PD 与 HiSparse：MUSA 状态闭环

### 9.1 总体链路

```text
Prefill Compress State
→ Online C128
→ PD Transfer / PP Stage
→ Decode/MTP Consume
→ Cache Lifecycle Update
→ 必要时 HiSparse Host Offload
```

这一章主要解决状态在不同设备、阶段、Page 空间和 Graph Buffer 之间是否仍指向正确位置。

### 9.2 Online C128 × MTP Deferred Commit

MTP Draft 中部分 Token 会被拒绝；C128 是有损持久状态，不能先写后回滚。

```text
Verify 前 Reset 临时 Cache
→ Forward 每层只缓存候选 kv_score
→ Verify 得到 Accepted Rows
→ 按请求、按层、按接受顺序回放
→ 更新持久 C128 State
→ 清空临时 Cache
```

Accepted 信息必须是 Flattened `kv_score` 的全局 Row Index，不能只传数量或请求内局部编号。

提交地址链：

```text
position = floor((seq_len-1)/128)×128
→ raw_loc = req_to_token[req,position]
→ swa_loc = translate_full_to_swa(raw_loc)
→ state_loc = swa_page×ring_size + swa_loc%ring_size
→ state_index = floor(state_loc/128)
```

跨 127→128 边界必须恰好执行一次 Compress/Store，判断必须基于最终 Accepted Token，放在 Commit 阶段。

这是本地系统设计，但编排的是已有 Triton Kernel，不应计作新 C128 Kernel。

### 9.3 PP/PD 的五层地址坐标

| 坐标 | 含义 |
| --- | --- |
| Full Token Location | Token 在 Full KV Pool 的物理位置 |
| Full Global Page | Full Location 除以 Full KV Page Size |
| SWA Token/Page | Token 在 SWA Pool 的位置 |
| Request-Relative Page | 相对于请求首 Page 的位置 |
| CP-Rank Page Subset | 当前 CP Rank 负责的 Global Page 子集 |

Full KV `page_size` 与 SWA `swa_page_size` 可能不同，不能混用。

PD 两端物理 Page 不同，因此通过 Request-Relative Page 对齐：

```text
Prefill physical page 105, first_page 100 → relative 5
Decode first_page 500                 → physical 505
```

Prefill 侧构造 `(full_global_page,swa_pool_page)` Pair，按 Pair 去重、筛选 CP Rank、转换 Relative Page 并排序；Decode 侧建立 `relative_page→destination_state_index`，按 Prefill 顺序重排目标。

不能分别对两列 `unique`，否则会破坏 Full/SWA 对应关系。

### 9.4 PP Layer Pointer Layout

压缩 MLA Pointer Table 按类型分段，不是简单逐层排列。

Compact Layout：

```text
[C4 Attention States]
[C4 Indexer States]
[C128 States]

length = 2*c4_full + c128_full
```

PP Stage 切片时要分别统计 Stage 前后的 C4/C128 层数，再在三个分段内切片。

带 SWA 的布局：

```text
[SWA per layer]
[Compress State for ratio != 0]
[Indexer State for ratio == 4]
```

后两段需要分别按 `count(ratio!=0)` 和 `count(ratio==4)` 计算 Offset。最终必须严格校验 Source/Destination Pointer 数相等。

### 9.5 Layer Transfer 可见性

```text
Layer L Transfer 完成
    happens-before
Attention/Indexer/Graph 读取 Layer L Buffer
```

`wait_layer_transfer(layer_id)` 等待的是 `layer_id-stage_start_layer`。它不是全局 Barrier，只保证指定层可见。

验证需要故意让后层先完成、前层延迟，检查读取 Layer L 不会早读，也不会等待错误层。PP Graph Proxy Tensor 在 Capture 后只能更新内容，不能替换对象。

### 9.6 HiSparse Host Offload

官方 CUDA Helper 无法处理 MUSA Tensor，因此实现固定 HiSparse Packed Layout 的 TileLang Offload：

```text
GPU/CPU Pointer Table + Indices
→ MUSA Router/Guard
→ TileLang Kernel
→ Host Destination
```

固定布局：

```text
page_size       = 64
GPU page stride = 4680 uint64
value width     = 72 uint64
scale base      = 4608 uint64
CPU item width  = 73 uint64
```

每个 Item/Layer 复制 72 个 Value 和 1 个 Scale。

Guard 检查 Device、`uint64` Pointer、`int64` Index、Shape、Contiguous；空输入 No-Op。裸指针无法证明背后 Layout，调用方必须保证 Page64/Stride 契约。Kernel Launch 完成也不代表 CPU 立即可见，Host 消费前仍需 Stream/Event 同步。

### 9.7 归属表

| 工作 | 分类 |
| --- | --- |
| Online C128、PP/PD、HiSparse、MTP 系统目标 | 当前官方已有 |
| MUSA Memory Pool/Connection/Graph Proxy | 平台移植与状态适配 |
| Full/SWA/Relative/CP 坐标转换 | PD/CP正确性适配 |
| PP Pointer 分段切片、Transfer Wait | PP正确性适配 |
| C128 Deferred Commit | 本地系统设计 |
| 127→128 Commit 边界 | 本地状态机制 |
| HiSparse TileLang Offload | 本地 MUSA Kernel |
| HiSparse Router/Guard | MUSA平台适配 |

### 9.8 面试表述

> 官方提供 Online C128、PP/PD 和 HiSparse 的系统语义。我在 MUSA 上补齐状态闭环：为 MTP Verify 实现 Deferred Commit，避免 Rejected Draft 污染 C128 Ring；梳理 Full KV、SWA、Relative Page 和 CP Rank 地址转换；按压缩类型切分 PP Pointer Table并建立逐层可见性；同时实现固定 HiSparse Packed Layout 的 MUSA Host-Offload Kernel。

这章没有可信的统一性能倍数，价值应表述为 32K/128K 可承载性、显存、State Transfer、Graph Replay 和长稳正确性。

---

## 10. Runtime Hardening：证明 S5000 Fast Path 真实命中

### 10.1 这一章解决什么问题

```text
写了快 Kernel
→ Wrapper Guard 没命中
→ 静默走 Torch/CUDA Fallback
→ 数值正确但性能测试无效
```

Runtime Hardening 的目标不是再写一个快 Kernel，而是证明 Serving 真正执行了预期 MUSA Fast Path。

### 10.2 Production Fail-Closed

Debug：

```text
Fast Path Miss → 允许 Reference/Torch → 记录 Fallback Reason
```

Production/Benchmark：

```text
Fast Path Miss → 抛异常
```

必须禁止：

- MUSA Tensor 进入 CUDA-only Helper。
- Graph Capture 中 Torch Fallback。
- Compress/TopK Fast Path Miss 后静默慢跑。
- 不支持 Dtype/Layout/Page Size 时隐式转换。

### 10.3 JIT Cache Key

原 C4 Decode Kernel 接收：

```text
head_dim / extra_data_cols / threads
```

但旧名字没有 `extra_data_cols`：

```text
dsv4_c4_decode_page4_t{threads}
```

`extra_data_cols=1` 与 `4` 的 Tensor Shape 和地址公式不同，却可能命中同一 Artifact。

修复：

```python
f"dsv4_c4_decode_page{extra_data_cols}_t{threads}"
```

测试必须覆盖：

```text
page1 → page4 → page1
page4 → page1 → page4
首次编译/热 Cache
多进程共享 Cache
```

通用规则：任何改变 Rank、Shape、Stride、常量循环或地址公式的参数都必须进入 JIT Key。

### 10.4 MCCL Device Binding

创建 Process Group 时绑定：

```python
if backend == "mccl" and local_rank >= 0:
    device_id = torch.device("musa", local_rank)
```

必须在 `init_process_group` 前传入；之后 `set_device` 无法回溯修复已经创建的 WORLD Group。非 MCCL 和 `local_rank<0` 保持原语义。

### 10.5 Ops Facade Host Dispatch

旧方案为支持 Monkeypatch，每次调用都扫描并同步 Facade/Impl，给高频 Decode 增加 Python 开销。

新方案：

```text
setattr(facade,name,value)
→ 赋值时同步到 target/domain
→ 正常 Warm Call 不再逐次扫描
```

因此需要分别报告：

```text
Direct Op
Facade Warm Call
Host-Inclusive Latency
```

这优化的是 Host Dispatch，不是 Kernel Latency。

### 10.6 No-Copy 契约

Source 与 Destination 要区别处理：

```text
Row-Strided Source
→ Kernel 显式支持 Stride时可以 No-Copy

Packed Cache Destination 非连续
→ 地址公式不成立，必须 Fail-Closed
```

不能偷偷 `.contiguous()`，因为它会增加 Allocation/Copy、破坏 Graph，并让 Benchmark 混入隐藏 Materialization。

### 10.7 Graph Preflight 与 Replay

Capture 前固定：

```text
MUSA Backend
已预编译的 JIT Specialization
Dtype/Shape/Stride/Page Layout
Output/Workspace/Proxy Tensor 地址
无 Torch Fallback
已初始化 MCCL Group
当前 C128/MTP State
```

Capture 中禁止动态编译、Host Value Read、隐式 Allocation、Unsupported Sync 和 Fallback。

Replay 正确性至少使用两组不同输入复用同一 Graph，并分别对比 Eager，证明 `seq_len/cache_loc/accept_index/request mapping` 没被固化成旧值。

Failure Injection 应改变 Bucket、Stride、Page Layout 或关闭 Kernel，确认在 Capture 前拒绝，而不是 Replay 旧路径。

### 10.8 可观测性

统一记录：

```text
Backend/Op/Specialization Key
Device/Dtype/Shape/Stride/Page Layout
Fast-Path Hit/Fallback Reason
JIT Cold/Warm/Compile Count
Capture ID/Bucket/Replay Count
Kernel-Only/Host-Inclusive Latency
MCCL Rank/Local-Rank/Current-Device
```

验收目标：

```text
Unexpected Fallback = 0
JIT Cache Collision = 0
Graph Capture/Replay 正确
多卡 Rank-Device 一致
Host Dispatch 未抵消 Small Kernel 收益
```

### 10.9 归属表

| 工作 | 分类 |
| --- | --- |
| Router/JIT/Fallback/Graph/Distributed 框架 | 官方已有 |
| MUSA JIT/FFI/Tensor ABI | 平台移植 |
| Production Fail-Closed/Graph-Safe Fallback | MUSA稳定性适配 |
| `extra_data_cols` JIT Key 修复 | 本地正确性修复 |
| MCCL `device_id` | MUSA多卡适配 |
| Forwarding `__setattr__` 同步 | 本地Runtime重构 |
| Source Stride No-Copy/Cache Fail-Closed | MUSA Kernel契约与正确性防护 |
| Fast-Path观测和失败注入 | 验证体系 |

### 10.10 简历边界

Runtime Hardening 主要是 ABI、Guard、Fallback、Device Binding、JIT Key 和 Facade 重构，证据等级以 C 为主，而且部分 No-Copy 内容与第7章重复。因此不适合作为核心性能故事，最多合并成一句：

> 完善 MUSA Runtime Fast-Path 保障：生产环境禁止 Silent Fallback，将 Page Layout 纳入 JIT Cache Key，补齐 MCCL Device Binding 与 Graph Preflight，并优化高频 Decode Ops Facade 的 Host Dispatch。

不能写“Runtime Hardening 将模型加速 X%”。

---

## 11. 统一性能数字口径

| 数字 | 正确说法 |
| --- | --- |
| MoE 算子约 `1.6×` | 项目口径，配置待补 |
| Prefill 端到端约 `+8%` | 项目口径，配置待补 |
| Quant/Compress `1.8～2×` | 两个算子混合口径，不可当单项结果 |
| Sparse 前处理/Metadata 约 `2×` | 局部链路，不等于 Sparse Attention/TTFT |
| Prefill Cache Store 约 `1.8×` | 只覆盖 Prefill Cache Store |
| TPOT `7.3→6.6 ms/token` | 项目口径，硬件/模型/并发/MTP配置待补 |
| Quant-Scatter 约省 `311 MB/层` | 指定 Shape 下的流量推导，不是实测 |

统一表述：

> 我可以说明项目已有口径和完整测量方法；缺少硬件、Shape、Baseline Commit 和原始日志时，我不会把它说成本人复测结果。

---

## 12. 第 3～10 章闭卷自测

1. Expert-Contiguous、`src2dst`、`m_indices` 分别是谁提供的，作用是什么？
2. Fused Quant-Scatter 为什么必须以源 Token 行为并行方向？
3. Fixed-Bucket 为什么有利于 Graph，Overflow 为什么不能截断 Route？
4. `[2048,64]→[2048,512]` 与 `[16,262208]→[16,512]` 为什么不能共用一个 TopK512 Kernel？
5. Candidate Overflow 为什么与 Graph Capture 冲突？Exact/Candidate 双跑如何解决？
6. Half-Warp Quant 的 `16 lanes×8 values` 是原创还是 MUSA 重实现？
7. 为什么 `total_groups>=16K` 时要主动阻断 Generic Cast？
8. Seq-Pack 中 `-1` 与 `combined_len` 分别是什么？
9. Pack8 为什么要求每个 Request 的 `extend_seq_len` 分别被 8 整除？
10. Fused Norm-RoPE-Cache 到底消除了哪些中间 Tensor？
11. Strided-Input 测试如何证明没有隐藏 `.contiguous()`？
12. 为什么 Cache Store 必须分 Decode/Prefill 两套 Kernel？
13. Page32 为什么不是简单修改一个 Assert？
14. `is_paged=True` 为什么要求 Prefill 和 Decode 成对修改？
15. MTP Graph Replay 中哪些 Metadata 每轮需要更新？
16. Rejected Draft 为什么不能先写 C128 再回滚？
17. Full KV Page、SWA Page、Request-Relative Page 为什么不能混用？
18. PP Compressed Pointer Table 为什么不能直接 `[start:end]`？
19. HiSparse Offload Kernel 为什么不是通用 GPU-to-CPU Copy？
20. 如何证明 Benchmark 命中 Fast Path，而不是 Silent Fallback？
21. JIT Key 缺少 `extra_data_cols` 为什么可能产生顺序相关的错误？
22. Kernel-Only 与 Host-Inclusive Latency 为什么都要测？

完成标准：每题都能回答“官方基线、S5000问题、具体机制、验证方式和边界”，而不是只背优化名称。
