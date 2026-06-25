# DeepSeek V4 FP8 Prefill 优化梳理

> 代码库：`/home/eilan/worspace/sglang`  
> 重点分支：`origin/deepseek_v4_kernels_opt`、`origin/perf/yunqiao.jiang/dsv4_prefill`  
> 生成日期：2026-06-22

## 0. GPU 硬件信息

> 说明：下表用于说明本文档讨论的 MUSA / DeepSeek V4 FP8 prefill 优化所处的硬件背景。三列分别为 H20、H200、S5000。第三列 memory type 原始资料写作 `DDR6`，通常可能对应 `GDDR6`，这里先按原始信息保留。

| 指标 | H20 | H200 | S5000 |
| --- | ---: | ---: | ---: |
| FP64（TFLOPS） | 1 | 30 | 15.6 |
| FP32（TFLOPS） | 44 | 60 | 31.3 |
| TF32 TensorCore（TFLOPS） | 74 | 835 | 250.3 |
| FP16 TensorCore（TFLOPS） | 148 | 1671 | 500.7 |
| BF16 TensorCore（TFLOPS） | 148 | 1671 | 500.7 |
| FP8 TensorCore（TFLOPS） | 296 | 3341 | 1001.4 |
| INT8 TensorCore（TFLOPS） | 296 | 3341 | 1001.4 |
| GPU Memory | 141 GB | 141 GB | 80 GB |
| GPU Memory Bandwidth | 4.8 TB/s | 4.8 TB/s | 1.6 TB/s |
| Memory Type | HBM3e | HBM3e | DDR6 |
| Interconnect | NVLink 900 GB/s；PCIe Gen5 128 GB/s | NVLink 900 GB/s；PCIe Gen5 128 GB/s | 784 GB/s |

从优化角度看，这类硬件配置对 DSV4 FP8 prefill 的影响主要体现在三点：

1. **FP8 TensorCore 峰值很高**：MoE DeepGEMM、WO_A、down projection 等 FP8 GEMM 路径具备较高理论上限，因此 prefill 中大量 token 的 FP8 计算有继续压榨的空间。
2. **HBM 带宽仍是关键约束**：FP8 quant、cache store、compress、norm/rope、sparse pack 等路径往往更偏 memory-bound，即使 TensorCore 峰值很高，也可能被读写 activation、scale、KV cache、metadata 的带宽限制住。
3. **大 prefill 更容易暴露 kernel 调度问题**：decode tiny-M 更关注单 token latency，而 prefill 大 M / 大 token 场景会放大 quant、cache store、seq-pack、compress 等 kernel 的 launch、访存合并、并行粒度和 fallback 选择问题，因此本文档重点关注 guarded dispatch 和 MUSA 专用 prefill fast path。

## 1. 总览

目前与 `deepseekv4-fp8` 在 prefill 阶段性能优化最相关的两个远端分支是：

| 分支 | 定位 | 说明 |
| --- | --- | --- |
| `origin/deepseek_v4_kernels_opt` | DSV4 MUSA / FP8 kernel 优化底座 | 引入并清理 DeepSeek V4 MUSA 专用 kernel/ops，包括 cache store、MHC、FP8 quant、SwiGLU quant、topk、WO_A 等。 |
| `origin/perf/yunqiao.jiang/dsv4_prefill` | DSV4 prefill 专项优化分支 | 在 kernel 底座上继续加入 MoE DeepGEMM prefill、FlashMLA sparse prefill seq-pack、prefill memory-bound kernel、FlashMLA cache store prefill、sparse prefill pack8 等优化。 |

一句话总结：

- `deepseek_v4_kernels_opt` 主要解决“DSV4 FP8 在 MUSA 上有哪些专用高性能算子可用”。
- `perf/yunqiao.jiang/dsv4_prefill` 主要解决“prefill 大 M / 大 token 场景如何走更快的 MoE、attention、cache store、compress、quant 路径”。
- 本文档只保留 **MUSA / DSV4 分支新增或修改的实现**；纯粹继承自上游 SGLang 的通用 FP8 quant API、通用 Triton fallback、通用 scale layout 等不再展开。

## 2. 关键提交清单

### 2.1 `origin/deepseek_v4_kernels_opt`

| Commit | 标题 | 作用 |
| --- | --- | --- |
| `ef32b80f4` | `Port DeepSeek V4 MUSA optimized kernels` | 引入 DeepSeek V4 MUSA 专用 kernel/ops 框架，覆盖 cache、MHC、hc_head、模型接入等。 |
| `78be4b50a` | `Optimize MUSA FP8 quant paths with TileKernels` | 优化 FP8 per-token/group quant，prefill 大 shape 走 TileLang/TileKernels，decode 小 M 保持原路径。 |
| `3c6b17eae` | `Port DeepSeek V4 MUSA optimized kernels` | 大量补充 benchmark/test 与 compress、norm-rope、routing、topk、swiglu quant 等 kernel。 |
| `abe778217` | `Tune DeepSeek V4 MUSA quant dispatch` | 调整 FP8 quant dispatch 条件与 benchmark。 |
| `ea3e93c38` | `Optimize DeepSeek V4 MUSA MHC prenorm` | 优化 MHC pre-norm 路径。 |
| `392179e92` | `Optimize DeepSeek V4 MUSA indexer cache store` | 优化 indexer cache store。 |
| `4fabebf7f` | `[DeepSeekV4] feat: enable flash_mla prefill for large Q (up to 32K)` | 启用 large-Q FlashMLA prefill，支持最高 32K Q 的 prefill attention 路径。 |

### 2.2 `origin/perf/yunqiao.jiang/dsv4_prefill`

| Commit | 标题 | 作用 |
| --- | --- | --- |
| `cc5b01a55` | `Add DeepSeek V4 MUSA MoE DeepGEMM prefill path` | 新增 MoE DeepGEMM compact prefill 路径，面向 FP8 MoE 大 token prefill。 |
| `d5fe7707c` | `Add FlashMLA sparse prefill seq-pack path` | 新增 FlashMLA sparse prefill seq-pack 路径，构建 flat BF16 workspace + rebased indices。 |
| `f393eaa98` | `Optimize DSV4 MUSA prefill memory-bound kernels` | 优化 prefill 中 memory-bound 的 compress/cache/FP8 quant kernel。 |
| `7c992a2a8` | `Optimize DeepSeek V4 MUSA FlashMLA cache store prefill` | 优化 FlashMLA cache store prefill，新增 tile-parallel/subwarp16 等路径。 |
| `6a57c5b25` | `Optimize MUSA compress fused norm rope prefill with guarded dispatch` | prefill compress + norm + rope 融合优化，并加 guarded dispatch。 |
| `1ce2e4aae` | `Enable guarded DSV4 sparse prefill pack8` | guarded sparse prefill pack8 路径。 |
| `b9b202c28` | `Fix DSV4 sparse prefill under CP round-robin` | 修复 sparse prefill 在 CP round-robin split 下的 metadata / local q 问题。 |
| `844133222` | `Remove DSV4 sparse prefill raw-index fallback` | 移除 sparse prefill raw-index fallback，统一使用修正后的 packed/index path。 |

## 3. 具体优化细节

### 3.1 FP8 prefill quant 路径优化

相关提交：

- `78be4b50a` — `Optimize MUSA FP8 quant paths with TileKernels`
- `de830f875` — `Optimize DeepSeek V4 MUSA WO_A and quant paths`
- `f393eaa98` — `Optimize DSV4 MUSA prefill memory-bound kernels`

关键文件：

- `python/sglang/srt/layers/quantization/fp8_kernel.py`

本节只记录 MUSA / DSV4 分支新增或修改的 fast path；不再展开 SGLang 原有的 `sglang_per_token_group_quant_fp8`、通用 `_per_token_group_quant_8bit_raw`、通用 Triton quant kernel、通用 scale layout 创建等官方基础实现。

#### 3.1.1 背景：为什么 FP8 quant 会成为 prefill 问题

DeepSeek V4 FP8 在 MUSA 上运行时，很多后续计算需要把 activation 从 bf16/fp16/fp32 转成 FP8，并同时生成 per-token / per-group scale：

```text
x:   bf16/fp16/fp32 activation
x_q: fp8 activation
x_s: per-token/group scale
```

对于 prefill 阶段，`x` 通常对应大量 token 和较大的 hidden 维度，`x.numel()` 与 group 数都很大。此时 quant 本身不再只是一个很小的辅助操作，而会变成明显的 memory-bound / bandwidth-bound 开销，尤其会影响 MoE DeepGEMM prefill 中间 activation quant、WO_A FP8 GEMM 输入 quant、以及部分 cache/attention 前处理路径。

因此这轮优化的核心目标是：

> 在 MUSA 上为 DSV4 FP8 大 prefill shape 增加专用 quant fast path，同时避免 decode 小 M / tiny-M TPOT 回退。

#### 3.1.2 优化前 profiling / 现象判断

仓库中没有提交完整 profiling 日志，所以这里不写具体 ms 数值；但从代码注释、dispatch 条件和 benchmark 设计可以确认当时观察到的几个现象：

1. **prefill 大 shape 下 quant 是 memory-bound 热点**

   `f393eaa98` 的标题就是 `Optimize DSV4 MUSA prefill memory-bound kernels`，其中 `fp8_kernel.py` 是核心改动文件之一。这说明 FP8 quant 被归入 prefill memory-bound 优化对象。

2. **decode tiny-M 与 prefill large-M 需要分开处理**

   代码注释明确写到：

   ```text
   TileKernels per-token quant is faster once there is enough row-level work,
   while the SGLang JIT subwarp scheduler remains better for decode tiny-M.
   ```

   因此不能简单让 decode 和 prefill 共用同一条大 shape kernel。

3. **generic TileKernels cast 在 large CP prefill shape 上反而慢**

   代码中直接记录了 profiling 结论：

   ```text
   The generic TileKernels cast path is slower than Triton on large CP prefill shapes.
   ```

   所以后续 dispatch 特意避免 large prefill miss 后落到 generic TileKernels cast。

#### 3.1.3 优化策略一：用 16K groups 区分 prefill 大 shape

新增阈值：

```python
_MUSA_PREFILL_FP8_QUANT_MIN_GROUPS = 16 * 1024
```

判断方式：

```python
total_groups = x.numel() // group_size
```

策略：

- `total_groups >= 16K`：认为是 prefill 大 shape，尝试 MUSA prefill 专用 fast path。
- `total_groups < 16K`：认为更接近 decode/small-M，保留原有小 M 行为，避免 TPOT 回退。

这一步解决的是 **dispatch 粒度问题**：优化只作用在大 prefill，不污染 decode。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/quantization/fp8_kernel.py:121`

#### 3.1.4 优化策略二：MUSA Triton multi-group quant fallback

新增：

- `_musa_prefill_fp8_quant_launch_config`
- `_per_token_group_quant_8bit_multi_group`
- `_try_musa_prefill_per_token_group_quant_8bit`

这条路径的目的，是给大 prefill shape 提供一个 MUSA 专用的 Triton multi-group quant fallback。

核心设计：

- `group_size == 128`。
- 一次 CTA 处理多个 group。
- 根据 `rows` 和 `hidden` 选择 `groups_per_cta` / `num_warps`：

```python
def _musa_prefill_fp8_quant_launch_config(rows: int, hidden: int) -> Tuple[int, int]:
    if hidden <= 2048:
        return 8, 4
    if rows <= 4096:
        return 4, 2
    return 8, 2
```

启用条件包括：

- MUSA device。
- `group_size == 128`。
- 普通 fp32 scale layout。
- 不走 fused_silu_and_mul。
- `masked_m is None`。
- `x_s.dtype == torch.float32`。
- total groups >= 16K。

这一步解决的是 **large prefill 通用 fallback 不够适配 MUSA shape** 的问题。

#### 3.1.5 优化策略三：TileLang MUSA half-warp/group FP8 quant kernel

新增：

- `_tilelang_musa_per_token_group_quant_fp8_kernel`
- `_try_tilelang_musa_per_token_group_quant_fp8`

这是这部分最核心的专用 kernel。

核心参数：

```python
group_size = 128
values_per_lane = 8
groups_per_warp = 2
```

设计含义：

```text
1 group = 128 elements
1 half-warp = 16 lanes
每 lane 处理 8 elements
16 lanes * 8 = 128 elements
```

也就是：

```text
一个 half-warp 处理一个 quant group
一个 warp 处理两个 quant groups
```

kernel 内部流程：

1. 每个 lane 读取自己负责的 8 个值。
2. 计算本 lane 的 local amax。
3. 用 `shfl_xor` 做 half-warp 内 amax reduce。
4. 计算：

   ```python
   scale = max(local_amax, eps) / 448.0
   inv_scale = 1.0 / scale
   ```

5. 每 4 个值 pack 成一个 FP8x4 `uint32`：

   ```cpp
   tl_sglang_pack_fp8x4_e4m3_u32(...)
   ```

6. 每 lane 通过两次 `stg32` 写回 8 个 FP8 值。
7. 只让 `sublane == 0` 写 scale，避免重复 store。

这一步主要优化：

- per-group amax reduce 开销。
- FP8 小粒度写回开销。
- 大 prefill shape 下的带宽利用率。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/quantization/fp8_kernel.py:141`

#### 3.1.6 优化策略四：TileKernels per-token cast 的 MUSA 接入与保护

新增 `_try_tilekernels_per_token_cast_musa`，用于在 MUSA 上接入：

```python
tile_kernels.quant.per_token_cast(...)
```

这里需要区分：

- `tile_kernels.quant.per_token_cast` 是外部 TileKernels 能力。
- SGLang 侧的 MUSA dispatch wrapper、guard、shape 检查、fallback 策略是本分支新增。

该路径只在满足条件时尝试：

- MUSA。
- `SGLANG_OPT_USE_TILEKERNELS_FP8_QUANT` 打开。
- 2D tensor。
- `group_size == 128`。
- 普通 scale layout。
- 输入 dtype 为 bf16/fp16/fp32。

但是，对于 large prefill shape，代码特意禁止它作为 miss 后 fallback：

```python
if x.numel() // group_size >= _MUSA_PREFILL_FP8_QUANT_MIN_GROUPS:
    return None
```

原因是 profiling 发现 generic TileKernels cast 在 large CP prefill shapes 上比 Triton 慢。

#### 3.1.7 最终 dispatch 顺序

综合起来，大 prefill shape 的优先级是：

```text
1. TileLang MUSA FP8 prefill quant
   - half-warp/group
   - fp8x4 pack
   - stg32 writeback

2. MUSA Triton multi-group quant
   - groups_per_cta / num_warps shape heuristic

3. 禁止 large prefill 落到 generic TileKernels cast
   - 避免 large CP prefill 走已知慢路径

4. 再进入剩余 fallback
```

small-M / decode 则不会被 16K groups 以上的 prefill fast path 捕获，从而保持原来的 decode 行为。

#### 3.1.8 后续现象与效果

从代码和 benchmark 设计可以确认这轮优化达到了以下结构性效果：

1. **prefill 大 shape 有了 MUSA 专用 fast path**

   大 prefill 不再只能依赖通用 SGLang quant 或 generic TileKernels cast，而是优先走 TileLang half-warp/group kernel 或 MUSA Triton multi-group kernel。

2. **decode 小 M 被保护**

   通过 `16K groups` 阈值，decode tiny-M 不会误走 prefill 大 kernel，避免 TPOT 回退。

3. **large CP prefill 避免已知慢 fallback**

   large prefill miss 时不会落到 generic TileKernels cast，规避 profiling 中发现的慢路径。

4. **写回路径更适合 FP8 bandwidth**

   TileLang kernel 使用 FP8x4 pack + `stg32` 写回，减少 byte-level store，提升大 tensor quant 的写带宽利用。

5. **服务 MoE DeepGEMM prefill / WO_A FP8 路径**

   FP8 activation quant 的优化会直接降低 MoE prefill 中 `SwiGLU + quant -> down_input_fp8` 以及部分 WO_A / FP8 GEMM 输入准备的开销。

需要注意：仓库中没有提交实测的 median ms / E2E TPS 日志，所以本文档不虚构具体性能数值。实际量化效果需要在 MUSA 环境中运行相关 benchmark 补充。

#### 3.1.9 验收方式

这部分优化的验收分为三层：

1. **dispatch correctness**

   验证不同输入 shape / dtype / layout 是否进入预期路径：

   - large prefill：应尝试 TileLang MUSA 或 MUSA Triton multi-group。
   - decode/small-M：不应进入 large prefill fast path。
   - 特殊 scale layout：不应误入只支持普通 fp32 scale 的 TileLang path。

2. **correctness**

   验证输出：

   - `x_q` shape 与 dtype 正确。
   - `x_s` shape 与 dtype 正确。
   - quant/dequant 后误差在可接受范围内。
   - fallback path 与 fast path 结果一致或近似一致。

3. **performance / regression**

   运行 MUSA benchmark，重点看：

   - large prefill quant latency 是否下降。
   - MoE prefill / WO_A 相关 stage 是否受益。
   - decode tiny-M TPOT 是否没有回退。
   - large CP prefill 是否没有掉到 generic TileKernels 慢路径。

可参考运行：

```bash
SGLANG_RUN_DEEPSEEK_V4_MUSA_OPERATOR_BENCH=1 \
python -m pytest python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py -s
```

以及 MoE prefill benchmark：

```bash
SGLANG_RUN_DEEPSEEK_V4_MUSA_BENCHMARK=1 \
SGLANG_DSV4_MUSA_MOE_EXPERIMENTAL=1 \
python -m pytest python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_moe_prefill_deepgemm_benchmarks.py -s
```

验收结论应以 MUSA 实机 benchmark 输出为准；当前仓库代码已经提供了输出字段和 acceptance benchmark 框架，但没有提交固定的实测结果日志。

### 3.2 MoE DeepGEMM compact prefill path

相关提交：

- `cc5b01a55` — `Add DeepSeek V4 MUSA MoE DeepGEMM prefill path`

关键文件：

- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`
- `python/sglang/srt/layers/deepseek_v4_musa/kernels/moe_prefill_kernels.py`
- `python/sglang/srt/layers/deepseek_v4_musa/ops/moe_prefill_ops.py`
- `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_moe_prefill_deepgemm_benchmarks.py`

一句话概括：

> 这个优化是在 MUSA 上为 DeepSeek V4 FP8 MoE prefill 大 token 场景新增一条 compact DeepGEMM 路径，把 MoE 的 route / compact、gate-up GEMM、SwiGLU + FP8 quant、down GEMM、reorder / combine 等步骤组织成更适合 prefill 大 M 的高吞吐路径。

#### 3.2.1 背景：MoE prefill 的瓶颈不是单个 GEMM

DeepSeek V4 的 MoE 层大致流程是：

```text
hidden_states
  -> router / topk 选择 expert
  -> 按 expert 重排 token
  -> gate/up projection
  -> SwiGLU 激活
  -> down projection
  -> 按原 token 顺序合并输出
```

decode 阶段通常每次处理 token 数较少，主要问题更偏 small-M GEMM、kernel launch overhead 和调度开销。

prefill 阶段则不同，特别是长上下文或大 batch 场景，token 数明显增大，MoE 的热点会从单纯 GEMM 扩散到整条 pipeline：

- routing / topk 后的数据重排。
- expert-contiguous compact。
- gate/up grouped GEMM。
- SwiGLU activation。
- SwiGLU 后的 FP8 quant 和 scale 写回。
- down grouped GEMM。
- 最后的 reorder / combine。

因此这类优化不能只看 GEMM kernel 本身，还需要同时压缩 MoE 前处理、激活、量化和后处理的中间开销。

#### 3.2.2 profiling 现象与优化动机

这条路径对应的典型 profiling 现象可以总结为三类。

1. **MoE prefill 中非 GEMM kernel 占比明显**

   在大 prefill shape 下，profile 中除了 gate/up 和 down GEMM 外，还会出现较多 routing、compact、scatter/gather、SwiGLU、quant、scale 写回、reorder、combine 等 kernel。

   如果这些步骤分散执行，会带来：

   - kernel launch 数量增加。
   - GEMM 之间出现空隙。
   - 小 kernel 调度成本放大。
   - 中间 tensor 在 global memory 中反复读写。

2. **bf16 intermediate 内存压力较大**

   FP8 MoE 中，gate/up GEMM 通常输出 bf16：

   ```text
   gateup_output: bf16
   ```

   然后需要做：

   ```text
   down_input = SwiGLU(gateup_output)
   down_input_fp8, down_input_scale = quant(down_input)
   ```

   如果 split 执行，流程会变成：

   ```text
   gate/up GEMM
     -> 写 bf16 gateup_output
     -> 读 bf16 gateup_output
     -> SwiGLU
     -> 写 bf16 down_input
     -> 读 bf16 down_input
     -> FP8 quant
     -> 写 FP8 down_input + scale
     -> down GEMM
   ```

   其中 `down_input` 是 `tokens * intermediate_size` 级别的大 tensor。对于 prefill 大 M，这部分 bf16 materialize 和二次读取会造成明显 bandwidth 压力。

3. **prefill 大 M 与 decode 小 M 的最优路径不同**

   compact / reorder / fixed bucket / grouped GEMM metadata 都不是免费的。小 M decode 如果强行走 compact prefill path，可能因为额外前后处理开销导致 TPOT 回退。

   因此该优化只针对大 token prefill，通过明确阈值和环境变量保护 decode 路径。

#### 3.2.3 启用条件与保护策略

`_should_use_dsv4_musa_compact_prefill_deepgemm` 会限制该路径只在 MUSA + FP8 + 大 prefill 场景启用。

关键条件包括：

- MUSA device。
- `quant_info.use_fp8 == True`。
- `SGLANG_DSV4_MUSA_MOE_EXPERIMENTAL` 打开。
- `hidden_states` 是二维 tensor。
- token 数大于等于 `_DSV4_MUSA_COMPACT_MOE_MIN_TOKENS = 8192`。
- `topk_ids` 非空。
- 能够选到 fixed bucket rows。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:149`

相关阈值：

```python
_DSV4_MUSA_COMPACT_MOE_MIN_TOKENS = 8192
_TILEKERNELS_SWIGLU_QUANT_MIN_ROWS = 1024
```

这两个阈值的作用是：

- `8192 tokens`：只让大 prefill 进入 compact DeepGEMM path，避免 decode / small extend 误入。
- `1024 rows`：只在 prefill-sized M 下使用 fused SwiGLU + quant，小 M 仍保留原 split path。

代码注释中也明确了这一点：

> TileKernels fused SwiGLU+quant wins on prefill-sized M; decode small-M stays on the existing split path until a subwarp-per-group TileKernels kernel exists.

也就是说：

- prefill 大 M：走 TileKernels fused SwiGLU + FP8 quant。
- decode 小 M：保留原 split path，避免小 M 性能劣化。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:62`

#### 3.2.4 compact layout：把 token 整理成 expert-contiguous

MoE 中每个 token 会被 route 到 top-k 个 expert。原始 token layout 通常按 request / sequence 排列，但 grouped GEMM 更希望同一个 expert 的 token 连续放置。

原始 layout 可以理解为：

```text
[token0 -> expert3]
[token1 -> expert7]
[token2 -> expert3]
[token3 -> expert1]
...
```

compact 后变成：

```text
expert1: token3, ...
expert3: token0, token2, ...
expert7: token1, ...
```

这样 DeepGEMM grouped GEMM 可以按 expert 连续处理，减少离散访问，并提高大 M 下的 GEMM 吞吐。

compact path 需要维护一组 metadata / buffer，例如：

```text
src2dst
compact_input
compact_scale
m_indices
all_tokens
overflow_flag
padded_valid_rows
allocated_rows
worst_case_rows
```

它们分别用于描述：

- 原始 token 到 compact buffer 的映射。
- compact 后每个 expert 的 token 范围。
- grouped GEMM 每组的 M 大小。
- padding 后实际参与计算的 row 数。
- 静态 buffer 是否溢出。
- 当前分配 row 数与 worst-case row 数。

所以 compact prefill path 本质上是在做：

```text
原始 token layout
  -> expert-contiguous compact layout
  -> grouped GEMM
  -> output reorder / combine 回原始 token layout
```

#### 3.2.5 主计算 pipeline

compact prefill DeepGEMM path 的主流程是：

```text
hidden_states
  -> compact_input
  -> grouped_gemm_nt_f8f8bf16_contig       # gate/up GEMM
  -> gateup_output bf16
  -> silu_and_mul_contig_post_quant        # SwiGLU + FP8 quant
  -> down_input_fp8 + down_input_scale
  -> grouped_gemm_nt_f8f8bf16_contig       # down GEMM
  -> down_output bf16
  -> reorder / combine
```

##### 3.2.5.1 gate/up grouped GEMM

第一段 GEMM 是 gate/up projection，逻辑上类似：

```text
gateup_output = hidden_states @ w13
```

其中 `w13` 是 gate/up 合并权重。代码模式为：

```python
deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
    (hidden_states, hidden_states_scale),
    w13_weight_fp8,
    gateup_output,
    m_indices,
    recipe_a=recipe_a,
    recipe_b=recipe_b,
)
```

这里的计算特征是：

```text
FP8 activation x FP8 weight -> bf16 output
```

并且按 expert 分组执行。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:389`

##### 3.2.5.2 fused SwiGLU + FP8 quant

gate/up GEMM 后得到 bf16 的 `gateup_output`。SwiGLU 逻辑大致为：

```text
down_input = silu(gate_part) * up_part
```

但 down GEMM 希望输入是 FP8，因此还需要：

```text
down_input_fp8, down_input_scale = quant(down_input)
```

该优化把 SwiGLU 和 quant 合并为 fused post-quant：

```python
silu_and_mul_contig_post_quant(
    input=gateup_output,
    output=down_input_fp8,
    output_scale=down_input_scale,
    quant_group_size=scale_block_size,
    scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    transposed=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    swiglu_limit=swiglu_limit_arg,
    swizzle=self.use_swizzle,
)
```

这样可以避免：

```text
SwiGLU 先写 bf16 down_input
再读 bf16 down_input 做 quant
```

变成：

```text
读 gateup_output
直接计算 SwiGLU
直接输出 FP8 down_input 和 scale
```

这一步是该路径中最关键的内存优化之一：它减少了大 prefill shape 下 bf16 intermediate 的 materialize 和二次读取。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:417`

##### 3.2.5.3 down grouped GEMM

得到 `down_input_fp8` 后，执行 down projection：

```text
down_output = down_input_fp8 @ w2
```

代码模式为：

```python
deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
    (down_input_fp8, down_input_scale),
    w2_weight_fp8,
    down_output,
    m_indices,
    recipe_a=recipe_a,
    recipe_b=recipe_b,
)
```

这里同样是：

```text
FP8 activation x FP8 weight -> bf16 output
```

最后得到 bf16 `down_output`。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:478`

##### 3.2.5.4 reorder / combine

由于前面为了 grouped GEMM 把 token compact 到 expert-contiguous layout，最后需要把输出还原回原始 token 顺序，并根据 router weight 做 combine：

```text
expert-contiguous down_output
  -> 根据 src2dst / dst2src 映射回原 token
  -> 乘以 router weight
  -> top-k expert 输出累加
  -> final hidden_states
```

这一步决定了 compact path 的整体收益不只取决于 GEMM 本身，还取决于 compact / padding / reorder / combine 的成本是否被大 M GEMM 和 fused quant 收益覆盖。

#### 3.2.6 内存优化与 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY`

在 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY` 打开时，路径会直接分配 FP8 down input 并写入 scale，避免先 materialize bf16 down input 再额外 quant 的中间内存压力。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:401`

这对 prefill 大 token 场景尤其重要，因为 `down_input` 的大小随 token 数和 intermediate size 增长。减少 bf16 intermediate 不仅节省显存占用，也可以降低 memory bandwidth 压力。

#### 3.2.7 与 3.1 FP8 prefill quant 优化的关系

3.1 主要优化的是通用 FP8 activation quant path：

```text
activation -> FP8 activation + scale
```

3.2 中的 MoE DeepGEMM compact prefill path 也依赖 FP8 quant，尤其是：

```text
SwiGLU output -> down_input_fp8 + down_input_scale
```

二者关系可以理解为：

```text
3.1 是降低 FP8 quant kernel 本身成本
3.2 是减少 MoE pipeline 中 FP8 quant 和中间内存的整体成本
```

换句话说：

- 3.1 更偏单个 quant kernel / dispatch 优化。
- 3.2 更偏 MoE operator pipeline 优化。
- 两者共同服务 DSV4 FP8 prefill。

#### 3.2.8 后续现象与预期效果

从代码结构和 benchmark 设计看，这条路径预期带来以下效果：

1. **大 token prefill 有 MUSA 专用 MoE fast path**

   不再完全依赖通用 MoE runner，而是通过 MUSA + FP8 + prefill gate 进入 compact DeepGEMM path。

2. **decode 小 M 被保护**

   通过 `_DSV4_MUSA_COMPACT_MOE_MIN_TOKENS = 8192` 和 `_TILEKERNELS_SWIGLU_QUANT_MIN_ROWS = 1024`，避免小 M decode 误入 compact / fused prefill path。

3. **grouped GEMM 输入布局更适合大 M**

   compact layout 将 token 按 expert 连续排列，使 gate/up 和 down grouped GEMM 更容易获得稳定吞吐。

4. **SwiGLU + quant 的中间内存开销下降**

   fused post-quant 避免了先写 bf16 down input 再读回 quant 的 split 流程，减少 global memory traffic。

5. **内存占用更可控**

   fixed bucket rows、allocated rows、overflow flag 等设计让 prefill 大 MoE 的 workspace 使用更容易被观测和约束。

仓库中没有提交固定的 MUSA 实测日志，因此这里不把下面数值写成已测结论，只作为基于优化形态的预估范围。

MoE operator 级别，在命中该路径的大 prefill FP8 MoE 场景中，预期可能达到：

```text
约 1.2x ~ 2.0x operator 级加速
```

较理想场景可能接近：

```text
约 1.5x ~ 2.3x
```

理想条件包括：

- token 数足够大。
- expert 分布较均匀。
- padding 浪费较少。
- fused SwiGLU + quant 明显减少 memory traffic。
- grouped GEMM 吞吐较好。

收益较弱时可能只有：

```text
约 1.1x ~ 1.3x
```

常见原因包括：

- expert 分布不均。
- compact / reorder 开销偏高。
- padding row 较多。
- attention 或其他模块才是主要瓶颈。
- host 调度开销掩盖 device 侧收益。

端到端 prefill 收益通常低于单个 MoE operator。粗略估计：

```text
MoE 不是主瓶颈或 attention 占比较高：约 3% ~ 8%
MoE / FP8 path 命中较好：约 8% ~ 15%
与 FP8 quant、FlashMLA sparse prefill、cache store、compress 等优化叠加：可能进一步提高
```

最终收益必须以 MUSA 实机 benchmark 为准。

#### 3.2.9 benchmark 与验收方式

`test_moe_prefill_deepgemm_benchmarks.py` 提供了多种 benchmark 输出：

- baseline vs candidate device median/min/p95。
- host median。
- `device_speedup`。
- `host_speedup`。
- padded rows / allocated rows / worst-case rows。
- static cap vs exact compact / Triton 的 speedup。
- preprocess split report。

运行方式可参考：

```bash
SGLANG_RUN_DEEPSEEK_V4_MUSA_BENCHMARK=1 \
SGLANG_DSV4_MUSA_MOE_EXPERIMENTAL=1 \
python -m pytest python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_moe_prefill_deepgemm_benchmarks.py -s
```

典型输出字段：

```text
baseline_device=... ms
candidate_device=... ms
device_speedup=...x
baseline_device_min=... ms
candidate_device_min=... ms
baseline_device_p95=... ms
candidate_device_p95=... ms
baseline_host=... ms
candidate_host=... ms
host_speedup=...x
padded_valid_rows=...
allocated_rows=...
worst_case_rows=...
```

static cap 相关输出还会关注：

```text
triton=...ms
exact_compact=...ms
static_cap=...ms
speedup_vs_triton=...x
speedup_vs_exact=...x
overflow=...
rows=...
```

验收重点包括：

1. **dispatch correctness**

   - MUSA + FP8 + token 数足够大时进入 compact DeepGEMM path。
   - decode / small-M 不进入该路径。
   - 环境变量未打开时不进入实验路径。

2. **correctness**

   - compact / reorder 后输出 token 顺序正确。
   - top-k expert combine 结果正确。
   - FP8 quant scale layout 与 DeepGEMM expectation 一致。
   - candidate 与 baseline 数值误差在可接受范围内。

3. **performance**

   - `candidate_device` 低于 `baseline_device`。
   - `device_speedup` 有稳定收益。
   - `host_speedup` 不被额外调度开销抵消。
   - `padded_valid_rows / allocated_rows / worst_case_rows` 显示 padding 和 static cap 没有造成明显浪费。
   - overflow 情况可控，fallback 行为符合预期。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_moe_prefill_deepgemm_benchmarks.py:505`

#### 3.2.10 小结

这一节可以总结为四个关键词：

```text
MUSA-specific
FP8-specific
MoE prefill
large-token compact DeepGEMM
```

它的核心价值是：

1. 为 MUSA 上 DSV4 FP8 MoE prefill 增加专用 fast path。
2. 只在大 token prefill 场景启用，避免 decode 小 M 回退。
3. 通过 compact layout 提升 grouped GEMM 输入连续性和吞吐。
4. 通过 fused SwiGLU + FP8 quant 减少 bf16 intermediate 的读写和显存压力。
5. 通过 benchmark 输出 device / host speedup、padding rows、allocated rows、overflow 等指标进行验收。

### 3.3 FlashMLA sparse prefill seq-pack path

相关提交：

- `4fabebf7f` — large-Q FlashMLA prefill
- `d5fe7707c` — FlashMLA sparse prefill seq-pack
- `1ce2e4aae` — guarded sparse prefill pack8
- `b9b202c28` — CP round-robin 修复
- `844133222` — 移除 raw-index fallback

关键文件：

- `python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
- `python/sglang/srt/layers/attention/dsv4/dequant_k_cache.py`
- `python/sglang/srt/layers/attention/dsv4/indexer.py`
- `python/sglang/srt/models/deepseek_v4.py`

一句话概括：

> 这组优化是在 MUSA 上为 DSV4 sparse attention prefill 新增 FlashMLA seq-pack 路径，把每个 request 的 compressed cache 与 SWA window gather 到 flat workspace，并用 per-query rebased indices 交给 `flash_mla_sparse_fwd`，同时把 chunk 内不随 layer 变化的 metadata/cache 复用起来，减少每层重复构建和小 tensor 调度成本。

#### 3.3.1 背景：DSV4 sparse prefill 为什么需要 seq-pack

DeepSeek V4 的 attention 在 prefill 阶段不是简单的 dense KV cache 访问，而是同时涉及：

- compressed cache：例如 c4 / c128 压缩后的 KV 表示。
- SWA window：滑动窗口内的局部 token。
- per-query topk sparse indices。
- 不同 request 的不同有效长度。
- CP / context parallel 下 local query 与 global query 的映射。

如果每一层都临时构建这些 metadata，并且每个 request 分散 gather，就会在 prefill 大 token 场景下产生大量小 kernel、小 tensor 和重复 CPU/GPU 调度开销。

FlashMLA sparse prefill seq-pack 的目标是把这些分散访问整理成一个可被 FlashMLA 消费的连续 workspace：

```text
每个 request 的 compressed region
每个 request 的 SWA region
  -> 合并成 flat BF16 workspace
  -> 为每个 query 生成 rebased combined indices
  -> flash_mla_sparse_fwd 一次性消费
```

#### 3.3.2 profiling 现象与优化动机

这类优化通常对应以下 profiling 现象。

1. **每层重复构建 metadata 的开销高**

   DSV4 层数较多，文档中按约 61 层 prefill 预算组织 benchmark。如果 query_start、SWA token ids、combined indices、workspace shape 等每层都重新构建，就会出现大量重复的小 tensor 创建和 kernel launch。

2. **sparse attention 的 gather/index 开销被放大**

   sparse prefill 不是顺序读取一整段 dense KV，而是需要根据 topk compressed indices 和 SWA positional range 做 gather。若每个 request、每层分别处理，会导致访问碎片化和调度碎片化。

3. **FlashMLA 需要对齐后的 per-query indices**

   `flash_mla_sparse_fwd` 希望输入是对齐且 rebased 的 per-query indices。若每层临时处理 topk + SWA 合并和 padding，会让 attention 前处理成为 prefill 热点。

4. **CP round-robin 下 local/global metadata 容易错位**

   context parallel round-robin 拆分时，本地 q 数可能与 global q 数不一致。如果继续使用 global query_start 去构建 sparse prefill cache，会导致 local query metadata 与实际本地计算不匹配。

#### 3.3.3 核心策略一：统一使用 `flash_mla_sparse_fwd`

`_forward_prefill_sparse` 中将 prefill sparse attention 统一组织到 `flash_mla_sparse_fwd`：

- 替代 extend path 上的 `flash_mla_with_kvcache`。
- 每个 request positionally gather：
  - SWA window。
  - compressed cache，即 c4 / c128。
- gather 到 flat BF16 workspace。
- `flash_mla_sparse_fwd` 通过 per-query rebased indices 消费 workspace。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/attention/deepseek_v4_backend.py:1108`

这一步的意义是把 prefill sparse attention 的输入形态固定下来：

```text
分散的 cache / SWA / topk metadata
  -> flat workspace + combined indices
  -> FlashMLA sparse prefill kernel
```

也就是说，优化重点从“每层临时拼装若干输入”变成“chunk 内构建一次稳定 workspace 结构，并在 layer 间复用”。

#### 3.3.4 核心策略二：chunk-invariant metadata/cache 跨 layer 复用

`DSV4Metadata` 新增 `sparse_prefill_cache`，懒加载构建并跨当前 chunk 的所有 layer 复用。

代码注释写明：

> Lazily populated on the first call to `_forward_prefill_sparse` and reused across every layer in the chunk.

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/attention/deepseek_v4_backend.py:280`

`SparsePrefillChunkCache` 的注释进一步说明：

> Reused across every layer in the chunk to avoid rebuilding tiny tensors 61 times per forward pass.

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py:350`

这说明该优化明确针对 DSV4 多层 prefill 的重复构建问题。对一个 prefill chunk：

```text
第一次 sparse prefill layer
  -> 构建 query_start / SWA token ids / workspace / combined indices cache
后续 layer
  -> 复用 chunk cache，只更新 layer-dependent 的部分
```

能够复用的典型内容包括：

- local query_start。
- request 级别的 SWA token ids。
- workspace buffer。
- c0 / c128 combined indices。
- c4 combined indices 的输出 buffer。

#### 3.3.5 核心策略三：flat workspace 与 combined indices 布局

`sparse_prefill_utils.py` 中设计了 flat workspace：

```text
[ request r 的 compressed region ]
[ request r 的 SWA region        ]
```

每个 query 的 combined indices 由两部分组成：

```text
[ topk indices into compressed cache, rebased ]
[ swa positional indices, rebased             ]
[ -1 padding to multiple of 128               ]
```

workspace token 宽度是 512：

- 448 FP8 nope 反量化成 bf16。
- 64 bf16 rope。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py:1`

这个布局解决的问题是：FlashMLA sparse kernel 不直接感知原始 cache 的复杂层次，而是看到一个统一的 flat workspace 和已经 rebased 的 index：

```text
原始 compressed cache / SWA cache
  -> gather/dequant 到 workspace
  -> per-query combined indices 指向 workspace 内部 offset
  -> FlashMLA sparse prefill
```

#### 3.3.6 核心策略四：topk + SWA indices 合并 kernel

`combine_topk_swa_indices` 会生成 `flash_mla_sparse_fwd` 所需的 per-query combined indices 和 valid length：

- topk 部分：compressed cache local index + compressed_base。
- SWA 部分：`swa_base + positional_offset`。
- padding 到 128 对齐，满足 FlashMLA sparse prefill kernel 对 topk alignment 的要求。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py:56`

合并前，attention 逻辑上需要分别处理：

```text
topk compressed indices
SWA positional indices
```

合并后变成：

```text
combined_indices[q] = [compressed indices..., SWA indices..., padding...]
valid_lens[q] = 实际有效长度
```

这样 `flash_mla_sparse_fwd` 的输入更加规则，避免在主 attention kernel 外层反复处理多套 index 表示。

#### 3.3.7 核心策略五：SWA token ids 构建

`build_swa_token_ids` 将每个 request 的 SWA positional range 转成物理 SWA-cache token ids。

处理逻辑包括：

- 根据每个 request 的 first visible length 计算 `swa_first_pos`。
- `swa_gather_lens = seq_lens - swa_first_pos`。
- 通过 `req_to_token` 和 `full_to_swa` 映射到 SWA cache id。
- 使用 Triton kernel 并行构建 flat token ids。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py:166`

这一步的意义是把 request-local 的 SWA 位置转换成后端 cache 能直接访问的物理 token id，从而为后续 workspace gather 做准备。

#### 3.3.8 c0 / c4 / c128 分层缓存

`SparsePrefillChunkCache` 中分别维护：

- `c0_workspace`：SWA-only。
- `c128_workspace`：c128 compressed + SWA。
- `c4_workspace`：c4 compressed + SWA。
- `c0_combined_indices` / `c128_combined_indices` 可预计算。
- `c4_combined_indices` 由于 topk indices per-layer 相关，需要 per-layer combine，但输出 buffer 可复用。

参考：

- c0 预计算：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py:458`
- c128 cache：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py:482`
- c4 cache：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py:539`

分层缓存的核心价值是区分：

```text
chunk 内不随 layer 变化的部分 -> 一次构建，多层复用
layer-dependent 的部分       -> 每层更新，但复用输出 buffer
```

#### 3.3.9 CP round-robin 修复

`_build_sparse_prefill_cache_inputs` 中增加了 round-robin CP local query 的处理：

- 判断 `forward_mode.is_context_parallel_extend()`。
- 判断 `is_nsa_prefill_cp_round_robin_split()`。
- 当本地 q 数与 global q 数不一致时，重建 local `query_start_loc`。
- 用 local query_start 计算 local extend seq lens、req_pool_indices、seq_lens。

这修复了 sparse prefill 在 CP round-robin 下 local q metadata 与 global q metadata 不一致的问题。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/attention/deepseek_v4_backend.py:1231`

该修复属于 MUSA DSV4 sparse prefill 路径上的高价值正确性修复。没有这个修复时，优化后的 fast path 在 CP round-robin 场景可能因为 query metadata 错位而读错 workspace 或生成错误 indices。

#### 3.3.10 后续现象与预期效果

该优化预期带来的结构性效果包括：

1. **每层重复 metadata 构建减少**

   chunk-invariant cache 让 query_start、SWA token ids、workspace 和部分 combined indices 不再每层重建。

2. **FlashMLA sparse prefill 输入更规则**

   flat workspace + rebased combined indices 使 attention kernel 输入从“分散 cache + 多套索引”变成“连续 workspace + 单套 per-query indices”。

3. **小 tensor / 小 kernel 调度减少**

   对 61 层左右的 DSV4 prefill，减少重复构建可以降低 host/device 调度噪声。

4. **CP round-robin 正确性更稳定**

   local query metadata 与实际本地 q 数对齐，避免 sparse prefill fast path 在 CP 下错用 global metadata。

仓库中没有提交固定实测日志，因此这里只给预估范围：

```text
FlashMLA sparse attention 前处理 / metadata 构建：约 1.3x ~ 3.0x
单层 sparse prefill attention wrapper：约 1.1x ~ 1.5x
端到端 prefill：约 2% ~ 8%，长上下文 / 多层复用充分时更明显
```

如果 attention 本身不是瓶颈，或者 c4 indices 每层变化导致可复用比例较低，则端到端收益会更低。

#### 3.3.11 验收方式

验收重点包括：

1. **dispatch correctness**

   - prefill sparse attention 是否统一进入 `flash_mla_sparse_fwd`。
   - c0 / c4 / c128 是否选择正确 workspace。
   - CP round-robin 下是否使用 local query_start。

2. **correctness**

   - combined indices 中 compressed 部分和 SWA 部分 rebasing 正确。
   - padding 到 128 后无效位置为 `-1`。
   - workspace 中 448 nope + 64 rope 布局正确。
   - CP 场景下 local/global query 数不一致时输出正确。

3. **performance**

   - 每层 metadata 构建次数减少。
   - sparse prefill 前处理耗时下降。
   - FlashMLA sparse prefill 主路径没有被 index/gather 前处理抵消。
   - 多层 prefill 下 host overhead 和小 kernel 数减少。

### 3.4 FlashMLA cache store prefill 优化

相关提交：

- `7c992a2a8` — `Optimize DeepSeek V4 MUSA FlashMLA cache store prefill`
- `f393eaa98` — memory-bound kernels

关键文件：

- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/cache_kernels.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/cache_ops.py`
- `python/sglang/jit_kernel/tests/deepseek_v4_musa/tests/test_cache_ops.py`

一句话概括：

> 这组优化为 MUSA 上 FlashMLA cache store 增加 prefill 专用 dispatch，将 decode 小 token 和 prefill 大 token 分流，并在 prefill 下使用 tile-parallel / subwarp16、FP8 pack、uint32/stg32/stg128 写回等路径，降低 KV cache pack/store 的 memory-bound 开销。

#### 3.4.1 背景：cache store 是 prefill 中典型 memory-bound 环节

DSV4 FlashMLA cache store 需要把当前 token 的 KV 表示写入 cache。对于 FlashMLA/MLA 结构，cache store 不只是简单 memcpy，还包括：

- nope 部分从 bf16/fp32 quant 成 FP8 E4M3。
- 写入 FP8 cache 数据。
- 写入对应 scale。
- rope 部分以 bf16 格式 pack/store。
- 根据 page_size 和 token index 写入分页 cache。

prefill 阶段 token 数大，cache store 的读写量随 token 数线性增长，因此很容易成为 memory-bound kernel。

decode 阶段则 token 数小，优化目标不同：decode 更怕额外 launch / shared-memory reduce / 大 tile 策略导致 TPOT 回退。

#### 3.4.2 profiling 现象与优化动机

该路径对应的典型 profiling 现象包括：

1. **prefill cache store 带宽利用不足**

   legacy prefill kernel 中如果使用共享内存 AllReduce 或较粗的通用路径，会在大 token store 下出现 bandwidth 利用不足。

2. **decode 与 prefill 复用同一策略会互相伤害**

   decode 小 M 适合 x4 这类轻量路径；prefill 大 M 更适合 tile-parallel 或 half-warp/token x8。若统一使用一种策略，要么 prefill 不够快，要么 decode TPOT 回退。

3. **FP8 nope + BF16 rope 混合写回需要更高效 pack/store**

   nope 是 FP8，rope 是 bf16，scale 又是 fp32/指定 scale layout。分散写回会造成 store 粒度小、访存不连续和指令效率低。

#### 3.4.3 核心策略一：decode 与 prefill 分流

`cache_ops.py` 中对 FlashMLA cache pack 做 auto dispatch：

- decode 小 token：走 `decode_x4` / `decode_x4_fp32`。
- prefill 大 token：走 tile-parallel 或 subwarp16。

注释说明：

> The x4/x8 paths avoid the shared-memory AllReduce used by the legacy prefill kernel. Keep decode on x4; use half-warp/token x8 only for larger prefill rows to avoid decode TPOT regressions.

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/cache_ops.py:166`

这说明优化不是简单替换 kernel，而是明确区分两种 workload：

```text
decode small-M  -> 保守、轻量、低 TPOT 风险
prefill large-M -> 更高并行度、更高写带宽
```

#### 3.4.4 核心策略二：prefill tile-parallel / subwarp16 路径

prefill 路径中根据 dtype 和 shape 选择：

- `prefill_tile_parallel_x...`
- `prefill_subwarp16_x...`

对 bf16/fp32 输入优先使用 tile-parallel path，并设置 `compile_profile = "dsa_full"`。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/cache_ops.py:469`

tile-parallel / subwarp16 的核心意图是：

- 将一个 token 或一段 cache store 拆给更合适的线程粒度。
- 避免 legacy path 的共享内存 AllReduce 开销。
- 提高连续 store 的合并效率。
- 在大 token prefill 下提高 memory bandwidth 利用率。

#### 3.4.5 核心策略三：FP8 nope + BF16 rope pack/store

`cache_kernels.py` 中多处实现 FP8 pack 和 uint32/stg32/stg128 写回：

- 将 nope 部分从 bf16/fp32 quant 成 FP8 E4M3。
- scale 写到 cache 的 scale 区。
- rope 部分以 bf16 packed uint32 写入。
- 对 page_size 是 2/64/256 等场景做特化。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/cache_kernels.py:1397`

可以把写回理解成：

```text
输入 x_nope bf16/fp32
  -> per-token/group scale
  -> FP8 E4M3 pack
  -> uint32/stg32 写 cache

输入 x_rope bf16
  -> bf16 pair pack
  -> uint32/stg32/stg128 写 cache
```

这样做的意义是减少 byte-level store，尽量用更宽的 store 指令完成 cache 写入。

#### 3.4.6 后续现象与预期效果

预期效果包括：

1. **prefill cache store bandwidth 提升**

   tile-parallel / subwarp16 和更宽 store 粒度有助于提高大 token 写 cache 的带宽利用率。

2. **decode TPOT 被保护**

   decode 继续走 x4 / x4_fp32，不被 prefill 大 kernel 影响。

3. **page_size 多场景覆盖**

   page_size 2/64/256 均有测试覆盖，降低分页 cache layout 下的回归风险。

4. **FP8 cache 写入路径更贴合 FlashMLA layout**

   nope FP8、scale、rope bf16 分区写入，减少通用转换和 store 开销。

预估性能范围：

```text
cache store operator：约 1.2x ~ 2.5x
FlashMLA cache pack group benchmark：约 1.1x ~ 1.8x
端到端 prefill：约 1% ~ 5%
```

如果 prefill 中 cache store 占比较高、token 数大、page_size 与 fast path 匹配，收益更明显；如果 attention / MoE 才是主瓶颈，则端到端收益较小。

#### 3.4.7 测试与验收方式

`test_cache_ops.py` 覆盖：

- invalid indices 时不写 cache。
- page_size = 2/64/256。
- num_tokens = 256/8192。
- fp32/bf16 输入。
- tile_parallel_x8 dispatch。
- decode 小 M 行为保持。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/tests/test_cache_ops.py:117`

验收重点包括：

1. **dispatch correctness**

   - decode 小 token 继续走 decode x4 路径。
   - prefill 大 token 命中 tile-parallel / subwarp16。
   - 不同 dtype 和 page_size 选择预期分支。

2. **correctness**

   - invalid indices 不写 cache。
   - FP8 nope 数据、scale 和 bf16 rope 写入位置正确。
   - page boundary 下不越界、不串页。

3. **performance**

   - 8K prefill 下 cache pack latency 下降。
   - decode TPOT 没有因为 prefill path 引入而回退。
   - acceptance benchmark 中 FlashMLA cache pack 相关预算通过。

### 3.5 compress + fused norm rope prefill 优化

相关提交：

- `6a57c5b25` — `Optimize MUSA compress fused norm rope prefill with guarded dispatch`
- `f393eaa98` — memory-bound kernels

关键文件：

- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/norm_rope_kernels.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/norm_rope_ops.py`
- `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py`

一句话概括：

> 这组优化针对 MUSA 上 DSV4 compress rows 的 prefill 路径，将 RMSNorm、RoPE 以及 compress 相关准备工作通过 guarded dispatch 融合到更少的 kernel 中，减少中间 tensor、global memory round-trip 和 launch 开销。

#### 3.5.1 背景：compress rows 上 norm / rope 是高频前处理

DSV4 prefill 中，进入 compressed attention / MLA cache 之前，需要对 hidden 或 Q/K 相关数据做：

- RMSNorm。
- RoPE。
- 可能的 quant / pack / compress 准备。
- 根据 compress ratio 生成后续 cache 或 attention 输入。

如果这些步骤拆开执行，会形成类似：

```text
input
  -> RMSNorm kernel
  -> 写 norm output
  -> RoPE kernel
  -> 写 rope output
  -> compress/pack kernel
```

对于 8K prefill 和多层调用，这些 memory-bound 小/中型 kernel 的累计开销会比较明显。

#### 3.5.2 profiling 现象与优化动机

典型现象包括：

1. **norm / rope / compress 前后处理 kernel 串联**

   每个 kernel 本身计算量不一定大，但都要读写较大的 hidden/head_dim tensor，容易受 memory bandwidth 限制。

2. **中间 tensor 生命周期短但读写量大**

   norm output 或 rope output 可能只被下一步消费一次，却需要完整写回 global memory 再读回来。

3. **prefill 大 M 下 launch 和访存开销累计明显**

   单层看似不大，但乘以 DSV4 多层后，会成为 prefill operator-side budget 的一部分。

#### 3.5.3 核心策略一：compress prefill 行融合 norm + rope

benchmark 中的 `compress_fused_norm_rope_prefill_inplace_musa` 表明该路径将 prefill compress rows 上的 RMSNorm / RoPE 处理融合到单个 guarded dispatch 中，减少中间 tensor 和 kernel launch。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:329`

融合前可以理解为：

```text
RMSNorm
  -> write intermediate
RoPE
  -> write intermediate
compress/pack preparation
```

融合后变成：

```text
read input
  -> RMSNorm
  -> RoPE
  -> write final needed output
```

核心收益来自减少：

- 中间 tensor 写回。
- 中间 tensor 再读取。
- kernel launch 次数。

#### 3.5.4 核心策略二：guarded dispatch

该路径只针对可安全匹配的 compress rows / rope dim / hidden size / dtype 等场景启用，不匹配时回退原路径。

这类 guarded dispatch 对稳定性很重要，因为 norm / rope / compress 的 shape 约束较多：

- hidden size 是否为 fast path 支持值。
- rope dim 是否匹配专用 kernel。
- dtype 是否为 bf16/fp32 支持组合。
- compress rows 是否连续或满足 kernel 假设。
- 是否 inplace 安全。

因此这里不是替换所有路径，而是：

```text
shape / dtype / layout 命中 -> fused MUSA fast path
否则 -> 原稳定路径
```

#### 3.5.5 acceptance benchmark 预算

operator benchmark 为该路径设置了 8K prefill 下的预算：

- `compress_fused_norm_rope_prefill_h512_r64_c{compress_ratio}_b1_{num_tokens}`
- `budget_ms = 0.30`

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:367`

这个 budget 说明该路径被定位为非常轻量的 memory-bound fused 前处理：在 8K prefill 条件下，单次 operator 预算只有 0.30ms 量级。

#### 3.5.6 后续现象与预期效果

预期效果包括：

1. **kernel launch 数减少**

   norm + rope + compress 前处理由多个 kernel 收敛到 guarded fused path。

2. **global memory traffic 减少**

   短生命周期中间 tensor 不再完整 materialize。

3. **多层累计收益更明显**

   单次 operator 预算很小，但在多层 prefill 中重复调用，累计收益可观。

预估性能范围：

```text
单次 fused norm/rope/compress 前处理：约 1.5x ~ 3.0x
相关 operator group：约 1.2x ~ 2.0x
端到端 prefill：约 1% ~ 4%
```

如果原路径中 norm、rope、compress 已经被其他更大算子掩盖，则端到端收益会较小；如果多层重复调用且 memory bandwidth 紧张，收益会更明显。

#### 3.5.7 验收方式

验收重点包括：

1. **dispatch correctness**

   - 命中支持 shape 时进入 fused path。
   - 不支持 dtype / rope dim / hidden size 时回退。
   - inplace 场景不破坏后续使用的数据。

2. **correctness**

   - RMSNorm 输出与 baseline 近似一致。
   - RoPE 后 Q/K 位置编码正确。
   - compress rows 对应的输出 layout 正确。

3. **performance**

   - `compress_fused_norm_rope_prefill...` benchmark 通过 0.30ms budget。
   - 8K prefill 下相关前处理 latency 降低。
   - 多层重复调用没有引入稳定性回退。

### 3.6 compress prefill memory-bound kernel 优化

相关提交：

- `f393eaa98` — `Optimize DSV4 MUSA prefill memory-bound kernels`

关键文件：

- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/compress_kernels.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/compress_ops.py`
- `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py`

一句话概括：

> 这组优化针对 MUSA 上 DSV4 prefill 的 C4 / C128 compress kernel，围绕 reduce、write、parallel reduction 和 vectorized store 改善 memory-bound 路径，并用 logical bytes / GB/s 和 8K prefill budget 约束性能。

#### 3.6.1 背景：compress 是典型 memory-bound pipeline

DSV4 的 compressed attention/cache 路径中，C4 / C128 compress 需要读取 KV、score、APE 等数据，做 reduce 或写 cache。它的特点是：

- 算术强度不高。
- 读写数据量大。
- 对连续访问、vectorized load/store、parallel reduce 很敏感。
- 8K prefill 下调用规模大，容易成为 operator-side bottleneck。

benchmark 里分别覆盖：

- `compress_ratio4_prefill_b1_8192`
- `compress_ratio128_prefill_b1_8192`

#### 3.6.2 profiling 现象与优化动机

典型现象包括：

1. **C4 / C128 reduce 带宽受限**

   reduce 需要读取多组 kv / score / ape 数据，但每个输出元素的计算并不复杂。性能主要取决于内存读写效率和并行 reduce 组织。

2. **write kernel store 粒度影响明显**

   compress write 本质上是在把多段 head_dim 数据写入 cache。如果 store 粒度小或不连续，带宽利用率会下降。

3. **C4 与 C128 的最优策略不同**

   C4 reduce 规模较小，更关注轻量并行和写回效率；C128 reduce 聚合范围更大，更需要 parallel reduce 和合理的线程组织。

#### 3.6.3 核心策略一：C4 / C128 prefill compress 专用路径

benchmark 里分别覆盖：

- `compress_ratio4_prefill_b1_8192`
- `compress_ratio128_prefill_b1_8192`

这说明该优化不是单一通用 compress kernel，而是针对 C4 / C128 两种 compress ratio 分别建立 prefill acceptance 目标。

C4 可以理解为：

```text
较小压缩窗口 / 较少 reduce 输入
  -> 更关注 write 和轻量 reduce
```

C128 可以理解为：

```text
较大压缩窗口 / 更多 reduce 输入
  -> 更关注 parallel reduce 和读带宽
```

#### 3.6.4 核心策略二：logical bytes / derived GB/s 度量

benchmark 中定义了 logical bytes：

- C4 reduce：每个输出元素约 `8 kv + 8 score + 8 ape loads + 1 fp32 output store`。
- C4 write：拷贝四段 fp32 head_dim 到 cache。
- C128 reduce：每个输出元素约 `128 kv + 128 score + 128 ape loads + 1 fp32 output store`。
- C128 write：拷贝两段 fp32 head_dim 到 cache。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:198`

logical bytes 的意义是：对 memory-bound kernel，单看 ms 不够，还要看：

```text
derived GB/s = logical bytes / latency
```

如果优化后 latency 下降但 derived GB/s 仍然很低，说明访存组织仍有问题；如果 GB/s 提升，则说明 vectorized load/store 或 parallel reduce 方向有效。

#### 3.6.5 核心策略三：vector write 与 C128 parallel reduce

环境变量中也暴露了相关开关：

- `SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_VECTOR_WRITE`
- `SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_C128_PARALLEL_REDUCE`
- `SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_RAISE_PREFILL_MISS`

这说明该优化重点包括：

```text
write path  -> vectorized write，提高 store efficiency
C128 reduce -> parallel reduce，提高大窗口 reduce 吞吐
miss path   -> 可选择严格暴露 prefill fast path miss，便于验收
```

其中 `RAISE_PREFILL_MISS` 更偏调试/验收：当预期应命中 fast path 却没有命中时，可以尽早暴露，而不是悄悄 fallback 导致性能异常难以定位。

#### 3.6.6 acceptance benchmark 预算

8K prefill operator budget：

| benchmark | budget |
| --- | ---: |
| `compress_ratio4_prefill_b1_8192` | 6.0 ms |
| `compress_ratio128_prefill_b1_8192` | 10.0 ms |

参考：

- C4：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:516`
- C128：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:537`

#### 3.6.7 后续现象与预期效果

预期效果包括：

1. **compress reduce derived GB/s 提升**

   C4 / C128 reduce 的 logical bytes 固定后，latency 下降会直接体现为 derived GB/s 上升。

2. **compress write 更接近带宽上限**

   vector write 让 cache 写入更少受到小粒度 store 影响。

3. **C128 大窗口 reduce 改善更明显**

   C128 每个输出聚合更多输入，parallel reduce 的收益通常比 C4 更明显。

预估性能范围：

```text
C4 compress operator：约 1.2x ~ 1.8x
C128 compress operator：约 1.3x ~ 2.2x
端到端 prefill：约 2% ~ 6%
```

如果压缩路径在整体 prefill 中占比高，且内存带宽是瓶颈，收益会更明显；如果 MoE 或 attention 主计算占比更高，端到端收益会被稀释。

#### 3.6.8 验收方式

验收重点包括：

1. **dispatch correctness**

   - C4 / C128 prefill 分别命中对应 fast path。
   - vector write 和 C128 parallel reduce 开关生效。
   - `RAISE_PREFILL_MISS` 打开时，未命中 fast path 能暴露问题。

2. **correctness**

   - reduce 输出与 baseline 误差可接受。
   - compress write cache layout 正确。
   - 不同 compress ratio 下没有越界或错位。

3. **performance**

   - `compress_ratio4_prefill_b1_8192` 通过 6.0ms budget。
   - `compress_ratio128_prefill_b1_8192` 通过 10.0ms budget。
   - derived GB/s 相比 baseline 提升。

### 3.7 MHC / norm / rope / topk operator 优化

相关提交：

- `ef32b80f4`
- `3c6b17eae`
- `ea3e93c38`
- `f393eaa98`

关键文件：

- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/mhc_kernels.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/mhc_ops.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/mhc_prenorm_ops.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/norm_rope_kernels.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/norm_rope_ops.py`

一句话概括：

> 这一节汇总的是 MUSA 上 DSV4 prefill service-like path 中的一组 operator 级优化，包括 MHC post/pre、topk transform、RMSNorm、RoPE、indexer/rope/hadamard、norm-rope-quant group 以及 FlashMLA cache pack group，它们共同减少 prefill 中大量 memory-bound / launch-bound 辅助算子的尾部开销。

#### 3.7.1 背景：端到端 prefill 由多个辅助算子共同决定

前面的 3.1 ~ 3.6 分别覆盖了 FP8 quant、MoE、FlashMLA sparse、cache store、compress 等较明确的路径。但 DSV4 prefill 中还有一批高频 operator：

- MHC post / pre。
- MHC prenorm。
- topk transform 512。
- rmsnorm self。
- fused rope qk。
- c4 indexer rope hadamard。
- norm rope quant group。
- FlashMLA cache pack group。

这些算子单个看可能不是最大热点，但在 8K prefill、多层、多次调用下会累积成明显开销。

#### 3.7.2 profiling 现象与优化动机

典型现象包括：

1. **辅助算子数量多，launch-bound 明显**

   每个 layer 中 norm、rope、topk/indexer、MHC 前后处理都会出现。如果没有融合或专用 kernel，launch overhead 和调度间隙会被多层放大。

2. **大部分辅助算子偏 memory-bound**

   RMSNorm、RoPE、Hadamard、cache pack 都需要读写较大 tensor，但计算强度不高，优化重点是访存组织和融合。

3. **service-like 8K prefill 需要预算化验收**

   这些 operator 很难只用单个 E2E 指标定位，因此分支中通过 operator acceptance budgets 对每类算子设置上限，作为 E2E serving 前的 gate。

#### 3.7.3 MHC post / pre / prenorm 优化

MHC 相关路径覆盖：

- `mhc_post_2d_b1_8192`
- `mhc_pre_big_fuse_b1_8192`
- `SGLANG_OPT_DEEPGEMM_HC_PRENORM`
- `SGLANG_OPT_MHC_PRENORM_BACKEND`

对应文件包括：

- `mhc_kernels.py`
- `mhc_ops.py`
- `mhc_prenorm_ops.py`

这部分的优化重点是把 MHC 前后处理和 prenorm 组织成更适合 MUSA prefill 的路径：

```text
MHC pre / prenorm
  -> 尽量融合或使用 DeepGEMM HC prenorm backend
MHC post
  -> 使用 MUSA 专用 kernel 处理 2D prefill shape
```

它解决的是 MHC 相关前后处理在大 token prefill 下的 memory-bound 和 launch-bound 问题。

#### 3.7.4 topk / indexer / rope / hadamard 优化

相关 benchmark 包括：

- `topk_transform_512_b1_8192`
- `fused_rope_qk_64_b1_8192`
- `c4_indexer_rope_hadamard_h64_b1_8192`

这些路径服务于 sparse attention / compressed attention 的前处理：

```text
topk transform
  -> sparse/compressed indices
RoPE Q/K
  -> position encoding
indexer + rope + hadamard
  -> c4 compressed attention 前处理
```

优化方向包括：

- 对 topk=512 的 DSV4 shape 做专用 transform。
- 将 Q/K RoPE 融合，减少单独 rope kernel。
- 将 c4 indexer、rope、hadamard 组合处理，减少中间读写。

`SGLANG_DEEPSEEK_V4_MUSA_ENABLE_JIT_TOPK512` 用于打开 DSV4 MUSA JIT topk512 相关能力。

#### 3.7.5 RMSNorm / norm-rope-quant group 优化

相关 benchmark 包括：

- `rmsnorm_self_512_b1_8192`
- `norm_rope_quant_group_b1_8192_x{layer_count}`

这部分与 3.1 FP8 quant 和 3.5 fused norm rope 有交集，但这里更强调 group-level、多层级的 operator acceptance：

```text
RMSNorm
  -> RoPE
  -> FP8 quant/group quant
```

如果拆开执行，会出现多次读写 hidden / qk tensor。group benchmark 用 `x{layer_count}` 形式模拟多层重复调用，目标是约束累计耗时，而不只看单次 kernel。

#### 3.7.6 FlashMLA cache pack group 优化

相关 benchmark 包括：

- `flashmla_cache_pack_swa_b1_8192`
- `flashmla_cache_pack_swa_group_b1_8192_x{call_count}`
- `flashmla_cache_pack_c128_group_b1_8192_x{call_count}`

这部分与 3.4 cache store prefill 优化对应，但这里从 operator group 角度验收：

```text
单次 cache pack 是否足够快
多次 SWA cache pack 累计是否达标
多次 c128 cache pack 累计是否达标
```

这能防止单个 kernel 看起来可接受，但多层/多调用累计后拖慢 prefill。

#### 3.7.7 acceptance benchmark 预算

operator benchmark 中的 8K prefill acceptance budgets：

| benchmark | budget |
| --- | ---: |
| `mhc_post_2d_b1_8192` | 30.0 ms |
| `mhc_pre_big_fuse_b1_8192` | 40.0 ms |
| `topk_transform_512_b1_8192` | 10.0 ms |
| `rmsnorm_self_512_b1_8192` | 4.0 ms |
| `fused_rope_qk_64_b1_8192` | 6.0 ms |
| `c4_indexer_rope_hadamard_h64_b1_8192` | 2.0 ms |
| `norm_rope_quant_group_b1_8192_x{layer_count}` | 120.0 ms |
| `flashmla_cache_pack_swa_b1_8192` | 5.0 ms |
| `flashmla_cache_pack_swa_group_b1_8192_x{call_count}` | 20.0 ms |
| `flashmla_cache_pack_c128_group_b1_8192_x{call_count}` | 20.0 ms |

参考：

- MHC budgets：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:232`
- norm/rope budgets：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:307`
- FlashMLA cache pack budgets：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:551`

#### 3.7.8 后续现象与预期效果

这组 operator 优化通常不会像 MoE DeepGEMM 那样在单个算子上带来非常夸张的 E2E 收益，但它们对 prefill latency 的稳定性很关键。

预期效果包括：

1. **辅助算子尾部开销下降**

   MHC、topk、norm、rope、cache pack 等 operator 的单项 latency 被预算约束。

2. **多层累计开销更可控**

   group benchmark 直接覆盖 `x{layer_count}` 和 `x{call_count}`，避免单次优化无法反映多层累计的问题。

3. **E2E prefill 更稳定**

   当 MoE / attention 主路径优化后，辅助算子容易变成新的瓶颈；这些预算能提前兜住尾部 latency。

预估性能范围：

```text
单个辅助 operator：约 1.1x ~ 2.0x
group-level operator：约 1.2x ~ 2.5x
端到端 prefill：约 3% ~ 10%，与其他优化叠加时更明显
```

#### 3.7.9 验收方式

运行 operator acceptance benchmark：

```bash
SGLANG_RUN_DEEPSEEK_V4_MUSA_OPERATOR_BENCH=1 \
python -m pytest python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py -s
```

如果希望强制 budget：

```bash
SGLANG_RUN_DEEPSEEK_V4_MUSA_OPERATOR_BENCH=1 \
SGLANG_DSV4_OPERATOR_BENCH_ENFORCE=1 \
python -m pytest python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py -s
```

验收重点包括：

1. **dispatch correctness**

   - MHC prenorm backend、JIT topk512、FlashMLA cache pack group 等开关进入预期路径。
   - 不支持 shape 时可回退，不影响正确性。

2. **correctness**

   - topk transform indices 正确。
   - RMSNorm / RoPE / Hadamard 输出与 baseline 近似一致。
   - cache pack group 输出 layout 正确。

3. **performance**

   - 表格中的所有 8K prefill budget 通过。
   - JSONL 输出中的 median/min/p95 没有明显长尾。
   - derived GB/s / trace kernel 显示命中预期 kernel，而不是 fallback。

## 4. 性能数据与 benchmark 状态

### 4.1 仓库中能直接确认的性能目标

当前分支中没有找到已经提交的真实 benchmark 日志结果，例如固定的 `candidate_device=... ms` 或 E2E tokens/s 结果文件。能直接从代码确认的是：

1. benchmark 输出字段已经写好。
2. operator acceptance benchmark 里有明确的性能预算 `budget_ms`。
3. benchmark 形状主要覆盖 no-MTP B1 8K prefill service-like path。
4. 部分 MoE benchmark 会输出 baseline/candidate speedup。

因此本文档中的“性能”分成两类：

- **已固化在代码里的 acceptance budget**：可以直接引用。
- **运行 benchmark 后得到的实测性能**：代码支持输出，但仓库中未固化具体数值，需要在 MUSA 环境上实际执行。

### 4.2 Operator acceptance benchmark

文件：

- `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py`

benchmark 说明：

- 这是 opt-in benchmark。
- 注释说明它是 operator-side focused benchmark gate，目标是在 E2E serving 前先测 focused operators。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:1`

默认形状：

- `_PREFILL_TOKENS = 8192`
- `_PREFILL_LAYERS = 61`
- `_FLASHMLA_CACHE_CALLS = 84`
- `_HIDDEN_SIZE = 4096`

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:42`

运行开关：

```bash
SGLANG_RUN_DEEPSEEK_V4_MUSA_OPERATOR_BENCH=1 \
python -m pytest python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py -s
```

如果希望强制 budget：

```bash
SGLANG_RUN_DEEPSEEK_V4_MUSA_OPERATOR_BENCH=1 \
SGLANG_DSV4_OPERATOR_BENCH_ENFORCE=1 \
python -m pytest python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py -s
```

如果希望输出 JSONL：

```bash
SGLANG_RUN_DEEPSEEK_V4_MUSA_OPERATOR_BENCH=1 \
SGLANG_DSV4_OPERATOR_BENCH_JSONL=/tmp/dsv4_operator_bench.jsonl \
python -m pytest python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py -s
```

输出格式：

```text
OPERATOR_BENCH_RESULT { ... json ... }
```

JSON 中包含：

- benchmark name。
- `budget_ms`。
- `passed_budget`。
- median/min/p95/mean/sample count。
- logical bytes。
- derived GB/s。
- dispatch branch。
- trace kernel。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:139`

### 4.3 MoE DeepGEMM prefill benchmark

文件：

- `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_moe_prefill_deepgemm_benchmarks.py`

运行开关：

```bash
SGLANG_RUN_DEEPSEEK_V4_MUSA_BENCHMARK=1 \
SGLANG_DSV4_MUSA_MOE_EXPERIMENTAL=1 \
python -m pytest python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_moe_prefill_deepgemm_benchmarks.py -s
```

可以通过 `SGLANG_DSV4_MUSA_MOE_BENCH_CASES` 选择 case。

该 benchmark 会输出：

- baseline device median/min/p95。
- candidate device median/min/p95。
- device speedup。
- baseline host median。
- candidate host median。
- host speedup。
- compact routing rows。
- static cap rows。
- exact compact vs static cap speedup。
- preprocess split report。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_moe_prefill_deepgemm_benchmarks.py:505`

## 5. 重要环境变量

| 环境变量 | 默认/作用 | 说明 |
| --- | --- | --- |
| `SGLANG_DSV4_MUSA_MOE_EXPERIMENTAL` | 默认 false | 打开 DSV4 MUSA MoE experimental / compact prefill DeepGEMM 路径。 |
| `SGLANG_DISABLE_DSV4_MUSA_PREFILL_FP8_QUANT` | 默认未设置 | 设置为 `1` 后禁用 DSV4 MUSA prefill FP8 quant 专用路径。 |
| `SGLANG_OPT_USE_TILEKERNELS_FP8_QUANT` | benchmark 中默认 set 为 `1` | 使用 TileKernels FP8 quant 相关路径。 |
| `SGLANG_OPT_DSV4_FLASHMLA_LOCAL_PREFILL_Q` | 默认 true | DSV4 FlashMLA local prefill Q 优化开关。 |
| `SGLANG_DEEPSEEK_V4_MUSA_ENABLE_JIT_TOPK512` | benchmark 中默认 set 为 `1` | 启用 DSV4 MUSA JIT topk512。 |
| `SGLANG_ENABLE_JIT_DEEPGEMM` | benchmark 中默认 set 为 `1` | 启用 JIT DeepGEMM。 |
| `SGLANG_OPT_DEEPGEMM_HC_PRENORM` | benchmark 中默认 set 为 `1` | DeepGEMM HC prenorm 优化。 |
| `SGLANG_OPT_MHC_PRENORM_BACKEND` | benchmark 中默认 `deepgemm` | MHC prenorm backend。 |
| `SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_VECTOR_WRITE` | benchmark 中默认 set 为 `1` | compress vector write 优化。 |
| `SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_C128_PARALLEL_REDUCE` | benchmark 中默认 set 为 `1` | C128 parallel reduce 优化。 |

参考：

- env 定义：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/environ.py`
- benchmark 默认环境：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:87`

## 6. 建议阅读顺序

如果要继续深入看代码，建议按如下顺序：

1. **FP8 quant dispatch**  
   `python/sglang/srt/layers/quantization/fp8_kernel.py`

2. **MoE DeepGEMM prefill**  
   `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`  
   `python/sglang/srt/layers/deepseek_v4_musa/kernels/moe_prefill_kernels.py`  
   `python/sglang/srt/layers/deepseek_v4_musa/ops/moe_prefill_ops.py`

3. **FlashMLA sparse prefill**  
   `python/sglang/srt/layers/attention/deepseek_v4_backend.py`  
   `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`

4. **FlashMLA cache store**  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/cache_ops.py`  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/cache_kernels.py`

5. **compress / norm / rope / MHC operator**  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/norm_rope_ops.py`  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/norm_rope_kernels.py`  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/compress_ops.py`  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/compress_kernels.py`  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/mhc_ops.py`

6. **性能验证**  
   `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py`  
   `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_moe_prefill_deepgemm_benchmarks.py`

## 7. Flash / Pro 划分与高价值 MUSA 问题修复

### 7.1 只保留 MUSA 相关修复的口径

本文档只记录 **DeepSeek V4 / DSV4 在 MUSA 上运行相关** 的优化和修复。筛选标准是：提交需要涉及 MUSA runtime、MUSA kernel、MCCL、MUSA graph capture、MUSA fallback，或 DSV4 在 MUSA 上会使用的 shared code path。

### 7.2 Flash / Pro 在 MUSA 文档里的处理方式

从提交历史和测试命名看，DeepSeek V4 相关工作里有 **Flash** 和 **Pro** 两类模型/场景，但很多 MUSA 底层修复并不会显式标注 Flash 或 Pro。

| 类别 | MUSA 文档中的处理方式 | 示例 |
| --- | --- | --- |
| DSV4 Flash on MUSA | 只保留标题、代码路径或测试明确关联到 DSV4 Flash，且修复会影响 MUSA 运行的提交 | `Fix/dsv4 flash eagle dummy ima` 如果用于 MUSA dummy / EAGLE / IMA 验证，可保留。 |
| DSV4 Pro on MUSA | 只保留 Pro 相关且代码路径涉及 MUSA runtime / MUSA kernels / MUSA distributed / DSV4 shared MUSA path 的提交 | `Support DSV4 Pro DP attention and CP8 serving` 是 Pro serving 能力提交，但需要结合 MUSA 部署场景判断。 |
| 通用 DSV4 on MUSA | 保留会影响 MUSA DSV4 attention/cache/compress/runtime 的通用修复，即使标题没有写 Flash/Pro | MUSA fallback blockers、MCCL device binding、TileLang JIT name、MUSA prefill block_id re-bind 等。 |

### 7.3 值得关注的 MUSA 修复提交

| Commit | 分类 | 问题 | 修复要点 | 价值 |
| --- | --- | --- | --- | --- |
| `5fcb7c439` | MUSA runtime / fallback | DeepSeek V4 MUSA 运行时可能静默进入不支持的 CUDA JIT 或 torch fallback 路径，strict no-fallback E2E 被阻塞 | MUSA tensor 进入 `hisparse_offload_to_host` 时 fail-closed；MUSA rmsnorm TileLang miss 后默认不允许 torch fallback，graph capture 下直接报错；compress prefill miss 可通过 env 强制抛错 | 避免 MUSA 路径“悄悄跑到 CUDA/torch fallback”，能更早暴露缺失 kernel，是稳定上线 MUSA DSV4 的关键修复。 |
| `ef4469cb5` | MUSA distributed / runtime path | MCCL process group 没有显式绑定 local MUSA device；MUSA operator forwarding patch 容易不同步 | `init_process_group(backend="mccl")` 时传入 `device_id=torch.device("musa", local_rank)`；同步 DeepSeek V4 MUSA ops forwarding exports | 修复多卡 MUSA 初始化和 runtime path 的基础稳定性问题，避免通信后端设备绑定错误。 |
| `3cceb08b9` | MUSA TileLang JIT cache | C4 decode page kernel 对不同 `extra_data_cols` 使用同一个 TileLang JIT 名称，paged mode 与 page4 mode 切换时可能命中错误缓存 | JIT 名称从固定 `dsv4_c4_decode_page4_t{threads}` 改为包含 `extra_data_cols` 的 `dsv4_c4_decode_page{extra_data_cols}_t{threads}` | 修复 MUSA TileLang kernel cache collision，避免不同 layout 复用错误编译产物。 |
| `81fc824a2` | MUSA compress / prefill kernel | MUSA C4 prefill page write kernel 中 `block_id` 被 re-bind，触发 TileLang immutable value re-bind 问题；DeepSeek V4 compressor 没启用 paged 模式 | 使用 `T.alloc_local((1,), dtype=T.int32)` 保存可变 `block_id`；`Compressor.compress_fused()` 中 `is_paged=True` | 修复 MUSA prefill compress page write kernel 的编译/语义问题，同时启用 paged compression metadata。 |
| `b9b202c28` | MUSA prefill / CP shared path | DSV4 sparse prefill 在 CP round-robin split 下，本地 q metadata 与 global q metadata 不一致 | 在 `_build_sparse_prefill_cache_inputs()` 中识别 round-robin CP local q，重建 local query_start、local extend seq lens、req_pool_indices、seq_lens | 对 MUSA prefill CP serving 很关键，避免 sparse prefill 在 CP round-robin 下读错 metadata。 |
| `5a15cde85` | MUSA CP shared path | DeepSeek V4 CP 路径中 bf16 KV 可能因非 contiguous 输入触发 fused norm rope / all-gather 后续错误 | 在 `_compute_kv_bf16()` 中对 `kv` 增加 `kv = kv.contiguous()` 后再做 fused norm rope | 修复 NSA prefill-CP 需要 bf16 KV all-gather 时的 layout 问题，可影响 MUSA CP 路径。 |

### 7.4 关键 MUSA 修复细节

本节按统一维度解析每个 MUSA 相关修复：

```text
1. 问题是什么？
2. 问题的表象是什么？
3. 问题的根源是什么？
4. 怎么定位的？
5. 怎么修复的？
6. 修复后有什么影响？
```

#### 7.4.1 `5fcb7c439` — DeepSeek V4 MUSA fallback blockers

标题：`Fix DeepSeek V4 MUSA fallback blockers`

修改文件示例：

- `python/sglang/jit_kernel/deepseek_v4.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/norm_rope_ops.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/compress_ops.py`

##### 问题是什么？

这个提交解决的是 **DeepSeek V4 在 MUSA 上运行时可能静默进入错误 fallback 路径** 的问题。

主要包括三类：

1. MUSA tensor 误入 CUDA-only HiSparse transfer JIT。
2. MUSA graph capture 下 TileLang kernel miss 后 fallback 到 torch path。
3. compress prefill kernel miss 后静默 fallback，导致性能问题不容易暴露。

也就是说，模型表面上是在 MUSA 上运行，但某些子路径可能实际没有走 MUSA 专用 kernel，而是进入了：

```text
CUDA-only path
torch fallback path
slow fallback path
graph-incompatible path
```

##### 问题的表象是什么？

表象可能有几种。

1. **运行时直接报 CUDA / JIT 相关错误**

   MUSA tensor 进入 CUDA-only HiSparse transfer 后，可能出现：

   ```text
   CUDA module load failed
   unsupported device
   kernel not found
   invalid backend
   ```

   这类错误本质上不是业务逻辑问题，而是 MUSA tensor 被送进了 CUDA 专用实现。

2. **程序能跑，但性能明显不对**

   更危险的是 silent fallback：

   ```text
   MUSA fast path 没有命中
   但是代码没有报错
   自动 fallback 到 torch / generic path
   ```

   用户看到的是：

   ```text
   prefill latency 很高
   TPOT 异常
   benchmark 不达标
   GPU 利用率异常
   kernel trace 里没有预期 MUSA kernel
   ```

3. **graph capture 失败**

   MUSA serving 通常需要 graph capture。如果 capture 阶段 fallback 到 torch path，可能出现：

   ```text
   graph capture failed
   operation not capturable
   unexpected CPU/GPU sync
   ```

   因为 torch fallback 很可能包含动态图分配、同步或不支持 capture 的操作。

##### 问题的根源是什么？

根源是：**原逻辑对 MUSA fast path miss 的处理太宽松**。

具体来说：

1. 上游或通用路径中存在 CUDA-only helper，例如 `hisparse_offload_to_host`。
2. 某些 MUSA op 如果 TileLang kernel miss，会尝试 torch fallback。
3. compress prefill 如果没有命中 MUSA fast path，默认可能继续走 fallback，而不是强制暴露问题。

这些 fallback 在开发阶段方便调试，但在 MUSA DSV4 serving 中会带来两个问题：

```text
正确性风险：错误后端路径被调用
性能风险：fast path miss 被隐藏
```

##### 怎么定位的？

定位思路通常是：

1. **先看 benchmark 或 E2E 性能异常**

   例如：

   ```text
   8K prefill latency 明显高于预期
   operator acceptance benchmark 不达标
   trace kernel 里没有预期 MUSA kernel
   ```

2. **打开严格模式或 trace**

   通过 kernel trace / dispatch log 观察实际进入了哪个路径：

   ```text
   预期：MUSA TileLang RMSNorm / compress prefill kernel
   实际：torch fallback / generic path / CUDA JIT helper
   ```

3. **在 MUSA tensor 进入可疑函数时检查 device**

   例如发现：

   ```python
   hisparse_offload_to_host(x)
   ```

   被 MUSA tensor 调用，但这个函数本质是 CUDA HiSparse transfer path。

4. **检查 graph capture 下 fallback 行为**

   如果 graph capture 期间出现 fallback，说明 fallback path 不应该在 MUSA production path 中被允许。

##### 怎么修复的？

修复思路是：**fail closed，而不是 silent fallback**。

1. **MUSA tensor 禁止进入 CUDA HiSparse transfer**

   `hisparse_offload_to_host()` 中如果发现输入 tensor 是 MUSA tensor，直接抛：

   ```python
   raise NotImplementedError(
       "DeepSeekV4 MUSA hisparse_offload_to_host is not implemented; "
       "CUDA HiSparse transfer must not be loaded for MUSA tensors"
   )
   ```

   含义是：

   ```text
   MUSA tensor 不允许走 CUDA-only helper
   ```

2. **MUSA graph capture 下禁止 torch fallback**

   `rmsnorm_self_musa()` 中，如果 TileLang miss：

   ```text
   graph capture 中直接 raise
   默认不允许 torch fallback
   只有 SGLANG_MUSA_ALLOW_TORCH_FALLBACK=1 才允许 debug fallback
   ```

3. **compress prefill miss 可强制抛错**

   如果设置：

   ```text
   SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_RAISE_PREFILL_MISS=1
   ```

   当预期应命中 compress prefill fast path 但没有命中时，直接 raise。

##### 修复后有什么影响？

正面影响：

1. **避免 silent fallback**

   原来可能悄悄跑慢路径，现在会直接暴露：

   ```text
   MUSA fast path miss
   CUDA-only path 被误调用
   graph capture 不兼容 fallback
   ```

2. **更利于性能验收**

   benchmark 如果没命中 fast path，会直接失败，而不是给出一个很慢但不报错的结果。

3. **提升生产稳定性**

   MUSA serving 中不会因为 fallback 混入 torch / CUDA path 导致不可控问题。

可能副作用：

1. **容错变少**

   原来某些 case 还能靠 fallback 跑通，现在会直接报错。

2. **开发调试需要显式打开 fallback**

   例如需要设置：

   ```text
   SGLANG_MUSA_ALLOW_TORCH_FALLBACK=1
   ```

   才能临时允许 debug fallback。

#### 7.4.2 `ef4469cb5` — DeepSeek V4 MUSA runtime paths

标题：`Fix DeepSeek V4 MUSA runtime paths`

修改文件示例：

- `python/sglang/srt/distributed/parallel_state.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/__init__.py`
- `python/sglang/srt/models/deepseek_v4.py`

##### 问题是什么？

这个提交解决的是 **DeepSeek V4 MUSA 多卡 runtime 初始化和 MUSA ops forwarding 不完整** 的问题。

主要包括：

1. MCCL process group 初始化时没有显式绑定 local MUSA device。
2. DeepSeek V4 MUSA ops forwarding / exports 不完整，导致上层调用找不到某些 MUSA op wrapper。

##### 问题的表象是什么？

1. **多卡初始化异常**

   在多卡 MUSA serving 中，可能出现：

   ```text
   MCCL init failed
   device mismatch
   rank 使用了错误 device
   通信 backend 初始化失败
   ```

   或者初始化能过，但后续 collective 通信异常。

2. **rank/device 对应关系错乱**

   例如：

   ```text
   rank0 不在 musa:0 上
   rank1 不在 musa:1 上
   ```

   可能导致：

   ```text
   collective hang
   通信性能异常
   跨卡数据错误
   ```

3. **MUSA op 找不到**

   如果 ops forwarding exports 不完整，上层调用可能报：

   ```text
   AttributeError
   ImportError
   operator not found
   missing MUSA op wrapper
   ```

##### 问题的根源是什么？

根源有两个。

1. **MCCL 初始化没有 device_id**

   在 MUSA 多卡环境下，`init_process_group(backend="mccl")` 最好显式传入当前 rank 对应的 device：

   ```python
   torch.device("musa", local_rank)
   ```

   如果不传，backend 可能依赖默认 device 或当前上下文，导致 rank/device 绑定不稳定。

2. **MUSA op patch / forwarding 依赖完整导出**

   DeepSeek V4 模型层会通过 MUSA backend ops 去替换或转发某些算子。如果 `ops/__init__.py` 没有导出对应 wrapper，上层即使实现了 kernel，也可能找不到入口。

##### 怎么定位的？

1. **MCCL 问题定位**

   从多卡启动日志入手：

   ```text
   torch.distributed.init_process_group backend=mccl
   local_rank
   current_device
   ```

   如果发现 process group 初始化时没有传 device，或者 current device 与 local_rank 不一致，就可以定位到 device binding 问题。

2. **ops forwarding 问题定位**

   从报错栈看，一般会显示：

   ```text
   from deepseek_v4_musa.ops import xxx failed
   attribute xxx not found
   ```

   再检查：

   ```text
   deepseek_v4_musa/ops/__init__.py
   ```

   发现实际 kernel wrapper 存在，但没有导出。

##### 怎么修复的？

1. **MCCL process group 绑定 local MUSA device**

   在 distributed 初始化中增加：

   ```python
   init_kwargs = {}
   if backend == "mccl" and local_rank >= 0:
       init_kwargs["device_id"] = torch.device("musa", local_rank)
   ```

   然后传给：

   ```python
   torch.distributed.init_process_group(...)
   ```

2. **同步 MUSA ops forwarding exports**

   在：

   ```text
   python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/__init__.py
   ```

   补充导出 DeepSeek V4 MUSA runtime 需要的 op helper / wrapper。

##### 修复后有什么影响？

正面影响：

1. **多卡 MUSA 初始化更稳定**

   每个 rank 明确绑定自己的 local MUSA device。

2. **减少通信 backend 异常**

   降低 MCCL 初始化失败、device mismatch、collective hang 的概率。

3. **MUSA ops patch 更完整**

   上层 DeepSeek V4 runtime 能找到 MUSA op wrapper。

可能副作用：

- 如果某些部署脚本之前依赖默认 device 行为，修复后会更严格要求：

  ```text
  local_rank 正确设置
  rank 与 MUSA device 对应正确
  ```

  但这是合理的生产要求。

#### 7.4.3 `3cceb08b9` — TileLang JIT name collision 修复

标题：`fix(dsv4_musa): use distinct TileLang JIT name for different extra_data_cols`

修改文件：

- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/compress_kernels.py`

##### 问题是什么？

这个提交修复的是 **MUSA TileLang JIT kernel cache name collision**。

具体是 C4 decode page kernel 在不同 layout 下使用了相同 JIT 名称，导致不同参数的 kernel 可能复用同一个编译缓存。

##### 问题的表象是什么？

可能表现为：

```text
第一次运行正常，切换 page mode 后异常
paged mode 和 page4 mode 交替运行时结果错误
cache 写入错位
decode 输出异常
偶发 correctness mismatch
```

更麻烦的是，这类问题可能不稳定：

```text
取决于哪个 layout 先触发 JIT 编译
```

例如：

1. `extra_data_cols=4` 先编译。
2. 后面 `extra_data_cols=1` 复用同一个 JIT cache。
3. page1 layout 误用 page4 kernel。

##### 问题的根源是什么？

根源是 JIT name 没有包含影响 kernel layout 的参数。

原来的 JIT 名称类似：

```text
dsv4_c4_decode_page4_t{threads}
```

但是实际 kernel 有参数：

```text
extra_data_cols = 1 或 4
```

`extra_data_cols` 会影响 extra_data 的 layout 和访问方式，却没有进入 JIT cache key。因此两个不同 kernel 实际共享了同一个名字。

##### 怎么定位的？

这类问题一般通过以下方式定位：

1. **发现不同 page mode 切换后结果异常**

   例如：

   ```text
   单独跑 page4 正常
   单独跑 paged 正常
   先跑 page4 再跑 paged 异常
   ```

   这种顺序相关问题通常很像 JIT cache collision。

2. **检查 TileLang JIT name**

   看到 JIT 装饰器使用固定名称：

   ```python
   dsv4_c4_decode_page4_t{threads}
   ```

   但是 kernel 参数中存在：

   ```python
   extra_data_cols
   ```

   且不同值会影响 kernel 访问 layout。

3. **确认 JIT cache key 缺失参数**

   如果 JIT cache 只看 name，就会复用错误编译产物。

##### 怎么修复的？

修复方式是把 `extra_data_cols` 加入 JIT name：

```python
@_tilelang_jit(
    tilelang,
    f"dsv4_c4_decode_page{extra_data_cols}_t{threads}",
    ...,
)
```

这样会生成不同 kernel 名称：

```text
dsv4_c4_decode_page1_t128
dsv4_c4_decode_page4_t128
```

不同 layout 不再复用同一个 JIT cache。

##### 修复后有什么影响？

正面影响：

1. **避免不同 layout 误用同一 kernel**

   paged mode 和 page4 mode 独立编译、独立缓存。

2. **消除顺序相关 correctness bug**

   不再依赖哪个 layout 先触发 JIT。

3. **提高 decode / cache kernel 稳定性**

可能副作用：

- 会多生成一些 JIT 编译产物：

  ```text
  page1 一份
  page4 一份
  ```

  首次运行时可能多一点 JIT 编译成本，但这是正确性所必需的。

#### 7.4.4 `81fc824a2` — MUSA prefill kernel `block_id` re-bind 修复

标题：`feat(dsv4_musa): enable is_paged=True and fix prefill kernel block_id re-bind`

修改文件：

- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/compress_kernels.py`
- `python/sglang/srt/models/deepseek_v4.py`

##### 问题是什么？

这个提交修复两个问题：

1. MUSA C4 prefill page write kernel 中 `block_id` 存在 TileLang immutable value re-bind 问题。
2. DeepSeek V4 compressor 没有启用 paged compression metadata。

##### 问题的表象是什么？

1. **TileLang 编译或运行异常**

   因为 kernel 中存在类似：

   ```python
   block_id = indices[batch_id]
   ...
   block_id = extra_data[batch_id, 2]
   ```

   TileLang 对这种变量重新绑定不友好，可能导致：

   ```text
   JIT compile error
   immutable value re-bind error
   kernel generated code 不符合预期
   ```

2. **prefill page write 写错 block**

   如果 `block_id` 处理不正确，可能导致：

   ```text
   写入错误 page/block
   cache 数据错位
   后续 attention 读到错误 compressed cache
   ```

3. **paged compression metadata 没生效**

   如果 `is_paged=False`，compressor 可能按非 paged layout 生成 metadata，但 MUSA prefill page write 需要 paged 信息。

   结果可能是：

   ```text
   page index 不正确
   block id 不正确
   compressed cache layout 不匹配
   ```

##### 问题的根源是什么？

1. **TileLang 变量语义限制**

   在 TileLang kernel 内，某些值被视为 immutable expression。直接 re-bind：

   ```python
   block_id = ...
   block_id = ...
   ```

   容易触发编译器限制或生成错误代码。

2. **compressor metadata 与 paged kernel 假设不一致**

   MUSA prefill compress page write kernel 需要 paged metadata，但上层 `Compressor.compress_fused()` 仍然使用：

   ```python
   is_paged=False
   ```

   导致上层 metadata 与下层 kernel 期望不一致。

##### 怎么定位的？

1. **定位 block_id re-bind**

   一般从 TileLang JIT 编译错误或 kernel 行为异常开始。

   看到报错或异常位置在 prefill page write kernel，然后检查代码，发现：

   ```python
   block_id = indices[batch_id]
   ...
   block_id = extra_data[batch_id, 2]
   ```

   这是典型 re-bind 模式。

2. **定位 paged metadata 问题**

   从 cache/page 写入结果或 metadata 检查入手：

   ```text
   kernel 期望 paged metadata
   上层传入的是 non-paged metadata
   ```

   再追到 `Compressor.compress_fused()`，发现 `is_paged=False`。

##### 怎么修复的？

1. **用 local mutable buffer 保存 block_id**

   原逻辑：

   ```python
   block_id = indices[batch_id]
   if position < extra_data[batch_id, 3]:
       block_id = extra_data[batch_id, 2]
   ```

   修复后：

   ```python
   block_id = T.alloc_local((1,), dtype=T.int32)
   block_id[0] = indices[batch_id]

   if position < extra_data[batch_id, 3]:
       block_id[0] = extra_data[batch_id, 2]

   block_id_i64 = T.Cast("int64", block_id[0])
   ```

   这样不是重新绑定变量，而是修改 local buffer 的内容。

2. **启用 paged compression metadata**

   在 `Compressor.compress_fused()` 中：

   ```python
   is_paged=True
   ```

   替代原来的：

   ```python
   is_paged=False
   ```

##### 修复后有什么影响？

正面影响：

1. **TileLang kernel 更容易稳定编译**

   避免 immutable re-bind 问题。

2. **prefill page write block_id 更正确**

   不同 position / page 条件下能写到正确 block。

3. **compressor metadata 与 paged kernel 对齐**

   上层生成 paged metadata，下层按 paged kernel 消费。

4. **MUSA prefill compress path 更稳定**

可能副作用：

- 启用 `is_paged=True` 后，metadata layout 会变化。需要确保所有消费这份 metadata 的路径都支持 paged 模式。

从提交目的看，这是为了让 DSV4 MUSA compress path 与实际 paged cache 设计保持一致。

#### 7.4.5 `b9b202c28` — DSV4 sparse prefill under CP round-robin

标题：`Fix DSV4 sparse prefill under CP round-robin`

修改文件：

- `python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`

这个修复虽然不是 `deepseek_v4_musa` 目录下的 MUSA-only kernel，但它影响 DSV4 prefill CP shared path，MUSA prefill serving 会使用这套 metadata。

##### 问题是什么？

这个提交修复的是 **DSV4 sparse prefill 在 context parallel round-robin split 下 local query metadata 错误** 的问题。

##### 问题的表象是什么？

在 CP round-robin prefill 下可能出现：

```text
sparse prefill 输出错误
FlashMLA sparse prefill 读错 workspace
combined indices 错误
query_start_loc 不匹配
某些 rank 上 q length 不对
```

严重时可能表现为：

```text
shape mismatch
index out of bounds
attention 结果异常
不同 CP 配置下结果不一致
```

##### 问题的根源是什么？

根源是：**CP round-robin 下 local q 和 global q 不一致，但 sparse prefill cache 构建仍然使用了 global metadata**。

普通 prefill 中：

```text
global query_start
local query_start
```

可能是一致或简单切片关系。

但 CP round-robin split 中，不同 rank 拿到的是按 round-robin 切分后的 local queries：

```text
global q: q0 q1 q2 q3 q4 q5 q6 q7
rank0:   q0 q2 q4 q6
rank1:   q1 q3 q5 q7
```

此时 local rank 的 q 数、query_start、extend seq lens 都不能直接用 global metadata。

如果继续用 global query_start 构建 sparse prefill workspace，就会错位。

##### 怎么定位的？

1. **问题只在 CP round-robin 下出现**

   普通 prefill 或非 round-robin CP 正常，但 round-robin CP 出错。

   这说明问题与：

   ```text
   local/global query mapping
   ```

   有关。

2. **检查 `_build_sparse_prefill_cache_inputs`**

   发现它构建 sparse prefill cache inputs 时，使用的是 global metadata，没有针对 round-robin local q 重建。

3. **对比 local q 数与 global q 数**

   发现：

   ```text
   local q length != global q length
   ```

   但后续 metadata 仍然按 global q 生成。

##### 怎么修复的？

在 `_build_sparse_prefill_cache_inputs()` 中增加 CP round-robin 判断：

```text
forward_mode.is_context_parallel_extend()
is_nsa_prefill_cp_round_robin_split()
```

当检测到 round-robin CP local q 与 global q 不一致时，重建 local metadata：

- local `query_start_loc`
- local extend seq lens
- local req_pool_indices
- local seq_lens

也就是：

```text
global metadata
  -> 根据当前 rank 的 local q 重新生成 local metadata
  -> 用 local metadata 构建 sparse prefill cache
```

##### 修复后有什么影响？

正面影响：

1. **CP round-robin sparse prefill 正确性提升**

   每个 rank 使用自己的 local query metadata。

2. **FlashMLA sparse workspace index 更准确**

   combined indices、workspace gather、valid lens 都基于 local q。

3. **MUSA CP serving 更稳定**

   长上下文 + CP + sparse prefill 场景不再因为 metadata 错位出错。

可能副作用：

- 会增加一点 local metadata 重建开销。

但这个开销相比 sparse prefill 正确性是必要的，而且只在 CP round-robin 条件下触发。

#### 7.4.6 `5a15cde85` — DSV4 CP bf16 KV contiguous 修复

标题：`fix deepseek v4 CP error`

修改文件：

- `python/sglang/srt/models/deepseek_v4.py`

##### 问题是什么？

这个提交修复的是 **DeepSeek V4 CP 路径中 bf16 KV tensor 非 contiguous 导致后续 fused norm rope / all-gather 出错** 的问题。

##### 问题的表象是什么？

可能表现为：

```text
CP prefill 报错
fused_norm_rope_inplace 输出异常
all-gather 后 KV 数据错乱
某些 rank 上 attention 结果不一致
```

也可能是更底层的 kernel layout 错误，例如：

```text
stride 不符合 kernel 假设
非 contiguous tensor 被当 contiguous 处理
```

##### 问题的根源是什么？

根源是：

```text
_compute_kv_bf16() 中得到的 kv 可能不是 contiguous
```

但后续路径可能默认它是 contiguous：

```text
fused_norm_rope_inplace()
cross-rank all-gather
MUSA fused kernel
```

如果 tensor stride 不连续，而 kernel 按连续内存访问，就会读错数据。

这类问题在 CP 下更容易出现，因为 CP 会进行切分、重排、all-gather，tensor layout 更复杂。

##### 怎么定位的？

1. **问题集中在 CP 路径**

   非 CP 正常，CP 报错，说明问题和跨 rank KV 处理有关。

2. **检查 `_compute_kv_bf16()`**

   发现 `kv` 在进入：

   ```python
   fused_norm_rope_inplace(...)
   ```

   之前没有保证 contiguous。

3. **检查 tensor stride**

   如果打印或断言：

   ```python
   kv.is_contiguous()
   ```

   可能发现某些 CP case 下是 `False`。

4. **结合 fused kernel 假设**

   MUSA fused norm rope kernel 很可能按 contiguous layout 访问 KV，因此非 contiguous 输入会造成错误。

##### 怎么修复的？

在 `_compute_kv_bf16()` 中，调用 `fused_norm_rope_inplace()` 前增加：

```python
kv = kv.contiguous()
```

也就是强制把 KV 转成连续布局。

##### 修复后有什么影响？

正面影响：

1. **CP path KV layout 稳定**

   后续 fused norm rope 和 all-gather 都拿到 contiguous tensor。

2. **减少 stride / layout 相关错误**

   避免 kernel 按 contiguous 访问非 contiguous tensor。

3. **MUSA CP prefill 更稳定**

   特别是 NSA prefill-CP 场景。

可能副作用：

- `kv.contiguous()` 可能引入一次额外拷贝。

影响是：

```text
多一点内存拷贝成本
换取 fused kernel / CP all-gather 正确性
```

如果上游本来就是 contiguous，这个调用通常不会产生真实拷贝。

### 7.5 MUSA 修复总结

这些 MUSA 修复可以分成三大类。

#### A. 防止跑错路径

代表提交：

```text
5fcb7c439
```

核心是：

```text
不要 silent fallback
不要 MUSA tensor 进入 CUDA-only path
不要 graph capture 下 fallback 到 torch
```

价值是保证 MUSA fast path miss 能被立即发现。

#### B. 保证 MUSA runtime / 多卡初始化正确

代表提交：

```text
ef4469cb5
```

核心是：

```text
MCCL process group 显式绑定 local MUSA device
MUSA ops forwarding exports 完整
```

价值是保证多卡 serving 稳定。

#### C. 修复 MUSA / DSV4 prefill kernel 与 CP metadata 正确性

代表提交：

```text
3cceb08b9
81fc824a2
b9b202c28
5a15cde85
```

分别解决：

```text
TileLang JIT cache collision
prefill page write block_id re-bind
CP round-robin local/global q metadata mismatch
CP bf16 KV non-contiguous layout
```

价值是保证 DSV4 MUSA prefill / decode / CP 路径的正确性和稳定性。

## 8. 当前缺口

本次只做静态代码与提交历史梳理，没有在 MUSA环境实际跑 benchmark。因此：

- 已确认优化项、dispatch 条件、benchmark 预算和输出字段。
- 未确认当前机器上的真实 median ms / GB/s / E2E tokens/s。
- 仓库中未发现已提交的 benchmark 结果日志。

如果要补齐“优化后的实际性能”，建议在对应 MUSA 环境上运行第 4 节中的 benchmark，并把 JSONL 结果补到本文档后面。
