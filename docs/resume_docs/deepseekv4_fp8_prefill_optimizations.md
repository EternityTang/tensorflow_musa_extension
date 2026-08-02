# DeepSeek V4 FP8 MUSA Serving：官方能力、MUSA 移植与 S5000 性能重构

> 本文审查两个代码库：
>
> - 官方当前基线：`/home/eilan/combine/sglang`，`main@5c6635d8f3`（2026-07-30）；
> - MUSA 历史库：`/home/eilan/combine/sglang-musa`。其工作树停在较早的 detached HEAD，本文涉及的 2026 年 5～6 月特性必须按 `git show <commit>:<path>` 阅读，不能用当前工作树是否存在文件来判断功能是否存在。
>
> 本文只把 MUSA 实际修改过的移植、调优、Kernel/Pipeline 重写和系统适配计入成果。官方算法、接口或后来进入官方主线的实现保留为比较基线，不计为 MUSA 从零自研。

## 1. 审查与归属口径

### 1.1 双时间基线和四份代码快照

每个优化点必须同时回答两个不同问题：

1. **开发时官方基线**：MUSA 开发该特性时，上游是否已有同等语义或实现？
2. **当前官方基线**：截至 `sglang@5c6635d8f3`，上游是否后来加入了相同或类似实现？

具体对照四份代码：

| 快照 | 用途 |
| --- | --- |
| MUSA 特性提交的父版本 | 确认“修改前原来是什么”，避免只看提交标题 |
| 特性开发时可定位的官方提交 | 判断是上游移植、并行开发还是平台首发 |
| MUSA 特性提交 | 确认新增 Kernel、布局、guard、fallback 和测试 |
| 当前官方 `main` | 判断现在能否收敛、复用或删除重复分支 |

MUSA 私有父提交通常不在官方对象图中，因此不能机械地以父 hash 在官方仓库中查找同名提交。本文使用共同上游提交、提交日期、`git log -S/-G`、函数调用链和 patch 内容共同判定。当前官方后来合入的功能标记为“后续上游合入”，不能倒推为 MUSA 开发时官方早已有完整实现。

### 1.2 五类归属标签

| 标签 | 定义 | 是否计入 MUSA 工作 |
| --- | --- | --- |
| `O`：官方原生 | 官方已经提供算法、接口或完整实现，MUSA 未作实质修改 | 只作基线，不计成果 |
| `P`：功能移植 | 保留官方语义，解决 MUSA 编译、设备、dtype、layout 或正确性 | 计入平台适配 |
| `T`：平台调优 | Kernel 结构基本不变，调整 tile、warp、stage、vector 或阈值 | 计入平台优化 |
| `R`：平台重写 | 保留语义，但为 S5000 重写 Kernel、数据组织、融合或 dispatch | 计入性能重构 |
| `S`：系统适配 | Graph、Scheduler、通信、PP/PD、Runtime、fallback 或 serving 状态闭环 | 计入系统工作 |

同一特性可以是混合归属。例如 Sparse Prefill 是 `O`（FlashMLA sparse 接口）+ `P`（MUSA 接入）+ `R`（Seq-Pack）+ `S`（跨层 cache/CP metadata）。删除的是“把官方能力算作 MUSA 发明”的表述，不是删除基于上游能力完成的 S5000 二次优化。

### 1.3 证据等级

| 等级 | 可写结论 |
| --- | --- |
| `A` | 代码、测试和可复现实测配置齐全；可写明确性能数字 |
| `B` | 有用户/项目结果，但硬件、shape 或 baseline 配置待补；数字必须带限定 |
| `C` | 有 benchmark/correctness 框架和代码路径；只写结构性收益或验证目标 |
| `D` | 只有代码静态分析或 Roofline 推导；不得写成 profiler 实测 |

本文把“代码事实”“硬件推导”和“实测结果”分开。没有日志时不使用“观察到”“实测证明”等措辞。

### 1.4 明确不计为 MUSA 自研的基础能力

- DeepSeek V4 模型语义和服务框架；
- Expert-contiguous token layout、`m_indices`、DeepGEMM contiguous grouped GEMM；
- `flash_mla_sparse_fwd` 和 Sparse Attention 数学语义；
- FP8 per-token/group quant、SwiGLU/SiLU、C4/C128 压缩语义；
- TopK/routing、MHC、Paged Attention、MTP/EAGLE、Sampling、PP/PD 的上层语义；
- 官方 CUDA/NVIDIA Kernel 或双方共享且 MUSA 未修改的提交。

例如官方 `c2942907d5`（2025-04-22）已经接入预编译 DeepGEMM，`acc816d8a2`（2025-05-08）已经支持 DeepEP normal + DeepGEMM contiguous；所以 Expert-Contiguous 绝不能整体归为 MUSA 自研。

## 2. 逐项审查总览

| 优化点 | 开发时/当前官方基线 | MUSA 实质工作 | 归属 | 关键提交 | 证据 |
| --- | --- | --- | --- | --- | --- |
| S5000 Fixed-Bucket Compact MoE Prefill | 官方已有 contiguous grouped GEMM、`m_indices`、permute；当前仍有完整 DeepGEMM runner | MUSA runner/permute bring-up；static-cap/compact quant scatter、容量与 overflow、prefill/decode guard | `O+P+R+T` | `cc5b01a55`、`d2919327a`、`57832a04d`、`594d4950f` | `B/C` |
| Fused Gate Candidate 与 Routing | 官方已有 routing/TopK；当前实现更完整 | MUSA candidate、routing JIT 与数据布局衔接 | `O+R` | `4112a8f34`、`0d67c53c9` | `C` |
| TopK V1/V2/TopK512 | 官方已有通用/专用 TopK 语义 | MUSA JIT/CUH Kernel、shape mapping、exact/non-exact gate | `P+T+R` | `b31ab813e`、`01ed296dd`、`84d7beb65` | `C` |
| FP8 Prefill Quant | 官方已有 group quant；官方 `d0913fca8` 已有 fused SiLU+clamp+FP8 quant | MUSA TileKernels 接入、16K-group dispatch、TileLang half-warp 和 Triton multi-group | `O+P+T+R` | `78be4b50a`、`abe778217`、`f393eaa98` | `B/C` |
| C4/C128 Compress | 官方定义压缩语义 | MUSA vector write、C128 parallel reduce 与 shape profile | `P+T+R` | `f393eaa98`、`a18fcbf07` | `B/C` |
| Sparse Prefill Seq-Pack | `flash_mla_sparse_fwd`/chunk cache 后由官方 `93173b27e8`（2026-06-03）合入 | MUSA 在 2026-05-28 提交 Seq-Pack、flat workspace、index rebase，后续 Pack8/padding | `O+P+R+S` | `d5fe7707c`、`1ce2e4aae`、`373af4d99` | `B/C` |
| Cache Store 双路径 | 官方有 KV/cache store 语义 | decode x4/vec2 与 prefill tile-parallel/subwarp 分流 | `P+T+R` | `7c992a2a8`/`c2eafb495`（同 patch） | `B/C` |
| Norm/RoPE/Cache 与 MHC | 官方有语义；`2f0686712` 是双方共有上游提交 | MUSA materialization 消除、fused cache、prenorm、mixed handoff | `O+P+R+S` | `d65f0b4b2`、`5eab15edc`、`ea3e93c38` | `C` |
| Decode/Paged/MTP/Sampling | 官方提供框架和基础 Kernel | Queue4/prefix-tail/Page32/Head256、MUSA small-M、paged 闭环、graph metadata、sampling backend | `O+P+T+R+S` | `8094ab938`、`17be8f1bf`、`81fc824a2`、`22023f1df` | `B/C` |
| Online C128/PP/PD/HiSparse | 当前官方已有 Online C128、PP/PD、HiSparse；开发时间存在上游与 MUSA 并行演进 | MUSA C128/MTP 状态、PP4 graph/通信、TileLang offload | `O+P+R+S` | `66464a505`、`1dbba2604`、`6e7f541ae` | `C/D` |
| Runtime Hardening | 官方有 backend/JIT/fallback 框架 | MUSA fail-closed、JIT key、MCCL device、facade dispatch | `P+S` | `1b6399413`、`3cceb08b9`、`ef4469cb5` | `C` |

重复提交不重复计数：`7c992a2a8/c2eafb495`、`6e7f541ae/65093d411`、`90b4437a8/fecc03a86` 分别具有相同 stable patch-id；它们是同一 patch 在不同分支/历史中的副本，而不是两次独立实现。

## 3. MoE FP8 Prefill：基于官方 Expert-Contiguous 的 S5000 Fixed-Bucket Compact Pipeline

> **结论先行**：Expert-contiguous、`m_indices`、token pre/post reorder 和 contiguous grouped GEMM 是官方已有基础能力（`O`）。MUSA 的贡献分为两层：使官方 DeepGEMM/DeepEP normal permute 在 MUSA 上成立（`P`），以及为 S5000 大 Token Prefill 新增 fixed/static-cap bucket、compact quant scatter、容量/overflow 管理和 workload dispatch（`R+T`）。不能把整条 Expert-Contiguous Pipeline 写成从零自研。

### 3.1 双基线和代码证据

| 对象 | 代码/提交 | 结论 |
| --- | --- | --- |
| 开发前官方能力 | `c2942907d5`（2025-04-22）、`acc816d8a2`（2025-05-08） | 已有 DeepGEMM contiguous、`m_indices` 和 DeepEP normal 支持 |
| MUSA bring-up | `d2919327a`、`57832a04d`、`594d4950f` | 放开 MUSA tensor/normal permute 限制，属于移植兼容，不是新算法 |
| S5000 重构 | `cc5b01a55`（2026-05-26） | 新增 MUSA MoE Prefill ops/kernel、static-cap/compact scatter、benchmark 与 dispatch |
| 当前官方 | `sglang@5c6635d8f3` 的 `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py` | 仍调用 `grouped_gemm_nt_f8f8bf16_contig`；官方能力更完整不改变历史归属 |

MUSA 特性提交中的关键快照文件：

```text
cc5b01a55:python/sglang/srt/layers/moe/moe_runner/deep_gemm.py
cc5b01a55:python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/moe_prefill_kernels.py
cc5b01a55:python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/moe_prefill_ops.py
cc5b01a55:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_moe_prefill_deepgemm_benchmarks.py
```

注意这些是 commit snapshot 路径；`sglang-musa` 当前 detached 工作树较老，不能据其当前目录缺失而否认提交中的实现。

### 3.2 官方原始调用链和 Kernel

官方 contiguous 路径已经形成如下语义链：

```text
Router / TopK
  -> DeepEP normal pre-permute / token reorder
  -> contiguous FP8 activations + scales + m_indices
  -> grouped_gemm_nt_f8f8bf16_contig (Gate/Up)
  -> SwiGLU / activation quant
  -> grouped_gemm_nt_f8f8bf16_contig (Down)
  -> post-permute / combine
```

它解决了“不同 expert 的行如何连续交给 grouped GEMM”这一通用问题。`m_indices` 指示每一行所属 expert，DeepGEMM contiguous Kernel 据此执行分组矩阵乘；这些概念不是 S5000 新增。

直接复用时，需要区分两种瓶颈：

- **GEMM 主体**：大 M 时为高算术强度，规则的 expert-contiguous 行有利于 FP8 计算吞吐；
- **GEMM 周边**：count/prefix、quant/scatter、SwiGLU、scale store、post-combine 多为低算术强度，容易受 HBM 和 launch 支配。

因此“官方不适合 S5000”并不意味着官方 contiguous GEMM 设计错误；准确说法是：它提供了正确且可复用的计算基线，但没有自动消除 MUSA 平台 bring-up、碎片路由、BF16 中间数据和 prefill/decode 共用策略的成本。

### 3.3 修改前 MUSA 路径的问题

`cc5b01a55` 的父版本已经有官方语义和部分 MUSA 接入，但缺少大 Prefill 专用闭环：

| 原路径问题 | 代码级后果 | S5000 影响 |
| --- | --- | --- |
| normal pre/post permute 的设备检查过严 | MUSA tensor 无法进入预期 DeepGEMM 路径 | 可能 fallback 或无法复用 contiguous runner |
| expert token 数不规则 | 每个 expert 的 M 破碎，padding/调度难稳定 | FP8 GEMM 吃不满，前处理占比升高 |
| quant 与 scatter 分开或输出布局不直达 GEMM | 生成额外 FP8/scale 临时张量并再次搬运 | 增加 HBM Bytes 和 launch |
| 大 Prefill 与 tiny-M Decode 共用决策 | compact/prefix/bucket 固定成本侵入 Decode | TPOT 可能回退 |
| bucket 容量无显式边界 | 热 expert 可能超过静态容量 | 必须检测 overflow，不能静默截断 |

这里“HBM/launch 更敏感”是基于 S5000 高脊点的 Roofline 推导（`D`），不是仓库中自带 profiler 结论；实际幅度仍需 MUSA 设备数据确认。

### 3.4 S5000 新路径：保留什么、重写什么

```text
保留 O：TopK 语义、expert-contiguous、m_indices、DeepGEMM grouped GEMM
移植 P：MUSA tensor/device、FP8 scale layout、normal pre/post permute、JIT 调用
重写 R：count/prefix -> fixed/static-cap bucket -> quant+scatter -> compact rows
调优 T：token threshold、groups_per_cta、post-combine 选择、overflow fallback
```

#### 3.4.1 Fixed/Static-Cap Bucket 与容量契约

新路径维护：

```text
src2dst
valid_routes
padded_valid_rows
allocated_rows
worst_case_rows
cap_per_expert
m_indices
compact_input / compact_scale
overflow flag
```

其目的不是重新发明 expert-contiguous，而是在进入官方 contiguous GEMM 前，把大 Prefill 的路由结果按固定容量组织：正常 case 用更规则的 compact rows；热点 expert 超过容量时显式 fallback。`padded_valid_rows`、`allocated_rows` 和 `worst_case_rows` 必须分开记录，否则 benchmark 可能只看到较小有效行数，却忽略真实 workspace/padding 成本。

#### 3.4.2 Compact Quant Scatter Kernel

`moe_prefill_kernels.py` 新增的 `_tilelang_moe_deepgemm_*quant_scatter_kernel` 保持 `group_size=128`，以 warp/group 映射把以下步骤合并：

```text
读取原 token BF16
  -> 组内 amax / scale
  -> FP8 cast
  -> 按 src2dst 直接写 compact_input
  -> 写 compact_scale
```

这减少“先量化到原顺序，再单独 permute/scatter”的一次中间落盘。它只融合 **quant + scatter**；Gate/Up GEMM、SwiGLU 和 Down GEMM仍是独立阶段，不能误写成一个端到端全融合 Kernel。

#### 3.4.3 SwiGLU + Quant 与 Masked Rows

官方已经有通用 fused SiLU/SwiGLU + FP8 quant 能力；MUSA 的工作是将其接到本地 JIT/TileKernels 路径并处理 compact padding。`415346d8c` 保持 masked SwiGLU quant 走 JIT，是为了保证 padding 行不污染 scale/输出，归属是 `P+R`，不是宣称激活量化算法本身由 MUSA 发明。

#### 3.4.4 Fused Gate Candidate

`4112a8f34` 的 `moe_fused_gate_ops.py` 是候选路径：尝试减少 Gate、routing weight 和 expert index 之间的中间张量。它有独立 correctness/benchmark 测试，正文应称 Candidate/Experimental；在 exactness、负载均衡和端到端收益未完整记录前，不能写成默认生产路径。

### 3.5 Dispatch、fallback 与验证边界

`cc5b01a55` 中可直接核验的 guard：

```python
SGLANG_DSV4_MUSA_MOE_EXPERIMENTAL = 1
_DSV4_MUSA_COMPACT_MOE_MIN_TOKENS = 8192
_TILEKERNELS_SWIGLU_QUANT_MIN_ROWS = 1024
```

- `hidden_states.shape[0] >= 8192` 才尝试 compact Prefill；
- SwiGLU quant 行数小于 `1024` 时不进入大行数 TileKernels 路径；
- `group_size`、dtype、contiguous、scale shape、top-k 必须满足 Kernel 契约；
- static cap overflow 必须回退，不能丢 route；
- Decode/small-M 继续走低固定成本路径。

对应 benchmark 覆盖 device/host 时间、固定 cap、worst-case rows、prefix、scatter、两段 GEMM、post-combine 和 overflow。正式结果至少应给出 Token、expert/top-k、TP/EP、cap、padding ratio、baseline/optimized commit、median/P95 和 fallback 次数。

### 3.6 效果和最终归属

用户提供的口径为 MoE 算子约 `1.6×`、端到端 Prefill 约 `+8%`，当前只能标为 `B`：已有结果，但仓库文档没有同时给出完整硬件、shape 和运行参数，不能冒充本文复现实测。

最终成果名：

> **基于官方 Expert-Contiguous Grouped GEMM 的 S5000 Fixed-Bucket Compact MoE Prefill Pipeline（`O+P+R+T`）**

其中 `O` 仅作基线；MUSA 可计工作是 bring-up、compact quant scatter、容量/overflow、dispatch 和 masked-row 兼容。

### 3.7 源码级执行流程

下面按 `DeepGemmRunnerCore` 的执行顺序展开，主要证据为 `cc5b01a55:python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`。

#### 阶段 0：Dispatch 判定

`_should_use_dsv4_musa_compact_prefill_deepgemm` 不只看 token 数，而是联合设备、实验开关、forward mode、输入形状和 runner 能力：

```python
use_compact = (
    is_musa
    and SGLANG_DSV4_MUSA_MOE_EXPERIMENTAL
    and hidden_states.ndim == 2
    and hidden_states.shape[0] >= 8192
    and 当前为 extend/prefill 语义
    and DeepGEMM contiguous runner 可用
)
```

任一条件不满足即返回原 `DeepGemmRunnerCore` 路径，不先分配 workspace。回退发生在 count/prefix/scatter 前，因此 Decode 不承担 compact 固定成本。

#### 阶段 1：Route Count、Prefix 和 `src2dst`

设 `T=token 数`、`K=top-k`、`E=本地 expert 数`、`B=DEEPGEMM_BLOCK_M`。路由展开后最多有 `T×K` 个 route；按 expert 计数后：

```text
padded[e] = ceil_div(count[e], B) * B
padded_valid_rows = Σe padded[e]
allocated_rows(static-cap) = E * cap_per_expert
```

| 字段 | 含义 | 错用风险 |
| --- | --- | --- |
| `valid_routes` | 命中本地 expert 的 route 数 | 只看它会漏掉 padding 成本 |
| `padded_valid_rows` | 按 expert/GEMM block 对齐后的有效容量 | 不能直接代表实际分配 |
| `allocated_rows` | compact buffer 实际行数 | 与有效行混淆会高估利用率 |
| `worst_case_rows` | 极端不均衡路由的审计上界 | 用于常规分配会浪费显存 |

`src2dst[t,k]` 保存 route 到 compact row 的映射；`m_indices[row]` 保存 compact row 所属 expert，供官方 contiguous GEMM 使用。前者是 token→row，后者是 row→expert，不能互换。

#### 阶段 2：Compact Quant Scatter 契约

`try_moe_deepgemm_compact_quant_scatter_tilelang_musa` 的输入约束：

| Tensor | Shape | Dtype/约束 |
| --- | --- | --- |
| `hidden_states` | `[T,H]` | BF16/FP16/FP32，MUSA，contiguous |
| `topk_ids` | `[T,K]` | INT32，contiguous |
| `route_ranks` | `T*K` elements | INT32，contiguous |
| `src2dst` | `[T,K]` | INT32，contiguous |
| `offsets` | 至少 `E` elements | INT32，contiguous |
| `compact_input` | `[allocated_rows,H]` | FP8 E4M3FN，contiguous |
| `compact_scale` | `[allocated_rows,H/128]` | FP32，contiguous |

同时要求 `H % 128 == 0`、`group_size == 128`。不满足时 wrapper 返回 `False`，不会在 Kernel 内偷偷转换 dtype/stride。

Kernel 映射为：

```text
threads       = groups_per_cta * 32
group_linear  = block_id * groups_per_cta + warp
token_id      = group_linear // (H / 128)
scale_group   = group_linear %  (H / 128)
elem_base     = scale_group * 128 + lane * values_per_lane
```

组内完成 load → abs-max reduce → scale → FP8 cast/pack；写地址是 `src2dst` 指向的 compact row。因此它融合的是 **quant + scatter**，不是整条 MoE。

#### 阶段 3～6：GEMM、激活与 Combine

```text
compact_input/scale + gate/up weight
  -> grouped_gemm_nt_f8f8bf16_contig
  -> gateup_output [allocated_rows,2I] BF16
  -> masked SwiGLU + per-group FP8 quant
  -> down_input_fp8 [allocated_rows,I]
     down_input_scale [allocated_rows,I/128]
  -> grouped_gemm_nt_f8f8bf16_contig
  -> compact_down_output [allocated_rows,H] BF16
  -> post-combine(src2dst,routing_weights)
  -> output [T,H]
```

Gate/Up、SwiGLU+Quant、Down 是三个 Kernel 边界。Masked path 防止 padded row 污染 scale；`_TILEKERNELS_SWIGLU_QUANT_MIN_ROWS=1024` 只控制中间激活 Kernel，不能与 compact 的 `8192 tokens` 条件混为一谈。

Post-combine 将 route 输出乘 routing weight 后累加回 `[T,H]`。提交注释记录某些 `m=8192` case 的 TileLang combine 慢于 Triton scatter-add，说明 dispatch 是按 shape 选择，而非“新 Kernel 永远优先”。

```text
前置 guard miss          -> 原 DeepGEMM runner
quant-scatter wrapper miss -> 稳定 quant + permute/scatter
static-cap overflow      -> 动态/原路径，不截断 route
masked row               -> masked JIT SwiGLU quant
combine shape miss       -> 已验证 scatter-add/combine
```

### 3.8 测试如何对应实现阶段

`test_moe_prefill_deepgemm_benchmarks.py` 分别构造均衡/不均衡 route、dynamic/static cap，并记录 prefix、quant scatter、两段 GEMM、SwiGLU-Quant、post-combine、host/device latency 和 overflow。正确验收顺序是：先对齐 `src2dst/m_indices/scale/output`，再比较分项 latency，最后检查 MoE module 和 Prefill；只报告 GEMM 变快不能证明 pipeline 收益。

## 4. TopK、Routing 与 C4 Indexer：从官方语义到 S5000 Hot-Shape Kernel

> **归属与双基线**：routing、TopK expert selection、weight/index 输出格式属于官方 `O`；MUSA 的 V1/V2/TopK512、shape mapping、exact/non-exact gate 和 C4 fused indexer 属于 `P+T+R`。当前官方也有多套 TopK/DSV4 实现，但不抹去 MUSA 在 2026 年 5～6 月提交中的平台实现；同样不能把整个 TopK 算法记为 MUSA 自研。

关键 commit snapshot：

```text
b31ab813e:.../deepseek_v4_musa/kernels/topk_kernels.py
b31ab813e:.../deepseek_v4_musa/ops/topk_ops.py
b31ab813e:python/sglang/srt/layers/moe/deepseek_v4_topk.py
84d7beb65:python/sglang/jit_kernel/csrc/deepseek_v4/topk_musa.cuh
01ed296dd:python/sglang/jit_kernel/csrc/deepseek_v4/topk_musa.cuh
897b18c00:python/sglang/srt/layers/attention/dsv4/indexer.py
```

原调用链与新调用链：

```text
官方/移植基线：logits -> 通用 reduce/select/sort -> topk weights/ids -> routing/indexer
S5000 热点路径：shape guard -> MUSA TopK512 或 V1/V2 -> 直接输出约定 layout
C4 indexer：indexer -> RoPE -> Hadamard 由多段 materialization 改为 guarded fused path
```

原路径为覆盖不同 `k`、token 和 expert 数保留通用分支；对固定热点 shape，这会引入额外 reduce/sort、临时 mask/index 和 launch。MUSA 新 Kernel 固定线程/归约映射并按 shape dispatch，目标是降低控制和 launch 成本；unsupported shape 回 exact stable path。`7e5f3dfb8` 的 non-exact 路径必须显式 opt-in，并用 expert id/weight、load balance 和端到端 accuracy A/B 验证，不能作为 exact TopK 的透明替代。



```text
b31ab813e Optimize DeepSeek V4 MUSA topk v2 kernels
01ed296dd Optimize DeepSeek V4 MUSA topk512 V1
84d7beb65 Add DeepSeek V4 MUSA JIT topk512 path
7e5f3dfb8 Add opt-in nonexact MUSA topk dispatch
b4018950f Optimize DeepSeek V4 MUSA topk dispatch
897b18c00 Auto-dispatch C4 indexer fused RoPE Hadamard
```

### 4.1 上游特性、官方实现与跟进判断

官方 DSV4 使用 TopK 完成 MoE Expert Routing、Sparse Indexer 和 Hash TopK。NV 路径通常使用 CUDA/JIT/通用 TopK Kernel，根据 Vocab/Expert 数和 `k` 做 Reduce、Select 或排序。

值得跟进的原因：TopK 虽不如 GEMM 重，但每层重复调用；在长 Prefill 和 MoE 模型中，Router/Indexer 的小 Kernel 会被几十层放大，并决定后续 Compact 和 Sparse Workspace 的输入质量。

### 4.2 MUSA 直接移植差异与 S5000 归因

直接复用通用路径可能出现：

```text
完整排序或多轮 Reduce
中间 Mask/Index Tensor
TopK512/1024 Shape 覆盖不足
Kernel Launch 多
Torch Fallback
```

TopK 属于 Reduce/Memory/Launch 混合型算子，难以利用 S5000 的高峰值算力；相对低带宽和高脊点意味着应减少全量数据遍历、中间 Tensor 和 Launch，而不是简单增加线程。

### 4.3 适配决策与我们的实现

```text
保留：TopK 语义、输出格式、Router/Indexer 接口
替换：通用/CUDA-only Kernel -> MUSA TopK V1/V2/TopK512
调优：按 k、Token、Batch 建立 Shape Mapping
融合：C4 Indexer + RoPE + Hadamard
治理：Exact/Non-Exact 与 Torch Fallback 显式 Gate
```

MUSA TopK V2 针对不同 Shape 选择线程和 Reduce 策略；TopK512/1024 使用专用路径，避免不必要的通用排序。Non-Exact 只允许显式 opt-in，不能默认替代 Exact Routing。

C4 Indexer 路径将：

```text
Indexer -> RoPE -> Hadamard
```

合并为一个 MUSA Fused Path，减少三段之间的中间读写。

### 4.4 NV/MUSA 对比与验证

至少对比：

```text
NV official exact TopK
MUSA direct-port/generic TopK
MUSA TopK V1/V2
MUSA non-exact（仅实验）
```

验证矩阵：TopK=8/512/1024、Token=Decode 到 32K Prefill、不同 Expert 分布、不同 input-id dtype。指标包括 latency、读取 Bytes、Launch 数、Index/Weight 一致性、Expert Load Balance 和端到端 MoE/Sparse 收益。

### 4.5 效果与反哺

仓库有独立 benchmark、Shape Mapping 和 correctness coverage，但当前没有统一实测倍数日志，按 **证据 C：Benchmark Framework/Budget** 记录，不写虚构的加速比。

反哺：

```text
支持 Shape -> MUSA TopK V2/专用 TopK512
不支持 Shape -> Exact Stable Path
Non-Exact   -> 业务显式开启并做 Accuracy A/B
Graph       -> 禁止隐式 Torch Fallback
C4 H64      -> 满足 Guard 时进入 Fused Indexer/RoPE/Hadamard
```

### 4.6 源码级 Dispatch 和算法差异

`b31ab813e:.../topk_ops.py:topk_transform_512_musa` 的输入不是传统 MoE `[tokens, experts]` TopK，而是 sparse/page transform：

| Tensor | Shape/契约 |
| --- | --- |
| `scores` | `[rows, max_seq_len]` FP32；最后一维 stride 为 1 |
| `seq_lens` | `[rows]` INT32 |
| `page_tables` | `[rows, num_pages]` INT32，且 `num_pages*page_size >= max_seq_len` |
| `out_page_indices` | `[rows, topk]` INT32；热点为 `topk=512` |
| `out_raw_indices` | 可选 `[rows,topk]` INT32 |
| `page_size` | 正整数且 2 的幂 |

选中 raw position `r` 后，输出不是直接复制 `r`，而是：

```text
logical_page  = r >> log2(page_size)
offset        = r & (page_size - 1)
physical_page = page_tables[row, logical_page]
page_index    = (physical_page << log2(page_size)) | offset
invalid       -> -1
```

#### 4.6.1 实际分支顺序

源码的优先级很重要；环境开关并非互相平行：

```text
1. page_size 非 2 的幂                         -> ValueError
2. MUSA + MATE opt-in                          -> mate candidate
3. sequential TileLang                         -> 首个默认尝试
4. topk<=32 && seq_len<=256                    -> small_n
5. seq_len>=1024 && topk=32 && rows>=32        -> two_phase
6. 各实验 env：v2/split-hist/radix/chunk/...    -> 显式候选
7. topk=512 && seq_len>512 && 未 disable        -> long select
8. graph capture + MUSA 仍 miss                 -> NotImplementedError
9. 非 graph                                    -> Torch reference fallback
```

Graph capture 中 fail-closed 是因为 Torch fallback 可能分配、同步或执行不可 capture 操作；因此 `NotImplementedError` 是 correctness/runtime 约束，不是性能优化。

#### 4.6.2 两个真实生产 Shape

`topk_shape_mapping.md` 记录了两个性质完全不同的热点：

| 场景 | `scores` | 输出 | 为什么走不同 Kernel |
| --- | --- | --- | --- |
| Prefill flattened | `[2048,64]` | `[2048,512]` | `seq_len <= topk`，实质是按有效长度搬运/映射，sequential 路径适合 |
| 长上下文 paged load | `[16,262208]` | `[16,512]` | 每行扫描 262K 分数，只保留 512，需要真正 select/radix/histogram |

这解释了为什么“TopK512”不能对应一个固定 Kernel。前者 rows 多而每行短；后者 rows 少、每行极长，瓶颈从并行 row 搬运变成单行多轮扫描和阈值候选管理。

#### 4.6.3 Exact 与 Approximate 不是一个性能档位

长行候选算法通常先做 coarse histogram，再找 threshold bin，最后 refine ties。若 tie buffer 有界，候选溢出会导致结果与 exact Torch TopK 不同。代码和 benchmark 明确区分：

- `v2_large`：四轮 byte-radix count/threshold + gather/prefix/write，保持 exact；
- `v2_large_candidate`：先 high-16 radix，再把 threshold bin 收进 bounded candidate buffer；
- safe candidate 模式：先运行 exact 输出，candidate 无 overflow 时 device-side 覆盖，因此 graph-safe 但两条路径都执行；
- `FAST_CLIP=1` 或 NV-compatible approximate：跳过 preserve fallback，只能显式启用；
- `candidate_capacity=1024` 在真实长 shape 曾失败，不能作为 exact 默认值。

`topk_transform_512_v2_musa` 在该提交中还只是 compatibility wrapper：它验证 metadata shape 为 `[(batch+1),4]`、INT32、同 device，然后仍调用 v1。CUDA v2 的 cluster-persistent scheduling 尚未实现；文档不能仅凭函数名把它描述成完整 V2 Kernel。

#### 4.6.4 负面实验也是 Dispatch 证据

仓库 benchmark 给出了为什么某些候选不能进默认路径：

| 候选 | 代码结构 | 已记录问题 |
| --- | --- | --- |
| split-radix 原型 | 11 次 launch、五次长行扫描 | exact 但 launch/scan 太多 |
| split-hist | coarse/sub histogram + scatter/refine | 多 Kernel、同步与原子开销高 |
| chunk-merge | 每 chunk 仍做 512 次 serial select | 长行呈数量级回退 |
| direct gather | 全局 atomic counter 写 greater/equal 候选 | 原子争用极重 |
| warp-hist | `match_any/popcount` 聚合 bin | 分布中同 bin 冲突不足以摊销指令 |

这比“V2 更适合 S5000”更准确：真正结论是**只有通过 exactness 和具体 shape latency 的候选才能进入 dispatch**，其余实现保留为负面基线或 env-gated 实验。

### 4.7 C4 Indexer Fusion 的边界

`897b18c00` 的 acceptance benchmark 对应的是：

```text
原链：index selection/write -> RoPE transform -> Hadamard transform
新链：满足 H64/layout/dtype guard -> fused C4 indexer/RoPE/Hadamard op
```

融合消除的是阶段间 Tensor materialization 和 launch；TopK select 本身仍由前述 Kernel 完成。验证必须同时比较 page/raw index、RoPE 数值、Hadamard 输出和 fused/unfused latency，不能只检查最终 shape。

## 5. FP8 Activation Quant 与 C4/C128 Compress：围绕 S5000 带宽重写

> **归属与双基线**：per-token/group quant、amax/scale/FP8 cast 和 C4/C128 压缩语义属于官方 `O`；官方 `d0913fca8`（2026-05-13）还已合入 fused SiLU+clamp+FP8 quant。MUSA 可计成果是 TileKernels bring-up、S5000 专用 dispatch 和 Kernel 重写（`P+T+R`），不是 FP8 quant 或 fused activation 的发明。

关键 commit snapshot：

```text
78be4b50a / abe778217:python/sglang/srt/layers/quantization/fp8_kernel.py
f393eaa98:python/sglang/srt/layers/quantization/fp8_kernel.py
f393eaa98:.../deepseek_v4_musa/kernels/compress_kernels.py
a18fcbf07:.../deepseek_v4_musa/kernels/compress_kernels.py
```

开发演进需要按提交区分：`78be4b50a` 首先接入 TileKernels，`abe778217` 根据 hidden size 加入最小 rows 选择；文档所述 `16K groups` 和 half-warp/multi-group 路径实际首次出现在后续 `f393eaa98`，不能错误归到前两个提交。


### 5.1 上游实现与跟进价值

官方 Quant 典型流程：

```text
Load Activation
  -> Per-Group Amax
  -> Scale
  -> FP8 Cast
  -> Store FP8 + Scale
```

官方 Compress 读取 KV/Score/APE，进行 C4/C128 Reduce 并写入压缩状态。两类算子算术强度都低，是典型 Memory-bound。

S5000 的 FP8 算力只有在 GEMM 输入准备足够快时才有价值；如果 Quant/Compress 被带宽卡住，后续 Tensor Core 空闲。因此值得跟进并优先减少 Bytes 和小粒度 Store。

### 5.2 原 Kernel、S5000 不匹配与新 Kernel

官方/直接移植的通用 group quant 通常让一个 program 处理一个 group：读取 128 个值、归约 amax、计算 scale、cast/store。它覆盖 shape 广，但在两端有不同问题：small-M 时 program 数少且 launch 占比高；large Prefill 时每组独立调度、scale/store 粒度和中间流量可能成为瓶颈。Generic TileKernels cast 也不是在所有 CP 大 shape 上都优于 Triton。

`f393eaa98` 的实际新路径是：

```text
total_groups = x.numel() / group_size
  < 16 * 1024 -> 保留 small-M/既有路径
  >=16 * 1024 -> MUSA Prefill 专用路径
                  ├─ TileLang half-warp：16 lanes × 8 values = group_size 128
                  │   两个 half-warp/warp；groups_per_cta=8
                  └─ MUSA Triton multi-group：按 total_groups 选择 groups_per_cta/num_warps
```

Half-warp Kernel 在一次 load/reduce 中完成 amax、scale、FP8 pack 和写回，不创建独立 amax tensor。它增加每 CTA 覆盖的 group 数并提高 store 粒度，适合大 Prefill；但 small-M 不进入该路径，以免大配置的固定成本伤害 Decode。这是有明确 guard 的 workload specialization，不是“half-warp 对所有 shape 都更快”。

Compress 也必须拆开描述：C4 使用短 reduce + vector write；C128 的归约长度更大，`f393eaa98/a18fcbf07` 使用 parallel reduce/profile 选择，避免将长窗口压到单 warp 串行关键路径。两者共用压缩语义，但并行映射不同。

| 项目 | 保留/变化 | 归属 |
| --- | --- | --- |
| quant 数学、scale layout、输出 dtype | 保留官方契约 | `O` |
| TileKernels 可用性和 dtype/layout guard | MUSA 接入 | `P` |
| 16K groups、rows/hidden dispatch | MUSA 阈值与 launch 选择 | `T` |
| TileLang half-warp、Triton multi-group | S5000 Kernel 重写 | `R` |
| C4 vector write、C128 parallel reduce | S5000 Kernel/映射重写 | `T+R` |

### 5.3 S5000 性能归因与适配决策

S5000 高脊点使 Quant/Compress 更容易长期停留在 Memory-bound 区域。决策顺序：

```text
减少访问次数
  -> 提高 Store 粒度
  -> 一个 CTA/Warp 处理更多 Group
  -> 最后再调 Warp/Tile
```

选择：

```text
保留：官方 Quant/Compress 数学语义和输出 Layout
重写：Half-Warp Group Quant、C128 Parallel Reduce
调优：16K Groups Dispatch、groups_per_cta、num_warps
替换：Large Prefill 的 Generic TileKernels Cast
保留 fallback：MUSA Triton Multi-Group
隔离：Decode Tiny-M 不走 Prefill Kernel
```

### 5.4 我们的实现

#### Half-Warp Group Quant

```text
group_size=128
16 lanes × 8 values = 128
1 half-warp / group
2 groups / warp
```

融合：

```text
Amax Reduce + Scale + FP8x4 Pack + stg32 Writeback
```

#### Shape-Aware Dispatch

```python
_MUSA_PREFILL_FP8_QUANT_MIN_GROUPS = 16 * 1024
```

```text
>=16K Groups -> Half-Warp 或 MUSA Triton Multi-Group
<16K Groups  -> Small-M/Decode Path
```

#### MUSA Triton Multi-Group

根据 Rows/Hidden 选择 `groups_per_cta` 和 `num_warps`，作为大 Prefill 的稳定 fallback；禁止大 CP Prefill 落入已知较慢的 generic cast。

#### C4/C128 Compress

```text
C4   -> 轻量 Reduce + Vector Write
C128 -> Parallel Reduce + Vector Write
```

使用 logical bytes/derived GB/s 判断优化是否真正改善访存，而不仅是偶然降低一次 latency。

### 5.5 客户验证、效果与反哺

Case：Decode Tiny-M、8K/16K/32K Prefill、Large CP、BF16/FP16/FP32 输入、C4/C128。验证 Quant-Dequant 误差、Scale Layout、GB/s、MoE/Projection 输入准备和 Decode TPOT。

用户提供结果：FP8 Quant/Compress 前处理约 **1.8×～2×**，按 **证据 B** 记录；需要拆分 Quant 与 Compress 的独立 Case，补齐硬件、Shape、Baseline Commit，避免把两个算子合并成不可复现的单一数字。

反哺：

```text
Groups >=16K             -> Prefill Quant
Large CP                  -> 禁止 Generic TileKernels Cast
C128 Large Reduce         -> Parallel Reduce
Unsupported Scale/Layout  -> Stable Fallback
Decode                     -> Small-M Path
```

### 5.6 Quant 的真实调用优先级与 Tensor 布局

`f393eaa98:.../fp8_kernel.py:sglang_per_token_group_quant_fp8` 先计算输出，再按如下顺序尝试 backend：

```text
输入 x contiguous，且 H % group_size == 0
  -> 预分配 x_q 与 x_s
  -> _try_musa_prefill_per_token_group_quant_8bit
       ├─ TileLang MUSA prefill
       └─ MUSA Triton multi-group
  -> _try_tilekernels_per_token_cast_musa（仅 small/medium）
  -> sgl-kernel v2 / JIT
  -> legacy group quant
```

这意味着 `16K groups` 的作用不只是“选择一个更快 Kernel”：当 `total_groups>=16K` 时，generic TileKernels helper 主动返回 `None`，防止 large CP Prefill miss 后落到已知不合适的 generic cast。

#### 5.6.1 Shape 公式

普通 quant（无 fused SwiGLU）对 `x=[R,H]`：

```text
G = group_size = 128
total_groups = R * H / G
x_q.shape = [R,H]                  dtype=FP8 E4M3
x_s.shape = [R,H/G]                dtype=FP32（row-major case）
```

若 `fuse_silu_and_mul=True`，输入最后一维包含 gate/up 两半，输出最后一维除以 2。Column-major scale、TMA-aligned scale、UE8M0、masked-M 各有不同契约；当前 TileLang Prefill helper明确拒绝这些模式，交给支持它们的后端，不能为了命中 fast path 改 layout。

#### 5.6.2 Half-Warp Kernel

`_tilelang_musa_per_token_group_quant_fp8_kernel` 的 `group_size=128` 映射为：

```text
half-warp = 16 lanes
每 lane    = 8 contiguous values
每 half-warp = 16*8 = 128 values = 1 group
每 warp    = 2 groups
groups_per_cta = 8
```

每个 group 的执行顺序：

1. 每 lane 向量 load 8 个 BF16/FP16/FP32 值并转 FP32；
2. lane 内求局部 abs-max；
3. half-warp shuffle/reduce 得到 group amax；
4. `scale=max(amax/448, eps)`；
5. 每 lane 将 8 个值 clamp/cast/pack 为 FP8；
6. 写连续 `x_q`，由一个 lane 写 `x_s[group_id]`。

没有单独的 amax Tensor，也没有第二个 cast Kernel。Wrapper 只接受 MUSA、contiguous、row-major FP32 scale、非 UE8M0，异常时返回 `False`。

#### 5.6.3 Shape-Aware Launch 与 Fallback

代码中 `_musa_prefill_fp8_quant_launch_config(rows,hidden)` 给出 Triton multi-group 配置：

| 条件 | `groups_per_cta` | `num_warps` |
| --- | ---: | ---: |
| `hidden <= 2048` | 8 | 4 |
| `hidden > 2048 && rows <= 4096` | 4 | 2 |
| 其他大 rows/hidden | 8 | 2 |

TileLang helper当前固定 `groups_per_cta=8`。两条 prefill backend 都要求 `total_groups>=16*1024`；其下才考虑 TileKernels，而 TileKernels 又通过 `_tilekernels_fp8_quant_min_rows(hidden)` 拒绝过小 rows。故完整决策维度是 `total_groups + rows + hidden + layout mode`，不是单一 token threshold。

### 5.7 SwiGLU Quant、C4 与 C128 不是同一个 Kernel

| 路径 | 入口/核心符号 | 数据语义 | 关键边界 |
| --- | --- | --- | --- |
| 普通 group quant | `sglang_per_token_group_quant_fp8` | `[R,H] -> FP8 + scale` | 可选多种 scale layout |
| fused SwiGLU quant | `silu_and_mul_contig_post_quant_musa` | `[R,2I] -> [R,I] FP8 + scale` | 同时做 SiLU×up，不等于 MoE 全融合 |
| masked post-quant | `silu_and_mul_masked_post_quant_musa` | 只处理有效 expert rows | padding/masked-M 是正确性契约 |
| C4 page reduce | `_tilelang_compress_ratio4_prefill_page_reduce_cached_kernel` | 页内 C4 状态 reduce/cache | 短 reduce、page/cache layout |
| C128 reduce | C128 parallel reduce/profile path | 长窗口压缩状态 | 多执行单元协作，避免单 warp 长链 |

`a18fcbf07` 的 benchmark 通过 `_compress_c4_reduce_logical_bytes`、`_compress_c128_reduce_logical_bytes` 等函数估算逻辑读写量，并以：

```text
logical_GB/s = logical_bytes / device_time
```

报告带宽。这个数是按算法必须读写的数据估算的 effective bandwidth，不等同于硬件计数器的物理 HBM traffic；cache hit、重复 load 和 write combining 都可能让两者不同。

## 6. FlashMLA Sparse Prefill：从官方 Kernel 接口到 S5000 Seq-Pack

> **归属与双基线**：`flash_mla_sparse_fwd`、Sparse Attention 数学和 compressed/SWA 输入语义属于官方 `O`。时间上，MUSA `d5fe7707c` 于 2026-05-28 提交 Seq-Pack；官方共享提交 `93173b27e8` 于 2026-06-03 合入 `flash_mla_sparse_fwd` 和 `SparsePrefillChunkCache`。这说明开发期存在分支/并行演进，不能只看当前官方主线后倒推“当时 MUSA 没有二次工作”。可计 MUSA 工作是接口 bring-up、flat workspace/index rebase、Pack8/padding 和 CP/cache 状态闭环（`P+R+S`），不是 FlashMLA Kernel 本身。

关键 snapshot：

```text
d5fe7707c:python/sglang/srt/layers/attention/deepseek_v4_backend.py
d5fe7707c:python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py
d5fe7707c:python/sglang/srt/layers/attention/dsv4/metadata.py
1ce2e4aae:python/sglang/srt/layers/attention/deepseek_v4_backend.py
373af4d99:python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py
```


### 6.1 上游特性与官方 NV 路径

官方新特性使用 FlashMLA Sparse Kernel 消费：

```text
Query
KV/Compressed Cache Workspace
Per-Query Sparse Indices
Valid Length
```

官方 NV 路径依赖 FlashMLA CUDA Kernel，并需要将 Compressed Cache、SWA Window 和 TopK Indices 转换成 Kernel 可消费的格式。这项特性对 32K～128K 长上下文非常重要，因为 Dense Attention 成本过高。

### 6.2 原数据准备链、S5000 不匹配与 Seq-Pack

原 sparse 主 Kernel 并不直接消费分散的请求 cache；调用前仍需：

```text
逐 request gather compressed cache
+ gather SWA window
+ 构造 query_start / valid length
+ 合并 TopK 与 SWA indices
+ 对齐/padding
-> flash_mla_sparse_fwd
```

如果每层重做，DSV4 多层会重复分配 workspace、构造 chunk 内不变 metadata，并进行不连续 gather。这里的问题在主 Attention Kernel **之前**；所以优化归属应称“输入组织和生命周期重构”，不能称“自研 FlashMLA sparse Kernel”。

Seq-Pack 新链路：

```text
各 request compressed region + SWA region
  -> 一次 pack 到 flat BF16 workspace
  -> TopK/SWA index rebased 到 workspace 坐标
  -> -1 padding + valid length
  -> flash_mla_sparse_fwd 消费连续 workspace
```

`SparsePrefillChunkCache` 复用 local query start、SWA token ids、C0/C128 workspace/indices 和 C4 buffer；仅 layer-dependent C4 top-k 需要逐层更新。`Pack8` 还有明确 guard：query 总行数和每个 `extend_seq_len` 都需被 8 整除，且对应 pack8 symbol/metadata 存在；不满足时回非 pack8 路径，不应通过隐式 padding 改变有效 query。

CP round-robin 不是性能发明，而是正确性修复（`P/S`）：local query 不是 global query 的简单连续切片，因此必须重建 local `query_start_loc`、`extend_seq_lens`、`req_pool_indices` 和 `seq_lens`。缺少该步骤即使 Kernel 更快也会消费错位 metadata。

### 6.3 S5000 归因与适配决策

该路径主要是 Memory-bound + Launch-bound，而非 Compute-bound。选择：

```text
保留：官方 Sparse Attention/FlashMLA Kernel 接口
重写：输入数据组织为 Flat BF16 Workspace
复用：Chunk-Level Metadata/Workspace/Buffer
调优：Pack8、H64、Request Padding
修复：CP Round-Robin Local Metadata
禁用：Raw-Index Silent Fallback
```

### 6.4 我们的实现

#### Seq-Pack

```text
Request Compressed Region
+ Request SWA Region
-> Flat BF16 Workspace
```

每个 Query 的 Combined Indices：

```text
Rebased Compressed TopK
+ Rebased SWA Position
+ -1 Padding to Alignment
```

#### Chunk-Level Cache

跨约 61 层复用：

```text
local query_start
SWA token ids
c0/c128 workspace/indices
c4 output buffer
```

仅更新 layer-dependent 的 C4 TopK Indices。

#### Pack8 和 Request Padding

对满足 Tile Alignment 的 Shape 进入 Pack8/H8；不规则 Request 通过 Padding 规则化，同时维护 Valid Length 和 `-1` Indices，避免 Padding Query 参与 Attention。

#### CP Round-Robin

重建：

```text
local query_start_loc
local extend_seq_lens
local req_pool_indices
local seq_lens
```

### 6.5 NV/MUSA 和客户验证

对比：

```text
NV official FlashMLA + official preparation
MUSA direct preparation
MUSA Seq-Pack without cache
MUSA Seq-Pack + Chunk Cache + Pack8
```

Case：8K/32K/128K、C0/C4/C128、SWA、CP1/CP多卡、Round-Robin、Mixed Batch。指标：Workspace Build、Metadata Build、Gather、Sparse Kernel、Layer 累计、TTFT、显存和 CP Correctness。

### 6.6 效果与反哺

用户提供 Sparse Attention 前处理及 Metadata 构建约 **2×**，按 **证据 B** 记录；不将其等同于完整 Sparse Attention Kernel 或端到端 2×。

反哺：

```text
Chunk 内不变 Metadata -> 必须缓存
C0/C128               -> 预计算
C4                     -> 每层更新 Index，复用 Buffer
CP Round-Robin         -> Local Metadata
Pack8                   -> Alignment Guard + Padding
Raw Index Miss          -> 不允许 Silent Fallback
```

### 6.7 `SparsePrefillChunkCache` 的字段生命周期

`d5fe7707c:.../sparse_prefill_utils.py` 的 docstring 明确把字段分成 chunk-invariant 和 layer-dependent：

| 字段 | Shape | 生命周期 |
| --- | --- | --- |
| `seq_lens` | `[num_reqs]` INT32 | chunk 内固定 |
| `query_start_loc` | `[num_reqs+1]` INT32 | 由 `cumsum(extend_seq_lens)` 一次构造 |
| `swa_token_ids` | `[total_swa]` INT32 | chunk 内固定 |
| `swa_first_pos/gather_lens` | `[num_reqs]` | chunk 内固定 |
| `swa_offsets` | `[num_reqs+1]` | chunk 内固定 |
| `c0_combined_indices/lens` | `[num_q,aligned_width]` / `[num_q]` | TopK=0，可预计算 |
| `c0_workspace` | `[total_swa,1,512]` BF16 | 跨层复用 |
| `c128_*` | fixed positional layout | 可预计算并复用 |
| `c4_workspace/bases` | flat compressed+SWA layout | buffer/base 复用 |
| `c4_combined_indices` | `[num_q,aligned_width]` | tail 常驻 `-1`，有效 prefix 逐层覆盖 |

SWA gather 长度不是简单 `min(seq_len,window)`，而是：

```text
swa_gather_len = min(seq_len, extend_len + window - 1)
swa_first_pos  = seq_len - swa_gather_len
```

因为一个 prefill chunk 内所有 query 的滑窗并集，要覆盖最早 query 的左边界到最后 query。`build_swa_token_ids` 再执行：

```text
(request, seq_position)
  -> req_to_token[req_pool_index,seq_position]   # full KV id
  -> full_to_swa[full_kv_id]                    # physical SWA id
```

`total_swa=int(swa_offsets[-1].item())` 会产生每 chunk 一次 CPU sync；cache 的意义之一就是不让这个 sync 每层重复。

### 6.8 Combined Index 的精确坐标变换

`combine_topk_swa_indices` 输入的 `topk_indices[T,K]` 已经是 **request-local compressed region** 坐标。输出宽度为：

```text
combined_width = ceil_align(topk + window_size, 128)
combined_indices.shape = [num_tokens,combined_width]
combined_lens.shape = [num_tokens]
```

以两个 request 为例：

```text
request 0: compressed_base=0,   swa_base=40
request 1: compressed_base=80,  swa_base=116
query 属于 request 1
local compressed topk = [2,9,-1]
local SWA positions   = [0,1,2,3]
```

重映射后有效 prefix 为：

```text
compressed: [80+2,80+9]
SWA:        [116+0,116+1,116+2,116+3]
combined:   [82,89,116,117,118,119,-1,...]
combined_len = 6
```

`-1` 只作 sentinel；`combined_len` 才是每行真实有效长度。复用预分配 buffer 时 Kernel 只覆写有效 prefix，所以调用者必须保证 tail 预先填 `-1`，并保证复用期间有效 prefix 长度契约不变。

`query_start_loc` 可以处于跨 chunk 的 global 空间；Kernel 通过减 `query_start_loc[0]` rebasing。CP round-robin 下 local query 顺序不是 global 连续片段，必须先重建 local start/length/request mapping，不能仅做该减法。

### 6.9 Pack8 的两类 Metadata

`1ce2e4aae` 新增：

- `pack_row_level_indices_for_pack8`：把普通逐 query row index/lens 打包；
- `pack_structured_dsv4_indices_for_pack8`：利用 DSV4 compressed+SWA 的结构化区间生成 pack metadata。

进入 Pack8 至少要求：

```python
q.shape[0] % 8 == 0
all(extend_seq_len % 8 == 0 for each request)
flash_mla_sparse_fwd_pack8 is not None
相应 c0/c128 pack indices、lens、row mask 已构造
```

Row mask 保证 padding lane 不贡献 Attention；pack lens 保证每个 pack 内 query 仍读取自己的有效 prefix。`373af4d99` 的 request padding 是为了将 request 边界调整到 Pack8 可表达的形状，并同时维护 valid length/mask，不是简单向 `q` 末尾补零。

### 6.10 调用链和验证断点

```text
DeepSeekV4 attention backend
  -> SparsePrefillChunkCache.build（每 chunk 一次）
  -> dequant compressed/SWA 直接写预分配 workspace
  -> combine_topk_swa_indices（C0/C128 预计算；C4 更新 prefix）
  -> [optional] structured/row-level Pack8
  -> flash_mla_sparse_fwd[_pack8]
```

建议逐断点验证：workspace 每 request offset 不重叠；combined index 全在 workspace 范围或为 `-1`；`combined_lens` 与有效 prefix 相符；C0/C128 第二层不重建；C4 只改有效 prefix；CP1 与 CP round-robin 输出匹配 reference；Pack8 与非 Pack8 在有效 query 上数值一致。

## 7. Cache、Norm、RoPE 与 MHC：官方语义上的 Workload 分流与 Materialization 消除

> **归属与双基线**：KV/cache store、Norm、RoPE 和 MHC 数学/状态语义是官方 `O`。`2f0686712` 在官方与 MUSA 仓库中是同一 hash 的共享上游提交，包含 DeepGEMM、fused norm、fused HC head，不能列为 MUSA 自研。MUSA 可计工作是 cache/store backend 接入、prefill/decode Kernel 分流、layout/materialization 消除、fused norm-rope-cache、prenorm 和 mixed handoff（`P+T+R+S`）。

关键 snapshot：

```text
7c992a2a8 / c2eafb495:.../deepseek_v4_musa/{kernels,ops}/cache_*.py  # 同 patch
d65f0b4b2:.../deepseek_v4_musa/{kernels,ops}/norm_rope_*.py
5eab15edc:python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_flashmla_musa.cuh
ea3e93c38:.../deepseek_v4_musa/{kernels,ops}/mhc*.py
1b03ac81e:.../deepseek_v4_musa/kernels/mhc_kernels.py
```


### 7.1 上游能力和官方 NV 路径

官方提供 KV Cache Store、Norm、RoPE、MHC 和相关 Fusion 语义。NV 路径通常针对 CUDA 数据布局和 NVIDIA Warp/Memory Primitive 调优。它们值得跟进，因为每层、每 Token 都会调用，是 S5000 高脊点下典型的 Memory/Launch Tail。

### 7.2 MUSA 直接移植差异

常见问题：

```text
Decode/Prefill 共用一套 Store Kernel
为满足 Kernel Layout 额外 contiguous/materialize
FP8 nope、BF16 rope、scale 分散写回
Norm/RoPE/Cache 分成多个 Kernel
MHC Prefill Split 误用于 Decode/Mixed Batch
```


在 S5000 上，大 Prefill 的总 Bytes 和 Decode 的固定 Launch 成本分别成为瓶颈，不能用单一 Kernel 解决。

### 7.3 适配决策总览

| 子路径 | 官方语义 | S5000 决策 |
| --- | --- | --- |
| Cache Store | 写 KV/Scale/RoPE | Decode/Prefill 分流 |
| Norm/RoPE | 独立或部分融合 | 消除 Materialization、融合 Cache |
| Compress Norm/RoPE | 前处理链 | Guarded Fusion |
| MHC | Pre/Post/Prenorm | Shape Split、Mixed Handoff、Prewarm |

### 7.4 FlashMLA Cache Store 双路径

原路径若让 Decode 和 Prefill 共用大而通用的 pack/store Kernel，会在两个方向妥协：Decode 的 1～几十行承担大 tile/shared-memory 固定成本；Prefill 的成千上万行又缺少足够的跨 token 并行和向量写回。MUSA commit snapshot 中的 dispatch 明确以 workload 分流，而不是单纯替换同名 Kernel：

```text
Decode / small rows
  -> block-per-token decode_x4 / decode_x4_fp32 / vec2
  -> 优先低固定延迟；检查 dtype、contiguous 和 page_size

Prefill / input.shape[0] >= 128
  -> prefill_subwarp16 或 prefill_tile_parallel
  -> 多 token/CTA、NoPE 分 tile、向量化 rope/scale/cache store
  -> 编译或 shape miss 时回稳定路径并可 trace reason
```

Indexer cache 同样以 `128` rows 为 prefill x8 guard。代码还针对大 shape 选择 threads/profile，并检查 page size 为正的 2 次幂、index dtype/shape 与 cache layout。因而“prefill 更快”的原因是提高并行和写回粒度，“decode 更快”的原因是避免大配置固定成本；不能笼统写为一个 cache Kernel 同时优化两者。


### 7.5 Materialization 消除与 Fusion

扩展 MUSA Kernel 对 Stride/Layout 的支持，减少：

```text
contiguous copy
Norm Output Tensor
Cache Temporary Tensor
跨 Kernel HBM Round-Trip
```

Fused Norm-RoPE-Cache 将 Norm、RoPE、Pack 和部分 Cache Store 合并。Compress Fused Norm/RoPE 使用 Hidden、Rope Dim、Dtype、Layout 和 Inplace Guard，不支持时稳定 fallback。

### 7.6 MHC 的 S5000 适配

官方 MHC Pipeline 的 DeepGEMM、Fused Norm 和 HC Head 作为上游能力；MUSA 侧补齐：

```text
MHC Prenorm Backend
Prefill/Decode Shape Split
Mixed Prefill Handoff
Token-Count Prewarm
HC Head/Cache Kernel
```

目标是避免首次 JIT 抖动和 Prefill 大配置污染 Decode TPOT。

### 7.7 NV/MUSA 与客户验证

对 Cache 分别测 1～64 Decode Token 和 8K～32K Prefill；对 Norm/RoPE 测 Materialize/Fused；对 MHC 测 Decode、Prefill、Mixed Batch 和 Graph Replay。

指标：Logical Bytes、GB/s、Launch 数、临时显存、Page Boundary、Invalid Index、TPOT、P95 和 Graph Stability。

### 7.8 效果与反哺

用户提供 Prefill Cache Store 约 **1.8×**，按 **证据 B** 记录；Norm/RoPE/MHC 没有统一实测倍数，按 **证据 C/D** 记录结构性收益。

反哺：

```text
Decode Small-M       -> x4/vec2
Prefill Large-M      -> Tile-Parallel/Subwarp
Layout 可直接消费    -> 禁止 Materialize
支持 Shape           -> Fused Norm-RoPE-Cache
Mixed/Decode MHC     -> 专用 Split
首次常用 Shape       -> Prewarm
```

### 7.9 Cache Page Layout 与 Kernel 分支细节

以 FlashMLA cache store 为例，逻辑输入不是普通 KV `[T,H]`。代码验证的主要布局为：

```text
NoPE quant payload : 448 FP8 bytes/token
RoPE payload       : 64 BF16 values/token = 128 bytes
scale              : 7 groups/token，按 UE8M0/byte 形式存储
index              : INT32 physical slot
```

`page_index_and_offset(index,page_size)` 将 physical slot 拆成 page 和页内 offset，再根据 `cache.shape[1]` 计算 NoPE、RoPE、scale 区域地址。因而 page size、每页字节数、scale offset 都是 Kernel ABI；不能在 wrapper 中改一个 page_size 而继续复用旧 JIT binary。

#### Indexer Cache 分支

`_try_tilelang_pack_store_indexer_cache_musa` 的 guard 是：input/cache/index 同 MUSA device；input 为 `[T,128]` FP32/BF16 且最后一维 contiguous；cache 为 contiguous UINT8 2-D；indices 为 `[T]` INT32 contiguous。随后：

```python
use_vectorized_pack = page_size > 0 and cache.shape[1] % 4 == 0
use_prefill_x8 = use_vectorized_pack and T >= 128
```

- `T<128`：decode x4，每个 token 独立，降低固定开销；
- `T>=128`：prefill x8，half-warp/token，一次覆盖更多 token；
- 不满足 vector/layout：返回 `False`，由 JIT/稳定实现接手。

#### FlashMLA Cache 分支

`_tilelang_pack_store_flashmla_cache_musa` 先验证 `[T,*]` 输入、`indices[T]`、contiguous、dtype 和 page layout，再按顺序尝试：

```text
显式 pack_impl             -> 用户指定 decode candidate
T <= decode_max_tokens     -> decode_x4 / decode_x4_fp32
T >= 128                   -> prefill subwarp16
大且满足 tile 条件          -> prefill tile-parallel
编译/shape miss            -> trace 原因并回稳定 no-copy/JIT 路径
```

FP32 decode x4 还限制 page size（提交中为 `{2,64}`）并要求 contiguous。Prefill tile-parallel 将 7 个 NoPE quant tile 分成连续 `4+3` 组，多个 half-warp 协作完成 amax、scale byte、FP8 pack 和 RoPE vector store；`full_tiles` 只在 `T % tokens_per_cta == 0` 时启用，尾块必须保留 bounds guard。

### 7.10 Norm-RoPE-Cache 融合的读写账本

分解路径至少 materialize：

```text
x --RMSNorm--> norm_out
norm_out --RoPE--> q/k_rope
k_nope --quant--> fp8 + scale
fp8/scale/rope --pack--> cache page
```

融合 `fused_norm_rope_flashmla_store_musa_jit` / `v2` 保留数学顺序，但在一个调用内读取 residual/weight/freq，计算 norm 和旋转，再直接按 page layout 写 NoPE/scale/RoPE。验证重点不是“数值大致相等”而是：

- inplace tensor 是否与 reference 更新同一部分；
- strided weight/freq 是否由 Kernel 直接消费，避免隐式 `.contiguous()`；
- cache-tail 和无效 index 是否不写；
- UE8M0 scale byte 与分解 quant 相同；
- graph replay 时 cache/workspace 地址是否仍有效。

对应测试 `test_flashmla_fused_norm_rope_store_v2_matches_decomposed_tolerance` 将 fused 输出与分解实现比较；`d65f0b4b2` 的 strided-input tests 则专门证明“消除 materialization”没有偷换为 wrapper 内 copy。

### 7.11 MHC Prenorm 和 Mixed Handoff

官方共享提交 `2f0686712` 已给出 MHC DeepGEMM/fused norm/HC-head 基线。MUSA `ea3e93c38` 进一步围绕 prenorm 拆 stage/profile：BF16 `A@A`/平方和 probe 用于定位数值和内存阶段，TME workspace pass config 与 split sweep用于选择 Decode 的工作分割。它不是重新定义 MHC 数学。

`1b03ac81e` 修复的是大 Prefill fuse 与 mixed batch 的 handoff：Prefill 可走 big-fuse，但 Decode/Mixed 不能错误复用同一 workspace/shape 假设。`test_mhc_pre_big_fuse_kernel_matches_reference_on_musa` 对比 Sinkhorn/reference，`test_mhc_prenorm_x_tme_cast_stage0_handles_9216_tokens_on_musa` 固定覆盖 9216-token 边界。文档中的 prewarm 应理解为提前编译常用 `(tokens,hidden,split)` 组合，而不是改变 dispatch correctness。

## 8. Decode、Paged Cache、MTP/EAGLE 与 Sampling：官方框架的 MUSA 闭环和 Small-M 专用路径

> **归属与双基线**：Decode Attention、Paged KV、Prefix Cache、MTP/EAGLE、Sampling 和 Graph 框架是官方 `O`。MUSA 工作必须逐项计为：官方 TileLang/CUDA 路径的平台扩展（`P/T`）、S5000 small-M cache/HC/MHC Kernel（`R`）、paged/graph/metadata/sampling 闭环（`P/S`），而不是把整个 Decode 或推测解码框架列为自研。

关键 snapshot：

```text
8094ab938:python/sglang/srt/layers/attention/tilelang_unified_attention_v2_queue4_prefix_tail.py
018145486 / 4ef9320d9:python/sglang/srt/layers/attention/tilelang_unified_attention_v2.py
17be8f1bf:.../deepseek_v4_musa/kernels/{cache,hc_head}_kernels.py
81fc824a2 / eb255ea13:.../deepseek_v4_musa/kernels/compress_kernels.py（及早期 facade 路径）
6d8e20f98:python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py
22023f1df:python/sglang/srt/layers/sampler.py
```


### 8.1 上游特性和跟进价值

官方社区提供 TileLang Unified Decode Attention、Paged Cache、MTP/EAGLE Speculative Decode 和 Sampling 框架。它们直接决定 TPOT、长 Decode 和在线稳定性，必须跟进。

### 8.2 官方 NV 路径

官方 NV 路径通常依赖 CUDA Graph、CUDA/TileLang Decode Kernel、Paged KV Layout 和 CUDA Sampling/`torch.multinomial`。Decode 每步 M 很小，性能不取决于峰值 FP8 算力，而取决于：

```text
Kernel Launch
KV Read/Write
Page/Prefix-Tail Dispatch
Graph Replay
Host Synchronization
Sampling
```

### 8.3 为什么 Prefill Kernel 不能直接用于 Decode

Decode 每步 token 很少、KV 长度可能很大，瓶颈是 page/cache read、同步与 launch，而不是大 M 的 FP8 峰值。如果套用 Prefill 大 tile：有效线程不足、尾块和 padding 增加，shared-memory/寄存器配置也成为固定成本。MUSA 因此按实际 shape 扩展 Queue4、prefix-tail、page size 32、head dimension 256，并为 cache/HC/MHC 提供 small-M 路径：

```text
Queue4            -> 小批量 query 的队列映射
Prefix-Tail       -> prefix 主体与尾块分治，减少尾块空转
Page32            -> 与 paged KV 的 32-token page 契约一致
Head256           -> 专门覆盖固定 head dimension，避免 unsupported fallback
Decode x4/vec2    -> 少量 token 的低延迟 load/store
```

这些扩展保留官方 Attention 语义，属于 `P+T`；MUSA 新写的 small-M cache/HC Kernel 属于 `R`。是否更快仍需按 active token、KV length、page/head shape 和 host-inclusive latency 测量，不能只由名称推断。

Paged 路径还要求 prefill/decode 成对闭环：`81fc824a2` 使 `is_paged=True` 并修复 prefill `block_id` rebind，`eb255ea13` 补 decode paged Kernel。只开 paged prefill 而 decode 无对应布局消费会成为正确性缺口，不是可接受 fallback。

MTP/EAGLE 的 `6d8e20f98`、`4c49bdcf9`、`bb3e58863` 修改 draft graph、D2H `seq_len`、accept index、expert remap 和 extend metadata，归属为 `S/P`。`22023f1df` 将 MUSA no-seed sampling 从有问题的 `torch.multinomial` fallback 切到可用 backend，属于平台稳定性适配；seeded deterministic path 仍应保留自身契约。

### 8.4 适配决策与我们的实现

```text
保留：官方 Decode/Paged/MTP/Sampling 语义
扩展：Queue4、Prefix-Tail、Page32、Head256
重写：MUSA Decode Cache/HC Head/MHC Small-M Kernel
补齐：Paged Decode Kernel
适配：Draft Token Graph、D2H seq_len、Metadata Cache Loc
替换：MUSA No-Seed torch.multinomial -> sgl_kernel Sampling
```

MTP/EAGLE 还修复 Accept Index、Expert Remap、Draft Metadata 和 Extend Graph，使 Graph Replay 真正闭环。

### 8.5 客户验证

Case：不同 Page/Head、1～64 Active Token、4K 输入/1K 输出、并发 1/8/32/64、MTP on/off、Paged Prefill+Decode、Seeded/No-Seed Sampling。

指标：TPOT/P95/P99、Graph Capture/Replay、Kernel/Fallback Trace、Accepted Tokens、Draft Latency、Cache Correctness、Sampling Stability。

### 8.6 效果与反哺

文档已有用户给定口径：TPOT `7.3 -> 6.6 ms/token`（约 -9.6%），但仍需补齐硬件、模型、并发和 MTP 配置，按 **证据 B** 记录。各子 Kernel 未提供统一倍数，不臆造。

反哺：

```text
Page32/Head256       -> 专用 Decode Path
is_paged=True        -> 必须 Paged Decode 闭环
Graph Capture        -> 禁止 Torch Fallback
MUSA + No Seed       -> sgl_kernel Sampling
Seeded Sampling      -> 保留 Deterministic Path
Decode               -> 禁止 Prefill 大 Kernel
```

### 8.7 Queue4、Prefix-Tail 和 Page32 的控制流

`8094ab938` 将 queue4 作为 wrapper 可选择的 variant，而不是另建一套 Attention 语义：

```text
unified attention wrapper
  -> _queue4_decode_mode
  -> _select_queue4_decode_variant(shape/runtime config)
  -> _get_cached_3d_plan/workspace/kernels
  -> _run_3d_native
       ├─ RunFullTile
       ├─ RunTailTile
       ├─ RunGroup
       └─ RunQueue
```

Full tile 处理规则 KV 区间，tail tile处理不能填满 tile 的末段；Queue 将多个小 query/group 排入可复用计划，避免每个 query 单独 launch。Plan、workspace 和编译 Kernel 按 runtime config 缓存，所以验证既要覆盖首次编译，也要覆盖相同/不同 shape 的 cache hit。

Page32 不是仅放宽 `assert page_size`。`018145486` 新增 `_get_page32_logical_kv_views`，将物理 page32 cache 解释成 Kernel 所需 logical K/V view；`_choose_num_segments`/`_fallback_num_segments` 决定 split-reduce 段数，`RunFullTilePage32Aligned` 利用 page 对齐。测试矩阵同时覆盖 fused、split、queue4 mixed-tail 和 MUSA graph，说明 page view、tail 和 graph 缺一不可。

Head256 的 `4ef9320d9:_choose_block_h` 调整 head 方向 block 划分，测试覆盖 page32+Qwen3.5 H256 的 fused/split/queue4，以及 `num_segments=64`。因此 Page32 与 Head256 是两个正交维度，文档和 benchmark 不应只测二者同时开启的单一 case。

### 8.8 Small-M Cache 与 HC-Head Split-K

`17be8f1bf` 新增两类不同 Kernel：

1. `pack_store_flashmla_cache_decode_x4_fp32_kernel`：以 token 为主并行，直接完成 FP32 输入的 quant/scale/NoPE/RoPE page store；对应 benchmark 有 dispatch latency guard，要求 host-inclusive latency不因 wrapper 增长而回退。
2. HC-head linear split-K：stage0 按 K 维 split 计算 partial accumulation，stage1 对 partial 做 reduce/写最终输出；另有 warp variant。它适合 Decode 的小 M、大 K，代价是 partial workspace 和第二次 launch，所以 split 数必须 sweep，不能固定写成越多越好。

### 8.9 `is_paged=True` 为什么需要 Prefill/Decode 配套

`81fc824a2` 将 `Compressor.compress_fused(...is_paged=False)` 改为 `True`。这改变了 metadata 语义：

```text
非 paged/page4：extra_data[batch] 含 4 列 page/write 信息
paged：extra_data[token] 只有 1 列/按 physical block 解释
```

同一提交还将 prefill write Kernel 中 `block_id` 改为 local scalar再根据 `position < extra_data[batch,3]` rebind，避免编译器/SSA 路径在条件覆盖时使用错误 block。随后 `eb255ea13` 新增 `_tilelang_compress_forward_ratio4_decode_paged_kernel`，专门消费 paged `(num_tokens,1)` metadata，完成：

```text
block_id = indices[token]
write_pos = (seq_len + 3) % 4
写入当前四路 kv_score
遍历有效 source slot，计算 exp-weighted numerator/denominator
out = acc / denom
```

Ops wrapper 根据 `extra_data` shape 选择 paged 或 page4 decode。只改模型端 `is_paged=True` 而没有该 decode branch，会让 Decode 用错误列数和地址解释 metadata。

### 8.10 MTP/EAGLE Graph 中 Metadata 的生命周期

Graph capture 固定 tensor address，但每轮 replay 的值会变：draft token、`seq_len`、cache location、accept index、expert remap 都必须 copy/update 到被 capture 的 buffer。`6d8e20f98:_set_replay_forward_batch` 修复 D2H `seq_len` capture，并加入 draft-token graph；若 host 读取未更新的旧值，下一轮 workspace/page 计算会沿用上一 batch 长度。

`4c49bdcf9` 将 accept-index 防护拆成：

- `_flatten_accept_index`：统一输入 rank；
- `_sanitize_accept_index`：过滤越界/无效值；
- `_safe_index_select_1d`：在选择前验证 index；
- `_evict_indices_from_accept_index`：只释放真实被接受/淘汰的 cache slot；
- `_sanitize_num_correct_drafts`：约束 draft 计数。

它同时提供 allocator over-free repair，说明错误不仅会产生错误 token，还可能重复释放 SWA cache slot。`bb3e58863` 再修 expert remap 和 EAGLE extend graph，确保 draft/target 的 expert id 与 extend replay metadata 使用同一映射。

### 8.11 Sampling 路径边界

No-seed 路径允许随机采样但仍需设备可运行；`22023f1df` 对 MUSA 避免有问题的 `torch.multinomial` fallback，改走 `sgl_kernel` 可用实现。Seeded path 则承担 deterministic/reproducibility 契约，不能因为 no-seed backend 更快而替换。验证至少分 greedy、seeded multinomial、no-seed multinomial、graph on/off，并检查分布统计和重复运行一致性。

## 9. Online C128、PP/PD 与 HiSparse：上游系统语义的 MUSA 状态闭环

> **归属与双基线**：Online C128、PP/PD、MTP 状态和 HiSparse 的系统目标属于官方 `O`。当前官方已有 Online C128/MTP（例如 `063ab89ac1`，2026-06-16）、PP/PD 和 HiSparse 实现；MUSA 分支同期也有连续演进。因此本章按接口/状态语义（`O`）、MUSA graph/通信/runtime 移植（`P+S`）和 MUSA TileLang offload Kernel（`R`）拆分，不把整套长上下文系统包装为 MUSA 从零自研。

关键 snapshot：

```text
66464a505:python/sglang/srt/layers/attention/dsv4/c128_online_{eagle,triton}.py
66464a505:python/sglang/srt/disaggregation/dsv4_pd_disagg.py
1dbba2604:python/sglang/srt/model_executor/{cuda_graph_runner,model_runner}.py
1dbba2604:python/sglang/srt/disaggregation/
6e7f541ae:.../deepseek_v4_musa/kernels/hisparse_kernels.py
6e7f541ae:.../deepseek_v4_musa/ops/hisparse_ops.py
```

`6e7f541ae/65093d411` 是相同 patch-id 的重复历史，只计一次实现。


### 9.1 上游特性与官方 NV 状态流

官方社区的 Online C128、Pipeline Parallel、Prefill/Decode Disaggregation 和 HiSparse 解决的是：

```text
长上下文状态存储
单卡显存容量
Prefill/Decode 解耦
跨 Stage/节点传输
Sparse Cache Offload
```

NV 路径通常依赖 CUDA/NCCL、CUDA Graph 和官方存储/传输 Backend。

### 9.2 为什么值得在 S5000 跟进

32K～128K 客户负载下，问题不再只是单 Kernel：

- KV/Compressed State 占用大；
- Prefill 和 Decode 资源需求不同；
- 多卡 PP/CP/PD 成为容量和吞吐前提；
- S5000 带宽较低，重复构建/传输 C128 State 代价高；
- HiSparse Offload 若走 CUDA-only Helper 无法运行。

### 9.3 原状态流不匹配与 MUSA 实现边界

直接移植不仅是把 `cuda` 字符串替换为 `musa`。Online C128 在 Prefill 产生压缩状态，跨 PD/PP 传输后由 Decode/MTP 消费；生命周期同时涉及 request id、cache location、pending/commit、memory pool、graph proxy tensor 和 stage ownership。任一 metadata 仍指向旧 buffer，graph replay 都可能读错状态。

```text
官方状态语义 O
  Prefill compress -> online C128 -> transfer/stage -> Decode/MTP consume

MUSA P+S
  MUSA memory pool + connection backend + metadata/cache-loc
  + PP4 graph/proxy tensor + EAGLE deferred/pending state

MUSA R
  TileLang HiSparse swap/offload Kernel
  + MUSA tensor/layout/page guard
```

这类工作的主要效果应以“可承载上下文、显存峰值、state rebuild/transfer 次数、graph replay 正确率和长稳”衡量。只有 HiSparse Kernel 本身适合用 operator latency/GB/s；不能把系统适配笼统写成单算子加速。Unsupported layout/page、graph bucket 或连接 backend 必须进入显式受测 fallback 或 fail-closed，不能复用错误的 C128/cache-location 状态。

### 9.4 我们的实现

相关提交：

```text
66464a505 / a570eb0f8 Support online_c128 & MTP
1dbba2604 / 5cc145e9d Enable DSV4 Pro PP migration and PP4 graph
6e7f541ae / 65093d411 Add MUSA TileLang hisparse offload kernel
```

状态闭环：

```text
Prefill Compress State
 -> Online C128
 -> PD Transfer / PP Stage
 -> Decode/MTP Consume
 -> Cache Lifecycle Update
```

### 9.5 客户验证、效果与反哺

测试 32K/128K、PP/CP/TP 组合、PD 分离、MTP、C128 on/off、Offload on/off。指标：TTFT、TPOT、传输时间、显存峰值、State Rebuild 次数、Graph Replay、P95、长稳和故障恢复。

当前没有统一性能倍数，按 **证据 C/D** 记录；效果应以“可承载上下文、显存下降、State Transfer、系统吞吐和稳定性”表达，而不是虚构单算子加速比。

反哺形成：PP/PD 组合推荐、C128 生命周期规则、HiSparse Backend 路由、长上下文显存水位和回归 Case。

### 9.6 Full KV、SWA 与 CP Rank 的五层坐标

`66464a505` 中的 PD helper 不是简单地对一组 page id 做切片。它同时处理五种不同坐标；如果把它们都称为“cache index”，最容易在 CP、decode DP 或 SWA ring 下把数据传到合法但错误的位置。

| 坐标 | 生成方式 | 所属空间 | 用途 |
|---|---|---|---|
| Full token location | `req_to_token[req, position]` | Full KV pool token | 请求 token 到物理 KV slot 的基础映射 |
| Full global page | `full_loc // page_size` | Full KV pool page | PD 两端共同识别的绝对 page |
| SWA token/page | `swa_loc = translate_loc_from_full_to_swa(full_loc)`；`swa_page = swa_loc // swa_page_size` | SWA pool | 找到 C4/C128 state 实际存放页 |
| Request-relative global page | `full_page - first_page` | 当前请求 `[0, total_pages)` | 让 prefill/decode 即使物理起始页不同也可对齐 |
| CP-rank page subset | 对请求相对范围按 `cp_size` 均分后平移回 global page | 当前 CP rank | 只传该 rank 拥有的 Full KV/SWA state |

这里有两个不能省略的 page size：`page_size` 用于 Full KV，`swa_page_size` 用于 SWA pool。两者数值可以不同，因此 `full_loc // swa_page_size` 或 `swa_loc // page_size` 都不是合法替换。

#### 9.6.1 `unique_swa_page_indices`：只做 SWA 页去重

`dsv4_pd_disagg.py::unique_swa_page_indices` 的操作很窄：

```text
window_swa_locs
  -> np.asarray(int64)
  -> // swa_page_size
  -> np.unique
  -> int32 SWA page ids
```

空输入返回空 `int32` 数组。它不保留 Full page 对应关系，也不执行 CP 分片。因此它适用于“只需传唯一 SWA state 页”的路径；一旦 decode 侧还需要知道每个 SWA state 对应请求中的哪个 global page，就必须使用 pair helper。

#### 9.6.2 `build_swa_state_index_pairs`：保留 Full/SWA 对应关系

该函数对同一窗口的 `window_full_locs` 和 `window_swa_locs` 分别整除各自 page size，构造：

```text
(full_global_page, swa_pool_page)
```

随后按 pair 去重、按 Full global page 排序，并返回两个同序数组。传入 `first_page` 时，第一列进一步变为 request-relative global page：

```text
relative_global_page = full_global_page - first_page
```

这一步不能只对两列分别 `unique`：Full page 与 SWA page 的配对关系才是 decode 侧重排 state destination 的依据。任一输入为空时返回两个空数组；调用方还应验证两组 token location 长度相等，因为 `np.stack` 本身会拒绝不一致长度。

#### 9.6.3 `page_indices_to_cp_rank_page_indices`：先按请求分片，再回到全局页

`disaggregation/utils.py` 先把当前请求视为连续局部范围 `[0, total_pages)`。设：

```text
base = total_pages // cp_size
rem  = total_pages % cp_size
```

当不能整除时，前 `rem` 个 rank 各多一页：

```text
local_start = cp_rank * base + min(cp_rank, rem)
local_count = base + (cp_rank < rem)
```

然后以 `first_page` 平移为 `[start_page, end_page)` 的 Full global page 范围，并与输入 `page_indices` 取交集。`cp_size <= 1` 时原样返回，空输入保持为空；没有显式 `first_page` 时以输入最小页推断，但 PD 请求已经掌握起始页时应显式传入，避免稀疏输入把“最小观测页”误当请求起点。

函数名里的 “CP rank page indices” 表示“属于该 rank 的 global page 子集”，并不是重新编号为 rank-local page。后续若需要 request-relative 坐标，仍要减去 `first_page`。

#### 9.6.4 `prepare_prefill_swa_state_indices`：Prefill 侧的组合变换

`prepare_prefill_swa_state_indices` 返回：

```text
(swa_state_indices, relative_global_pages_for_decode_alignment)
```

真实分支如下：

1. 先通过 `unique_swa_page_indices` 得到唯一 SWA page；为空则返回 `(None, None)`。
2. CP transfer 未启用时只返回 SWA page list，global-page metadata 为 `None`。
3. CP transfer 启用时构造逐 token `(full_global_page, swa_page)` pair。
4. 调用 `page_indices_to_cp_rank_page_indices` 找到当前 CP rank 拥有的 Full page。
5. 仅保留这些 Full page 对应的 pair。
6. 同一 SWA page 若映射到多个 Full page，保留最小 request-relative page。
7. 按 relative page 排序，输出同序 SWA state index 与 relative global page。

第 6 步是 SWA ring 语义的关键：SWA 物理页可被窗口映射重复引用，传输列表必须按 state 页去重，但仍要保存一个稳定的请求相对坐标供 decode 对齐。

#### 9.6.5 `align_dsv4_cp_state_dst_indices`：Decode 侧按相对页重排

Mooncake 路径拿到 prefill 的 relative global pages 后，将 decode 自己的：

```text
(dst_state_global_pages, dst_state_indices)
```

构成 `relative_page -> destination_state_index` 映射，再按 prefill relative-page 顺序生成 aligned destination。这样即使两端 SWA pool 的物理 page id 不同，传输顺序仍按请求内页坐标一致。

当前实现的边界必须如实记录：

| 条件 | 当前行为 | 验证要求 |
|---|---|---|
| 非 DSV4 CP transfer | 原样返回 destination indices | 确认没有意外重排 |
| 任一 global-page metadata 缺失/为空 | 原样返回 | 上层应确认此路径是否允许无对齐 |
| decode global-page 数量与 destination 数量不同 | warning，并按 prefill 数量截断 destination | 生产配置应把 warning 提升为失败或证明截断安全 |
| 某些 prefill relative page 在 decode 映射中缺失 | 只返回找到的项并 warning | 不得把缩短后的传输当完整成功 |
| 完全匹配 | 严格按 prefill 顺序返回 | 校验值和顺序，而不只校验长度 |

因此该函数是“带告警的兼容实现”，不是所有 metadata 错误都 fail-closed。长稳测试必须抓取 warning，并验证实际 transfer count；否则一次部分对齐可能直到后续 attention 才表现为结果偏差。

### 9.7 Online C128 与 MTP Verify 的 Deferred-Commit 状态机

普通 decode 的每个 token 都会进入在线 C128 累积状态；MTP/EAGLE verify 则一次执行多个候选 row，其中一部分会被拒绝。若 verify forward 直接更新持久状态，被拒绝的 draft 也会污染 ring。因此 `66464a505` 采用“forward 缓存、verify 后回放”的两阶段提交。

```text
verify 开始
  -> reset_c128_verify_kv_cache
  -> TARGET_VERIFY forward
  -> 每层 cache_c128_verify_kv_score（只缓存候选 kv_score）
  -> EagleVerifyInput.verify 产生 accepted rows
  -> accepted rows/bonus token 按请求排序
  -> update_c128_online_state_after_mtp_verify
  -> 逐 accepted token replay 到持久 C128 state
  -> reset cache
```

`eagle_worker.py` 和 `eagle_worker_v2.py` 都在 verify 前清空旧 cache，并在 `spec_info.verify(...)` 返回 accepted metadata 后调用 backend wrapper。wrapper 再把 `req_to_token` 一并传入 `c128_online_eagle.py`。这保证 commit 使用的是 verify 结束后的请求映射，而不是某个临时 graph row 的位置。

#### 9.7.1 Accepted row 的含义

`accepted_kv_score_indices_per_req_cpu` 中每个列表保存 verify `kv_score` 扁平张量的全局 row index，顺序就是提交顺序，并包含 bonus token。它不能替换成每个请求的局部 `0..n-1`，也不能只传 accepted token 数；回放必须准确选择该请求对应的候选 row。

每个请求先记录 `seq_lens_pre_verify`。第 `step` 个 accepted row 的提交长度为：

```text
seq_len = base_seq_len + step + 1
```

这使 `_online_state_index` 能用真实序列位置计算：

```text
position    = floor((seq_len - 1) / 128) * 128
raw_loc     = req_to_token[req_pool_index, position]
swa_loc     = translate_loc_from_full_to_swa(raw_loc)
state_loc   = swa_page * ring_size + (swa_loc % ring_size)
state_index = floor(state_loc / 128)
```

这里同时出现 Full location、SWA location、ring offset 与 C128 state index，不能用第 9.6 节的 PD page index直接替代。

#### 9.7.2 逐层、逐请求、逐 accepted token 回放

`update_c128_online_state_after_mtp_verify` 的执行顺序为：

1. Online compress 未启用时直接返回。
2. verify cache 不存在或为空时直接返回。
3. 遍历模型层；只处理有 compressor 且 `ratio == 128` 的层。
4. 某层没有 cached `kv_score` 时跳过该层。
5. 对每个请求按 accepted list 顺序逐 row 调用 `c128_online_decode_triton(..., disable_state_update=False)`。
6. 当 `seq_len % 128 == 0` 时，对输出执行 fused norm/RoPE；配置要求时执行 `rotate_activation`；再写入 compressed KV page。
7. 所有层完成后清空 verify cache。

compressed store 也有两个显式分支：启用 fused-store 时调用 `set_extra_key_buffer_fused`；否则先通过 `quant_to_nope_fp8_rope_bf16_pack_triton` 打包，再调用普通 setter。二者必须对同一个 `c128_loc = raw_loc // 128` 产生一致内容。

#### 9.7.3 MTP 状态失败矩阵

| 注入条件 | 正确结果 | 不能接受的结果 |
|---|---|---|
| Online C128 off | 显式 no-op | 分配或修改 C128 state |
| verify cache 空 | no-op | 复用上一 batch 的 score |
| 某请求 accepted list 空 | 该请求不提交 | 仍推进 ring |
| 某层非 C128/无 score | 跳过该层 | 使用其他层 score |
| accepted flat index 越界 | fail-closed | Python slice 为空后继续写 state |
| accepted 顺序打乱 | 测试必须失败 | 只比较最终 shape |
| 跨过 127→128 边界 | 恰好一次 compress/store | 少写或重复写 C128 page |
| verify 抛异常 | 后续 batch 开始前 cache 必须 reset | 陈旧 cache 被下一 batch 消费 |
| worker v1/v2 | accepted 语义一致 | 两条路径只覆盖其一 |

源码已经处理常规空路径和层筛选，但越界 row、异常后的清理以及 v1/v2 一致性仍应通过失败注入验证；不能把 `finally` 语义写成现状中已经存在的保证。

### 9.8 PP Layer Range 与压缩 MLA Pointer Layout

`1dbba2604` 的关键不是简单的 `dst[start:end]`。DSV4 的 pointer table 按压缩类别分段，且 decode 侧可能持有完整模型、SWA pool 和 draft model，而 prefill PP stage 只拥有一段 layer。

`get_mla_kv_ptrs_with_pp` 先处理三种情况：

```text
src/dst 长度相同     -> 不切片
存在 mla_ratios      -> _mla_slice_ptrs_for_pp
普通 MLA             -> dst[prefill_start_layer : start + len(src)]
```

对 compressed MLA，`_mla_slice_ptrs_for_pp` 强制要求 `prefill_end_layer`，并按全模型 `mla_ratios` 统计：

```text
c4_full   = count(ratio == 4)
c128_full = count(ratio == 128)
compact layout length = 2 * c4_full + c128_full
```

compact layout 的三段是 C4 attention state、C4 indexer state 和 C128 state。函数分别统计 stage 前后 C4/C128 layer 数作为 offset，再从每一段切出同一 PP layer range，最后校验切片后 src/dst pointer 数严格相等。

另一种带 SWA buffer 的 layout 为：

```text
[SWA per layer]
[compress state for every ratio != 0 layer]
[indexer state for ratio == 4 layer]
```

此时不能用 layer id直接切后两段；必须分别使用 `count(ratio != 0)` 和 `count(ratio == 4)` 计算段内 offset。`swa_L < 0`、`swa_L > num_layers`、`prefill_end_layer > swa_L` 或最终 pointer 数不匹配都会显式失败。

MHA helper 还单独覆盖 decode 含 draft KV 而 prefill 没有 speculative model 的布局。该分支的 pointer offset 与 compressed MLA 不同，文档和测试不能混为同一算法。

### 9.9 `wait_layer_transfer` 建立的可见性边界

DeepSeek V4 memory pool 在 PP migration 中用 `layer_transfer_counter` 跟踪逐层传输。`wait_layer_transfer(layer_id)` 调用：

```text
layer_transfer_counter.wait_until(layer_id - start_layer)
```

随后 `get_attention_compress_states`、`get_indexer_compress_states`、`get_swa_key_buffer` 和 radix SWA getter 在返回底层 buffer 前先等待。这一顺序建立了：

```text
PP transfer for layer L 完成
  happens-before
attention/indexer/graph replay 读取 layer L buffer
```

它不是全局 barrier，也不表示所有层都完成；local index 必须减去当前 stage 的 `start_layer`。验证要故意让后层先完成、前层延迟，确认读取 layer L 只等待正确 counter，不会 off-by-one 或错误跳过。对没有调用 getter 的直接 raw pointer 访问，则不能自动继承此保证，必须审查调用点是否另有 event/stream 同步。

PP4 graph 的 proxy tensor 测试还需同时验证：capture shape 固定、stage 输入输出 key 一致、replay 更新的是 proxy 内容而不是替换对象，以及 pointer 所指层已经通过 transfer wait。只证明 graph capture API 返回成功不等于迁移数据可见。

### 9.10 HiSparse Host Offload 的 MUSA 生命周期

`6e7f541ae` 把 `jit_kernel/deepseek_v4.py::hisparse_offload_to_host` 的 MUSA 分支从 `NotImplementedError` 改为调用 `hisparse_offload_to_host_musa`。router 只要四个输入中任一个是 MUSA tensor，就不会加载 CUDA HiSparse transfer module，而进入 MUSA guard。

```text
调用方构造 gpu_ptrs/cpu_ptrs + gpu_indices/cpu_indices
  -> 顶层 router 识别 MUSA
  -> hisparse_offload_to_host_musa 校验契约
  -> 获取缓存的 TileLang kernel
  -> grid = (num_items, num_layers, copy_blocks)
  -> 每个 item/layer 拷贝 72 个 value uint64 + 1 个 scale uint64
  -> 返回调用方；可见性由当前 stream/后续同步保证
```

Kernel 把 GPU cache 固定解释为 page size 64：

```text
gpu_page        = gpu_index >> 6
gpu_page_offset = gpu_index & 63
gpu page stride = 4680 uint64
value width      = 72 uint64
scale base       = 4608 uint64
CPU item width   = 73 uint64
```

因此这是特定 HiSparse packed layout 的 direct pointer copy，不是任意 tensor copy API。`cpu_ptrs` 虽指向 host-offload destination，但 pointer table 本身仍要求位于 MUSA device；kernel launch 返回也不自动证明 CPU consumer 已同步看到数据。

#### 9.10.1 Guard 与失败语义

| 条件 | 实现行为 |
|---|---|
| pointer tables 非 MUSA | `NotImplementedError` |
| index tensor 与 pointer table device 不同 | `NotImplementedError` |
| pointer table 非 `uint64` | `NotImplementedError` |
| indices 非 `int64` | `NotImplementedError` |
| pointer table 非匹配的一维 shape | `NotImplementedError` |
| index 非匹配的一维 shape | `NotImplementedError` |
| pointer 或 index 非 contiguous | `NotImplementedError` |
| `num_items == 0` 或 `num_layers == 0` | 显式 no-op |
| page/layout 与固定常量不一致 | 当前 guard 无法从裸 pointer 推断；调用方必须前置拒绝 |
| kernel launch 失败 | 异常向上传播，不应切回 CUDA/Torch copy |
| launch 后立即由 CPU 读取 | 调用方必须等待 stream/event；源码不能据此声称同步完成 |

该提交的测试能静态证明 MUSA routing 和输入 guard 存在，但不等于已经覆盖 page 常量错误、异步 host 可见性、进程退出时 pending copy 或多 stream 重用。客户长稳应记录 pending item 数、offload latency、host memory 水位、同步超时和恢复行为。

### 9.11 第 9 章的分层验证矩阵

| 层级 | 核心 Case | 判定 |
|---|---|---|
| 纯函数 | Full/SWA pair 去重、非整除 CP 分片、relative-page align | 数组值、顺序、dtype 全部一致 |
| 状态单测 | accepted/rejected/bonus row，127→128 边界 | rejected 不改 state；边界只 store 一次 |
| PP 单测 | compact 与 SWA layout、stage 首/中/尾、draft destination | pointer 段和数量严格一致 |
| 同步测试 | layer transfer 乱序完成 | getter 不早读且不等待错误 layer |
| HiSparse guard | device/dtype/shape/contiguous/空输入 | exception 或 no-op 与表格一致 |
| 集成 | CP×PP×PD、MTP、Online C128、graph on/off | token 结果、state 内容、transfer count 一致 |
| 长稳 | 32K/128K、offload on/off、故障恢复 | 无陈旧 state、部分 transfer 或 host 可见性错误 |

`66464a505`、`1dbba2604`、`6e7f541ae` 已通过历史提交静态定位；其中 `6e7f541ae` 与 `65093d411` 的 stable patch-id 均为 `f1f7f0a076c6d6905dd6b51f96df843cfef95570`，本文只计一次实现。以上仍是源码/测试证据，不是当前 S5000 实机通过声明。

## 10. Runtime Hardening：证明 S5000 Fast Path 真实命中

> **归属与双基线**：backend router、JIT、fallback、op registration、distributed init 和 graph capture 是官方框架 `O`；MUSA commit 修改的是平台行为与可观测性（`P+S`）。它不是模型算法或新 Kernel 性能本身，却决定 benchmark 测到的是新路径还是 silent fallback。

关键 snapshot：

```text
1b6399413:python/sglang/jit_kernel/{csrc,include}/...          # MUSA JIT/FFI/tensor 兼容
3cceb08b9:.../deepseek_v4_musa/kernels/compress_kernels.py    # extra_data_cols JIT name
ef4469cb5:python/sglang/srt/distributed/parallel_state.py     # MCCL device binding
90b4437a8 / fecc03a86:.../deepseek_v4_musa/_forwarding.py    # 同 patch facade dispatch
```


### 10.1 上游行为与跟进价值

官方框架为覆盖不同硬件保留 CUDA、Triton、Torch 和 JIT Fallback。NV 上某些 fallback 可运行且性能尚可，但在 MUSA 上可能：

```text
进入 CUDA-only Helper
Graph Capture 失败
静默走 Torch 慢路径
命中错误 JIT Cache
多卡 Device 未绑定
Python Dispatch 抵消 Small Kernel 收益
```

因此 Runtime Hardening 是所有性能数字可信的前提。

### 10.2 Fallback：从宽松容错到 MUSA Fail-Closed

相关提交：

```text
1b6399413 Fix DSV4 MUSA JIT compatibility and runtime fallbacks
5fcb7c439 Fix DSV4 MUSA fallback blockers
5742d2df8 Make DSV4 MUSA fallback graph-safe
```

决策：开发阶段允许显式 Debug Fallback，生产默认 Fail-Closed。MUSA Tensor 不进入 CUDA-only HiSparse；Graph Capture 不允许 Torch Fallback；Compress/TopK Fast-Path Miss 可强制报错。

### 10.3 JIT Cache：把 Layout 参数纳入 Key

不同 `extra_data_cols` 影响 Page Layout，必须使用不同 JIT Name，避免 Page1/Page4 复用错误编译产物。此项效果用 Correctness、Cache Hit 和首次编译次数衡量，不写加速倍数。

### 10.4 MCCL Device Binding

`init_process_group(backend="mccl")` 显式传入：

```python
device_id=torch.device("musa", local_rank)
```

验证 Rank/Local Rank/Current Device、一致性、Collective 和多卡长稳。

### 10.5 Ops Facade Dispatch

减少高频 Decode 路径中的 Python Wrapper、重复 Backend 判断和 forwarding。对比 Kernel-only 与 Host-inclusive latency，确认优化没有被 Host Dispatch 抵消。

### 10.6 客户验证和效果指标

Runtime 类按 **证据 C** 记录：提交和测试能证明 guard、backend 与错误处理存在，但本文没有 MUSA 实机数据，不能写统一加速倍数。使用以下可观测指标：

```text
Fast-Path Hit Rate
Unexpected Fallback = 0
Graph Capture Success Rate
Replay P95/P99
多卡启动成功率
24h/长稳错误数
Host Dispatch Latency
JIT Cache Collision = 0
```

反哺规则：

```text
Production -> Fail-Closed
Debug      -> 显式 Allow Fallback
Graph      -> No Torch Fallback
JIT Key    -> 包含全部 Codegen/Layout 参数
MCCL       -> 显式 Local Device
Benchmark  -> 同时记录 Kernel 与 Host Inclusive Time
```

### 10.7 JIT Key 必须覆盖 Codegen/Layout 参数

`3cceb08b9` 修复前，`_tilelang_compress_forward_ratio4_decode_page_kernel(head_dim, extra_data_cols, threads)` 虽然接收 `extra_data_cols`，JIT 名称却固定为：

```text
dsv4_c4_decode_page4_t{threads}
```

这会把两种不同函数签名映射到同一个编译缓存名字：

```text
paged metadata: extra_data_cols = 1
page4 metadata: extra_data_cols = 4
```

两者的 `extra_data` shape 与生成代码不同。若先编译 page4 再运行 page1，或顺序反过来，缓存命中不代表命中了正确 Kernel；最危险的结果不是编译报错，而是用旧 layout 解释新 metadata。

修复后名称为：

```python
f"dsv4_c4_decode_page{extra_data_cols}_t{threads}"
```

即至少形成 `dsv4_c4_decode_page1_t128` 和 `dsv4_c4_decode_page4_t128` 两个 artifact。验证不能只清空 cache 后各运行一次，必须覆盖：

```text
page1 -> page4 -> page1
page4 -> page1 -> page4
多进程共享 cache
首次编译与热 cache replay
```

并同时断言 factory 收到 `(head_dim, extra_data_cols)`、artifact 名不同、数值结果与 reference 一致。该修复说明通用规则是：所有改变 tensor rank、shape、stride 假设、常量循环或地址公式的参数都必须进入 JIT key；仅把 `threads` 纳入 key 不够。

### 10.8 MCCL Device Binding 的构造顺序

`ef4469cb5` 并不是无条件给所有 process group 加 MUSA device。实现先构造局部字典：

```python
init_kwargs = {}
if backend == "mccl" and local_rank >= 0:
    init_kwargs["device_id"] = torch.device("musa", local_rank)
```

然后一次性调用：

```python
torch.distributed.init_process_group(
    backend=backend,
    init_method=distributed_init_method,
    world_size=world_size,
    rank=rank,
    timeout=timeout,
    **init_kwargs,
)
```

这有三个语义边界：

1. 只有 MCCL 接收 MUSA `device_id`，Gloo/NCCL 等 backend 保持原接口。
2. `local_rank < 0` 时不构造非法 device；上层仍需在初始化后处理 local-rank/current-device 关系。
3. device kwargs 必须在 `init_process_group` 前准备好；在 group 创建后再 `set_device` 不能回溯修正 WORLD group 的绑定。

测试应 monkeypatch `init_process_group` 捕获 kwargs，覆盖 MCCL rank 0/非 0、`local_rank=-1` 和非 MCCL；多卡集成再校验 `rank -> local_rank -> current_device`、collective 内容、重复启动和异常退出清理。源码静态存在该 kwargs 不等于多卡 S5000 已长稳通过。

### 10.9 Forwarding Module：把 Patch 同步从每次调用移到赋值时

MUSA ops facade 同时暴露聚合模块和 domain modules。测试或运行时可能 monkeypatch facade symbol；若 patch 只留在 facade，而 wrapper 最终调用 `_impl`，就会出现“赋值成功但执行旧函数”。旧方案每次 wrapper call 都扫描 patch，能保持语义却把 Python 开销放进高频 decode 路径。

`90b4437a8`/`fecc03a86` 的同 patch 修改由两部分构成。

第一部分是正常 import 快路径：

```python
_CALL_SYNC_REQUIRED = __name__ not in sys.modules
```

wrapper 只在特殊加载场景要求时执行 `_sync_patches_to_impl()`；正常模块已经注册在 `sys.modules` 时，不再为每个 op call 扫描。

第二部分把同步移到 `_ForwardingModule.__setattr__`：

```text
setattr(facade, name, value)
  -> setattr(target_module, name, value)
  -> target_module._sync_patch_to_domain(name, value)（若存在）
  -> facade 保存同一 value
```

`__dunder__`、`_target_module` 和 `__all__` 是保留属性，不进入普通转发。验证矩阵需要覆盖：

| Case | 期望 |
|---|---|
| 正常 import 后重复调用 | 不发生 per-call 全量同步 |
| facade 上 monkeypatch op | target 与 domain 立即看到同一对象 |
| domain patch 后经同步入口传播 | facade 实际调用新对象 |
| 特殊 loader/模块未注册 | `_CALL_SYNC_REQUIRED` 保留兼容扫描 |
| patch 恢复 | facade、target、domain 三处恢复一致 |

该优化减少的是 host dispatch overhead，不应与 Kernel latency 混为同一个加速数字。应同时报告 direct-op、facade warm-call 和 monkeypatch 场景的 host-inclusive latency。

### 10.10 FlashMLA/Indexer Cache 的 No-Copy 契约

“no-copy”不是对任意输入都不 materialize，而是 fast path 接受调用方现有 storage/stride，并在不满足契约时拒绝偷偷 `.contiguous()`。静默 materialization 有两类问题：它既可能掩盖错误 layout，也会把额外 allocation/copy 带进 graph capture 和 benchmark。

对 FlashMLA 与 Indexer cache store，生产判定至少包含：

```text
device == musa
支持的 input/cache dtype
明确的 rank、last-dim 和 packed page layout
loc/index dtype 与 shape 合法
cache contiguous 或 Kernel 明确支持其 stride
input row stride 在 Kernel 支持集合内
TileLang/JIT Kernel 已可用
```

满足这些 guard 才能直接把现有 pointer/stride 传给 cache kernel。当前测试名称已经体现这一区分，例如：

- `test_fused_store_cache_musa_flashmla_rejects_non_contiguous_cache`；
- `...flashmla_invokes_tilelang_store_fast_path`；
- `...flashmla_accepts_row_strided_input_on_musa`；
- `...flashmla_rejects_dtype_and_layout_edges`；
- Indexer multiple-page 与 layout reference cases；
- compress metadata 的 `rejects_non_contiguous_*_without_copy` cases。

需要特别区分两种 non-contiguous：cache destination 若地址公式要求连续 packed page，应 fail-closed；row-strided source 若 Kernel 参数显式携带 stride，则可以 no-copy 支持。不能用一个全局 `is_contiguous()` 规则描述二者。

Production 和 Debug 的边界为：

```text
Production fast-path miss
  -> 带 device/dtype/shape/stride/page/runtime metadata 的异常

显式 Debug fallback 开启
  -> 允许 reference/Torch 路径
  -> 必须记录 fallback reason
  -> 不得把结果计入 fast-path benchmark
```

`64e803bd3`/`49eb8067d`、`fc1e88584`/`392179e92`、`c2eafb495`/`7c992a2a8` 分别代表 cache hardening、Indexer store 和 FlashMLA prefill store 的重复历史快照；引用时应按 patch 实现去重，而不是把镜像提交累计成多次优化。

### 10.11 Graph Capture 前置验证与 Replay 验证

Graph-safe 不是“在 capture 中 try 一次，没抛异常就通过”。capture 前应把所有可能动态改变控制流或内存行为的条件固定下来：

| 前置项 | 必须确认的内容 |
|---|---|
| Backend | MUSA path 已选择，不能落入 CUDA-only helper |
| Kernel | 对当前 shape/layout 的 artifact 已预编译或可在 capture 外构建 |
| Fallback | capture 中 Torch/reference fallback 禁止 |
| Metadata | dtype、rank、shape、stride、page size 与 bucket 固定 |
| Buffer | output/workspace/proxy tensor 已分配，replay 不新建 storage |
| PP proxy | key 集合、shape 和对象 identity 稳定，replay 只更新内容 |
| Distributed | MCCL device/group 已初始化，capture 内不建立 group |
| 状态 | C128/MTP pending cache 与 request mapping 属于当前 batch |

`5742d2df8` 提供 graph-safe fallback 基础，`ef4469cb5` 中的 `test_graph_capture_ops.py` 覆盖一组 MUSA op capture，`1dbba2604` 的 `test_cuda_graph_pp_proxy_tensors.py` 覆盖 PP proxy；但这些测试服务于不同层级，不能用单个 operator capture case替代 PP/MTP 集成 replay。

验证应拆成四步：

1. **Preflight**：在 capture 外主动运行 guard，并预热所有 JIT specialization。
2. **Capture**：确认没有动态编译、host value read、隐式 allocation、unsupported sync 或 fallback。
3. **Replay correctness**：至少两组不同输入复用同一 graph，输出与 eager reference 一致，证明 graph 没固化旧 metadata。
4. **Failure injection**：改变 bucket/stride/page layout 或关闭 Kernel，确认在 capture 前拒绝，而不是 capture 中途失败或 replay 旧路径。

现有 `topk_transform_512_musa_rejects_torch_fallback_during_graph_capture` 等测试说明部分 op 已把 fallback 与 capture 状态联动；同一原则需要覆盖 compress、cache、HiSparse、C128 replay 和 PP proxy。

### 10.12 `1b6399413` 的准确边界

`1b6399413` 的提交标题是 “Fix DeepSeek V4 MUSA JIT compatibility and runtime fallbacks”，但实际 diff 主要落在：

```text
jit_kernel/csrc/deepseek_v4/{fused_norm_rope_v2,main_norm_rope,topk_1024,topk_v2}.cuh
jit_kernel/csrc/nsa/fused_store_index_cache.cuh
jit_kernel/csrc/distributed/tp_qknorm.cuh
jit_kernel/include/sgl_kernel/{ffi.h,tensor.h,utils.cuh,utils.h}
srt/layers/mhc.py
srt/layers/moe/moe_runner/deep_gemm.py
srt/models/deepseek_v4.py
```

因此它可以作为 MUSA JIT/FFI/tensor ABI 与 MHC/deep-gemm fallback 修复的证据，却不能替代：

- `3cceb08b9` 的 `extra_data_cols` JIT key；
- `ef4469cb5` 的 MCCL device binding；
- `90b4437a8` 的 forwarding 同步；
- 后续 cache no-copy/hardening commits；
- `66464a505` 的 Online C128/MTP 状态闭环。

按实际文件归属拆分提交证据，可以避免把 Runtime Hardening 写成一个不可审计的大 patch。

### 10.13 单测、集成与失败注入矩阵

| 层级 | 测试项 | 必须观测 |
|---|---|---|
| JIT 单测 | page1/page4 交叉编译和热 cache | artifact/key 不碰撞，结果一致 |
| MCCL 单测 | backend×local_rank kwargs | 仅合法 MCCL case带 MUSA device |
| Forwarding 单测 | facade/domain monkeypatch、恢复、特殊 loader | 对象同步且 warm call 不扫描 |
| Cache 单测 | FlashMLA/Indexer dtype/layout/stride/page matrix | no-copy 命中或明确 exception |
| Graph 单测 | preflight、capture、双输入 replay | 无旧 metadata、无 capture fallback |
| C128/PP 集成 | MTP×PP×CP×PD×graph | state、pointer slice、token 结果一致 |
| HiSparse 集成 | offload launch、event wait、CPU consume | host 可见前存在明确同步 |
| 多卡集成 | MCCL init/collective/重启 | rank-device 一致，无陈旧 group |
| 长稳 | 32K/128K、bucket 切换、cache cold/warm | fallback=0、collision=0、无 state 泄漏 |

失败注入应至少覆盖：

```text
错误 extra_data_cols / 交叉 cache 顺序
local_rank 缺失或 backend 非 MCCL
facade patch 后 domain 未同步
FlashMLA cache 非连续、Indexer layout 错误
source row stride 支持/不支持边界
capture 期间 Kernel miss 或 Torch fallback
PP proxy shape/key 改变
C128 accepted row 越界或顺序错误
HiSparse pointer dtype/device/layout 错误
layer transfer event 延迟或缺失
```

每个 case 的预期必须明确分类为：**正确 fast path**、**显式 no-op**、**warning 且上层阻断**、**fail-closed exception** 或 **仅 Debug fallback**。只检查“不 crash”会把 silent fallback、部分 transfer 和错误 JIT hit 都算成通过。

### 10.14 可观测性与证据结论

建议为 benchmark/客户验证统一记录：

```text
backend/op 名称与 specialization key
input device/dtype/shape/stride/page layout
fast-path hit / fallback reason
capture id、bucket、replay 次数
JIT cold/warm 与 compile count
Kernel-only / host-inclusive latency
MCCL rank/local-rank/current-device
PP transfer layer/counter/event
C128 accepted row 数与 state-store 数
HiSparse pending/completed item 数
```

静态核实结果：`1b6399413`、`3cceb08b9`、`ef4469cb5` 和 `90b4437a8` 均能在 `sglang-musa` 历史中解析；`90b4437a8` 与 `fecc03a86` 的 stable patch-id 均为 `db7a40a562d09ac7ae168ee36f707df190f24b91`，只计一次 facade 实现。本文没有在 MUSA/S5000 上执行上述矩阵，因此 Runtime 类仍按证据 C 表述，指标是验收条件而不是已取得的实测结果。

## 11. 官方新特性合入与 S5000 适配优化闭环

本节明确描述“从官方 SGLang 合入新特性，再针对 MUSA/S5000 做优化”的完整工作方法。这里的适配不是简单搬代码，而是要回答：官方实现为什么在 NV 上成立、在 S5000 上哪里不成立、我们保留了什么、替换了什么、重写了什么，以及最终是否转化为客户场景收益。

### 11.1 S5000 的硬件前提：高脊点、 高算力、低带宽

S5000 的关键特征不是单纯“算力低”或“带宽低”，而是：

```text
高计算峰值
相对较低的显存带宽
较高 Roofline Ridge Point
```

因此同一个官方 Kernel 在 NVIDIA 上和 S5000 上的最优点可能不同。对算术强度 `AI = FLOPs / Bytes` 的判断是：

```text
AI < Ridge Point  -> Memory-bound
AI > Ridge Point  -> Compute-bound
```

在高脊点 S5000 上，更多算子会落在 Memory-bound 区域。也就是说，单个算子即使计算量不大，只要频繁读写 HBM，就可能无法发挥 S5000 的计算能力；而真正的大矩阵 GEMM 如果数据布局规则、Tile 足够大，则更容易进入 Compute-bound 区域并发挥高算力。

因此 S5000 的优化原则不是盲目追求更大 GEMM，而是：

```text
1. 对大 GEMM：尽量提高算术强度和计算单元利用率。
2. 对 quant/compress/cache/norm/rope：尽量减少 HBM 往返。
3. 对 decode 小算子：减少 launch、Python dispatch 和 graph miss。
4. 对官方数据布局：优先判断是否引入额外 materialization。
5. 对 fallback：不能只看能否运行，要看是否掩盖了带宽和调度损失。
```

### 11.2 第一步：新特性进入社区

持续跟踪官方 SGLang、vLLM 和相关 Kernel 项目的：

```text
Feature/PR/Commit
新增算子
Backend Router
TileLang/Triton/CUDA 实现
DeepGEMM/FlashMLA 版本变化
正确性修复
性能 Patch
Graph/CP/TP/PD 运行时变化
```

每个新特性先建立记录：

| 项目 | 要回答的问题 |
| --- | --- |
| 目标 | 解决 TTFT、TPOT、吞吐、显存还是稳定性？ |
| 实现 | 是融合、布局、调度、GEMM 还是 Runtime？ |
| 后端 | Triton、TileLang、CUDA、Torch 还是 custom op？ |
| 触发条件 | Shape、Dtype、Batch、Page、Graph 限制是什么？ |
| 官方基线 | 哪个 commit/版本作为对比？ |
| MUSA 价值 | 是否会影响 S5000 客户负载？ |

不能因为官方标题包含“optimize”就直接合入。必须先判断它是：

```text
算法普适优化
NVIDIA 架构专用优化
编译器专用优化
Runtime/Serving 优化
```

### 11.3 第二步：判断是否值得跟进

采用四维判断：

#### 11.3.1 业务相关性

优先跟进会影响以下场景的特性：

```text
32K~128K 长 Prefill
高并发 Prefill
Decode TPOT
MTP/EAGLE
CP/TP/PP/PD
Online C128
KV/HiSparse/Prefix Cache
```

#### 11.3.2 算子相关性

优先分析出现在 DSV4 热路径中的算子：

```text
MoE Router/TopK
FP8 Activation Quant
Gate/Up/Down Grouped GEMM
SwiGLU
Indexer/RoPE/Hadamard
FlashMLA Sparse Gather
Compress
KV Cache Store
MHC
Decode Attention
Sampling
```

#### 11.3.3 S5000 硬件匹配度

如果官方优化主要依赖：

```text
SM90/SM100 特有 Tensor Core
CUDA PTX/inline assembly
NVIDIA-specific shared-memory primitive
CUDA-only graph/runtime
```

不能直接假设 S5000 受益，需要先做可编译性和性能 A/B。

反之，如果官方优化的核心是：

```text
减少 HBM 往返
减少 Kernel Launch
复用 Workspace
提高算术强度
改善数据布局
```

通常值得适配，因为这些目标与 S5000 的高脊点特征高度相关。

#### 11.3.4 维护成本

需要同时评估：

```text
MUSA 是否已有等价 kernel
是否需要新增 custom op
是否会分叉官方接口
是否影响 Decode
是否需要维护 exact/non-exact 两套路径
是否能加入自动 benchmark 和 regression gate
```

### 11.4 第三步：分析官方实现路径

对官方新特性建立完整调用链：

```text
Model
  -> Scheduler/Forward Mode
  -> Backend Router
  -> Python Op Wrapper
  -> TileLang/Triton/CUDA Kernel
  -> Compiler/Runtime
  -> Cache/Graph/Distributed
```

重点记录官方实现的“原始样子”：

```text
1. 官方在哪个条件下进入 fast path？
2. 默认 Tile/warp/stage/vector width 是什么？
3. 输入是否必须 contiguous？
4. 是否存在 CUDA-only helper？
5. miss 后是 Torch、Triton 还是另一个 Kernel？
6. Graph capture 是否允许该 fallback？
7. 是否重复构建 workspace 或 metadata？
8. Decode 和 Prefill 是否共用同一实现？
```

官方实现必须作为一个可运行的 baseline，而不是只阅读代码。至少要记录：

```text
official dispatch branch
official kernel name
official latency
official effective GB/s/FLOPS
official launch count
official memory/graph behavior
```

### 11.5 第四步：在 NV/MUSA 上建立公平对比

对比至少分三层。

#### 11.5.1 相同逻辑跨平台对比

```text
相同模型
相同 Shape
相同 Dtype
相同 Batch/Token
相同 Kernel 逻辑
相同 Warmup/迭代
相同测量方法
```

用于回答：

```text
同一算法在 NV/MUSA 上的性能差异是否来自硬件？
是否来自编译器？
是否来自 launch 参数？
```

#### 11.5.2 官方实现与 S5000 实现对比

```text
Official path on NV
Official/ported path on MUSA
MUSA tuned path on MUSA
```

不要只比较：

```text
NV official vs MUSA tuned
```

因为这会把硬件差异、代码差异和参数差异混在一起。必须增加：

```text
MUSA official-port baseline vs MUSA tuned
```

#### 11.5.3 模块和服务对比

```text
Operator
  -> MoE/Attention/Cache module
  -> Model Prefill/Decode
  -> Server workload
```

指标：

| 层级 | 指标 |
| --- | --- |
| Kernel | latency、GB/s、FLOPS、occupancy、register/shared memory |
| Runtime | launch 数、dispatch 时间、graph capture/replay、fallback |
| Model | Prefill latency、Decode latency、TTFT、TPOT、tokens/s |
| Service | concurrency、P50/P95/P99、OOM、SLO、长序列稳定性 |

### 11.6 第五步：定位 NV/MUSA 性能差异

#### 11.6.1 Memory-bound 差异

典型特征：

```text
计算单元利用率低
显存带宽接近平台上限
latency 随 bytes 近似增长
kernel FLOPS 不高
```

S5000 上的处理顺序：

```text
先减少 bytes
  -> 再做融合
  -> 再做 vectorized store
  -> 再调 Tile/warp
```

不能在 memory-bound 算子上只增加线程或 Tile，因为可能只会增加访存竞争。

适用对象：

```text
FP8 Quant
C4/C128 Compress
Cache Store
Norm/RoPE
Sparse Workspace Gather
```

#### 11.6.2 Compute-bound 差异

典型特征：

```text
Tensor Core/计算单元利用率高
算术强度高
GEMM Tile 影响明显
增加 M/Batch 后吞吐改善
```

S5000 上的处理顺序：

```text
提高 M/Token 规整度
  -> Expert-Contiguous/Compact
  -> 调整 Tile/warp/stage
  -> 减少 pipeline bubble
  -> 提高 GEMM 利用率
```

适用对象：

```text
Gate/Up Grouped GEMM
Down Grouped GEMM
大 Token MoE
大 Prefill Attention
```

#### 11.6.3 Launch-bound/Runtime-bound 差异

典型特征：

```text
单次 Kernel 很小
CPU/Host dispatch 占比高
Decode 每 token 反复调用
Graph capture/replay 不稳定
```

S5000 上的处理顺序：

```text
减少 wrapper/dispatch
  -> 小 Shape 专用 Kernel
  -> 融合小算子
  -> Graph capture
  -> 删除不必要同步和 materialization
```

适用对象：

```text
Decode Cache Store
RoPE Decode
MHC
Sampling
MTP/EAGLE
Ops Facade
```

### 11.7 第六步：选择保留、替换或重写

对每个官方新特性，最终只能选择一种生产策略：

#### 策略 A：直接保留

条件：

```text
MUSA 可编译
数值正确
MUSA 性能可接受
没有额外 materialization
Graph/Runtime 兼容
```

结果：

```text
官方逻辑
  -> MUSA 直接复用
```

#### 策略 B：保留算法，调整参数

条件：

```text
算法和数据布局没有问题
只是 NV 默认 Tile/warp/stage 不适合 S5000
```

调整：

```text
Tile Size
num_warps
num_stages
subwarp width
vector width
compile profile
threshold
```

#### 策略 C：保留接口，替换 Kernel

条件：

```text
官方接口有价值
官方 Kernel 在 MUSA 不可编译或明显慢
输入输出语义可以保持
```

结果：

```text
统一 Python/Backend 接口
  ├── NV official kernel
  └── MUSA/S5000 kernel
```

例如：

```text
FP8 Half-Warp Group Quant
MUSA TopK V2
C128 Parallel Reduce
MUSA Cache Store
MUSA Fused Norm-RoPE-Cache
```

#### 策略 D：重写数据组织或 Pipeline

条件：

```text
官方 Kernel 的性能问题来自数据布局或中间 Tensor
不是简单参数能解决
```

结果：

```text
官方 Compressed Cache/SWA
  -> MUSA Seq-Pack Workspace

官方 MoE Token Layout
  -> S5000 Fixed-Bucket Compact（复用官方 Expert-Contiguous）
```

#### 策略 E：部分保留/部分禁用

一个官方特性可能包含多个子路径：

```text
调度优化 + Kernel A + Kernel B + layout conversion
```

如果实测：

```text
调度优化有效
Kernel A 有效
Kernel B 变慢
layout conversion 吃掉收益
```

则只保留有效部分，禁用或缩小 Kernel B 的 guard，不能机械整体合入。

### 11.8 第七步：在客户场景验证

建议至少覆盖：

| 场景 | Input | Output | Concurrency | 重点 |
| --- | ---: | ---: | ---: | --- |
| 短 Decode | 1K | 128 | 1 | small-M TPOT |
| 中等 Prefill | 8K | 256 | 8 | TTFT、cache store |
| 长 Prefill | 32K | 128 | 8 | MoE、Sparse、workspace |
| 超长 Prefill | 128K | 128 | 1 | CP、显存、C128 |
| 高并发 Decode | 4K | 1K | 32/64 | TPOT/P95/sampling |
| MTP/EAGLE | 4K | 1K | 8/32 | accept rate、draft graph |
| PP/PD/Online C128 | 32K~128K | 128~1K | 多卡 | state/metadata/graph |

必须同时观察：

```text
TTFT 随输入长度
TPOT 随并发
Prefill/Decode 吞吐
P50/P95/P99
GPU 利用率
显存和 KV Cache 水位
OOM
Scheduler 排队
fallback 命中
graph capture/replay
输出正确性和 sampling 稳定性
```

算子收益只有在以下条件成立时，才能写成端到端收益：

```text
operator latency 下降
  -> module latency 下降
  -> model stage latency 下降
  -> TTFT/TPOT 或吞吐改善
```

如果算子变快但 Scheduler、通信或其他阶段成为瓶颈，应如实记录为“算子收益未完全传导”。

### 11.9 第八步：结果反哺适配策略

每次适配验证后沉淀四类结果：

#### 适配规则

```text
S5000 + group_size=128 + groups>=16K
  -> Half-Warp Quant

Decode + small M
  -> x4/vec2 cache store

Prefill + large M
  -> tile-parallel/subwarp cache store

C128 + large reduce
  -> parallel reduce
```

#### 禁用规则

```text
large CP prefill
  -> 禁止 generic TileKernels cast

graph capture
  -> 禁止 Torch fallback

unsupported page/layout
  -> 不进入 fused kernel
```

#### 回归 Case

```text
page1/page4 交替 JIT
CP round-robin sparse prefill
paged prefill + paged decode
MTP/EAGLE draft/extend
TopK exact/non-exact
MoE overflow
```

#### 参数推荐

```text
max_num_seqs
max_num_batched_tokens
chunked_prefill
mem_fraction_static
tp/ep/cp/pp
KV block size
attention backend
graph size
prefill chunk size
```

### 11.10 适配项目的标准结论模板

对每个新特性使用以下模板：

```text
【上游特性】
官方新增了什么，解决 TTFT/TPOT/吞吐/显存中的什么问题。

【官方实现】
调用链、Kernel、layout、dispatch、fallback、Graph 约束。

【S5000 对比】
相同 Shape/Dtype/Batch 下，NV 与 MUSA 结果如何；
MUSA official-port 与 tuned path 差异如何。

【差异归因】
属于 Memory-bound、Compute-bound、Launch-bound，
还是编译器、layout、Runtime、通信问题。

【适配决策】
直接保留、参数调优、替换 Kernel、重写 Pipeline，
或部分禁用某个子路径。

【我们的实现】
新增哪些 MUSA/S5000 Kernel、Fusion、Dispatch、Fallback 和 Runtime 修复。

【客户验证】
在什么 Input/Output/Concurrency/TP/EP/CP/PP 配置下验证。

【效果】
Operator、Module、Prefill、Decode、TTFT、TPOT、吞吐、P95 的结果。

【反哺规则】
最终形成哪些 Shape 规则、Backend 规则、参数推荐和回归测试。
```


## 12. 性能归因与验证方法

### 12.1 Compute-Bound

主要是：

```text
Compact MoE Gate/Up/Down Grouped GEMM
大 Token DeepGEMM
```

优化：

```text
Expert-contiguous layout
Fixed bucket
更规则的 grouped GEMM
融合 SwiGLU + Quant
```

### 12.2 Memory-Bound

主要是：

```text
FP8 Quant
C4/C128 Compress
Cache Store
Norm/RoPE/Cache
Sparse Workspace Gather
C4 Indexer/RoPE/Hadamard
```

优化：

```text
减少 materialization
Vectorized load/store
FP8/BF16 pack
Chunk-level workspace reuse
Fusion
```

### 12.3 Launch/Runtime-Bound

主要是：

```text
Decode Cache Store
RoPE Decode
MHC Mixed Handoff
Sampling
MTP/EAGLE Draft Graph
Ops Facade
```

优化：

```text
small-M 专用 kernel
Graph capture/replay
减少 host dispatch
显式 fast-path policy
```

### 12.4 建议验证矩阵

| 层级 | 重点指标 |
| --- | --- |
| TopK/Routing | latency、exactness、fallback |
| FP8 Quant | latency、scale correctness、derived GB/s |
| MoE | compact rows、overflow、device/host speedup |
| Sparse Prefill | workspace build、metadata build、CP correctness |
| Cache/Compress | page correctness、GB/s、vector write |
| Decode | TPOT、graph replay、small-M latency |
| MTP/EAGLE | accepted tokens、draft latency、metadata copy |
| Serving | TTFT、TPOT、P50/P95/P99、显存、长上下文稳定性 |

## 13. MUSA 优化的最终主线

```text
MUSA TopK / Fused Gate
        ↓
S5000 Fixed-Bucket Compact / DeepGEMM Permute
        ↓
Gate/Up GEMM
        ↓
Masked SwiGLU + FP8 Quant
        ↓
Down GEMM / Reorder
        ↓
C4 Indexer + RoPE + Hadamard
        ↓
FlashMLA Seq-Pack + Chunk-Level Workspace Cache
        ↓
C4/C128 Compress + Norm/RoPE/Cache Fusion
        ↓
Decode/Prefill Cache Store 分流
        ↓
MUSA Decode Attention / Paged Decode
        ↓
MTP/EAGLE Graph + No-Seed Sampling
        ↓
Online C128 / PP / PD / HiSparse
        ↓
MUSA Runtime Hardening
```

## 14. 当前性能数据口径

本文档只在有实际 MUSA benchmark 证据时使用“实测提升”。需要同时记录：

```text
GPU 型号和数量
模型版本与 FP8 配置
Token 数、Batch、Concurrency
TP/EP/CP/PP 配置
Baseline commit
Optimized commit
Median/P95
Kernel trace
Fallback 状态
```

如果只有代码、benchmark budget 或静态分析，应使用：

```text
预期降低
目标提升
结构性收益
```

不要将官方纯 CUDA/NVIDIA 实现的性能结果直接归因于 MUSA 优化。

## 15. 建议阅读路径（按 Commit Snapshot）

`sglang-musa` 当前工作树早于多项特性，以下路径应通过左侧提交读取，例如 `git -C sglang-musa show cc5b01a55:<path>`：

1. `b31ab813e:python/sglang/srt/layers/moe/deepseek_v4_topk.py`
2. `b31ab813e:python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/topk_ops.py`
3. `cc5b01a55:python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`
4. `cc5b01a55:python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/moe_prefill_ops.py`
5. `f393eaa98:python/sglang/srt/layers/quantization/fp8_kernel.py`
6. `d5fe7707c:python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`
7. `d5fe7707c:python/sglang/srt/layers/attention/deepseek_v4_backend.py`
8. `7c992a2a8:python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/cache_ops.py`
9. `5eab15edc:python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/norm_rope_ops.py`
10. `ea3e93c38:python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/mhc_ops.py`
11. `bb3e58863:python/sglang/srt/speculative/eagle_worker_v2.py`
12. `6d8e20f98:python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`
13. `66464a505:python/sglang/srt/disaggregation/dsv4_pd_disagg.py`
14. `cc5b01a55:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_moe_prefill_deepgemm_benchmarks.py`

当前官方基线则直接阅读 `sglang@5c6635d8f3` 的同语义模块；路径改名时以 `git log -S/-G` 追踪，不以文件名相同作为来源判据。

## 16. 说明

本版本按 `O/P/T/R/S` 重构归属：纯官方能力只作为基线；MUSA 功能移植、平台调优、S5000 Kernel/Pipeline 重写及 Runtime/系统适配分别统计。所有历史实现以对应 commit snapshot 为准，性能数字按证据等级表达；本文未执行 MUSA 实机 benchmark。

## 17. 面试表达与技术深挖手册

> **本章用途**：第 1～16 章是技术证据线，用于回答源码、提交、调用链、归属和验证细节；第 17 章是面试表达线，用于把这些证据组织成可在 30 秒、3 分钟和 10 分钟内讲清楚的项目故事。本章不改变前文结论，也不新增未经验证的性能数字。

### 17.1 项目定位：一句话先讲清“我们做了什么”

推荐开场：

> 我们做的是 DeepSeek V4 在 MUSA S5000 上的高性能推理适配。官方已经提供模型语义、SGLang serving 框架以及 DeepGEMM、FlashMLA、MTP/EAGLE 等基础能力；我们的重点不是重新发明这些算法，而是在保持官方数值与接口语义的前提下，完成 MUSA backend bring-up，并针对 S5000 的高脊点、相对低带宽和 Decode Small-M 特征，自主重写一批 Kernel、数据组织、融合 Pipeline 和 Runtime/Serving 闭环，最终把优化从算子传导到 TTFT、TPOT、吞吐、显存和长稳。

这句话包含四层边界：

```text
官方已有：模型/算法语义、通用框架、NV 基线
我们移植：device/dtype/layout/compiler/backend correctness
我们重写：MUSA/S5000 Kernel、数据组织、Fusion、Dispatch
我们闭环：Graph、JIT、Fallback、通信、PP/PD、状态和服务验证
```

面试中不要把“自主研发”理解成所有数学算法都由团队提出。更准确的表述是：

> 我们自主研发的是 MUSA/S5000 高性能执行路径，而不是 DeepSeek V4、FlashMLA、TopK、FP8 Quant 或 MTP 的数学定义。

### 17.2 把 `O/P/T/R/S` 翻译成面试语言

| 技术标签 | 面试表达 | 典型例子 | 不应如何表达 |
|---|---|---|---|
| `O` 官方原生 | “这是我们复用和对齐的官方基线” | Expert-Contiguous、FlashMLA sparse 语义、MTP | “这是我们发明的算法” |
| `P` 功能移植 | “我们让官方语义在 MUSA 上正确运行” | device/dtype/layout、JIT/FFI 接入 | “只改了一个设备字符串” |
| `T` 平台调优 | “我们针对 S5000 调整并行参数和触发阈值” | warp/tile/stage/vector、shape gate | “所有 Shape 都更快” |
| `R` 平台重写 | “我们保持接口/数值语义，重写了 Kernel 或 Pipeline” | Compact Quant Scatter、Seq-Pack、Queue4 | “整个上层特性完全自研” |
| `S` 系统适配 | “我们打通 Graph、通信、状态和服务生命周期” | Online C128、PP/PD、Fail-Closed | “只是外围工程，不影响性能” |

一个优化可以同时含多类贡献。例如 MoE Prefill 的标准说法是：

```text
O：复用官方 Expert-Contiguous 与 Grouped GEMM 语义
P：完成 MUSA FP8、Permute 和 Runner 接入
R：重写 Fixed-Bucket + Compact Quant Scatter Pipeline
T：建立 Prefill/Decode 阈值、容量和 Shape Dispatch
```

这种拆分既不会把官方成果据为己有，也不会把真正的 MUSA 重写弱化成“简单移植”。

### 17.3 三种时长的项目介绍

#### 17.3.1 30 秒版本

> 官方 SGLang 已经具备 DeepSeek V4 的 MoE、Sparse Attention、压缩 KV 和 MTP 等能力，但其高性能路径主要围绕 NVIDIA/CUDA 生态设计，直接迁移到 S5000 后会遇到编译与后端不兼容、内存访问和数据布局不匹配、Decode Small-M launch 开销高，以及 silent fallback、Graph 和通信状态不闭环等问题。我们先建立官方 NV、MUSA 原始移植和 MUSA 优化路径三组基线，再按 Compute、Memory、Launch/Runtime 三类瓶颈归因，保留官方数值语义，重写 MUSA Kernel、数据组织和 Runtime。代表工作包括 Compact MoE、Half-Warp FP8 Quant、Sparse Seq-Pack、Cache No-Copy、Small-M Decode 和 Online C128/PP/PD 闭环。

#### 17.3.2 2～3 分钟版本

> 这个项目的目标，是让 DeepSeek V4 的官方 serving 能力在 MUSA S5000 上不仅“能跑”，而且真正命中适合本地硬件的高性能路径。官方已经实现了模型结构、Expert-Contiguous Grouped GEMM、FlashMLA Sparse Attention、C4/C128、MTP/EAGLE 和 PP/PD 等上层语义，我们把这些作为 baseline，而不是算作本地原创。
>
> 直接迁移不够主要有四类原因。第一，部分路径依赖 CUDA/NCCL、CUDA Graph 或 NVIDIA 专用 Kernel，MUSA 上存在编译和 Runtime 兼容问题；第二，S5000 高脊点、相对低带宽，一些在 NV 上可接受的中间 tensor、scatter 和 cache materialization 会更容易变成 Memory-bound；第三，Decode 的 M 很小，Python wrapper、Kernel launch 和 metadata 构建会吞掉算子收益；第四，silent fallback、JIT cache collision 或设备绑定错误会让 benchmark 以为测到了新 Kernel，实际执行的是旧路径。
>
> 我们的定位方法不是直接比较 NV official 和 MUSA tuned，而是建立三段基线：NV official、MUSA official-port、MUSA tuned。然后把测量拆成 Kernel-only、host-inclusive、module 和 server 四层，通过 Shape、dtype、bytes、launch count、fast-path hit、fallback reason 和 Graph replay 判断瓶颈属于 Compute、Memory 还是 Launch/Runtime。
>
> 方案上，我们保留官方输入输出和数值语义，但按问题选择不同策略：MoE Prefill 重写 Fixed-Bucket Compact Quant Scatter，减少碎片行和中间落盘；FP8 Quant、Compress 和 Cache 使用 Half-Warp、parallel reduce、no-copy 与 Prefill/Decode 双路径；Sparse Prefill 重写 Seq-Pack、index rebase 和 chunk cache；Decode 针对 Small-M 增加 Queue4、Prefix-Tail、Page32 和 Split-K；Runtime 则补齐 JIT specialization key、MCCL device binding、Graph-safe fallback、PP/PD 和 Online C128 状态生命周期。
>
> 我们不只看单 Kernel。Fast Path 必须通过 dispatch trace、specialization key、fallback=0、Graph capture/replay 和 reference correctness 证明真实命中；收益也要依次验证 Operator、Module、Model/Service 是否传导。目前文档中 MoE 约 `1.6×`、端到端 Prefill `+8%` 属于用户/项目口径的证据 B，不是本文复测；其他没有完整日志的部分只描述结构性收益和待测指标。

#### 17.3.3 5～10 分钟版本：按三类瓶颈展开

面试官允许展开时，不要按提交时间罗列，而按性能归因讲三条线。

**第一条：Compute-bound——让大 GEMM 吃到规则的 M。**

```text
官方基础：Expert-Contiguous + m_indices + Grouped GEMM
本地问题：expert M 碎片化、padding/容量不稳定、前处理占比高
定位：随 Token/M 增大 GEMM 吞吐提高；前处理与无效行占比明显
方案：Fixed-Bucket、Compact Rows、Quant+Scatter、Overflow Guard
验证：有效/分配/最坏行数、两段 GEMM、MoE block、Prefill TTFT
```

**第二条：Memory-bound——先减少 Bytes，再调线程。**

```text
对象：FP8 Quant、C4/C128 Compress、Cache、Norm/RoPE、Sparse metadata
本地问题：中间 tensor、重复 HBM 往返、非连续访问、隐式 contiguous
定位：latency 随 bytes 近似增长；计算利用率低；融合前后 launch/bytes 可计数
方案：Half-Warp、Parallel Reduce、No-Copy、Vector Store、Seq-Pack、Fusion
验证：derived GB/s、materialization 次数、cache latency、module/TTFT
```

**第三条：Launch/Runtime-bound——证明新 Kernel 真的在线上执行。**

```text
对象：Decode Small-M、MTP、Sampling、Ops Facade、Graph、Distributed
本地问题：Kernel 短，host dispatch 占比高；fallback/JIT/Graph 可能走错路
定位：Kernel-only 快但 host-inclusive/TPOT 不变；trace 显示 fallback 或 graph miss
方案：Small-M Kernel、减少 wrapper、预热 JIT、Graph-safe、Fail-Closed、MCCL binding
验证：host-inclusive、TPOT、fast-path hit、fallback=0、capture/replay、P95/P99
```

最后收束：

> 这三条线共同说明，我们做的不是把 CUDA 文件改成 MUSA，而是从官方特性评估、平台基线、瓶颈归因，到 Kernel/Pipeline 重写和服务验收的完整闭环。

### 17.4 每个优化都必须回答的八个问题

面试时可把任何技术点放入以下模板。

#### 17.4.1 官方已经实现了什么

回答要包含：

```text
官方特性解决的业务问题
官方调用链和默认 Kernel
官方数据布局和 Fast Path 条件
官方 fallback、Graph、通信约束
开发时基线与当前官方基线
```

合格示例：

> 官方已经有 Expert-Contiguous、`m_indices` 和 contiguous grouped GEMM，解决不同 expert token 如何连续进入两段 GEMM；我们没有重写这部分数学语义。

不合格示例：

> 官方有一个 MoE，性能不太好。

#### 17.4.2 为什么直接迁移到 S5000 不够

不要笼统说“硬件不一样”，而要指明具体不匹配：

```text
编译/设备：CUDA-only、dtype、FFI、MCCL
计算映射：Tile/warp/stage 与 S5000 不匹配
数据组织：额外 materialization、碎片 M、随机访存
Workload：Prefill 大 M 与 Decode Small-M 共用通用路径
Runtime：Python dispatch、JIT collision、Graph miss、silent fallback
系统状态：CP/PP/PD page 坐标、MTP accepted state、异步 transfer
```

#### 17.4.3 我们怎么定位瓶颈

推荐固定回答顺序：

```text
1. 锁定相同模型、Shape、dtype、batch、并行配置。
2. 建立 NV official / MUSA port / MUSA tuned 三组基线。
3. 分离 Kernel-only 与 host-inclusive。
4. 记录 bytes、FLOPs、launch、workspace、materialization。
5. 检查 dispatch branch、specialization key、fallback reason、graph state。
6. 从 Operator 继续向 Module、Prefill/Decode 和 Server 追踪。
```

诊断逻辑：

| 现象 | 更可能的根因 | 下一步 |
|---|---|---|
| 增大 M 后吞吐明显改善 | Compute/occupancy | 调数据规整度和 Tile |
| latency 与 bytes 近似线性 | Memory-bound | 减 materialization/融合 |
| Kernel 快、host-inclusive 慢 | Launch/dispatch | 查 wrapper、sync、Graph |
| 单测快、服务不快 | 收益未传导 | 查 Scheduler、通信、其他阶段 |
| 冷启动或切 Shape 异常 | JIT specialization | 查 key、预热和 cache |
| 日志看似成功但性能异常 | silent fallback | 强制 fail-closed/trace |

#### 17.4.4 我们保留了哪些官方语义

面试官实际在确认改动是否可维护、是否可能破坏正确性。回答应覆盖：

```text
输入输出 dtype/shape
数学定义与数值容差
Expert/Token 顺序
Page/Cache 的逻辑语义
MTP accepted/rejected token 语义
PP/PD 请求与 layer ownership
上层 Python/backend 接口
```

标准表达：

> 我们保留官方接口、数值定义和状态语义，只替换执行层；这样 NV 继续走官方 Kernel，MUSA 走 S5000 专用实现，并可共享上层 correctness 与 serving 流程。

#### 17.4.5 我们替换或重写了哪些执行路径

按改动强度回答：

```text
参数调优：Tile/warp/stage/vector/threshold
Kernel 替换：同接口的 MUSA TileLang/Triton/CUH Kernel
Pipeline 重写：改变中间布局、融合边界、workspace 生命周期
系统重写：Graph、JIT、通信、状态提交、fallback policy
```

必须讲“为什么需要这一层改动”。例如：

> 如果瓶颈只是默认 warp 不合适，我们调参数；如果存在额外 HBM 中间落盘，调 warp 无法消除 bytes，就必须重写 Pipeline。

#### 17.4.6 如何证明真正命中

至少给出以下证据链：

```text
dispatch guard 满足
  -> backend/op 名称正确
  -> JIT specialization key 正确
  -> Kernel launch 可观测
  -> fallback reason 为空
  -> graph capture/replay 使用同一目标路径
  -> 输出与 reference 一致
```

最有说服力的测试不是“运行不报错”，而是：

- monkeypatch 旧 fallback 使其一旦调用就抛错；
- 交叉运行不同 JIT Shape，证明 artifact 不碰撞；
- graph replay 输入变化后输出随之变化，证明没有读旧 metadata；
- 记录 Fast-Path Hit Rate 和 Unexpected Fallback；
- 同时测 direct Kernel 与 host-inclusive wrapper。

#### 17.4.7 算子、模块和服务分别取得什么效果

不要把三层效果混成一个数字：

| 层级 | 问题 | 指标 |
|---|---|---|
| Operator | Kernel 本身是否更高效 | latency、GB/s、FLOPS、launch、workspace |
| Module | 上下游是否吃到收益 | MoE/Attention/Cache block latency、materialization |
| Model | Prefill/Decode 是否改善 | Prefill latency、Decode latency、TTFT、TPOT |
| Service | 客户负载是否改善 | tokens/s、concurrency、P50/P95/P99、OOM、长稳 |

标准表达：

> 算子优化是必要但不充分条件。只有 Operator 下降传导到 Module，再传导到 TTFT/TPOT 或吞吐，我们才把它称为端到端收益；否则会明确记录收益被 dispatch、通信或其他阶段抵消。

#### 17.4.8 适用范围和失败边界是什么

每个 Fast Path 必须说明：

```text
支持的 device/dtype/shape/page/group size
Prefill/Decode、Graph/Eager 条件
contiguous 或 stride contract
精确/近似语义
容量和 overflow
不支持时 fallback 还是 fail-closed
失败是否会污染 cache/state
```

标准表达：

> 我们不会把专用 Kernel 扩大到未经验证的 Shape。生产路径默认 fail-closed 或进入明确受测的 fallback；Graph 中禁止 Torch fallback；Debug fallback 必须记录 reason，且其结果不能计入 Fast-Path benchmark。

### 17.5 回答质量自检表

讲完一个优化后，用下面十项快速自检：

- [ ] 是否明确了开发时官方 baseline，而不是只看当前代码？
- [ ] 是否区分官方语义和 MUSA 自主实现？
- [ ] 是否给出具体 Shape、dtype、page 或 workload？
- [ ] 是否说明瓶颈属于 Compute、Memory 还是 Launch/Runtime？
- [ ] 是否解释为何选择调参、换 Kernel 或重写 Pipeline？
- [ ] 是否说清 Fast Path 的 dispatch guard？
- [ ] 是否证明没有 silent fallback？
- [ ] 是否同时关注 Kernel-only 与 host-inclusive？
- [ ] 是否区分 Operator、Module、Service 效果？
- [ ] 是否给出不支持条件、fallback 和失败边界？

### 17.6 深挖案例一：S5000 Fixed-Bucket Compact MoE Prefill

> **技术证据入口**：第 3 章；主要 snapshot 为 `cc5b01a55`，官方基线见 `c2942907d5`、`acc816d8a2`。

#### 17.6.1 官方已经实现了什么

官方已经具备完整的 Expert-Contiguous 计算语义：Router/TopK 给出 expert assignment，DeepEP normal 或 reorder 把 token 按 expert 连续组织，`m_indices` 标记每行所属 expert，两段 `grouped_gemm_nt_f8f8bf16_contig` 完成 Gate/Up 和 Down GEMM，中间执行 SwiGLU 与量化。

我们明确保留：

```text
TopK 与 routing 数学语义
Expert-Contiguous 行语义
m_indices 含义
Grouped GEMM 输入输出契约
Gate/Up -> SwiGLU -> Down 的模型计算顺序
```

因此项目起点不是“重新实现 MoE”，而是分析官方 contiguous baseline 在 S5000 客户 Prefill workload 上，GEMM 周边是否成为新的瓶颈。

#### 17.6.2 为什么直接迁移不够

大 Prefill 中不同 expert 的 token 数不规则。即使 Grouped GEMM 本体高效，前后的 count/prefix、quant、scatter、padding、SwiGLU quant 和 combine 仍可能产生：

- expert M 碎片化，GEMM tile 利用不稳定；
- 先按原顺序量化、再 permute/scatter 的额外 FP8 和 scale 落盘；
- 动态容量、padding 与 workspace 不规则；
- 热 expert 超出固定容量时存在 correctness 风险；
- 大 Prefill 方案若误用于 Decode，会让固定调度成本恶化 TPOT。

这里不是“官方设计错误”，而是官方通用能力没有自动解决 S5000 的 workload specialization。

#### 17.6.3 我们如何定位

面试中按下面的测量链回答：

```text
总 MoE block
  -> route count/prefix
  -> quant/scatter
  -> Gate/Up GEMM
  -> SwiGLU/quant
  -> Down GEMM
  -> combine
```

记录每个 expert 的有效 M、padded M、allocated rows、worst-case rows、padding ratio、overflow 次数，同时分别测 Kernel-only 与包含 Python/dispatch 的 device/host 时间。若 GEMM 随规则 M 增大明显提高吞吐，而 quant/scatter 与 padding 占比仍高，说明问题同时含 Compute 规整度与 Memory/Launch 成本，不能只调 GEMM tile。

公平 A/B 是：

```text
同一 S5000、同一 tokens/expert/top-k/dtype
MUSA official-port contiguous baseline
vs
MUSA fixed-bucket compact tuned path
```

而不是只拿 NV official 与 MUSA tuned 比较。

#### 17.6.4 我们替换或重写了什么

我们保留官方 contiguous GEMM，在其前后重写数据组织：

```text
route count/prefix
  -> fixed/static-cap expert bucket
  -> 生成 src2dst 与 m_indices
  -> 读取 BF16 token
  -> group amax/scale + FP8 cast
  -> 按 src2dst 直接写 compact_input/compact_scale
  -> 官方语义的两段 Grouped GEMM
  -> masked SwiGLU quant
  -> post-combine
```

核心不是把所有阶段融合成一个 Kernel，而是把 **quant + scatter** 合并，减少一次中间落盘；同时引入 capacity、overflow 和 Prefill/Decode dispatch，保证专用路径只覆盖验证过的大 Token Shape。

#### 17.6.5 如何证明真正命中

- 开启实验开关并满足 `tokens >= 8192` 等 guard；
- trace 确认进入 compact MoE op，而不是普通 permute/quant 路径；
- 记录 `valid_routes/padded_valid_rows/allocated_rows`；
- overflow case 强制发生，确认完整 fallback 而非丢 route；
- monkeypatch fallback，验证正常 hot shape 不调用它；
- 比较 compact 前后 token/expert 映射和最终输出；
- 分别报告两段 GEMM、前后处理、MoE block 和 Prefill。

#### 17.6.6 效果、范围和失败边界

当前可用结果是用户/项目口径：MoE 算子约 `1.6×`、端到端 Prefill 约 `+8%`，按证据 `B` 表述。面试时必须补充：本文没有复现该数字，完整硬件、Shape、TP/EP、baseline commit 和 P95 尚需原始记录支持。

适用范围：大 Token Prefill、指定 FP8/group/top-k/layout、容量未 overflow。失败边界：small-M Decode 不进入；dtype/shape/layout 不满足时显式 fallback；overflow 不能静默截断；Candidate fused gate 未经完整验证不能说成生产默认。

#### 17.6.7 90 秒回答示例

> 官方已经有 Expert-Contiguous 和 Grouped GEMM，我们保留了它的 routing、`m_indices` 和两段 GEMM 语义。迁移后发现，大 Prefill 的瓶颈不只在 GEMM，而在 expert M 碎片、padding，以及 quant 和 scatter 分开造成的中间 HBM 写回。我们把 MoE block 拆阶段 profile，并记录有效行、分配行和 overflow，确认需要同时提高 GEMM 输入规整度和减少前处理 bytes。最终在 S5000 上增加 Fixed-Bucket Compact Pipeline，把 group quant 与 scatter 融合，直接生成 grouped GEMM 需要的 compact FP8 和 scale，同时为热 expert 加 capacity/overflow guard，并限制只在大 Token Prefill 命中。命中通过 dispatch trace、compact row metadata、fallback=0 和 reference correctness 证明。当前项目口径是算子约 1.6 倍、Prefill 约 8% 收益，但这属于证据 B，我们不会说成本文复测，也不会把官方 Grouped GEMM 算成自研。

### 17.7 深挖案例二：FP8 Quant、C4/C128 Compress 与 Cache Store

> **技术证据入口**：第 5、7、10 章；代表 snapshots 为 `f393eaa98`、`a18fcbf07`、`7c992a2a8/c2eafb495`、`fc1e88584/392179e92`。

#### 17.7.1 官方已经实现了什么

官方定义了 FP8 per-token/group quant、C4/C128 compression、SwiGLU 和 KV/Indexer/FlashMLA cache 的数据语义。我们保留量化公式、scale 含义、压缩比、page/cache logical mapping、RoPE/NoPE pack 和上层 setter 接口。

#### 17.7.2 为什么直接迁移不够

这些算子 FLOPs 不高，却频繁读写 activation、scale、state 和 cache，是 S5000 高脊点环境中的典型 Memory-bound 区域。通用实现可能包含：

```text
整行归约宽度与本地 warp 映射不匹配
多个 group 逐次处理，launch/occupancy 不稳定
C128 reduce 串行度高
Norm/RoPE 后产生临时 tensor
cache store 前隐式 contiguous/materialization
Prefill 大 M 与 Decode small-M 共用同一写入策略
```

#### 17.7.3 我们如何定位

先计算理论 bytes 和 derived GB/s，再对照 latency 随 rows/hidden/page 的变化；用 allocation/trace 检查 `.contiguous()`、临时 pack 和重复 store；拆开 Norm、RoPE、Quant、Compress 和 Cache Store；最后比较 direct Kernel 与 wrapper。若增加线程不改善，而减少一次中间落盘直接降低 latency，说明应该先改数据流而不是继续堆并行度。

#### 17.7.4 我们替换或重写了什么

- FP8 Quant：Half-Warp group mapping、16K-group/shape-aware dispatch、TileLang/Triton multi-group；
- C128：parallel reduce 与 boundary specialization；
- C4：vector write、paged/page4 specialization；
- Norm/RoPE/Cache：融合或直接 handoff，减少中间 materialization；
- FlashMLA Cache：Decode x4/vec2 与 Prefill tile-parallel/subwarp 双路径；
- Indexer Cache：多 page store、row-stride-aware no-copy；
- Production guard：destination layout 不满足则 fail-closed，不用隐式 copy 掩盖。

这里需要强调：no-copy 不是“所有 non-contiguous 都拒绝”。Packed cache destination 若 Kernel 要求连续必须拒绝；source 若 Kernel 显式接收 row stride，则可 no-copy 支持。

#### 17.7.5 如何证明真正命中

```text
shape/dtype/stride guard
  -> 对应 Quant/Compress/Cache specialization
  -> 无 .contiguous() 和临时 allocation
  -> trace 中无 Torch/reference fallback
  -> cache byte layout 与 reference 一致
  -> graph capture/replay 可重复
```

测试要覆盖 contiguous 与支持的 row-stride、非法 destination layout、page1/page4、Prefill/Decode、cold/warm JIT。Fast Path miss 的异常应携带 device/dtype/shape/stride/page metadata。

#### 17.7.6 效果、范围和失败边界

当前文档中，Prefill Cache Store 约 `1.8×` 是用户/项目口径，按证据 `B` 表述；完整硬件、Shape、baseline 和日志仍需补齐，不能说成本文复测。其余大部分 Quant/Compress/Norm-RoPE 优化只有代码与 benchmark framework，按证据 `C/D`，不能报统一加速倍数。可验证的结构性效果是减少 HBM 往返、临时 allocation 和通用路径开销；需要补齐的指标是 Kernel latency、derived GB/s、materialization 次数、cache block latency、TTFT/TPOT 传导。

失败边界包括非法 dtype/page/layout/stride、Graph 中 Kernel miss、JIT specialization 错误，以及 Debug fallback 被误计入生产 benchmark。

#### 17.7.7 90 秒回答示例

> FP8 Quant、C4/C128 和 Cache Store 的共同特点是计算量低、数据搬运多，所以我们先按 Memory-bound 路径分析，而不是先调大 tile。官方的量化和 cache 语义保持不变，我们测 bytes、derived GB/s、临时 allocation 和 host-inclusive latency，定位到通用 group 映射、串行 reduce、Norm/RoPE 中间 tensor，以及 cache store 的隐式 materialization。对应地，我们实现 Half-Warp Quant、C128 Parallel Reduce、Prefill/Decode cache 双路径和 stride-aware no-copy，并在生产路径对不支持的 destination layout fail-closed。是否命中通过 specialization trace、fallback=0、无 `.contiguous()`、cache byte reference 和 graph replay 证明。当前没有统一可复现的 S5000 数字，所以面试中会讲清结构性收益和验收指标，而不会虚构加速比。

### 17.8 深挖案例三：FlashMLA Sparse Prefill Seq-Pack

> **技术证据入口**：第 6 章；主要 snapshot 为 `d5fe7707c`、`1ce2e4aae`、`373af4d99`。

#### 17.8.1 官方已经实现了什么

官方提供 `flash_mla_sparse_fwd` 的 Sparse Attention 数学语义和 Kernel 接口，定义 query、combined index、cache/page metadata 如何进入 sparse attention。当前官方后续也具备 chunk/cache 类能力，但这不能倒推为 MUSA 开发时没有独立工作。

#### 17.8.2 为什么直接迁移不够

Sparse Kernel 快不代表完整 Prefill 快。调用前仍需按 request/chunk 处理 index、page、position、padding 和 CP 分片；若每层重复构建 workspace、保留 raw index fallback 或执行小而散的 gather，metadata preparation 会成为 TTFT 的显著部分。多请求变长时，绝对 index、request-local index、packed index 和 CP rank index混用还可能产生 correctness 问题。

#### 17.8.3 我们如何定位

将流程拆成：

```text
raw/combined index preparation
  -> request/chunk metadata
  -> workspace pack/index rebase
  -> flash_mla_sparse_fwd
  -> output unpack
```

分别统计 metadata build、workspace allocation、跨层重建次数、Kernel latency、TTFT；按 request 数、sequence length、chunk size、Pack8/padding 和 CP rank变化做扩展性测试。若 sparse Kernel 时间稳定而 TTFT 随 metadata 重建增长，瓶颈就在 Kernel 外。

#### 17.8.4 我们替换或重写了什么

我们保留 FlashMLA sparse 接口与 attention 数值，重写其输入准备 Pipeline：

- Seq-Pack 将不同 request/chunk 组织进 flat workspace；
- index rebase 把各坐标转换为 Kernel 需要的 packed 坐标；
- `SparsePrefillChunkCache` 跨层复用可共享 metadata；
- Pack8 和 request padding 满足本地向量/布局要求；
- CP round-robin 下显式处理 combined index；
- 删除未经证明安全的 raw-index silent fallback。

#### 17.8.5 如何证明真正命中

- trace 显示 Seq-Pack 而非 raw-index 路径；
- workspace/cache build count 不随层数重复增长；
- 多 request 的 index rebase 与 dense/reference 结果一致；
- Pack8 padding 行不参与有效 attention；
- CP 每个 rank 的 index 子集无重叠、无遗漏，重组输出一致；
- graph/eager 与不同 chunk bucket 均覆盖。

#### 17.8.6 效果、范围和失败边界

效果应分 metadata build、Sparse Attention module 和 TTFT 三层。当前归属为 `O+P+R+S`，性能证据为 `B/C`；没有完整日志的部分不能报统一倍数。适用边界包含特定 sparse backend、pack/chunk/page layout、CP split 与 cache key；unsupported metadata 必须拒绝或走明确 reference，不能把 raw index 当作 combined index。

#### 17.8.7 90 秒回答示例

> 官方已经有 FlashMLA Sparse Attention Kernel，但迁移后我们发现端到端 Prefill 不只受 Kernel 控制，request/chunk metadata、workspace、index rebase 和 CP 分片可能每层重复发生。我们把 Sparse Prefill 拆成 metadata、pack、Kernel 和 unpack 四段，发现需要优化的是 Kernel 前的数据组织。因此保留官方 attention 语义，重写 Seq-Pack flat workspace、combined index rebase、跨层 chunk cache、Pack8/padding 和 CP round-robin。命中通过 build count、workspace identity、index reference、CP 重组和 TTFT 分段证明。这个案例体现的是 Pipeline 重写，而不是声称 Sparse Attention 算法由我们提出。

### 17.9 深挖案例四：Decode Small-M、Paged Cache 与 MTP Graph

> **技术证据入口**：第 8、10 章；代表 snapshots 为 `8094ab938`、`17be8f1bf`、`81fc824a2`、`bb3e58863`、`5742d2df8`。

#### 17.9.1 官方已经实现了什么

官方提供 Paged Attention、Decode、MTP/EAGLE、Sampling 和 Graph 框架。我们保留 token/page 语义、draft/target verify、accepted token、sampling correctness 和上层 forward mode。

#### 17.9.2 为什么直接迁移不够

Decode 每步 M 很小，通用 Prefill Kernel、较大固定 workspace、Python backend 判断和多次 launch 都可能比有效计算更贵。Paged metadata 必须在 Prefill store 与 Decode consume 两端一致；MTP 又增加 draft/extend/verify metadata 生命周期。若 graph capture 中发生 Torch fallback、动态编译或旧 metadata 固化，可能既慢又错。

#### 17.9.3 我们如何定位

同时测：

```text
Kernel-only latency
Python wrapper/host-inclusive latency
每 token launch 数
Graph capture 与 replay 时间
Paged metadata build/copy
MTP draft/verify latency与 accept rate
端到端 TPOT/P95/P99
```

典型判断是：Kernel 已明显变快但 TPOT 不变，说明 host dispatch、Graph miss 或其他 Decode 阶段成为瓶颈；不能继续只优化 Kernel。

#### 17.9.4 我们替换或重写了什么

- Queue4、Prefix-Tail、Page32、Head256 等 hot-shape 路径；
- Small-M cache x4/vec2；
- HC-Head Split-K；
- Paged Prefill/Decode metadata 配套；
- MTP/EAGLE draft/extend/verify graph metadata 生命周期；
- Ops facade 减少 per-call patch scan；
- capture 前 JIT 预热、buffer/proxy 固定；
- Graph 内禁止 Torch fallback。

#### 17.9.5 如何证明真正命中

- 对每个 bucket 记录 Kernel 名、page layout、graph id 和 replay count；
- direct Kernel 与 host-inclusive 都下降；
- capture 后用两组不同输入 replay，输出随输入变化并与 eager/reference 一致；
- fallback 被 monkeypatch 为抛错，正常 graph replay不触发；
- paged Prefill 写入与 Decode 读取 byte/page 对齐；
- MTP accepted/rejected/bonus token 的 cache/state 更新正确；
- 最终 TPOT 和 P95/P99 而非只看微基准。

#### 17.9.6 效果、范围和失败边界

当前效果多为 `B/C`，应以 Small-M latency、host dispatch、TPOT、Graph success/replay 和 fallback=0 表达，缺少日志时不写统一倍数。范围由 batch/M、head/page、graph bucket、dtype 和 metadata layout 限定。失败边界包括 unsupported page、bucket 变化、capture 中 JIT miss、Torch fallback、旧 proxy metadata，以及 allocator over-free/state 污染。

#### 17.9.7 90 秒回答示例

> Decode 的关键问题是 Small-M，Kernel 很短，所以通用 Kernel 和 Python/launch 开销会占很大比例。官方 Paged Attention 和 MTP 语义保持不变，我们同时测 Kernel-only、host-inclusive、launch count、graph replay 和 TPOT，避免只优化微基准。针对 hot shape，我们实现 Queue4、Prefix-Tail、Page32、Split-K 和 Small-M cache，并把 Paged Prefill/Decode、MTP metadata 与 graph bucket配套；Runtime 上禁止 capture 中 Torch fallback并减少 facade per-call dispatch。命中必须有 Kernel trace、graph id、fallback=0 和双输入 replay correctness，最终看 TPOT/P95，而不是只看一个 Kernel 数字。

### 17.10 深挖案例五：Online C128、PP/PD、HiSparse 与 Runtime Hardening

> **技术证据入口**：第 9、10 章；主要 snapshots 为 `66464a505`、`1dbba2604`、`6e7f541ae`、`3cceb08b9`、`ef4469cb5`、`90b4437a8`。

#### 17.10.1 官方已经实现了什么

官方定义 Online C128、MTP、PP/PD、HiSparse、distributed process group 和 backend/JIT/fallback 框架。我们保留请求、cache、accepted token、layer ownership、传输与压缩状态语义。

#### 17.10.2 为什么直接迁移不够

这一层的问题不再是单 Kernel：

- Full KV、SWA、request-relative page 和 CP-rank page 是不同坐标；
- MTP verify 中 rejected draft 不能提前提交 C128 state；
- PP stage 只拥有部分 layer，而 pointer table按压缩类别分段；
- 异步 transfer/offload 后，consumer 必须等待正确 layer/event；
- HiSparse CUDA helper 不能接收 MUSA pointer；
- JIT key、MCCL device 和 forwarding patch 任一错误都会让正确 Kernel 无法稳定执行。

#### 17.10.3 我们如何定位

除 latency 外，更重要的是建立状态与可观测性：

```text
request/page 坐标映射
accepted row 与 state-store 次数
PP layer range 与 pointer count
transfer counter/event
HiSparse pending/completed
JIT key 与 compile count
rank/local-rank/current-device
fast-path/fallback reason
```

用失败注入复现 metadata 错长、accepted row 越界、layer event 延迟、page1/page4 交叉 JIT 和错误 local rank。复杂系统中，能够稳定失败比 silent corruption 更重要。

#### 17.10.4 我们替换或重写了什么

- CP/PD：Full/SWA pair、rank page subset 与 relative-page destination align；
- MTP/Online C128：verify 只缓存 `kv_score`，accepted 后逐 token deferred commit；
- PP：按 `prefill_start/end_layer` 和 C4/C128/SWA layout切 pointer；
- Transfer：`wait_layer_transfer` 建立 per-layer happens-before；
- HiSparse：MUSA TileLang pointer-table host-offload Kernel 与输入 guard；
- JIT：`extra_data_cols` 进入 specialization name，隔离 page1/page4；
- MCCL：在 `init_process_group` 前为合法 local rank构造 MUSA `device_id`；
- Forwarding：正常调用不再每次扫描 patch，`setattr` 时即时同步；
- Production：Graph/unsupported layout默认 fail-closed。

#### 17.10.5 如何证明真正命中和正确

- page transform 用纯函数 case 验证值、顺序和 dtype；
- MTP accepted/rejected/bonus row 与 127→128 边界逐项验证；
- PP compact/SWA/draft pointer layout覆盖首、中、尾 stage；
- transfer 乱序完成，consumer 不早读且不等错 layer；
- HiSparse 非 MUSA/device/dtype/layout 输入显式失败；
- page1→page4→page1 与反向顺序无 JIT collision；
- MCCL kwargs 和 rank-device mapping 可观测；
- forwarding monkeypatch 在 facade/target/domain 同步；
- Graph replay 与长稳无陈旧 state、部分 transfer 或 fallback。

#### 17.10.6 效果、范围和失败边界

这部分主要按证据 `C/D`：效果首先是 Online C128/MTP、PP4 graph、PD/CP 和 HiSparse 在 MUSA 上形成可运行、可验证的状态闭环；性能指标包括可承载上下文、显存峰值、state rebuild/transfer、offload latency、Graph replay、TTFT/TPOT 和长稳，不能凭静态源码报统一加速倍数。

最关键的失败边界是：metadata 缺失时当前部分 helper 可能 warning/截断而非完全 fail-closed；accepted row 越界不能继续提交；裸 pointer layout 无法由 Kernel自行推断；异步 host offload 不表示 CPU 已可见；JIT/Graph fallback 不能静默进入生产数据。

#### 17.10.7 90 秒回答示例

> Online C128、PP/PD 和 HiSparse 体现的是系统闭环能力。官方有上层语义，但 MUSA 上需要处理 Full KV、SWA、请求相对页和 CP rank页四类坐标；MTP verify 还不能让 rejected draft 污染持久 C128 state。我们实现了 page pair和 rank align、accepted-token deferred commit、按压缩布局的 PP pointer slicing、per-layer transfer wait，以及 MUSA TileLang HiSparse offload。为了证明这些路径稳定执行，又补了 JIT layout key、MCCL device binding、forwarding patch 同步和 fail-closed。验证重点不是单个 Kernel 倍数，而是状态、pointer、event、fallback 和 replay：通过边界 token、乱序 event、交叉 JIT 和错误 metadata 注入，确保不是 silent corruption。当前这部分按证据 C/D 讲可运行性和验收指标，不报没有实测支持的性能数字。

### 17.11 三类岗位如何选择重点

用户希望兼顾 `1/2/3` 三类岗位，因此同一项目可按面试官方向切换重点，而不是准备三套互相矛盾的故事。

#### 17.11.1 AI Infra / 推理优化岗位

重点讲：

```text
三组公平 baseline
Operator -> Module -> Service 收益传导
TTFT/TPOT/吞吐/P95/P99
KV Cache、Graph、MTP、PP/PD
fallback、可观测性与长稳
客户 workload 和参数策略
```

优先案例：MoE Prefill + Decode/Graph + Online C128/PP/PD。

#### 17.11.2 GPU Kernel 岗位

重点讲：

```text
Roofline 与 Compute/Memory/Launch 分类
Shape、dtype、stride、page layout
Half-Warp、Tile、Parallel Reduce、Split-K
Vector load/store 与 materialization
JIT specialization 与数值 reference
为何调参不够、必须重写数据流
```

优先案例：Compact Quant Scatter + FP8 Quant/Compress/Cache + Small-M Decode。

#### 17.11.3 推理框架岗位

重点讲：

```text
Model -> Backend Router -> Op -> Kernel 调用链
Prefill/Decode 分流与 Paged metadata
Graph capture/replay 和 proxy tensor
MTP accepted state
CP/PP/PD 坐标、pointer 和 transfer
Fail-Closed、MCCL、JIT、forwarding
```

优先案例：Sparse Seq-Pack + Decode/MTP Graph + Runtime Hardening。

### 17.12 高频追问与建议回答

#### 追问 1：为什么不直接复用官方实现？

> 我们优先复用官方语义和能在 MUSA 上成立的实现，不为重写而重写。先做 MUSA official-port baseline；如果只是 tile/warp 参数不合适，就做平台调优；只有确认存在 CUDA-only 依赖、额外 materialization、数据布局或 Small-M 固定成本时，才替换 Kernel 或重写 Pipeline。每次选择都由 A/B、bytes/launch 和调用链证据支持。

#### 追问 2：如何证明收益不是 NV/MUSA 硬件差异？

> 关键对比是同一 S5000 上的 MUSA official-port 与 MUSA tuned，而不是只比较 NV official 和 MUSA tuned。NV/MUSA 相同逻辑对比用来理解硬件/编译器差异，本地前后对比才用于归因我们的代码收益。

#### 追问 3：为什么算子更快，端到端可能不变？

> 算子可能占比低，或收益被 Python dispatch、metadata、通信、Scheduler 和其他阶段抵消。我们使用 Kernel-only 与 host-inclusive，并沿 Operator、Module、Model、Service 四层追踪；没有传导到 TTFT/TPOT 的，只报告算子收益。

#### 追问 4：怎样防止 silent fallback？

> Production 默认 fail-closed；Debug fallback 必须由显式开关开启并记录 reason。测试中 monkeypatch fallback 为抛错，Graph 中禁止 Torch fallback，同时记录 backend/op、specialization key 和 Fast-Path Hit Rate。

#### 追问 5：最典型的隐蔽 Runtime 问题是什么？

> 一个例子是 `extra_data_cols` 没进入 JIT name。page1 和 page4 单独冷启动都可能正确，但交叉运行会复用错误 artifact。我们把 layout 参数加入 key，并用双向交叉顺序、cold/warm cache验证。它说明“cache hit”本身不是成功，必须命中正确 specialization。

#### 追问 6：有没有失败或负面实验？

> 有。通用思路是记录哪些候选在特定 Shape 变慢，并缩小 guard，而不是机械全量启用。例如大 Prefill 的 compact pipeline不应进入 Small-M Decode；Cache destination不满足 packed layout时不做隐式 contiguous；TopK exact/non-exact 不混用；算子快但 host-inclusive 不变时转向 dispatch/Graph，而不是继续堆 Kernel 优化。

#### 追问 7：如何保证重写没有破坏官方正确性？

> 保持上层接口、dtype/shape、数学定义和状态语义；用官方/eager reference 做数值比较，再加 layout、overflow、page、accepted token、CP/PP 和 Graph replay 的边界测试。性能 guard 和 correctness guard 分开，unsupported case不扩大覆盖。

#### 追问 8：这些都是团队成果还是个人成果？

> 这是团队完成的 MUSA/S5000 适配与优化体系。面试中我们按模块说明团队在官方基线之上完成的移植、Kernel/Pipeline 重写和系统闭环，不把仓库中所有提交归为单个人，也不把官方算法归为团队原创。如需要进一步拆分，应依据实际负责模块、代码评审和验证记录如实说明。

#### 追问 9：后续官方也实现了类似能力，如何看待？

> 我们用开发时基线和当前基线分开判断。后续上游出现同类能力不会抹掉当时的 MUSA 平台工作，但会改变维护策略：能收敛到官方接口就减少分叉，只保留 S5000-specific Kernel、参数和 guard；相同 patch-id 也只计一次实现。

#### 追问 10：如果让你再做一次，最先补什么？

> 最先补统一、可复现的 benchmark artifact：固定 baseline/optimized commit、硬件、Shape、并行配置、median/P95、trace 和 fallback 状态。现有技术链已经足以解释为什么优化，但证据 A 需要把用户结果、benchmark 和服务数据完整串起来。

### 17.13 效果数据表：面试前必须补齐

不要背孤立的“提升百分比”。每个数字至少关联 baseline、workload 和证据等级。

| 优化 | Baseline → Optimized | Workload/Shape | Operator | Module | Service | 证据 |
|---|---|---|---:|---:|---:|---|
| Compact MoE | MUSA contiguous → fixed-bucket compact | Token/expert/top-k/TP/EP 待补 | 约 `1.6×`（用户口径） | 待补 MoE block | Prefill `+8%`（用户口径） | `B` |
| FP8 Quant | 通用 group quant → half-warp/shape-aware | rows/hidden/group 待补 | 待补 | 待补 | 待补 | `C/D` |
| C4/C128 | baseline reduce/store → parallel/vector path | ratio/seq/page 待补 | 待补 | 待补 | 待补 | `C/D` |
| Sparse Seq-Pack | per-layer/raw prep → pack/cache | req/seq/chunk/CP 待补 | metadata 待补 | attention 待补 | TTFT 待补 | `B/C` |
| Cache Store | generic/materialize → no-copy 双路径 | M/page/layout 待补 | Prefill 约 `1.8×`（用户口径） | cache block 待补 | TTFT/TPOT 待补 | `B/C` |
| Decode Small-M | generic → Queue/Page/Split-K | batch/context/head 待补 | 待补 | decode block 待补 | TPOT/P95 待补 | `B/C` |
| Online C128/PP | baseline state/transfer → MUSA 闭环 | 32K/128K、CP/PP/PD 待补 | 不适用单一倍数 | transfer/state 待补 | 容量/显存/长稳待补 | `C/D` |

每个“待补”项必须来自可追溯日志；不允许在面试前凭印象填写。

### 17.14 可以说与不要说

| 可以说 | 不要说 |
|---|---|
| “我们基于官方语义重写了 MUSA execution path” | “DeepSeek V4 整套算法是我们自研的” |
| “官方提供 Grouped GEMM，我们重写其前后的 compact pipeline” | “Expert-Contiguous 是我们发明的” |
| “结构上减少一次 quant/scatter 中间落盘” | 无日志时说“实测 HBM 带宽提升 X%” |
| “项目口径约 1.6×，当前证据 B、配置待补” | “我们复测稳定提升 1.6×” |
| “在已验证 Shape 下命中专用 Kernel” | “所有 Shape 都更快” |
| “Graph 中禁止 Torch fallback” | “能 capture 就代表 replay 正确” |
| “no-copy 支持明确 stride contract” | “任意 non-contiguous 都能零拷贝” |
| “团队完成了该优化体系” | 未核实时把所有提交说成个人独立完成 |
| “当前官方后来已有类似能力” | 用当前状态倒推开发时没有本地创新 |
| “没有实测的部分按结构性收益和验收指标讲” | 把 benchmark framework 当成性能结果 |

### 17.15 面试讲述顺序建议

默认优先讲三项：

1. **Compact MoE Prefill**：最适合展示 Compute + 数据组织 + 端到端收益；
2. **FP8/Compress/Cache**：最适合展示 Memory-bound、no-copy 和 Kernel 能力；
3. **Decode/Runtime/Graph**：最适合展示 Small-M、host-inclusive 和生产闭环。

面试官继续深挖系统能力，再讲：

4. **Sparse Seq-Pack**：metadata/Pipeline/CP；
5. **Online C128/PP/PD/HiSparse**：状态、通信、异步生命周期与失败注入。

推荐节奏：

```text
30 秒项目定位
  -> 选一个最强案例讲 90 秒
  -> 根据追问切到 Kernel / Framework / Service 证据
  -> 用失败边界和数据口径收尾
```

不要一开始罗列十几个 Kernel 名；先讲问题、归因和决策，函数名只在追问时作为证据。

### 17.16 最终收尾话术

> 这个项目最核心的价值，不是完成了某一个 MUSA Kernel，而是建立了一套从官方新特性到 S5000 生产路径的适配方法：先区分官方语义和平台贡献，建立 NV official、MUSA port 和 MUSA tuned 三组基线；再按 Compute、Memory、Launch/Runtime 定位瓶颈；根据证据选择保留、调参、替换 Kernel 或重写 Pipeline；最后用 Fast-Path 命中、fallback、Graph、正确性和 Operator-to-Service 指标完成验收。我们的自主工作集中在 MUSA/S5000 的 Kernel、数据组织、融合 Pipeline 和系统闭环，效果只按可追溯证据表达。

如果面试官要求精确数字，使用以下回答：

> 我可以说明当前已有的用户/项目口径及其证据等级，也可以给出完整的测量设计；但如果缺少硬件、Shape、baseline commit 和原始日志，我不会把它说成自己复测的结果。

如果面试官要求拆分团队贡献，使用以下回答：

> 这份材料描述的是团队完成的整体技术链。我们可以按实际模块和记录进一步说明分工，但不会把官方能力或其他成员提交归为单个人成果。

## 18. 把技术工作整理成有含金量的实习产出

> **本章用途**：第 1～16 章是技术证据，第 17 章是面试表达。本章解决另一个问题——**如何把这些工作组织成一个“有含金量的实习产出”**：即让人一眼看出这不是打杂、跑通 demo 或改配置，而是产生了真实、可复用、可验证、体现工程判断的交付物。含金量不取决于“算法是不是你发明的”，而取决于**产出物的质量、覆盖的难度、以及是否沉淀成可复用的能力**。

### 18.1 “有含金量的实习产出”的判断标准

招聘方/导师判断一个实习产出是否有分量，看的不是工时，而是四条：

```text
1. 真实交付物：有没有进生产/被复用的 kernel、pipeline、工具、方法，而不是一次性脚本
2. 难度稀缺性：这件事是不是“别人不容易做”——跨架构、底层 kernel、系统闭环都算
3. 可验证性：产出是否有 benchmark/正确性/回归支撑，而不是“我觉得变快了”
4. 可复用/方法沉淀：是不是留下了能规模化的规则、流程、回归集，而不是走了就没了
```

把前面章节的工作对到这四条上，会发现它**全都命中**——问题只是之前没按“产出物”视角组织，而是按“提交/优化点”视角罗列。本章就是做这个视角转换。

### 18.2 从“优化点”到“交付物”：产出清单（Deliverables Inventory）

面试/答辩不要说“我做了 MoE、quant、sparse……”（这是任务列表），要说“我交付了以下东西”（这是产出）。把散在各章的工作重组为可清点的交付物：

| # | 交付物 | 对应章节 | 交付物类型 | 含金量点 |
|---|---|---|---|---|
| D1 | 一套 MUSA/S5000 专用算子 kernel（FP8 quant、cache store、compress、norm-rope、TopK、MoE compact scatter） | 3/4/5/7 | 手写 kernel | 上游 CUDA kernel 无法运行，必须重写；有 half-warp/parallel-reduce/宽写等微架构级设计 |
| D2 | MoE Prefill 的 S5000 Fixed-Bucket Compact Pipeline | 3 | 数据组织/pipeline 重构 | 在官方 grouped GEMM 前后重建数据流，含容量/overflow 契约 |
| D3 | FlashMLA Sparse Prefill 的 MUSA 落地 + CP 修复 | 6 | 移植+正确性修复 | workspace gather/dequant 的 MUSA 实现 + CP round-robin metadata 修复 |
| D4 | Decode Small-M 路径（Queue4/Page32/Head256/Split-K/paged 闭环） | 8 | 平台扩展+kernel | 保护 TPOT，prefill kernel 不能直接复用 |
| D5 | MUSA Runtime 硬化（fail-closed、MCCL 绑定、JIT key、forwarding、graph-safe） | 10 | 系统闭环 | 上游 NVIDIA 不存在的问题，从 0 到 1 |
| D6 | Online C128 / PP / PD / HiSparse 的 MUSA 状态闭环 | 9 | 系统适配 | 多坐标系、deferred-commit 状态机、跨层 transfer 可见性 |
| D7 | 一套上游→MUSA 适配方法论（三基线 + Roofline 归因 + 四类决策 + 分层验证） | 11 | 方法/流程 | 可复用到任意新特性，是规模化能力 |
| D8 | Operator/MoE/Cache/Sparse 的 benchmark 与回归框架 | 3.8/5/8/12 | 测试基建 | 让性能可验收、防回归，是工程成熟度信号 |
| D9 | 归属分级(O/P/T/R/S) + 证据分级(A/B/C/D) 的技术审计文档 | 全文 | 文档/规范 | 能清晰界定“官方 vs 平台工作”，本身是资深工程习惯 |

**关键**：D7、D8、D9 常被忽略，但它们恰恰是把“实习生”和“会写代码的人”区分开的东西——**方法论、测试基建、审计规范**证明你不是只完成任务，而是建立了可复用的工程能力。

### 18.3 这个实习产出体现的核心能力矩阵

把交付物翻译成招聘方关心的能力标签：

| 能力 | 由哪些交付物证明 | 为什么值钱 |
|---|---|---|
| **底层 GPU kernel 工程** | D1、D2、D4 | half-warp/shfl/宽写/parallel reduce/split-K，硬功夫 |
| **性能分析与归因** | D7、D8 | Roofline 分 memory/compute/launch-bound，不是瞎调 |
| **跨架构移植** | D1、D3、D6 | CUDA→MUSA，ISA 级重写，稀缺经验 |
| **硬件感知优化** | D2、D5 | 按 S5000 高脊点/带宽饥饿反向优化 |
| **系统/分布式工程** | D5、D6 | graph capture、多卡、PP/PD、状态机 |
| **工程方法与规范** | D7、D8、D9 | 方法论、回归、归属审计——可规模化、可维护 |

一份实习产出能同时覆盖“kernel + 性能 + 系统 + 方法”四个层面，本身就说明它含金量不低。

### 18.4 诚实的量化：即使没有实测倍数，也能量化“产出规模”

“有含金量”不等于“必须有性能倍数”。在没有证据 A 实测数字时，用**结构性/规模性量化**同样有力，而且不会被问穿：

```text
覆盖算子：为 N 类 DSV4 热点算子（quant/cache/compress/norm-rope/topk/MoE/sparse）提供 MUSA 实现
kernel 数：手写/重写 M 个 MUSA kernel（含 half-warp、split-K、双路径 dispatch）
场景覆盖：覆盖 8K/32K/128K prefill、decode small-M、CP/PP/PD、MTP 等场景
系统闭环：打通 K 项 MUSA 特有 runtime/多卡/graph 问题（上游不存在）
可验证性：建立 operator/module/service 三层 benchmark + 回归 case
方法沉淀：形成 O/P/T/R/S 归属 + A/B/C/D 证据 + 四类决策的适配规范
```

这些数字来自代码事实，**不依赖实测日志就成立**，属于证据 C，可以放心写。真正的性能倍数(1.6x/1.8x/+8%)按证据 B 单独标注“用户/项目口径、配置待补”。

### 18.5 一句话实习成果定位

用于简历标题栏或开场：

> 在摩尔线程 S5000 上完成 DeepSeek V4 FP8 推理栈的后端落地与性能优化：手写一整套 MUSA 专用算子 kernel，重构 MoE/Sparse 数据流水线，打通 graph/多卡/PP-PD 系统闭环，并沉淀出一套“上游特性→MUSA 适配”的可复用方法论与三层 benchmark/回归体系。

它同时传达：**真实交付物(kernel/pipeline) + 难度(跨架构/系统) + 可复用(方法论/回归)**，正好命中 18.1 的四条标准。

### 18.6 简历里怎么写这段实习经历（bullet 范式）

每条 bullet 的公式：**动词(产出物) + 难点/规模 + 可验证性/影响**，避免“参与/协助/负责”这类模糊词。

- 在 MUSA(S5000) 上**手写/重写了 N 类 DSV4 FP8 热点算子 kernel**（FP8 quant、cache store、compress、TopK、MoE compact scatter），采用 half-warp 映射、parallel reduce、宽写等按硬件微架构设计的实现，因上游 CUDA kernel 无法在 MUSA 执行。
- **重构了 MoE Prefill 的 S5000 数据流水线**：在官方 grouped GEMM 前后引入 fixed-bucket compact + quant/scatter 融合与容量/overflow 契约，减少中间落盘，并用 token 阈值隔离 decode。
- **打通了 MUSA 后端从 0 到 1 的系统闭环**：fail-closed dispatch、MCCL 设备绑定、JIT specialization key、graph-safe fallback、PP/PD 与 Online C128 状态生命周期——均为上游 NVIDIA 版本不存在的问题。
- **建立了一套上游→MUSA 适配方法论与三层验证体系**：三组基线 + Roofline 瓶颈归因 + 四类决策 + operator/module/service benchmark 与回归 case，可复用到后续任意上游新特性。
- （有数据时）项目口径下 MoE 算子约 1.6×、端到端 prefill 约 +8%、TPOT −9.6%（用户/项目口径，配置/复现待补）。

### 18.7 转正/答辩视角：为什么这是“能独立干活”的证据

导师/评委真正想确认的是：**离开手把手指导，你能不能独立产出。** 用这条链证明：

```text
能读懂上游前沿实现（DeepGEMM/FlashMLA/MTP 调用链）
  -> 能判断它在新硬件上哪里不成立（Roofline + 三基线归因）
  -> 能自己写出 MUSA kernel 解决（不是等人给方案）
  -> 能保证正确性和可验收（benchmark/回归/归属审计）
  -> 能把经验沉淀成规则供后续复用（方法论）
```

这条链每一环都是“独立工程能力”的证据，比“完成了导师安排的 N 个任务”强得多。答辩时按这条链讲，而不是按提交时间罗列。

### 18.8 避免把实习产出讲成“打杂”的三个陷阱

| 陷阱 | 会被听成 | 改成 |
|---|---|---|
| 按提交/任务罗列（“我做了 A、B、C……”） | 打杂、执行 | 按交付物 + 能力组织（D1~D9） |
| 只强调“移植/适配/基于官方” | 搬运工 | 承认上游后，重心放在“MUSA 上重写 + 反向优化 + 系统闭环” |
| 没有可验证性（“应该变快了”） | 不严谨 | 挂上 benchmark/回归/证据等级，即使是结构性量化 |

### 18.9 一句话收尾

> 这段实习的产出不是“帮忙优化了几个算子”，而是**独立交付了一整套 MUSA/S5000 上 DeepSeek V4 FP8 推理的执行层**——从手写 kernel、重构数据流水线、打通系统闭环，到建立可复用的适配方法论和验证体系。含金量在于它同时覆盖了 kernel、性能、系统和方法四个层面，且每一层都有可验证的交付物，而不是一次性的调通。
