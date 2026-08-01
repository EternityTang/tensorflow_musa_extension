# DeepSeek V4 FP8 MUSA — 简历最终版(全文审计后)

> 依据:对 `docs/resume_docs/deepseekv4_fp8_prefill_optimizations.md`(3508 行)全文精读审计 + `docs/deepseekv4_fp8_prefill_optimizations.md` 第 0 章硬件表 + 上游 SGLang/DeepSeek 公开 PR 核实
> 组织方式:按大模型推理阶段展开(Prefill → Decode → 跨阶段底座)
> 原则:只保留有 kernel/参数/机制级证据、且上游确实没有的内容

---

## 一、简历正文

**DeepSeek V4 FP8 推理栈 MUSA(S5000)移植与性能优化** · SGLang · 实习

> S5000:FP8 峰值约 1001 TFLOPS、带宽 1.6 TB/s,FP8 Roofline 脊点约 626——是 H20(约 62)的 10 倍,而绝对带宽只有 H 系列(4.8 TB/s)的 1/3。上游 kernel 的调优取向不能直接迁移,memory-bound 算子的壁钟代价被放大约 3 倍,优化重心从"提升计算效率"转为"减少 HBM 往返与 launch 次数"。CUDA kernel 无法在 MUSA 上执行,以下算子均为 MUSA(TileLang / 手写 `.cuh`)重新实现。

### ▍Prefill / TTFT — MoE FP8 数据通路

上游 DeepGEMM contiguous grouped GEMM 要求 expert-contiguous 输入,但 permute/scatter 与量化在上游是**两个独立算子**,中间要物化一整个 `[T,H]` FP8 张量。实现 **fused quant-scatter MUSA kernel**:warp-per-group 线程映射(`group_linear = block*groups_per_cta + warp`,`token_id`/`scale_group` 由 hidden 维分解得到),组内 abs-max 并行归约 → scale → FP8 cast/pack → **按 `src2dst` 直接写 compact row 与 compact scale**,消除该中间张量的一次写与一次散读。按流量公式测算(H=4096, T=8192, k=8, M=65536)**单层省约 311 MB、整栈约 18 GB,在 S5000 的 1.6 TB/s 下约合 11 ms**——同一优化在 4.8 TB/s 的 H200 上只值约 3.8 ms。

同时设计 **fixed-bucket 容量契约**:`allocated_rows = E × cap_per_expert` 静态分配以支持 graph capture,`valid / padded / allocated / worst_case` 四口径分账,热 expert 超容时置 overflow flag 并显式回退动态路径、不截断 route。按 token ≥ 8192 与 `hidden`/`rows` 分档做 dispatch,使 decode 不承担 compact 固定成本。

### ▍Prefill / TTFT — Sparse Indexer 选择与 Pack8 打包

实现 MUSA **TopK-512 select kernel**:不是普通 MoE TopK,而是 score → page-index 变换(`logical_page = r >> log2(page)`、`physical = page_tables[row, logical]`、`page_index = (physical << log2) | offset`,invalid → −1)。大 shape 走**四轮 byte-radix count/threshold + gather/prefix/write 的 exact select**;并设计 **graph-capture 安全的 exact/candidate 双跑机制**——bounded candidate 无 overflow 时在 device 侧覆盖 exact 结果,避免 host 同步而破坏 graph 可捕获性。定位出 `candidate_capacity=1024` 在 `[16, 262208] → [16, 512]` 这类单行扫 262K 的生产 shape 上失败,据此确定 candidate 不能作默认路径。

另实现上游没有的 **Pack8 变体**(`flash_mla_sparse_fwd_pack8` + `pack_row_level_indices_for_pack8` / `pack_structured_dsv4_indices_for_pack8`),以 row mask + pack lens 保证 padding lane 不贡献 attention、pack 内每 query 只读自身有效 prefix,并处理 `q.shape[0] % 8` / 每 request `extend_seq_len % 8` 的对齐与 request padding。

### ▍Decode / TPOT — 小 M 专用 kernel 与投机解码状态机

Decode 阶段算术强度约 2B(B 为 batch),S5000 脊点约 626 意味着 **B ≳ 313 才能脱离 memory-bound,约为 H20(B ≳ 31)的 10 倍**——在此之前 FP8 峰值算力无法兑现,耗时仅由权重流量与 launch 延迟决定。据此为 decode 单独实现:

- **FlashMLA cache store decode kernel**:block-per-token 的 x4 / x4_fp32 / vec2,一趟内完成 quant / scale / NoPE / RoPE / page store
- **HC-head linear Split-K 两段 kernel**:stage0 按 K 维 partial accumulation、stage1 reduce,另有 warp variant,针对小 M 大 K,split 数经 sweep 选定
- **paged C4 decode compress kernel**:`block_id = indices[token]`、`write_pos = (seq_len+3) % 4`,遍历有效 source slot 做 exp-weighted numerator/denominator

另实现 **Online C128 × MTP verify 的 deferred-commit 状态机**:verify 前 reset,forward 时每层仅缓存 kv score,verify 产出 accepted rows 后逐层逐行回放到持久 state,`position → raw_loc → swa_loc → ring → state_index` 换算贯通,并处理 127→128 边界恰好一次 compress/store,保证被拒绝的 draft 不污染 C128 ring。

### ▍跨阶段 — 访存算子融合与 MUSA kernel 底座

这些算子 FLOPs 极低、耗时完全由实测带宽决定,实现围绕**减少 HBM 往返、加宽访存、消除中间张量物化**展开:

- **Fused Norm-RoPE-Cache**(手写 `fused_norm_rope_flashmla_musa.cuh`):一次调用内完成 RMSNorm + 旋转,并直接按 page layout 写出 NoPE FP8 / UE8M0 scale byte / RoPE,消掉 `norm_out`、`q/k_rope`、`fp8+scale` 三个中间张量的 HBM 往返;配等价性测试与 **strided-input 测试**,证明"消除物化"不是在 wrapper 里偷偷 `.contiguous()`
- **Cache store prefill 路径**:`subwarp16` / `tile_parallel`,把 7 个 NoPE quant tile 拆成连续 4+3 两组由多个 half-warp 协作完成 amax → scale byte → FP8 pack → RoPE vector store,`full_tiles` 仅在 `T % tokens_per_cta == 0` 时启用、尾块保留 bounds guard,`T ≥ 128` 作为分流阈值
- **FP8 per-token-group quant** 的 TileLang/Triton MUSA 重实现(group=128 → 16 lane × 8 值,half-warp shuffle 归约,FP8x4 pack + 连续宽写回,不物化 amax 张量),以及 **C4 page reduce / C128 parallel reduce**
- **Shape-aware dispatch**:`total_groups ≥ 16K` 时主动阻断 generic cast helper,防止大 CP prefill 落入已知慢路径;`hidden`/`rows` 三分档选 `groups_per_cta`/`num_warps`

### ▍验证

自建算子级 benchmark 与回归套件(MoE prefill 分项计时、cache/MHC benchmark、`logical_bytes → GB/s` 有效带宽口径并显式声明其不等于硬件计数器 HBM traffic),验收顺序为先对齐 `src2dst`/`m_indices`/`scale`/`output` 数值再比 latency;关键路径配等价性与反作弊测试。

> 项目口径参考值:MoE 算子约 1.6×、prefill 端到端约 +8%、TPOT 7.3 → 6.6 ms/token。**均为项目内提供的口径,缺完整硬件/shape/baseline commit 记录,本人未独立复测。**

---

## 二、硬件立论(修正版)

**老文档第 0 章硬件表(唯一有数字的地方):**

| 指标 | H20 | H200 | S5000 |
|---|---:|---:|---:|
| FP8 TensorCore | 296 TFLOPS | 3341 TFLOPS | 1001.4 TFLOPS |
| BF16 TensorCore | 148 | 1671 | 500.7 |
| 显存带宽 | 4.8 TB/s | 4.8 TB/s | **1.6 TB/s** |
| 显存类型 | HBM3e | HBM3e | DDR6(疑为 GDDR6) |
| 互联 | NVLink 900 GB/s | NVLink 900 GB/s | 784 GB/s |
| **FP8 脊点(算力÷带宽)** | **≈62** | **≈696** | **≈626** |

**正确立论(两条,缺一不可):**
1. S5000 脊点 ≈626 是 H20(≈62)的 **10 倍**。上游大量 kernel 按 H20/H800 这类国内推理卡调优,那里几乎一切都不是 memory-bound;搬到 S5000 后大量算子掉进 memory-bound 区,调优方向必须反向。
2. 绝对带宽 1.6 TB/s 只有 H 系列 4.8 TB/s 的 **1/3**,同样字节数壁钟时间是 3 倍。**"减字节"的收益被放大 3 倍**——这才是所有 memory-bound 优化的真正依据。

**⚠️ 不能再说的话:**
- ❌ "S5000 脊点特别高" —— 按此表 H200 是 696,比 S5000 还高。脊点高不是 S5000 独有特征。
- ❌ "H200 脊点约 412" —— 之前的错误数字,按此表应为 696。
- ⚠️ 该表 H200 的 3341 疑似含 2:4 稀疏(H200 FP8 dense 通常引作 ~1979),与 S5000 口径可能不一致。**跨厂商比较要谨慎,H20 对比相对可靠(两者应同为 dense)。**

**⚠️ 新文档里没有任何数字。** 全文 3508 行:零 TFLOPS、零 GB/s、零脊点数值、零算术强度数值。"高脊点"是纯定性断言,文档第 141 行自评证据等级 `D`(仅静态分析或 Roofline 推导,不得写成 profiler 实测)。**面试若被问"你们脊点算出来多少"必须用上表回答,不能说"文档里推导过"。**

**decode 推导(可白板复现):**
```
权重 [K,N] FP8:FLOPs ≈ 2·B·K·N,Bytes ≈ K·N(权重主导)→ AI ≈ 2B
2B > 脊点 → H20: B ≳ 31 | S5000: B ≳ 313
```

---

## 三、审计结论:哪些是真的

### 确认为自己写的 MUSA kernel(有线程映射/归约/索引公式级描述)

| kernel | 证据强度 | 备注 |
|---|---|---|
| Fused Norm-RoPE-Cache(`fused_norm_rope_flashmla_musa.cuh`) | ★★★ | 全文唯一有等价性测试 + strided-input 反作弊测试;唯一明确的手写 `.cuh` |
| Cache Store 双路径族(decode x4/x4_fp32/vec2;prefill subwarp16/tile_parallel) | ★★★ | 机制描述最完整:7 tile 拆 4+3、full_tiles 条件、tail guard |
| MoE fused quant-scatter | ★★ | 上游 `ep_scatter` 与 `per_token_group_quant_fp8` 是分离的,融合是本地的 |
| Sparse TopK-512 select(+ exact/candidate graph-safe 机制) | ★★ | 含真实定位:`candidate_capacity=1024` 在 262K 长行失败 |
| HC-head Split-K 两段(stage0/stage1 + warp variant) | ★★ | Split-K 是通用范式,MUSA 实现是本地的 |
| Paged C4 decode compress | ★★ | 有完整索引与数值公式 |
| `flash_mla_sparse_fwd_pack8` + 两个 metadata 打包函数 | ★★ | **上游无 pack8 变体**,是 sparse 章唯一差异化产出 |
| C4 page reduce / C128 parallel reduce | ★ | 机制只有"parallel reduce"几个字,描述偏薄 |
| FP8 per-token-group quant(TileLang + Triton multi-group) | ★ | 结构与上游 sgl-kernel `per_token_group_quant_8bit.cu` 同款,只能说"MUSA 重实现"不能说"设计" |
| Indexer cache pack/store 双路径 | ★ | 与 cache store 同模式的复刻 |
| HiSparse MUSA host-offload copy kernel | ★ | 固定 layout 的 pointer copy,复杂度低 |

### 确认为系统语义工作(非 kernel,但有含量)

| 项 | 内容 |
|---|---|
| Online C128 × MTP deferred-commit 状态机 | 两阶段提交完整,`position→raw_loc→swa_loc→ring→state_index` 换算链能扛追问;但编排的是已有 triton kernel,无新 kernel |
| PP compressed-MLA pointer table 分段切片 | `2*c4_full + c128_full` 三段 offset + SWA layout 变体 + pointer 数校验,易错的正确性工作 |
| fixed-bucket 容量契约 + overflow 回退 | 概念借自上游 DeepEP masked/low-latency,用在 contiguous prefill 是本地的 |
| Shape-aware quant dispatch(16K guard) | "主动返回 None 挡住 generic cast"是真发现 |
| CP round-robin local metadata 重建 | 真实系统语义(round-robin 是交错分配,global `query_start_loc` 不能 slice);但**文档无任何定位过程**,不能当"疑难 bug 故事"讲 |

---

## 四、必须放弃的(写了会被问穿)

| 项 | 原因 |
|---|---|
| **"实现 TopK V2"** | ⛔ `topk_transform_512_v2_musa` 只校验 metadata shape 后**转调 v1**,cluster-persistent scheduling 未实现(文档 484 行自曝)。上游有真 V2(#23882 / #25406) |
| **"SwiGLU + FP8 量化融合"** | 上游已有(官方 `d0913fca8`、PR #4199/#4343)。文档原话是"**保持** masked SwiGLU quant 走 JIT",作者只做接线 |
| **expert-contiguous / m_indices / src2dst** | 上游 `ep_scatter`/`ep_gather` 已产出;作者只放开了 MUSA 设备 guard |
| **seq-pack / flat workspace / rebased indices / chunk 级跨层复用** | 上游主分支全部已有(PR #25418 起)。文档唯一辩护是 6 天时间差,不可采信为个人贡献 |
| **FP8 nope(448)→BF16 反量化** | ⛔ 这是上游 `flash_mla_sparse_fwd` 的 **dtype 契约**,NV 路径同样要付(vLLM 亦然);且 448B→896B 是**流量膨胀的成本,不是优化**。上游另有 `flashmla_sparse_q8` 可免此往返 |
| **Fused Gate** | 未落地实验、机制未描述、无收益记录;sgl-kernel 已有同名 `moe_fused_gate` |
| **C4 Indexer fused RoPE+Hadamard** | 上游 `fused_q_indexer_rope_hadamard_quant`(PR #27705)已有;文档只写了 guard 与 dispatch,没有一句"我写了这个 kernel" |
| **"设计了 half-warp 量化算法"** | 16 lane × 8 值 + shuffle amax + 向量 pack 就是上游 sgl-kernel CUDA 的标准结构;文档也无任何与上游实现的差异对比或参数 sweep 数据 |
| **"三层 benchmark 体系"** | 算子层已落地;module/service 层文档措辞是"**建议**覆盖",是设计不是已运行 |
| **Runtime Hardening 整章** | 老文档 2167 行只含 4 个独立提交,其余为同内容三次复述;新文档第 10 章净增的 3 个 commit 全是 C 级(ABI 头文件对齐 / env 开关断言 / 自家 facade 重构),且把已在 ch7 计过的 cache-store 提交改名"no-copy 契约"重列。**最多在最后合并成一行。** |

---

## 五、性能数字口径

文档中全部数字及等级:

| 数字 | 对象 | 文档自评 | 可用性 |
|---|---|---|---|
| 约 1.6× | MoE 算子 | B(用户/项目口径) | 需标注"项目口径、配置待补" |
| +8% | 端到端 prefill | B | 同上 |
| 1.8×~2× | FP8 Quant/Compress 前处理 | B | ⚠️ 两个算子被合并成一个不可复现的数字 |
| 约 2× | Sparse 前处理及 metadata | B | ⚠️ 文档明确警告不等于端到端 |
| 约 1.8× | **Prefill** cache store | B | 仅 prefill 路径 |
| 7.3 → 6.6 ms/token(−9.6%) | TPOT | B | 硬件/模型/并发/MTP 配置待补 |

**文档第 2747 行:「本文未执行 MUSA 实机 benchmark。」** 全文无 profiler 输出、无 nsys/torch.profiler 痕迹、无 A/B 实测。第 12 章"性能归因"实为先验分类学(把算子塞进 Compute/Memory/Launch 三桶),不是归因。

**安全表述**:"项目口径约 1.6×,证据等级 B、配置待补"
**危险表述**:"我们复测稳定提升 1.6×"

**可以不带限定词写的是结构性量化**:自写 kernel 与变体数量、消除的 HBM 往返次数、消除的中间张量个数、覆盖的算子类别数。

### 关于 311 MB / 18 GB / 11 ms 这组数

这组是**推导值,不是实测值**,证据等级 D(纯算术 + Roofline),但它和上面的 B 级数字**性质不同,可以放心写**:

- 它是**可当场白板复现的算术**:给定 H/T/k/group,split 与 fused 两条路径的读写字节都是确定的,没有任何拟合或外推。
- 它的**全部假设都写明了**:H=4096, T=8192, k=8, M=65536, 58 MoE 层为 `[推断]`,S5000 带宽取 1.6 TB/s 标称值。面试官换一组 shape,你能当场重算。
- 措辞用"**按流量公式测算**"而不是"实测/达成/提升"。推导值说成推导值,不算夸大。

**唯一的软肋**:它假设 fused kernel 按源行并行(即量化一次、写 k 份)。三条论据见 **§7.3.2 推理链**,其中"若按目标行并行则整条路径是负收益、与 1.6× 矛盾"这条是独立自洽论证,即使前两条记错也仍成立。回代码扫一眼 `src2dst` 遍历方向即可坐实。

**同时要记住:11 ms 只是这一条 kernel 的流量下限,而上游那个 SwiGLU fusion 在同样口径下值 19 ms。** 被问"你这条贡献占多少",诚实答案是"比上游那个 fusion 小,大概是它的六成"——主动给出这个对比,比等对方算出来强。

---

## 六、面试策略

**讲的顺序 ≠ 简历排的顺序。** 被问"讲一个最有把握的":

1. **先讲 Fused Norm-RoPE-Cache** —— 证据链最完整(手写 kernel + 读写账本 + 等价性测试 + 反作弊测试),追问到底都答得出。
2. **再讲 Cache Store 双路径** —— 微架构故事最完整:为什么 decode 要 block-per-token(避免大 tile 固定成本)、为什么 prefill 要 subwarp16/tile-parallel(跨 token 并行 + 宽写回)、7 个 tile 怎么拆 4+3、尾块怎么处理。
3. **"最难的 bug"** → TopK candidate_capacity 在 262K 长行失败,或 TileLang JIT cache key 未含 layout 参数导致顺序相关的错误结果。
4. **被追问 DeepGEMM/FlashMLA 细节** → 先主动划清"库和算法是上游的,quant-scatter kernel、pack8、cache store 是我写的"。

**待核实(直接影响措辞):**
1. 这些 kernel 常数(`groups_per_cta=8`、`T≥128`、`16K`、`8192`)有没有 sweep 数据?**文档全文只有 MHC prenorm 一处出现"sweep"字样。** 有数据 → 可写"标定";无数据 → 只能写"按 shape 分档选型"。
2. `git log --author` 确认哪些 commit 是本人的。
3. 跑一次 benchmark,把 B 级数字升成实测。

---

# 七、深挖手册

> **用法**:面试官说"挑一条详细讲讲"或顺着简历往下追时,答案在这里。每条按 **归属 → 物理动机 → 机制 → 能扛的追问 → 软肋** 五段。
>
> **⚠️ 关于 `[待补]` 标记**:机制细节全部来自对源文档的静态审计。凡是审计没有建立起来的,一律标 `[待补]`,**没有编造**。被追问到 `[待补]` 处,正确答法是"这块的具体实现我要回去确认一下",**不要现场推一个听起来合理的答案**——面试官对自己领域的 kernel 细节比你熟,编的会被当场戳破,而"我不确定"只是扣一点分。
>
> **讲述优先级**(按证据链强度,不是按简历顺序):7.1 → 7.2 → 7.4 → 7.3 → 其余

---

## 7.1 Fused Norm-RoPE-Cache ★ 最该先讲的一条

**归属**:`[我的]` 手写 MUSA C++,文件 `fused_norm_rope_flashmla_musa.cuh`,入口 `fused_norm_rope_flashmla_store_musa_jit` / `_v2`。

**为什么这条排第一**:它是**全部工作里唯一同时具备"手写 kernel + 读写账本 + 等价性测试 + 反作弊测试"的**。别的条目最多有前两项。

### 物理动机

MLA 的 KV 写入路径上,原本是四个独立算子串起来:

```
hidden ──RMSNorm──> norm_out ──RoPE──> q_k_rope ──quant──> fp8 + scale ──store──> paged cache
         [写HBM]              [读写HBM]         [读写HBM]              [读写HBM]
```

`norm_out`、`q_k_rope`、`fp8+scale` 三个中间张量的生命周期都只跨越一个 kernel,产出后立刻被下一个消费。为它们做完整的 write-back + read-back 是纯浪费。这类算子的 FLOPs 极低(RMSNorm 是逐元素、RoPE 是两个乘加),**耗时 100% 由 HBM 流量决定**——正好落在 S5000 脊点(≈626)的最左侧。

### 机制

一次 kernel 调用内完成:

```
读 hidden 一次
  → RMSNorm(组内归约求平方和 → rsqrt → scale)
  → RoPE 旋转(cos/sin 配对,寄存器内完成)
  → NoPE 部分:求 amax → scale = amax/448 → FP8 E4M3 cast
  → 直接按 paged cache 的 page layout 写出:
       NoPE FP8 | UE8M0 scale byte | RoPE
```

关键在最后一步:**不是"算完再搬到 cache",而是直接按 cache 的物理排布写**。三个中间张量一个都不物化。

`[待补]` 线程映射(每 block 处理几个 token、head 维怎么分)、`v2` 相对 v1 改了什么。

### ★ 测试设计(这条是加分项,一定要讲)

| 测试 | 验什么 |
|---|---|
| **等价性测试** | fused 输出与 `RMSNorm → RoPE → quant → store` 四段串行的结果逐元素一致 |
| **strided-input 测试** | 输入张量**非连续**时结果仍正确 |

**strided 测试为什么是反作弊**:消除中间张量最容易的偷懒方式,是在 Python wrapper 里悄悄插一个 `.contiguous()` —— 这样 kernel 内部只需处理连续内存,实现简单,但**你在 wrapper 里刚刚制造了一个新的全量拷贝**,等于把省下的流量又花掉了,而单测因为只喂连续输入根本发现不了。加一个 strided 输入的用例,就把这条退路堵死了:kernel 必须真的按 stride 访存。

> **这段话本身就是很强的信号**——它说明你知道"性能优化可以自欺",并且主动设计了防线。审计发现全项目只有这一条有这个测试。

### 能扛的追问

| 问 | 答 |
|---|---|
| 省了多少? | 三个中间张量的 write+read。按 shape 能算,方法同 MoE 那条(§三 的流量账法) |
| 为什么值得融?RMSNorm 不是很快吗? | 正因为它快才值得融——FLOPs 低意味着 AI 低,耗时全在访存;S5000 带宽 1.6 TB/s 只有 H 系列 1/3,减字节的收益 ×3 |
| 融合会不会影响数值精度? | 有等价性测试兜底。而且寄存器内连算反而少了中间的 bf16 舍入 |
| **怎么证明你真的消除了物化,而不是 wrapper 里加了 contiguous?** | **主动讲 strided 测试** —— 这题就是为你准备的 |
| 为什么手写 `.cuh` 而不用 TileLang? | `[待补]` 大概率是 page layout 的位级写出用 DSL 表达不了 |

### 软肋

- `v1` / `v2` 的差异说不清 → 别主动提 v2。
- 没有实测数字,只有推导。

---

## 7.2 FlashMLA Cache Store 双路径 ★ 微架构故事最完整

**归属**:`[我的]` MUSA kernel 族。
- decode 路径:`decode_x4` / `decode_x4_fp32` / `vec2`
- prefill 路径:`prefill_subwarp16` / `prefill_tile_parallel`

**为什么值得讲**:这是**唯一一条能完整回答"同一个功能为什么要写五个 kernel"的**——而这正是 GPU 岗最想听的思考方式。

### 物理动机:decode 和 prefill 是两个完全不同的并行问题

| | decode | prefill |
|---|---|---|
| token 数 T | 1 ~ 几十 | 几千 ~ 几万 |
| 并行度来源 | **不够**,只能靠 head/hidden 维 | 充足,token 维就够 |
| 瓶颈 | kernel launch + 尾延迟 | 纯访存带宽 |
| 正确策略 | 每 token 一个 block,尽量少的固定开销 | 跨 token 并行 + 宽写回 |

**一个 kernel 通吃必然两头不讨好**:按 prefill 设计的大 tile kernel,在 T=1 时绝大部分 block 空转,固定开销吃掉一切;按 decode 设计的 block-per-token kernel,在 T=8192 时写回宽度不足,跑不满带宽。

### 机制

**decode 路径**:block-per-token,一趟内完成 `quant → scale → NoPE → RoPE → page store`。
三个变体 `x4` / `x4_fp32` / `vec2` 的区别在向量化宽度与累加精度 —— `[待补:各自的选择条件]`。

**prefill 路径**(细节最丰富的部分):

```
7 个 NoPE quant tile,拆成连续的 4 + 3 两组
  → 由多个 half-warp 协作完成:
       amax 归约 → scale byte → FP8 pack → RoPE vector store

full_tiles 快路径:仅当 T % tokens_per_cta == 0 时启用(整除,无尾块)
尾块:保留 bounds guard,不走快路径
分流阈值:T >= 128 才进 prefill 路径
```

**7 拆成 4+3 而不是别的**:`[待补]` —— 合理猜测是 4 对齐向量化宽度、3 是余数,但**别当成事实讲**。这题可以答"分组是为了让每个 half-warp 拿到对齐的连续 tile,具体 4+3 的边界我要回去看一下"。

### 能扛的追问

| 问 | 答 |
|---|---|
| **为什么要写两套?** | 核心答案见上表:decode 缺并行度、prefill 缺写回宽度,是两个不同的并行问题 |
| `full_tiles` 为什么要判整除? | 整除时所有 block 的工作量一致,可以去掉每次访存的边界检查;不整除的尾块单独走带 guard 的路径。**用一次 host 侧判断换掉 kernel 内每个 tile 的分支** |
| T>=128 这个阈值怎么来的? | 诚实:`[待核实]` 是否有 sweep。没有就说"按并行度够不够分档,128 是经验值" |
| 为什么 decode 要 block-per-token? | T 小时 block 数就是并行度上限,每 token 一个 block 能让 SM 尽量占满;大 tile 会让大部分 SM 空转 |
| x4 / vec2 / fp32 三个变体怎么选? | `[待补]` |

### 软肋

- 三个 decode 变体的选择条件说不清。
- 4+3 的理由是推测。
- 项目口径 "prefill cache store ≈1.8×" 只覆盖 prefill 路径,别说成整体。

---

## 7.3 MoE Compact DeepGEMM(含 Fused Quant-Scatter)

> 本节由原独立文档 `docs/interview_moe_compact_deepgemm.md` 合并而来(该文件已删除),内容最完整,可当白板推导稿用。
> 参数约定:`[推断] H=4096, E=256, k=8, I=2048, T=8192, M=T×k=65536, group=128`

### 归属总表 —— 讲之前先看这张

| 环节 | 上游已有 | 我在 MUSA 上做的 |
|---|---|---|
| expert-contiguous 布局 / `m_indices` / `src2dst` | ✅ `ep_scatter` / `ep_gather` 已产出全套 metadata | 放开 MUSA 设备 guard、接线 |
| 两段 grouped FP8 GEMM | ✅ `grouped_gemm_nt_f8f8bf16_contig` | 调用与 shape/layout 对齐 |
| **量化 + scatter 融合** | ❌ 上游是**两个独立算子** | ✅ **`_tilelang_moe_deepgemm_*quant_scatter_kernel`** |
| SwiGLU + FP8 quant 融合 | ✅ `silu_and_mul_contig_post_quant`(PR #4199 / #4343) | 仅接线;源文档原话是"**保持** masked SwiGLU quant 走 JIT" |
| fixed-bucket 静态容量 | ⚠️ 概念源自 DeepEP masked / low-latency 的 `num_max_dispatch_tokens_per_rank` | ✅ 用在 **contiguous prefill** 上,四口径分账 + `overflow_flag` 显式回退 |
| dispatch 阈值 | ❌ | ✅ `token ≥ 8192`、`rows ≥ 1024`、`hidden`/`rows` 三分档 |

> **这张表就是这条经历的全部诚实内容。** 6 行里只有 1 行能称"自己写的 kernel"。**面试第一句就主动划清界限** —— 面试官知道 DeepGEMM 和 ep_scatter 是开源的,你先说是加分,被问出来是减分。

---

### 7.3.1 Roofline:为什么"瓶颈不在 GEMM"是可计算的

三卡脊点表见 **§二**(S5000 ≈626 / H20 ≈62 / H200 ≈696)。用它算 MoE 两类环节:

**gate/up GEMM**(H=4096, 2I=4096):
```
每行 FLOP ≈ 2 × 2I × H = 33.5M
每行访存 ≈ H(激活 fp8) + 2I×2(输出 bf16) ≈ 12 KB   ← 权重被组内多行摊薄,忽略
AI ≈ 33.5M / 12K ≈ 2700 FLOP/byte  >>  626   → compute-bound
```

**SwiGLU + quant**(elementwise):
```
每行访存 ≈ 读 2I bf16(8KB) + 写 I fp8(2KB) ≈ 10 KB
每行 FLOP ≈ silu + mul + quant ≈ 20K
AI ≈ 20K / 10K ≈ 2 FLOP/byte  <<  626        → 深度 memory-bound
```

**结论**:GEMM 的 AI 是脊点的 4 倍以上,DeepGEMM 已贴着 FP8 TensorCore 峰值跑,**再调没有收益**;scatter / SwiGLU / quant / reorder 的 AI 只有个位数,耗时 100% 由 HBM 流量决定。
**优化 MoE = 砍 memory-bound 环节的流量和 launch 次数,不是碰 GEMM。**

> ⚠️ 算 GEMM 的 AI 时**别把权重算进访存** —— 组内多行摊薄了,算进去 AI 会算低,结论就反了。这是最常见的翻车点。

---

### 7.3.2 流量账:方法 + 两个算例

#### 方法(通用,面试真正考的是这个)

```
1. 列出 split 路径每个 kernel 的读 / 写字节
2. 列出 fused 路径的读 / 写字节
3. 差值 ÷ 带宽 = 时间下限(是下限:不含 launch、不含 L2 命中)
4. × 层数 = 整栈收益
```

两个必须自己盯住的坑:
- 别把权重算进 GEMM 访存(见上)。
- **别默认 fused 一定省** —— 融合有可能把"读一次写 k 份"变成"读 k 次",raw traffic 反而涨。见下面的推理链。

#### 算例 A:SwiGLU + quant 融合 `[上游]`

`down_input` 生命周期极短 —— SwiGLU 产出后立刻被 down GEMM 消费一次,为它做完整 write-back + read-back 是纯浪费。

| 路径 | 读 | 写 | 小计 |
|---|---|---|---:|
| **split** SwiGLU | gateup `[M,2I]` bf16 = 537 MB | down_input `[M,I]` bf16 = 268 MB | |
| **split** quant | down_input bf16 = 268 MB | down_input_fp8 = 134 MB(+scale) | **≈ 1207 MB** |
| **fused** | gateup bf16 = 537 MB | down_input_fp8 = 134 MB(+scale) | **≈ 671 MB** |

```
单层省 ≈ 536 MB;整栈 58 层 ≈ 31 GB
S5000: 31 GB / 1.6 TB/s ≈ 19 ms      H200: 31 GB / 4.8 TB/s ≈ 6.5 ms
```

> ⚠️ **这笔账是上游那个 fusion 的收益,不是我的产出。**
> 但它仍然值得讲 —— **它精确演示了"为什么在 S5000 上要优先做减字节的事":同一个融合,S5000 省 19 ms,H200 只省 6.5 ms。** 措辞是"上游这个 fusion 在 S5000 上的杠杆是 H200 的 3 倍,这是我判断优化优先级的依据",**不是**"我做了这个 fusion"。

#### 算例 B:Fused Quant-Scatter `[我的]` ★

```
上游:  quant([T,H] bf16 → [T,H] fp8)  →  ep_scatter([T,H] fp8 → [M,H] compact)
                                 ↑ 中间物化一整个 [T,H] fp8 张量
我的:  一个 kernel:读 hidden → 组内 amax → scale → FP8 pack → 按 src2dst 直写 compact
```

| 路径 | 读 | 写 | 小计 |
|---|---|---|---:|
| **split** ① `per_token_group_quant_fp8` | hidden `[T,H]` bf16 = 67.1 MB | x_q `[T,H]` fp8 = 33.6<br>scale `[T,32]` = 1.0 | 101.7 MB |
| **split** ② `ep_scatter` | x_q 按 `src2dst` 散读 = 268.4<br>scale 散读 = 8.4 | compact `[M,H]` fp8 = 268.4<br>compact scale = 8.4 | 553.6 MB |
| | | | **≈ 655 MB** |
| **fused** | hidden bf16 = 67.1 | compact fp8 = 268.4<br>compact scale = 8.4 | **≈ 344 MB** |

```
单层省 ≈ 311 MB = x_q 一次写(33.6) + 一次散读(268.4) + scale 往返(9.5)
整栈 58 层 ≈ 18 GB → S5000 ≈ 11 ms;H200 ≈ 3.8 ms
```

**最能体现理解深度的两句**:
1. **fused 的读端(67 MB bf16)与写端(268 MB compact)和 split 完全相同,省下的 100% 是中间张量的物化** —— 纯粹的物化消除,没有算法取巧。
2. **量化必须在源行上算** —— 128 元素组的 amax 是源 token 的属性,k 份拷贝的量化字节和 scale 完全相同。按源行并行是唯一不做冗余计算的选择,`src2dst`(形状 `[T,k]`,源侧索引)天然就是为这个遍历方向准备的。

#### 推理链:凭什么确定是"源行并行 + quant 在前"

| 环 | 论据 | 排除 |
|---|---|---|
| ① | 上游 `ep_scatter(recv_x, recv_x_scale, ...)` 签名带 scale → **输入已量化** → quant 在前 | 排除"先 scatter bf16 再 quant"(那种情形 fused 省 1074 MB) |
| ② | `src2dst` 形状 `[T,k]`,按 `(token, topk槽位)` 查目标行号,是**源侧索引**;目标行并行需要逆映射 `dst2src` | 指向源行并行 |
| ③ | **反证**:若目标行并行,每个 compact 行各读一次源 bf16 = M×H×2 = 537 MB,fused 总流量 814 MB **>** split 655 MB,应为**负收益** —— 与项目口径 1.6× 直接矛盾 | 排除目标行并行 |

> `[待核实]` 环 ① 依赖对上游签名的记忆,回代码扫一眼 `recv_x_scale` 即可坐实。**环 ③ 是独立自洽论证,即使环 ① 记错也仍成立** —— 所以 311 MB 这个数不会翻盘。

---

### 7.3.3 逐手段:what / why / how / 谁做的

#### 手段 1 — Expert-contiguous compact layout `[上游]`

- **Why**:grouped GEMM 要求同一 expert 的 token 内存连续;topk 后 token 按 request 打散,散着做 = 离散访存 + M 攒不大 → GEMM 效率低。
- **How**:
  ```
  原始:    token0→e3, token1→e7, token2→e3, token3→e1
  compact: e1:[t3], e3:[t0,t2], e7:[t1]
  ```
- **Metadata**:`src2dst`(原→compact,combine 反向用)、`m_indices`(每行属哪个 expert)、`padded_valid_rows` / `allocated_rows` / `worst_case_rows`、`overflow_flag`。
- **归属**:布局约定、`ep_scatter`/`ep_gather`、全套 metadata **上游已产出**;我放开 MUSA guard 并接进 MoeRunner。
- **代价**:compact 是一次 scatter、reorder/combine 是一次 gather,均 memory-bound → **这就是必须设 token 阈值的原因**。

#### 手段 2 — 两段 grouped FP8 GEMM `[上游]`

`grouped_gemm_nt_f8f8bf16_contig` **逐字拆解**(这段拆解本身就是加分项):

| 片段 | 含义 |
|---|---|
| **grouped** | 一次 launch 处理多 expert,避免 per-expert 单独 launch |
| **nt** | `Normal × Transposed` —— 激活 `[M,K]` 正常存、权重 `[N,K]` 转置存,**K 维连续** → TensorCore 读取最优 |
| **f8f8bf16** | FP8 激活 × FP8 权重 → bf16 累加。吃满 FP8 TensorCore(S5000 上 FP8 是 BF16 的 2 倍:1001 vs 501 TFLOPS) |
| **contig** | 输入必须是 compact 后的 expert-contiguous 布局 |

- **`m_indices` 机制** `[推断]`:标记每行属哪个 expert,让 kernel 按变长 M 分组、无需 padding 到统一 M。
- **★ 要点**:**DeepGEMM 不做 permute** —— 它只吃已经排好的输入。compact 是调用方的义务。**这就是手段 1/3 存在的全部理由。**

#### 手段 3 — ★ Fused quant-scatter kernel `[我的]`

```
warp-per-group 线程映射:
    group_linear = block_id * groups_per_cta + warp_id
    token_id / scale_group 由 hidden 维分解得到
组内 abs-max 并行归约(half-warp shuffle,不物化 amax 张量)
    → scale = amax / 448
    → FP8 E4M3 cast + pack
    → 按 src2dst 直接写 compact row 与 compact scale
```

收益见 7.3.2 算例 B:单层 ≈311 MB,整栈 ≈18 GB / S5000 ≈11 ms。另外少一次 kernel launch 和一个全量中间张量的分配。

#### 手段 3′ — Fused SwiGLU + FP8 quant `[上游]`

```python
silu_and_mul_contig_post_quant(
    input=gateup_output,          # 读一次 bf16
    output=down_input_fp8,        # 直接输出 FP8
    output_scale=down_input_scale,
    quant_group_size=scale_block_size,
    scale_ue8m0=..., transposed=..., swizzle=self.use_swizzle,
    swiglu_limit=...,
)
```
数据流 `读gateup→写bf16→读bf16→quant→写FP8` 变成 `读gateup→(寄存器)SwiGLU+quant→写FP8`。
配 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY`:直接分配 FP8 down input 并写 scale,连 bf16 buffer 都不分配。
**归属:上游 PR #4199 / #4343(V3 时代)。我只做接线。**

#### 手段 4 — reorder / combine `[上游]`

```
expert-contiguous down_output → 按 src2dst 回原序 → × router weight → top-k 累加 → final hidden
```
净收益 = (GEMM 连续化 + fusion 收益) − (compact + reorder/combine 成本)。

#### 手段 5 — Dispatch guard 与容量契约 `[我的]`

`_should_use_dsv4_musa_compact_prefill_deepgemm` 启用条件:
- MUSA + `use_fp8=True` + `SGLANG_DSV4_MUSA_MOE_EXPERIMENTAL` + 2D hidden + **token ≥ 8192**(`_DSV4_MUSA_COMPACT_MOE_MIN_TOKENS`)+ topk_ids 非空 + 能选到 fixed bucket rows
- 独立阈值 `_TILEKERNELS_SWIGLU_QUANT_MIN_ROWS = 1024`:fused SwiGLU+quant 只在 ≥1024 行启用,decode small-M 保留 split path
- `overflow_flag`:static bucket 撑爆 → 显式回退动态路径,**不截断 route**(截断会静默改变模型输出)

**本质**:compact 有固定成本,token 少时净收益为负。阈值是**盈亏平衡点**,不是保守 —— `[待核实]` 8192 / 1024 是否真有 sweep 数据支撑。

---

### 7.3.4 Static-cap / Graph Capture `[我的]`

`[推断,与源文档 is_paged 闭环逻辑一致]`

**矛盾**:MoE 的 token→expert 分配运行时动态(每步不同),compact buffer 大小和每组 M 都动态;但 serving 要 graph capture,capture 要求 shape 和地址静态。

**解法**:预分配**固定**大小 compact buffer + 四口径分账:

| 口径 | 含义 |
|---|---|
| `valid_rows` | 真实有效行 M = T×k |
| `padded_valid_rows` | 对齐到 GEMM tile 后的行数 |
| `allocated_rows` | 本次实际使用的 buffer 行数 |
| `worst_case_rows` | **静态 buffer 容量上界 = E × cap_per_expert** |

**取舍**(benchmark 里 `static cap vs exact compact speedup` 测的就是这个):

| | 优 | 劣 |
|---|---|---|
| static cap | buffer 固定、capture 友好 | expert 不均时 padding 浪费大 |
| exact compact | 无浪费 | shape 动态、capture 不友好 |

**★ 要点**:`worst_case_rows` **不是监控指标,是静态 buffer 容量上界**,直接决定能否 graph capture 和浪费多少。这是整个设计里最能体现"想过工程约束"的地方。

---

### 7.3.5 FP8 scale 布局陷阱(correctness)

- **为何需要 scale**:FP8 E4M3 范围仅 ±448。per-group 量化:每 128 元素算 amax,`scale = amax / 448`,缩放进 FP8 范围,scale 单独存。
- **参数含义** `[推断,DeepGEMM 通用约定]`:
  - `quant_group_size=128`:每 128 元素共享一个 scale
  - `scale_ue8m0`:scale 用 UE8M0(只存 2 的幂),反量化用移位、利于对齐
  - `transposed` / `swizzle`:scale 的内存排布必须与 GEMM 读 tile 的顺序匹配,让每个 tile 连续读到对应 scale
- **★ 陷阱**:quant 写 scale 的布局与 down GEMM 读 scale 的布局不一致 → 用错位 scale 反量化 → **数值系统性偏差,shape 对、不 crash、不 NaN**,极难查。
- **验收顺序**:先对齐 `src2dst` / `m_indices` / `scale` / `output` 的数值,**再**比 latency。数值没对齐之前的 speedup 没有意义。
- **附带收益**:融合让 quant 直接产出 GEMM 期望的格式,**消除了布局不匹配的风险面**。

---

### 7.3.6 完整 shape 走查(白板用)

```
1. hidden_states           [8192, 4096] bf16
2. router → topk           topk_ids [8192, 8]
3. topk 展开               M = 8192 × 8 = 65536
4. ★ fused quant-scatter   → compact_input [~65536(+pad), 4096] fp8
   [我的]                    + compact scale + m_indices + src2dst
                           ── memory-bound;省掉 x_q [T,H] 的写+散读 ≈ 311 MB/层 ──
5. gate/up grouped GEMM    [M,4096] fp8 × w13[E,8192,4096] fp8 → gateup [M,8192] bf16
   [上游]                  ── compute-bound, AI ≈ 2700 ──
6. fused SwiGLU + quant    读 gateup [M,8192] bf16 → silu(gate)*up → quant
   [上游]                    → down_input_fp8 [M,4096] + scale [M,32]
                           ── memory-bound;省 536 MB/层(上游的收益)──
7. down grouped GEMM       [M,4096] fp8 × w2[E,4096,4096] fp8 → down_output [M,4096] bf16
   [上游]                  ── compute-bound ──
8. reorder + combine       按 src2dst 回原序 × router_weight,8 expert 累加 → [8192,4096] bf16
   [上游]                  ── memory-bound (gather) ──
```

第 5/7 步是 compute-bound(DeepGEMM 负责,不用动);**第 4/6/8 步是 memory-bound 战场,其中只有第 4 步是我写的 kernel。**

---

### 7.3.7 验收与 benchmark

`test_moe_prefill_deepgemm_benchmarks.py`
```bash
SGLANG_RUN_DEEPSEEK_V4_MUSA_BENCHMARK=1 \
SGLANG_DSV4_MUSA_MOE_EXPERIMENTAL=1 \
python -m pytest .../test_moe_prefill_deepgemm_benchmarks.py -s
```
输出:`baseline/candidate device median/min/p95`、`device_speedup`、`host_speedup`、`padded/allocated/worst_case rows`、static cap vs exact speedup、preprocess split report。
**device 时间才是 kernel 真实耗时;`host_speedup` 用来防止调度开销吃掉收益。**

**三层验收**:
1. **dispatch**:MUSA + FP8 + 大 token 才进;decode / small-M 不进;env 关不进
2. **correctness**:token 顺序对、top-k combine 对、FP8 scale layout 与 DeepGEMM 一致、误差可接受
3. **performance**:`candidate_device < baseline_device`、speedup 稳定、padding / overflow 可控

---

### 7.3.8 边界 / 局限(主动承认)

1. **padding 浪费**:static bucket + 对齐,expert 越不均浪费越大(靠 `worst_case_rows` / static cap 对比监控)
2. **只覆盖大 prefill**:8192 阈值以下(含 decode)走原路径,**刻意保护 TPOT**
3. **scale 布局强耦合 DeepGEMM**:不一致即难查数值 bug,且不报错
4. **experimental gate 默认关**:说明当时未默认上线,有稳定性顾虑 —— 主动承认,别装成已上线

---

### 7.3.9 一句话主线(背这个)

> DeepGEMM 的 contiguous grouped GEMM 和 `ep_scatter` 都是上游的,我的工作是把这条路径在 MUSA 上跑通,并针对 S5000 的访存特性重排优先级。
>
> 先用 Roofline 定优先级:S5000 的 FP8 脊点约 626 —— 和 H200 同级,但**绝对带宽只有它的三分之一**,而且是 H20 的十倍脊点,所以上游按 H20 调的取向不能直接搬。算下来 gate/up 和 down GEMM 的 AI 有 2700,远在脊点右侧,DeepGEMM 已经吃满 TensorCore,不用动;真正拖时间的是 GEMM 之间的 scatter / SwiGLU / quant / reorder,AI 只有个位数,纯 memory-bound。同一笔 536 MB/层的流量,H200 上值 6.5 ms,S5000 上值 19 ms —— **这就是我把重心放在消除中间张量而不是调 GEMM 的依据**。
>
> 具体做了三件事:第一,上游把量化和 scatter 做成两个独立算子、中间要落一整个 `[T,H]` 的 fp8 张量,我写了一个 MUSA kernel 把它们融进一趟 —— warp-per-group 映射、组内 shuffle 求 amax、FP8 pack 后按 `src2dst` 直写 compact row,**按流量公式算单层省约 311 MB、整栈约 18 GB、S5000 上约 11 ms**;值得说明的是融合前后读端和写端的必需流量完全一样,省下的 100% 是中间张量的物化。第二,设计 fixed-bucket 静态容量契约,四口径分账,让动态的 MoE 路由能在静态 shape 下 graph capture,热 expert 超容时置 flag 显式回退、不截断 route。第三,按 token ≥ 8192 和 shape 分档 dispatch,因为 compact 有固定成本,小 batch 净收益为负,decode 必须走原路径。
>
> 项目口径 MoE 算子约 1.6×、端到端 prefill 约 +8%,但这两个数我没有独立复测,证据等级只能算 B。

---

### 7.3.10 追问速查

| 问 | 答 |
|---|---|
| 怎么知道该优化哪里? | 算脊点 + 两个 AI:S5000 脊点 626,GEMM 的 AI 2700 是 compute-bound,glue 是个位数 memory-bound |
| 为什么这在 MUSA 上比 NV 上更重要? | 脊点是 H20 的 10 倍(上游取向不通用)+ 绝对带宽只有 H 系列 1/3(减字节收益 ×3) |
| **DeepGEMM / ep_scatter 不是开源的吗?** | **主动先说**:是,库和布局约定都是上游的。我的产出是 quant-scatter 融合、容量契约、dispatch 分档 |
| 收益主要来自哪? | 诚实:上游的 SwiGLU fusion 省 536 MB/层是最大一块;我的 quant-scatter 省 311 MB/层,约是它六成 |
| quant-scatter 具体省多少? | 单层 311 MB、整栈 18 GB、S5000 约 11 ms。拆开是 x_q 一次写 33.6 + 一次散读 268.4 + scale 往返 9.5 |
| 怎么确定是源行并行? | 三条,见 7.3.2 推理链;**其中反证独立成立** |
| compact 有成本,怎么保证净收益正? | token ≥ 8192 阈值 + benchmark 的 padding / static-cap 对比;分布不均靠 overflow fallback |
| down GEMM 为何要 FP8 输入? | 吃 FP8 TensorCore 峰值(S5000 上是 BF16 的 2 倍),代价是 SwiGLU 后要 quant,所以上游把它融进了 SwiGLU |
| `worst_case_rows` 干嘛的? | 静态 buffer 容量上界,为 graph capture 服务,**不是监控指标** |
| **1.6× 怎么测的?** | 诚实:项目内口径,证据等级 B,配置待补,我没独立复测。要复测就跑那个 benchmark 取 device median |
| 为什么 experimental 默认关? | 当时未默认上线,有稳定性顾虑 —— 这是事实,不掩饰 |
| 和你 FP8 quant 那条(7.9)什么关系? | 那条优化单独的 quant kernel,这条优化 MoE pipeline 里的 quant + 中间内存,二者叠加 |

### 软肋

- **最大的一块流量收益不是自己的** → 必须主动说,别等对方算出来。
- `[T,H]` vs `[M,H]`、源行 vs 目标行,讲的时候容易口误 → **白板上先把 T、M、k 写出来**。
- 所有阈值(8192 / 1024)都缺 sweep 证据。

---

## 7.4 Sparse Indexer TopK-512 Select

**归属**:`[我的]` MUSA kernel + graph-safe 机制。

> ⛔ **绝对不要说"实现了 TopK V2"。** `topk_transform_512_v2_musa` 只校验 metadata shape 后**转调 v1**,cluster-persistent scheduling 未实现(源文档 484 行自曝)。而上游有真 V2(PR #23882 引入 / #25406 设为默认)。这题一旦说错,面试官只要问一句"你们的 cluster persistent 是怎么调度的"就穿了。

### 这不是普通 TopK

普通 MoE TopK 输出的是 expert id。这个 kernel 输出的是 **paged KV cache 的物理页内偏移**,中间要做一次索引变换:

```
logical_page = r >> log2(page_size)
physical     = page_tables[row, logical_page]
page_index   = (physical << log2(page_size)) | offset
invalid      → -1(哨兵,下游 padding 到 128)
```

也就是说:**选 top-512 和"把逻辑位置翻译成物理页地址"是在同一个 kernel 里完成的**,省掉一趟全量索引张量的往返。

### 机制:四轮 byte-radix exact select

大 shape 走精确选择:

```
对 32-bit key 按字节分 4 轮,每轮:
    count(直方图) → 定位第 512 名所在的 bucket → 确定阈值
最后:gather → prefix sum → write
```

比排序快,因为只需要第 512 名的阈值,不需要全序。

### ★ graph-capture 安全的 exact/candidate 双跑

这是最能体现工程判断的一段:

```
问题:candidate 路径(先取有界候选集再精选)快,但可能 overflow;
      判断有没有 overflow 需要把 flag 读回 host  →  host 同步  →  破坏 graph capture

解法:先无条件跑 exact(保证正确),再跑 bounded candidate;
      candidate 没 overflow 时,在 device 侧直接覆盖 exact 的结果
      →  全程无 host 同步,graph 可捕获
```

**代价是多跑一遍 exact,换来的是可 graph capture。** 这个取舍要讲出来 —— 它说明你知道 serving 场景下 graph capture 的价值高于单 kernel 的极致性能。

### ★ 真实定位(可以当"最难的 bug"讲)

```
现象:candidate 路径在生产 shape [16, 262208] → [16, 512] 上失败
原因:candidate_capacity = 1024。单行要扫 262208 个候选,
      top-512 的真实分布下 1024 的候选窗口装不住
结论:candidate 不能作默认路径,必须 exact 兜底
```

这条的价值在于:**它是一个只有在真实生产 shape 下才暴露的问题**(小 shape 测试全过),而且结论直接改变了架构决策(默认路径的选择)。

### 能扛的追问

| 问 | 答 |
|---|---|
| 为什么不直接排序? | 只需要第 512 名的阈值,不需要全序;byte-radix 4 轮就能定阈值 |
| 为什么要 exact + candidate 都跑? | graph capture 不允许 host 同步,而判断 candidate 是否 overflow 需要读回 flag。跑两遍换可捕获性 |
| 那不是浪费一倍时间? | exact 是兜底,candidate 命中时结果被覆盖。净成本是 exact 一遍,收益是整个 decode 图可捕获 |
| **候选容量为什么不调大到装得下?** | 容量要按最坏分布定,262K 行的最坏情况接近全量,那就退化成 exact 了 |
| −1 哨兵怎么处理? | 下游 padding 到 128,由 attention kernel 的 mask 消化 |

### 软肋

- ⛔ V2 那道坎(见开头)。
- byte-radix 每轮的具体 bucket 划分 `[待补]`。

---

## 7.5 Pack8(sparse 章唯一上游没有的)

**归属**:`[我的]`。上游 **没有** pack8 变体 —— 这是整个 FlashMLA sparse 章节里唯一能算个人产出的东西。

> ⚠️ 同一章里的 **seq-pack / flat workspace / rebased indices / chunk 级跨层复用全部是上游的**(PR #25418 起已在 main)。源文档给的辩护是"我们比上游早 6 天",这个理由**不能用**。讲 sparse 时必须先划清:"这一章大部分是上游的,我的增量是 pack8。"

### 是什么

`flash_mla_sparse_fwd_pack8` + 两个 metadata 打包函数:
- `pack_row_level_indices_for_pack8`
- `pack_structured_dsv4_indices_for_pack8`

把 8 个 query 打包进一次 attention 调用。

### 正确性约束(这部分是真功夫)

打包引入了两类风险,都要在 metadata 层解决:

| 风险 | 解法 |
|---|---|
| padding lane 参与了 attention → 污染结果 | **row mask** —— padding 行不贡献 |
| pack 内不同 query 的有效长度不同 → 读到别人的 KV | **pack lens** —— 每个 query 只读自身有效 prefix |
| shape 不对齐 | 硬 guard:`q.shape[0] % 8 == 0`,且**每个 request** 的 `extend_seq_len % 8 == 0` |
| request 尾部不足 8 | request padding |

**注意第三行是"每个 request 都要整除",不是总数整除** —— 因为 pack 不能跨 request 边界。这个细节能证明你真的实现过。

### 能扛的追问

| 问 | 答 |
|---|---|
| 为什么打包能加速? | 小 query 数下 attention kernel 的固定开销占比高,打包摊薄 launch 与 tile 固定成本 |
| 为什么是 8? | `[待补]` —— 大概率对齐 warp/tile 结构 |
| padding 怎么保证不污染? | row mask + pack lens 双重约束,见上表 |
| 为什么要求每个 request 单独整除? | pack 不能跨 request 边界,否则一个 pack 内会混入两个 request 的 KV |

### 软肋

- 整章其余部分是上游的 → 必须主动交代。
- 8 这个数的理由 `[待补]`。

---

## 7.6 HC-head Split-K 两段 Decode Kernel

**归属**:`[我的]` MUSA kernel。Split-K 是通用范式,MUSA 实现是本地的。

### 物理动机

decode 阶段的 linear 层是**小 M、大 K**:M 是 batch(几十),K 是 hidden(几千)。

```
常规 GEMM 按 M×N 切 tile 分配 block
  →  M 只有几十  →  block 数远少于 SM 数  →  大部分 SM 空转
```

**Split-K**:把 K 维也切开,每个 block 算一段 K 的部分和,再归约。用"多一次归约"换"并行度够用"。

### 机制

```
stage0:按 K 维切 split 份,每份独立累加出 partial 结果
stage1:reduce,把 split 份 partial 相加
另有 warp variant `[待补:与主版本的分工]`
split 数经 sweep 选定
```

> **`split 数经 sweep 选定` 是全项目唯一有 sweep 证据的参数**(源文档 MHC prenorm 处)。这条可以理直气壮说"标定过";**其他常数(8192 / 1024 / 16K / T≥128 / groups_per_cta=8)都没有 sweep 证据,只能说"按 shape 分档"。**

### 能扛的追问

| 问 | 答 |
|---|---|
| 为什么 decode 要 Split-K? | M 小 → block 数不足 → SM 空转。切 K 维补并行度 |
| Split-K 的代价? | 多一次全局归约 + partial buffer 的读写;split 太大时归约开销反超收益 |
| split 怎么定的? | **sweep 选的** —— 这条有证据,可以直说 |
| 和 decode 的 memory-bound 属性矛盾吗? | 不矛盾。AI≈2B,B 小的时候本来就是访存受限,Split-K 解决的是并行度不足导致的**带宽跑不满**,不是算力问题 |

### 软肋

- warp variant 的定位 `[待补]`。

---

## 7.7 Paged C4 Decode Compress Kernel

**归属**:`[我的]` MUSA kernel。

### 机制(索引与数值公式都完整,这是它的强项)

```
block_id  = indices[token]           ← 从 page table 拿物理块
write_pos = (seq_len + 3) % 4        ← C4 是 4 槽压缩,轮转写入位置
遍历有效 source slot:
    exp-weighted numerator / denominator 累加
```

`(seq_len + 3) % 4` 这个式子值得解释:C4 压缩把 4 个 token 压成 1 个槽位,decode 每步新增 1 token,写入位置在 4 个槽间轮转。`+3` 是因为 `seq_len` 是**新增后**的长度,要换算回**待写槽位**。

**能把这个 off-by-one 讲清楚,比讲什么高深算法都有说服力** —— 它说明你真的在这段代码里 debug 过。

### 能扛的追问

| 问 | 答 |
|---|---|
| 为什么是 `+3`? | seq_len 是新增后长度,减 1 再对 4 取模等价于 `(seq_len+3)%4`。写的是本次 token 应落的槽 |
| exp-weighted 是什么加权? | `[待补]` —— 大概率是 attention score 的指数加权,与 online softmax 同构 |
| 为什么 decode 要单独一个 compress kernel? | prefill 是批量压缩、decode 是每步增量一个 token,并行结构完全不同 |

### 软肋

- exp-weighted 的具体权重语义 `[待补]`,别硬答。

---

## 7.8 Online C128 × MTP Deferred-Commit 状态机

**归属**:`[我的]` 系统设计。**注意:编排的是已有 triton kernel,没有新 kernel** —— 讲的时候定位成"状态机设计"而非"kernel 优化"。

### 物理动机:投机解码和有状态压缩天然冲突

```
MTP(投机解码):draft 模型先猜 n 个 token,target 模型 verify,可能只接受前 k 个
C128(在线压缩):每 128 个 token 压一次,压缩状态是持久的、跨步累积的

冲突:如果 forward 时就把 draft token 写进 C128 持久状态,
      而 verify 拒绝了其中一部分  →  被拒绝的 draft 已经污染了压缩状态  →  无法回滚
```

### 机制:两阶段提交

```
1. reset_c128_verify_kv_cache            ← verify 前清空暂存区
2. cache_c128_verify_kv_score(每层)      ← forward 时只往暂存区写,不碰持久状态
3. update_c128_online_state_after_mtp_verify
                                          ← verify 产出 accepted rows 后,
                                            逐层逐行把被接受的回放进持久 state
```

**这就是数据库的两阶段提交搬到 KV cache 上。**

### 索引换算链(能扛追问的硬细节)

```
position  = floor((seq_len - 1) / 128) * 128    ← 对齐到 128 边界
          → raw_loc                              ← 原始 KV 位置
          → swa_loc                              ← sliding window 内位置
          → state_loc = swa_page * ring_size + (swa_loc % ring_size)
          → / 128                                ← 落到 C128 槽
```

四层地址空间(逻辑序列位置 → 原始 cache → 滑窗 → ring buffer),**每一层都可能 off-by-one**。

### 127→128 边界

跨越 128 边界时必须**恰好触发一次** compress/store:
- 少一次 → 压缩状态漏掉一个块,后续全错位
- 多一次 → 重复压缩,数值污染

而在 MTP 场景下,draft 可能跨越边界后又被拒绝 —— 所以边界判断必须发生在 commit 阶段,不能在 forward 阶段。

### 能扛的追问

| 问 | 答 |
|---|---|
| 为什么不直接在 forward 时写,拒绝了再回滚? | C128 是有损压缩,状态被合并进去就回滚不了。只能延迟提交 |
| 暂存区的内存开销? | 每层一份 kv score,`[待补:具体大小]` |
| 127→128 边界怎么保证恰好一次? | 判断放在 commit 阶段,按 accepted rows 的最终 position 算 |
| 这算 kernel 优化吗? | 诚实:不算,这是状态机设计。kernel 是已有的 triton |

### 软肋

- 无新 kernel → 别放在"我写了 N 个 kernel"里凑数。
- 暂存区开销说不清。

---

## 7.9 FP8 Per-Token-Group Quant + Shape-Aware Dispatch

**归属**:⚠️ **混合,要小心措辞。**
- kernel **结构**(16 lane × 8 值 + shuffle amax + 向量 pack)= 上游 sgl-kernel `per_token_group_quant_8bit.cu` 的标准结构 → **只能说"MUSA 重实现",不能说"设计"**
- **dispatch 策略**(16K guard + 三分档)= `[我的]`

### kernel 结构

```
group = 128 元素共享一个 scale
  → 16 lane × 8 值 = 128     ← 每 lane 8 个值 = 128-bit 宽加载
  → half-warp(16 lane)内 shfl_xor 归约求 amax,寄存器内完成,不物化 amax 张量
  → scale = amax / 448       ← FP8 E4M3 范围 ±448
  → FP8x4 pack + stg32 宽写回
2 groups/warp,groups_per_cta = 8
```

**为什么是 16 lane × 8 值**:8 个值恰好是 128-bit,是单指令最宽的加载粒度;16 lane 凑满 128 元素的组。这个映射让"一个 group 的归约"落在 half-warp 内,用 `shfl_xor` 就能完成,不需要 shared memory。

> ⚠️ 但这套结构上游就是这么写的。**被问"这个映射是你设计的吗",答"这是 sgl-kernel 的标准做法,我在 MUSA 上重新实现了"** —— 抢答比被拆穿好。审计发现源文档也自己承认:"MUSA 可计成果是 TileKernels bring-up、S5000 专用 dispatch 和 Kernel 重写,**不是 FP8 quant 或 fused activation 的发明**"。

### ★ Shape-Aware Dispatch(这部分才是自己的)

```python
_MUSA_PREFILL_FP8_QUANT_MIN_GROUPS = 16 * 1024

# total_groups >= 16K 时,主动返回 None,阻断 generic cast helper
# → 防止大 CP prefill 落进已知的慢路径
```

三分档 launch config:

| 条件 | groups_per_cta | num_warps |
|---|---|---|
| hidden ≤ 2048 | 8 | 4 |
| hidden > 2048 且 rows ≤ 4096 | 4 | 2 |
| 其余 | 8 | 2 |

**"主动返回 None 挡住 generic cast"是真发现** —— 它的意思是:框架里有一个通用的 cast helper,在大 group 数下性能很差,但它会静默接管。与其等它接管,不如显式拒绝,强制走专用路径。这是"知道框架哪里会坑你"的体现。

### 能扛的追问

| 问 | 答 |
|---|---|
| 这个线程映射是你设计的吗? | **诚实**:sgl-kernel 的标准结构,我在 MUSA 上重实现 |
| 为什么每 lane 8 个值? | 8×fp16/bf16 = 128 bit,单指令最宽加载 |
| 为什么用 shfl 不用 shared memory? | 128 元素的组正好落在 half-warp 内,寄存器归约省掉 shared 的读写和同步 |
| 16K 这个阈值怎么来的? | `[待核实]` 是否有 sweep。没有就说"经验值,目的是挡住 generic helper" |
| 三分档的依据? | hidden 大时每 CTA 要处理的数据多,减少 groups_per_cta 避免寄存器压力;`[待核实]` 是否 sweep 过 |

### 软肋

- **算法非原创**,一定要抢先说明。
- 所有常数都没有 sweep 证据。

---

## 7.10 其余可提但不宜深挖的

这几条有实质内容,但证据链不足以支撑长时间追问。**简历可以带一句,面试主动讲则不划算。**

| 项 | 内容 | 为什么不宜深挖 |
|---|---|---|
| **PP compact pointer table 分段切片** | `2*c4_full + c128_full` 三段 offset + SWA layout 变体 + pointer 数校验 | 易错的正确性工作,但没有性能故事,也没有定位过程 |
| **CP round-robin local metadata 重建** | round-robin 是交错分配,global `query_start_loc` 不能直接 slice,必须重建 local metadata | 语义是真的(上游 PR #13959),但**源文档没有任何定位过程** —— 讲成"我 debug 出来的"会露馅 |
| **C4 page reduce / C128 parallel reduce** | 归约 kernel | 源文档只有"parallel reduce"几个字,机制描述太薄 |
| **Indexer cache pack/store 双路径** | 与 7.2 同模式的 decode/prefill 分路 | 是 7.2 的复刻,讲了会显得在凑数量 |
| **HiSparse MUSA host-offload copy kernel** | 固定 layout 的 pointer copy | 复杂度低 |

---

## 7.11 全局追问:三道必答题

这三道不针对具体某条,但**几乎一定会问**,答不好会否定整份简历。

### Q1 「这些不都是 SGLang 开源的吗?你做了什么?」

> 对,DeepGEMM、FlashMLA、ep_scatter、SwiGLU fusion、seq-pack 这些都是上游的。**CUDA kernel 在 MUSA 上一行都跑不了**,所以移植的工作量在于用 TileLang 和手写 MUSA C++ 重新实现这些算子,并且按 S5000 的访存特性重新做取舍。
>
> 真正算我自己设计的有三类:一是上游没有的融合 —— quant-scatter、norm-rope-cache;二是上游没有的变体 —— pack8;三是平台相关的机制 —— fixed-bucket 容量契约、graph-safe 的 exact/candidate 双跑、MTP 的 deferred commit。

### Q2 「性能数据是怎么测的?」

> **诚实**:MoE 约 1.6×、prefill 约 +8%、TPOT 7.3→6.6 ms 这几个是项目内提供的口径,**我没有独立复测,完整的硬件/shape/baseline commit 记录也不全**。
>
> 我能负责的是推导值:比如 quant-scatter 单层省 311 MB,这个是按 split 和 fused 两条路径的读写字节算出来的,给定 shape 可以当场复现。
>
> **不要说**"我们复测稳定提升 1.6×"。

### Q3 「S5000 和 NV 卡的差异对你的优化有什么影响?」

> S5000 的 FP8 脊点约 626 —— 和 H200(约 696)同级,但**是 H20(约 62)的十倍**。上游很多 kernel 是按 H20 这类"算力砍了、带宽留着"的推理卡调的,在那里几乎一切都还没掉进 memory-bound 区,搬过来后大批算子落到脊点左侧,调优方向要反过来。
>
> 另一条是绝对带宽:1.6 TB/s 只有 H 系列 4.8 TB/s 的三分之一,**同样的字节数壁钟时间是三倍**。所以"少读写一次中间张量"这件事在 S5000 上的杠杆是 H200 的三倍 —— 这是我把工作重心放在算子融合和消除中间物化、而不是调 GEMM 的直接原因。
>
> ⚠️ **不要说"S5000 脊点特别高"** —— H200 比它还高。准确表述是"**脊点像 H200,带宽只有三分之一**"。
