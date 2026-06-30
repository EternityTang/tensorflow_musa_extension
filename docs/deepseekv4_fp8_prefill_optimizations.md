# DeepSeek V4 FP8 MUSA Serving Bring-up：Prefill / Decode 优化与 Runtime 修复梳理

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



## 1. 故事主线：DSV4 FP8 MUSA serving bring-up

这批优化和修复可以放在同一个背景下理解：目标不是只让 DeepSeek V4 / DSV4 FP8 做一次离线 forward，而是把它推进到 **MUSA 上可稳定、高性能服务化运行** 的状态。

服务化推理相比单次 forward 多了很多约束：

```text
1. prefill 要能处理长上下文和大 batch
2. decode 不能因为 prefill fast path 污染而 TPOT 回退
3. FP8 activation / scale / weight path 要稳定
4. MoE 大 token 场景要能吃满 DeepGEMM
5. FlashMLA sparse attention 要支持 compressed cache + SWA + CP
6. KV cache / compressed cache 要正确、高带宽写入
7. 多卡 MCCL / CP metadata / graph capture 要稳定
8. sampling 要稳定，不能在最后一步生成异常 token
9. fast path miss 不能 silent fallback 到 CUDA-only / torch 慢路径
```

因此本文档按下面的故事顺序组织：

| 阶段 | 目标 | 对应内容 |
| --- | --- | --- |
| 第一阶段：runtime hardening | 先保证 MUSA serving 路径不跑错、不 silent fallback、多卡和 CP metadata 正确 | MUSA sampling、fallback blockers、MCCL device binding、TileLang JIT/cache、CP round-robin、KV contiguous 等修复 |
| 第二阶段：decode / TPOT 路径 | 补齐 decode fast path、TileLang decode attention、RoPE decode、cache store decode 分流和 sampling，回答“不是只优化 prefill” | DSV4 MUSA decode fast paths、TileLang decode queue4/prefix-tail、page32/head256、RoPE decode、paged decode、MHC decode split、sampling |
| 第三阶段：FP8 基础瓶颈 | 先把 prefill 中大量 FP8 GEMM 依赖的 activation quant 做快 | FP8 prefill quant 路径优化 |
| 第四阶段：prefill 主路径 | 优化 DSV4 prefill 里最重的 MoE 和 sparse attention 主路径 | MoE DeepGEMM compact prefill、FlashMLA sparse prefill seq-pack |
| 第五阶段：shared memory-bound 算子 | 当主路径变快后，补齐 cache store、compress、norm、rope、topk、MHC 等辅助算子；其中 cache / norm / rope / MHC 有 decode 与 prefill 双路径 | FlashMLA cache store decode/prefill 分流、compress fused norm rope、C4/C128 compress、MHC/norm/rope/topk |
| 验收阶段 | 用 operator benchmark、MoE benchmark、decode/cache/MHC benchmark、环境变量和 strict miss 机制确认路径命中与性能 | benchmark 状态、环境变量、建议阅读顺序、当前缺口 |

一句话概括：

> 这批工作是一条完整的 **DeepSeek V4 FP8 MUSA serving bring-up** 路线：先保证不跑错，再补齐 decode / TPOT 路径，然后优化 FP8 prefill 基础算子、MoE 和 FlashMLA 主路径，最后补齐 shared memory-bound 辅助算子，并用 benchmark 和 strict fallback 把性能与正确性兜住。


### 1.1 面试总回答模板

如果面试官让你整体介绍这批工作，可以按下面结构回答：

> 这批工作不是单点 kernel 优化，而是 DeepSeek V4 FP8 在 MUSA 上的完整 serving bring-up。第一阶段先做 runtime hardening，解决 sampling、fallback、MCCL device binding、TileLang JIT cache、CP metadata、KV contiguous 等稳定性问题，确保 MUSA 路径不 silent fallback、不跑错。第二阶段补齐 decode / TPOT 路径，包括 DSV4 MUSA decode fast path、TileLang decode attention、RoPE decode、cache store decode 分流、MHC decode split 和 paged decode kernel。第三阶段开始优化 prefill 的基础瓶颈 FP8 quant；第四阶段优化 prefill 主路径 MoE DeepGEMM compact 和 FlashMLA sparse seq-pack；第五阶段补齐 cache store、compress、norm、rope、topk、MHC 等 shared memory-bound 辅助算子。整体目标是让 DSV4 FP8 在 MUSA 上从“能跑”变成“能稳定、高性能地做 prefill / decode / CP / 多卡 serving”。


## 2. 总览与关键提交清单

### 2.1 总览


目前与 `deepseekv4-fp8` 在 prefill 阶段性能优化最相关的两个远端分支是：

| 分支 | 定位 | 说明 |
| --- | --- | --- |
| `origin/deepseek_v4_kernels_opt` | DSV4 MUSA / FP8 kernel 优化底座 | 引入并清理 DeepSeek V4 MUSA 专用 kernel/ops，包括 cache store、MHC、FP8 quant、SwiGLU quant、topk、WO_A 等。 |
| `origin/perf/yunqiao.jiang/dsv4_prefill` | DSV4 prefill 专项优化分支 | 在 kernel 底座上继续加入 MoE DeepGEMM prefill、FlashMLA sparse prefill seq-pack、prefill memory-bound kernel、FlashMLA cache store prefill、sparse prefill pack8 等优化。 |

一句话总结：

- `deepseek_v4_kernels_opt` 主要解决“DSV4 FP8 在 MUSA 上有哪些专用高性能算子可用”。
- `perf/yunqiao.jiang/dsv4_prefill` 主要解决“prefill 大 M / 大 token 场景如何走更快的 MoE、attention、cache store、compress、quant 路径”。
- 本文档只保留 **MUSA / DSV4 分支新增或修改的实现**；纯粹继承自上游 SGLang 的通用 FP8 quant API、通用 Triton fallback、通用 scale layout 等不再展开。

### 2.2 关键提交清单

#### 2.2.1 `origin/deepseek_v4_kernels_opt`

| Commit | 标题 | 作用 |
| --- | --- | --- |
| `ef32b80f4` | `Port DeepSeek V4 MUSA optimized kernels` | 引入 DeepSeek V4 MUSA 专用 kernel/ops 框架，覆盖 cache、MHC、hc_head、模型接入等。 |
| `78be4b50a` | `Optimize MUSA FP8 quant paths with TileKernels` | 优化 FP8 per-token/group quant，prefill 大 shape 走 TileLang/TileKernels，decode 小 M 保持原路径。 |
| `3c6b17eae` | `Port DeepSeek V4 MUSA optimized kernels` | 大量补充 benchmark/test 与 compress、norm-rope、routing、topk、swiglu quant 等 kernel。 |
| `abe778217` | `Tune DeepSeek V4 MUSA quant dispatch` | 调整 FP8 quant dispatch 条件与 benchmark。 |
| `ea3e93c38` | `Optimize DeepSeek V4 MUSA MHC prenorm` | 优化 MHC pre-norm 路径。 |
| `392179e92` | `Optimize DeepSeek V4 MUSA indexer cache store` | 优化 indexer cache store。 |
| `4fabebf7f` | `[DeepSeekV4] feat: enable flash_mla prefill for large Q (up to 32K)` | 启用 large-Q FlashMLA prefill，支持最高 32K Q 的 prefill attention 路径。 |

#### 2.2.2 `origin/perf/yunqiao.jiang/dsv4_prefill`

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


## 3. 第一阶段：runtime hardening 与高价值 MUSA 修复

### 3.1 只保留 MUSA 相关修复的口径

本文档只记录 **DeepSeek V4 / DSV4 在 MUSA 上运行相关** 的优化和修复。筛选标准是：提交需要涉及 MUSA runtime、MUSA kernel、MCCL、MUSA graph capture、MUSA fallback，或 DSV4 在 MUSA 上会使用的 shared code path。

### 3.2 Flash / Pro 在 MUSA 文档里的处理方式

从提交历史和测试命名看，DeepSeek V4 相关工作里有 **Flash** 和 **Pro** 两类模型/场景，但很多 MUSA 底层修复并不会显式标注 Flash 或 Pro。

| 类别 | MUSA 文档中的处理方式 | 示例 |
| --- | --- | --- |
| DSV4 Flash on MUSA | 只保留标题、代码路径或测试明确关联到 DSV4 Flash，且修复会影响 MUSA 运行的提交 | `Fix/dsv4 flash eagle dummy ima` 如果用于 MUSA dummy / EAGLE / IMA 验证，可保留。 |
| DSV4 Pro on MUSA | 只保留 Pro 相关且代码路径涉及 MUSA runtime / MUSA kernels / MUSA distributed / DSV4 shared MUSA path 的提交 | `Support DSV4 Pro DP attention and CP8 serving` 是 Pro serving 能力提交，但需要结合 MUSA 部署场景判断。 |
| 通用 DSV4 on MUSA | 保留会影响 MUSA DSV4 attention/cache/compress/runtime 的通用修复，即使标题没有写 Flash/Pro | MUSA fallback blockers、MCCL device binding、TileLang JIT name、MUSA prefill block_id re-bind 等。 |

### 3.3 值得关注的 MUSA 修复提交

| Commit | 分类 | 问题 | 修复要点 | 价值 |
| --- | --- | --- | --- | --- |
| `sampler.py` | MUSA runtime / sampling shared path | DeepSeek V4 MUSA 在线服务中，`sampling_seed is None` 的 no-seed sampling 可能因 MUSA 上 `torch.multinomial` 路径不稳定而出现偶发异常 token / 提前 EOS / 空输出等问题 | 在 `Sampler._sample_from_probs` 中对 `is_musa() and sampling_seed is None` 单独走 `sgl_kernel` sampling：`top_p_sampling_from_probs` / `top_k_top_p_sampling_from_probs` / `min_p_sampling_from_probs`；seeded sampling 保留 torch path | 提升 DeepSeek V4 MUSA 普通在线生成稳定性，同时不改变 seeded deterministic sampling 语义。 |
| `5fcb7c439` | MUSA runtime / fallback | DeepSeek V4 MUSA 运行时可能静默进入不支持的 CUDA JIT 或 torch fallback 路径，strict no-fallback E2E 被阻塞 | MUSA tensor 进入 `hisparse_offload_to_host` 时 fail-closed；MUSA rmsnorm TileLang miss 后默认不允许 torch fallback，graph capture 下直接报错；compress prefill miss 可通过 env 强制抛错 | 避免 MUSA 路径“悄悄跑到 CUDA/torch fallback”，能更早暴露缺失 kernel，是稳定上线 MUSA DSV4 的关键修复。 |
| `ef4469cb5` | MUSA distributed / runtime path | MCCL process group 没有显式绑定 local MUSA device；MUSA operator forwarding patch 容易不同步 | `init_process_group(backend="mccl")` 时传入 `device_id=torch.device("musa", local_rank)`；同步 DeepSeek V4 MUSA ops forwarding exports | 修复多卡 MUSA 初始化和 runtime path 的基础稳定性问题，避免通信后端设备绑定错误。 |
| `3cceb08b9` | MUSA TileLang JIT cache | C4 decode page kernel 对不同 `extra_data_cols` 使用同一个 TileLang JIT 名称，paged mode 与 page4 mode 切换时可能命中错误缓存 | JIT 名称从固定 `dsv4_c4_decode_page4_t{threads}` 改为包含 `extra_data_cols` 的 `dsv4_c4_decode_page{extra_data_cols}_t{threads}` | 修复 MUSA TileLang kernel cache collision，避免不同 layout 复用错误编译产物。 |
| `81fc824a2` | MUSA compress / prefill kernel | MUSA C4 prefill page write kernel 中 `block_id` 被 re-bind，触发 TileLang immutable value re-bind 问题；DeepSeek V4 compressor 没启用 paged 模式 | 使用 `T.alloc_local((1,), dtype=T.int32)` 保存可变 `block_id`；`Compressor.compress_fused()` 中 `is_paged=True` | 修复 MUSA prefill compress page write kernel 的编译/语义问题，同时启用 paged compression metadata。 |
| `b9b202c28` | MUSA prefill / CP shared path | DSV4 sparse prefill 在 CP round-robin split 下，本地 q metadata 与 global q metadata 不一致 | 在 `_build_sparse_prefill_cache_inputs()` 中识别 round-robin CP local q，重建 local query_start、local extend seq lens、req_pool_indices、seq_lens | 对 MUSA prefill CP serving 很关键，避免 sparse prefill 在 CP round-robin 下读错 metadata。 |
| `5a15cde85` | MUSA CP shared path | DeepSeek V4 CP 路径中 bf16 KV 可能因非 contiguous 输入触发 fused norm rope / all-gather 后续错误 | 在 `_compute_kv_bf16()` 中对 `kv` 增加 `kv = kv.contiguous()` 后再做 fused norm rope | 修复 NSA prefill-CP 需要 bf16 KV all-gather 时的 layout 问题，可影响 MUSA CP 路径。 |

### 3.4 DeepSeek V4 MUSA sampling 与运行时适配

> 说明：该修复位于 shared sampler runtime path，主要文件是 `python/sglang/srt/layers/sampler.py`，不是 `deepseek_v4.py` 中的模型专属逻辑。但 DeepSeek V4 / DSV4 在 MUSA 上做在线服务化推理时会经过这条 sampler 路径，因此本文按 **DeepSeek V4 MUSA runtime shared-path 修复** 记录。

#### 3.4.1 背景问题

DeepSeek V4 在 MUSA 上服务化推理时，不只是 forward / prefill / decode kernel 能跑就行，还需要 sampler、KV Cache、graph capture、分布式 runtime 等整条 serving path 稳定。

其中 sampler 是 decode 阶段的最后一步：

```text
logits
  -> softmax / probability
  -> top-k / top-p / min-p filtering
  -> sampling
  -> next_token_id
```

当时 MUSA 上一个关键问题是：

> `torch.multinomial` 在 MUSA 的 no-seed sampling 场景下不稳定。

这里的 **no-seed sampling** 指：

```python
sampling_seed is None
```

也就是普通在线推理场景。用户没有显式指定随机种子，每一步 decode 都从 softmax 后的概率分布中随机采下一个 token。

原始 PyTorch fallback 路径会走类似：

```python
torch.multinomial(probs, num_samples=1)
```

这里说的“不稳定”不是指 sampling 本身有随机性，而是指：

> MUSA 后端上 PyTorch 原生 `torch.multinomial` 在无 seed 的在线 serving 场景中行为不够可靠。

对于 DeepSeek V4 来说，这个问题尤其容易在长 decode、高并发、pressure test 或大 batch serving 中暴露，因为 decode sampling 会被频繁调用。

#### 3.4.2 问题的表象是什么？

这个问题的表象不是“DeepSeek V4 完全跑不起来”，而是更像 **decode 阶段生成异常**。

典型表现包括：

- server 可以正常启动。
- `/health` 正常。
- prefill 阶段通常不直接报错。
- logits 没有明显 NaN / Inf。
- greedy decoding 相对正常。
- 普通在线生成时偶发异常，例如：
  - 输出为空。
  - 提前 EOS。
  - 重复特殊 token。
  - 乱码。
  - 回答明显不符合 prompt。
  - batch 内个别 request 输出异常。
- 高并发、pressure test 或长 decode 场景下更容易暴露。
- 关键特征是：问题主要出现在 `sampling_seed is None` 的普通在线采样路径。

需要注意，这类问题和 DeepSeek V4 FP8 权重加载错误、MoE expert 权重错误、KV cache 错位等问题不同。

如果是模型权重或 MoE 路径错误，通常会表现为：

```text
logits 本身已经异常
greedy 也异常
生成稳定崩坏
```

而 sampling 不稳定更像是：

```text
forward / logits 大体正常
greedy 或 seeded path 相对正常
只有 no-seed 随机采样路径偶发异常
```

所以它属于 **MUSA runtime sampler shared path 修复**。

#### 3.4.3 问题的根源是什么？

这个问题的根源可以总结为：

```text
DeepSeek V4 MUSA serving 的 shared sampler path
在 no-seed sampling 场景下依赖 PyTorch 原生 torch.multinomial
但 MUSA 后端上的该路径在在线推理中不够稳定
```

更具体地说：

```text
模型 forward 正常
logits 正常
probability 正常
top-k / top-p / min-p filtering 正常
最后一步 multinomial sampling 在 MUSA no-seed 下不稳定
```

所以修复点不在 DeepSeek V4 模型结构本身，而在：

```text
SGLang sampler runtime
MUSA backend sampling implementation selection
```

#### 3.4.4 怎么定位的？

定位时按下面顺序排查。

1. **先排除 DeepSeek V4 forward / 权重 / KV cache 问题**

   首先确认不是模型主路径错误：

   - 模型能正常加载。
   - server 能启动。
   - prefill 不报错。
   - decode forward 不报错。
   - logits 没有明显 NaN / Inf。
   - greedy decode，例如 `temperature=0`，表现相对正常。
   - seeded sampling 表现相对稳定。
   - KV cache 没有明显 shape / index / page 错误。
   - MoE expert 权重和 route 没有明显异常。

   如果 greedy 正常，说明：

   ```text
   DeepSeek V4 forward
   attention
   MoE
   KV cache
   logits 计算
   ```

   大概率不是根因。

2. **对比 greedy、seeded sampling 和 no-seed sampling**

   对比三种 decode 模式：

   - greedy，例如 `temperature = 0`。如果 greedy 正常，说明模型 forward 和 logits 基本正常。
   - seeded sampling，例如 `sampling_seed = 42`。如果 seeded sampling 相对正常，说明 probability 计算、top-k / top-p / min-p filtering、sampling 参数处理大概率没问题。
   - no-seed sampling，也就是 `sampling_seed is None`。如果异常集中出现在 no-seed sampling，就能把问题范围缩小到 MUSA no-seed random sampling implementation。

3. **做 sampling kernel A/B test**

   保持以下逻辑不变：

   ```text
   DeepSeek V4 模型
   prefill
   decode forward
   logits
   softmax
   top-k / top-p / min-p filtering
   KV cache
   batch scheduling
   ```

   只切换最后一步 sampling kernel：

   ```text
   A: torch.multinomial
   B: sgl_kernel sampling
   ```

   如果切换到 `sgl_kernel` sampling 后：

   - no-seed 生成异常消失。
   - pressure test 更稳定。
   - batch 内异常 token 减少或消失。
   - server 长时间运行更稳定。

   就可以定位为：

   > MUSA 上 PyTorch 原生 `torch.multinomial` no-seed sampling 路径不可靠。

#### 3.4.5 怎么修复的？

主要文件：

```text
python/sglang/srt/layers/sampler.py
```

核心修改点在：

```python
Sampler._sample_from_probs
```

也就是从 probability 分布中采样 next token 的地方。

##### simple sampling

原始逻辑会走 torch sampling：

```python
batch_next_token_ids = sampling_from_probs_torch(
    probs,
    sampling_seed=sampling_info.sampling_seed,
    positions=positions,
)
```

在 MUSA 且 `sampling_seed is None` 时，改成走 `sgl_kernel` sampling：

```python
if is_musa() and sampling_info.sampling_seed is None:
    batch_next_token_ids = top_p_sampling_from_probs(
        probs.contiguous(),
        1.0,
        check_nan=self.use_nan_detection,
    )
    batch_next_token_ids = batch_next_token_ids.view(-1).to(torch.int32)
else:
    batch_next_token_ids = sampling_from_probs_torch(
        probs,
        sampling_seed=sampling_info.sampling_seed,
        positions=positions,
    )
```

这里：

```python
top_p = 1.0
```

等价于不做 top-p 截断。

也就是说，这里不是改变 sampling 策略，而是：

```text
保持原概率分布不变
只把最终 multinomial sampling 实现
从 torch.multinomial
切换为 sgl_kernel sampling
```

##### complex sampling

如果开启 top-k / top-p / min-p，MUSA no-seed 下按实际需求走不同 `sgl_kernel` sampling path：

```text
need_min_p_sampling -> top_k_renorm_prob / top_p_renorm_prob -> min_p_sampling_from_probs
need_top_k_sampling -> top_k_top_p_sampling_from_probs
otherwise           -> top_p_sampling_from_probs
```

也就是：

1. 如果需要 min-p sampling：

   ```text
   先根据 top-k / top-p 做 renorm
   再调用 min_p_sampling_from_probs
   ```

2. 如果需要 top-k / top-p sampling：

   ```text
   调用 top_k_top_p_sampling_from_probs
   ```

3. 如果只是普通 sampling：

   ```text
   调用 top_p_sampling_from_probs
   top_p = 1.0
   ```

最后统一输出：

```python
batch_next_token_ids = batch_next_token_ids.view(-1).to(torch.int32)
```

保证输出 dtype 和 shape 与原 sampler path 一致。

#### 3.4.6 `sgl_kernel` sampling 底层实现补充

这里的 `sgl_kernel` sampling 不是 Python 里手写的采样逻辑，也不是 Triton / TileLang kernel。它整体是：

```text
Python wrapper
  -> torch.ops.sgl_kernel custom op
    -> C++ extension 注册
      -> CUDA / MUSA 自定义设备 kernel
```

MUSA 侧 Python 封装主要在：

```text
sgl-kernel/python/sgl_kernel/musa.py
```

其中：

```python
top_p_sampling_from_probs(...)
top_k_top_p_sampling_from_probs(...)
min_p_sampling_from_probs(...)
```

会分别调用：

```python
torch.ops.sgl_kernel.top_p_sampling_from_probs.default(...)
torch.ops.sgl_kernel.musa_top_k_top_p_sampling_from_probs.default(...)
torch.ops.sgl_kernel.min_p_sampling_from_probs.default(...)
```

MUSA custom op 注册在：

```text
sgl-kernel/csrc/common_extension_musa.cc
```

其中 `top_k_top_p_sampling_from_probs` 的 MUSA 专门 kernel 在：

```text
sgl-kernel/csrc/musa/top_k_top_p_sampling.mu
```

`.mu` 是 MUSA 设备侧 kernel 文件，可以类比 CUDA 的 `.cu`。所以这条路径本质是 **SGLang 自定义 sampling op + MUSA 设备 kernel**，不是 `torch.multinomial`。

##### 三个 sampling path 分别做什么？

1. `top_p_sampling_from_probs(probs, top_p)`

   做 top-p / nucleus sampling：

   ```text
   按概率从大到小累计
   保留累计概率达到 top_p 的候选 token
   在候选 token 中重新归一化并采样
   ```

   当：

   ```python
   top_p = 1.0
   ```

   时，基本等价于不做 top-p 截断，直接从完整 `probs` 分布中采样。因此 simple sampling 中：

   ```python
   top_p_sampling_from_probs(probs.contiguous(), 1.0)
   ```

   语义上是在替代：

   ```python
   torch.multinomial(probs, num_samples=1)
   ```

   也就是不改变采样策略，只替换底层采样实现。

2. `top_k_top_p_sampling_from_probs(probs, top_k, top_p)`

   做 top-k + top-p 组合采样：

   ```text
   top-k：只允许概率最高的 k 个 token 参与采样
   top-p：只允许 nucleus 累计概率范围内的 token 参与采样
   最终从同时满足 top-k / top-p 约束的 token 中采样
   ```

   MUSA no-seed complex sampling 中使用：

   ```python
   top_k_top_p_sampling_from_probs(
       probs.contiguous(),
       sampling_info.top_ks,
       top_ps,
       filter_apply_order="joint",
   )
   ```

   `joint` 路径会走 MUSA fused kernel：

   ```python
   torch.ops.sgl_kernel.musa_top_k_top_p_sampling_from_probs.default(...)
   ```

3. `min_p_sampling_from_probs(probs, min_p)`

   做 min-p sampling。min-p 不是看累计概率，而是看相对最大概率：

   ```text
   threshold = min_p * max_prob
   保留 prob >= threshold 的 token
   丢掉相对最大概率太小的尾部 token
   再采样
   ```

   如果 min-p 和 top-k / top-p 同时启用，代码会先做：

   ```python
   probs = top_k_renorm_prob(probs, sampling_info.top_ks)
   probs = top_p_renorm_prob(probs, sampling_info.top_ps)
   ```

   然后再调用：

   ```python
   min_p_sampling_from_probs(probs, sampling_info.min_ps)
   ```

   也就是先应用 top-k / top-p 过滤并重新归一化，再执行 min-p sampling。

##### `top_k_top_p_sampling_from_probs` kernel 内部大致怎么做？

直觉上，如果 top-k / top-p sampling 先完整排序 vocab，再构造 mask、renorm、multinomial，在 32K / 128K vocab 下会很重。因此 MUSA `top_k_top_p_sampling` 不是朴素完整排序实现。

它的核心思路是 **fused sampling + block-level scan/reduce**：

```text
每个 batch row 一个 block
block 内线程并行扫描这一行 vocab 概率
用 MUSA Philox / murand 生成随机数 u
用 block reduce / block scan 在候选概率上做 CDF sampling
先采出一个 candidate token
再检查 candidate 是否满足 top-k / top-p 约束
如果不满足，提高概率阈值并重试
满足后输出 sampled_id
```

具体来说，kernel 里会对当前行概率做并行扫描。采到 candidate token 后，设它的概率为：

```text
pivot = probs[sampled_id]
```

然后统计两件事：

```text
count = 有多少 token 的概率 > pivot
value = 这些概率更大的 token 的概率和
```

这两个量分别用于判断 top-k 和 top-p：

```text
count < top_k 说明 candidate 位于 top-k 范围内
value < top_p 说明 candidate 位于 top-p nucleus 范围内
```

因此接受条件可以理解成：

```text
candidate 同时满足 top-k 和 top-p 约束
```

如果 candidate 不满足条件，kernel 会提高概率阈值 `low`，下一轮只从更高概率 token 集合中采样，而不是完整排序整个 vocab。

这种实现的性能取舍是：

```text
优点：
1. 不做完整 sort，避免 O(vocab log vocab) 排序。
2. 不显式 materialize top-k / top-p mask。
3. top-k / top-p 判断和 sampling 融合在一个设备 kernel 中。
4. 减少中间 tensor 写回和 kernel launch。

代价：
1. 仍然需要扫描 vocab，复杂度近似 O(vocab * retry 次数)。
2. vocab 很大时 sampling 仍然可能是可见开销。
3. top_k 很小或 top_p 很窄时，candidate 被拒绝并重试的概率可能上升。
```

所以这不是“零成本”的采样优化，而是一个服务化场景下合理的工程折中：

```text
绕开 MUSA torch.multinomial no-seed 不稳定路径
避免最差的完整 sort/top-k/top-p 实现
用 fused scan/reduce kernel 减少中间开销
```

面试时可以这样表达：

> `sgl_kernel` 的 top-k/top-p sampling 不是先 sort 完整 vocab 的朴素实现。MUSA 侧有 fused `.mu` kernel，每个 batch row 一个 block，block 内通过 reduce / scan 对概率分布做 CDF sampling。采到 candidate 后统计比它概率更大的 token 数量和概率和，分别判断它是否落在 top-k 和 top-p 范围内；如果不满足就提高概率阈值继续采样。这样避免完整排序和中间 mask/renorm tensor，虽然仍然需要扫描 vocab，但比朴素 top-k/top-p + multinomial 更适合 serving，同时绕开了 MUSA no-seed `torch.multinomial` 的不稳定问题。

#### 3.4.7 为什么 seeded sampling 不切换？

这里保留了一个重要设计：

```text
无 seed 的 MUSA sampling：走 sgl_kernel
有 seed 的 deterministic sampling：继续走 torch path
```

原因是 seeded sampling 通常用于：

- 确定性推理。
- 单测。
- benchmark 对齐。
- 复现问题。
- A/B correctness 检查。

如果把 seeded path 也切到 `sgl_kernel`，可能改变原来的 deterministic behavior。

所以这里选择只切换：

```python
is_musa() and sampling_info.sampling_seed is None
```

也就是普通在线服务场景。

可以这样理解：

```text
no-seed path 更关注 serving 稳定性和吞吐
seeded path 更关注 deterministic reproducibility
```

因此：

- no-seed MUSA sampling 用 `sgl_kernel` 绕开不稳定的 `torch.multinomial`。
- seeded sampling 保留 torch path，避免改变复现语义。

#### 3.4.7 修复后有什么影响？

正面影响：

1. **DeepSeek V4 MUSA 在线生成更稳定**

   no-seed 普通 serving path 不再依赖 MUSA 上不稳定的 `torch.multinomial`。

2. **减少偶发异常 token**

   例如：

   ```text
   提前 EOS
   重复特殊 token
   空输出
   乱码
   batch 内个别 row 异常
   ```

3. **pressure test 更稳定**

   高并发、长 decode 场景下 sampling 相关 runtime 异常减少。

4. **不改变 seeded sampling 语义**

   有 seed 的 deterministic path 仍然保留 torch 实现，便于复现和测试。

5. **对 DeepSeek V4 属于 shared runtime 修复**

   虽然代码在 `sampler.py`，不是 `deepseek_v4.py`，但 DeepSeek V4 MUSA serving 会经过这条 sampler runtime path，因此对 DSV4 在线推理稳定性有直接价值。

可能影响或注意点：

1. **sampling kernel 实现发生变化**

   no-seed MUSA 下从 torch path 切到 `sgl_kernel` path，因此需要验证输出分布没有系统性偏差。

2. **需要确认 top-k / top-p / min-p 组合正确**

   complex sampling path 下要验证：

   ```text
   top-k only
   top-p only
   top-k + top-p
   min-p
   top-k/top-p + min-p
   ```

3. **需要确认 batch 维度行为一致**

   尤其是 batch 内不同 row 使用不同 sampling 参数时，要确认输出 shape / dtype / row 对齐正确。

#### 3.4.8 验收方式

建议从 correctness、stability 和 serving behavior 三个层面验收。

1. **correctness**

   验证：

   - `batch_next_token_ids` shape 正确。
   - dtype 为 `torch.int32`。
   - 每个 row 的 token id 在合法 vocab 范围内。
   - top-k / top-p / min-p 后不会采到被过滤掉的 token。
   - `check_nan` 能正常发现异常 probability。
   - batch 内不同 sampling 参数时 row 对齐正确。

2. **stability**

   对比修复前后：

   ```text
   torch.multinomial path
   sgl_kernel sampling path
   ```

   在 MUSA 上进行：

   - 长 decode。
   - 高并发 pressure test。
   - 多 batch size。
   - 多 top-k / top-p / min-p 参数组合。
   - no-seed online serving。

   重点观察：

   ```text
   runtime error 是否消失
   device-side assert 是否消失
   非法 token 是否消失
   提前 EOS / 空输出是否减少
   ```

3. **serving behavior**

   用 DeepSeek V4 MUSA server 测：

   - `/health` 正常。
   - 普通 chat completion 正常。
   - 多轮生成稳定。
   - pressure test 下无 sampling 相关异常。
   - seeded sampling 仍可复现。
   - greedy 行为不受影响。

#### 3.4.9 面试回答模板

> 当时 DeepSeek V4 在 MUSA 上 server 可以启动，prefill 和 decode forward 也能正常跑，但普通在线生成时结果偶发异常。我们先排除了模型权重、MoE、KV cache 和 logits 问题，因为 greedy decode 和 seeded sampling 相对正常，问题主要集中在 `sampling_seed is None` 的 no-seed sampling path。原逻辑在这个路径下会用 PyTorch 的 `torch.multinomial` 从 softmax 后的 probability 里采 token，但 MUSA 后端上这个算子在无 seed 的 serving 场景中不够稳定。
>
> 定位时我们做了 A/B test，保持 DeepSeek V4 forward、logits、top-k/top-p/min-p 处理逻辑不变，只把 MUSA no-seed 下的最终 sampling kernel 从 `torch.multinomial` 切到 `sgl_kernel` 的 `top_p_sampling_from_probs`、`top_k_top_p_sampling_from_probs` 或 `min_p_sampling_from_probs`。切换后生成异常和 pressure test 中的 sampling 问题消失，所以确认根因在 MUSA 上 PyTorch 原生 no-seed sampling path。
>
> 修复时我们只切换 `is_musa() and sampling_seed is None` 的路径；有 seed 的 deterministic sampling 继续走 torch path，因为它更强调复现语义，不能随便改变随机实现。这个修复本身在 shared `sampler.py`，不是 DeepSeek 模型文件里的专属逻辑，但 DeepSeek V4 MUSA serving 会走这条 sampler runtime path，所以它属于 DeepSeek V4 MUSA 在线推理稳定性修复。

### 3.5 关键 MUSA 修复细节

本节按统一维度解析每个 MUSA 相关修复：

```text
1. 问题是什么？
2. 问题的表象是什么？
3. 问题的根源是什么？
4. 怎么定位的？
5. 怎么修复的？
6. 修复后有什么影响？
```

#### 3.5.1 `5fcb7c439` — DeepSeek V4 MUSA fallback blockers

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

#### 3.5.2 `ef4469cb5` — DeepSeek V4 MUSA runtime paths

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

#### 3.5.3 `3cceb08b9` — TileLang JIT name collision 修复

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

#### 3.5.4 `81fc824a2` — MUSA prefill kernel `block_id` re-bind 修复

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

#### 3.5.5 `b9b202c28` — DSV4 sparse prefill under CP round-robin

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

#### 3.5.6 `5a15cde85` — DSV4 CP bf16 KV contiguous 修复

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

### 3.6 MUSA 修复总结

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



### 3.7 面试版逐项解析：runtime hardening / 修复类

#### 3.7.1 MUSA no-seed sampling 修复

**1. 简介**

这是 decode runtime 的最后一环。DeepSeek V4 MUSA 在线服务时，`sampling_seed is None` 的普通随机采样不再使用 MUSA 上不稳定的 `torch.multinomial`，而是切到 `sgl_kernel` sampling，例如 `top_p_sampling_from_probs`、`top_k_top_p_sampling_from_probs`、`min_p_sampling_from_probs`。seeded sampling 继续保留 torch path，避免改变 deterministic 复现语义。

**2. 当时表面现象是什么？**

server 能启动，prefill / decode forward 通常也能跑，logits 没有明显 NaN/Inf，但普通在线生成偶发异常：空输出、提前 EOS、重复特殊 token、乱码、batch 内个别 row 异常，或者 pressure test / 长 decode 时出现 sampling 相关 runtime 问题。

**3. 根源是什么？**

根源不在 DeepSeek V4 模型结构，而在 shared sampler runtime：MUSA 后端上 PyTorch 原生 `torch.multinomial` 的 no-seed 路径在在线 serving 场景中不够稳定。模型 forward、logits、top-k/top-p/min-p 过滤大体正常，最后一步 multinomial sampling 出问题。

**4. 怎么定位到这个原因？**

先排除权重、MoE、KV cache、logits 问题：greedy decode 和 seeded sampling 相对正常。然后做 A/B test，保持模型、logits、过滤逻辑不变，只切换最终 sampling kernel：`torch.multinomial` vs `sgl_kernel sampling`。切到 `sgl_kernel` 后异常消失或明显减少，就能定位到 MUSA no-seed sampling path。

**5. 怎么修复的？**

在 `Sampler._sample_from_probs` 中增加 guard：

```python
if is_musa() and sampling_info.sampling_seed is None:
    # use sgl_kernel sampling
else:
    # keep torch sampling
```

simple sampling 用 `top_p_sampling_from_probs(probs.contiguous(), 1.0)`；complex sampling 根据 top-k/top-p/min-p 组合走对应 `sgl_kernel` path；最后统一 `view(-1).to(torch.int32)`。

**6. 修复后的现象是什么？**

no-seed 在线生成更稳定，pressure test 中 sampling 相关异常减少；seeded sampling 和 greedy 行为保持原语义。

**7. 改善了什么？**

改善的是 DeepSeek V4 MUSA decode runtime 稳定性，尤其是普通在线服务场景的生成质量和长 decode 稳定性。

#### 3.7.2 MUSA fallback blockers

**1. 简介**

这项修复的目标是防止 MUSA 路径 silent fallback 到 CUDA-only、torch fallback 或 graph capture 不兼容路径。它不是直接提速，而是保证 DSV4 MUSA serving 不会“表面跑 MUSA，实际跑错路径”。

**2. 当时表面现象是什么？**

可能出现 CUDA JIT / CUDA module 相关错误，也可能程序能跑但性能明显不对：trace 里没有预期 MUSA kernel、prefill latency 或 TPOT 异常、graph capture 失败、operator benchmark 不达标。

**3. 根源是什么？**

原 runtime 对 fast path miss 处理太宽松：MUSA tensor 可能进入 CUDA-only HiSparse helper；TileLang miss 后可能 fallback 到 torch；compress prefill miss 后可能静默走慢路径。这会隐藏真实问题。

**4. 怎么定位到这个原因？**

通过 benchmark / trace 发现没有命中预期 MUSA kernel，或者 graph capture 期间出现 fallback。进一步检查可疑函数入参 device，发现 MUSA tensor 进入 CUDA-only helper，或 TileLang miss 后走 torch path。

**5. 怎么修复的？**

采用 fail-closed 策略：MUSA tensor 进入 `hisparse_offload_to_host` 直接 raise；MUSA graph capture 下 TileLang miss 不允许 torch fallback；compress prefill miss 可以通过 `SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_RAISE_PREFILL_MISS=1` 强制暴露。

**6. 修复后的现象是什么？**

fast path miss 不再被隐藏，错误更早暴露；benchmark 失败时更容易知道是没有命中 MUSA kernel，而不是悄悄跑慢路径。

**7. 改善了什么？**

改善的是生产稳定性、性能可验收性和问题定位效率。代价是容错变少，调试时需要显式打开 fallback。

#### 3.7.3 MUSA runtime paths / MCCL device binding

**1. 简介**

这是多卡 MUSA runtime 修复。核心是 `init_process_group(backend="mccl")` 时显式绑定 `torch.device("musa", local_rank)`，并补齐 DeepSeek V4 MUSA ops forwarding exports。

**2. 当时表面现象是什么？**

多卡启动可能出现 MCCL 初始化失败、device mismatch、collective hang、rank/device 对应错乱；也可能出现 MUSA op wrapper 找不到、ImportError / AttributeError。

**3. 根源是什么？**

MCCL 初始化没有显式 device_id，容易依赖默认 device 或错误上下文；同时 MUSA ops `__init__.py` 导出不完整，上层模型 patch 找不到对应 wrapper。

**4. 怎么定位到这个原因？**

从多卡启动日志检查 backend、local_rank、current_device，发现 process group 初始化没有绑定 local MUSA device；从 import stack 发现 kernel wrapper 存在但没有被 ops package 导出。

**5. 怎么修复的？**

MCCL 初始化时传入 `device_id=torch.device("musa", local_rank)`；补齐 `deepseek_v4_musa/ops/__init__.py` 中的 MUSA op exports。

**6. 修复后的现象是什么？**

多卡初始化更稳定，rank/device 绑定明确；上层 DeepSeek V4 runtime 能找到 MUSA op wrapper。

**7. 改善了什么？**

改善多卡 serving 基础稳定性，降低 collective hang、device mismatch 和 ops forwarding 缺失风险。

#### 3.7.4 TileLang JIT name collision 修复

**1. 简介**

这是典型 TileLang-specific 修复。C4 decode page kernel 中，不同 `extra_data_cols` 影响 layout，但原来使用同一个 JIT name，导致不同 layout 可能复用错误编译缓存。

**2. 当时表面现象是什么？**

单独跑某个 page mode 可能正常，但 paged mode 和 page4 mode 交替运行时出现 cache 写入错位、decode 输出异常、偶发 correctness mismatch，而且结果可能和运行顺序相关。

**3. 根源是什么？**

TileLang JIT cache key / name 没包含影响 kernel layout 的 `extra_data_cols`。`extra_data_cols=1` 和 `extra_data_cols=4` 复用了同一个 JIT cache entry。

**4. 怎么定位到这个原因？**

观察到错误和运行顺序相关，怀疑 JIT cache collision。检查 TileLang JIT name，发现 name 固定为类似 `dsv4_c4_decode_page4_t{threads}`，没有包含 `extra_data_cols`。

**5. 怎么修复的？**

把 `extra_data_cols` 加入 JIT name：

```python
f"dsv4_c4_decode_page{extra_data_cols}_t{threads}"
```

这样 page1 和 page4 会生成不同 JIT 编译产物。

**6. 修复后的现象是什么？**

不同 page layout 不再误用同一 kernel，顺序相关的 correctness bug 消失。

**7. 改善了什么？**

改善 TileLang kernel cache 的正确性和 decode/cache kernel 稳定性。代价是首次运行可能多编译几份 kernel。

#### 3.7.5 prefill kernel `block_id` re-bind + paged metadata 修复

**1. 简介**

这个修复包含两部分：一是 TileLang kernel 中 `block_id` re-bind 导致 codegen/语义问题；二是 DSV4 compressor 启用 `is_paged=True`，让 paged compression metadata 与下游 kernel 一致。

**2. 当时表面现象是什么？**

可能出现 TileLang JIT compile error、immutable value re-bind error、prefill page write 写错 block/page、compressed cache layout 不匹配。

**3. 根源是什么？**

在 TileLang kernel 中直接写：

```python
block_id = indices[batch_id]
block_id = extra_data[batch_id, 2]
```

会触发 TileLang 对 immutable expression / scalar re-bind 的限制。另一方面，上层 `Compressor.compress_fused()` 仍用 `is_paged=False`，与 paged kernel 期望不一致。

**4. 怎么定位到这个原因？**

从 TileLang 编译错误或 page write 异常定位到 prefill page write kernel；检查代码发现 `block_id` re-bind。再追踪 metadata 构建，发现 compressor 没启用 paged metadata。

**5. 怎么修复的？**

用 local mutable buffer 替代 scalar re-bind：

```python
block_id = T.alloc_local((1,), dtype=T.int32)
block_id[0] = indices[batch_id]
...
block_id[0] = extra_data[batch_id, 2]
```

同时在 compressor 中设置 `is_paged=True`。

**6. 修复后的现象是什么？**

TileLang kernel 编译和语义更稳定，prefill page write 能写到正确 block，paged metadata 与 kernel 对齐。

**7. 改善了什么？**

改善 MUSA prefill compress / paged cache 路径正确性，为后续 decode paged kernel 支持打基础。

#### 3.7.6 sparse prefill CP round-robin 修复

**1. 简介**

这是 DSV4 sparse prefill 在 context parallel round-robin split 下的 metadata 正确性修复。它不是 TileLang 问题，而是 local/global query metadata 问题。

**2. 当时表面现象是什么？**

CP round-robin 下 sparse prefill 可能输出错误、combined indices 错误、workspace index 错位、shape mismatch、index out of bounds，或者不同 CP 配置下结果不一致。

**3. 根源是什么？**

CP round-robin 下每个 rank 看到的是 local q，例如 rank0 拿 q0/q2/q4，rank1 拿 q1/q3/q5；但 sparse prefill cache 构建仍然使用 global query_start / seq_lens，导致 metadata 错位。

**4. 怎么定位到这个原因？**

问题只在 CP round-robin 出现，普通 prefill 或非 round-robin CP 正常。对比 local q length 与 global q length，发现 local q metadata 没有重建。

**5. 怎么修复的？**

在 `_build_sparse_prefill_cache_inputs()` 中识别 `is_context_parallel_extend()` 和 `is_nsa_prefill_cp_round_robin_split()`，为当前 rank 重建 local `query_start_loc`、local extend seq lens、req_pool_indices、seq_lens。

**6. 修复后的现象是什么？**

每个 rank 使用自己的 local query metadata，FlashMLA sparse workspace index 与 local q 对齐。

**7. 改善了什么？**

改善 MUSA/DSV4 长上下文 CP serving 中 sparse prefill 的正确性和稳定性。

#### 3.7.7 CP bf16 KV contiguous 修复

**1. 简介**

这是 DSV4 CP shared path 的 tensor layout 修复。在 `_compute_kv_bf16()` 中，调用 fused norm rope 前对 `kv` 做 `contiguous()`。

**2. 当时表面现象是什么？**

CP prefill 可能报错，fused norm rope 输出异常，all-gather 后 KV 数据错乱，或者某些 rank attention 结果不一致。

**3. 根源是什么？**

CP 切分、重排或 all-gather 前后，`kv` 可能是非 contiguous tensor；但后续 MUSA fused norm rope / all-gather path 可能隐含 contiguous layout 假设。

**4. 怎么定位到这个原因？**

问题集中在 CP path，非 CP 正常。检查 `_compute_kv_bf16()`，发现进入 fused kernel 前未保证 `kv.is_contiguous()`；打印 stride 后确认某些 case 非连续。

**5. 怎么修复的？**

在 fused norm rope 前增加：

```python
kv = kv.contiguous()
```

**6. 修复后的现象是什么？**

后续 fused norm rope 和 all-gather 都拿到连续布局，stride/layout 相关错误减少。

**7. 改善了什么？**

改善 CP prefill / CP serving 中 KV layout 稳定性。代价是某些情况下可能多一次拷贝。


### 3.8 面试深挖版：runtime hardening 常见追问

> 本节补充 3.7 的深挖问题，重点回答“为什么会怀疑这个点、如何排除其他可能、为什么这样修、有什么副作用、如何验收”。

#### 3.8.1 MUSA no-seed sampling 修复

**为什么会怀疑这个点？**

因为表象集中在生成阶段，而不是模型完全无法 forward：server、prefill、decode forward、logits 通常正常，但 no-seed 在线生成偶发异常。只要 greedy 或 seeded sampling 相对稳定，就说明模型权重、MoE、KV cache、logits 主路径大概率不是根因，问题更可能在最后一步 sampling。

**怎么排除其他可能？**

- 用 greedy / `temperature=0` 排除大部分 forward 与 logits 问题。
- 用 seeded sampling 排除 top-k/top-p/min-p 参数处理和 probability 计算问题。
- 检查 logits 是否有 NaN/Inf，排除数值崩坏。
- 保持 logits 和过滤逻辑不变，只替换最终 sampling kernel 做 A/B test。

**为什么选择这个方案？**

只在 `is_musa() and sampling_seed is None` 下切到 `sgl_kernel`，是因为问题集中在 MUSA no-seed serving path；seeded path 用于复现和测试，保留 torch path 可以避免改变 deterministic 语义。这样修复范围最小，风险也最可控。

**有什么副作用？**

no-seed sampling 的底层随机实现发生变化，理论上需要验证分布没有系统性偏差；另外要覆盖 top-k/top-p/min-p 多种组合，避免 batch 内不同 sampling 参数时 row 对齐错误。

**怎么验收？**

- correctness：token id 合法、不采被过滤 token、shape/dtype 正确。
- stability：长 decode、pressure test、不同 batch size 下无异常 token / device assert。
- regression：seeded sampling 可复现，greedy 不受影响。

**常见追问怎么答？**

> Q：为什么不把 seeded sampling 也切到 `sgl_kernel`？  
> A：seeded sampling 更强调 deterministic reproducibility，常用于测试和问题复现。当前问题集中在 no-seed 在线 serving，因此只切 no-seed，避免扩大语义变化范围。

#### 3.8.2 MUSA fallback blockers

**为什么会怀疑这个点？**

因为很多性能异常不是直接 crash，而是 trace 中没有预期 MUSA kernel、benchmark latency 异常、graph capture 失败。这类现象通常说明 kernel dispatch 没按预期命中 fast path，可能发生 silent fallback。

**怎么排除其他可能？**

- 看 trace / dispatch log，确认实际 kernel 名称。
- 打开 strict miss env，观察是否立即暴露 fast path miss。
- 检查 graph capture 期间是否调用 torch fallback。
- 检查 MUSA tensor 是否进入 CUDA-only helper。

**为什么选择 fail-closed？**

生产 MUSA serving 中 silent fallback 比直接失败更危险：它会隐藏性能回退，甚至在 graph capture 下埋入不兼容路径。fail-closed 可以让问题在上线前暴露。

**有什么副作用？**

调试容错下降，之前靠 fallback 跑通的 case 现在可能直接报错。因此需要保留显式 debug 开关，例如 `SGLANG_MUSA_ALLOW_TORCH_FALLBACK=1`。

**怎么验收？**

- strict mode 下 fast path miss 可复现地报错。
- graph capture 期间没有 torch fallback。
- trace 中能看到预期 MUSA kernel。
- 关闭 debug fallback 后生产路径不走 CUDA-only / torch fallback。

**常见追问怎么答？**

> Q：为什么不保留 fallback 提高鲁棒性？  
> A：开发调试可以保留显式 fallback，但生产 serving 更需要性能和 graph capture 可控。silent fallback 会让性能问题很晚才暴露，所以生产默认 fail-closed 更合理。

#### 3.8.3 MUSA runtime paths / MCCL device binding

**为什么会怀疑这个点？**

多卡问题常表现为 hang、device mismatch 或 rank 间行为不一致。如果单卡正常、多卡异常，就要优先检查 distributed 初始化、local rank、device binding 和 backend 参数。

**怎么排除其他可能？**

- 单卡运行确认模型和 kernel 本身可用。
- 打印 rank/local_rank/current_device，确认是否一一对应。
- 检查 `init_process_group` 参数是否包含 MUSA device。
- 如果是 import/attribute error，则检查 op wrapper 是否实现但未导出。

**为什么选择显式 `device_id`？**

MCCL 后端需要明确知道当前 rank 绑定哪个 MUSA device。依赖默认 device 容易受启动脚本、环境变量、上下文切换影响。

**有什么副作用？**

部署脚本必须正确设置 `local_rank`。如果 rank/device 映射本来就错，修复后会更早暴露，而不是以不确定方式运行。

**怎么验收？**

- 多卡初始化不 hang。
- 每个 rank current device 与 local_rank 对齐。
- collective 通信正常。
- DeepSeek V4 MUSA op import / forwarding 不再报 missing symbol。

**常见追问怎么答？**

> Q：为什么这个属于 DeepSeek V4 修复？  
> A：它是 shared MUSA runtime 修复，但 DSV4 MUSA 多卡 serving 必须经过这条路径，所以属于 DSV4 MUSA bring-up 的基础稳定性工作。

#### 3.8.4 TileLang JIT name collision

**为什么会怀疑这个点？**

因为错误可能和运行顺序相关：单独跑 page1/page4 正常，混合或切换顺序后异常。这类“先编译谁影响后续结果”的现象非常像 JIT cache key collision。

**怎么排除其他可能？**

- 固定输入分别单独跑不同 page mode。
- 改变执行顺序观察是否影响结果。
- 检查 kernel name 是否包含所有影响 layout 的参数。
- 清理 JIT cache 后复现，看是否和编译缓存有关。

**为什么选择把 `extra_data_cols` 放入 JIT name？**

因为 `extra_data_cols` 直接影响 kernel 对 extra_data 的访问 layout，是 codegen 语义的一部分。它必须进入 cache key，否则不同 layout 会复用错误编译产物。

**有什么副作用？**

JIT 产物数量增加，首次编译成本略高；但这是正确性必要成本。

**怎么验收？**

- page1/page4 单独和交替运行都正确。
- 清 cache / 不清 cache 结果一致。
- trace 中 JIT name 能区分不同 `extra_data_cols`。

**常见追问怎么答？**

> Q：这是不是 TileLang 独有问题？  
> A：这个具体修复是 TileLang-specific，因为它依赖 TileLang JIT name/cache key。但类似“cache key 必须包含 layout 参数”的原则对所有 JIT/codegen 框架都成立。

#### 3.8.5 `block_id` re-bind + paged metadata

**为什么会怀疑这个点？**

因为错误出现在 TileLang prefill page write kernel，且涉及 block/page 选择。看到 kernel 里同一个 scalar `block_id` 被重新赋值，就会怀疑 TileLang IR/codegen 对 immutable expression 的限制。

**怎么排除其他可能？**

- 检查 indices / extra_data 输入是否正确。
- 对比 non-paged 与 paged metadata。
- 简化 kernel，只保留 block_id 选择逻辑复现编译/结果问题。
- 检查上层 compressor 的 `is_paged` 是否和下层 kernel 假设一致。

**为什么选择 local buffer？**

TileLang 中 local buffer mutation 比 scalar re-bind 更符合 kernel IR 语义。用 `T.alloc_local` 后，`block_id[0]` 是可变存储，避免重新绑定表达式。

**有什么副作用？**

`is_paged=True` 改变 metadata layout，要求所有消费方都支持 paged 语义；local buffer 也会引入极小的局部存储开销。

**怎么验收？**

- TileLang kernel 能稳定编译。
- prefill page write block/page 正确。
- paged metadata 下 prefill 写入与 decode 读取一致。
- 与后续 decode paged kernel 联合测试。

**常见追问怎么答？**

> Q：为什么普通 Python 变量赋值在这里不行？  
> A：TileLang 不是普通 Python 执行，它会生成 kernel IR。某些 scalar expression 被视为不可变表达式，直接 re-bind 可能造成 codegen 语义问题，所以要用 local mutable buffer。

#### 3.8.6 CP round-robin sparse prefill

**为什么会怀疑这个点？**

问题只在 CP round-robin 下出现，而普通 prefill、非 CP 或非 round-robin CP 正常。这个特征强烈指向 local/global query metadata 差异。

**怎么排除其他可能？**

- 单卡和非 round-robin CP 对比，排除 FlashMLA kernel 本身问题。
- 打印 local q length 与 global q length。
- 检查 local `query_start_loc` 是否按当前 rank 重建。
- 验证 combined indices 是否指向当前 local workspace。

**为什么选择重建 local metadata？**

round-robin 后每个 rank 拿到的 q 不是全局连续切片，不能简单复用 global query_start。只有按 local q 重建 seq lens、req_pool_indices、query_start，workspace/index 才一致。

**有什么副作用？**

只在 CP round-robin 条件下增加一点 metadata 重建开销；相比正确性收益可以接受。

**怎么验收？**

- CP round-robin 下 sparse prefill 不再 index out of bounds。
- 不同 CP 配置输出一致或误差可接受。
- combined indices / valid lens 与 local q 对齐。

**常见追问怎么答？**

> Q：为什么 global metadata 不能直接 slice？  
> A：round-robin 不是连续切片，而是交错分配 query。local rank 的 query_start 和 seq lens 语义已经变化，直接 slice global metadata 会错位。

#### 3.8.7 CP bf16 KV contiguous

**为什么会怀疑这个点？**

问题集中在 CP path，而 CP 会引入切分、转置、all-gather 等 layout 变化。如果 fused kernel 输出异常但输入值本身合理，就要检查 stride/contiguous。

**怎么排除其他可能？**

- 比较 CP 与非 CP。
- 打印 `kv.shape`、`kv.stride()`、`kv.is_contiguous()`。
- 在 fused norm rope 前手动 `.contiguous()` 做 A/B。
- 若 A/B 后错误消失，则定位为 layout 问题。

**为什么选择 `.contiguous()`？**

这是最小风险修复：不改变数值语义，只保证后续 fused kernel 的内存布局假设成立。

**有什么副作用？**

如果输入本来非 contiguous，会多一次拷贝；但如果本来 contiguous，基本无额外成本。

**怎么验收？**

- CP prefill 不再因 stride/layout 报错。
- all-gather 后 KV 正确。
- fused norm rope 输出与 baseline 对齐。

**常见追问怎么答？**

> Q：为什么不改 kernel 支持任意 stride？  
> A：可以，但成本和风险更高。当前问题出在 CP shared path，先用 `.contiguous()` 保证 layout，是最小改动、最稳定的修复。



## 4. 第二阶段：decode / TPOT 路径优化与保护

这一阶段补齐的是 DeepSeek V4 在 MUSA 上的 **decode / TPOT** 路径。前面的 prefill 专项解决的是大 token / large-M 吞吐问题；decode 则是 small-M / per-token latency 场景，核心指标是 TPOT、graph replay latency、cache lookup/store、sampling 和小 shape kernel 调度。二者优化方法不同，因此不能把 prefill 的大 tile / 大并行 kernel 直接套到 decode 上。

从仓库分支和提交历史看，DSV4 MUSA bring-up 中确实存在一组 decode 相关优化和修复，主要集中在：

```text
MUSA DeepSeekV4 decode fast paths
TileLang decode attention
RoPE decode path
FlashMLA / indexer cache store decode path
MHC decode split correctness
paged decode kernel
sampling runtime
```

相关提交线索包括：

| Commit | 标题 | 作用 |
| --- | --- | --- |
| `17be8f1bf` | `Add MUSA DeepSeekV4 decode fast paths` | 新增 DSV4 MUSA decode fast path，覆盖 cache / HC head / MHC benchmark / 模型接入等。 |
| `8094ab938` | `Add queue4 prefix-tail decode path` | 为 TileLang decode attention 增加 queue4 prefix-tail 路径。 |
| `018145486` | `Support page size 32 for TileLang decode attention` | TileLang decode attention 支持 page size 32。 |
| `4ef9320d9` | `Support head size 256 in TileLang decode attention` | TileLang decode attention 支持 head size 256。 |
| `d71563167` | `Add production TileLang decode dispatch policy` | 增加生产可用的 TileLang decode dispatch policy。 |
| `b5547f17c` | `Remove TileLang fused decode disable switch` | 移除 fused decode 禁用开关，使生产策略更直接。 |
| `9f6d9ce9c` | `Simplify TileLang decode queue4 segment policy` | 简化 queue4 segment policy，降低 dispatch 复杂度。 |
| `85fd9cc2b` / `ec3ab4cbd` | `Optimize DeepSeek V4 MUSA RoPE decode path` | 优化 DSV4 MUSA RoPE decode 小 shape 高频路径。 |
| `392179e92` / `58d682390` | `Optimize DeepSeek V4 MUSA indexer cache store` | 优化 indexer cache store，包含 decode/prefill 分流相关能力。 |
| `be50bfc0d` / `6ce8e1357` | `Optimize MUSA FlashMLA cache store paths` / `Optimize MUSA FlashMLA cache pack store` | 优化 FlashMLA cache pack/store，包含 decode x4 / vec2 / fp32 与 prefill 路径。 |
| `b64efe165` | `Fix MHC pre big fuse decode split correctness` | 修复 MHC pre big fuse 在 decode split 下的正确性。 |
| `eb255ea13` | `feat(dsv4_musa): add decode paged kernel for is_paged=True support` | 为 `is_paged=True` 场景补齐 decode paged kernel。 |

> 注：这些提交来自仓库分支观察和提交历史静态梳理；如果要写入最终性能结论，还需要在对应 MUSA 环境上跑 decode/cache/MHC benchmark 和 E2E TPOT。

### 4.1 为什么 decode 需要单独一章

decode 和 prefill 的 workload 差异很大：

| 阶段 | 典型 shape | 主要指标 | 典型瓶颈 | 优化方向 |
| --- | --- | --- | --- | --- |
| prefill | 8K / 16K / 32K token，大 M | TTFT、prefill throughput | memory bandwidth、large-M GEMM、workspace 构建、cache write | fusion、vector write、compact layout、workspace reuse、large tile |
| decode | 1 token 或少量 token，small-M / tiny-M | TPOT、per-token latency | launch overhead、cache lookup、decode attention、small-M GEMM、sampling、graph replay | persistent/queue policy、decode attention fast path、小 shape kernel、graph-friendly dispatch |

因此这轮故事不能简单说“只优化 prefill”。更准确的说法是：

```text
prefill 侧：解决 large-M 吞吐与 memory-bound 热点
decode 侧：补齐 small-M fast path、dispatch policy、cache store 和 sampling runtime
```

### 4.2 DSV4 MUSA decode fast paths

`17be8f1bf Add MUSA DeepSeekV4 decode fast paths` 是 decode 主线里最关键的提交之一。

涉及文件包括：

- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/cache_kernels.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/hc_head_kernels.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/cache_ops.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/hc_head_ops.py`
- `python/sglang/srt/models/deepseek_v4.py`
- `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_cache_benchmarks.py`
- `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_mhc_benchmarks.py`

可以把它理解为 DSV4 MUSA decode 的底座：

```text
decode token
  -> cache / hc_head / MHC 相关 MUSA fast path
  -> model path 接入
  -> cache/MHC benchmark 验收
```

它解决的是 decode 阶段不能只依赖 prefill kernel 或通用 fallback 的问题。decode small-M 对 TPOT 很敏感，需要更轻量、更低调度开销的 MUSA fast path。

### 4.3 TileLang decode attention：queue4、prefix-tail、page32、head256

TileLang decode attention 相关提交包括：

- `8094ab938` — queue4 prefix-tail decode path。
- `018145486` — page size 32 支持。
- `4ef9320d9` — head size 256 支持。
- `d71563167` — production decode dispatch policy。
- `b5547f17c` — 移除 fused decode disable switch。
- `9f6d9ce9c` — 简化 queue4 segment policy。

对应文件主要是：

- `python/sglang/srt/layers/attention/tilelang_unified_attention_v2.py`
- `python/sglang/srt/layers/attention/tilelang_unified_attention_v2_queue4_prefix_tail.py`
- `python/sglang/srt/layers/attention/tilelang_decode_runtime_dispatch_policy.md`
- `test/srt/test_tilelang_unified_decode_page32.py`

这组优化的核心是 decode attention 的生产化：

```text
page_size / head_size 支持更完整
queue4 / prefix-tail 处理更适合 decode serving
runtime dispatch policy 更明确
不再依赖手动 disable switch
```

decode attention 的难点不是大 M 吞吐，而是大量小步重复调用，因此 dispatch policy、page layout、prefix-tail segment 这些“看起来不像大 kernel”的细节会直接影响 TPOT 和稳定性。

### 4.4 RoPE decode path 优化

相关提交：

- `85fd9cc2b` — `Optimize DeepSeek V4 MUSA RoPE decode path`
- `ec3ab4cbd` — `Optimize DeepSeek V4 MUSA RoPE decode path`

关键文件：

- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/norm_rope_ops.py`

RoPE 在 decode 中每步都会调用，单次开销不大，但 TPOT 对尾部 latency 很敏感。decode RoPE 优化的重点不是像 prefill 那样追求大规模吞吐，而是：

```text
小 shape 快速命中 MUSA fast path
减少不必要 materialization
减少 fallback / dispatch overhead
保证 graph capture 友好
```

这类优化和 6.x 里的 norm/rope prefill/group benchmark 有交集，但 decode 侧关注的是 small-M 高频调用。

### 4.5 cache store decode path：decode_x4 / vec2 / fp32 与 prefill 分流

cache store 不是纯 prefill 优化，它同时有 decode 与 prefill 两套策略。

相关提交：

- `392179e92` / `58d682390` — indexer cache store。
- `be50bfc0d` — FlashMLA cache store paths。
- `6ce8e1357` — FlashMLA cache pack store。
- `49eb8067d` — Harden DeepSeek V4 MUSA cache store paths。

`cache_ops.py` 中能看到 decode 相关实现名称：

```text
decode_x4
decode_x4_fp32
decode_x4_i32addr
decode_x4_flat_i32addr
decode_vec2_nvstyle
decode_vec2_i32addr
```

对应策略是：

```text
decode small-M  -> decode_x4 / vec2 / fp32 等低延迟路径
prefill large-M -> tile-parallel / subwarp16 / x8 等高吞吐路径
```

这也是为什么文档后面要把 cache store 从“纯 prefill 优化”改成“decode/prefill 双路径分流”：

```text
prefill 需要写带宽
decode 需要低 TPOT
两者不能使用同一套 kernel 策略
```

### 4.6 MHC decode split correctness

相关提交：

- `b64efe165` — `Fix MHC pre big fuse decode split correctness`

关键文件：

- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/mhc_kernels.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/mhc_ops.py`

MHC pre big fuse 在 prefill 和 decode 中的 split 策略不同。decode small-M 下，如果沿用不合适的 split 或 fuse 策略，可能出现 correctness 问题或 TPOT 回退。

该修复说明 decode 不是只需要性能优化，还需要专门的 correctness guard：

```text
prefill/mid-prefill 可以使用较大 hidden_block / split 策略
decode/tiny-decode 需要更保守的小 shape 配置
```

### 4.7 paged decode kernel 与 `is_paged=True`

相关提交：

- `eb255ea13` — `feat(dsv4_musa): add decode paged kernel for is_paged=True support`

这个提交可以和前面 `81fc824a2` 串起来理解：

```text
prefill compress / cache path 启用 is_paged=True
=> decode 也必须有对应 paged kernel 支持
```

否则会出现：

```text
prefill 写入的是 paged metadata / paged cache
decode 读取或更新时没有对应 paged kernel
```

这类问题不是单纯性能问题，而是 serving path 的一致性问题：prefill 写入什么 layout，decode 就必须按同样 layout 消费。

### 4.8 sampling 是 decode runtime 的最后一环

文档第 3.4 节已经写了 MUSA no-seed sampling 修复。放在完整 decode 故事里，它对应的是：

```text
decode logits
  -> probability
  -> top-k / top-p / min-p
  -> sampling
  -> next token
```

也就是说 sampling 修复不是 prefill，而是 decode runtime 稳定性修复：

```text
MUSA no-seed 下不用不稳定的 torch.multinomial
改用 sgl_kernel sampling
seeded sampling 保留 torch path 保证 deterministic behavior
```

### 4.9 小结：为什么现在不是“只优化 prefill”

观察整个仓库分支和提交后，更准确的结论是：

```text
prefill 是这轮性能文档最早梳理出来的主线
但 DeepSeek V4 MUSA bring-up 实际上同时包含 decode 路径
```

其中：

- prefill 侧解决 large-M / TTFT / memory-bound / MoE / FlashMLA sparse attention。
- decode 侧解决 small-M / TPOT / TileLang decode attention / RoPE decode / cache store / MHC split / paged kernel / sampling。
- shared runtime 侧解决 fallback、MCCL、CP metadata、KV contiguous、JIT cache 等稳定性。

所以后续文档标题和故事应统一为：

> DeepSeek V4 FP8 MUSA Serving Bring-up：Prefill / Decode 优化与 Runtime 修复。


### 4.10 面试版逐项解析：decode / TPOT 优化类

#### 4.10.1 DSV4 MUSA decode fast paths

**1. 简介**

这是 DSV4 MUSA decode 的底座优化，代表提交是 `17be8f1bf Add MUSA DeepSeekV4 decode fast paths`。它把 decode 阶段涉及的 cache、HC head、MHC 等路径接入 MUSA fast path，并配套 cache / MHC benchmark。

**2. 当时表面现象是什么？**

decode 不是完全不能跑，而是可能依赖通用 fallback 或 prefill-oriented path，导致 TPOT 偏高、trace 中缺少预期 MUSA decode kernel、cache/MHC 小 shape benchmark 不稳定。

**3. 根源是什么？**

DeepSeek V4 decode 是 small-M / per-token 场景，不能直接复用 prefill 大 M kernel。cache、HC head、MHC 等小 shape 高频路径需要专门 MUSA decode fast path。

**4. 怎么定位到这个原因？**

通过 decode TPOT、cache/MHC benchmark、kernel trace 观察 decode 阶段实际走的 kernel；如果看到 fallback 或 large-M path，就说明 decode fast path 缺失。

**5. 怎么优化的？**

新增/接入 `cache_kernels.py`、`hc_head_kernels.py`、`cache_ops.py`、`hc_head_ops.py` 等 MUSA decode path，并在 `deepseek_v4.py` 中接入模型路径，同时增加 `test_cache_benchmarks.py`、`test_mhc_benchmarks.py`。

**6. 优化后的现象是什么？**

decode cache / HC head / MHC 等路径能命中 MUSA fast path，benchmark 能单独观测小 shape decode 行为。

**7. 改善了什么？**

改善 decode TPOT 基础路径，为后续 attention、cache store、RoPE、MHC split 等优化打底。

#### 4.10.2 TileLang decode attention：queue4 / prefix-tail / page32 / head256

**1. 简介**

这组优化补齐 TileLang decode attention 的生产能力，包括 queue4 prefix-tail、page size 32、head size 256、production dispatch policy、queue4 segment policy 简化等。

**2. 当时表面现象是什么？**

某些 decode serving shape 不能命中 fused TileLang decode attention；page size 或 head size 不支持时会 fallback；queue/prefix-tail 处理不稳定会导致 TPOT 波动或 dispatch 复杂。

**3. 根源是什么？**

TileLang decode attention 的生产适配不完整：page layout、head size、prefix-tail segment、runtime dispatch policy 都会影响 decode 小步高频调用。

**4. 怎么定位到这个原因？**

观察 decode attention dispatch branch 和测试覆盖，发现 page32/head256/queue4 prefix-tail 等 serving shape 缺 kernel 或策略不明确。

**5. 怎么优化的？**

增加 queue4 prefix-tail decode path；支持 page size 32 和 head size 256；加入 production decode dispatch policy；移除 fused decode disable switch；简化 queue4 segment policy。

**6. 优化后的现象是什么？**

更多真实 serving decode shape 能命中 TileLang decode attention fast path，dispatch policy 更稳定、可预测。

**7. 改善了什么？**

改善 decode attention 的 TPOT、覆盖面和生产可用性，减少 fallback 和手动开关依赖。

#### 4.10.3 RoPE decode path 优化

**1. 简介**

RoPE decode path 优化面向每步 decode 都会调用的小 shape 高频路径，关键文件是 `norm_rope_ops.py`。

**2. 当时表面现象是什么？**

单次 RoPE 开销不大，但在 decode 每 token 重复调用下会进入 TPOT 尾部；如果 miss fast path 或发生 materialization，TPOT 会变差。

**3. 根源是什么？**

decode RoPE 是 small-M、高频、graph-sensitive 路径，通用 prefill/fused kernel 未必适合；dispatch/fallback/materialization 开销会被每步放大。

**4. 怎么定位到这个原因？**

通过 decode trace 或 TPOT 分解观察 RoPE/norm_rope 相关 kernel 是否频繁出现、是否 miss TileLang path、是否发生 fallback。

**5. 怎么优化的？**

优化 MUSA RoPE decode path，让小 shape 更快命中 MUSA fast path，减少不必要 materialization 和 fallback，并保持 graph capture 友好。

**6. 优化后的现象是什么？**

decode RoPE 的 tail latency 降低，kernel trace 更稳定地命中 MUSA path。

**7. 改善了什么？**

改善 decode TPOT 中的高频辅助算子开销。

#### 4.10.4 cache store decode path：decode_x4 / vec2 / fp32

**1. 简介**

cache store 同时服务 decode 和 prefill。decode 侧保留 `decode_x4` / `decode_vec2` / `decode_x4_fp32` 等低延迟路径；prefill 侧走 tile-parallel / subwarp16 / x8 高吞吐路径。

**2. 当时表面现象是什么？**

如果 decode 使用 prefill 大 kernel，TPOT 可能回退；如果 cache store fallback 到通用路径，则每 token cache 写入开销偏高。

**3. 根源是什么？**

decode 是 small-M，对 launch 和每 token store latency 敏感；prefill 是 large-M，对写带宽敏感。两者最优策略不同。

**4. 怎么定位到这个原因？**

通过 cache benchmark 和 dispatch trace 对比 decode token 数下不同 pack/store 实现的 latency，发现 decode 需要 x4/vec2 等轻量路径，而 prefill 需要更高并行写带宽。

**5. 怎么优化的？**

在 `cache_ops.py` 中引入 decode impl auto dispatch，包括 `decode_x4`、`decode_x4_fp32`、i32 address、flat i32 address、vec2 等路径；同时保留 prefill tile-parallel / subwarp16 分支。

**6. 优化后的现象是什么？**

decode 小 token 继续走低延迟 cache store path，prefill 大 token 走高吞吐 path，二者互不污染。

**7. 改善了什么？**

改善 decode TPOT，同时保护 prefill 写带宽，是典型 decode/prefill 分流优化。

#### 4.10.5 MHC decode split correctness

**1. 简介**

`b64efe165` 修复 MHC pre big fuse 在 decode split 下的正确性问题。

**2. 当时表面现象是什么？**

decode small-M 下 MHC pre big fuse 可能出现结果不一致、split 配置不适配、或者某些 shape 下 correctness benchmark 失败。

**3. 根源是什么？**

MHC pre big fuse 的 split / hidden_block / pass_config 对 num_tokens 很敏感。prefill/mid-prefill 的配置不一定适合 decode tiny-M。

**4. 怎么定位到这个原因？**

对比不同 num_tokens、n_splits、decode-like shape 下 MHC 输出，发现问题集中在 decode split 策略。

**5. 怎么修复的？**

在 MHC ops 中根据 `num_tokens <= 32/64` 等 decode-like 条件选择更合适的 threads、hidden_block、pass_config 和 decode split kernel。

**6. 修复后的现象是什么？**

decode MHC split 输出稳定，small-M correctness 不再被 prefill-oriented split 策略影响。

**7. 改善了什么？**

改善 decode MHC correctness 和 TPOT 保护。

#### 4.10.6 paged decode kernel 与 `is_paged=True`

**1. 简介**

当前面 compressor / cache path 启用 `is_paged=True` 后，decode 也必须有对应 paged kernel，否则 prefill 写入的 paged layout decode 无法正确消费。

**2. 当时表面现象是什么？**

prefill 写入 paged metadata/cache 后，decode 阶段可能因为没有 paged kernel 支持而读错 cache、fallback 或 shape/layout 不匹配。

**3. 根源是什么？**

serving path 必须保证 prefill 写入 layout 与 decode 消费 layout 一致。只改 prefill `is_paged=True` 而不补 decode paged kernel，会造成路径不闭环。

**4. 怎么定位到这个原因？**

从 `is_paged=True` 后的 decode cache 访问异常或 kernel miss 出发，检查 decode kernel 支持矩阵，发现缺 paged decode kernel。

**5. 怎么修复的？**

新增 DSV4 MUSA decode paged kernel，使 decode 能按 paged metadata 正确访问 cache。

**6. 修复后的现象是什么？**

prefill paged 写入和 decode paged 读取一致，paged serving path 闭环。

**7. 改善了什么？**

改善 paged cache serving 的正确性和稳定性。


### 4.11 面试深挖版：decode / TPOT 常见追问

#### 4.11.1 DSV4 MUSA decode fast paths

**为什么会怀疑这个点？**

如果 prefill 已经优化但 TPOT 仍然高，trace 中 decode 阶段还在走通用 fallback 或 prefill-oriented kernel，就说明缺 decode fast path。decode 每 token 高频调用，小路径缺失会被每步放大。

**怎么排除其他可能？**

- 分开测 prefill latency 和 decode TPOT。
- 用 cache/MHC benchmark 单独测小 shape。
- trace decode kernel，确认是否命中 DSV4 MUSA decode kernel。
- 对比不同 batch / token 数，观察是否 small-M 特有问题。

**为什么选择新增 decode fast path？**

decode 的目标是 per-token latency，不是 large-M throughput。cache、HC head、MHC 等路径需要小 shape 专用策略，不能直接套 prefill 大 kernel。

**有什么副作用？**

路径变多，dispatch 条件更复杂，需要避免 decode/prefill 分支误选；还要增加 benchmark 防止 TPOT regression。

**怎么验收？**

cache/MHC decode benchmark、E2E TPOT、kernel trace、graph capture 兼容性。

**常见追问怎么答？**

> Q：为什么不是只优化 prefill？  
> A：仓库里确实有 decode fast path。prefill 解决 TTFT/large-M，decode 解决 TPOT/small-M，两者是不同专项。

#### 4.11.2 TileLang decode attention

**为什么会怀疑这个点？**

decode attention 是每 token 核心路径。如果某些 page_size/head_size 下 fallback，或 prefix-tail 处理不稳定，TPOT 会明显波动。

**怎么排除其他可能？**

- 固定模型和 cache store，只替换 decode attention dispatch。
- 测 page32/head256 是否命中 fast path。
- 用 `test_tilelang_unified_decode_page32.py` 覆盖 page layout。
- trace queue4 / prefix-tail branch。

**为什么选择 queue4 / prefix-tail / production policy？**

decode serving 中请求长度和 cache page 分布不均，prefix-tail 和 queue segment policy 会影响小步 attention 的组织方式。production policy 可以减少手动开关和不确定 fallback。

**有什么副作用？**

dispatch policy 更复杂，JIT 编译覆盖面增加；需要防止某些 shape 命中错误分支。

**怎么验收？**

page32/head256 测试、decode TPOT、dispatch branch trace、不同 batch/request length 的 correctness。

**常见追问怎么答？**

> Q：这是不是 TileLang-only？  
> A：当前实现依赖 TileLang decode attention，但优化目标是 decode attention small-M latency；理论上也可用其他后端实现类似策略。

#### 4.11.3 RoPE decode path

**为什么会怀疑这个点？**

RoPE 单次开销不大，但 decode 每 token 都调用。如果 TPOT profile 里 RoPE/norm_rope 高频出现，或者 miss fast path 后 fallback，就值得优化。

**怎么排除其他可能？**

- 对比开启/关闭 RoPE fast path 的 TPOT。
- trace 是否出现 materialization 或 torch fallback。
- 检查 small-M shape 是否满足 guard。

**为什么选择小 shape fast path？**

decode 不是大吞吐场景，重点是低 launch/低 tail latency。小 shape fast path 比复用 prefill fusion 更合适。

**有什么副作用？**

需要保证不同 rope dim、position layout、dtype 下 correctness；guard 过宽可能误入不支持 shape，guard 过窄则收益不足。

**怎么验收？**

RoPE 输出与 baseline 对齐，decode TPOT 下降，trace 命中 MUSA path，graph capture 无 fallback。

**常见追问怎么答？**

> Q：RoPE 这么小，值得优化吗？  
> A：decode 是每 token 重复调用，小算子尾部开销会乘以生成长度，TPOT 对这类高频小 kernel 很敏感。

#### 4.11.4 cache store decode / prefill 分流

**为什么会怀疑这个点？**

cache store 同时出现在 decode 和 prefill。若 decode 使用高吞吐大 kernel，TPOT 可能回退；若 prefill 使用轻量 decode kernel，带宽不够。

**怎么排除其他可能？**

- 分别测 decode tokens 和 8K prefill cache pack。
- trace `decode_x4` / `decode_vec2` / prefill x8/subwarp16 分支。
- 对比 page_size、dtype、contiguous 条件。

**为什么选择分流？**

decode 的目标是低延迟，prefill 的目标是高带宽。单一 kernel 很难同时最优，分流能避免互相伤害。

**有什么副作用？**

dispatch 复杂度上升，需要更多测试覆盖 page_size / dtype / invalid indices。

**怎么验收？**

decode TPOT 不回退，prefill cache bandwidth 提升；invalid indices 不写 cache；page boundary 正确。

**常见追问怎么答？**

> Q：为什么 decode 用 x4，prefill 用 x8/subwarp？  
> A：decode token 少，x4 轻量、低调度开销；prefill token 多，x8/subwarp 更能提高并行度和写带宽。

#### 4.11.5 MHC decode split correctness

**为什么会怀疑这个点？**

MHC 问题如果只在 small-M decode 出现，而 prefill 正常，说明 split/fuse 配置可能没有适配 decode-like shape。

**怎么排除其他可能？**

- 对比 num_tokens <=32/64 与大 token。
- 对比不同 n_splits 输出。
- 用 baseline unfused path 验证数值。

**为什么选择按 token 数选择配置？**

MHC pre big fuse 的最优 threads、hidden_block、split 策略随 token 数变化。decode tiny-M 需要更保守配置。

**有什么副作用？**

分支更多，某些边界 token 数需要测试；过度保守可能牺牲一点性能。

**怎么验收？**

decode correctness benchmark、不同 token 数边界测试、TPOT 不回退。

**常见追问怎么答？**

> Q：为什么 correctness 修复也算 decode 优化故事？  
> A：decode fast path 首先要正确。small-M split 错误会让 fast path 无法上线，所以 correctness 是 decode bring-up 的一部分。

#### 4.11.6 paged decode kernel

**为什么会怀疑这个点？**

当 prefill/cache path 启用 `is_paged=True` 后，如果 decode 读取异常，就要检查 decode 是否支持同样 paged layout。

**怎么排除其他可能？**

- 检查 prefill 写入 metadata 是否 paged。
- 检查 decode kernel 支持矩阵。
- 对比 non-paged 与 paged cache 结果。

**为什么选择补 decode paged kernel？**

serving 是闭环：prefill 写什么 layout，decode 必须按同样 layout 消费。只改 prefill metadata 不够。

**有什么副作用？**

需要维护 paged/non-paged 两套路由和测试；dispatch 需要准确识别 metadata。

**怎么验收？**

paged prefill + paged decode 连续生成正确，cache read/write layout 一致，E2E serving 无 fallback。

**常见追问怎么答？**

> Q：这个和 prefill `is_paged=True` 修复是什么关系？  
> A：前者让 prefill 写 paged metadata，后者让 decode 能消费 paged metadata，两者共同保证 serving path 闭环。



## 5. 第三阶段：FP8 prefill 基础瓶颈优化

这一阶段先解决 DSV4 FP8 prefill 的基础瓶颈：大量后续 FP8 GEMM 都依赖 activation quant。如果 quant 路径慢，MoE、WO_A、down projection 等主计算再快也会被输入准备拖住。

### 5.1 FP8 prefill quant 路径优化

相关提交：

- `78be4b50a` — `Optimize MUSA FP8 quant paths with TileKernels`
- `de830f875` — `Optimize DeepSeek V4 MUSA WO_A and quant paths`
- `f393eaa98` — `Optimize DSV4 MUSA prefill memory-bound kernels`

关键文件：

- `python/sglang/srt/layers/quantization/fp8_kernel.py`

本节只记录 MUSA / DSV4 分支新增或修改的 fast path；不再展开 SGLang 原有的 `sglang_per_token_group_quant_fp8`、通用 `_per_token_group_quant_8bit_raw`、通用 Triton quant kernel、通用 scale layout 创建等官方基础实现。

#### 5.1.1 背景：为什么 FP8 quant 会成为 prefill 问题

DeepSeek V4 FP8 在 MUSA 上运行时，很多后续计算需要把 activation 从 bf16/fp16/fp32 转成 FP8，并同时生成 per-token / per-group scale：

```text
x:   bf16/fp16/fp32 activation
x_q: fp8 activation
x_s: per-token/group scale
```

对于 prefill 阶段，`x` 通常对应大量 token 和较大的 hidden 维度，`x.numel()` 与 group 数都很大。此时 quant 本身不再只是一个很小的辅助操作，而会变成明显的 memory-bound / bandwidth-bound 开销，尤其会影响 MoE DeepGEMM prefill 中间 activation quant、WO_A FP8 GEMM 输入 quant、以及部分 cache/attention 前处理路径。

因此这轮优化的核心目标是：

> 在 MUSA 上为 DSV4 FP8 大 prefill shape 增加专用 quant fast path，同时避免 decode 小 M / tiny-M TPOT 回退。

#### 5.1.2 优化前 profiling / 现象判断

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

#### 5.1.3 优化策略一：用 16K groups 区分 prefill 大 shape

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

#### 5.1.4 优化策略二：MUSA Triton multi-group quant fallback

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

#### 5.1.5 优化策略三：TileLang MUSA half-warp/group FP8 quant kernel

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

#### 5.1.6 优化策略四：TileKernels per-token cast 的 MUSA 接入与保护

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

#### 5.1.7 最终 dispatch 顺序

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

#### 5.1.8 后续现象与效果

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

#### 5.1.9 验收方式

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



### 5.2 面试版逐项解析：FP8 prefill quant 路径优化

**1. 简介**

这是 DSV4 FP8 prefill 的基础优化，把大 shape activation quant 从通用路径切到 MUSA 专用 fast path。它服务于后续 MoE、WO_A、down projection 等 FP8 GEMM 输入准备。

**2. 当时表面现象是什么？**

prefill 大 token 下 quant kernel latency 明显，FP8 GEMM 前后出现 memory-bound gap；large CP prefill shape 下 generic TileKernels cast 反而慢；decode tiny-M 又不能使用大 kernel。

**3. 根源是什么？**

FP8 quant 需要读 activation、计算 group amax/scale、写 FP8 和 scale，是 memory-bound。prefill 大 M 会放大读写量；decode small-M 则更怕 launch 和调度开销。

**4. 怎么定位到这个原因？**

通过 profiling / benchmark 观察 quant 在 prefill 大 shape 中占比；对比 large prefill 与 decode tiny-M，发现二者最优实现不同；代码注释也记录 generic TileKernels cast 在 large CP prefill shape 上慢。

**5. 怎么优化的？**

用 `16K groups` 区分 prefill 大 shape；优先尝试 TileLang half-warp/group FP8 quant kernel，使用 FP8x4 pack 和 `stg32` 写回；增加 MUSA Triton multi-group fallback；禁止 large prefill miss 后落到已知慢的 generic TileKernels cast；decode small-M 保持原路径。

**6. 优化后的现象是什么？**

大 prefill shape 有专用 MUSA quant fast path，decode tiny-M 不被污染，large CP prefill 避免已知慢 fallback。

**7. 改善了什么？**

改善 prefill FP8 activation quant latency 和带宽利用，为 MoE DeepGEMM、WO_A、down projection 等 FP8 路径打基础。


### 5.3 面试深挖版：FP8 prefill quant 常见追问

**为什么会怀疑这个点？**

prefill profile 中 GEMM 前后出现明显 quant gap，且 large token / large CP shape 下 quant latency 随 token 数线性放大。FP8 模型里很多后续 GEMM 都依赖 activation quant，因此 quant 慢会拖住整条 FP8 pipeline。

**怎么排除其他可能？**

- 分离测 GEMM 与 quant，确认不是 GEMM 本身慢。
- 对比 decode tiny-M 和 prefill large-M，确认问题集中在 large shape。
- 对比 Triton、TileKernels、TileLang 路径，确认 generic TileKernels cast 在某些 large CP shape 上慢。
- 检查 scale layout / dtype / group_size 是否满足 fast path。

**为什么选择 16K groups？**

它是 workload 分界：total groups 足够大时，row-level work 足以摊薄 launch/调度开销，适合 half-warp/group 或 multi-group 大 kernel；低于该阈值更像 decode/small-M，保留原路径更安全。

**为什么选择 half-warp/group？**

group_size=128 时，一个 half-warp 16 lanes，每 lane 处理 8 elements，正好覆盖一个 group；warp 内可处理两个 group，amax reduce 和 FP8x4 pack/writeback 都比较自然。

**有什么副作用？**

guard 过宽会污染 decode，guard 过窄会错过 prefill 收益；TileLang kernel 只支持特定 dtype/layout/group_size，需要 fallback 覆盖；不同硬件/shape 下阈值可能需要调优。

**怎么验收？**

- large prefill 命中 TileLang/MUSA Triton fast path。
- decode small-M 不命中 large prefill path，TPOT 不回退。
- quant/dequant 误差可接受。
- large CP prefill 不落到已知慢的 generic TileKernels cast。

**常见追问怎么答？**

> Q：为什么不统一用一个最快 kernel？  
> A：prefill 和 decode shape 完全不同。large-M 适合高并行/带宽型 kernel，decode tiny-M 更怕 launch 和调度开销，统一 kernel 往往会让其中一边回退。

> Q：这是 TileLang-only 吗？  
> A：不是。half-warp/group、FP8x4 pack、宽写这些思想可以用 MUSA native/Triton/其他 codegen 实现；TileLang只是当前实现载体。



## 6. 第四阶段：prefill 主路径优化

这一阶段集中优化 prefill 中最重的两条主路径：MoE 和 sparse attention。MoE 侧通过 compact DeepGEMM 提高大 token FP8 grouped GEMM 吞吐；attention 侧通过 FlashMLA sparse seq-pack 减少 workspace/index metadata 的重复构建。

### 6.1 MoE DeepGEMM compact prefill path

相关提交：

- `cc5b01a55` — `Add DeepSeek V4 MUSA MoE DeepGEMM prefill path`

关键文件：

- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`
- `python/sglang/srt/layers/deepseek_v4_musa/kernels/moe_prefill_kernels.py`
- `python/sglang/srt/layers/deepseek_v4_musa/ops/moe_prefill_ops.py`
- `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_moe_prefill_deepgemm_benchmarks.py`

一句话概括：

> 这个优化是在 MUSA 上为 DeepSeek V4 FP8 MoE prefill 大 token 场景新增一条 compact DeepGEMM 路径，把 MoE 的 route / compact、gate-up GEMM、SwiGLU + FP8 quant、down GEMM、reorder / combine 等步骤组织成更适合 prefill 大 M 的高吞吐路径。

#### 6.1.1 背景：MoE prefill 的瓶颈不是单个 GEMM

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

#### 6.1.2 profiling 现象与优化动机

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

#### 6.1.3 启用条件与保护策略

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

#### 6.1.4 compact layout：把 token 整理成 expert-contiguous

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

#### 6.1.5 主计算 pipeline

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

##### 6.1.5.1 gate/up grouped GEMM

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

##### 6.1.5.2 fused SwiGLU + FP8 quant

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

##### 6.1.5.3 down grouped GEMM

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

##### 6.1.5.4 reorder / combine

由于前面为了 grouped GEMM 把 token compact 到 expert-contiguous layout，最后需要把输出还原回原始 token 顺序，并根据 router weight 做 combine：

```text
expert-contiguous down_output
  -> 根据 src2dst / dst2src 映射回原 token
  -> 乘以 router weight
  -> top-k expert 输出累加
  -> final hidden_states
```

这一步决定了 compact path 的整体收益不只取决于 GEMM 本身，还取决于 compact / padding / reorder / combine 的成本是否被大 M GEMM 和 fused quant 收益覆盖。

#### 6.1.6 内存优化与 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY`

在 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY` 打开时，路径会直接分配 FP8 down input 并写入 scale，避免先 materialize bf16 down input 再额外 quant 的中间内存压力。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:401`

这对 prefill 大 token 场景尤其重要，因为 `down_input` 的大小随 token 数和 intermediate size 增长。减少 bf16 intermediate 不仅节省显存占用，也可以降低 memory bandwidth 压力。

#### 6.1.7 与 5.1 FP8 prefill quant 优化的关系

5.1 主要优化的是通用 FP8 activation quant path：

```text
activation -> FP8 activation + scale
```

6.1 中的 MoE DeepGEMM compact prefill path 也依赖 FP8 quant，尤其是：

```text
SwiGLU output -> down_input_fp8 + down_input_scale
```

二者关系可以理解为：

```text
3.1 是降低 FP8 quant kernel 本身成本
5.1 是减少 MoE pipeline 中 FP8 quant 和中间内存的整体成本
```

换句话说：

- 3.1 更偏单个 quant kernel / dispatch 优化。
- 5.1 更偏 MoE operator pipeline 优化。
- 两者共同服务 DSV4 FP8 prefill。

#### 6.1.8 后续现象与预期效果

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

#### 6.1.9 benchmark 与验收方式

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

#### 6.1.10 小结

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


### 6.2 FlashMLA sparse prefill seq-pack path

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

#### 6.2.1 背景：DSV4 sparse prefill 为什么需要 seq-pack

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

#### 6.2.2 profiling 现象与优化动机

这类优化通常对应以下 profiling 现象。

1. **每层重复构建 metadata 的开销高**

   DSV4 层数较多，文档中按约 61 层 prefill 预算组织 benchmark。如果 query_start、SWA token ids、combined indices、workspace shape 等每层都重新构建，就会出现大量重复的小 tensor 创建和 kernel launch。

2. **sparse attention 的 gather/index 开销被放大**

   sparse prefill 不是顺序读取一整段 dense KV，而是需要根据 topk compressed indices 和 SWA positional range 做 gather。若每个 request、每层分别处理，会导致访问碎片化和调度碎片化。

3. **FlashMLA 需要对齐后的 per-query indices**

   `flash_mla_sparse_fwd` 希望输入是对齐且 rebased 的 per-query indices。若每层临时处理 topk + SWA 合并和 padding，会让 attention 前处理成为 prefill 热点。

4. **CP round-robin 下 local/global metadata 容易错位**

   context parallel round-robin 拆分时，本地 q 数可能与 global q 数不一致。如果继续使用 global query_start 去构建 sparse prefill cache，会导致 local query metadata 与实际本地计算不匹配。

#### 6.2.3 核心策略一：统一使用 `flash_mla_sparse_fwd`

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

#### 6.2.4 核心策略二：chunk-invariant metadata/cache 跨 layer 复用

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

#### 6.2.5 核心策略三：flat workspace 与 combined indices 布局

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

#### 6.2.6 核心策略四：topk + SWA indices 合并 kernel

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

#### 6.2.7 核心策略五：SWA token ids 构建

`build_swa_token_ids` 将每个 request 的 SWA positional range 转成物理 SWA-cache token ids。

处理逻辑包括：

- 根据每个 request 的 first visible length 计算 `swa_first_pos`。
- `swa_gather_lens = seq_lens - swa_first_pos`。
- 通过 `req_to_token` 和 `full_to_swa` 映射到 SWA cache id。
- 使用 Triton kernel 并行构建 flat token ids。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py:166`

这一步的意义是把 request-local 的 SWA 位置转换成后端 cache 能直接访问的物理 token id，从而为后续 workspace gather 做准备。

#### 6.2.8 c0 / c4 / c128 分层缓存

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

#### 6.2.9 CP round-robin 修复

`_build_sparse_prefill_cache_inputs` 中增加了 round-robin CP local query 的处理：

- 判断 `forward_mode.is_context_parallel_extend()`。
- 判断 `is_nsa_prefill_cp_round_robin_split()`。
- 当本地 q 数与 global q 数不一致时，重建 local `query_start_loc`。
- 用 local query_start 计算 local extend seq lens、req_pool_indices、seq_lens。

这修复了 sparse prefill 在 CP round-robin 下 local q metadata 与 global q metadata 不一致的问题。

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/srt/layers/attention/deepseek_v4_backend.py:1231`

该修复属于 MUSA DSV4 sparse prefill 路径上的高价值正确性修复。没有这个修复时，优化后的 fast path 在 CP round-robin 场景可能因为 query metadata 错位而读错 workspace 或生成错误 indices。

#### 6.2.10 后续现象与预期效果

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

#### 6.2.11 验收方式

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



### 6.3 面试版逐项解析：prefill 主路径优化

#### 6.3.1 MoE DeepGEMM compact prefill path

**1. 简介**

这是 DSV4 FP8 MoE prefill 主路径优化。大 token 下把 token 按 expert compact，gate/up 和 down 走 FP8 grouped DeepGEMM，中间融合 SwiGLU + FP8 quant。

**2. 当时表面现象是什么？**

MoE prefill 不是单个 GEMM 慢，而是 route/compact、gate-up GEMM、SwiGLU、quant、down GEMM、reorder/combine 整条 pipeline 开销高，中间 bf16 tensor 读写压力大。

**3. 根源是什么？**

MoE token 分散到不同 expert，原 layout 不适合 grouped GEMM；SwiGLU 后如果先 materialize bf16 down_input 再 quant，会产生大规模 memory traffic。

**4. 怎么定位到这个原因？**

通过 MoE prefill benchmark / profiling 看 gate/up、SwiGLU、quant、down、reorder 等 stage；发现大 token 下 compact layout 和 fused post-quant 有明确收益空间。

**5. 怎么优化的？**

在 MUSA + FP8 + token >= 8192 + env enabled 时启用 compact DeepGEMM path：token 按 expert compact，gate/up grouped GEMM 输出 bf16，`silu_and_mul_contig_post_quant` 直接生成 `down_input_fp8 + scale`，再做 down grouped GEMM，最后 reorder/combine。

**6. 优化后的现象是什么？**

大 token MoE prefill 命中专用 fast path，bf16 intermediate 读写减少，grouped GEMM 输入更连续；decode small-M 通过阈值保护不进入该路径。

**7. 改善了什么？**

改善 MoE prefill operator latency 和 FP8 pipeline 吞吐，降低 memory traffic，提升 TTFT / 长上下文 prefill 能力。

#### 6.3.2 FlashMLA sparse prefill seq-pack path

**1. 简介**

这是 sparse attention prefill 主路径优化。它把 compressed cache 和 SWA window gather 到 flat BF16 workspace，并生成 per-query rebased combined indices 交给 `flash_mla_sparse_fwd`。

**2. 当时表面现象是什么？**

每层都重复构建 query_start、SWA token ids、combined indices、workspace 等小 tensor；sparse gather/index 前处理开销高；CP round-robin 下 local/global metadata 容易错位。

**3. 根源是什么？**

sparse prefill 输入不是 dense KV，而是 compressed cache + SWA + topk indices 的组合。很多 metadata 在 chunk 内不随 layer 变化，却被重复构建。

**4. 怎么定位到这个原因？**

分析 `_forward_prefill_sparse` 和 sparse prefill utils，发现大量 chunk-invariant metadata 可复用；观察 DSV4 多层下小 tensor 构建重复出现。

**5. 怎么优化的？**

统一走 `flash_mla_sparse_fwd`；设计 flat workspace；合并 topk + SWA indices 并 padding 到 128；构建 `SparsePrefillChunkCache` 跨 layer 复用 c0/c128 workspace 和 indices；c4 per-layer 更新但复用 buffer；补 CP round-robin local metadata。

**6. 优化后的现象是什么？**

FlashMLA sparse prefill 输入更规则，每层重复 metadata 构建减少，CP round-robin 下 sparse prefill metadata 正确。

**7. 改善了什么？**

改善 sparse attention prefill 前处理和多层复用效率，降低 TTFT 和长上下文 sparse prefill overhead。


### 6.4 面试深挖版：prefill 主路径常见追问

#### 6.4.1 MoE DeepGEMM compact prefill

**为什么会怀疑这个点？**

DeepSeek V4 是 MoE 模型，大 prefill 下 MoE 层占比高。profile 中不仅 GEMM 重，route/compact、SwiGLU、quant、reorder/combine 也有明显开销，说明要看整条 MoE pipeline。

**怎么排除其他可能？**

- 分 stage 测 gate/up GEMM、SwiGLU、quant、down GEMM、reorder。
- 对比 expert 分布和 padding rows，判断是否 compact/padding 浪费。
- 对比 split SwiGLU+quant 与 fused post-quant。
- 小 token decode 下验证 compact path 不进入。

**为什么选择 compact + fused SwiGLU/quant？**

compact 让同 expert token 连续，适合 grouped GEMM；fused SwiGLU+quant 避免先写 bf16 down_input 再读回 quant，直接输出 FP8 down_input + scale，减少 memory traffic。

**有什么副作用？**

compact/reorder 本身有成本，expert 分布不均或 padding 多时收益变小；需要阈值保护 small-M decode；experimental env 控制上线风险。

**怎么验收？**

MoE benchmark 中看 `candidate_device`、`device_speedup`、`padded_valid_rows`、`allocated_rows`、`overflow`；同时验证输出误差和 decode 不回退。

**常见追问怎么答？**

> Q：为什么 token >=8192 才启用？  
> A：compact/reorder 有固定成本，只有大 token 时 grouped GEMM 和 fused quant 的收益才能覆盖前后处理成本。

#### 6.4.2 FlashMLA sparse prefill seq-pack

**为什么会怀疑这个点？**

DSV4 sparse prefill 每层都要处理 compressed cache、SWA、topk indices。若每层重复构建 workspace/indices，小 tensor 和 metadata 开销会被 60+ 层放大。

**怎么排除其他可能？**

- 观察每层是否重复构建 query_start、SWA token ids、combined indices。
- 区分 layer-invariant 和 layer-dependent metadata。
- 对比 c0/c128 可预计算部分与 c4 per-layer 部分。
- CP round-robin 下检查 local/global q 是否一致。

**为什么选择 flat workspace + rebased indices？**

FlashMLA sparse kernel 更适合规则输入。把 compressed region 和 SWA region gather 到 flat workspace，再用 rebased combined indices 表达每个 query 的可见 token，可以把复杂 cache layout 转成统一 kernel 输入。

**有什么副作用？**

workspace 需要额外显存；c4 topk per-layer 变化时仍需 per-layer combine；indices rebasing/padding 错误会导致难查的 attention 错误。

**怎么验收？**

验证 combined indices 的 compressed/SWA 区间、padding=-1、valid_lens、workspace 448+64 layout；CP round-robin 下 local metadata 正确；多层复用后输出与 baseline 对齐。

**常见追问怎么答？**

> Q：为什么不是每层重新构建，逻辑更简单？  
> A：DSV4 层数多，chunk 内大量 metadata 不随 layer 变化。每层重建会放大小 tensor 和 launch overhead，seq-pack/cache 复用能显著降低前处理开销。



## 7. 第五阶段：shared memory-bound 算子优化

当 FP8 quant、MoE 和 sparse attention 主路径优化后，cache store、compress、norm、rope、topk、MHC 等辅助算子的尾部开销会更明显。本阶段的目标是补齐这些 memory-bound / launch-bound operator，避免它们成为新的 prefill 瓶颈。

### 7.1 FlashMLA cache store decode / prefill 分流优化

相关提交：

- `7c992a2a8` — `Optimize DeepSeek V4 MUSA FlashMLA cache store prefill`
- `f393eaa98` — memory-bound kernels

关键文件：

- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/cache_kernels.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/cache_ops.py`
- `python/sglang/jit_kernel/tests/deepseek_v4_musa/tests/test_cache_ops.py`

一句话概括：

> 这组优化为 MUSA 上 FlashMLA cache store 增加 prefill 专用 dispatch，将 decode 小 token 和 prefill 大 token 分流，并在 prefill 下使用 tile-parallel / subwarp16、FP8 pack、uint32/stg32/stg128 写回等路径，降低 KV cache pack/store 的 memory-bound 开销。

#### 7.1.1 背景：cache store 是 prefill 中典型 memory-bound 环节

DSV4 FlashMLA cache store 需要把当前 token 的 KV 表示写入 cache。对于 FlashMLA/MLA 结构，cache store 不只是简单 memcpy，还包括：

- nope 部分从 bf16/fp32 quant 成 FP8 E4M3。
- 写入 FP8 cache 数据。
- 写入对应 scale。
- rope 部分以 bf16 格式 pack/store。
- 根据 page_size 和 token index 写入分页 cache。

prefill 阶段 token 数大，cache store 的读写量随 token 数线性增长，因此很容易成为 memory-bound kernel。

decode 阶段则 token 数小，优化目标不同：decode 更怕额外 launch / shared-memory reduce / 大 tile 策略导致 TPOT 回退。

#### 7.1.2 profiling 现象与优化动机

该路径对应的典型 profiling 现象包括：

1. **prefill cache store 带宽利用不足**

   legacy prefill kernel 中如果使用共享内存 AllReduce 或较粗的通用路径，会在大 token store 下出现 bandwidth 利用不足。

2. **decode 与 prefill 复用同一策略会互相伤害**

   decode 小 M 适合 x4 这类轻量路径；prefill 大 M 更适合 tile-parallel 或 half-warp/token x8。若统一使用一种策略，要么 prefill 不够快，要么 decode TPOT 回退。

3. **FP8 nope + BF16 rope 混合写回需要更高效 pack/store**

   nope 是 FP8，rope 是 bf16，scale 又是 fp32/指定 scale layout。分散写回会造成 store 粒度小、访存不连续和指令效率低。

#### 7.1.3 核心策略一：decode 与 prefill 分流

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

#### 7.1.4 核心策略二：prefill tile-parallel / subwarp16 路径

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

#### 7.1.5 核心策略三：FP8 nope + BF16 rope pack/store

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

#### 7.1.6 后续现象与预期效果

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

#### 7.1.7 测试与验收方式

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


### 7.2 compress + fused norm rope prefill 优化

相关提交：

- `6a57c5b25` — `Optimize MUSA compress fused norm rope prefill with guarded dispatch`
- `f393eaa98` — memory-bound kernels

关键文件：

- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/norm_rope_kernels.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/norm_rope_ops.py`
- `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py`

一句话概括：

> 这组优化针对 MUSA 上 DSV4 compress rows 的 prefill 路径，将 RMSNorm、RoPE 以及 compress 相关准备工作通过 guarded dispatch 融合到更少的 kernel 中，减少中间 tensor、global memory round-trip 和 launch 开销。

#### 7.2.1 背景：compress rows 上 norm / rope 是高频前处理

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

#### 7.2.2 profiling 现象与优化动机

典型现象包括：

1. **norm / rope / compress 前后处理 kernel 串联**

   每个 kernel 本身计算量不一定大，但都要读写较大的 hidden/head_dim tensor，容易受 memory bandwidth 限制。

2. **中间 tensor 生命周期短但读写量大**

   norm output 或 rope output 可能只被下一步消费一次，却需要完整写回 global memory 再读回来。

3. **prefill 大 M 下 launch 和访存开销累计明显**

   单层看似不大，但乘以 DSV4 多层后，会成为 prefill operator-side budget 的一部分。

#### 7.2.3 核心策略一：compress prefill 行融合 norm + rope

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

#### 7.2.4 核心策略二：guarded dispatch

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

#### 7.2.5 acceptance benchmark 预算

operator benchmark 为该路径设置了 8K prefill 下的预算：

- `compress_fused_norm_rope_prefill_h512_r64_c{compress_ratio}_b1_{num_tokens}`
- `budget_ms = 0.30`

参考：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:367`

这个 budget 说明该路径被定位为非常轻量的 memory-bound fused 前处理：在 8K prefill 条件下，单次 operator 预算只有 0.30ms 量级。

#### 7.2.6 后续现象与预期效果

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

#### 7.2.7 验收方式

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


### 7.3 compress prefill memory-bound kernel 优化

相关提交：

- `f393eaa98` — `Optimize DSV4 MUSA prefill memory-bound kernels`

关键文件：

- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/compress_kernels.py`
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/compress_ops.py`
- `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py`

一句话概括：

> 这组优化针对 MUSA 上 DSV4 prefill 的 C4 / C128 compress kernel，围绕 reduce、write、parallel reduction 和 vectorized store 改善 memory-bound 路径，并用 logical bytes / GB/s 和 8K prefill budget 约束性能。

#### 7.3.1 背景：compress 是典型 memory-bound pipeline

DSV4 的 compressed attention/cache 路径中，C4 / C128 compress 需要读取 KV、score、APE 等数据，做 reduce 或写 cache。它的特点是：

- 算术强度不高。
- 读写数据量大。
- 对连续访问、vectorized load/store、parallel reduce 很敏感。
- 8K prefill 下调用规模大，容易成为 operator-side bottleneck。

benchmark 里分别覆盖：

- `compress_ratio4_prefill_b1_8192`
- `compress_ratio128_prefill_b1_8192`

#### 7.3.2 profiling 现象与优化动机

典型现象包括：

1. **C4 / C128 reduce 带宽受限**

   reduce 需要读取多组 kv / score / ape 数据，但每个输出元素的计算并不复杂。性能主要取决于内存读写效率和并行 reduce 组织。

2. **write kernel store 粒度影响明显**

   compress write 本质上是在把多段 head_dim 数据写入 cache。如果 store 粒度小或不连续，带宽利用率会下降。

3. **C4 与 C128 的最优策略不同**

   C4 reduce 规模较小，更关注轻量并行和写回效率；C128 reduce 聚合范围更大，更需要 parallel reduce 和合理的线程组织。

#### 7.3.3 核心策略一：C4 / C128 prefill compress 专用路径

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

#### 7.3.4 核心策略二：logical bytes / derived GB/s 度量

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

#### 7.3.5 核心策略三：vector write 与 C128 parallel reduce

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

#### 7.3.6 acceptance benchmark 预算

8K prefill operator budget：

| benchmark | budget |
| --- | ---: |
| `compress_ratio4_prefill_b1_8192` | 6.0 ms |
| `compress_ratio128_prefill_b1_8192` | 10.0 ms |

参考：

- C4：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:516`
- C128：`origin/perf/yunqiao.jiang/dsv4_prefill:python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py:537`

#### 7.3.7 后续现象与预期效果

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

#### 7.3.8 验收方式

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


### 7.4 MHC / norm / rope / topk operator 优化

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

#### 7.4.1 背景：端到端 prefill 由多个辅助算子共同决定

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

#### 7.4.2 profiling 现象与优化动机

典型现象包括：

1. **辅助算子数量多，launch-bound 明显**

   每个 layer 中 norm、rope、topk/indexer、MHC 前后处理都会出现。如果没有融合或专用 kernel，launch overhead 和调度间隙会被多层放大。

2. **大部分辅助算子偏 memory-bound**

   RMSNorm、RoPE、Hadamard、cache pack 都需要读写较大 tensor，但计算强度不高，优化重点是访存组织和融合。

3. **service-like 8K prefill 需要预算化验收**

   这些 operator 很难只用单个 E2E 指标定位，因此分支中通过 operator acceptance budgets 对每类算子设置上限，作为 E2E serving 前的 gate。

#### 7.4.3 MHC post / pre / prenorm 优化

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

#### 7.4.4 topk / indexer / rope / hadamard 优化

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

#### 7.4.5 RMSNorm / norm-rope-quant group 优化

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

#### 7.4.6 FlashMLA cache pack group 优化

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

#### 7.4.7 acceptance benchmark 预算

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

#### 7.4.8 后续现象与预期效果

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

#### 7.4.9 验收方式

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




### 7.5 面试版逐项解析：shared memory-bound 算子优化类

#### 7.5.1 FlashMLA cache store decode / prefill 分流优化

**1. 简介**

这是 KV/cache pack-store 的 decode/prefill 双路径优化。decode 保留低延迟 x4/vec2/fp32 路径，prefill 使用 tile-parallel/subwarp16/x8 高吞吐路径。

**2. 当时表面现象是什么？**

cache store 在 prefill 大 token 下带宽利用不足；如果用大 prefill kernel 处理 decode，会造成 TPOT 回退；FP8 nope、scale、BF16 rope 混合写回效率不高。

**3. 根源是什么？**

decode 和 prefill 的最优 store 粒度不同。decode 要低 latency，prefill 要高 bandwidth。FlashMLA cache layout 又包含 FP8 nope、scale、BF16 rope，多种数据类型混合写回。

**4. 怎么定位到这个原因？**

通过 cache benchmark、dispatch trace 和 page_size/dtype 组合测试，发现 decode small-M 与 prefill large-M 需要分流；legacy prefill kernel 的 shared-memory AllReduce / 小粒度 store 不理想。

**5. 怎么优化的？**

decode 走 `decode_x4` / `decode_vec2` / `decode_x4_fp32` 等；prefill 走 tile-parallel / subwarp16；nope FP8 pack，rope BF16 pack，使用 uint32/stg32/stg128 宽写；对 page_size 2/64/256 等做特化。

**6. 优化后的现象是什么？**

decode TPOT 被保护，prefill cache store bandwidth 提升，不同 dtype/page_size 分支更明确。

**7. 改善了什么？**

同时改善 decode latency 和 prefill cache store 吞吐，避免单一 kernel 策略互相伤害。

#### 7.5.2 compress + fused norm rope prefill 优化

**1. 简介**

这是 compress rows 上的 fusion 优化，把 RMSNorm、RoPE、compress 前处理合并到更少 kernel 中。

**2. 当时表面现象是什么？**

norm / rope / compress 前处理 kernel 串联，单个 kernel 不大但多层累计明显；中间 tensor 生命周期短，却要完整写回再读。

**3. 根源是什么？**

这些算子算术强度不高，主要受 memory bandwidth 和 launch overhead 限制。拆开执行会产生多次 global memory round-trip。

**4. 怎么定位到这个原因？**

operator benchmark 中单独列出 `compress_fused_norm_rope_prefill...`，并设置 8K prefill budget；profiling 可见 norm/rope/compress 前处理串联。

**5. 怎么优化的？**

新增 guarded fused path，只在 compress rows、rope dim、hidden size、dtype 等匹配时融合 RMSNorm + RoPE + final output write；不匹配时回退原路径。

**6. 优化后的现象是什么？**

kernel launch 数减少，中间 tensor materialization 减少，8K prefill 前处理 latency 更可控。

**7. 改善了什么？**

改善 memory-bound 前处理开销和多层累计 tail latency。

#### 7.5.3 C4 / C128 compress prefill memory-bound kernel

**1. 简介**

这是 compressed attention/cache 中 C4 / C128 reduce 和 write 的 memory-bound kernel 优化。

**2. 当时表面现象是什么？**

C4/C128 compress operator 在 8K prefill 下 latency 较高，derived GB/s 不理想；C128 大窗口 reduce 尤其容易受读带宽和并行 reduce 组织影响。

**3. 根源是什么？**

compress reduce/write 算术强度低，读取 kv/score/ape 多，写 cache 多；性能主要取决于访存合并、vectorized store 和 parallel reduce。

**4. 怎么定位到这个原因？**

benchmark 定义 logical bytes 和 derived GB/s，分别覆盖 C4/C128 prefill；如果 latency 高且 GB/s 低，就说明访存组织是瓶颈。

**5. 怎么优化的？**

针对 C4/C128 建立专用路径；启用 vector write；C128 使用 parallel reduce；增加 `RAISE_PREFILL_MISS` 便于暴露 fast path miss。

**6. 优化后的现象是什么？**

C4/C128 compress latency 降低，derived GB/s 提升，8K prefill budget 更容易通过。

**7. 改善了什么？**

改善 compressed cache / attention 相关 memory-bound pipeline，降低 long-context prefill 尾部开销。

#### 7.5.4 MHC / norm / rope / topk operator 优化

**1. 简介**

这是一组高频辅助算子优化，包括 MHC pre/post/prenorm、topk transform 512、RMSNorm、fused RoPE Q/K、c4 indexer+rope+hadamard、norm-rope-quant group、FlashMLA cache pack group。

**2. 当时表面现象是什么？**

单个算子不一定是最大热点，但在 8K prefill、多层、多次调用下累计明显；当 MoE/attention 主路径变快后，这些辅助算子可能成为新的尾部瓶颈。

**3. 根源是什么？**

这些算子大多偏 memory-bound / launch-bound：读写 tensor 多、计算强度不高、调用频繁。通用 kernel 或拆分执行在 service-like 8K prefill 下不够稳定。

**4. 怎么定位到这个原因？**

operator acceptance benchmark 对每个算子设置 budget，并输出 median/min/p95、logical bytes、dispatch branch、trace kernel；通过 group benchmark 观察多层累计开销。

**5. 怎么优化的？**

MHC 使用更合适的 MUSA kernel / DeepGEMM HC prenorm backend；topk512 使用 JIT path；Q/K RoPE 融合；c4 indexer+rope+hadamard 组合处理；norm-rope-quant 和 cache pack 做 group-level 优化和验收。

**6. 优化后的现象是什么？**

辅助算子 latency 更可控，多层 group benchmark 更稳定，p95 长尾减少。

**7. 改善了什么？**

改善服务化 prefill 的尾部延迟，防止主路径优化后辅助算子成为新瓶颈。


### 7.6 面试深挖版：shared memory-bound 算子常见追问

#### 7.6.1 FlashMLA cache store decode / prefill 分流

**为什么会怀疑这个点？**

cache store 是每个 token 都绕不开的路径。prefill 大 token 下写带宽不足会拖 TTFT；decode 下每 token store 如果过重会拖 TPOT。

**怎么排除其他可能？**

单独跑 cache benchmark，按 token 数、dtype、page_size、decode/prefill 分支对比；trace 是否命中 decode_x4/vec2 或 prefill tile-parallel/subwarp。

**为什么选择 decode/prefill 分流？**

decode 关注低固定开销，prefill 关注带宽吞吐。x4/vec2 更适合 small-M，x8/subwarp/tile-parallel 更适合 large-M。

**有什么副作用？**

dispatch 和测试矩阵变大，需要覆盖 page_size、dtype、invalid indices、contiguous 输入等。

**怎么验收？**

decode TPOT 不回退；prefill cache pack latency 下降；cache layout、scale、rope 写入正确。

**常见追问怎么答？**

> Q：为什么 cache store 也放在 shared 阶段？  
> A：因为它同时影响 decode 和 prefill，只是两阶段选择不同 kernel 策略。

#### 7.6.2 compress + fused norm rope

**为什么会怀疑这个点？**

norm/rope/compress 都是读写型算子，单独看不重，但多层 prefill 下串联执行会产生大量中间 tensor 和 launch。

**怎么排除其他可能？**

用 operator benchmark 单独测 fused 前处理；对比拆分路径与 fused path；检查是否 memory bandwidth 受限而不是算力受限。

**为什么选择 fusion？**

中间结果生命周期短，只被下一步消费。融合后可以减少 global memory write/read 和 kernel launch。

**有什么副作用？**

fusion kernel 的 shape/dtype/rope dim 支持范围有限，需要 guarded dispatch；过宽 guard 会有 correctness 风险。

**怎么验收？**

fused output 与 baseline 对齐；unsupported shape 能 fallback；8K prefill budget 通过。

**常见追问怎么答？**

> Q：为什么不所有 norm/rope 都融合？  
> A：融合需要满足 layout、dtype、rope dim 等约束，不支持的 case 盲目融合风险更高，所以用 guarded dispatch。

#### 7.6.3 C4 / C128 compress memory-bound kernel

**为什么会怀疑这个点？**

compress reduce/write 的 logical bytes 很大，算术强度低。如果 latency 高且 derived GB/s 低，就说明访存组织是瓶颈。

**怎么排除其他可能？**

分别测 C4 和 C128；看 reduce 与 write 子项；用 logical bytes 计算 GB/s；对比 vector write / parallel reduce 开关。

**为什么选择 vector write 和 C128 parallel reduce？**

write path 需要提高 store 合并效率；C128 聚合窗口大，parallel reduce 可以提高读带宽和归约效率。

**有什么副作用？**

并行 reduce 的数值顺序可能和 baseline 略有差异，需要误差容忍；vector write 对 alignment/layout 更敏感。

**怎么验收？**

C4/C128 budget 通过；derived GB/s 提升；输出误差可接受；miss fast path 能通过 env 暴露。

**常见追问怎么答？**

> Q：为什么用 GB/s 而不是只看 ms？  
> A：这是 memory-bound kernel，GB/s 能反映访存组织是否接近硬件能力，单看 ms 不容易判断瓶颈性质。

#### 7.6.4 MHC / norm / rope / topk operator

**为什么会怀疑这个点？**

主路径优化后，辅助算子的累计开销会浮上来。MHC、topk、norm、rope、cache pack 单次可能不大，但多层/多调用会形成 tail latency。

**怎么排除其他可能？**

用 operator acceptance benchmark 分项测；看 median/min/p95 和 group benchmark；trace dispatch branch，确认不是 fallback。

**为什么选择 budget 化管理？**

这些算子多且分散，E2E 里难定位。给每类 operator 设置 8K prefill budget，可以在 serving 前先兜住尾部开销。

**有什么副作用？**

budget 需要随硬件和模型配置调整；过严会增加维护成本，过松又失去 gate 价值。

**怎么验收？**

所有 operator budget 通过；p95 无明显长尾；dispatch 命中预期 kernel；E2E prefill 没有新的尾部瓶颈。

**常见追问怎么答？**

> Q：这些小算子值得单独优化吗？  
> A：在多层长上下文 serving 中，小算子会被重复几十次甚至更多。主路径变快后，它们会成为新的尾部瓶颈，所以需要预算化管理。



## 8. 性能数据与 benchmark 状态

### 8.1 decode 相关 benchmark 补充

除了原先整理的 operator acceptance benchmark 和 MoE prefill benchmark，decode 侧还需要关注仓库中的 cache / MHC / TileLang decode attention 相关 benchmark 与测试，例如：

- `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_cache_benchmarks.py`
- `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_mhc_benchmarks.py`
- `test/srt/test_tilelang_unified_decode_page32.py`

decode 验收重点不是 large-M throughput，而是 TPOT、small-M latency、dispatch branch、graph capture 兼容性、cache store correctness、sampling 稳定性。

### 8.2 仓库中能直接确认的性能目标

当前分支中没有找到已经提交的真实 benchmark 日志结果，例如固定的 `candidate_device=... ms` 或 E2E tokens/s 结果文件。能直接从代码确认的是：

1. benchmark 输出字段已经写好。
2. operator acceptance benchmark 里有明确的性能预算 `budget_ms`。
3. benchmark 形状主要覆盖 no-MTP B1 8K prefill service-like path。
4. 部分 MoE benchmark 会输出 baseline/candidate speedup。

因此本文档中的“性能”分成两类：

- **已固化在代码里的 acceptance budget**：可以直接引用。
- **运行 benchmark 后得到的实测性能**：代码支持输出，但仓库中未固化具体数值，需要在 MUSA 环境上实际执行。

### 8.3 Operator acceptance benchmark

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

### 8.4 MoE DeepGEMM prefill benchmark

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



## 9. 重要环境变量

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



## 10. 建议阅读顺序

如果要继续深入看代码，建议按如下顺序：

1. **FP8 quant dispatch**  
   `python/sglang/srt/layers/quantization/fp8_kernel.py`

2. **MoE DeepGEMM prefill**  
   `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`  
   `python/sglang/srt/layers/deepseek_v4_musa/kernels/moe_prefill_kernels.py`  
   `python/sglang/srt/layers/deepseek_v4_musa/ops/moe_prefill_ops.py`

3. **Decode / TPOT 路径**  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/cache_ops.py`  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/norm_rope_ops.py`  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/mhc_ops.py`  
   `python/sglang/srt/layers/attention/tilelang_unified_attention_v2.py`  
   `python/sglang/srt/layers/attention/tilelang_unified_attention_v2_queue4_prefix_tail.py`

4. **FlashMLA sparse prefill**  
   `python/sglang/srt/layers/attention/deepseek_v4_backend.py`  
   `python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`

5. **FlashMLA cache store**  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/cache_ops.py`  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/cache_kernels.py`

6. **compress / norm / rope / MHC operator**  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/norm_rope_ops.py`  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/norm_rope_kernels.py`  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/compress_ops.py`  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/kernels/compress_kernels.py`  
   `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/mhc_ops.py`

7. **性能验证**  
   `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_operator_acceptance_benchmarks.py`  
   `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_moe_prefill_deepgemm_benchmarks.py`  
   `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_cache_benchmarks.py`  
   `python/sglang/jit_kernel/tests/deepseek_v4_musa/benchmarks/test_mhc_benchmarks.py`



## 11. 当前缺口

本次只做静态代码与提交历史梳理，没有在 MUSA 环境实际跑 benchmark。因此：

- 已确认优化项、dispatch 条件、benchmark 预算和输出字段。
- 未确认当前机器上的真实 median ms / GB/s / E2E tokens/s / decode TPOT。
- 仓库中未发现已提交的 benchmark 结果日志。

如果要补齐“优化后的实际性能”，建议在对应 MUSA 环境上运行第 4 节中的 benchmark，并把 JSONL 结果补到本文档后面。
