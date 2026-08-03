# Profile And Projects

## Current Resume Summary

- 核心经历：在摩尔线程 S5000 上参与 DeepSeek V4 FP8 Serving 的 MUSA 后端落地与性能优化，覆盖 Kernel、数据流水线、Graph/多卡状态闭环和验证方法。
- 当前材料中的性能数字均为项目口径或推导，缺完整硬件、shape、baseline commit 和原始日志时不得表述为本人复测。

## Skills Inventory

- GPU Kernel：half-warp/shuffle reduce、向量化访存、FP8 pack、parallel reduce、Split-K。
- 推理系统：MoE、FlashMLA、Paged Cache、MTP/EAGLE、PP/PD、CP metadata。
- 性能工程：Roofline、bytes/FLOPs/launch 分析、shape-aware dispatch、fast-path/fallback 验证。
- 工程验证：数值等价、strided-input 反作弊、graph capture/replay、host-inclusive benchmark。

## Projects

### DeepSeek V4 FP8 MUSA Serving

- Source: `docs/resume_docs/deepseekv4_fp8_prefill_optimizations.md`、`docs/resume_docs/resume_final.md`
- 一句话定位：保留 DeepSeek/SGLang 的官方算法与上层语义，在 S5000 上重写或调优执行路径，解决高算力、相对低带宽条件下的中间物化、碎片化数据、small-M 和 runtime 闭环问题。
- 贡献边界：
  - 上游已有：DeepSeek V4 语义、Expert-Contiguous、`m_indices/src2dst`、DeepGEMM grouped GEMM、FlashMLA sparse 数学语义、SwiGLU+FP8 quant 等。
  - MUSA 平台工作：设备/dtype/layout bring-up、S5000 专用 Kernel、quant-scatter 等融合路径、shape dispatch、overflow/fallback、Graph/通信/状态闭环。
  - 个人归属：材料中标记为“我的”的内容仍需用 `git log --author` 和提交记录确认，确认前统一说“团队完成/我负责其中某模块”。
- 建议优先掌握的四条故事：
  1. Fused Norm-RoPE-Cache：证据最完整，展示物化消除与测试意识。
  2. FlashMLA Cache Store 双路径：展示 Prefill/Decode workload 分流与微架构判断。
  3. MoE Compact + Fused Quant-Scatter：展示 Roofline、数据流重构、容量契约和端到端意识。
  4. Sparse TopK-512/Pack8：展示 exactness、Graph 安全和 metadata 正确性。
- 次级故事：HC-head Split-K、Paged C4 Compress、Online C128 × MTP deferred-commit、JIT key/fail-closed runtime hardening。
- 详细复习入口：`interviews/deepseekv4-relearning.md`。

## Reusable Stories

- 30 秒项目定位：官方提供算法和框架语义，我的工作重点是让这些能力在 S5000/MUSA 上正确、高效、可进入生产 fast path；通过 Kernel 重写、数据物化消除、Prefill/Decode 分流以及 Graph/多卡状态闭环，形成从上游评估到服务验收的完整适配方法。
- 90 秒主故事：优先使用 Fused Norm-RoPE-Cache；若岗位更偏性能分析则使用 MoE Quant-Scatter；若更偏系统则使用 Online C128 × MTP 状态机。
- 最难问题故事：TopK candidate capacity 在 262K 长行下失效，或 JIT cache key 缺 layout 参数导致顺序相关错误；具体定位过程需回代码确认后再讲。

## Project-To-Question Links

- “为什么融合？” → Fused Norm-RoPE-Cache / MoE Quant-Scatter。
- “为什么需要多个 Kernel？” → Cache Store Prefill/Decode 双路径。
- “如何判断 memory-bound？” → Roofline + bytes 账本。
- “如何保证优化真的命中？” → dispatch、JIT key、fallback reason、capture/replay、reference 对齐。
- “哪些是你做的？” → 上游/平台/个人三层归属表。
- “怎么处理动态 shape 与 Graph？” → MoE static-cap + overflow fallback。
- “投机解码如何避免污染状态？” → Online C128 deferred-commit。

## Expression Risks

- 个人贡献边界尚未通过提交作者信息确认。
- `Fused Norm-RoPE-Cache` 的线程映射、v1/v2 差异尚未掌握。
- Cache Store 的 decode 三变体选择条件、Prefill 7 tile 拆 4+3 的依据尚未掌握。
- `8192`、`1024`、`T>=128`、`16K groups` 等阈值是否来自 sweep 尚未确认。
- 所有性能数字缺完整复现实验上下文；只能使用“项目口径”或“按流量公式推导”。
- 不得宣称 TopK V2、SwiGLU+FP8 fusion、Expert-Contiguous、Seq-Pack 等上游能力为个人原创。

## Change Log

- 2026-08-01: Initialized resume and project material.
- 2026-08-01: 录入 DeepSeek V4 FP8 MUSA 项目，建立贡献边界、核心故事、问题映射和表达风险。
- 2026-08-01: 将 DeepSeek V4 第 3～10 章逐章讲解、归属边界、性能口径和闭卷问题完整汇总到 `interviews/deepseekv4-relearning.md`。
