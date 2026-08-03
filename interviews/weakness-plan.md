# Weakness Plan

## Confirmed Weaknesses

- No confirmed weaknesses yet.

## Potential Risks

- `potential_risk`：项目覆盖面过大，容易按 Kernel 名称罗列，缺少“问题→归因→决策→实现→验证→边界”的主线。
- `potential_risk`：对 FP8 scale layout、MoE metadata、Paged Cache 坐标等正确性细节记忆不牢。
- `potential_risk`：只会复述结论，不能现场重算 Roofline、访存量和 shape。

## Project Expression Risks

- `project_expression_risk`：把官方基础能力说成个人原创。
- `project_expression_risk`：把项目口径或静态推导说成本人实测。
- `project_expression_risk`：团队整体材料尚未拆分到个人提交。
- `project_expression_risk`：对 `[待补]` 细节进行合理猜测并当作事实。

## Learning Backlog

- P0：Fused Norm-RoPE-Cache 数据流、访存账本、strided-input 测试。
- P0：Cache Store Prefill/Decode 双路径与 dispatch 边界。
- P0：MoE Quant-Scatter 的 shape、metadata、流量账和 static-cap。
- P1：TopK-512 exact/candidate、Pack8 metadata 与 Graph 安全。
- P1：Online C128 × MTP deferred-commit 状态机。
- P2：HC-head Split-K、Paged C4 Compress、Runtime Hardening。

## Practice Plan

### 第一轮：建立骨架（约 2 小时）

- 不看材料说出 30 秒项目定位和上游/MUSA/个人三层边界。
- 画出 Prefill、Decode、跨阶段三条数据流。
- 给四条核心故事各写一张六格卡：问题、瓶颈、方案、实现、验证、边界。

### 第二轮：吃透核心（每条 45～60 分钟）

- Day 1：Fused Norm-RoPE-Cache + Cache Store 双路径。
- Day 2：MoE Compact + Quant-Scatter，现场重算 AI 和 311 MB/层流量差。
- Day 3：TopK-512 + Pack8，重点解释 exactness、capacity 和 metadata。
- Day 4：Online C128 × MTP + Graph/JIT/fallback。

### 第三轮：闭卷输出

- 每条故事进行 30 秒、90 秒、5 分钟三档讲述。
- 每条连续回答五个追问，不翻材料。
- 对未知细节使用“我确认过的事实是……；这个常数的标定过程我需要回代码核实”，不临场编造。

## Completion Evidence

- 能闭卷画出四条核心数据流，并标注 tensor shape/dtype/metadata。
- 能在白板上复算 MoE GEMM 与 elementwise AI，以及 Quant-Scatter 的流量差。
- 能准确说出至少五项“上游已有、不能算个人原创”的能力。
- 能为每个性能数字补齐“来源、证据等级、缺失配置”。
- 能完成一次 15 分钟模拟项目深挖，且无归属夸大和未经证实的实测表述。
