# LayerNorm 与 Linear 融合工作总结

## 1. 工作概述

本阶段围绕深度学习模型中常见的归一化和线性投影计算，完成了两类图融合优化：

1. **LayerNorm 融合**：将 TensorFlow 展开的 LayerNorm 基础算子子图融合为 MUSA 后端的 `MusaLayerNorm`；
2. **Linear/LinearRelu 融合**：将 Linear 相关的 `MatMul`、`BiasAdd`、`Relu` 等算子融合为 MUSA 后端的专用算子，并进一步切换到 muDNN epilogue 路径。

这两类优化针对 Transformer、视觉模型和其他神经网络中的高频计算模式，减少了算子调度次数、中间 Tensor 读写和显存带宽消耗，并提升了 MUSA 后端对典型模型结构的适配能力。

需要特别说明：当前仓库中没有发现一个名称明确为 `LayerNormLinear` 或 `MusaLayerNormLinear` 的单独融合算子。这里的工作是 **LayerNorm 融合**和**Linear/LinearRelu 融合**两条相关但独立的优化链路。

---

## 2. LayerNorm 融合

### 2.1 背景问题

TensorFlow 中的 LayerNorm 可能由多个基础算子展开实现，典型计算包括：

```text
x
 ↓
均值 / 方差计算
 ↓
Normalize
 ↓
Mul(gamma)
 ↓
Add(beta)
 ↓
输出
```

如果直接执行展开后的子图，会产生多个 Kernel Launch，并可能产生归一化结果、均值、方差等中间数据。对于 Transformer 等模型，LayerNorm 位于大量 Linear、Attention 和残差结构之间，额外调度和内存访问会反复累积。

### 2.2 融合方案

通过 Grappler 图优化阶段识别 LayerNorm 子图，将基础算子替换成：

```text
MusaLayerNorm(x, gamma, beta, epsilon)
```

主要匹配逻辑包括：

- 从最终的 `AddV2` 节点进入匹配；
- 检查 `AddV2` 是否对应 beta 加法；
- 向前检查 `Mul` 是否对应 gamma 缩放；
- 检查归一化节点和原始输入；
- 通过节点名称前缀确认各节点属于同一个 LayerNorm 子图；
- 检查 gamma、beta 是否为合法的常量或权重节点；
- 检查是否存在共享节点或分叉，避免破坏其他计算路径；
- 删除已经被融合的基础节点，并保留输出节点名称和外部消费者关系。

融合后，原始多算子子图被替换为单个 MUSA 专用 LayerNorm 算子，从而将归一化计算集中到 MUSA/muDNN 的执行路径中。

### 2.3 LayerNorm V2 融合

针对另一种 TensorFlow 生成的 LayerNorm 图结构，又增加了 `MusaFuseLayerNormV2Fusion`。该模式重点处理 `AddV2` 入口，并将匹配到的子图替换为 `MusaLayerNorm`。

该实现补充了：

- LayerNorm V2 子图匹配；
- gamma、beta 和 epsilon 属性传递；
- 融合节点替换和原节点清理；
- 与已有 LayerNorm 融合规则的兼容；
- 融合测试和 AddV2 相关算子测试。

### 2.4 后端能力

当前 `MusaLayerNorm` 后端 Kernel 支持：

- `float`；
- `half`；
- `bfloat16`。

算子调用 MUSA/muDNN LayerNorm 接口执行归一化，并完成 gamma、beta 参数应用。通过算子注册、Shape 处理以及图融合规则，形成了从 TensorFlow 图到 MUSA Kernel 的完整执行路径。

---

## 3. Linear/LinearRelu 融合

### 3.1 背景问题

模型中的 Linear 层通常会展开为：

```text
MatMul
  ↓
BiasAdd
  ↓
Relu
  ↓
输出
```

或在图中表现为带有 `AddV2`、`MatMul` 和激活算子的组合。如果这些算子分别执行，会产生：

- 多次 Kernel Launch；
- MatMul 输出写回显存；
- BiasAdd 再次读取和写回；
- Relu 再次读取和写回；
- 额外的中间 Tensor 管理开销。

### 3.2 初始 LinearReluFusion

初始版本实现了 `LinearReluFusion`，识别 Linear + Relu 计算模式，并替换为 MUSA 后端专用算子。

典型转换为：

```text
Linear / MatMul + BiasAdd + Relu
              ↓
        MusaLinearRelu
```

主要工作包括：

- 增加 LinearRelu 图融合匹配规则；
- 增加 `MusaLinearRelu` 算子注册；
- 增加 MUSA Kernel 实现；
- 处理 MatMul 的输入、权重、Bias 以及激活关系；
- 增加融合测试，验证图替换和结果正确性。

### 3.3 融合逻辑优化

后续对融合逻辑进行了优化，重点解决原始节点清理和图结构维护问题：

- 删除已经被融合的原始 Linear/Relu 节点；
- 保留融合节点的原始输出名称，减少下游节点改写；
- 维护外部消费者输入关系；
- 避免共享节点被错误删除；
- 调整图优化器中的融合执行顺序和注册逻辑。

同时补充了 `MatMul + BiasAdd` 相关融合支持，为后续的 Linear 激活融合提供基础。

### 3.4 切换到 muDNN epilogue

后续将 LinearRelu 的实现切换到 muDNN epilogue 路径。原先可能采用独立的矩阵乘、Bias 和激活流程，切换后由底层矩阵乘接口直接在计算尾部完成 Bias 和激活：

```text
MatMul
  + BiasAdd
  + Relu epilogue
        ↓
一次融合执行
```

这样可以：

- 减少中间矩阵写回；
- 减少全量结果再次读取；
- 减少独立 BiasAdd 和 Relu 的 Kernel Launch；
- 更好地利用 muDNN 对矩阵乘后处理的优化；
- 降低显存带宽压力和算子调度开销。

相关代码从 `linear_relu_fusion` 逐步演进为更通用的 `linear_activation_fusion`，为后续支持其他激活函数留下扩展空间。

---

## 4. 两类融合的共同技术难点

### 4.1 计算图结构不固定

TensorFlow 不同版本、不同模型和不同优化阶段可能生成不同的节点组合。因此融合规则不能只依赖固定节点名称，而需要结合：

- 算子类型；
- 节点输入输出关系；
- 节点名称前缀；
- 常量和权重节点类型；
- 节点是否存在共享消费者；
- MatMul 的属性和输入位置。

### 4.2 融合后的图必须保持拓扑和语义一致

在删除基础节点时，需要同步维护：

- 融合节点名称；
- 外部消费者的输入；
- 控制依赖；
- 共享节点；
- 输出端口；
- 原有数据类型和算子属性。

LayerNorm 融合中特别增加了共享节点检测，避免因为错误删除公共节点而影响图中其他计算路径。

### 4.3 融合顺序影响最终结果

LayerNorm、MatMul、BiasAdd、Linear 和激活函数之间可能存在多个候选融合规则。因此需要在图优化器中合理安排注册顺序和优先级，优先使用更大粒度的融合，避免前一个通用规则先执行后破坏更完整的融合模式。

---

## 5. 测试与验证

相关测试文件包括：

- `test/fusion/layernorm_fusion_test.py`；
- `test/fusion/fuselayernormv2_fusion_test.py`；
- `test/fusion/linear_relu_fusion_test.py`；
- LayerNorm 相关算子测试；
- Max、Add、AddN 等 LayerNorm 子图依赖算子测试。

测试重点覆盖：

- LayerNorm 子图是否能够正确匹配；
- LayerNorm V2 模式是否能够正确替换；
- LinearRelu 子图是否能够正确融合；
- 融合节点属性和输入是否正确；
- 原始节点是否被安全删除；
- 共享节点和分叉场景是否不会误删；
- 融合前后输出结果是否一致；
- 多种数据类型和典型输入形状的正确性。

当前提交记录中未发现与这两类融合统一对应的性能数据。若答辩中需要量化收益，应补充融合前后的端到端耗时、Kernel Launch 数量、显存读写量及典型模型加速比。

---

## 6. 主要提交与贡献者

### LayerNorm 相关

| 提交 | 内容 | 提交人 |
|---|---|---|
| `cc4f8d5` | `feat: Layernorm fusion pattern 1 (#130)`，实现 LayerNorm 融合模式 | Sun Shijie |
| `c791f4d` | `Newaloysha pr2 (#136)`，修复 LayerNorm 匹配并增加 `FuseLayerNormV2` | Aloyshaaaa |
| `ede837c` | `add layernormgrad ops`，增加 LayerNorm 反向算子 | eilan |
| `6b356b1` | `fix layernormgrad`，修复 LayerNormGrad | eilan |

### Linear/LinearRelu 相关

| 提交 | 内容 | 提交人 |
|---|---|---|
| `0c6c95d` | `feat: LinearReluFusion (#102)`，实现初始 LinearRelu 融合、Kernel 和测试 | Zheqin Yin |
| `061ba7b` | `feat: Optimizing the fuse logi, delete the origin node of linear_relu op (#162)`，优化融合逻辑和原始节点删除 | awexxxx |
| `700f219` | `feat: switch linear relu fusion to mudnn epilogue (#203)`，切换到 muDNN epilogue | EternityTang |
| `7890733` | `switch linear relu fusion to mudnn epilogue`，相关实现调整 | eilan |
| `ad7754f` | `feat: switch linear relu fusion to mudnn epilogue`，相关实现调整 | eilan |

因此，如果答辩需要说明个人贡献，应根据实际参与情况区分：

- **LayerNorm 融合初始模式**：Sun Shijie；
- **LayerNorm V2 和匹配完善**：Aloyshaaaa；
- **LayerNormGrad**：eilan；
- **LinearRelu 初始融合**：Zheqin Yin；
- **Linear 融合逻辑优化**：awexxxx；
- **muDNN epilogue 切换**：EternityTang、eilan。

---

## 7. 工作价值

1. 将 LayerNorm、Linear、BiasAdd 和激活等高频基础算子组合转换为后端专用融合算子；
2. 减少 Kernel Launch 和中间 Tensor 数量；
3. 降低显存中间结果读写以及全量数据遍历开销；
4. 通过 muDNN epilogue 充分利用矩阵乘后处理能力；
5. 提升 Transformer 等模型中归一化和线性投影模块的执行效率；
6. 完善 MUSA 图优化器的融合规则、节点生命周期管理和测试体系；
7. 建立从 TensorFlow 子图识别到 MUSA/muDNN 后端执行的完整优化闭环。

---

## 8. 转正答辩表述

> 在 LayerNorm 和 Linear 相关优化方面，我参与了两类高频计算模式的图融合建设。针对 TensorFlow 将 LayerNorm 展开为多个归一化、乘法和加法基础算子的问题，通过 Grappler 识别 LayerNorm 子图，并将其替换为 MUSA 后端的 `MusaLayerNorm`，同时处理了 LayerNorm V2 的不同图结构、参数传递、共享节点检测和原始节点安全删除。针对 Linear 模块中 `MatMul`、`BiasAdd` 和 `Relu` 分别执行带来的调度和中间结果读写开销，实现并完善了 LinearRelu 融合，随后将其切换到 muDNN epilogue 路径，使矩阵乘、Bias 和激活能够在一次融合执行中完成。通过这两类优化，减少了算子调度和显存访问，并完善了 MUSA 图优化器从模式匹配、算子替换到后端 Kernel 执行的完整链路。
