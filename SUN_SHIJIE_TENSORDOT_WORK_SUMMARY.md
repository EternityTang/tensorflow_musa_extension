# Sun Shijie 工作总结：TensorDot 算子融合

## 1. 工作概述

Sun Shijie 主要负责 TensorFlow MUSA Extension 中 TensorDot 算子融合能力的建设，完成了从计算图模式识别、融合算子实现到底层 MUSA Kernel 和测试验证的完整链路。

TensorFlow 中的 `tf.tensordot` 通常会展开为多个基础算子，例如 `Transpose`、`Reshape`、`MatMul` 和 `Reshape`。这种展开形式会带来较多中间 Tensor、额外的内存读写以及多次 Kernel Launch。针对这一问题，Sun Shijie 实现了 TensorDot 子图融合，将原始基础算子子图替换为 MUSA 后端的专用 `MusaTensorDot` 算子；在此基础上，又进一步实现了 `TensorDot + BiasAdd` 的二次融合。

## 2. 主要贡献

### 2.1 TensorDot 基础融合

通过 Grappler 图优化流程识别 TensorFlow 生成的 TensorDot 子图，并将其融合为 `MusaTensorDot`。

核心工作包括：

- 增加 TensorDot 图融合匹配规则；
- 从最终 `Reshape` 节点反向检查 `MatMul`、内部 `Reshape`、`Transpose` 及权重节点；
- 检查 TensorDot 子图节点是否属于同一命名空间，降低误匹配风险；
- 识别 `Pack`、`Prod`、`GatherV2`、`Shape` 等轴计算节点；
- 从常量节点中提取 TensorDot 的收缩轴；
- 将 `axes_a`、`axes_b` 作为融合算子属性传递给后端 Kernel；
- 将匹配规则注册到 MUSA 图优化器中。

典型转换如下：

```text
Transpose / Reshape
        ↓
      MatMul
        ↓
      Reshape
        ↓
      输出
```

转换为：

```text
MusaTensorDot(A, B, axes_a, axes_b)
```

### 2.2 TensorDot 后端算子实现

实现了 `MusaTensorDot` 算子及对应的 MUSA 执行逻辑，支持：

- `float`；
- `double`；
- `half`；
- `bfloat16`。

算子内部将任意维度、任意收缩轴的 TensorDot 统一转换为二维矩阵乘：

```text
A: [非收缩维度, 收缩维度]
B: [收缩维度, 非收缩维度]

[A_batch, A_contract]
        ×
[B_contract, B_batch]
        =
[A_batch, B_batch]
```

具体执行流程为：

1. 规范化正数轴和负数轴；
2. 校验 A、B 两侧收缩轴数量及维度是否匹配；
3. 构造 A、B 两侧的 permutation；
4. 对不满足目标布局的输入执行 Transpose；
5. 将输入 Reshape 为二维矩阵；
6. 调用 MUSA/muDNN MatMul；
7. 根据非收缩维度恢复最终输出形状。

同时增加了 Shape Function、输出形状推导、空 Tensor 快速返回以及运行时维度校验，保证融合算子的行为与 TensorFlow 原始 TensorDot 语义一致。

### 2.3 TensorDot + BiasAdd 二次融合

在 TensorDot 基础融合完成后，继续识别：

```text
MusaTensorDot
      ↓
    BiasAdd
```

并将其融合为：

```text
MusaTensorDotBias
```

主要工作包括：

- 增加 `MusaTensorDotBiasFusion` 图融合规则；
- 读取前一阶段 `MusaTensorDot` 保存的轴属性；
- 校验 Bias 输入节点及 Bias 形状；
- 增加 `MusaTensorDotBias` 算子注册和 Kernel 实现；
- 调用带 Bias 的矩阵乘接口，直接完成 MatMul 和 BiasAdd；
- 避免 TensorDot 输出中间结果写回后再次执行独立 BiasAdd。

相比原始执行方式，该方案可以减少一次独立算子调度，并减少中间结果的额外读写和全量遍历。

## 3. 技术难点与解决方案

### 3.1 TensorFlow 子图结构存在变化

TensorDot 展开后的节点可能包含 `Identity`、`ReadVariableOp`、`Const`、`Reshape`、`Transpose`、`GatherV2`、`Pack`、`Prod`、`ConcatV2` 和 `Shape` 等节点。不同模型、TensorFlow 版本及优化阶段可能产生不同的节点组合。

对此，融合规则没有仅依赖固定节点名称，而是综合使用：

- 算子类型；
- 节点前驱关系；
- TensorDot 子图命名空间；
- 常量属性；
- 输出端口和控制依赖；
- 权重节点类型。

同时支持通过 `Identity` 链查找最终的 `Const` 或目标算子，提升了模式匹配的兼容性。

### 3.2 任意轴 TensorDot 到矩阵乘的映射

TensorDot 的收缩轴可能位于任意位置，也可能包含多个轴，并且 TensorFlow 支持使用负数表示轴。后端 MatMul 则要求输入满足固定的二维布局。

通过构造：

```text
A permutation = A 的非收缩轴 + A 的收缩轴
B permutation = B 的收缩轴 + B 的非收缩轴
```

将通用 TensorDot 统一映射为标准矩阵乘，同时保持输出维度顺序与 TensorFlow 原始实现一致。对于已经满足目标布局的输入，则跳过不必要的 Transpose。

### 3.3 多阶段融合的执行顺序

Bias 融合依赖 TensorDot 融合的输出，因此采用分阶段融合：

```text
原始 TensorDot 子图
        ↓
   MusaTensorDot
        ↓
      BiasAdd
        ↓
 MusaTensorDotBias
```

这样既复用了前一阶段提取的轴信息，也避免在一个复杂匹配规则中同时处理所有情况。

## 4. 测试与验证

该项工作同步增加了完整的融合测试和子图样例，包括：

- TensorDot 融合测试：`test/fusion/tensordot_fusion_test.py`；
- TensorDot 子图样例：`test/tensordot_subgraph.pb`；
- TensorDot + BiasAdd 融合测试：`test/fusion/tensordot_bias_fusion_test.py`；
- TensorDot + BiasAdd 子图样例：`test/tensordot_bias_subgraph.pb`。

测试覆盖了图模式匹配、融合算子替换、算子属性传递以及融合结果正确性等环节。融合实现还增加了分步骤的 VLOG 日志，便于定位子图结构不匹配、轴信息提取失败和权重节点类型不符合等问题。

## 5. 代码提交记录

| 提交 | 内容 | 提交人 |
|---|---|---|
| `4f36df1` | `feat: tensordot fusion2 (#92)`，实现 TensorDot 基础融合、后端算子、MUSA Kernel 和测试 | Sun Shijie |
| `d17f67d` | `feat: add tensordot bias fusion (#105)`，实现 TensorDot + BiasAdd 二次融合及测试 | Sun Shijie |
| `1d8e169` | `fix tensordot fusion op (#110)`，修复 TensorDot 融合算子相关问题 | Sun Shijie |

以上三笔核心提交的 Git 作者均为 **Sun Shijie**，邮箱为 `120322880+ssjcode@users.noreply.github.com`。

## 6. 工作价值

这项工作的价值主要体现在以下几个方面：

1. **减少计算图节点数量**：将多个基础算子收敛为后端专用算子；
2. **减少中间 Tensor 和内存读写**：降低 Transpose、Reshape 及 BiasAdd 带来的额外开销；
3. **减少 Kernel Launch 次数**：将 TensorDot 相关计算集中到专用执行路径；
4. **提升 MUSA 后端适配能力**：充分利用 MUSA/muDNN 的矩阵乘及带 Bias 矩阵乘能力；
5. **支持通用 TensorDot 语义**：覆盖多维输入、任意轴、多轴收缩、负轴和多种数据类型；
6. **形成完整的图优化闭环**：打通 TensorFlow 图识别、属性传递、算子注册、Kernel 执行和测试验证。

需要说明的是，当前 Git 提交记录能够确认上述功能实现和测试内容，但未包含统一的性能测试数据。因此，答辩中如需量化收益，应补充融合前后的端到端耗时、Kernel Launch 数量、显存读写量或典型模型加速比等实测数据。

## 7. 转正答辩表述

> 在 TensorDot 算子优化方面，我主要负责了从计算图融合到 MUSA 后端 Kernel 实现的完整链路建设。针对 TensorFlow 中 `tf.tensordot` 会展开为多个 `Transpose`、`Reshape` 和 `MatMul` 基础算子，导致中间 Tensor 较多、内存搬运和 Kernel Launch 开销较大的问题，我在 Grappler 中实现了 TensorDot 子图匹配规则，将符合条件的子图替换为 `MusaTensorDot` 专用算子。算子内部通过轴规范化、必要的 Transpose 和二维化处理，将任意轴的 TensorDot 映射为 MUSA/muDNN 的矩阵乘，并恢复原始输出形状。
>
> 在此基础上，我进一步实现了 `TensorDot + BiasAdd` 二次融合，将两个算子合并为 `MusaTensorDotBias`，直接调用带 Bias 的矩阵乘接口，减少中间结果读写和额外 Kernel 调度。同时，我补充了多数据类型支持、Shape 推导、轴和维度校验、空 Tensor 处理以及融合测试，打通了从计算图识别到后端执行的完整优化链路。
