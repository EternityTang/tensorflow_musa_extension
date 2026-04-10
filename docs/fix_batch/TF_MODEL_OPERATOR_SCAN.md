# TF Model 算子扫描报告

生成时间：2026-03-26

## 说明

这份文档用于扫描 `tf_test_model/` 下面各个模型大致会覆盖到哪些算子族，方便后续做：

- 融合规则验证
- 单算子补测选型
- 整网回归入口选择
- 性能分析模型挑选

本报告使用了两类数据来源：

- `prunedGraph`：直接在 `py38` 容器环境里解析真实的 `GraphDef(.pb)`，因此这里的统计是真实图里的算子分布。
- 其他模型目录：静态扫描 `model/*.py` 中的 `tf.*` 调用和 `keras.layers.*` 使用情况，因此这里更接近“算子族覆盖范围”，而不是精确的运行时 op histogram。

需要注意：

- 大多数非 `prunedGraph` 模型是 Python/Keras 组图，不是现成冻结好的 `.pb`。
- 所以 `Dense(MatMul/BiasAdd)`、`Embedding/Gather` 这类条目是按算子族归类，而不是按 TensorFlow 最终下沉后的精确 op 名来写。
- `Dropout` 这类算子在推理模式下可能会被裁掉，运行时未必真的保留。

## prunedGraph

来源：

- `prunedGraph/graph_def.pb`

图规模：

- 总节点数：`4948`

图中主要算子分布如下：

| 算子类型 | 数量 |
| --- | ---: |
| `Const` | 2070 |
| `Identity` | 545 |
| `AddV2` | 384 |
| `StridedSlice` | 315 |
| `Mul` | 255 |
| `Select` | 144 |
| `Equal` | 132 |
| `LogicalOr` | 120 |
| `ExpandDims` | 107 |
| `Sub` | 96 |
| `MatMul` | 80 |
| `Reshape` | 76 |
| `BiasAdd` | 70 |
| `Mean` | 66 |
| `GatherV2` | 50 |
| `Prod` | 50 |
| `RealDiv` | 43 |
| `Transpose` | 40 |
| `Pack` | 40 |
| `ConcatV2` | 36 |
| `Shape` | 33 |
| `Maximum` | 31 |
| `Minimum` | 31 |
| `Sqrt` | 31 |
| `Relu` | 23 |
| `Placeholder` | 17 |
| `Neg` | 12 |
| `Erf` | 11 |
| `Fill` | 8 |
| `Sum` | 6 |
| `FusedBatchNormV3` | 4 |
| `BatchMatMulV2` | 4 |
| `Greater` | 4 |
| `StopGradient` | 4 |
| `Sigmoid` | 3 |
| `SelectV2` | 2 |
| `Cast` | 2 |
| `Abs` | 2 |
| `SplitV` | 1 |

结论：

- `prunedGraph` 非常适合做图融合和整网精度验证。
- 它已经覆盖了 `FusedBatchNormV3`、`LogicalOr`、`MatMul`、`BatchMatMulV2`、`GatherV2`，以及大量 shape 变换类算子。

## 模型覆盖总表

| 模型 | 主要算子族 |
| --- | --- |
| `deepfm` | `Embedding/Gather`、`Dense(MatMul/BiasAdd)`、`BatchNorm(FusedBatchNormV3)`、`Dropout`、`ConcatV2`、`Pack`、`Reshape`、`Square`、`Sum`、`Relu` |
| `dien` | `Embedding/Gather`、`Dense(MatMul/BiasAdd)`、`BatchNorm(FusedBatchNormV3)`、`MatMul/BatchMatMulV2`、`SequenceMask`、`Softmax`、`Sigmoid`、`ConcatV2`、`ExpandDims`、`Squeeze`、`Where/Select`、`Transpose`、`Reshape`、`Sum` |
| `dsin` | `Embedding/Gather`、`Dense(MatMul/BiasAdd)`、`BatchNorm(FusedBatchNormV3)`、`MatMul/BatchMatMulV2`、`MatrixSetDiag`、`SequenceMask`、`Softmax`、`Tanh`、`Tile`、`Split`、`Equal`、`ExpandDims`、`Mean`、`Max`、`ConcatV2`、`Transpose` |
| `fgcnn` | `Embedding/Gather`、`Dense(MatMul/BiasAdd)`、`BatchNorm(FusedBatchNormV3)`、`Dropout`、`ConcatV2`、`ExpandDims`、`Reshape`、`Pack`、`Sum`、`Relu` |
| `fwfm` | `Embedding/Gather`、`Dense(MatMul/BiasAdd)`、`BatchNorm(FusedBatchNormV3)`、`MatMul/BatchMatMulV2`、`MatrixBandPart`、`ConcatV2`、`ExpandDims`、`Reshape`、`Sum`、`Relu` |
| `onetrans` | `Embedding/Gather`、`Dense(MatMul/BiasAdd)`、`BatchNorm(FusedBatchNormV3)`、`Einsum`、`MatMul/BatchMatMulV2`、`Softmax`、`Sigmoid`、`Mean`、`Range`、`Sqrt`、`Square`、`ConcatV2`、`Transpose`、`Reshape` |
| `rankmixer` | `Embedding/Gather`、`Dense(MatMul/BiasAdd)`、`BatchNorm(FusedBatchNormV3)`、`Einsum`、`Softmax`、`Tanh`、`Pow`、`Sqrt`、`Mean`、`Split`、`ExpandDims`、`ConcatV2`、`Transpose`、`Reshape`、`Relu` |
| `tokenmixer-large` | `Embedding/Gather`、`Dense(MatMul/BiasAdd)`、`BatchNorm(FusedBatchNormV3)`、`TopKV2`、`Softmax`、`Sigmoid`、`Rsqrt`、`Pow`、`Square`、`Equal`、`ExpandDims`、`ConcatV2`、`Transpose`、`Reshape`、`Relu` |
| `wukong` | `Embedding/Gather`、`Dense(MatMul/BiasAdd)`、`BatchNorm(FusedBatchNormV3)`、`LayerNorm`、`MatMul/BatchMatMulV2`、`ConcatV2`、`Transpose`、`Reshape`、`Pack`、`Relu` |
| `xdeepfm` | `Embedding/Gather`、`Dense(MatMul/BiasAdd)`、`BatchNorm(FusedBatchNormV3)`、`Conv1D`、`Einsum`、`Flatten/Reshape`、`ConcatV2`、`Transpose`、`Pack`、`Sum`、`Relu` |

## 各模型说明

### deepfm

- 来源：`deepfm/model/deepfm.py`
- 特点：
  - 典型的 embedding lookup + dense tower 结构
  - 有显式特征交互，能覆盖 `Square`、`Sum`、`ConcatV2`
  - MLP 尾部会走 `BatchNormalization`、`Dropout`、`ReLU`

### dien

- 来源：`dien/model/dien.py`
- 特点：
  - 偏序列建模，带有 attention/mask 风格路径
  - 能覆盖 `Softmax`、`Sigmoid`、`Where/Select`、`SequenceMask`
  - 同时包含较多张量布局变换，如 `Transpose`、`Squeeze`、`ExpandDims`

### dsin

- 来源：`dsin/model/dsin.py`
- 特点：
  - 会覆盖更复杂的 session-interest 建模逻辑
  - 能打到 `MatrixSetDiag`、`Tile`、`Split`、`Tanh`、`Softmax`
  - 比一般 CTR 模型包含更多 mask 和结构变换

### fgcnn

- 来源：`fgcnn/model/fgcnn.py`
- 特点：
  - embedding + 特征生成路径
  - 覆盖面比 `dsin`、`onetrans` 更轻
  - 适合验证常见 CTR 栈是否正常工作

### fwfm

- 来源：`fwfm/model/fwfm.py`
- 特点：
  - 以 field interaction 为主
  - 能覆盖 `MatMul` + `MatrixBandPart`
  - reduction 较多，适合看交互项相关实现

### onetrans

- 来源：`onetrans/model/onetrans.py`
- 特点：
  - 偏 transformer/token mixing 风格
  - 能覆盖 `Einsum`、`MatMul`、`Softmax`、`Range`、`Mean`
  - 归一化逻辑里也会走 `Square` / `Sqrt`

### rankmixer

- 来源：`rankmixer/model/rankmixer.py`
- 特点：
  - 偏 mixing/attention 数学路径
  - 能覆盖 `Einsum`、`Softmax`、`Tanh`、`Pow`、`Sqrt`
  - reduction 和 reshape 也比较多

### tokenmixer-large

- 来源：`tokenmixer-large/model/tokenmixerlarge.py`
- 特点：
  - token mixing + 稀疏 MoE 路由
  - 能覆盖 `TopKV2`、`Softmax`、`Sigmoid`、`Equal`
  - 归一化逻辑里会走 `Mean`、`Pow`、`Square`、`Rsqrt`

### wukong

- 来源：`wukong/model/wukong.py`
- 特点：
  - interaction block 里有大量 `MatMul`
  - 带 `LayerNormalization`
  - embedding tensor 周围有较多 `Transpose` / `Reshape` / `ConcatV2`

### xdeepfm

- 来源：`xdeepfm/model/xdeepfm.py`
- 特点：
  - 有卷积风格交互分支
  - 除了常见 embedding + dense 路径外，还能覆盖 `Conv1D` 和 `Einsum`
  - 比较适合看 conv 相关 lowering 是否正常

## 如果你想看某类算子，优先选这些模型

- `FusedBatchNormV3`：
  `deepfm`、`dien`、`dsin`、`fgcnn`、`fwfm`、`onetrans`、`rankmixer`、`tokenmixer-large`、`wukong`、`xdeepfm`、`prunedGraph`
- `LogicalOr`：
  `prunedGraph`
- `LayerNorm` / RMS 风格归一化：
  `wukong`、`onetrans`、`tokenmixer-large`
- `TopKV2`：
  `tokenmixer-large`
- `MatrixSetDiag`：
  `dsin`
- `MatrixBandPart`：
  `fwfm`
- `Conv1D`：
  `xdeepfm`
- `Einsum`：
  `onetrans`、`rankmixer`、`xdeepfm`

## 使用时的注意点

- 非 `prunedGraph` 模型部分来自静态代码扫描，所以可能和最终运行时下沉后的实际 op 有差异。
- `Dense(MatMul/BiasAdd)`、`Embedding/Gather` 这里是按算子族写的，目的是帮助选模型，不是为了精确等同于 profiler 输出。
- `Dropout` 在 inference-only 路径里可能会被裁掉或退化，不一定能在最终 profile 里看见。
