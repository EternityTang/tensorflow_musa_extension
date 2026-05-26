# prunedGraph 融合算子性能影响测试报告

## 1. 测试目的

评估 TensorFlow MUSA extension 中各个融合 pattern 对 `prunedGraph` 整网推理性能的影响，并结合 `musa_fusion_ops_summary.md` 中的图融合结果，说明哪些融合算子在当前模型中实际出现、数量多少、以及它们对性能的贡献。

## 2. 背景：prunedGraph 中的融合情况

根据 `musa_fusion_ops_summary.md`，`musa_optimizer` 对该模型图执行融合后，标准 TensorFlow 算子被合并为 9 种自定义 `Musa*` 融合算子。

### 2.1 节点数量变化

| 阶段 | 节点数 | 变化 |
|---|---:|---:|
| before_fusion | 1,875 | - |
| after_fusion | 1,113 | -762 |
| final | 1,064 | -49 |
| **总计减少** | - | **-811** |

补充记录：实际开启 dump 时，观察到最开始节点数为 4,948，最终节点数为 1,023。

从融合阶段看，节点减少 762 个；再经过后续清理，额外减少 49 个节点。说明该模型中融合优化不仅替换了大量计算子图，也为后续死代码消除创造了机会。

### 2.2 当前模型中实际生成的融合算子

| 融合算子 | 数量 | 主要功能 |
|---|---:|---|
| `MusaShiftedAffineMap` | 132 | 将多个 `Mul`、`AddV2`、归一化相关算子融合为偏移仿射映射操作 |
| `MusaMatMulBiasAdd` | 37 | 将 `MatMul + BiasAdd` 融合为带偏置矩阵乘法 |
| `MusaTensorDotBias` | 25 | 将 `Tensordot + BiasAdd` 融合为带偏置张量点积 |
| `MusaLayerNorm` | 21 | 将 LayerNorm 计算链融合为单一算子 |
| `MusaGelu` | 11 | 将基于 `Erf` 的 GELU 计算链融合为单一算子 |
| `MusaNormalize` | 8 | 自定义归一化算子，包含 `epsilon`、`max_std` 等属性 |
| `MusaLinearActivation` | 8 | 将线性层与激活函数融合为单一算子 |
| `MusaPRelu` | 6 | 将 PReLU 相关子图融合为单一算子 |
| `MusaTokenMixer` | 3 | Token 混合算子，实现类似注意力机制的 token 交互 |
| **合计** | **251** | **9 种融合算子** |

### 2.3 被融合消除或大幅减少的原始 TensorFlow 算子

完全消失的算子包括：

| 算子 | 融合前数量 | 融合后数量 | 减少 |
|---|---:|---:|---:|
| `BiasAdd` | 70 | 0 | -70 |
| `GatherV2` | 50 | 0 | -50 |
| `Prod` | 50 | 0 | -50 |
| `Square` | 25 | 0 | -25 |
| `Sqrt` | 25 | 0 | -25 |
| `Minimum` | 25 | 0 | -25 |
| `Maximum` | 25 | 0 | -25 |
| `Erf` | 11 | 0 | -11 |
| `Fill` | 8 | 0 | -8 |
| `Neg` | 6 | 0 | -6 |
| `FusedBatchNormV3` | 4 | 0 | -4 |
| **小计** | **299** | **0** | **-299** |

数量大幅减少的算子包括：

| 算子 | 融合前数量 | 融合后数量 | 减少 |
|---|---:|---:|---:|
| `Mul` | 223 | 31 | -192 |
| `AddV2` | 218 | 48 | -170 |
| `MatMul` | 79 | 9 | -70 |
| `Reshape` | 76 | 12 | -64 |
| `Mean` | 54 | 4 | -50 |
| `ExpandDims` | 61 | 7 | -54 |
| `RealDiv` | 26 | 1 | -25 |
| `Relu` | 23 | 3 | -20 |
| `Sub` | 27 | 2 | -25 |
| `Pack` | 32 | 3 | -29 |
| `Shape` | 32 | 7 | -25 |
| `ConcatV2` | 36 | 11 | -25 |
| `Transpose` | 15 | 12 | -3 |

这些减少解释了为什么关闭全部融合后性能下降明显：未融合时需要执行更多图节点、更多 kernel launch，并产生更多中间 tensor 读写。

## 3. 测试方法

测试目录：

```bash
/workspace/tf_test_model/inference/prunedGraph
```

基础测试命令：

```bash
python run_inference.py --device musa --batch-size 100 --infer-iters 500 --warmup-iters 20
```

由于当前 `run_inference.py` 没有直接暴露禁用单个融合 pattern 的命令行参数，本次测试使用临时 wrapper 向 `ConfigProto` 注入 `disabled_fusion_patterns`，不修改项目源码。

原始测试输出：

```text
/tmp/prunedgraph_fusion_bench/summary.tsv
/tmp/prunedgraph_fusion_bench/*.log
```

## 4. 指标说明

### 4.1 latency

表中的 `avg ms`、`P50 ms`、`P95 ms`、`P99 ms` 表示一次 `sess.run()` 的耗时，单位为毫秒。

本次测试使用：

```bash
--batch-size 100
```

也就是一次 `sess.run()` 处理 100 条输入样本。

### 4.2 samples/s

`samples/s` 表示每秒处理的样本数，也就是吞吐量。

脚本中的计算方式为：

```python
throughput = 1000 / avg_time_ms * batch_size
```

例如 baseline：

```text
avg_time_ms = 6.3753 ms
batch_size = 100
throughput = 1000 / 6.3753 * 100 ≈ 15685 samples/s
```

这里的 sample 指 batch 中的一条模型输入。当前脚本使用 mock 输入数据，因此这里的 sample 是一条 mock 出来的模型输入，用于评估整网推理性能。

## 5. 统计口径

- `baseline`：默认开启 MUSA graph optimizer 和所有可用融合。
- `disable all fusion`：设置 `disabled_fusion_patterns = "all"`。
- 单项测试：每次只禁用一个 fusion pattern。
- `avg Δ` / `P50 Δ` / `P95 Δ` / `throughput Δ` 均相对 baseline 计算。
- 延迟变化为正数表示变慢。
- 吞吐变化为负数表示变慢。

## 6. 基线与禁用全部融合

| case | avg ms | P50 ms | P95 ms | P99 ms | throughput |
|---|---:|---:|---:|---:|---:|
| baseline | 6.3753 | 6.2497 | 7.0883 | 7.3604 | 15685.42 samples/s |
| disable all fusion | 12.0656 | 12.0344 | 13.2662 | 14.0729 | 8288.04 samples/s |

禁用全部融合后：

- 平均延迟：+89.26%
- P50 延迟：+92.56%
- P95 延迟：+87.16%
- 吞吐：-47.16%

结论：`prunedGraph` 整网推理性能明显依赖融合优化。

该结果也和 `musa_fusion_ops_summary.md` 中记录的 S3000 测试趋势一致：

```text
开启 fusion：约 20 ms
不开启 fusion：约 39 ms
```

即关闭融合后耗时接近翻倍。

## 7. fusion pattern 与实际融合算子的对应关系

单项禁用接口使用的是 fusion pattern 名称，而不是最终图中的 `Musa*` op 名称。两者大致对应如下：

| fusion pattern | 生成/影响的主要融合算子 | 当前模型中该算子数量 | 说明 |
|---|---|---:|---|
| `MusaShiftedAffineMapFusion` | `MusaShiftedAffineMap` | 132 | 当前模型中数量最多的融合算子 |
| `MatMulBiasAddFusion` | `MusaMatMulBiasAdd` | 37 | MatMul + BiasAdd 融合 |
| `MusaTensorDotBiasFusion` | `MusaTensorDotBias` | 25 | Tensordot + BiasAdd 融合 |
| `MusaLayerNormFusion` | `MusaLayerNorm` | 21 | LayerNorm 计算链融合 |
| `MusaGeluFusion` | `MusaGelu` | 11 | GELU 计算链融合 |
| `MusaNormalizeFusion` | `MusaNormalize` | 8 | 自定义 normalize 融合 |
| `LinearActivationFusion` | `MusaLinearActivation` | 8 | Linear + activation 融合 |
| `MusaPReluFusion` | `MusaPRelu` | 6 | PReLU 子图融合 |
| `MusaTokenMixerFusion` | `MusaTokenMixer` | 3 | Token mixer 融合 |
| `MusaTensorDotFusion` | `MusaTensorDot` | 未在 summary 中单列 | 可能命中未统计口径中的 TensorDot 融合，或影响 TensorDot 相关融合链 |
| `MusaFuseLayerNormV2Fusion` | LayerNorm V2 相关融合 | 未在 summary 中单列 | 需要结合 graph dump 确认实际生成节点 |
| `MusaPlnCascadeFusion` | `MusaPlnCascade` | 未在 summary 中单列 | 当前 summary 未记录该 op 数量 |
| `MusaPlnCascadeBlockFusion` | `MusaPlnCascadeBlock` | 未在 summary 中单列 | 当前 summary 未记录该 op 数量 |
| `MusaClipFusion` | `MusaClip` | 未在 summary 中单列 | 当前 summary 未记录该 op 数量 |
| `MusaReshapeMatMulFusion` | `MusaReshapeMatMul` | 未在 summary 中单列 | 当前 summary 未记录该 op 数量 |
| `ConcatMatMulFusion` | `MusaConcatMatMul` | 未在 summary 中单列 | 当前 summary 未记录该 op 数量 |
| `BiasAddReluMatMulFusion` | `MusaBiasAddReluMatMul` | 未在 summary 中单列 | 当前 summary 未记录该 op 数量 |
| `MusaSigmoidCalibrationFusion` | `MusaSigmoidCalibration` | 未在 summary 中单列 | 当前 summary 未记录该 op 数量 |

注意：某些 pattern 禁用后，可能不只是少生成一个对应 op，也可能改变其它 fusion pattern 的匹配顺序。因此单项禁用结果不能简单等同于“这个 op 自身的 isolated 性能收益”。

## 8. 单个融合 pattern 禁用结果

| disabled pattern | avg ms | avg Δ | P50 ms | P50 Δ | P95 ms | P95 Δ | throughput | throughput Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `MusaTensorDotFusion` | 7.7949 | +22.27% | 7.7032 | +23.26% | 9.4319 | +33.06% | 12828.86 | -18.21% |
| `MusaPlnCascadeBlockFusion` | 7.0809 | +11.07% | 6.8951 | +10.33% | 8.7284 | +23.14% | 14122.59 | -9.96% |
| `MusaNormalizeFusion` | 6.7506 | +5.89% | 6.4056 | +2.49% | 8.0380 | +13.40% | 14813.57 | -5.56% |
| `MusaFuseLayerNormV2Fusion` | 6.7117 | +5.28% | 6.6472 | +6.36% | 7.0882 | -0.00% | 14899.34 | -5.01% |
| `MusaClipFusion` | 6.6161 | +3.78% | 6.5885 | +5.42% | 6.8265 | -3.69% | 15114.56 | -3.64% |
| `MusaReshapeMatMulFusion` | 6.4964 | +1.90% | 6.4906 | +3.85% | 7.6502 | +7.93% | 15393.25 | -1.86% |
| `MusaPlnCascadeFusion` | 6.4678 | +1.45% | 6.3828 | +2.13% | 6.9164 | -2.43% | 15461.30 | -1.43% |
| `MatMulBiasAddFusion` | 6.4252 | +0.78% | 6.1924 | -0.92% | 7.7931 | +9.94% | 15563.69 | -0.78% |
| `MusaGeluFusion` | 6.1959 | -2.81% | 6.0499 | -3.20% | 7.1950 | +1.51% | 16139.67 | +2.90% |
| `ConcatMatMulFusion` | 6.1975 | -2.79% | 5.9316 | -5.09% | 7.5510 | +6.53% | 16135.52 | +2.87% |
| `MusaTokenMixerFusion` | 6.2005 | -2.74% | 6.0713 | -2.85% | 7.7147 | +8.84% | 16127.64 | +2.82% |
| `MusaShiftedAffineMapFusion` | 5.9300 | -6.98% | 5.8720 | -6.04% | 6.4563 | -8.92% | 16863.31 | +7.51% |
| `MusaTensorDotBiasFusion` | 5.8848 | -7.69% | 5.8439 | -6.49% | 6.1780 | -12.84% | 16992.97 | +8.34% |
| `BiasAddReluMatMulFusion` | 5.8788 | -7.79% | 5.8112 | -7.02% | 6.1746 | -12.89% | 17010.36 | +8.45% |
| `MusaLayerNormFusion` | 5.8289 | -8.57% | 5.7822 | -7.48% | 6.1684 | -12.98% | 17155.89 | +9.37% |
| `MusaSigmoidCalibrationFusion` | 5.7758 | -9.40% | 5.7427 | -8.11% | 6.0446 | -14.72% | 17313.75 | +10.38% |
| `
` | 5.7648 | -9.58% | 5.6756 | -9.19% | 6.0465 | -14.70% | 17346.62 | +10.59% |
| `MusaPReluFusion` | 5.6139 | -11.94% | 5.5981 | -10.43% | 5.7627 | -18.70% | 17812.86 | +13.56% |

## 9. 结果解读

### 9.1 明确提升整网性能的融合

以下 fusion pattern 禁用后性能明显下降，说明它们对当前 `prunedGraph` 模型有正向收益：

1. `MusaTensorDotFusion`
   - 禁用后平均延迟 +22.27%
   - 吞吐 -18.21%
   - 是本轮测试中影响最大的单个融合。

2. `MusaPlnCascadeBlockFusion`
   - 禁用后平均延迟 +11.07%
   - 吞吐 -9.96%

3. `MusaNormalizeFusion`
   - 当前模型中 `MusaNormalize` 数量为 8
   - 禁用后平均延迟 +5.89%
   - 吞吐 -5.56%

4. `MusaFuseLayerNormV2Fusion`
   - 禁用后平均延迟 +5.28%
   - 吞吐 -5.01%

这些融合对性能的贡献可能来自：减少 kernel launch、减少中间 tensor 显存读写、缩短 TensorFlow graph 执行路径，以及使用专门优化过的融合 kernel。

### 9.2 数量很多但单项禁用未明显变慢的融合

`MusaShiftedAffineMap` 在 summary 中数量最多，共 132 个。但本轮禁用 `MusaShiftedAffineMapFusion` 后平均延迟反而低于 baseline。

这类现象说明：

- 融合算子数量多，不一定等价于单项性能贡献最大；
- 该 pattern 可能改变其它融合的匹配机会；
- 该融合 kernel 在当前 shape 上可能不是瓶颈；
- 单轮 benchmark 可能存在抖动。

因此，融合收益需要结合数量、实际命中、执行耗时和多轮稳定性一起判断。

### 9.3 单项禁用后本轮反而更快的融合

以下 pattern 在本轮单次测试中，禁用后平均延迟低于 baseline：

- `MusaPReluFusion`
- `LinearActivationFusion`
- `MusaSigmoidCalibrationFusion`
- `MusaLayerNormFusion`
- `BiasAddReluMatMulFusion`
- `MusaTensorDotBiasFusion`
- `MusaShiftedAffineMapFusion`

这不一定表示这些融合真实负收益，可能原因包括：

- benchmark 单轮结果存在抖动；
- 该 pattern 在当前 `prunedGraph` 中未命中或命中次数很少；
- 禁用某个 pattern 后改变了其它融合 pattern 的匹配顺序；
- 当前融合 kernel 在该图上的收益不稳定；
- baseline 与单项测试不是同一个进程内连续切换，运行环境可能存在频率、缓存、调度波动。

需要通过多轮重复和 graph dump 进一步确认。

## 10. 总结

1. 从整网角度看，融合优化收益非常明显：禁用全部融合后平均延迟从 6.3753 ms 增加到 12.0656 ms，吞吐从 15685.42 samples/s 降到 8288.04 samples/s。
2. 从图结构看，融合阶段将大量 TensorFlow 原始算子替换为 251 个 `Musa*` 融合算子，并减少了数百个节点。
3. 本轮单项禁用中，`MusaTensorDotFusion`、`MusaPlnCascadeBlockFusion`、`MusaNormalizeFusion`、`MusaFuseLayerNormV2Fusion` 对整网性能影响最明显。
4. 某些融合数量很多但单项禁用未表现出明显收益，例如 `MusaShiftedAffineMapFusion`，需要结合实际 graph dump 和多轮 benchmark 判断。
5. 单项禁用测试只能反映“禁用该 pattern 后整张图最终性能变化”，不能完全等价于该融合 kernel 的 isolated 性能。

## 11. 建议后续验证

1. 对所有单项禁用测试重复 3-5 轮，取中位数或均值再判断。
2. 开启 graph dump，对比 baseline 与单项禁用后的实际融合节点数量。
3. 优先分析 `MusaTensorDotFusion` 和 `MusaPlnCascadeBlockFusion`，它们对整网性能影响最大。
4. 对“禁用后更快”的 pattern 单独做稳定性测试，确认是否为抖动、未命中或真实负收益。
5. 对 summary 中数量最多的 `MusaShiftedAffineMap` 做专项分析：统计实际耗时占比，确认其数量多但单项禁用表现反向的原因。
6. 若要做更精确的归因，建议增加 profile 数据，例如 kernel launch 次数、各融合 op 执行时间、显存读写量。