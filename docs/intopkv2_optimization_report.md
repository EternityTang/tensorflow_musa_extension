# InTopKV2 算子优化记录

## 背景

`InTopKV2` 用于判断每个样本的目标类别是否位于预测分数的 top-k 内。本次测试场景主要关注大 `num_classes` 输入，例如：

```text
batch_size = 1024 / 8192 / 32768
num_classes = 10000
k = 5
target_dtype = int32
predictions dtype = float32
```

优化前 profiling 发现该算子利用率很低，尤其在 `batch_size=1024, num_classes=10000` 时：

```text
kernel 带宽利用率约 1.59%
算力利用率约 0.0008% ~ 0.0016%（按 800 TFLOPS 峰值估算）
```

主要原因是原始 kernel 的并行度不足。

## 原始实现

文件：

```text
musa_ext/kernels/math/musa_intopkv2_kernel.mu
```

原始实现的核心逻辑是：

```cpp
const int row = blockIdx.x * blockDim.x + threadIdx.x;
```

也就是：

```text
1 个线程处理 1 个 batch row
每个线程串行扫描整行 num_classes
```

对于 `batch_size=1024, num_classes=10000`：

```text
只有约 1024 个有效线程在工作
每个线程串行读取并比较 10000 个 class score
```

这导致 GPU 并行度无法充分展开，带宽和算力都没有被喂满。

## 优化方案

将 kernel 从：

```text
one thread per row
```

改为：

```text
one block per row
```

优化后的每个 row 由一个 block 内的 256 个线程协作处理：

```text
blockIdx.x 对应 row
threadIdx.x 以 stride=BLOCK_SIZE 扫描 num_classes
每个线程统计局部 count_higher
block 内做 reduction
thread 0 写 output[row]
```

核心思路：

```cpp
const int row = blockIdx.x;

for (int i = threadIdx.x; i < num_classes; i += BLOCK_SIZE) {
  const float score = LoadAsFloat(&row_predictions[i]);
  count_higher += score > target_score;
}
```

然后使用 block reduction 汇总 `count_higher`：

```text
warp 内使用 __shfl_xor_sync reduction
warp 间使用 shared memory 汇总
```

该 reduction 风格参考了仓库内已有实现：

```text
musa_ext/kernels/nn/musa_rmsnorm_kernel.mu
musa_ext/kernels/nn/musa_softmax_small_lastdim_kernel.mu
```

同时保留 TensorFlow `InTopKV2` 的语义：

```text
使用严格比较 score > target_score
不使用 >=
因此 ties 行为保持不变
```

还增加了两个快速路径：

```text
k == 0           -> output 全 false
k == num_classes -> output 全 true
```

## 修改文件

主要修改：

```text
musa_ext/kernels/math/musa_intopkv2_kernel.mu
```

新增/调整内容：

```text
1. 新增 kWarpSize 和 kInTopKBlockSize
2. 新增 int 类型 BlockReduceSum
3. 将 InTopKKernel 改成 block-per-row 并行扫描
4. 新增 SetBoolKernel 处理 k==0 和 k==num_classes
5. LaunchInTopKV2Int32 / LaunchInTopKV2Int64 改为 grid=batch_size, block=256
```

未修改 op 注册和输入校验逻辑：

```text
musa_ext/kernels/math/musa_intopkv2_op.cc
```

## 构建与正确性验证

构建命令：

```bash
cmake --build build -j$(nproc)
```

构建结果：

```text
[  0%] Compiling MUSA kernel: musa_intopkv2_kernel.mu
[  1%] Linking CXX shared library libmusa_plugin.so
Plugin is located at: /workspace/tensorflow_musa_extension/build/libmusa_plugin.so
Build completed successfully!
[100%] Built target musa_plugin
```

正确性测试命令：

```bash
cd test/ops && python intopkv2_op_test.py
```

测试结果：

```text
Ran 11 tests in 1.384s
OK
```

通过的测试包括：

```text
testInTopKV2BasicInt32
testInTopKV2BasicInt64
testInTopKV2ExactMatch
testInTopKV2KEquals1
testInTopKV2KEqualsAll
testInTopKV2KZero
testInTopKV2LargeBatch
testInTopKV2LargeClasses
testInTopKV2MixedResults
testInTopKV2RandomData
testInTopKV2SmallBatch
```

## Benchmark 命令

普通 benchmark：

```bash
python /workspace/tensorflow_musa_extension/test/ops/intopkv2_benchmark.py \
  --batch-sizes 1024,8192,32768 \
  --num-classes 10000 \
  --k-values 5 \
  --measure-iters 100
```

msys profile：

```bash
/workspace/tensorflow_musa_extension/test/ops/run_intopkv2_msys_profile.sh \
  --batch-sizes 8192 \
  --num-classes 10000 \
  --k-values 5 \
  --measure-iters 100
```

## 优化前性能数据

### batch=1024, classes=10000

```text
wall avg = 1.9134 ms
wall throughput = 535,185 samples/s
InTopKKernel avg = 1.606043 ms
```

估算：

```text
有效带宽 = 25.51 GB/s
带宽利用率 = 25.51 / 1600 ≈ 1.59%
算力利用率 ≈ 0.0008% ~ 0.0016%（按 800 TFLOPS）
```

### batch=8192, classes=10000

```text
wall avg = 2.0403 ms
wall throughput = 4,015,158 samples/s
InTopKKernel avg = 1.724354 ms
```

估算：

```text
有效带宽 = 190.08 GB/s
带宽利用率 = 190.08 / 1600 ≈ 11.88%
算力利用率 ≈ 0.0059% ~ 0.0119%（按 800 TFLOPS）
```

### batch=32768, classes=10000

```text
wall avg = 9.0828 ms
wall throughput = 3,607,689 samples/s
InTopKKernel avg = 8.758854 ms
```

估算：

```text
有效带宽 = 149.68 GB/s
带宽利用率 = 149.68 / 1600 ≈ 9.36%
算力利用率 ≈ 0.00468% ~ 0.00935%（按 800 TFLOPS）
```

## 优化后性能数据

普通 benchmark 输出：

```text
mode=op batch=1024 classes=10000 k=5 target=int32 avg=0.3666 ms min=0.3568 ms p50=0.3649 ms p90=0.3757 ms p99=0.3822 ms throughput=2793570.14 items/s true_count=0
mode=op batch=8192 classes=10000 k=5 target=int32 avg=0.6330 ms min=0.6157 ms p50=0.6327 ms p90=0.6384 ms p99=0.6520 ms throughput=12940660.89 items/s true_count=6
mode=op batch=32768 classes=10000 k=5 target=int32 avg=1.6240 ms min=1.6016 ms p50=1.6240 ms p90=1.6291 ms p99=1.6361 ms throughput=20177765.21 items/s true_count=14
```

msys profile，`batch=8192, classes=10000`：

```text
InTopKKernel<float, int, 256>
Avg = 350,107 ns = 0.350107 ms
Med = 346,681 ns
Min = 342,920 ns
Max = 392,920 ns
Instances = 151
```

## 前后对比

### wall latency

| batch | classes | old wall avg | new wall avg | speedup |
|---:|---:|---:|---:|---:|
| 1024 | 10000 | 1.9134 ms | 0.3666 ms | 5.22x |
| 8192 | 10000 | 2.0403 ms | 0.6330 ms | 3.22x |
| 32768 | 10000 | 9.0828 ms | 1.6240 ms | 5.59x |

### kernel latency，batch=8192

| case | InTopKKernel avg |
|---|---:|
| before | 1.724354 ms |
| after | 0.350107 ms |

kernel 级 speedup：

```text
1.724354 / 0.350107 ≈ 4.93x
```

## 优化后利用率估算

以 `batch=8192, num_classes=10000` 为例。

数据量估算：

```text
predictions = 8192 * 10000 * 4 B = 327,680,000 B
target_score extra read            =      32,768 B
targets read                       =      32,768 B
output write                       =       8,192 B
total                              ≈ 327,753,728 B
```

优化后 kernel time：

```text
0.350107 ms
```

有效带宽：

```text
327,753,728 B / 0.350107 ms ≈ 936.2 GB/s
```

带宽峰值按 1.6 TB/s：

```text
带宽利用率 = 936.2 / 1600 * 100% ≈ 58.5%
```

算力估算：

```text
elements = 8192 * 10000 = 81,920,000
```

按 1 op/element：

```text
81,920,000 / 0.350107 ms ≈ 234.0 GOPS = 0.234 TOPS
算力利用率 = 0.234 / 800 * 100% ≈ 0.029%
```

按 2 ops/element：

```text
468.0 GOPS = 0.468 TOPS
算力利用率 = 0.468 / 800 * 100% ≈ 0.059%
```

因此优化后：

```text
带宽利用率 ≈ 58.5%
算力利用率 ≈ 0.029% ~ 0.059%
```

## 结论

本次优化将 `InTopKV2` 的主要扫描逻辑从单线程串行扫描改为 block 内 256 线程并行扫描，显著提升了大 `num_classes` 场景下的并行度。

核心收益：

```text
batch=1024  wall speedup ≈ 5.22x
batch=8192  wall speedup ≈ 3.22x
batch=32768 wall speedup ≈ 5.59x
batch=8192  kernel speedup ≈ 4.93x
```

利用率变化：

```text
batch=8192 kernel 带宽利用率：约 11.88% -> 58.5%
```

算力利用率仍然很低，这是符合预期的，因为该算子主要是读取、比较和计数，不是 FMA 密集型计算。优化后它更接近带宽型 kernel，主要瓶颈从“并行度不足”转向“内存带宽和访存效率”。

## 第二轮优化：local early-exit

在 block-per-row 版本基础上，继续针对小 `k` 场景做了第二轮优化。

### 优化动机

`InTopKV2` 的输出条件是：

```text
count_higher < k
```

对于 `k=5` 这类小 k，只要某个线程在自己负责的 class 子集里已经发现 `k` 个更高分数，那么整行最终一定不在 top-k。这个线程继续扫描剩余 class 已经不会改变该线程对最终 reduction 的有效贡献。

因此可以在每个线程的局部扫描中提前停止：

```cpp
for (int i = threadIdx.x; i < num_classes && count_higher < k; i += BLOCK_SIZE) {
  const float score = LoadAsFloat(&row_predictions[i]);
  count_higher += score > target_score;
}
```

这个优化保持语义不变：

```text
1. 仍然使用严格比较 score > target_score
2. 只在局部 count_higher 已经达到 k 时停止
3. 最终 block reduction 后仍然判断 total_count < k
```

适用场景：

```text
k 较小
目标类别多数不在 top-k
score 分布较随机，较容易提前找到 k 个更高分数
```

收益较小或可能不明显的场景：

```text
目标类别经常在 top-k
k 较大
每个线程很难提前累积到 k 个更高分数
```

### 构建和安装

按以下命令重新构建并安装 wheel：

```bash
cd /workspace/tensorflow_musa_extension
./build.sh wheel
pip install dist/tensorflow_musa-0.1.0-py3-none-any.whl --no-deps --force-reinstall
```

构建结果：

```text
[SUCCESS] Wheel package built successfully!
-rw-r--r-- 1 root root 6.7M May 22 17:07 dist/tensorflow_musa-0.1.0-py3-none-any.whl
Successfully installed tensorflow-musa-0.1.0
```

### 正确性验证

测试命令：

```bash
cd /workspace/tensorflow_musa_extension/test/ops
python intopkv2_op_test.py
```

测试结果：

```text
Ran 11 tests in 1.394s
OK
```

### 第二轮优化后 benchmark

测试命令：

```bash
python /workspace/tensorflow_musa_extension/test/ops/intopkv2_benchmark.py \
  --batch-sizes 1024,8192,32768 \
  --num-classes 10000 \
  --k-values 5 \
  --measure-iters 100
```

输出：

```text
mode=op batch=1024 classes=10000 k=5 target=int32 avg=0.3177 ms min=0.2931 ms p50=0.3181 ms p90=0.3234 ms p99=0.3286 ms throughput=3223069.40 items/s true_count=0
mode=op batch=8192 classes=10000 k=5 target=int32 avg=0.5235 ms min=0.5094 ms p50=0.5227 ms p90=0.5312 ms p99=0.5379 ms throughput=15648998.29 items/s true_count=6
mode=op batch=32768 classes=10000 k=5 target=int32 avg=1.1924 ms min=1.1751 ms p50=1.1921 ms p90=1.1973 ms p99=1.2017 ms throughput=27481637.87 items/s true_count=14
```

### 第二轮优化前后对比

这里的“第一轮”指 block-per-row 并行 reduction 版本，“第二轮”指在其基础上加入 local early-exit。

| batch | classes | 第一轮 avg | 第二轮 avg | 第二轮相对第一轮 speedup |
|---:|---:|---:|---:|---:|
| 1024 | 10000 | 0.3666 ms | 0.3177 ms | 1.15x |
| 8192 | 10000 | 0.6330 ms | 0.5235 ms | 1.21x |
| 32768 | 10000 | 1.6240 ms | 1.1924 ms | 1.36x |

### msys profile，batch=8192

profile 命令：

```bash
cd /workspace/tensorflow_musa_extension
./test/ops/run_intopkv2_msys_profile.sh \
  --batch-sizes 8192 \
  --num-classes 10000 \
  --k-values 5 \
  --measure-iters 100
```

输出文件：

```text
/workspace/tensorflow_musa_extension/test/ops/benchmark_results/intopkv2_20260522_171036.msys-rep
/workspace/tensorflow_musa_extension/test/ops/benchmark_results/intopkv2_20260522_171036_musa_gpu_kern_sum.csv
/workspace/tensorflow_musa_extension/test/ops/benchmark_results/intopkv2_20260522_171036_musa_kern_exec_sum.csv
/workspace/tensorflow_musa_extension/test/ops/benchmark_results/intopkv2_20260522_171036_musa_api_gpu_sum.csv
```

`musa_gpu_kern_sum.csv` 中的 InTopK kernel 数据：

```text
InTopKKernel<float, int, 256>
Avg = 243,652 ns = 0.243652 ms
Med = 240,800 ns
Min = 238,200 ns
Max = 277,361 ns
Instances = 151
Time = 94.32%
```

第一轮 block-per-row 版本的同 case kernel 时间：

```text
Avg = 350,107 ns = 0.350107 ms
```

第二轮相对第一轮 kernel speedup：

```text
350107 / 243652 ≈ 1.44x
```

相对原始实现：

```text
原始 InTopKKernel avg = 1.724354 ms
第二轮 InTopKKernel avg = 0.243652 ms
总 kernel speedup = 1.724354 / 0.243652 ≈ 7.08x
```

### 第二轮优化后的等价带宽估算

以 `batch=8192, num_classes=10000` 为例，如果仍按完整扫描数据量计算：

```text
predictions = 8192 * 10000 * 4 B = 327,680,000 B
target_score extra read            =      32,768 B
targets read                       =      32,768 B
output write                       =       8,192 B
total                              ≈ 327,753,728 B
```

第二轮 kernel 时间：

```text
0.243652 ms
```

等价有效带宽：

```text
327,753,728 B / 0.243652 ms ≈ 1345.0 GB/s
```

按峰值带宽 `1.6 TB/s = 1600 GB/s`：

```text
等价带宽利用率 = 1345.0 / 1600 * 100% ≈ 84.1%
```

注意：这是“等价带宽利用率”，不是实际硬件读带宽。因为 local early-exit 后，kernel 不再保证读取完整 `batch_size * num_classes` 个 prediction 元素，实际读取数据量会小于完整扫描数据量。

### 第二轮优化总结

第二轮 local early-exit 在第一轮并行化基础上继续带来收益：

```text
batch=1024  wall avg: 0.3666 ms -> 0.3177 ms, speedup ≈ 1.15x
batch=8192  wall avg: 0.6330 ms -> 0.5235 ms, speedup ≈ 1.21x
batch=32768 wall avg: 1.6240 ms -> 1.1924 ms, speedup ≈ 1.36x
```

从最初原始实现到第二轮优化：

```text
batch=8192 InTopKKernel avg: 1.724354 ms -> 0.243652 ms
总 kernel speedup ≈ 7.08x
```

这个优化对当前 `k=5` 的性能测试有效，且 batch 越大收益越明显。后续如果要继续优化，可以考虑按 `k`、`num_classes` 和 batch 规模做更细的 kernel dispatch，例如不同 block size 或 warp-per-row 版本。
