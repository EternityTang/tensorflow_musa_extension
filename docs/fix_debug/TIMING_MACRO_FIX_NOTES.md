# Timing 宏修复说明

日期：2026-03-27

## 1. 背景

在 `tensorflow_musa_extension` 中：

- `build.sh release` 生成的 `libmusa_plugin.so` 可用于 `tf_test_model/prunedGraph` 稳定推理，长时间运行无报错。
- 切到 `build.sh debug` 后，问题只在 debug 版本出现。
- 该问题和 `MUSA_KERNEL_DEBUG` 打开的 timing 宏路径有关。

## 2. 根因判断

本次修复的重点不是算子功能逻辑，而是 timing instrumentation 本身的副作用。

原 timing 宏存在两个风险：

1. 宏在构造 `KernelTimingScope` 时会立即拿 shape/stream，即使 timing 实际没有启用，也会把额外逻辑带进每个 kernel。
2. timing 路径里会调用 `musaEventCreate/Record/Synchronize/ElapsedTime/Destroy`，但没有在这些 runtime API 前显式切到当前 `OpKernelContext` 对应的 MUSA device。
3. timing 路径对每个 stage 的 `END` 都会立刻 `musaEventSynchronize`，这会把原本异步的 stream 强制打成很多 host-side barrier；`release` 根本没有这层同步。
4. `build.sh debug` 会把 C++ 构建切成 `CMAKE_BUILD_TYPE=Debug`，也就是 `-g`/非 `-O3` 路径；这让 `debug` 和 `release` 多了一层与 timing 无关的差异。

在多卡/多线程情况下，这会导致：

- kernel 本体后续通过 `GetHandleByCtx()` 切到了正确设备；
- 但 timing event 可能先在错误的 current device 上执行；
- debug 版才会走到这条路径，因此表现为 release 正常、debug 才异常。

## 3. 本次修改

### 3.1 `tensorflow_musa_extension/musa_ext/kernels/utils_op.h`

新增：

- `GetMusaDeviceIdByCtx(OpKernelContext*)`

用途：

- 统一从 `OpKernelContext` 提取当前 kernel 所属的 MUSA device id。
- 避免 timing 宏和正常 kernel 路径各自用不同方式拿 device。

同时调整：

- `GetHandleByCtx()` 改为复用 `GetMusaDeviceIdByCtx()`。

### 3.2 `tensorflow_musa_extension/musa_ext/utils/logging.h`

修复点：

1. 为 `KernelTimingScope` 增加基于 `OpKernelContext*` 的构造路径。
2. 只有 timing 真启用且命中过滤器时，才去构造 input shape / stream。
3. 给 `KernelTimingScope` 保存 `device_id_`。
4. 在以下 API 前统一调用 `CachedMusaSetDevice(device_id_)`：
   - `musaEventCreate`
   - `musaEventRecord`
   - `musaEventSynchronize`
   - `musaEventElapsedTime`
   - `musaEventDestroy`
5. `MUSA_KERNEL_TIMING_GUARD_*` 宏改为把 `ctx` 直接传给 `KernelTimingScope`，不再在宏展开处提前构造 shape/stream。
6. stage 计时改为“记录 start/end event，析构时统一结算”，不再在每个 `TraceEnd()` 上立刻 `musaEventSynchronize`。
7. `level < 2` 时不再收集 stage 级 event，只保留总时长计时，避免 level 1 也引入细粒度同步和 event 开销。

### 3.3 `tensorflow_musa_extension/musa_ext/kernels/math/musa_neg_kernel.mu`

补充修正：

- 将 `int4/uint4` 改成显式 `::int4/::uint4`

原因：

- 该文件位于 `namespace tensorflow::musa` 内部。
- 在当前本地 shell 对应的 TensorFlow 2.20 头文件中，`tensorflow::int4` / `tensorflow::uint4` 已经被定义成 `ml_dtypes::int4` / `ml_dtypes::uint4`。
- 原代码直接写 `int4/uint4` 时，会优先命中 `tensorflow::int4` 这类 `ml_dtypes` 类型，而不是 MUSA 的全局向量类型。
- `ml_dtypes::int4` 不是带 `.x/.y/.z/.w` 成员的向量结构，因此会在 `musa_neg_kernel.mu` 中报错。

这不是 timing 宏根因，而是本地 TensorFlow 2.20 编译环境下额外暴露出来的命名兼容问题。

### 3.4 `tensorflow_musa_extension/musa_ext/kernels/array/musa_strided_slice_op.cc`

新增修正：

- 给 `StridedSlice<int32/int64>` 的输出路径补了完成性保护。
- 包括：
  - `is_identity` 分支中的 `musaMemcpyAsync` 返回值检查
  - `int32/int64` 输出场景下，对当前 stream 做显式 `musaStreamSynchronize`
  - 常规 `op.Run(...)` 分支结束后，对 `int32/int64` 输出场景做相同同步

修复原因：

- 新一轮 debug 长跑报错不再是 `RunOp failed`，而是：
  - `fwffm_pbp_mlp/Reshape_6`
  - `Input has 115200 elements, but target shape has 1217496214656 elements`
- 这类报错更像是 `Reshape` 的 shape 输入张量在上游被污染，而不是 `Reshape` 本身逻辑错误。
- 旧日志里最早暴露的问题点是：
  - `fwffm_pbp_mlp/strided_slice_1`
- 仓库中确实存在 MUSA 自定义 `StridedSlice`：
  - `tensorflow_musa_extension/musa_ext/kernels/array/musa_strided_slice_op.cc`
- 同时，MUSA 自定义 `Reshape` 明确要求第二个输入 `shape` 为 HostMemory。
- 结合这两点，较强的怀疑是：
  - `StridedSlice<int32/int64>` 在 shape 张量路径上虽然执行成功，但输出仍处于异步 device 流上；
  - 后续 HostMemory 消费者在某些时序下过早读取，导致偶发 shape 污染；
  - debug timing 改变执行节奏后，更容易把这个 race 暴露出来。

修复目标：

- 让 timing 宏在 debug 模式下尽量不改变 kernel 的原始执行语义。
- 避免 timing event 在错误 device 上执行。
- 避免 stage 级同步把 debug 版运行节奏改成和 release 完全不同。
- 给 shape 相关的 `StridedSlice<int32/int64>` 输出补足可见性保障，减少 `Reshape` 读到脏 shape 的概率。

## 4. 测试脚本

新增脚本：

- `tf_test_model/prunedGraph/run_musa_debug_1000.sh`

脚本行为：

- 默认 `export MUSA_ENABLE_TF32=0`
- 支持 1000/40000 轮 inference-only
- 支持整网精度对比

用法：

```bash
cd tf_test_model/prunedGraph

# 只跑 1000/40000 轮推理
./run_musa_debug_1000.sh inference

# 只跑整网精度对比
./run_musa_debug_1000.sh accuracy

# 两者都跑
./run_musa_debug_1000.sh all
```

## 5. 日志说明

### 5.1 修复前的旧报错

旧报错来自：

- `tf_test_model/prunedGraph/logs/graph_inference/2026-03-27-11.01.01_trace/main.log`

该日志中出现：

- `Internal: MUSA RunOp failed. Status: 7`
- 节点：`fwffm_pbp_mlp/strided_slice_1`

这是**修复前**的 debug 结果，不是本次补丁后的新结果。

### 5.2 修复后的新日志

修复后生成了新的日志目录：

- `tf_test_model/prunedGraph/logs/graph_inference/2026-03-27-12.17.40_trace/`
- `tf_test_model/prunedGraph/logs/graph_inference/2026-03-27-12.18.13_trace/`

从 `2026-03-27-12.17.40_trace/inference.log` 可以看到：

- 已经稳定跑过 `Inference round 500/1000`
- 日志继续推进到 `Inference round 595/1000`
- 没有出现修复前那种在 500 左右报错的记录

这至少说明：

- 本次 timing 宏修复已经明显改善了 debug 版的稳定性；
- “500 轮附近就炸”的旧现象当前没有在新日志中复现。

注意：

- 目前这两份新日志都没有完整收尾信息；
- 也就是说，这次看到的是“已明显好转”，但还不能仅凭当前日志直接下结论说 1000 轮和精度测试已经全部最终通过。

## 6. 结论

这次修改主要做了两件事：

1. 把 timing 宏的 device 上下文切换补完整。
2. 把 timing 宏改成按需启用，并移除了每个 stage 的即时同步。
3. 在当前本地 TensorFlow 2.20 环境下，补了 `musa_neg_kernel.mu` 的 `int4/uint4` 命名冲突兼容点。
4. 针对 `Reshape_6` 的 shape 污染现象，在 `musa_strided_slice_op.cc` 为 `int32/int64` shape 路径补了 stream 完成性保护。

从最新日志看，debug 版已经可以稳定越过之前容易出问题的 500 轮位置。

如果后续仍有异常，优先继续看：

- `tf_test_model/prunedGraph/logs/graph_inference/*/main.log`
- `tf_test_model/prunedGraph/logs/graph_inference/*/inference.log`

## 7. 当前编译状态

我在当前 shell 下重新执行过：

```bash
cd /home/fenghaoran/remote-dev/zhaoye-workspace/tensorflow_musa_extension
./build.sh debug
```

结果分两步：

1. 先在 `musa_neg_kernel.mu` 处暴露出 `int4/uint4` 命名冲突。
2. 修完该点后，编译继续推进，但随后卡在：
   - `tensorflow_musa_extension/musa_ext/mu/device/musa_stream.h`
   - 缺失头文件：`tensorflow/stream_executor/platform/port.h`

这说明当前 shell 实际使用的 TensorFlow 头文件布局，与这套工程原先依赖的 TensorFlow 版本并不一致。

更准确地说：

- 当前 shell 下 `python3` 指向的是本地 Python 3.10 / TensorFlow 2.20 环境。
- 该环境中已经没有 `tensorflow/stream_executor/platform/port.h` 这个旧路径。
- 因此当前 `debug build` 被拦住的直接原因，是 TensorFlow 头文件版本兼容问题，而不是 timing 宏本身的编译错误。

## 8. 已回退的尝试

以下尝试已经回退，不属于当前保留修改：

1. 改原始 `tf_test_model/prunedGraph/run_graph_tf_musa.py` 的行为。
2. 修改 `tensorflow_musa_extension/build.sh` 的 debug 构建模式。

当前保留的代码修改，只有：

1. `tensorflow_musa_extension/musa_ext/utils/logging.h`
2. `tensorflow_musa_extension/musa_ext/kernels/utils_op.h`
3. `tensorflow_musa_extension/musa_ext/kernels/math/musa_neg_kernel.mu`
4. `tensorflow_musa_extension/musa_ext/kernels/array/musa_strided_slice_op.cc`

## 9. 反思

这次排障里有一条边界需要明确：

- `release` 版本可以在**不修改原始 Python 测试脚本**的情况下跑完长轮次；
- 所以 debug 版本的问题，根因应优先归到 `tensorflow_musa_extension` 的 debug/timing 路径，而不是 `tf_test_model/prunedGraph/run_graph_tf_musa.py`。

因此后续原则应当是：

1. 不改原始 `.py` 测试入口。
2. 只修 `debug` 构建相对 `release` 多出来的那部分逻辑。
3. 优先比较：
   - `build.sh release`
   - `build.sh debug`
   - `MUSA_KERNEL_DEBUG` 打开的宏路径

这次真正应该保留的修复，是 `tensorflow_musa_extension` 里的 timing 宏与 device 上下文修复；不应该把 Python 脚本改成另一个测试体系来“掩盖” debug 版本问题。
