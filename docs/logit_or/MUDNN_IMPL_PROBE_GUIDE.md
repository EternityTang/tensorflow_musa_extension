# muDNN 底层实现排查复用指南

这份文档用于在 `tf_test_model` 侧复用一套固定流程，排查 TensorFlow MUSA 插件调用到的 muDNN 公开接口、动态库符号，以及当前环境里是否存在可继续阅读的内部实现源码。

当前最直接的使用场景：

- 新增或优化某个 MUSA op，例如 `LogicalOr`
- 想确认底层到底走的是 muDNN 公开 API，还是还能继续追到内部源码
- 想把排查结果沉淀到 `logs/mudnn_probe/` 里，后续和 `.sh` 验证脚本一起复用

## 1. 运行环境

建议在 docker 容器内、`tf26` 虚拟环境中执行。

```bash
docker exec -it zhaoye bash
conda activate tf26
cd /workspace
```

说明：

- `tf_test_model` 和 `tensorflow_musa_extension` 都挂载在 `/workspace`
- 日志统一写到 `/workspace/tf_test_model/logs/mudnn_probe/<timestamp>/`
- 容器内产生的日志，宿主机侧也能直接看到

## 2. 一次性采集日志

直接运行下面这段命令：

```bash
mkdir -p /workspace/tf_test_model/logs/mudnn_probe
ts="$(date +%Y-%m-%d-%H.%M.%S)"
LOG_DIR="/workspace/tf_test_model/logs/mudnn_probe/${ts}"
mkdir -p "${LOG_DIR}"

{
  echo "pwd=$(pwd)"
  echo "python=$(which python)"
  echo "conda_prefix=${CONDA_PREFIX:-<empty>}"
  echo
  echo "== /workspace =="
  ls -l /workspace
  echo
  echo "== MUSA include/lib =="
  ls -ld /usr/local/musa /usr/local/musa/include /usr/local/musa/lib 2>/dev/null
} > "${LOG_DIR}/00_env.log"

find /usr/local/musa/include /workspace/ls/include \
  -maxdepth 2 -type f 2>/dev/null \
  | grep -Ei 'mudnn|dnn' \
  | sort > "${LOG_DIR}/10_mudnn_headers.log"

grep -RIn -E 'class Binary|LOGICAL_OR|LOGICAL_AND|SetMode|SetNdInfo|SetFormat|SetAddr|SetType|TensorImpl|BinaryImpl|HandleImpl' \
  /usr/local/musa/include /workspace/ls/include 2>/dev/null \
  > "${LOG_DIR}/11_mudnn_key_hits.log"

nm -CD /usr/local/musa/lib/libmudnn.so 2>/dev/null \
  | grep -E 'musa::dnn::(Binary::|BinaryRun|Tensor::SetNdInfo|TensorImpl::SetNdInfo|_Binary)' \
  > "${LOG_DIR}/12_mudnn_symbols.log"

{
  echo "===== libmudnn.so dependencies ====="
  ls -l /usr/local/musa/lib/libmudnn.so
  ldd /usr/local/musa/lib/libmudnn.so
  echo
  echo "===== libmudnn.so dynamic section ====="
  readelf -d /usr/local/musa/lib/libmudnn.so
} > "${LOG_DIR}/13_mudnn_deps.log"

{
  echo "LOG_DIR=${LOG_DIR}"
  echo
  echo "===== headers ====="
  sed -n '1,120p' "${LOG_DIR}/10_mudnn_headers.log"
  echo
  echo "===== key hits ====="
  sed -n '1,200p' "${LOG_DIR}/11_mudnn_key_hits.log"
  echo
  echo "===== symbols ====="
  sed -n '1,120p' "${LOG_DIR}/12_mudnn_symbols.log"
} | tee "${LOG_DIR}/99_summary.log"
```

执行完成后，记下最后一行：

```bash
LOG_DIR=/workspace/tf_test_model/logs/mudnn_probe/<timestamp>
```

## 3. 针对 `LogicalOr` 的最小确认命令

如果只想确认 `LogicalOr` 的公开接口在哪个头文件里，可以额外执行：

```bash
sed -n '140,230p' /usr/local/musa/include/mudnn_tensor.h
sed -n '132,152p' /usr/local/musa/include/mudnn_base.h
```

当前环境下，`LogicalOr` 相关接口位于：

- `/usr/local/musa/include/mudnn_tensor.h`
- `/usr/local/musa/include/mudnn_base.h`

重点看这几类定义：

- `class Binary final : public ImplBase`
- `Binary::Mode::LOGICAL_OR`
- `Binary::SetMode`
- `Binary::Run`
- `Tensor::SetAddr`
- `Tensor::SetType`
- `Tensor::SetFormat`
- `Tensor::SetNdInfo(dim)`
- `Tensor::SetNdInfo(dim, stride)`

## 4. 怎么解读日志

### 4.1 看头文件位置

先看 `10_mudnn_headers.log`。

如果像当前环境这样出现：

- `mudnn.h`
- `mudnn_base.h`
- `mudnn_ops.h`
- `mudnn_tensor.h`

说明这个 container 的 muDNN 公开头是按功能拆分的，`Binary` 不一定在历史环境中的 `mudnn_math.h` 里。

### 4.2 看公开 API 命中

再看 `11_mudnn_key_hits.log`。

如果能看到类似下面这些命中：

- `mudnn_tensor.h: class Binary final : public ImplBase`
- `mudnn_tensor.h: LOGICAL_OR`
- `mudnn_tensor.h: Status SetMode(Mode m);`
- `mudnn_base.h: Status SetNdInfo(...);`

说明当前环境中至少能确认：

- `logical_or` 是通过 muDNN 的 `Binary` 类暴露出来的
- `Tensor` 支持只传 `dim` 的描述方式
- `Tensor` 也支持传 `dim + stride` 的描述方式

这对后续评估 `broadcast-view` / `stride` 路径很重要。

### 4.3 看动态库符号

再看 `12_mudnn_symbols.log`。

当前环境中可以看到：

- `musa::dnn::Binary::SetMode`
- `musa::dnn::Binary::Run`
- `musa::dnn::Tensor::SetNdInfo(...)`

如果某些条目前面是 `U`，例如：

- `U musa::dnn::BinaryRun(...)`
- `U musa::dnn::TensorImpl::SetNdInfo(...)`

这里的 `U` 表示未定义引用。

也就是说：

- 这个 `.so` 里看到了它要调用这些符号
- 但定义体不在当前可见符号里

这通常意味着：

- 实现藏在别的链接单元里
- 或者当前环境只有 ABI / 动态库接口，没有内部源码

### 4.4 看依赖链

最后看 `13_mudnn_deps.log`。

这一步用于确认：

- `libmudnn.so` 依赖了哪些运行时库
- 有没有明显额外的内部实现库可继续往下追

如果依赖里只看到运行时相关库，例如：

- `libmusart.so`
- `libmusa.so`

但没有额外可读的源码仓或实现库，那么通常就可以先判定：

- 当前 container 更像“公开安装包”
- 不是“可直接阅读 muDNN 内部实现源码”的开发环境

## 5. 如何判断有没有内部源码

如果怀疑机器上还挂了 muDNN 的内部 repo，可以额外跑：

```bash
find /workspace /opt /root \
  \( -path '*/.git' -o -path '*/build' -o -path '*/bazel-*' -o -path '*/.cache' \) -prune -o \
  -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cc' -o -name '*.cpp' \) -print 2>/dev/null \
  | grep -Ei 'mudnn|BinaryImpl|TensorImpl|HandleImpl|container\.h$|binary'
```

重点关注是否真的搜到这些内容：

- `BinaryImpl` 的类定义
- `TensorImpl` 的类定义
- `HandleImpl` 的类定义
- `_BinaryRun`
- `_BinaryContigRun`
- `_BinaryMidAlignRun`
- `_BinaryLastAlignRun`
- `BroadcastOneDimRun`
- `container.h`

如果没有搜到，就不要继续在公开 SDK 目录里硬找源码了，通常找不到。

## 6. 和 TensorFlow kernel 优化的关系

`tensorflow_musa_extension` 里像 `logical_or` 这种算子，常见执行链是：

1. TensorFlow kernel 做 shape/broadcast 判断
2. `CreateMTensor(...)` 包装输入输出
3. `Binary::SetMode(LOGICAL_OR)`
4. `Binary::Run(...)`

因此 muDNN 排查结果主要决定下面几件事：

- `logical_or` 能不能先做 same-shape fast path
- `broadcast-view` 有没有意义
- `dim + stride` 方式有没有机会触发更好的库内路径
- `output forward` 是否值得尝试

## 7. 关于 output forward 的安全边界

TensorFlow 提供的相关接口在：

- `tensorflow/core/framework/op_kernel.h`

关键接口包括：

- `forward_input_to_output_with_shape`
- `forward_input`
- `forward_input_or_allocate_output`

这些接口的含义不是“强行复用输入地址”，而是：

- 先尝试复用输入底层 buffer
- 只有框架判断安全时才复用
- 不安全就退回到正常分配 output

这里的“框架判断安全”，主要是图生命周期和 buffer 所有权层面的安全，例如：

- input/output 的 dtype 兼容
- shape 或元素数兼容
- allocator / memory type 兼容
- 底层 buffer 的 refcount 合法
- forwarding reservation 允许

但要注意：

- TensorFlow 框架层安全
- 不等于 muDNN 内核层 in-place alias 安全

即使框架允许 forward，也仍然要确认底层 `Binary::Run(...)` 是否支持：

- `out == in0`
- `out == in1`

如果底层库不支持 alias/in-place 语义，结果仍然可能错误。

因此，像 `logical_or` 这类 op 的优化建议通常是：

1. 先做 `IsExpensive() = false`
2. 先做 same-shape fast path
3. 再看 `broadcast-view` / `stride`
4. 最后才考虑 `forward_input_or_allocate_output`

## 8. 当前环境的结论模板

如果本次排查结果和现在一致，可以直接记录为：

- `LogicalOr` 的 muDNN 公开接口位于 `mudnn_tensor.h`
- `Tensor` 的 `dim/stride` 接口位于 `mudnn_base.h`
- 当前环境能看到公开 API 和部分动态库符号
- 当前环境看不到 `BinaryImpl / TensorImpl / HandleImpl` 的实现体
- 当前环境更像公开安装包，不像带内部源码的 muDNN 开发环境

## 9. 推荐和整网性能脚本配合使用

建议和下面的整网脚本配合使用：

- `run_or_pruned_graph_whole_model.sh`

使用方式：

1. 先用本指南确认 `logical_or` 对应的 muDNN 公开接口、符号和环境性质
2. 再改 `tensorflow_musa_extension` 中的 `logical_or`
3. 改完后用整网脚本做：
   - 精度对比
   - whole-network benchmark
   - `profile-ops`
4. 对比 `LogicalOr` 和相关链路 op 的耗时变化

这样后续做 `LogicalOr`、`Equal`、`Abs`、`Add` 等 op 优化时，都可以直接复用这份流程。
