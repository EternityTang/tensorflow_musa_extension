# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TensorFlow MUSA Extension is a high-performance TensorFlow plugin for Moore Threads MUSA GPU architecture. It provides native MUSA kernel implementations for GPU acceleration.

## Build Commands

```bash
# Release build (optimized, default)
./build.sh

# Debug build (enables kernel timing instrumentation)
./build.sh debug
```

Build output: `build/libmusa_plugin.so`

## Testing

```bash
cd test

# Run all operator tests (default)
python test_runner.py

# Run all fusion tests
python test_runner.py --fusion

# Run single test file
python test_runner.py --single ops/matmul_op_test.py
python test_runner.py --single fusion/layernorm_gelu_fusion_test.py

# Run individual test directly
python -m ops.add_op_test
python -m fusion.layernorm_gelu_fusion_test
```

Test naming conventions:
- Operator tests (`test/ops/`): `op_name_op_test.py`
- Fusion tests (`test/fusion/`): `*_fusion_test.py`

All tests inherit from `MUSATestCase` (in `test/musa_test_utils.py`) which handles plugin loading.

## Architecture

```
musa_ext/
├── kernels/         # MUSA kernel implementations (.mu files + .cc wrappers)
│   ├── math/        # Math ops: Add, MatMul, etc.
│   ├── nn/          # Neural network ops: Conv, LayerNorm, etc.
│   ├── training/    # Training ops: ApplyGradientDescent, etc.
│   └── ...
├── mu/
│   ├── device/      # MUSA device, allocator, executor, telemetry
│   ├── optimizer/   # Graph optimizer (musa_graph_optimizer.cc)
│   └── graph_fusion/ # Fusion patterns (LayerNorm, Gelu, etc.)
└── utils/           # Utility functions
```

Key components:
- **Device registration**: `mu/device_register.cc/h` registers the MUSA device type with TensorFlow's Stream Executor API
- **Graph optimizer**: `mu/optimizer/musa_graph_optimizer.cc` handles layout conversion and fusion
- **Fusion patterns**: `mu/graph_fusion/fusion_pattern_manager.cc` manages operator fusion rules

## Debugging

Kernel timing (requires debug build):
```bash
./build.sh debug
export MUSA_TIMING_KERNEL_LEVEL=2  # 1=total only, 2=total+segments
export MUSA_TIMING_KERNEL_NAME=ALL # Filter by kernel name
export MUSA_TIMING_KERNEL_STATS=1  # Print summary on exit
```

Telemetry for tracing:
```bash
export MUSA_TELEMETRY_ENABLED=1
export MUSA_TELEMETRY_LOG_PATH=/tmp/telemetry.json
```

See `docs/DEBUG_GUIDE.md` for full debugging documentation.

## Environment Requirements

- MUSA SDK: `/usr/local/musa` (musa runtime, muBLAS, muDNN)
- TensorFlow: == 2.6.1
- Python: >= 3.7
- CMake: >= 3.10

## Code Style

C++ code uses the `.clang-format` configuration. Pre-commit hooks are configured in `.pre-commit-config.yaml`.

## Important Notes

- MUSA kernels are `.mu` files compiled by `mcc` (MUSA compiler)
- The plugin uses TensorFlow's experimental Stream Executor C API for device registration
- NDEBUG is always enabled to match TensorFlow wheel ABI (see CMakeLists.txt comments)
- Device type name is `MUSA` (defined in `device_register.h` as `DEVICE_MTGPU`)