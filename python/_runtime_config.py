# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Helpers for configuring TensorFlow MUSA runtime options."""

def _runtime_config_bindings():
    from ._loader import load_plugin

    load_plugin()

    from . import _runtime_config_bindings as bindings

    return bindings


def set_musa_allow_growth(enabled=True):
    """Set process-wide MUSA BFC allocator allow_growth.

    This setting is applied to subsequently created MUSA devices. The
    `TF_FORCE_GPU_ALLOW_GROWTH` environment variable, when set to `true` or
    `false`, takes precedence over this Python setting.

    Args:
        enabled: Whether the MUSA device allocator should grow on demand.
    """
    _runtime_config_bindings().set_musa_allow_growth(bool(enabled))


def set_musa_telemetry_config(
    enabled=True,
    log_path=None,
    buffer_size=10000,
    flush_interval_ms=100,
    include_stack_trace=False,
):
    """Configure process-wide MUSA telemetry.

    This setting overrides the `MUSA_TELEMETRY_*` environment variables for the
    current process. Calling it repeatedly reconfigures the native telemetry
    manager; passing `enabled=False` disables telemetry and flushes pending
    events.

    Args:
        enabled: Whether telemetry should be enabled.
        log_path: Optional JSONL output path. When omitted or empty, telemetry
            writes to stderr.
        buffer_size: Maximum number of queued telemetry events.
        flush_interval_ms: Background flush interval in milliseconds.
        include_stack_trace: Whether telemetry should include stack traces for
            supported events.
    """
    if buffer_size <= 0:
        raise ValueError("buffer_size must be greater than 0")
    if flush_interval_ms <= 0:
        raise ValueError("flush_interval_ms must be greater than 0")

    _runtime_config_bindings().set_musa_telemetry_config(
        bool(enabled),
        "" if log_path is None else str(log_path),
        int(buffer_size),
        int(flush_interval_ms),
        bool(include_stack_trace),
    )


def enable_musa_telemetry(**kwargs):
    """Enable MUSA telemetry with optional configuration keyword arguments."""
    set_musa_telemetry_config(enabled=True, **kwargs)


def disable_musa_telemetry():
    """Disable MUSA telemetry and flush pending native events."""
    set_musa_telemetry_config(enabled=False)


def is_musa_telemetry_enabled():
    """Return whether native MUSA telemetry is currently enabled."""
    return bool(_runtime_config_bindings().is_musa_telemetry_enabled())


def get_musa_telemetry_health():
    """Return a JSON health snapshot for the native MUSA telemetry manager."""
    return _runtime_config_bindings().get_musa_telemetry_health()
