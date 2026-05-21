#include <pybind11/pybind11.h>

#include <string>

#include "mu/runtime_config_c_api.h"

namespace py = pybind11;

PYBIND11_MODULE(_runtime_config_bindings, m) {
  m.doc() = "TensorFlow MUSA runtime configuration bindings";

  m.def(
      "set_musa_allow_growth",
      [](bool enabled) { TFMusaSetAllowGrowth(enabled ? 1 : 0); },
      py::arg("enabled"));

  m.def(
      "set_musa_telemetry_config",
      [](bool enabled, const std::string& log_path,
         unsigned long long buffer_size, int flush_interval_ms,
         bool include_stack_trace) {
        TFMusaSetTelemetryConfig(enabled ? 1 : 0, log_path.c_str(), buffer_size,
                                 flush_interval_ms,
                                 include_stack_trace ? 1 : 0);
      },
      py::arg("enabled"), py::arg("log_path"), py::arg("buffer_size"),
      py::arg("flush_interval_ms"), py::arg("include_stack_trace"));

  m.def("is_musa_telemetry_enabled",
        []() { return TFMusaTelemetryIsEnabled() != 0; });

  m.def("get_musa_telemetry_health",
        []() { return std::string(TFMusaGetTelemetryHealthSnapshot()); });

  m.def(
      "set_musa_graph_dump_config",
      [](bool enabled, const std::string& dump_dir, bool dump_text,
         bool dump_slim) {
        TFMusaSetGraphDumpConfig(enabled ? 1 : 0, dump_dir.c_str(),
                                 dump_text ? 1 : 0, dump_slim ? 1 : 0);
      },
      py::arg("enabled"), py::arg("dump_dir"), py::arg("dump_text"),
      py::arg("dump_slim"));

  m.def("clear_musa_graph_dump_config", []() { TFMusaClearGraphDumpConfig(); });

  m.def("is_musa_graph_dump_enabled",
        []() { return TFMusaGraphDumpIsEnabled() != 0; });

  m.def("get_musa_graph_dump_directory",
        []() { return std::string(TFMusaGetGraphDumpDirectory()); });

  m.def("is_musa_graph_dump_text_enabled",
        []() { return TFMusaGraphDumpTextIsEnabled() != 0; });

  m.def("is_musa_graph_dump_slim_enabled",
        []() { return TFMusaGraphDumpSlimIsEnabled() != 0; });
}
