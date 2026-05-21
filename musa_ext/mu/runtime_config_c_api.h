#ifndef TENSORFLOW_MUSA_MUSA_EXT_MU_RUNTIME_CONFIG_C_API_H_
#define TENSORFLOW_MUSA_MUSA_EXT_MU_RUNTIME_CONFIG_C_API_H_

extern "C" {

void TFMusaSetAllowGrowth(int enabled);
void TFMusaSetTelemetryConfig(int enabled, const char* log_path,
                              unsigned long long buffer_size,
                              int flush_interval_ms,
                              int include_stack_trace);
int TFMusaTelemetryIsEnabled();
const char* TFMusaGetTelemetryHealthSnapshot();
void TFMusaSetGraphDumpConfig(int enabled, const char* dump_dir,
                              int dump_text, int dump_slim);
void TFMusaClearGraphDumpConfig();
int TFMusaGraphDumpIsEnabled();
const char* TFMusaGetGraphDumpDirectory();
int TFMusaGraphDumpTextIsEnabled();
int TFMusaGraphDumpSlimIsEnabled();

}  // extern "C"

#endif  // TENSORFLOW_MUSA_MUSA_EXT_MU_RUNTIME_CONFIG_C_API_H_
