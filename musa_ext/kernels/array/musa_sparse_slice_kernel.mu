#include <musa_runtime.h>

#include <cstdint>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace musa {

__global__ void SparseSliceMarkKernel(const int64_t* __restrict__ indices,
                                      const int64_t* __restrict__ start,
                                      const int64_t* __restrict__ size,
                                      int64_t* __restrict__ marks,
                                      int64_t nnz, int rank) {
  const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nnz) return;

  bool in_slice = true;
  for (int d = 0; d < rank; ++d) {
    const int64_t index = indices[row * rank + d];
    const int64_t begin = start[d];
    const int64_t end = begin + size[d];
    if (index < begin || index >= end) {
      in_slice = false;
      break;
    }
  }
  marks[row] = in_slice ? 1 : 0;
}

template <typename T>
__global__ void SparseSliceScatterKernel(
    const int64_t* __restrict__ indices, const T* __restrict__ values,
    const int64_t* __restrict__ start, const int64_t* __restrict__ marks,
    const int64_t* __restrict__ scanned, int64_t* __restrict__ output_indices,
    T* __restrict__ output_values, int64_t nnz, int rank) {
  const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nnz || marks[row] == 0) return;

  const int64_t output_row = scanned[row] - 1;
  output_values[output_row] = values[row];
  for (int d = 0; d < rank; ++d) {
    output_indices[output_row * rank + d] = indices[row * rank + d] - start[d];
  }
}

void LaunchSparseSliceMarkKernel(const int64_t* indices, const int64_t* start,
                                 const int64_t* size, int64_t* marks,
                                 int64_t nnz, int rank, musaStream_t stream) {
  if (nnz <= 0) return;
  const int threads = 256;
  const int blocks = static_cast<int>((nnz + threads - 1) / threads);
  SparseSliceMarkKernel<<<blocks, threads, 0, stream>>>(indices, start, size,
                                                        marks, nnz, rank);
}

template <typename T>
void LaunchSparseSliceScatterKernel(const int64_t* indices, const T* values,
                                    const int64_t* start,
                                    const int64_t* marks,
                                    const int64_t* scanned,
                                    int64_t* output_indices, T* output_values,
                                    int64_t nnz, int rank,
                                    musaStream_t stream) {
  if (nnz <= 0) return;
  const int threads = 256;
  const int blocks = static_cast<int>((nnz + threads - 1) / threads);
  SparseSliceScatterKernel<T><<<blocks, threads, 0, stream>>>(
      indices, values, start, marks, scanned, output_indices, output_values, nnz,
      rank);
}

#define INSTANTIATE_SPARSE_SLICE_SCATTER(T)                         \
  template void LaunchSparseSliceScatterKernel<T>(                   \
      const int64_t* indices, const T* values, const int64_t* start, \
      const int64_t* marks, const int64_t* scanned,                  \
      int64_t* output_indices, T* output_values, int64_t nnz,        \
      int rank, musaStream_t stream)

INSTANTIATE_SPARSE_SLICE_SCATTER(float);
INSTANTIATE_SPARSE_SLICE_SCATTER(double);
INSTANTIATE_SPARSE_SLICE_SCATTER(int32);
INSTANTIATE_SPARSE_SLICE_SCATTER(int64);
INSTANTIATE_SPARSE_SLICE_SCATTER(Eigen::half);
INSTANTIATE_SPARSE_SLICE_SCATTER(bfloat16);

#undef INSTANTIATE_SPARSE_SLICE_SCATTER

}  // namespace musa
}  // namespace tensorflow
