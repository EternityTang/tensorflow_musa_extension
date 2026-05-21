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

"""Tests for MUSA SparseSlice operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class SparseSliceOpTest(MUSATestCase):
  """Tests for MUSA SparseSlice operator."""

  def _compare_sparse_slice_output(self, output_name, indices_np, values_np,
                                   shape_np, start_np, size_np, dtype,
                                   rtol=1e-5, atol=1e-8):
    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
    indices = tf.constant(indices_np, dtype=tf.int64)
    values = tf.constant(values_np.astype(np_dtype), dtype=dtype)
    shape = tf.constant(shape_np, dtype=tf.int64)
    start = tf.constant(start_np, dtype=tf.int64)
    size = tf.constant(size_np, dtype=tf.int64)

    def op_func(indices_in, values_in, shape_in, start_in, size_in):
      result = tf.raw_ops.SparseSlice(
          indices=indices_in,
          values=values_in,
          shape=shape_in,
          start=start_in,
          size=size_in)
      return getattr(result, output_name)

    compare_dtype = dtype if output_name == "output_values" else tf.int64
    self._compare_cpu_musa_results(
        op_func, [indices, values, shape, start, size], compare_dtype,
        rtol=rtol, atol=atol)

  def _test_sparse_slice(self, indices_np, values_np, shape_np, start_np,
                         size_np, dtype, rtol=1e-5, atol=1e-8):
    for output_name in ["output_indices", "output_values", "output_shape"]:
      self._compare_sparse_slice_output(output_name, indices_np, values_np,
                                        shape_np, start_np, size_np, dtype,
                                        rtol=rtol, atol=atol)

  def testBasic2D(self):
    indices = np.array([[0, 0], [0, 3], [1, 1], [2, 2], [3, 0]], dtype=np.int64)
    values = np.array([1, 2, 3, 4, 5])
    shape = np.array([4, 5], dtype=np.int64)
    start = np.array([0, 1], dtype=np.int64)
    size = np.array([3, 3], dtype=np.int64)

    for dtype in [tf.float32, tf.float16, tf.bfloat16, tf.int32, tf.int64]:
      rtol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-5
      atol = 1e-2 if dtype in [tf.float16, tf.bfloat16] else 1e-8
      self._test_sparse_slice(indices, values, shape, start, size, dtype,
                              rtol=rtol, atol=atol)

  def testFullRange(self):
    indices = np.array([[0, 0], [1, 2], [3, 1]], dtype=np.int64)
    values = np.array([1.5, -2.0, 7.0])
    shape = np.array([4, 3], dtype=np.int64)
    start = np.array([0, 0], dtype=np.int64)
    size = shape

    self._test_sparse_slice(indices, values, shape, start, size, tf.float32)

  def testEmptyResult(self):
    indices = np.array([[0, 0], [1, 2], [3, 1]], dtype=np.int64)
    values = np.array([1.5, -2.0, 7.0])
    shape = np.array([4, 4], dtype=np.int64)
    start = np.array([2, 2], dtype=np.int64)
    size = np.array([1, 1], dtype=np.int64)

    self._test_sparse_slice(indices, values, shape, start, size, tf.float32)

  def test3D(self):
    indices = np.array(
        [[0, 0, 0], [0, 2, 3], [1, 1, 1], [2, 2, 2], [3, 0, 1]],
        dtype=np.int64)
    values = np.array([1, 2, 3, 4, 5])
    shape = np.array([4, 4, 5], dtype=np.int64)
    start = np.array([0, 1, 1], dtype=np.int64)
    size = np.array([3, 2, 3], dtype=np.int64)

    self._test_sparse_slice(indices, values, shape, start, size, tf.int32)

  def testBoundaries(self):
    indices = np.array([[1, 1], [1, 3], [2, 2], [3, 1], [3, 3]], dtype=np.int64)
    values = np.array([10, 20, 30, 40, 50])
    shape = np.array([5, 5], dtype=np.int64)
    start = np.array([1, 1], dtype=np.int64)
    size = np.array([2, 2], dtype=np.int64)

    self._test_sparse_slice(indices, values, shape, start, size, tf.int64)


if __name__ == "__main__":
  tf.test.main()
