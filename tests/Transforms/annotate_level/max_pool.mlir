// RUN: heir-opt --annotate-level %s | FileCheck %s

// kernel.max_pool drops the levels of Cyclops' MaxPool from its input level:
// the gather stages, two league multiplications, and the compaction stages.

// CHECK: func.func @aligned_windows
func.func @aligned_windows(%arg0: !secret.secret<tensor<1x8192xf32>>) -> !secret.secret<tensor<1x8192xf32>> {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x8192xf32>>) {
  ^body(%val: tensor<1x8192xf32>):
    // No gather, two league levels, two compaction stages.
    // CHECK: kernel.max_pool
    // CHECK-SAME: mgmt.level = 4 : index
    %1 = kernel.max_pool %val {num_slots = 8192 : i64, input_length = 4864 : i64, window_size = 2 : i64, stride = 2 : i64, value_bound = 0.5 : f64} : tensor<1x8192xf32> -> tensor<1x8192xf32>
    secret.yield %1 : tensor<1x8192xf32>
  } -> !secret.secret<tensor<1x8192xf32>>
  return %0 : !secret.secret<tensor<1x8192xf32>>
}

// CHECK: func.func @chained_pools
func.func @chained_pools(%arg0: !secret.secret<tensor<1x4096xf32>>) -> !secret.secret<tensor<1x4096xf32>> {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x4096xf32>>) {
  ^body(%val: tensor<1x4096xf32>):
    // A window of 3 pads to 4 slots: two gather stages, two league levels,
    // two compaction stages.
    // CHECK: kernel.max_pool
    // CHECK-SAME: mgmt.level = 6 : index
    %1 = kernel.max_pool %val {num_slots = 4096 : i64, input_length = 1000 : i64, window_size = 3 : i64, stride = 3 : i64, ceil_mode = true, value_bound = 0.5 : f64} : tensor<1x4096xf32> -> tensor<1x4096xf32>
    // A window of 1 is free.
    // CHECK: kernel.max_pool
    // CHECK-SAME: mgmt.level = 6 : index
    %2 = kernel.max_pool %1 {num_slots = 4096 : i64, input_length = 334 : i64, window_size = 1 : i64, stride = 1 : i64, value_bound = 0.5 : f64} : tensor<1x4096xf32> -> tensor<1x4096xf32>
    secret.yield %2 : tensor<1x4096xf32>
  } -> !secret.secret<tensor<1x4096xf32>>
  return %0 : !secret.secret<tensor<1x4096xf32>>
}
