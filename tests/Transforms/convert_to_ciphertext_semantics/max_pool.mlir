// RUN: heir-opt --layout-propagation=min-slot-count=64 --convert-to-ciphertext-semantics=min-slot-count=64 %s | FileCheck %s

// Four channels of 8 values lie end to end in the first 32 slots; one
// kernel.max_pool with windows of 2 pools all of them.

// CHECK: func.func @pool
// CHECK-SAME: !secret.secret<tensor<1x64xf32>>
// CHECK: secret.generic
// CHECK: ^body(%[[input:.*]]: tensor<1x64xf32>):
// CHECK: %[[pool:.*]] = kernel.max_pool %[[input]] {input_length = 32 : i64, num_slots = 64 : i64, stride = 2 : i64, value_bound = 1.000000e+00 : f64, window_size = 2 : i64} : tensor<1x64xf32> -> tensor<1x64xf32>
// CHECK: secret.yield %[[pool]]
module attributes {backend.cheddar} {
  func.func @pool(%arg0: !secret.secret<tensor<1x4x8xf32>>) -> !secret.secret<tensor<1x4x4xf32>> {
    %neg_inf = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1x4x4xf32>
    %1 = linalg.fill ins(%neg_inf : f32) outs(%0 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %window = tensor.empty() : tensor<2xf32>
    %r = secret.generic(%arg0 : !secret.secret<tensor<1x4x8xf32>>) {
    ^body(%x: tensor<1x4x8xf32>):
      %2 = linalg.pooling_ncw_max {dilations = dense<1> : vector<1xi64>, domain_lower = 0.0 : f64, domain_upper = 1.0 : f64, strides = dense<2> : vector<1xi64>} ins(%x, %window : tensor<1x4x8xf32>, tensor<2xf32>) outs(%1 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
      secret.yield %2 : tensor<1x4x4xf32>
    } -> !secret.secret<tensor<1x4x4xf32>>
    return %r : !secret.secret<tensor<1x4x4xf32>>
  }
}
