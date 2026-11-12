// RUN: heir-opt --layout-propagation=min-slot-count=16 --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK: #[[layout:.*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (1 + i0) mod 2 = 0 and (7 - 3i0 - 2i1 + 2slot) mod 16 = 0 and 0 < i0 <= 3 and 2 <= i1 <= 4 and 0 <= slot <= 15 and 32*floor((10 + 3i0 + 2i1)/32) <= -7 + 3i0 + 2i1 }">

module {
  // CHECK: func.func @fill_stride_2_1
  // CHECK-NOT: tensor_ext.assign_layout
  // CHECK: tensor.insert_slice
  // CHECK-SAME: tensor_ext.layout = #[[layout]]
  func.func @fill_stride_2_1(%arg0: !secret.secret<tensor<2x3xf32>>) -> !secret.secret<tensor<4x5xf32>> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x5xf32>
    %zeros = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x5xf32>) -> tensor<4x5xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x3xf32>>) {
    ^body(%input0: tensor<2x3xf32>):
      %1 = tensor.insert_slice %input0 into %zeros[1, 2] [2, 3] [2, 1] : tensor<2x3xf32> into tensor<4x5xf32>
      secret.yield %1 : tensor<4x5xf32>
    } -> !secret.secret<tensor<4x5xf32>>
    return %0 : !secret.secret<tensor<4x5xf32>>
  }
}

// -----

// CHECK: #[[layout:.*]] = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (1 + i2) mod 2 = 0 and (i1) mod 3 = 0 and (3 + 4i1 - 3i2 + 6slot) mod 24 = 0 and 0 <= i1 <= 3 and 0 < i2 <= 3 and 0 <= slot <= 15 }">

module {
  // CHECK: func.func @rank_reducing
  // CHECK-NOT: tensor_ext.assign_layout
  // CHECK: tensor.insert_slice
  // CHECK-SAME: tensor_ext.layout = #[[layout]]
  func.func @rank_reducing(%arg0: !secret.secret<tensor<2x2xf32>>) -> !secret.secret<tensor<1x4x4xf32>> {
    %zeros = arith.constant dense<0.000000e+00> : tensor<1x4x4xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x2xf32>>) {
    ^body(%input0: tensor<2x2xf32>):
      %1 = tensor.insert_slice %input0 into %zeros[0, 0, 1] [1, 2, 2] [1, 3, 2] : tensor<2x2xf32> into tensor<1x4x4xf32>
      secret.yield %1 : tensor<1x4x4xf32>
    } -> !secret.secret<tensor<1x4x4xf32>>
    return %0 : !secret.secret<tensor<1x4x4xf32>>
  }
}

// -----

module {
  func.func @concat_last_dim(%arg0: !secret.secret<tensor<2x3xf32>>, %arg1: !secret.secret<tensor<2x3xf32>>) -> !secret.secret<tensor<2x6xf32>> {
    %zeros = arith.constant dense<0.000000e+00> : tensor<2x6xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x3xf32>>, %arg1: !secret.secret<tensor<2x3xf32>>) {
    ^body(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>):
      %1 = tensor.insert_slice %a into %zeros[0, 0] [2, 3] [1, 1] : tensor<2x3xf32> into tensor<2x6xf32>
      // expected-error@+1 {{unsupported tensor.insert_slice}}
      %2 = tensor.insert_slice %b into %1[0, 3] [2, 3] [1, 1] : tensor<2x3xf32> into tensor<2x6xf32>
      secret.yield %2 : tensor<2x6xf32>
    } -> !secret.secret<tensor<2x6xf32>>
    return %0 : !secret.secret<tensor<2x6xf32>>
  }
}
