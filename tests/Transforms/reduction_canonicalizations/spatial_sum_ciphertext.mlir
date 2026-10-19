// RUN: heir-opt --reduction-canonicalizations --layout-propagation --convert-to-ciphertext-semantics %s | FileCheck %s

// The collapsed spatial sum lowers to rotations and adds on the row-major
// layout, without a layout conversion between reduction steps.

// CHECK: func @spatial_sum
// CHECK-NOT: tensor_ext.convert_layout
// CHECK-NOT: linalg.reduce
// CHECK: tensor_ext.rotate
// CHECK: return
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
func.func @spatial_sum(%arg0: !secret.secret<tensor<8x4x4xf32>>) -> !secret.secret<tensor<8xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<8x4x4xf32>>) {
  ^body(%input0: tensor<8x4x4xf32>):
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "reduction"]} ins(%input0 : tensor<8x4x4xf32>) outs(%cst : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.addf %in, %out : f32
      linalg.yield %2 : f32
    } -> tensor<8xf32>
    secret.yield %1 : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}
