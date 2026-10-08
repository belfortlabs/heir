// RUN: heir-opt --linalg-canonicalizations %s | FileCheck %s

// Torch exports BatchNorm statistics as dense resources and its epsilon as an
// f64 constant that is not exact in f32. Both are folded so that
// rsqrt(var + eps) becomes a constant.

// CHECK: func.func @batch_norm_inv_std
// CHECK-NEXT: %[[CST:.*]] = arith.constant dense<{{.*}}> : tensor<2xf32>
// CHECK-NEXT: return %[[CST]]
func.func @batch_norm_inv_std() -> tensor<2xf32> {
  %var = arith.constant dense_resource<var> : tensor<2xf32>
  %eps = arith.constant dense<1.000000e-05> : tensor<2xf64>
  %0 = arith.truncf %eps : tensor<2xf64> to tensor<2xf32>
  %1 = arith.addf %0, %var : tensor<2xf32>
  %2 = math.rsqrt %1 : tensor<2xf32>
  return %2 : tensor<2xf32>
}

// Resources used as weights of a contraction, convolution or transpose are not
// inlined.

// CHECK: func.func @weights_stay_resources
// CHECK: arith.constant dense_resource<weights> : tensor<2x2xf32>
func.func @weights_stay_resources(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %weights = arith.constant dense_resource<weights> : tensor<2x2xf32>
  %zero = arith.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = linalg.matvec ins(%weights, %arg0 : tensor<2x2xf32>, tensor<2xf32>) outs(%zero : tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      var: "0x040000000000803F00008040",
      weights: "0x040000000000803F0000803F0000803F0000803F"
    }
  }
#-}
