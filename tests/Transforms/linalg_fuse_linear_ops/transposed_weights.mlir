// RUN: heir-opt --linalg-fuse-linear-ops %s | FileCheck %s
// RUN: heir-opt --linalg-canonicalizations --linalg-fuse-linear-ops --drop-unit-dims --linalg-canonicalizations %s | FileCheck %s --check-prefix=PIPELINE

// Torch exports a linear layer as x * W^T. A per-channel scale (as from a
// BatchNorm) fuses into W before the transpose, so the weights stay a resource
// and the transpose stays next to the linear op.

// CHECK: func.func @fuse_through_transpose
// CHECK: %[[W:.*]] = arith.constant dense_resource<fused_weights{{.*}}> : tensor<2x3xf32>
// CHECK-NOT: arith.mulf
// CHECK: %[[T:.*]] = linalg.transpose ins(%[[W]] : tensor<2x3xf32>)
// CHECK: %[[RESULT:.*]] = linalg.matmul ins(%arg0, %[[T]]
// CHECK-NOT: arith.mulf
// CHECK: return %[[RESULT]]
// CHECK: fused_weights{{.*}}: "0x04000000000000400000004000000040000040400000404000004040"

// The 1xK matmul becomes a vecmat in drop-unit-dims, and then a matvec over
// the fused weights. A vecmat has no ciphertext lowering.

// PIPELINE: func.func @fuse_through_transpose
// PIPELINE: %[[W:.*]] = arith.constant dense_resource<fused_weights{{.*}}> : tensor<2x3xf32>
// PIPELINE-NOT: linalg.vecmat
// PIPELINE: linalg.matvec ins(%[[W]], %{{.*}} : tensor<2x3xf32>, tensor<3xf32>)
// PIPELINE-NOT: linalg.vecmat
// PIPELINE-NOT: arith.mulf
func.func @fuse_through_transpose(%arg0: tensor<1x3xf32>) -> tensor<1x2xf32> {
  %weights = arith.constant dense_resource<ones> : tensor<2x3xf32>
  %zero = arith.constant dense<0.000000e+00> : tensor<1x2xf32>
  %scale = arith.constant dense<[[2.0, 3.0]]> : tensor<1x2xf32>
  %empty = tensor.empty() : tensor<3x2xf32>
  %transposed = linalg.transpose ins(%weights : tensor<2x3xf32>) outs(%empty : tensor<3x2xf32>) permutation = [1, 0]
  %0 = linalg.matmul ins(%arg0, %transposed : tensor<1x3xf32>, tensor<3x2xf32>) outs(%zero : tensor<1x2xf32>) -> tensor<1x2xf32>
  %1 = arith.mulf %0, %scale : tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      ones: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F"
    }
  }
#-}
