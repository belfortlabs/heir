// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

// A vecmat over a transposed matrix becomes a matvec over the original matrix.

module {
  // CHECK: func @main
  // CHECK-SAME: %[[arg0:.*]]: tensor<512x784xf32>,
  // CHECK-SAME: %[[arg1:.*]]: tensor<784xf32>)
  // CHECK: %[[cst:.*]] = arith.constant dense<0.{{0*}}e+00> : tensor<512xf32>
  // CHECK-NOT: linalg.transpose
  // CHECK: %[[v0:.*]] = linalg.matvec ins(%[[arg0]], %[[arg1]] : tensor<512x784xf32>, tensor<784xf32>) outs(%[[cst]] : tensor<512xf32>)
  // CHECK: return %[[v0]] : tensor<512xf32>
  func.func @main(%arg0: tensor<512x784xf32>, %arg2: tensor<784xf32>) -> tensor<512xf32> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %0 = tensor.empty() : tensor<784x512xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<512x784xf32>) outs(%0 : tensor<784x512xf32>) permutation = [1, 0]
    %1 = linalg.vecmat ins(%arg2, %transposed : tensor<784xf32>, tensor<784x512xf32>) outs(%cst_0 : tensor<512xf32>) -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
}

// -----

// A matvec over a transposed matrix stays a matvec: there is no ciphertext
// lowering for a vecmat.

module {
  // CHECK: func @main
  // CHECK-SAME: %[[arg0:.*]]: tensor<784x512xf32>,
  // CHECK-SAME: %[[arg1:.*]]: tensor<784xf32>)
  // CHECK: %[[transposed:.*]] = linalg.transpose ins(%[[arg0]]
  // CHECK: %[[v0:.*]] = linalg.matvec ins(%[[transposed]], %[[arg1]] : tensor<512x784xf32>, tensor<784xf32>)
  // CHECK-NOT: linalg.vecmat
  // CHECK: return %[[v0]] : tensor<512xf32>
  func.func @main(%arg0: tensor<784x512xf32>, %arg2: tensor<784xf32>) -> tensor<512xf32> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %0 = tensor.empty() : tensor<512x784xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<784x512xf32>) outs(%0 : tensor<512x784xf32>) permutation = [1, 0]
    %1 = linalg.matvec ins(%transposed, %arg2 : tensor<512x784xf32>, tensor<784xf32>) outs(%cst_0 : tensor<512xf32>) -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
}

// -----

// A vecmat over any other matrix becomes a matvec over its transpose.

module {
  // CHECK: func @main
  // CHECK-SAME: %[[arg0:.*]]: tensor<784x512xf32>,
  // CHECK-SAME: %[[arg1:.*]]: tensor<784xf32>)
  // CHECK: %[[cst:.*]] = arith.constant dense<0.{{0*}}e+00> : tensor<512xf32>
  // CHECK: %[[transposed:.*]] = linalg.transpose ins(%[[arg0]] : tensor<784x512xf32>) outs(%{{.*}} : tensor<512x784xf32>) permutation = [1, 0]
  // CHECK: %[[v0:.*]] = linalg.matvec ins(%[[transposed]], %[[arg1]] : tensor<512x784xf32>, tensor<784xf32>) outs(%[[cst]] : tensor<512xf32>)
  // CHECK-NOT: linalg.vecmat
  // CHECK: return %[[v0]] : tensor<512xf32>
  func.func @main(%arg0: tensor<784x512xf32>, %arg2: tensor<784xf32>) -> tensor<512xf32> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %1 = linalg.vecmat ins(%arg2, %arg0 : tensor<784xf32>, tensor<784x512xf32>) outs(%cst_0 : tensor<512xf32>) -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
}

// -----

// An op between the transpose and the vecmat, such as a fused BatchNorm scale,
// does not block the rewrite.

module {
  // CHECK: func @main
  // CHECK-SAME: %[[arg0:.*]]: tensor<512x784xf32>,
  // CHECK-SAME: %[[arg1:.*]]: tensor<784x512xf32>,
  // CHECK-SAME: %[[arg2:.*]]: tensor<784xf32>)
  // CHECK: %[[transposed:.*]] = linalg.transpose ins(%[[arg0]]
  // CHECK: %[[scaled:.*]] = arith.mulf %[[transposed]], %[[arg1]]
  // CHECK: %[[transposed2:.*]] = linalg.transpose ins(%[[scaled]]
  // CHECK: %[[v0:.*]] = linalg.matvec ins(%[[transposed2]], %[[arg2]] : tensor<512x784xf32>, tensor<784xf32>)
  // CHECK-NOT: linalg.vecmat
  // CHECK: return %[[v0]] : tensor<512xf32>
  func.func @main(%arg0: tensor<512x784xf32>, %arg1: tensor<784x512xf32>, %arg2: tensor<784xf32>) -> tensor<512xf32> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %0 = tensor.empty() : tensor<784x512xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<512x784xf32>) outs(%0 : tensor<784x512xf32>) permutation = [1, 0]
    %scaled = arith.mulf %transposed, %arg1 : tensor<784x512xf32>
    %1 = linalg.vecmat ins(%arg2, %scaled : tensor<784xf32>, tensor<784x512xf32>) outs(%cst_0 : tensor<512xf32>) -> tensor<512xf32>
    return %1 : tensor<512xf32>
  }
}
