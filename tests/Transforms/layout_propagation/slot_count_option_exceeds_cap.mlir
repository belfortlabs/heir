// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --layout-propagation=min-slot-count=65536 --verify-diagnostics %s

// A min-slot-count above the 32768 slots of the backend's logN = 16 ring is
// rejected before propagation.

// expected-error@below {{min-slot-count 65536 exceeds the 32768 slots of the backend's largest ring degree 65536}}
module {
func.func @matvec(%arg0: !secret.secret<tensor<1500xf32>>) -> !secret.secret<tensor<4xf32>> {
  %cst = arith.constant dense<0.0> : tensor<4xf32>
  %matrix = arith.constant dense<1.0> : tensor<4x1500xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1500xf32>>) {
  ^body(%input0: tensor<1500xf32>):
    %1 = linalg.matvec ins(%matrix, %input0 : tensor<4x1500xf32>, tensor<1500xf32>) outs(%cst : tensor<4xf32>) -> tensor<4xf32>
    secret.yield %1 : tensor<4xf32>
  } -> !secret.secret<tensor<4xf32>>
  return %0 : !secret.secret<tensor<4xf32>>
}
}
