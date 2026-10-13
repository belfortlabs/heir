// RUN: heir-opt --layout-propagation="min-slot-count=16" --split-input-file --verify-diagnostics %s

// The flattened data has more elements than the min-slot-count slots of one
// ciphertext.
func.func @conv1d(%arg0: !secret.secret<tensor<20xf32>>) -> !secret.secret<tensor<18xf32>> {
  %cst = arith.constant dense<0.0> : tensor<18xf32>
  %filter = arith.constant dense<1.0> : tensor<3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<20xf32>>) {
  ^body(%input0: tensor<20xf32>):
    // expected-error@+1 {{Flattened data must fit into a single ciphertext, but got 20 elements and ciphertext size is 16}}
    %1 = linalg.conv_1d ins(%input0, %filter : tensor<20xf32>, tensor<3xf32>) outs(%cst : tensor<18xf32>) -> tensor<18xf32>
    secret.yield %1 : tensor<18xf32>
  } -> !secret.secret<tensor<18xf32>>
  return %0 : !secret.secret<tensor<18xf32>>
}

// -----

func.func @conv2d(%arg0: !secret.secret<tensor<5x5xf32>>) -> !secret.secret<tensor<3x3xf32>> {
  %cst = arith.constant dense<0.0> : tensor<3x3xf32>
  %filter = arith.constant dense<1.0> : tensor<3x3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<5x5xf32>>) {
  ^body(%input0: tensor<5x5xf32>):
    // expected-error@+1 {{Flattened data must fit into a single ciphertext, but got 25 elements and ciphertext size is 16}}
    %1 = linalg.conv_2d ins(%input0, %filter : tensor<5x5xf32>, tensor<3x3xf32>) outs(%cst : tensor<3x3xf32>) -> tensor<3x3xf32>
    secret.yield %1 : tensor<3x3xf32>
  } -> !secret.secret<tensor<3x3xf32>>
  return %0 : !secret.secret<tensor<3x3xf32>>
}
