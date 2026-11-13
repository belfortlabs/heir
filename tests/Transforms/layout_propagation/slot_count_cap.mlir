// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --layout-propagation --verify-diagnostics %s
// RUN: heir-opt --annotate-module="backend=openfhe scheme=ckks" --apply-config-override="config=max_ring_degree=65536" --layout-propagation --verify-diagnostics %s

// A diagonal of the 4x40000 matrix needs 40000 slots, more than the 32768 slots
// of the logN = 16 ring.
func.func @matvec(%arg0: !secret.secret<tensor<40000xf32>>) -> !secret.secret<tensor<4xf32>> {
  %cst = arith.constant dense<0.0> : tensor<4xf32>
  %matrix = arith.constant dense<1.0> : tensor<4x40000xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<40000xf32>>) {
  ^body(%input0: tensor<40000xf32>):
    // expected-error@+1 {{needs 40000 slots, but the backend's largest ring degree 65536 holds 32768}}
    %1 = linalg.matvec ins(%matrix, %input0 : tensor<4x40000xf32>, tensor<40000xf32>) outs(%cst : tensor<4xf32>) -> tensor<4xf32>
    secret.yield %1 : tensor<4xf32>
  } -> !secret.secret<tensor<4xf32>>
  return %0 : !secret.secret<tensor<4xf32>>
}
