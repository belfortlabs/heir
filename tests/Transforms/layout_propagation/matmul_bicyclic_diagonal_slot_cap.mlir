// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --apply-config-override="config=max_ring_degree=2048" --layout-propagation="min-slot-count=1024" --split-input-file --verify-diagnostics %s

// The bicyclic diagonal kernel needs the rotation reach plus the whole result:
// 31 * (30 - 1) + 31 * 29 = 1798 slots, more than the 1024 slots of the
// 2048 ring, though no operand has more than 930 elements.
func.func @matmul_secret_lhs(%arg0: !secret.secret<tensor<31x30xf32>>) -> !secret.secret<tensor<31x29xf32>> {
  %cst = arith.constant dense<0.0> : tensor<31x29xf32>
  %w = arith.constant dense<1.0> : tensor<30x29xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<31x30xf32>>) {
  ^body(%x: tensor<31x30xf32>):
    // expected-error@+1 {{needs 1798 slots, but the backend's largest ring degree 2048 holds 1024}}
    %1 = linalg.matmul ins(%x, %w : tensor<31x30xf32>, tensor<30x29xf32>) outs(%cst : tensor<31x29xf32>) -> tensor<31x29xf32>
    secret.yield %1 : tensor<31x29xf32>
  } -> !secret.secret<tensor<31x29xf32>>
  return %0 : !secret.secret<tensor<31x29xf32>>
}

// -----

// With a secret rhs the period is the column count: 29 * (30 - 1) + 31 * 29 =
// 1740 slots.
func.func @matmul_secret_rhs(%arg0: !secret.secret<tensor<30x29xf32>>) -> !secret.secret<tensor<31x29xf32>> {
  %cst = arith.constant dense<0.0> : tensor<31x29xf32>
  %w = arith.constant dense<1.0> : tensor<31x30xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<30x29xf32>>) {
  ^body(%x: tensor<30x29xf32>):
    // expected-error@+1 {{needs 1740 slots, but the backend's largest ring degree 2048 holds 1024}}
    %1 = linalg.matmul ins(%w, %x : tensor<31x30xf32>, tensor<30x29xf32>) outs(%cst : tensor<31x29xf32>) -> tensor<31x29xf32>
    secret.yield %1 : tensor<31x29xf32>
  } -> !secret.secret<tensor<31x29xf32>>
  return %0 : !secret.secret<tensor<31x29xf32>>
}
