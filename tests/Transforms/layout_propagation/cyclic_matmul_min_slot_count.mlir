// RUN: heir-opt --annotate-module="backend=cheddar scheme=ckks" --apply-config-override="config=max_ring_degree=65536" --layout-propagation="min-slot-count=128" --split-input-file --verify-diagnostics %s

// Each cyclic kernel needs its rotation reach plus its result in one
// ciphertext. The 65536 ring holds that, but the layouts hold only the
// min-slot-count slots, so the kernel does not fit. The 2-D shapes need one
// slot more than the shapes in cyclic_matmul_slot_fit.mlir.

// Reach 3 * (7 - 1) = 18, plus a 3x37 result: 129 slots.
func.func @bicyclic_diagonal_secret_lhs(%arg0: !secret.secret<tensor<3x7xf32>>) -> !secret.secret<tensor<3x37xf32>> {
  %cst = arith.constant dense<0.0> : tensor<3x37xf32>
  %w = arith.constant dense<1.0> : tensor<7x37xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<3x7xf32>>) {
  ^body(%x: tensor<3x7xf32>):
    // expected-error@+1 {{needs 129 slots, but min-slot-count is 128; raise min-slot-count to at least 129}}
    %1 = linalg.matmul ins(%x, %w : tensor<3x7xf32>, tensor<7x37xf32>) outs(%cst : tensor<3x37xf32>) -> tensor<3x37xf32>
    secret.yield %1 : tensor<3x37xf32>
  } -> !secret.secret<tensor<3x37xf32>>
  return %0 : !secret.secret<tensor<3x37xf32>>
}

// -----

// With a secret rhs the period is the column count: 3 * (37 - 1) = 108, plus a
// 7x3 result: 129 slots.
func.func @bicyclic_diagonal_secret_rhs(%arg0: !secret.secret<tensor<37x3xf32>>) -> !secret.secret<tensor<7x3xf32>> {
  %cst = arith.constant dense<0.0> : tensor<7x3xf32>
  %w = arith.constant dense<1.0> : tensor<7x37xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<37x3xf32>>) {
  ^body(%x: tensor<37x3xf32>):
    // expected-error@+1 {{needs 129 slots, but min-slot-count is 128; raise min-slot-count to at least 129}}
    %1 = linalg.matmul ins(%w, %x : tensor<7x37xf32>, tensor<37x3xf32>) outs(%cst : tensor<7x3xf32>) -> tensor<7x3xf32>
    secret.yield %1 : tensor<7x3xf32>
  } -> !secret.secret<tensor<7x3xf32>>
  return %0 : !secret.secret<tensor<7x3xf32>>
}

// -----

// Reach 2 * 11 - 1 + 9 * (2 - 1) = 30, plus a 9x11 result: 129 slots, though
// no operand or result has more than 99 elements.
func.func @bicyclic(%a: !secret.secret<tensor<9x2xf32>>, %b: !secret.secret<tensor<2x11xf32>>) -> !secret.secret<tensor<9x11xf32>> {
  %cst = arith.constant dense<0.0> : tensor<9x11xf32>
  %0 = secret.generic(%a : !secret.secret<tensor<9x2xf32>>, %b : !secret.secret<tensor<2x11xf32>>) {
  ^body(%x: tensor<9x2xf32>, %y: tensor<2x11xf32>):
    // expected-error@+1 {{needs 129 slots, but min-slot-count is 128; raise min-slot-count to at least 129}}
    %1 = linalg.matmul ins(%x, %y : tensor<9x2xf32>, tensor<2x11xf32>) outs(%cst : tensor<9x11xf32>) -> tensor<9x11xf32>
    secret.yield %1 : tensor<9x11xf32>
  } -> !secret.secret<tensor<9x11xf32>>
  return %0 : !secret.secret<tensor<9x11xf32>>
}

// -----

// Reach 2 * 5 * 7 - 1 + 2 * 3 * (5 - 1) = 93, plus a 2x3x7 result: 135 slots,
// though no operand or result has more than 70 elements.
func.func @tricyclic(%a: !secret.secret<tensor<2x3x5xf32>>, %b: !secret.secret<tensor<2x5x7xf32>>) -> !secret.secret<tensor<2x3x7xf32>> {
  %cst = arith.constant dense<0.0> : tensor<2x3x7xf32>
  %0 = secret.generic(%a : !secret.secret<tensor<2x3x5xf32>>, %b : !secret.secret<tensor<2x5x7xf32>>) {
  ^body(%x: tensor<2x3x5xf32>, %y: tensor<2x5x7xf32>):
    // expected-error@+1 {{needs 135 slots, but min-slot-count is 128; raise min-slot-count to at least 135}}
    %1 = linalg.batch_matmul ins(%x, %y : tensor<2x3x5xf32>, tensor<2x5x7xf32>) outs(%cst : tensor<2x3x7xf32>) -> tensor<2x3x7xf32>
    secret.yield %1 : tensor<2x3x7xf32>
  } -> !secret.secret<tensor<2x3x7xf32>>
  return %0 : !secret.secret<tensor<2x3x7xf32>>
}

// -----

// Reach 2 * 3 * (5 - 1) = 24, plus a 2x3x19 result: 138 slots.
func.func @tricyclic_diagonal(%arg0: !secret.secret<tensor<2x3x5xf32>>) -> !secret.secret<tensor<2x3x19xf32>> {
  %cst = arith.constant dense<0.0> : tensor<2x3x19xf32>
  %w = arith.constant dense<1.0> : tensor<2x5x19xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<2x3x5xf32>>) {
  ^body(%x: tensor<2x3x5xf32>):
    // expected-error@+1 {{needs 138 slots, but min-slot-count is 128; raise min-slot-count to at least 138}}
    %1 = linalg.batch_matmul ins(%x, %w : tensor<2x3x5xf32>, tensor<2x5x19xf32>) outs(%cst : tensor<2x3x19xf32>) -> tensor<2x3x19xf32>
    secret.yield %1 : tensor<2x3x19xf32>
  } -> !secret.secret<tensor<2x3x19xf32>>
  return %0 : !secret.secret<tensor<2x3x19xf32>>
}
