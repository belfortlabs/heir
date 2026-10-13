// RUN: heir-opt --layout-propagation=min-slot-count=128 %s | FileCheck %s --check-prefix=LAYOUT
// RUN: heir-opt --layout-propagation=min-slot-count=128 --convert-to-ciphertext-semantics=min-slot-count=128 %s | FileCheck %s --check-prefix=CTCS

// Each cyclic kernel needs its rotation reach plus its result in one
// ciphertext. The shapes below need exactly the 128 slots of min-slot-count
// (except the tricyclic diagonal one, which no shape fits exactly), so layout
// propagation still selects the cyclic kernel, and the lowering finds a valid
// prefix that holds the whole result. cyclic_matmul_min_slot_count.mlir has the
// shapes that need one slot more.

// LAYOUT-DAG: #[[BICYCLIC:kernel[0-9]*]] = #secret.kernel<name = "MatmulBicyclic", force = false>
// LAYOUT-DAG: #[[BICYCLIC_DIAG:kernel[0-9]*]] = #secret.kernel<name = "MatmulBicyclicDiagonal", force = false>
// LAYOUT-DAG: #[[TRICYCLIC:kernel[0-9]*]] = #secret.kernel<name = "BatchMatmulTricyclic", force = false>
// LAYOUT-DAG: #[[TRICYCLIC_DIAG:kernel[0-9]*]] = #secret.kernel<name = "BatchMatmulTricyclicDiagonal", force = false>

// Reach 1 * (2 - 1) = 1, plus a 1x127 result: 128 slots.
// LAYOUT: @bicyclic_diagonal
// LAYOUT: linalg.matmul {{.*}}secret.kernel = #[[BICYCLIC_DIAG]]
// CTCS: @bicyclic_diagonal
// CTCS-NOT: linalg.matmul
// CTCS: tensor_ext.remap
func.func @bicyclic_diagonal(%arg0: !secret.secret<tensor<1x2xf32>>) -> !secret.secret<tensor<1x127xf32>> {
  %cst = arith.constant dense<0.0> : tensor<1x127xf32>
  %w = arith.constant dense<1.0> : tensor<2x127xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x2xf32>>) {
  ^body(%x: tensor<1x2xf32>):
    %1 = linalg.matmul ins(%x, %w : tensor<1x2xf32>, tensor<2x127xf32>) outs(%cst : tensor<1x127xf32>) -> tensor<1x127xf32>
    secret.yield %1 : tensor<1x127xf32>
  } -> !secret.secret<tensor<1x127xf32>>
  return %0 : !secret.secret<tensor<1x127xf32>>
}

// Reach 3 * 25 - 1 + 2 * (3 - 1) = 78, plus a 2x25 result: 128 slots.
// LAYOUT: @bicyclic
// LAYOUT: linalg.matmul {{.*}}secret.kernel = #[[BICYCLIC]]
// CTCS: @bicyclic
// CTCS-NOT: linalg.matmul
// CTCS: tensor_ext.remap
func.func @bicyclic(%a: !secret.secret<tensor<2x3xf32>>, %b: !secret.secret<tensor<3x25xf32>>) -> !secret.secret<tensor<2x25xf32>> {
  %cst = arith.constant dense<0.0> : tensor<2x25xf32>
  %0 = secret.generic(%a : !secret.secret<tensor<2x3xf32>>, %b : !secret.secret<tensor<3x25xf32>>) {
  ^body(%x: tensor<2x3xf32>, %y: tensor<3x25xf32>):
    %1 = linalg.matmul ins(%x, %y : tensor<2x3xf32>, tensor<3x25xf32>) outs(%cst : tensor<2x25xf32>) -> tensor<2x25xf32>
    secret.yield %1 : tensor<2x25xf32>
  } -> !secret.secret<tensor<2x25xf32>>
  return %0 : !secret.secret<tensor<2x25xf32>>
}

// Reach 1 * 3 * 25 - 1 + 1 * 2 * (3 - 1) = 78, plus a 1x2x25 result: 128 slots.
// LAYOUT: @tricyclic
// LAYOUT: linalg.batch_matmul {{.*}}secret.kernel = #[[TRICYCLIC]]
// CTCS: @tricyclic
// CTCS-NOT: linalg.batch_matmul
// CTCS: tensor_ext.remap
func.func @tricyclic(%a: !secret.secret<tensor<1x2x3xf32>>, %b: !secret.secret<tensor<1x3x25xf32>>) -> !secret.secret<tensor<1x2x25xf32>> {
  %cst = arith.constant dense<0.0> : tensor<1x2x25xf32>
  %0 = secret.generic(%a : !secret.secret<tensor<1x2x3xf32>>, %b : !secret.secret<tensor<1x3x25xf32>>) {
  ^body(%x: tensor<1x2x3xf32>, %y: tensor<1x3x25xf32>):
    %1 = linalg.batch_matmul ins(%x, %y : tensor<1x2x3xf32>, tensor<1x3x25xf32>) outs(%cst : tensor<1x2x25xf32>) -> tensor<1x2x25xf32>
    secret.yield %1 : tensor<1x2x25xf32>
  } -> !secret.secret<tensor<1x2x25xf32>>
  return %0 : !secret.secret<tensor<1x2x25xf32>>
}

// Reach 2 * 3 * (5 - 1) = 24, plus a 2x3x7 result: 66 slots.
// LAYOUT: @tricyclic_diagonal
// LAYOUT: linalg.batch_matmul {{.*}}secret.kernel = #[[TRICYCLIC_DIAG]]
// CTCS: @tricyclic_diagonal
// CTCS-NOT: linalg.batch_matmul
// CTCS: tensor_ext.remap
func.func @tricyclic_diagonal(%arg0: !secret.secret<tensor<2x3x5xf32>>) -> !secret.secret<tensor<2x3x7xf32>> {
  %cst = arith.constant dense<0.0> : tensor<2x3x7xf32>
  %w = arith.constant dense<1.0> : tensor<2x5x7xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<2x3x5xf32>>) {
  ^body(%x: tensor<2x3x5xf32>):
    %1 = linalg.batch_matmul ins(%x, %w : tensor<2x3x5xf32>, tensor<2x5x7xf32>) outs(%cst : tensor<2x3x7xf32>) -> tensor<2x3x7xf32>
    secret.yield %1 : tensor<2x3x7xf32>
  } -> !secret.secret<tensor<2x3x7xf32>>
  return %0 : !secret.secret<tensor<2x3x7xf32>>
}
