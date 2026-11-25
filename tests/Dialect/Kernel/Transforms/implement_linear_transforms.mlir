// RUN: heir-opt --implement-linear-transforms %s | FileCheck %s --check-prefix=ROLLED
// RUN: heir-opt --implement-linear-transforms=unroll=true %s | FileCheck %s --check-prefix=UNROLLED

// Two diagonals share the baby step (input 0, shift 1) and two share the
// group (output 1, giant step 0), so the transform needs two ciphertext
// rotations: input 0 by 1 and input 1 by 2.

// ROLLED: @block
// ROLLED-SAME: (%[[X:.*]]: tensor<2x8xf32>, %[[D:.*]]: tensor<4x8xf32>)
// ROLLED-NOT: kernel.block_linear_transform
// Baby steps.
// ROLLED: scf.for
// ROLLED: tensor.extract_slice %[[X]]
// ROLLED: tensor_ext.rotate
// ROLLED: tensor.insert_slice
// One multiplication per diagonal, in a loop over the rows of the plan.
// ROLLED: scf.for
// ROLLED: tensor.extract_slice %[[D]]
// ROLLED: arith.mulf
// ROLLED: scf.if
// ROLLED: return

// UNROLLED: @block
// UNROLLED-SAME: (%[[X:.*]]: tensor<2x8xf32>, %[[D:.*]]: tensor<4x8xf32>)
// UNROLLED-NOT: scf.for
// UNROLLED-COUNT-2: tensor_ext.rotate
// UNROLLED-NOT: tensor_ext.rotate
// UNROLLED: return
func.func @block(%x: tensor<2x8xf32>, %diagonals: tensor<4x8xf32>) -> tensor<2x8xf32> {
  %0 = kernel.block_linear_transform %x, %diagonals <
    output_ciphertexts = [0, 1, 1, 1],
    input_ciphertexts = [0, 0, 0, 1],
    shifts = [1, 0, 1, 2]
  > : tensor<2x8xf32>, tensor<4x8xf32> -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
