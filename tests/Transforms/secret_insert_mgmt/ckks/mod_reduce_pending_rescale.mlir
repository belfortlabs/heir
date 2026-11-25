// RUN: heir-opt --secret-insert-mgmt-ckks="after-mul=false level-budget=12 min-slot-count=1024" %s | FileCheck %s
// RUN: heir-opt --secret-insert-mgmt-ckks="after-mul=false before-mul-include-first-mul=true level-budget=12 min-slot-count=1024" %s | FileCheck %s --check-prefix=FIRST

// With mod reduce before multiplication, a mod reduce is only inserted for a
// value that has a pending rescale: the result of a multiplication that was
// not mod reduced yet, or, when the first multiplication is included, an
// input. kernel.linear_transform and kernel.eval_chebyshev rescale their own
// result, so their results need no mod reduce.

module attributes {backend.lattigo, scheme.ckks} {
  // A chain of transforms and polynomials needs no mod reduce.
  // CHECK: func.func @transform_poly_transform
  // CHECK-SAME: level = 4
  // CHECK-NOT: mgmt.modreduce
  // CHECK: secret.yield

  // FIRST: func.func @transform_poly_transform
  // FIRST-SAME: level = 5
  // FIRST: ^body(%[[x:.*]]: tensor
  // FIRST-NEXT: %[[x_mr:.*]] = mgmt.modreduce %[[x]]
  // FIRST-NEXT: kernel.linear_transform %[[x_mr]]
  // FIRST-NOT: mgmt.modreduce
  // FIRST: secret.yield
  func.func @transform_poly_transform(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
    %diagonals = arith.constant dense<1.0> : tensor<2x1024xf32>
    %res = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%x: tensor<1x1024xf32>):
      %0 = kernel.linear_transform %x, %diagonals <diagonal_indices = [0, 1]> : tensor<1x1024xf32>, tensor<2x1024xf32> -> tensor<1x1024xf32>
      %1 = kernel.eval_chebyshev %0 <coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64, 4.0 : f64]> : tensor<1x1024xf32> -> tensor<1x1024xf32>
      %2 = kernel.linear_transform %1, %diagonals <diagonal_indices = [0, 1]> : tensor<1x1024xf32>, tensor<2x1024xf32> -> tensor<1x1024xf32>
      secret.yield %2 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %res : !secret.secret<tensor<1x1024xf32>>
  }

  // A sum of transform results is multiplied without a mod reduce; the product
  // is mod reduced before it is returned.
  // CHECK: func.func @add_of_transforms_then_mul
  // CHECK-NOT: mgmt.modreduce
  // CHECK: %[[sum:.*]] = arith.addf
  // CHECK-NEXT: %[[mul:.*]] = arith.mulf %[[sum]], %[[sum]]
  // CHECK-NEXT: %[[relin:.*]] = mgmt.relinearize %[[mul]]
  // CHECK-NEXT: %[[mr:.*]] = mgmt.modreduce %[[relin]]
  // CHECK-NEXT: secret.yield %[[mr]]

  // FIRST: func.func @add_of_transforms_then_mul
  // FIRST: %[[sum:.*]] = arith.addf
  // FIRST-NEXT: %[[mul:.*]] = arith.mulf %[[sum]], %[[sum]]
  // FIRST-NEXT: %[[relin:.*]] = mgmt.relinearize %[[mul]]
  // FIRST-NEXT: %[[mr:.*]] = mgmt.modreduce %[[relin]]
  // FIRST-NEXT: secret.yield %[[mr]]
  func.func @add_of_transforms_then_mul(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
    %diagonals = arith.constant dense<1.0> : tensor<2x1024xf32>
    %res = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%x: tensor<1x1024xf32>):
      %0 = kernel.linear_transform %x, %diagonals <diagonal_indices = [0, 1]> : tensor<1x1024xf32>, tensor<2x1024xf32> -> tensor<1x1024xf32>
      %1 = kernel.linear_transform %x, %diagonals <diagonal_indices = [0, 2]> : tensor<1x1024xf32>, tensor<2x1024xf32> -> tensor<1x1024xf32>
      %2 = arith.addf %0, %1 : tensor<1x1024xf32>
      %3 = arith.mulf %2, %2 : tensor<1x1024xf32>
      secret.yield %3 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %res : !secret.secret<tensor<1x1024xf32>>
  }

  // A product is mod reduced before a transform, whose result is then returned
  // as it is.
  // CHECK: func.func @mul_then_transform
  // CHECK: %[[mul:.*]] = arith.mulf
  // CHECK-NEXT: %[[relin:.*]] = mgmt.relinearize %[[mul]]
  // CHECK-NEXT: %[[mr:.*]] = mgmt.modreduce %[[relin]]
  // CHECK-NEXT: %[[lt:.*]] = kernel.linear_transform %[[mr]]
  // CHECK-NEXT: secret.yield %[[lt]]

  // FIRST: func.func @mul_then_transform
  // FIRST: %[[mul:.*]] = arith.mulf
  // FIRST-NEXT: %[[relin:.*]] = mgmt.relinearize %[[mul]]
  // FIRST-NEXT: %[[mr:.*]] = mgmt.modreduce %[[relin]]
  // FIRST-NEXT: %[[lt:.*]] = kernel.linear_transform %[[mr]]
  // FIRST-NEXT: secret.yield %[[lt]]
  func.func @mul_then_transform(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
    %diagonals = arith.constant dense<1.0> : tensor<2x1024xf32>
    %res = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%x: tensor<1x1024xf32>):
      %0 = arith.mulf %x, %x : tensor<1x1024xf32>
      %1 = kernel.linear_transform %0, %diagonals <diagonal_indices = [0, 1]> : tensor<1x1024xf32>, tensor<2x1024xf32> -> tensor<1x1024xf32>
      secret.yield %1 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %res : !secret.secret<tensor<1x1024xf32>>
  }

  // A product is mod reduced before a polynomial, whose result is multiplied
  // without another mod reduce.
  // CHECK: func.func @mul_poly_mul
  // CHECK: %[[mul:.*]] = arith.mulf
  // CHECK-NEXT: %[[relin:.*]] = mgmt.relinearize %[[mul]]
  // CHECK-NEXT: %[[mr:.*]] = mgmt.modreduce %[[relin]]
  // CHECK-NEXT: %[[poly:.*]] = kernel.eval_chebyshev %[[mr]]
  // CHECK-NEXT: %[[mul2:.*]] = arith.mulf %[[poly]], %[[poly]]
  // CHECK-NEXT: %[[relin2:.*]] = mgmt.relinearize %[[mul2]]
  // CHECK-NEXT: %[[mr2:.*]] = mgmt.modreduce %[[relin2]]
  // CHECK-NEXT: secret.yield %[[mr2]]

  // FIRST: func.func @mul_poly_mul
  // FIRST: %[[poly:.*]] = kernel.eval_chebyshev
  // FIRST-NEXT: %[[mul2:.*]] = arith.mulf %[[poly]], %[[poly]]
  func.func @mul_poly_mul(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
    %res = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%x: tensor<1x1024xf32>):
      %0 = arith.mulf %x, %x : tensor<1x1024xf32>
      %1 = kernel.eval_chebyshev %0 <coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64, 4.0 : f64]> : tensor<1x1024xf32> -> tensor<1x1024xf32>
      %2 = arith.mulf %1, %1 : tensor<1x1024xf32>
      secret.yield %2 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %res : !secret.secret<tensor<1x1024xf32>>
  }

  // Only the operand that has a pending rescale is mod reduced, which brings it
  // to the level of the transform result.
  // CHECK: func.func @mul_of_rescaled_and_pending
  // CHECK: %[[lt:.*]] = kernel.linear_transform
  // CHECK-SAME: level = 3
  // CHECK-NEXT: %[[mul:.*]] = arith.mulf
  // CHECK-NEXT: %[[relin:.*]] = mgmt.relinearize %[[mul]]
  // CHECK-NEXT: %[[mr:.*]] = mgmt.modreduce %[[relin]]
  // CHECK-SAME: level = 3
  // CHECK-NEXT: arith.mulf %[[lt]], %[[mr]]
  // CHECK-SAME: level = 3
  func.func @mul_of_rescaled_and_pending(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
    %diagonals = arith.constant dense<1.0> : tensor<2x1024xf32>
    %res = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%x: tensor<1x1024xf32>):
      %0 = kernel.linear_transform %x, %diagonals <diagonal_indices = [0, 1]> : tensor<1x1024xf32>, tensor<2x1024xf32> -> tensor<1x1024xf32>
      %1 = arith.mulf %x, %x : tensor<1x1024xf32>
      %2 = arith.mulf %0, %1 : tensor<1x1024xf32>
      secret.yield %2 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %res : !secret.secret<tensor<1x1024xf32>>
  }

  // The polynomial result is two levels below the input and is not mod reduced;
  // the product is brought down to its level.
  // CHECK: func.func @mul_of_deeper_rescaled_and_pending
  // CHECK: %[[poly:.*]] = kernel.eval_chebyshev
  // CHECK-SAME: level = 2
  // CHECK-NEXT: %[[mul:.*]] = arith.mulf
  // CHECK-NEXT: %[[relin:.*]] = mgmt.relinearize %[[mul]]
  // CHECK-NEXT: %[[lr:.*]] = mgmt.level_reduce %[[relin]]
  // CHECK-NEXT: %[[mr:.*]] = mgmt.modreduce %[[lr]]
  // CHECK-SAME: level = 2
  // CHECK-NEXT: arith.mulf %[[poly]], %[[mr]]
  // CHECK-SAME: level = 2
  func.func @mul_of_deeper_rescaled_and_pending(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
    %res = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%x: tensor<1x1024xf32>):
      %0 = kernel.eval_chebyshev %x <coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64, 4.0 : f64]> : tensor<1x1024xf32> -> tensor<1x1024xf32>
      %1 = arith.mulf %x, %x : tensor<1x1024xf32>
      %2 = arith.mulf %0, %1 : tensor<1x1024xf32>
      secret.yield %2 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %res : !secret.secret<tensor<1x1024xf32>>
  }
}
