// RUN: heir-opt --count-ops %s 2>&1 | FileCheck %s

// @helper runs one addition and two rotations in each of its four inner
// iterations. @main calls it three times in a loop.

// CHECK-DAG: count-ops{{[[:space:]]}}helper{{[[:space:]]}}arith.addf{{[[:space:]]}}4{{$}}
// CHECK-DAG: count-ops{{[[:space:]]}}helper{{[[:space:]]}}tensor_ext.rotate{{[[:space:]]}}8{{$}}
// CHECK-DAG: count-ops{{[[:space:]]}}helper{{[[:space:]]}}distinct-shifts{{[[:space:]]}}2{{$}}
// CHECK-DAG: count-ops{{[[:space:]]}}main{{[[:space:]]}}arith.addf{{[[:space:]]}}12{{$}}
// CHECK-DAG: count-ops{{[[:space:]]}}main{{[[:space:]]}}tensor_ext.rotate{{[[:space:]]}}24{{$}}
// CHECK-DAG: count-ops{{[[:space:]]}}main{{[[:space:]]}}func.call{{[[:space:]]}}3{{$}}
// CHECK-DAG: count-ops{{[[:space:]]}}main{{[[:space:]]}}distinct-shifts{{[[:space:]]}}2{{$}}
func.func @helper(%x: tensor<16xf32>) -> tensor<16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c2 iter_args(%acc = %x) -> (tensor<16xf32>) {
    %inner = affine.for %j = 0 to 2 iter_args(%acc2 = %acc) -> (tensor<16xf32>) {
      %rot1 = tensor_ext.rotate %acc2, %c1 : tensor<16xf32>, index
      %rot4 = tensor_ext.rotate %rot1, %c4 : tensor<16xf32>, index
      %sum = arith.addf %rot4, %acc2 : tensor<16xf32>
      affine.yield %sum : tensor<16xf32>
    }
    scf.yield %inner : tensor<16xf32>
  }
  return %r : tensor<16xf32>
}

func.func @main(%x: tensor<16xf32>) -> tensor<16xf32> {
  %r = affine.for %i = 0 to 3 iter_args(%acc = %x) -> (tensor<16xf32>) {
    %y = func.call @helper(%acc) : (tensor<16xf32>) -> tensor<16xf32>
    affine.yield %y : tensor<16xf32>
  }
  return %r : tensor<16xf32>
}
