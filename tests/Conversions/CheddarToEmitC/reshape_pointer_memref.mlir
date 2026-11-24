// RUN: heir-opt --convert-to-emitc=filter-dialects=cheddar,arith %s | FileCheck --implicit-check-not=memref.collapse_shape --implicit-check-not=memref.expand_shape %s
// RUN: heir-opt --convert-to-emitc=filter-dialects=cheddar,arith %s | heir-translate --mlir-to-cpp | FileCheck --check-prefix=CPP %s

// Reshapes of a primitive buffer reach the conversion directly; the consumer
// reads the source pointer.

// Loads the cheddar dialect, which the conversion requires.
func.func @context(%ctx: !cheddar.context) {
  return
}

// CHECK: func.func @expand_load(%[[ARG:.*]]: !emitc.ptr<f32>
// CHECK: emitc.subscript %[[ARG]]
// CPP-LABEL: float expand_load(float* v1
// CPP: v1[
func.func @expand_load(%arg0: memref<16xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %0 = memref.expand_shape %arg0 [[0, 1]] output_shape [1, 16] : memref<16xf32> into memref<1x16xf32>
  %1 = memref.load %0[%c0, %c5] : memref<1x16xf32>
  return %1 : f32
}

// CHECK: func.func @collapse_load(%[[ARG:.*]]: !emitc.ptr<f32>
// CHECK: emitc.subscript %[[ARG]]
// CPP-LABEL: float collapse_load(float* v1
// CPP: v1[
func.func @collapse_load(%arg0: memref<1x16xf32>) -> f32 {
  %c5 = arith.constant 5 : index
  %0 = memref.collapse_shape %arg0 [[0, 1]] : memref<1x16xf32> into memref<16xf32>
  %1 = memref.load %0[%c5] : memref<16xf32>
  return %1 : f32
}

// A rank-0 collapse goes through the converter's rank-0 branch.
// CHECK: func.func @collapse_rank0(%[[ARG:.*]]: !emitc.ptr<f32>
// CHECK: emitc.subscript %[[ARG]]
func.func @collapse_rank0(%arg0: memref<1x1xf32>) -> f32 {
  %0 = memref.collapse_shape %arg0 [] : memref<1x1xf32> into memref<f32>
  %1 = memref.load %0[] : memref<f32>
  return %1 : f32
}
