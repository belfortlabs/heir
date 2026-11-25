// RUN: heir-opt --activation-canonicalizations --canonicalize %s | FileCheck %s

// A clamp to [0, 6], as torch-mlir imports ReLU6: an ordered comparison that
// keeps the larger value, then one that keeps the smaller.

// CHECK: func.func @relu6
// CHECK-SAME: (%[[x:.*]]: f32) -> f32
// CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[six:.*]] = arith.constant 6.000000e+00 : f32
// CHECK: %[[lower:.*]] = arith.maximumf %[[x]], %[[zero]]
// CHECK: %[[upper:.*]] = arith.minimumf %[[lower]], %[[six]]
// CHECK-NOT: arith.select
// CHECK: return %[[upper]]
func.func @relu6(%x: f32) -> f32 {
  %zero = arith.constant 0.0 : f32
  %six = arith.constant 6.0 : f32
  %above = arith.cmpf ogt, %x, %zero : f32
  %lower = arith.select %above, %x, %zero : f32
  %below = arith.cmpf olt, %six, %lower : f32
  %upper = arith.select %below, %six, %lower : f32
  return %upper : f32
}

// The select chooses between other values than the compared pair.

// CHECK: func.func @not_a_clamp
// CHECK: arith.select
func.func @not_a_clamp(%x: f32, %y: f32, %z: f32) -> f32 {
  %c = arith.cmpf ogt, %x, %y : f32
  %r = arith.select %c, %x, %z : f32
  return %r : f32
}
