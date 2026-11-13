// RUN: heir-opt %s --cheddar-plan-evaluation-keys | FileCheck %s
// REQUIRES: cyclops-planner

// The requested rotation by 3 at level 0, and the rotations the width-8
// linear transform at level 1 needs, as (family, rotation, level, key mode,
// required num aux) tuples. The planning attributes are removed.
// CHECK: func.func @setup()
// CHECK-SAME: cheddar.evaluation_keys = array<i64: 0, 1, 1, 2, 1, 0, 2, 1, 2, 1, 0, 3, 0, 0, -1>
// CHECK-NOT: cheddar.rotation_keys
// CHECK-NOT: cheddar.linear_transform_keys
func.func @setup() attributes {cheddar.rotation_keys = array<i64: 3, 0>, cheddar.linear_transform_keys = [{indices = array<i32: 0, 1, 2>, width = 8 : i64, level = 1 : i64, bs = 0 : i64, gs = 0 : i64}], heir.interface = {roles = ["client.setup"]}} {
  %p = cheddar.make_parameter {parameterSet = #cheddar.parameter_set<logN = 12, logScale = 20, mainPrimes = [2013265921, 1811939329], auxPrimes = [469762049, 754974721]>} : !cheddar.parameter
  return
}
