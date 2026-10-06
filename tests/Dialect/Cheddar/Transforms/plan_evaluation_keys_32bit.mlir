// RUN: heir-opt %s --cheddar-plan-evaluation-keys | FileCheck %s

// A 32-bit chain whose levels mix main and terminal primes: the planner builds
// the runtime's 32-bit Parameter from the level layout.
// CHECK: func.func @setup()
// CHECK-SAME: cheddar.evaluation_keys = array<i64: 0, 3, 1,
// CHECK-NOT: cheddar.rotation_keys
func.func @setup() attributes {cheddar.rotation_keys = array<i64: 3, 1>, heir.interface = {roles = ["client.setup"]}} {
  %p = cheddar.make_parameter {parameterSet = #cheddar.parameter_set<logN = 12, logScale = 28, mainPrimes = [2013265921, 1811939329, 2113929217, 1711276033], auxPrimes = [469762049, 754974721], terminalPrimes = [167772161, 998244353], levelConfig = [0, 2, 2, 1, 4, 0], wordBits = 32>} : !cheddar.parameter
  return
}
