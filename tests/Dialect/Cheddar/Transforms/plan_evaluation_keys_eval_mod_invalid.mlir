// RUN: heir-opt %s --cheddar-plan-evaluation-keys --verify-diagnostics
// REQUIRES: cyclops-planner

// The default EvalMod approximation consumes 8 levels.
// expected-error@+1 {{the EvalMod approximation consumes 8 levels, but the bootstrap config reserves 7}}
func.func @eval_mod_level_mismatch() attributes {cheddar.rotation_keys = array<i64>, cheddar.bootstrap_slots = 1024 : i64, cheddar.bootstrap_config = #cheddar.bootstrap_config<numCtsLevels = 3, numStcLevels = 2, numEvalModLevels = 7>, heir.interface = {roles = ["client.setup"]}} {
  %params = cheddar.make_parameter {parameterSet = #cheddar.parameter_set<logN = 12, logScale = 20, mainPrimes = [2013265921, 1811939329], auxPrimes = [469762049, 754974721]>} : !cheddar.parameter
  return
}
