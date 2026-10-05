// RUN: heir-opt %s --cheddar-plan-evaluation-keys --split-input-file --verify-diagnostics

// expected-error@+1 {{cheddar.rotation_keys must be a dense i64 array of (distance, level) pairs}}
func.func @missing_rotations() attributes {cheddar.linear_transform_keys = [], heir.interface = {roles = ["client.setup"]}} { return }

// -----

// expected-error@+1 {{cheddar.rotation_keys must be a dense i64 array of (distance, level) pairs}}
func.func @ragged_rotations() attributes {cheddar.rotation_keys = array<i64: 1>, heir.interface = {roles = ["client.setup"]}} { return }

// -----

// expected-error@+1 {{cheddar.multiplication_keys must be a dense i64 array of levels}}
func.func @mistyped_multiplication_keys() attributes {cheddar.rotation_keys = array<i64>, cheddar.multiplication_keys = [1], heir.interface = {roles = ["client.setup"]}} { return }

// -----

// expected-error@+1 {{cheddar.linear_transform_keys must be an array}}
func.func @mistyped_transforms() attributes {cheddar.rotation_keys = array<i64>, cheddar.linear_transform_keys = 1 : i64, heir.interface = {roles = ["client.setup"]}} { return }

// -----

// expected-error@+1 {{cheddar.bootstrap_slots must be i64 when planning bootstrap keys}}
func.func @mistyped_bootstrap() attributes {cheddar.rotation_keys = array<i64>, cheddar.bootstrap_slots = "1024", heir.interface = {roles = ["client.setup"]}} { return }

// -----

// expected-error@+1 {{cheddar.bootstrap_slots must be i64 when planning bootstrap keys}}
func.func @orphaned_bootstrap_config() attributes {cheddar.rotation_keys = array<i64>, cheddar.bootstrap_config = #cheddar.bootstrap_config<numCtsLevels = 3, numStcLevels = 2, numEvalModLevels = 8>, heir.interface = {roles = ["client.setup"]}} { return }

// -----

// expected-error@+1 {{cheddar.bootstrap_config must be a #cheddar.bootstrap_config when planning bootstrap keys}}
func.func @incomplete_bootstrap() attributes {cheddar.rotation_keys = array<i64>, cheddar.bootstrap_slots = 1024 : i64, heir.interface = {roles = ["client.setup"]}} { return }

// -----

// expected-error@+1 {{cheddar.bootstrap_config must be a #cheddar.bootstrap_config when planning bootstrap keys}}
func.func @mistyped_bootstrap_config() attributes {cheddar.rotation_keys = array<i64>, cheddar.bootstrap_slots = 1024 : i64, cheddar.bootstrap_config = 3 : i64, heir.interface = {roles = ["client.setup"]}} { return }

// -----

// expected-error@+1 {{cheddar.bootstrap_slots must be i64 when planning bootstrap keys}}
func.func @wide_bootstrap_integer() attributes {cheddar.rotation_keys = array<i64>, cheddar.bootstrap_slots = 18446744073709551616 : i128, heir.interface = {roles = ["client.setup"]}} { return }

// -----

func.func @empty_aux_primes() attributes {cheddar.rotation_keys = array<i64>, heir.interface = {roles = ["client.setup"]}} {
  // expected-error@+1 {{requires at least one auxiliary prime for Cyclops key planning}}
  %params = cheddar.make_parameter {parameterSet = #cheddar.parameter_set<logN = 14, logScale = 30, mainPrimes = [65537], auxPrimes = []>} : !cheddar.parameter
  return
}

// -----

// The default EvalMod approximation consumes 8 levels.
// expected-error@+1 {{the EvalMod approximation consumes 8 levels, but the bootstrap config reserves 7}}
func.func @eval_mod_level_mismatch() attributes {cheddar.rotation_keys = array<i64>, cheddar.bootstrap_slots = 1024 : i64, cheddar.bootstrap_config = #cheddar.bootstrap_config<numCtsLevels = 3, numStcLevels = 2, numEvalModLevels = 7>, heir.interface = {roles = ["client.setup"]}} {
  %params = cheddar.make_parameter {parameterSet = #cheddar.parameter_set<logN = 12, logScale = 20, mainPrimes = [2013265921, 1811939329], auxPrimes = [469762049, 754974721]>} : !cheddar.parameter
  return
}
