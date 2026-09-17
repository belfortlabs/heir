// RUN: heir-opt %s --cheddar-plan-evaluation-keys --split-input-file --verify-diagnostics

// expected-error@+1 {{cheddar.rotation_keys must be a dense i64 array of (distance, level) pairs}}
func.func @missing_rotations() attributes {cheddar.linear_transform_keys = [], heir.interface = {roles = ["client.setup"]}} { return }

// -----

// expected-error@+1 {{cheddar.rotation_keys must be a dense i64 array of (distance, level) pairs}}
func.func @ragged_rotations() attributes {cheddar.rotation_keys = array<i64: 1>, heir.interface = {roles = ["client.setup"]}} { return }

// -----

// expected-error@+1 {{cheddar.linear_transform_keys must be an array}}
func.func @mistyped_transforms() attributes {cheddar.rotation_keys = array<i64>, cheddar.linear_transform_keys = 1 : i64, heir.interface = {roles = ["client.setup"]}} { return }

// -----

// expected-error@+1 {{cheddar.bootstrap_slots must be i64 when planning bootstrap keys}}
func.func @mistyped_bootstrap() attributes {cheddar.rotation_keys = array<i64>, cheddar.bootstrap_slots = "1024", heir.interface = {roles = ["client.setup"]}} { return }

// -----

// expected-error@+1 {{cheddar.bootstrap_slots must be i64 when planning bootstrap keys}}
func.func @orphaned_bootstrap_field() attributes {cheddar.rotation_keys = array<i64>, cheddar.bootstrap_num_cts = 3 : i64, heir.interface = {roles = ["client.setup"]}} { return }

// -----

// expected-error@+1 {{cheddar.bootstrap_num_stc must be i64 when planning bootstrap keys}}
func.func @incomplete_bootstrap() attributes {cheddar.rotation_keys = array<i64>, cheddar.bootstrap_slots = 1024 : i64, cheddar.bootstrap_num_cts = 3 : i64, heir.interface = {roles = ["client.setup"]}} { return }

// -----

// expected-error@+1 {{cheddar.bootstrap_slots must be i64 when planning bootstrap keys}}
func.func @wide_bootstrap_integer() attributes {cheddar.rotation_keys = array<i64>, cheddar.bootstrap_slots = 18446744073709551616 : i128, heir.interface = {roles = ["client.setup"]}} { return }
