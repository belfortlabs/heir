// RUN: heir-opt %s --cheddar-emitc-entry-interface --split-input-file --verify-diagnostics

// Reject a malformed or unresolved evaluation-key list before emitting C++.
// cheddar-plan-evaluation-keys produces cheddar.evaluation_keys and removes the
// requirement attributes it consumed; finding either problem here means the
// module reached the emitter in a state it cannot render.

module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry() attributes {heir.interface = {func_name = "entry", roles = ["entry", "server.evaluate"]}} { return }
  // expected-error@+1 {{still carries cheddar.bootstrap_num_cts; run cheddar-plan-evaluation-keys before emitting the entry interface}}
  func.func @orphaned_bootstrap_field() attributes {cheddar.bootstrap_num_cts = 3 : i64, heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
  func.func @keygen() attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
  func.func @server_setup() attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
}

// -----

module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry() attributes {heir.interface = {func_name = "entry", roles = ["entry", "server.evaluate"]}} { return }
  // expected-error@+1 {{still carries cheddar.rotation_keys; run cheddar-plan-evaluation-keys before emitting the entry interface}}
  func.func @unplanned_rotations() attributes {cheddar.rotation_keys = array<i64: 1, 0>, heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
  func.func @keygen() attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
  func.func @server_setup() attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
}

// -----

module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry() attributes {heir.interface = {func_name = "entry", roles = ["entry", "server.evaluate"]}} { return }
  // expected-error@+1 {{still carries cheddar.linear_transform_keys; run cheddar-plan-evaluation-keys before emitting the entry interface}}
  func.func @unplanned_transforms() attributes {cheddar.linear_transform_keys = [], heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
  func.func @keygen() attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
  func.func @server_setup() attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
}

// -----

module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry() attributes {heir.interface = {func_name = "entry", roles = ["entry", "server.evaluate"]}} { return }
  // expected-error@+1 {{still carries cheddar.bootstrap_slots; run cheddar-plan-evaluation-keys before emitting the entry interface}}
  func.func @unplanned_bootstrap() attributes {cheddar.bootstrap_slots = 1024 : i64, heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
  func.func @keygen() attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
  func.func @server_setup() attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
}

// -----

// A key list must be whole 5-tuples.
module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry() attributes {heir.interface = {func_name = "entry", roles = ["entry", "server.evaluate"]}} { return }
  // expected-error@+1 {{cheddar.evaluation_keys must hold 5-tuples}}
  func.func @ragged() attributes {cheddar.evaluation_keys = array<i64: 0, 1, 2>, heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
  func.func @keygen() attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
  func.func @server_setup() attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
}

// -----

// Family 4 is outside the range of supported key families.
module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry() attributes {heir.interface = {func_name = "entry", roles = ["entry", "server.evaluate"]}} { return }
  // expected-error@+1 {{has key family 4, which is outside 0..3}}
  func.func @bad_family() attributes {cheddar.evaluation_keys = array<i64: 4, 1, 7, 0, -1>, heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
  func.func @keygen() attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
  func.func @server_setup() attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
}

// -----

// Key modes are cyclops::KeyMode, which has three values.
module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry() attributes {heir.interface = {func_name = "entry", roles = ["entry", "server.evaluate"]}} { return }
  // expected-error@+1 {{has key mode 3, which is outside 0..2}}
  func.func @bad_mode() attributes {cheddar.evaluation_keys = array<i64: 0, 1, 7, 3, -1>, heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
  func.func @keygen() attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
  func.func @server_setup() attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
}

// -----

// -1 means unconstrained; nothing below it is meaningful.
module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry() attributes {heir.interface = {func_name = "entry", roles = ["entry", "server.evaluate"]}} { return }
  // expected-error@+1 {{has a required aux count of -2}}
  func.func @bad_aux() attributes {cheddar.evaluation_keys = array<i64: 0, 1, 7, 0, -2>, heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
  func.func @keygen() attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
  func.func @server_setup() attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
}

// -----

// A mistyped list would otherwise emit an empty table and fail at key
// generation instead.
module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry() attributes {heir.interface = {func_name = "entry", roles = ["entry", "server.evaluate"]}} { return }
  // expected-error@+1 {{cheddar.evaluation_keys must be a dense i64 array}}
  func.func @not_an_array() attributes {cheddar.evaluation_keys = 0 : i64, heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
  func.func @keygen() attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
  func.func @server_setup() attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
}
