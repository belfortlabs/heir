// RUN: heir-opt %s --cheddar-emitc-entry-interface --split-input-file --verify-diagnostics

// Reject incomplete or mistyped key-planning metadata before emitting C++.

module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry() attributes {heir.interface = {func_name = "entry", roles = ["entry", "server.evaluate"]}} { return }
  // expected-error@+1 {{cheddar.linear_transform_keys must be an array}}
  func.func @non_array() attributes {cheddar.linear_transform_keys = 1 : i64, heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
  func.func @keygen() attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
  func.func @server_setup() attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
}

// -----

module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry() attributes {heir.interface = {func_name = "entry", roles = ["entry", "server.evaluate"]}} { return }
  // expected-error@+1 {{cheddar.linear_transform_keys entry 0 must be a dictionary}}
  func.func @non_dictionary() attributes {cheddar.linear_transform_keys = [1 : i64], heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
  func.func @keygen() attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
  func.func @server_setup() attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
}

// -----

module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry() attributes {heir.interface = {func_name = "entry", roles = ["entry", "server.evaluate"]}} { return }
  // expected-error@+1 {{cheddar.linear_transform_keys entry 0 requires an i64 width field}}
  func.func @missing_width() attributes {cheddar.linear_transform_keys = [{bs = 0 : i64, gs = 0 : i64, indices = array<i32: 0, 1>, level = 7 : i64}], heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
  func.func @keygen() attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
  func.func @server_setup() attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
}

// -----

module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry() attributes {heir.interface = {func_name = "entry", roles = ["entry", "server.evaluate"]}} { return }
  // expected-error@+1 {{bootstrap key planning requires i64 attribute cheddar.bootstrap_num_cts}}
  func.func @missing_cts() attributes {cheddar.bootstrap_slots = 1024 : i64, heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
  func.func @keygen() attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
  func.func @server_setup() attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
}

// -----

module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry() attributes {heir.interface = {func_name = "entry", roles = ["entry", "server.evaluate"]}} { return }
  // expected-error@+1 {{cheddar.linear_transform_keys entry 0 requires an i64 width field}}
  func.func @wide_width() attributes {cheddar.linear_transform_keys = [{bs = 0 : i64, gs = 0 : i64, indices = array<i32: 0, 1>, level = 7 : i64, width = 18446744073709551616 : i128}], heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
  func.func @keygen() attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
  func.func @server_setup() attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
}
