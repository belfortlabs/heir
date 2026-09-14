// RUN: heir-opt --cheddar-build-entry-interface %s | FileCheck %s

// A Cyclops-style entry: the client owns a ClientContext, the server a
// BootContext; preprocessing loads resources and encodes with a client
// context; a cleartext scalar and an encrypted zero reach the evaluation.

!ct = !cheddar.ciphertext
!pt = !cheddar.plaintext
!context = !cheddar.context
!boot_context = !cheddar.boot_context
!client_context = !cheddar.client_context
!encoder = !cheddar.encoder
!ui = !cheddar.user_interface
!evk_map = !cheddar.evk_map
!debug = !cheddar.debug_handler
!dir = !preprocessing.resource_dir

func.func @entry__setup() -> tensor<!client_context>
    attributes {heir.interface = {func_name = "entry", roles = ["client.setup"]}} {
  %ctx = tensor.empty() : tensor<!client_context>
  return %ctx : tensor<!client_context>
}
func.func @entry__server_setup() -> tensor<!boot_context>
    attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} {
  %ctx = tensor.empty() : tensor<!boot_context>
  return %ctx : tensor<!boot_context>
}
func.func @entry__keygen(%ctx: tensor<!client_context>) -> (tensor<!client_context>, tensor<!ui>)
    attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} {
  %ui = tensor.empty() : tensor<!ui>
  return %ctx, %ui : tensor<!client_context>, tensor<!ui>
}
func.func @entry(%ctx: !boot_context {cheddar.support = "boot_context"},
                 %input: tensor<1x!ct>, %flag: i1,
                 %zero: tensor<1x!ct> {client.enc_zero_arg = {func_name = "entry", index = 0 : i64}}) -> tensor<1x!ct>
    attributes {heir.interface = {func_name = "entry", input_types = [tensor<4xf32>, i1], result_types = [tensor<4xf32>], roles = ["entry"]}} {
  return %input : tensor<1x!ct>
}
func.func @entry__preprocessing(%ctx: !client_context {cheddar.support = "client_context"}, %encoder: !encoder {cheddar.support = "encoder"}, %dir: !dir) -> tensor<1x!pt>
    attributes {heir.interface = {func_name = "entry", roles = ["server.preprocessing"]}} {
  %pt = tensor.empty() : tensor<1x!pt>
  return %pt : tensor<1x!pt>
}
func.func @entry__preprocessed(%ctx: !boot_context {cheddar.support = "boot_context"}, %evk_map: !evk_map {cheddar.support = "evk_map"}, %debug: !debug {cheddar.support = "debug_handler"},
                               %input: tensor<1x!ct>, %flag: i1, %zero: tensor<1x!ct>, %pt: tensor<1x!pt>) -> tensor<1x!ct>
    attributes {heir.interface = {func_name = "entry", roles = ["server.evaluate"]}} {
  return %input : tensor<1x!ct>
}
func.func @entry__encrypt__arg0(%ctx: !client_context {cheddar.support = "client_context"}, %encoder: !encoder {cheddar.support = "encoder"}, %ui: !ui {cheddar.support = "user_interface"}, %input: tensor<4xf32>) -> tensor<1x!ct>
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt"]}} {
  %ct = tensor.empty() : tensor<1x!ct>
  return %ct : tensor<1x!ct>
}
func.func @entry__encrypt__zero__0(%ctx: !client_context {cheddar.support = "client_context"}, %encoder: !encoder {cheddar.support = "encoder"}, %ui: !ui {cheddar.support = "user_interface"}) -> tensor<1x!ct>
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt_zero"]}} {
  %ct = tensor.empty() : tensor<1x!ct>
  return %ct : tensor<1x!ct>
}
func.func @entry__decrypt__result0(%ctx: !client_context {cheddar.support = "client_context"}, %encoder: !encoder {cheddar.support = "encoder"}, %ui: !ui {cheddar.support = "user_interface"}, %input: tensor<1x!ct>) -> tensor<4xf32>
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.decrypt"]}} {
  %cst = arith.constant dense<0.0> : tensor<4xf32>
  return %cst : tensor<4xf32>
}

// The cleartext flag has no helper and is left to the caller; the encrypted
// zero takes no input.
// CHECK: func.func @entry__encrypt_inputs(%[[CTX:[^:]+]]: !client_context {cheddar.support = "client_context"}, %[[UI:[^:]+]]: !user_interface {cheddar.support = "user_interface"}, %[[IN:[^:]+]]: tensor<4xf32> {cheddar.entry_input = 0 : i64}) -> (tensor<1x!ciphertext>, tensor<1x!ciphertext>)
// CHECK: %[[ENC:.*]] = cheddar.get_encoder %[[CTX]]
// CHECK: %[[CT:.*]] = call @entry__encrypt__arg0(%[[CTX]], %[[ENC]], %[[UI]], %[[IN]])
// CHECK: %[[ZERO:.*]] = call @entry__encrypt__zero__0(%[[CTX]], %[[ENC]], %[[UI]])
// CHECK: return %[[CT]], %[[ZERO]]

// CHECK: func.func @entry__decrypt_outputs

// CHECK: func.func @entry__evaluate(%[[CTX:[^:]+]]: !boot_context {cheddar.support = "boot_context"}, %[[MAP:[^:]+]]: !evk_map {cheddar.support = "evk_map"}, %[[DBG:[^:]+]]: !debug_handler {cheddar.support = "debug_handler"}, %[[IN:[^:]+]]: tensor<1x!ciphertext> {cheddar.entry_input = 0 : i64}, %[[FLAG:[^:]+]]: i1 {cheddar.entry_input = 1 : i64}, %[[ZERO:[^:]+]]: tensor<1x!ciphertext>, %[[PT:[^:]+]]: tensor<1x!plaintext> {cheddar.prepared}) -> tensor<1x!ciphertext>
// CHECK: call @entry__preprocessed(%[[CTX]], %[[MAP]], %[[DBG]], %[[IN]], %[[FLAG]], %[[ZERO]], %[[PT]])

// Preprocessing encodes with a client context the server also owns, and
// reads resources from the caller's directory.
// CHECK: func.func @entry__preprocess(%[[CTX:[^:]+]]: !boot_context {cheddar.support = "boot_context"}, %[[CCTX:[^:]+]]: !client_context {cheddar.support = "client_context"}, %[[DIR:[^:]+]]: !preprocessing.resource_dir {cheddar.support = "resource_dir"}) -> tensor<1x!plaintext>
// CHECK: %[[ENC:.*]] = cheddar.get_encoder %[[CTX]]
// CHECK: call @entry__preprocessing(%[[CCTX]], %[[ENC]], %[[DIR]])
