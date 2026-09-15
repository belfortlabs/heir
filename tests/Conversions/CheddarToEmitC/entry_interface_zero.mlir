// RUN: heir-opt %s --cheddar-emitc-entry-interface > %t
// RUN: heir-translate %t --mlir-to-cpp --file-id=client_source | FileCheck %s
// RUN: heir-translate %t --mlir-to-cpp --file-id=client_header | FileCheck %s --check-prefix=CLIENT-H
// RUN: heir-translate %t --mlir-to-cpp --file-id=server_source | FileCheck %s --check-prefix=SHARED
// RUN: heir-translate %t --mlir-to-cpp --file-id=server_header | FileCheck %s --check-prefix=SERVER-H
// RUN: not heir-opt %s --cheddar-emitc-entry-interface=runtime=cheddar 2>&1 | FileCheck %s --check-prefix=CONTRADICT

// The module records the runtime it was lowered for, so the pass needs no
// option; an option that disagrees is an error.
// CONTRADICT: contradicts the recorded runtime 'cyclops'
// SERVER-H: EncryptedOutputs Evaluate(Context&, const EvaluationKeys*, const PreparedInputs&, const EncryptedInputs&, const DebugSink*);
// CLIENT-H: EvaluationKeyRequest GetKeyRequest(Context&);

// A helper both sides use is defined, with internal linkage, in both.
// SHARED: static void shared_layout(
// SHARED: shared_layout(
// CHECK: static void shared_layout(
// CHECK: shared_layout(
// CHECK: EvaluationKeyRequest GetKeyRequest(
// CHECK: constexpr std::array<int, 3> linear_transform_indices_0{-2, 0, 5};
// CHECK: AddLinearTransformRequiredKeys
// CHECK-SAME: , 16, linear_transform_indices_0, 7, 2, 3);
// CHECK: AddBootstrapRequiredKeys
// CHECK-SAME: BootParameter({{.*}}.param_.max_level_, 4, 2, 8), 1024,
// CHECK-SAME: BootVariant::kImaginaryRemoving
// CHECK: PrepareRotationKey(GetKeyRequest(

!ctx = !emitc.ptr<!emitc.opaque<"Context<word>">>
!client_ctx = !emitc.ptr<!emitc.opaque<"ClientContext<word>">>
!ctx_owner = !emitc.opaque<"std::shared_ptr<ClientContext<word>>&">
!ctx_const = !emitc.opaque<"const std::shared_ptr<ClientContext<word>>&">
!server_owner = !emitc.opaque<"std::shared_ptr<Context<word>>&">
!ui = !emitc.ptr<!emitc.opaque<"UserInterface<word>">>
!ui_owner = !emitc.opaque<"std::unique_ptr<UserInterface<word>>&">
!evk_map = !emitc.opaque<"const EvkMap<word>&">
!ct = !emitc.array<1x!emitc.opaque<"Ciphertext<word>">>
!ct_const = !emitc.array<1x!emitc.opaque<"const Ciphertext<word>">>
!float = !emitc.ptr<f32>

module attributes {cheddar.runtime = "cyclops"} {
func.func private @shared_layout() { return }
func.func @prepare() attributes {heir.interface = {func_name = "entry", roles = ["server.preprocessing"]}} {
  call @shared_layout() : () -> ()
  return
}
func.func @entry__setup(%out: !ctx_owner {bufferize.result})
    attributes {
      cheddar.bootstrap_log_message_ratio = 8 : i64,
      cheddar.bootstrap_num_cts = 4 : i64,
      cheddar.bootstrap_num_stc = 2 : i64,
      cheddar.bootstrap_slots = 1024 : i64,
      cheddar.linear_transform_keys = [{bs = 2 : i64, gs = 3 : i64,
        indices = array<i32: -2, 0, 5>, level = 7 : i64, width = 16 : i64}],
      heir.interface = {func_name = "entry", roles = ["client.setup"]}
    } { return }
func.func @entry__server_setup(%out: !server_owner {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
func.func @entry__keygen(%ctx: !ctx_const, %out: !ui_owner {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
func.func @entry(%input: !ct_const, %flag: i1,
    %zero: !ct_const {client.enc_zero_arg = {func_name = "entry", index = 0 : i64}},
    %out0: !ct {bufferize.result}, %out1: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", input_types = [tensor<4xf32>, i1], result_types = [tensor<4xf32>, tensor<4xf32>], roles = ["entry", "server.evaluate"]}} { return }
func.func @encrypt(%ui: !ui {cheddar.support = "user_interface"}, %input: !float, %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt"]}} {
  call @shared_layout() : () -> ()
  return
}
func.func @encrypt_zero(%ui: !ui {cheddar.support = "user_interface"}, %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt_zero"]}} { return }
func.func @decrypt0(%ui: !ui {cheddar.support = "user_interface"}, %input: !ct_const, %out: !float {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.decrypt"]}} { return }
func.func @decrypt1(%ui: !ui {cheddar.support = "user_interface"}, %input: !ct_const, %out: !float {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 1 : i64, roles = ["client.decrypt"]}} { return }

func.func @entry__encrypt_inputs(%ctx: !client_ctx {cheddar.support = "client_context"}, %ui: !ui {cheddar.support = "user_interface"},
    %input: !float {cheddar.entry_input = 0 : i64},
    %out: !ct {bufferize.result}, %zero: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["facade.encrypt"]}} {
  emitc.call_opaque "encrypt"() : () -> ()
  emitc.call_opaque "encrypt_zero"() : () -> ()
  return
}
func.func @entry__decrypt_outputs(%ctx: !client_ctx {cheddar.support = "client_context"}, %ui: !ui {cheddar.support = "user_interface"},
    %input0: !ct_const, %input1: !ct_const,
    %out0: !float {bufferize.result}, %out1: !float {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["facade.decrypt"]}} {
  emitc.call_opaque "decrypt0"() : () -> ()
  emitc.call_opaque "decrypt1"() : () -> ()
  return
}
func.func @entry__evaluate(%ctx: !ctx {cheddar.support = "context"}, %evk_map: !evk_map {cheddar.support = "evk_map"},
    %input: !ct_const {cheddar.entry_input = 0 : i64}, %flag: i1 {cheddar.entry_input = 1 : i64}, %zero: !ct_const,
    %out0: !ct {bufferize.result}, %out1: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["facade.evaluate"]}} {
  emitc.call_opaque "entry"() : () -> ()
  return
}
func.func @entry__preprocess(%ctx: !ctx {cheddar.support = "context"})
    attributes {heir.interface = {func_name = "entry", roles = ["facade.preprocess"]}} {
  emitc.call_opaque "prepare"() : () -> ()
  return
}
}

// CHECK: EncryptedInputs Encrypt
// CHECK: std::get<1>
// CHECK: static_cast<bool>
// CHECK: std::get<0>
// CHECK: ::heir::generated::detail::entry__encrypt_inputs(
// CHECK: Outputs Decrypt
// CHECK: ::heir::generated::detail::entry__decrypt_outputs(
