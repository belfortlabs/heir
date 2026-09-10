// RUN: heir-opt %s --cheddar-emitc-entry-interface=runtime=cyclops > %t
// RUN: heir-translate %t --mlir-to-cpp --file-id=client_source | FileCheck %s
// RUN: heir-translate %t --mlir-to-cpp --file-id=server_source | FileCheck %s --check-prefix=SHARED
// RUN: heir-translate %t --mlir-to-cpp --file-id=client_header | FileCheck %s --check-prefix=CLIENT-H
// RUN: heir-translate %t --mlir-to-cpp --file-id=server_header | FileCheck %s --check-prefix=SERVER-H

// Setup outputs determine context types even when evaluation needs no context.
// CLIENT-H: using Context = ::cyclops::ClientContext<word>;
// CLIENT-H: std::shared_ptr<Context> Setup();
// SERVER-H: using Context = ::cyclops::Context<word>;
// SERVER-H: std::shared_ptr<Context> Setup();
// SERVER-H: EncryptedOutputs Evaluate(

// SHARED: void shared_layout(
// SHARED: shared_layout(
// CHECK: void shared_layout(
// CHECK: shared_layout(

!ctx_owner = !emitc.opaque<"std::shared_ptr<ClientContext<word>>&">
!ctx_const = !emitc.opaque<"const std::shared_ptr<ClientContext<word>>&">
!server_owner = !emitc.opaque<"std::shared_ptr<Context<word>>&">
!ui = !emitc.ptr<!emitc.opaque<"UserInterface<word>">>
!ui_owner = !emitc.opaque<"std::unique_ptr<UserInterface<word>>&">
!ct = !emitc.opaque<"std::array<Ciphertext<word>, 1>&">
!ct_const = !emitc.opaque<"const std::array<Ciphertext<word>, 1>&">
!float = !emitc.ptr<f32>

func.func private @shared_layout() { return }
func.func @prepare() attributes {heir.interface = {func_name = "entry", roles = ["server.preprocessing"]}} {
  call @shared_layout() : () -> ()
  return
}
func.func @entry__setup(%out: !ctx_owner {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
func.func @entry__server_setup(%out: !server_owner {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["server.setup"]}} { return }
func.func @entry__keygen(%ctx: !ctx_const, %out: !ui_owner {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
func.func @entry(%input: !ct_const, %flag: i1,
    %zero: !ct_const {client.enc_zero_arg = {func_name = "entry", index = 0 : i64}},
    %out0: !ct {bufferize.result}, %out1: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", input_types = [tensor<4xf32>, i1], result_types = [tensor<4xf32>, tensor<4xf32>], roles = ["entry", "server.evaluate"]}} { return }
func.func @encrypt(%ui: !ui, %input: !float, %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt"]}} {
  call @shared_layout() : () -> ()
  return
}
func.func @encrypt_zero(%ui: !ui, %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt_zero"]}} { return }
func.func @decrypt0(%ui: !ui, %input: !ct_const, %out: !float {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.decrypt"]}} { return }
func.func @decrypt1(%ui: !ui, %input: !ct_const, %out: !float {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 1 : i64, roles = ["client.decrypt"]}} { return }

// CHECK: EncryptedInputs Encrypt
// CHECK: ::heir::generated::detail_entry_client::encrypt(
// CHECK: std::get<1>
// CHECK: static_cast<bool>
// CHECK: std::get<2>
// CHECK: ::heir::generated::detail_entry_client::encrypt_zero(
// CHECK: Outputs Decrypt
// CHECK: ::heir::generated::detail_entry_client::decrypt0(
// CHECK: ::heir::generated::detail_entry_client::decrypt1(
