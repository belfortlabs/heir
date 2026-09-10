// RUN: heir-opt %s --cheddar-emitc-entry-interface | heir-translate --mlir-to-cpp --file-id=source | FileCheck %s

!ctx_owner = !emitc.opaque<"std::shared_ptr<Context<word>>&">
!ctx_const = !emitc.opaque<"const std::shared_ptr<Context<word>>&">
!ui = !emitc.ptr<!emitc.opaque<"UserInterface<word>">>
!ui_owner = !emitc.opaque<"std::unique_ptr<UserInterface<word>>&">
!ct = !emitc.opaque<"std::array<Ciphertext<word>, 1>&">
!ct_const = !emitc.opaque<"const std::array<Ciphertext<word>, 1>&">
!float = !emitc.ptr<f32>

func.func @entry__setup(%out: !ctx_owner {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
func.func @entry__keygen(%ctx: !ctx_const, %out: !ui_owner {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
func.func @entry(%input: !ct_const, %flag: i1,
    %zero: !ct_const {client.enc_zero_arg = {func_name = "entry", index = 0 : i64}},
    %out0: !ct {bufferize.result}, %out1: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", input_types = [tensor<4xf32>, i1], result_types = [tensor<4xf32>, tensor<4xf32>], roles = ["entry", "server.evaluate"]}} { return }
func.func @encrypt(%ui: !ui, %input: !float, %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt"]}} { return }
func.func @encrypt_zero(%ui: !ui, %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt_zero"]}} { return }
func.func @decrypt0(%ui: !ui, %input: !ct_const, %out: !float {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.decrypt"]}} { return }
func.func @decrypt1(%ui: !ui, %input: !ct_const, %out: !float {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 1 : i64, roles = ["client.decrypt"]}} { return }

// CHECK: EncryptedInputs Encrypt
// CHECK: ::heir::generated::detail::encrypt(
// CHECK: std::get<1>
// CHECK: static_cast<bool>
// CHECK: std::get<2>
// CHECK: ::heir::generated::detail::encrypt_zero(
// CHECK: Outputs Decrypt
// CHECK: ::heir::generated::detail::decrypt0(
// CHECK: ::heir::generated::detail::decrypt1(
