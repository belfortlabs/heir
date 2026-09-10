// RUN: heir-opt %s --cheddar-emitc-entry-interface | heir-translate --mlir-to-cpp --file-id=header | FileCheck %s

// Boundary signatures for an entry that forwards ciphertexts. After precise
// support threading, the evaluation and encode/decode helpers need no Context
// argument. Setup's output still determines the public context type.
!ctx_owner = !emitc.opaque<"std::shared_ptr<Context<word>>&">
!ctx_const = !emitc.opaque<"const std::shared_ptr<Context<word>>&">
!encoder = !emitc.opaque<"const Encoder<word>&">
!ui = !emitc.ptr<!emitc.opaque<"UserInterface<word>">>
!ui_owner = !emitc.opaque<"std::unique_ptr<UserInterface<word>>&">
!ct = !emitc.opaque<"std::array<Ciphertext<word>, 1>&">
!ct_const = !emitc.opaque<"const std::array<Ciphertext<word>, 1>&">
!float = !emitc.ptr<f32>

func.func @entry__setup(%out: !ctx_owner {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["client.setup"]}} { return }
func.func @entry__keygen(%ctx: !ctx_const, %out: !ui_owner {bufferize.result})
    attributes {heir.interface = {func_name = "entry", roles = ["client.keygen"]}} { return }
func.func @entry(%input: !ct_const, %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", input_types = [tensor<4xf32>], result_types = [tensor<4xf32>], roles = ["entry", "server.evaluate"]}} { return }
func.func @encrypt(%encoder: !encoder, %ui: !ui, %input: !float, %out: !ct {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.encrypt"]}} { return }
func.func @decrypt(%encoder: !encoder, %ui: !ui, %input: !ct_const, %out: !float {bufferize.result})
    attributes {heir.interface = {func_name = "entry", index = 0 : i64, roles = ["client.decrypt"]}} { return }

// CHECK: using Context = ::cheddar::Context<word>;
// CHECK: std::shared_ptr<Context> Setup();
// CHECK: EncryptedOutputs Evaluate(
