// RUN: heir-opt --cheddar-emitc-entry-interface %s | FileCheck %s --check-prefix=RECORDED
// RUN: heir-opt --cheddar-emitc-entry-interface=runtime=cyclops %s | FileCheck %s --check-prefix=RECORDED
// RUN: not heir-opt --cheddar-emitc-entry-interface=runtime=cheddar %s 2>&1 | FileCheck %s --check-prefix=MISMATCH

// RECORDED: verbatim "using namespace ::cyclops;"
// RECORDED: verbatim "using PublicKey = const ::cyclops::EvkMap<word>*;"

// MISMATCH: error: runtime option 'cheddar' contradicts the 'cyclops' runtime the module was lowered for

!ctx = !emitc.ptr<!emitc.opaque<"Context<word>">>
!boot_ctx = !emitc.ptr<!emitc.opaque<"BootContext<word>">>
!ctx_owner = !emitc.opaque<"std::shared_ptr<BootContext<word>>&">
!ctx_owner_const = !emitc.opaque<"const std::shared_ptr<BootContext<word>>&">
!encoder = !emitc.opaque<"const Encoder<word>&">
!ui = !emitc.ptr<!emitc.opaque<"UserInterface<word>">>
!ui_owner = !emitc.opaque<"std::unique_ptr<UserInterface<word>>&">
!evk = !emitc.opaque<"const EvaluationKey<word>&">
!evk_map = !emitc.opaque<"const EvkMap<word>&">
!ct = !emitc.opaque<"std::array<Ciphertext<word>, 1>&">
!ct_const = !emitc.opaque<"const std::array<Ciphertext<word>, 1>&">
!pt = !emitc.opaque<"std::array<Plaintext<word>, 2>&">
!pt_const = !emitc.opaque<"const std::array<Plaintext<word>, 2>&">


module attributes {cheddar.runtime = "cyclops"} {
  func.func @entry__setup(
      %out: !ctx_owner {bufferize.result})
      attributes {client.setup_func = {func_name = "entry"}} {
    return
  }

  func.func @entry__keygen(
      %ctx: !ctx_owner_const,
      %out: !ui_owner {bufferize.result})
      attributes {client.keygen_func = {func_name = "entry"}} {
    return
  }

  func.func @entry(
      %input0: !ct_const, %input1: !ct_const, %prepared: !pt_const,
      %out: !ct {bufferize.result})
      attributes {
        heir.entry_func = {func_name = "entry"},
        heir.entry_input_types = [tensor<4xf32>, tensor<2xf32>],
        heir.entry_result_types = [tensor<2xf32>]
      } {
    return
  }

  func.func @entry__encrypt__arg0(
      %ctx: !ctx, %encoder: !encoder, %ui: !ui, %input: !emitc.ptr<f32>,
      %out: !ct {bufferize.result})
      attributes {
        client.enc_func = {func_name = "entry", index = 0 : i64}
      } {
    return
  }

  func.func @entry__encrypt__arg1(
      %ctx: !ctx, %encoder: !encoder, %ui: !ui, %input: !emitc.ptr<f32>,
      %out: !ct {bufferize.result})
      attributes {
        client.enc_func = {func_name = "entry", index = 1 : i64}
      } {
    return
  }

  func.func @entry__preprocessing(
      %ctx: !boot_ctx, %encoder: !encoder, %out: !pt {bufferize.result})
      attributes {server.preprocessing_func = {func_name = "entry"}} {
    %data = emitc.literal "nullptr" : !emitc.ptr<f32>
    call @outlined_layout(%data) : (!emitc.ptr<f32>) -> ()
    return
  }

  func.func @entry__preprocessed(
      %ctx: !boot_ctx, %encoder: !encoder, %evk_map: !evk_map,
      %input0: !ct_const, %input1: !ct_const, %prepared: !pt_const,
      %out: !ct {bufferize.result})
      attributes {server.evaluate_func = {func_name = "entry"}} {
    return
  }

  func.func @entry__decrypt__result0(
      %ctx: !ctx, %encoder: !encoder, %ui: !ui, %evk: !evk,
      %input: !ct_const, %out: !emitc.ptr<f32> {bufferize.result})
      attributes {
        client.dec_func = {func_name = "entry", index = 0 : i64}
      } {
    return
  }

  func.func private @outlined_layout(%input: !emitc.ptr<f32>)
      attributes {client.pack_func = {func_name = "entry"}} {
    emitc.call_opaque "heir::loadResource"(%input) <{
      args = [#emitc.opaque<"\22data/weights.bin\22">, 0 : index,
              #emitc.opaque<"4">],
      template_args = [f32]
    }> : (!emitc.ptr<f32>) -> ()
    return
  }

  func.func private @call_preprocessing(
      %ctx: !boot_ctx, %encoder: !encoder, %out: !pt) {
    emitc.call_opaque "entry__preprocessing"(%ctx, %encoder, %out)
        : (!boot_ctx, !encoder, !pt) -> ()
    return
  }
}
