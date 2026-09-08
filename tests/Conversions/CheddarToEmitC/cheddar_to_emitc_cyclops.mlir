// RUN: heir-opt --cheddar-bufferize --fold-memref-alias-ops --canonicalize --drop-equivalent-buffer-results "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true" --canonicalize --convert-to-emitc=filter-dialects=cheddar,arith,scf --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!encoder = !cheddar.encoder
!evk_map = !cheddar.evk_map
!parameter = !cheddar.parameter
!plaintext = !cheddar.plaintext
!user_interface = !cheddar.user_interface

module attributes {cheddar.runtime = "cyclops"} {
  // CHECK: func.func @hrot
  // CHECK: emitc.verbatim "{}->HRot({}, {}, {}.GetRotationKey(5, {}->BootSecretId(), {}->param_, 4, KeyMode::kInherit), 5);"
  func.func @hrot(%ctx: !context, %evk: !evk_map, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
    %dest = tensor.empty() : tensor<!ciphertext>
    %result = cheddar.hrot %ctx, %evk, %ct, %dest {level = 4 : i64, static_distance = 5 : i64} : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }

  // CHECK: func.func @hconj_add
  // CHECK: emitc.verbatim "{}->HConjAdd({}, {}, {}, {}.GetConjugationKey({}->BootSecretId()));"
  func.func @hconj_add(%ctx: !context, %evk: !evk_map, %lhs: tensor<!ciphertext>, %rhs: tensor<!ciphertext>) -> tensor<!ciphertext> {
    %dest = tensor.empty() : tensor<!ciphertext>
    %result = cheddar.hconj_add %ctx, %evk, %lhs, %rhs, %dest : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }

  // A plaintext starts untagged and `Encrypt` rejects one, so the plaintext is
  // tagged with the secret it is encoded for before the call. The scale-snu
  // form in `cheddar_to_emitc.mlir` emits only the call.
  // CHECK: func.func @enc_chain
  // CHECK: emitc.verbatim "{}.EncodeSlots({}, 5, {}.GetScale(5), {});"
  // CHECK: emitc.verbatim "{}.SetSecretId({}->BootSecretId());"
  // CHECK: emitc.member_call_opaque %arg3 "Encrypt"
  func.func @enc_chain(%ctx: !context, %enc: !encoder, %msg: tensor<4xf64>, %ui: !user_interface) -> tensor<!ciphertext> {
    %dp = tensor.empty() : tensor<!plaintext>
    %pt = cheddar.encode %enc, %msg, %dp {level = 5 : i64, logScale = 37 : i64, useSlotsApi} : (!encoder, tensor<4xf64>, tensor<!plaintext>) -> tensor<!plaintext>
    %dc = tensor.empty() : tensor<!ciphertext>
    %ct = cheddar.encrypt %ctx, %ui, %pt, %dc : (!context, !user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %ct : tensor<!ciphertext>
  }

  // Every evaluation key is indexed by the secret it was generated under, so
  // the rotation-key call names one. The scale-snu form in
  // `cheddar_to_emitc.mlir` has no such argument.
  // CHECK: func.func @configure
  // CHECK: emitc.verbatim "{}->PrepareRotationKey(3, {}->BootSecretId(), 2);"
  func.func @configure() -> (tensor<!context>, tensor<!user_interface>) {
    %p = cheddar.make_parameter {logN = 14 : i64, logScale = 45 : i64, mainPrimes = array<i64: 1, 2, 3>, auxPrimes = array<i64: 4, 5>} : !parameter
    %context_dest = tensor.empty() : tensor<!context>
    %context = cheddar.create_context %p, %context_dest : (!parameter, tensor<!context>) -> tensor<!context>
    %ui_dest = tensor.empty() : tensor<!user_interface>
    %ui = cheddar.create_user_interface %context, %ui_dest : (tensor<!context>, tensor<!user_interface>) -> tensor<!user_interface>
    %prepared = cheddar.prepare_rot_key %context, %ui {distance = 3 : i64, maxLevel = 2 : i64} : (tensor<!context>, tensor<!user_interface>) -> tensor<!user_interface>
    return %context, %prepared : tensor<!context>, tensor<!user_interface>
  }
}
