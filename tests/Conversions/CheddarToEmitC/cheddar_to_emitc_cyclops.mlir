// RUN: heir-opt --cheddar-bufferize --fold-memref-alias-ops --canonicalize --drop-equivalent-buffer-results "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true" --canonicalize --convert-to-emitc=filter-dialects=cheddar,arith,scf --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!evk_map = !cheddar.evk_map

module attributes {cheddar.runtime = "cyclops"} {
  // CHECK: func.func @keygen
  // CHECK: emitc.verbatim "{} = std::make_unique<UserInterface<word>>({}, true);"
  func.func @keygen(%ctx: tensor<!cheddar.client_context>) -> tensor<!cheddar.user_interface> {
    %dest = tensor.empty() : tensor<!cheddar.user_interface>
    %ui = cheddar.create_user_interface %ctx, %dest : (tensor<!cheddar.client_context>, tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface>
    return %ui : tensor<!cheddar.user_interface>
  }

  // CHECK: func.func @hrot
  // CHECK: emitc.verbatim "{}->HRot({}, {}, {}.GetRotationKey(5, {}->NativeSecretId(), {}->param_, 4, KeyMode::kInherit), 5);"
  func.func @hrot(%ctx: !context, %evk: !evk_map, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
    %dest = tensor.empty() : tensor<!ciphertext>
    %result = cheddar.hrot %ctx, %evk, %ct, %dest {level = 4 : i64, static_distance = 5 : i64} : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }

  // A relinearization keyed by the map looks its key up for the op's level:
  // the default key when the ring holds one, the level-specific key otherwise.
  // CHECK: func.func @relinearize
  // CHECK: emitc.verbatim "{}->Relinearize({}, {}, {}.GetMultiplicationKey({}->NativeSecretId(), {}->param_, 3, KeyMode::kDefault));"
  // CHECK: emitc.verbatim "{}->RelinearizeRescale({}, {}, {}.GetMultiplicationKey({}->NativeSecretId(), {}->param_, 3, KeyMode::kDefault));"
  func.func @relinearize(%ctx: !context, %evk: !evk_map, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
    %dest = tensor.empty() : tensor<!ciphertext>
    %relin = cheddar.relinearize %ctx, %ct, %evk, %dest {level = 3 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
    %dest2 = tensor.empty() : tensor<!ciphertext>
    %result = cheddar.relinearize_rescale %ctx, %relin, %evk, %dest2 {level = 3 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }

  // CHECK: func.func @hmult
  // CHECK: emitc.verbatim "{}->HMult({}, {}, {}, {}.GetMultiplicationKey({}->NativeSecretId(), {}->param_, 2, KeyMode::kDefault), true);"
  func.func @hmult(%ctx: !context, %evk: !evk_map, %a: tensor<!ciphertext>, %b: tensor<!ciphertext>) -> tensor<!ciphertext> {
    %dest = tensor.empty() : tensor<!ciphertext>
    %result = cheddar.hmult %ctx, %a, %b, %evk, %dest {level = 2 : i64, rescale = true} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }

  // CHECK: func.func @hconj_add
  // CHECK: emitc.verbatim "{}->HConjAdd({}, {}, {}, {}.GetConjugationKey({}->NativeSecretId()));"
  func.func @hconj_add(%ctx: !context, %evk: !evk_map, %lhs: tensor<!ciphertext>, %rhs: tensor<!ciphertext>) -> tensor<!ciphertext> {
    %dest = tensor.empty() : tensor<!ciphertext>
    %result = cheddar.hconj_add %ctx, %evk, %lhs, %rhs, %dest : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
    return %result : tensor<!ciphertext>
  }
}
