// RUN: heir-opt --cheddar-bufferize --fold-memref-alias-ops --canonicalize --drop-equivalent-buffer-results "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true" --canonicalize --convert-to-emitc=filter-dialects=cheddar,arith,scf --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!evk_map = !cheddar.evk_map

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
}
