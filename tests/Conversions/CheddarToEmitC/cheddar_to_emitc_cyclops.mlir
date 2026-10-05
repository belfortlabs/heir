// RUN: heir-opt --cheddar-bufferize --fold-memref-alias-ops --canonicalize --drop-equivalent-buffer-results "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true" --canonicalize --convert-to-emitc=filter-dialects=cheddar,arith,scf --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!evk_map = !cheddar.evk_map

// The cheddar.max_pool test at the end marks the module.
// CHECK: module attributes {cheddar.runtime = "cyclops", cheddar.uses_max_pool}
module attributes {cheddar.runtime = "cyclops"} {
  // CHECK: func.func @keygen
  // CHECK: emitc.verbatim "{} = std::make_unique<UserInterface<word>>({}, true);"
  func.func @keygen(%ctx: tensor<!cheddar.client_context>) -> tensor<!cheddar.user_interface> {
    %dest = tensor.empty() : tensor<!cheddar.user_interface>
    %ui = cheddar.create_user_interface %ctx, %dest : (tensor<!cheddar.client_context>, tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface>
    return %ui : tensor<!cheddar.user_interface>
  }

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

  // One Cyclops MaxPool call on the boot context. The asserts fail fast if
  // Cyclops disagrees with HEIR's level model, and the module is marked so
  // that the entry interface includes extension/max/MaxPool.h (checked
  // before the module).
  // CHECK: func.func @max_pool
  // CHECK: emitc.verbatim "ConstContextPtr<word> _mp_cp(ConstContextPtr<word>(), {});"
  // CHECK: emitc.verbatim "std::shared_ptr<BootContext<word>> _mp_bc(std::shared_ptr<BootContext<word>>(), {});"
  // CHECK: emitc.verbatim "_mp_cfg.total_slots = 8192;"
  // CHECK: emitc.verbatim "_mp_cfg.input_length = 4736;"
  // CHECK: emitc.verbatim "_mp_cfg.window_size = 2;"
  // CHECK: emitc.verbatim "_mp_cfg.stride = 2;"
  // CHECK: emitc.verbatim "_mp_cfg.dilation = 1;"
  // CHECK: emitc.verbatim "_mp_cfg.value_bound = 0.5;"
  // CHECK: emitc.verbatim "_mp_cfg.pad_value = -0.5;"
  // CHECK: emitc.verbatim "_mp_cfg.compact_output = true;"
  // CHECK: emitc.verbatim "_mp_cfg.ceil_mode = false;"
  // CHECK: emitc.verbatim "[](auto& _c) { if constexpr (requires { _c.cap_league_level; }) _c.cap_league_level = false; }(_mp_cfg);"
  // CHECK: emitc.verbatim "int _mp_lvl = {}->param_.NPToLevel({}[0].GetNP());"
  // CHECK: emitc.verbatim "AssertTrue(_mp_lvl == 9,
  // CHECK: emitc.verbatim "MaxPool<word> _mp(_mp_cfg, _mp_lvl, {}[0].GetScale());"
  // CHECK: emitc.verbatim "_mp.Compile(_mp_cp, _mp_bc);"
  // CHECK: emitc.verbatim "AssertTrue(_mp.GetOutputLevel() == 5,
  // CHECK: emitc.verbatim "_mp.EvaluateMax(_mp_cp, {}[0], {}[0], {}, _mp_bc);"
  func.func @max_pool(%ctx: !cheddar.boot_context, %evk: !evk_map, %ct: tensor<1x!ciphertext>) -> tensor<1x!ciphertext> {
    %dest = tensor.empty() : tensor<1x!ciphertext>
    %result = cheddar.max_pool %ctx, %ct, %evk, %dest {num_slots = 8192 : i64, input_length = 4736 : i64, window_size = 2 : i64, stride = 2 : i64, dilation = 1 : i64, ceil_mode = false, level = 9 : i64, levelConsumption = 4 : i64, value_bound = 0.5 : f64} : (!cheddar.boot_context, tensor<1x!ciphertext>, !evk_map, tensor<1x!ciphertext>) -> tensor<1x!ciphertext>
    return %result : tensor<1x!ciphertext>
  }
}
