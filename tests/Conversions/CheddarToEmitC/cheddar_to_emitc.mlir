// RUN: heir-opt --cheddar-bufferize --fold-memref-alias-ops --cse --canonicalize --drop-equivalent-buffer-results "--buffer-results-to-out-params=hoist-static-allocs=true modify-public-functions=true add-result-attr=true" --canonicalize --convert-to-emitc=filter-dialects=cheddar,arith,scf --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

// End-to-end op coverage for the cheddar -> EmitC lowering, now driven by stock
// `--convert-to-emitc` (cheddar's dialect interface) + the
// `--cheddar-emitc-boundary` pass, over destination-passing-style cheddar ops.
// Every payload-producing op carries a `tensor.empty` `$output`
// destination; bufferization + `--buffer-results-to-out-params` turn func
// results into trailing out-params, and the boundary pass re-types move-only
// payload args as C++ references (`const T&` inputs, `T&` out-params).

!ciphertext = !cheddar.ciphertext
!plaintext = !cheddar.plaintext
!constant = !cheddar.constant
!context = !cheddar.context
!encoder = !cheddar.encoder
!eval_key = !cheddar.eval_key
!evk_map = !cheddar.evk_map
!user_interface = !cheddar.user_interface
!parameter = !cheddar.parameter
!boot_context = !cheddar.boot_context
!linear_transform = !cheddar.linear_transform

// CHECK: emitc.global static @resource : !emitc.array<4xf32> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]>
memref.global "private" constant @resource : memref<4xf32> = dense_resource<weights>

// C++ zero-initializes private globals with no initializer. Dropping a
// positive-zero splat keeps large generated arrays compact.
// CHECK: emitc.global static @zero_splat : !emitc.array<4x4xf32>{{$}}
memref.global "private" constant @zero_splat : memref<4x4xf32> = dense<0.000000e+00>

// Negative floating-point zero must remain explicit.
// CHECK: emitc.global static @negative_zero_splat : !emitc.array<4xf32> = dense<-0.000000e+00>
memref.global "private" constant @negative_zero_splat : memref<4xf32> = dense<-0.000000e+00>

// CHECK: func.func @configure
// CHECK-SAME: !emitc.opaque<"std::shared_ptr<Context<word>>&">
// CHECK-SAME: !emitc.opaque<"std::unique_ptr<UserInterface<word>>&">
// CHECK: emitc.verbatim "static Parameter<word> cheddar_param
// CHECK-SAME: std::vector<word>{1ULL, 2ULL, 3ULL}
// CHECK-SAME: std::vector<word>{4ULL, 5ULL}
// CHECK: emitc.verbatim "{} = Context<word>::Create({});"
// CHECK: emitc.verbatim "{} = std::make_unique<UserInterface<word>>({});"
// CHECK: emitc.verbatim "{}->PrepareRotationKey(3, 2);"
func.func @configure() -> (tensor<!context>, tensor<!user_interface>) {
  %p = cheddar.make_parameter {logN = 14 : i64, logScale = 45 : i64, mainPrimes = array<i64: 1, 2, 3>, auxPrimes = array<i64: 4, 5>} : !parameter
  %context_dest = tensor.empty() : tensor<!context>
  %context = cheddar.create_context %p, %context_dest : (!parameter, tensor<!context>) -> tensor<!context>
  %ui_dest = tensor.empty() : tensor<!user_interface>
  %ui = cheddar.create_user_interface %context, %ui_dest : (tensor<!context>, tensor<!user_interface>) -> tensor<!user_interface>
  %prepared = cheddar.prepare_rot_key %ui {distance = 3 : i64, maxLevel = 2 : i64} : (tensor<!user_interface>) -> tensor<!user_interface>
  return %context, %prepared : tensor<!context>, tensor<!user_interface>
}

// A transform whose split the runtime plans contributes no distances of its
// own; `prepare_linear_transform_keys` asks the runtime instead, through a
// shape-only transform over a zero matrix with the same diagonals and width.
// CHECK: func.func @configure_cyclops
// CHECK: emitc.verbatim "{}->PrepareRotationKey(3, {}->BootSecretId(), 2);"
// CHECK: emitc.verbatim "{"
// CHECK: emitc.verbatim "ConstContextPtr<word> _ltk_cp(ConstContextPtr<word>(), {});"
// CHECK: emitc.verbatim "StripedMatrix _ltk_matrix(8, 8);"
// CHECK: emitc.verbatim "_ltk_matrix[0] = std::vector<Complex>(8, Complex(0.0, 0.0));"
// CHECK: emitc.verbatim "_ltk_matrix[3] = std::vector<Complex>(8, Complex(0.0, 0.0));"
// CHECK: emitc.verbatim "LinearTransform<word> _ltk(_ltk_cp, _ltk_matrix, 1, {}->param_.GetScale(1), 0, 0, -1, PlaintextCacheConfig(), KeyMode::kInherit, PlaintextMode::kShapeOnly);"
// CHECK: emitc.verbatim "EvkRequest _ltk_req;"
// CHECK: emitc.verbatim "_ltk.AddRequiredRotations(_ltk_req);"
// CHECK: emitc.verbatim "{}->PrepareRotationKey(_ltk_req, {}->BootSecretId());"
// CHECK: emitc.verbatim "}"
func.func @configure_cyclops() -> (tensor<!context>, tensor<!user_interface>) {
  %p = cheddar.make_parameter {logN = 14 : i64, logScale = 45 : i64, mainPrimes = array<i64: 1, 2, 3>, auxPrimes = array<i64: 4, 5>} : !parameter
  %context_dest = tensor.empty() : tensor<!context>
  %context = cheddar.create_context %p, %context_dest : (!parameter, tensor<!context>) -> tensor<!context>
  %ui_dest = tensor.empty() : tensor<!user_interface>
  %ui = cheddar.create_user_interface %context, %ui_dest : (tensor<!context>, tensor<!user_interface>) -> tensor<!user_interface>
  %prepared = cheddar.prepare_rot_key %context, %ui {distance = 3 : i64, maxLevel = 2 : i64} : (tensor<!context>, tensor<!user_interface>) -> tensor<!user_interface>
  %keys = cheddar.prepare_linear_transform_keys %context, %prepared
      {diagonal_indices = array<i32: 0, 3>, width = 8 : i64, level = 1 : i64}
      : (tensor<!context>, tensor<!user_interface>) -> tensor<!user_interface>
  return %context, %keys : tensor<!context>, tensor<!user_interface>
}

// A two-op chain: each op is `ctx->Method(out, a, b)`. The first op's result is
// an intermediate local (`emitc.variable`); the second writes the function
// out-param. Inputs are `const Ciphertext<word>&`, the out-param is mutable.
// CHECK: func.func @arith(
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"Context<word>">>
// CHECK-SAME: !emitc.opaque<"const Ciphertext<word>&">
// CHECK-SAME: !emitc.opaque<"Ciphertext<word>&">
// CHECK: emitc.member_call_opaque %arg0 "Add"(%[[V:.*]], %arg1, %arg2)
// CHECK: emitc.member_call_opaque %arg0 "Mult"(%arg3, %[[V]], %arg2)
func.func @arith(%ctx: !context, %a: tensor<!ciphertext>, %b: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.add %ctx, %a, %b, %d0 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %s = cheddar.mult %ctx, %r, %b, %d1 : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %s : tensor<!ciphertext>
}

// A semantic ciphertext copy lowers to CHEDDAR's deep-copy API, never C++
// copy-assignment on the move-only payload.
// CHECK: func.func @copy
// CHECK: emitc.member_call_opaque %arg0 "Copy"(%arg2, %arg1)
func.func @copy(%ctx: !context, %input: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %dest = tensor.empty() : tensor<!ciphertext>
  %result = cheddar.copy %ctx, %input, %dest : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %result : tensor<!ciphertext>
}

// A memref.copy that survives alias folding has true copy semantics and lowers
// through the same CHEDDAR deep-copy API.
// CHECK: func.func @memref_copy
// CHECK: emitc.member_call_opaque %arg0 "Copy"(%arg2, %arg1)
func.func @memref_copy(%ctx: !context, %input: memref<!ciphertext>,
                       %output: memref<!ciphertext>) {
  memref.copy %input, %output
      : memref<!ciphertext> to memref<!ciphertext>
  return
}

// CHEDDAR's Context overloads Add/Sub/Mult on the second operand type, so the
// `*_plain` / `*_const` ops dispatch to the base method name.
// CHECK: func.func @plain_const
// CHECK: emitc.member_call_opaque %arg0 "Add"
// CHECK: emitc.member_call_opaque %arg0 "Mult"
func.func @plain_const(%ctx: !context, %ct: tensor<!ciphertext>, %pt: tensor<!plaintext>, %c: tensor<!constant>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r1 = cheddar.add_plain %ctx, %ct, %pt, %d0 : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %r2 = cheddar.mult_const %ctx, %r1, %c, %d1 : (!context, tensor<!ciphertext>, tensor<!constant>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r2 : tensor<!ciphertext>
}

// The level_down target level is appended as a trailing opaque constant arg.
// CHECK: func.func @unary
// CHECK: emitc.member_call_opaque %arg0 "Neg"
// CHECK: emitc.member_call_opaque %arg0 "LevelDown"
// CHECK-SAME: #emitc.opaque<"2">
func.func @unary(%ctx: !context, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %n = cheddar.neg %ctx, %ct, %d0 : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %l = cheddar.level_down %ctx, %n, %d1 {targetLevel = 2 : i64} : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %l : tensor<!ciphertext>
}

// CHECK: func.func @relin
// CHECK: emitc.member_call_opaque %arg0 "Relinearize"
// CHECK: emitc.member_call_opaque %arg0 "Rescale"
func.func @relin(%ctx: !context, %ct: tensor<!ciphertext>, %k: !eval_key) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r1 = cheddar.relinearize %ctx, %ct, %k, %d0 : (!context, tensor<!ciphertext>, !eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %r2 = cheddar.rescale %ctx, %r1, %d1 : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r2 : tensor<!ciphertext>
}

// The HMult `rescale` flag is appended as the trailing opaque constant arg.
// CHECK: func.func @hmult
// CHECK: emitc.member_call_opaque %arg0 "HMult"
// CHECK-SAME: #emitc.opaque<"true">
func.func @hmult(%ctx: !context, %a: tensor<!ciphertext>, %b: tensor<!ciphertext>, %k: !eval_key) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.hmult %ctx, %a, %b, %k, %d0 {rescale = true} : (!context, tensor<!ciphertext>, tensor<!ciphertext>, !eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// Encode bridges a float message buffer through a std::vector<Complex> and
// uses CHEDDAR's canonical per-level scale; encrypt is an out-param method
// call.
// CHECK: func.func @enc_chain
// CHECK-SAME: !emitc.ptr<f64>
// CHECK: emitc.verbatim "{}.Encode({}, 5, {}.GetScale(5), {});"
// CHECK: emitc.member_call_opaque %arg2 "Encrypt"
func.func @enc_chain(%enc: !encoder, %msg: tensor<4xf64>, %ui: !user_interface) -> tensor<!ciphertext> {
  %dp = tensor.empty() : tensor<!plaintext>
  %pt = cheddar.encode %enc, %msg, %dp {level = 5 : i64, logScale = 37 : i64} : (!encoder, tensor<4xf64>, tensor<!plaintext>) -> tensor<!plaintext>
  %dc = tensor.empty() : tensor<!ciphertext>
  %ct = cheddar.encrypt %ui, %pt, %dc : (!user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %ct : tensor<!ciphertext>
}

// Decrypt is an out-param method call; decode reads into a temporary
// std::vector<Complex> then copies the real parts into the float buffer.
// CHECK: func.func @dec_chain
// CHECK-SAME: !emitc.ptr<f32>
// CHECK: emitc.member_call_opaque %arg1 "Decrypt"
// CHECK: emitc.verbatim "{}.Decode({}, {});"
// CHECK: emitc.verbatim "for (size_t _i = 0; _i < 4; ++_i) {}[_i] = {}.at(_i).real();"
func.func @dec_chain(%enc: !encoder, %ui: !user_interface, %ct: tensor<!ciphertext>, %dst: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %dp = tensor.empty() : tensor<!plaintext>
  %pt = cheddar.decrypt %ui, %ct, %dp : (!user_interface, tensor<!ciphertext>, tensor<!plaintext>) -> tensor<!plaintext>
  %msg = cheddar.decode %enc, %pt, %dst : (!encoder, tensor<!plaintext>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %msg : tensor<1x4xf32>
}

// With `useSlotsApi` (the Cyclops runtime), encode bridges the message
// through a std::vector<double> and calls EncodeSlots.
// CHECK: func.func @enc_chain_slots
// CHECK: emitc.verbatim "{} = std::vector<double>({}, {} + 4);"
// CHECK: emitc.verbatim "{}.EncodeSlots({}, 5, {}.GetScale(5), {});"
func.func @enc_chain_slots(%enc: !encoder, %msg: tensor<4xf64>, %ui: !user_interface) -> tensor<!ciphertext> {
  %dp = tensor.empty() : tensor<!plaintext>
  %pt = cheddar.encode %enc, %msg, %dp {level = 5 : i64, logScale = 37 : i64, useSlotsApi} : (!encoder, tensor<4xf64>, tensor<!plaintext>) -> tensor<!plaintext>
  %dc = tensor.empty() : tensor<!ciphertext>
  %ct = cheddar.encrypt %ui, %pt, %dc : (!user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %ct : tensor<!ciphertext>
}

// Cyclops containers start untagged and Encrypt rejects an untagged plaintext,
// so a `$ctx` operand makes encode tag it first, with the same secret
// UserInterface::EncryptMessage uses.
// CHECK: func.func @enc_chain_tagged
// CHECK: emitc.verbatim "{}.SetSecretId({}->BootSecretId());"
// CHECK: emitc.verbatim "{}.EncodeSlots({}, 5, {}.GetScale(5), {});"
// CHECK: emitc.member_call_opaque %arg3 "Encrypt"
func.func @enc_chain_tagged(%ctx: !context, %enc: !encoder, %msg: tensor<4xf64>, %ui: !user_interface) -> tensor<!ciphertext> {
  %dp = tensor.empty() : tensor<!plaintext>
  %pt = cheddar.encode %ctx, %enc, %msg, %dp {level = 5 : i64, logScale = 37 : i64, useSlotsApi} : (!context, !encoder, tensor<4xf64>, tensor<!plaintext>) -> tensor<!plaintext>
  %dc = tensor.empty() : tensor<!ciphertext>
  %ct = cheddar.encrypt %ui, %pt, %dc : (!user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %ct : tensor<!ciphertext>
}

// With `useSlotsApi`, decode reads real doubles straight out of DecodeSlots;
// no `.real()` projection.
// CHECK: func.func @dec_chain_slots
// CHECK: emitc.verbatim "{}.DecodeSlots({}, {});"
// CHECK: emitc.verbatim "for (size_t _i = 0; _i < 4; ++_i) {}[_i] = {}.at(_i);"
func.func @dec_chain_slots(%enc: !encoder, %ui: !user_interface, %ct: tensor<!ciphertext>, %dst: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %dp = tensor.empty() : tensor<!plaintext>
  %pt = cheddar.decrypt %ui, %ct, %dp : (!user_interface, tensor<!ciphertext>, tensor<!plaintext>) -> tensor<!plaintext>
  %msg = cheddar.decode %enc, %pt, %dst {useSlotsApi} : (!encoder, tensor<!plaintext>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %msg : tensor<1x4xf32>
}

// HRot/HConj keep their verbatim form, looking the key up on the EvkMap operand
// with the secret handle and parameters taken off the context -- so an
// evaluating process needs no UserInterface. Rotations pass the op's level so
// the lookup best-fits the key prepared for it. Static distance bakes the
// distance into the format string; dynamic distance threads the SSA value twice.
// CHECK: func.func @hrot_static
// CHECK: emitc.verbatim "{}->HRot({}, {}, {}.GetRotationKey(5, {}->BootSecretId(), {}->param_, 4, KeyMode::kInherit), 5);"
func.func @hrot_static(%ctx: !context, %evk: !evk_map, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.hrot %ctx, %evk, %ct, %d0 {level = 4 : i64, static_distance = 5 : i64} : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// CHECK: func.func @hrot_dyn
// CHECK: emitc.verbatim "{}->HRot({}, {}, {}.GetRotationKey({}, {}->BootSecretId(), {}->param_, 4, KeyMode::kInherit), {});"
// CHECK-SAME: %arg3, %arg0, %arg0, %arg3
func.func @hrot_dyn(%ctx: !context, %evk: !evk_map, %ct: tensor<!ciphertext>, %d: index) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.hrot %ctx, %evk, %ct, %d0, %d {level = 4 : i64} : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>, index) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// CHECK: func.func @hconj_add
// CHECK: emitc.verbatim "{}->HConjAdd({}, {}, {}, {}.GetConjugationKey({}->BootSecretId()));"
func.func @hconj_add(%ctx: !context, %evk: !evk_map, %a: tensor<!ciphertext>, %b: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.hconj_add %ctx, %evk, %a, %b, %d0 : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// Boot is a BootContext method; cheddar.boot requires a !cheddar.boot_context
// (lowered to BootContext<word>*) so no downcast is needed.
// CHECK: func.func @boot
// CHECK: emitc.member_call_opaque %arg0 "Boot"
func.func @boot(%ctx: !boot_context, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.boot %ctx, %ct, %evk, %d0 : (!boot_context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// eval_poly lowers to the real EvalPoly<word> class -- there is no
// `RunEvalPoly` in cheddar. It mirrors cheddar's own EvalMod: the level/scale
// are taken from the actual input ciphertext (NPToLevel(in.GetNP()),
// in.GetScale()) and target_scale is the square/divide recurrence over
// GetRescalePrimeProd -- seeding the constructor with anything else silently
// blows the Chebyshev basis recurrence up. The construct/Compile/Evaluate
// (which need the move-only ciphertext as a method receiver + ctx->param_ + the
// recurrence) are emitted as verbatim real-cheddar statements in a `{ }` block
// scope so the EvalPoly (and its GPU power basis) is destroyed right after use.
// CHECK: func.func @eval_poly
// CHECK-SAME: !emitc.opaque<"const Ciphertext<word>&">
// CHECK: emitc.verbatim "{"
// CHECK: emitc.verbatim "ConstContextPtr<word> _ep_cp(ConstContextPtr<word>(), {}
// CHECK: emitc.verbatim "int _ep_lvl = {}->param_.NPToLevel({}.GetNP());"
// CHECK: emitc.verbatim "double _ep_is = {}.GetScale();"
// CHECK: emitc.verbatim "_ep_ts = _ep_ts * _ep_ts / {}->param_.GetRescalePrimeProd
// CHECK: emitc.verbatim "EvalPoly<word> _ep({1, 2, 3}, _ep_lvl, _ep_is, _ep_ts, true);"
// CHECK: emitc.verbatim "_ep.Compile(_ep_cp);"
// CHECK: emitc.verbatim "_ep.Evaluate(_ep_cp, {}, {}, {}.GetMultiplicationKey());"
// CHECK: emitc.verbatim "}"
func.func @eval_poly(%ctx: !context, %enc: !encoder, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.eval_poly %ctx, %ct, %evk, %d0 {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64], levelConsumption = 2 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// CHECK: func.func @eval_poly_level_key
// CHECK: emitc.verbatim "_ep.Evaluate(_ep_cp, {}, {}, MultKeySelector<word>({}));"
func.func @eval_poly_level_key(%ctx: !context, %ct: tensor<!ciphertext>, %evk: !evk_map) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.eval_poly %ctx, %ct, %evk, %d0 {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64], levelConsumption = 2 : i64, selectMultKeyAtUseLevel} : (!context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// scale-snu evaluates a single ciphertext, while HEIR represents ciphertext
// payloads as one-element tensors. Both direct and prepared transforms must
// therefore pass the array element, not the std::array itself.
// CHECK: func.func @linear_transform
// CHECK: emitc.verbatim "_lt_matrix[0] = std::vector<Complex>({} + 4, {} + 8);"
// CHECK: emitc.verbatim "_lt_matrix[1] = std::vector<Complex>({} + 12, {} + 16);"
// CHECK: emitc.verbatim "LinearTransform<word> _lt(_lt_cp, _lt_matrix, 1, {}->param_.GetScale(1), 2, 1);"
// CHECK: emitc.verbatim "_lt.Evaluate(_lt_cp, {}[0], {}[0], {});"
func.func @linear_transform(
    %ctx: !context, %ct: tensor<1x!ciphertext>, %evk: !evk_map,
    %diagonals: tensor<4x4xf64>) -> tensor<1x!ciphertext> {
  %out = tensor.empty() : tensor<1x!ciphertext>
  %result = cheddar.linear_transform %ctx, %ct, %evk, %diagonals, %out
      {diagonal_indices = array<i32: 0, 1>, source_row_indices = array<i32: 1, 3>, level = 1 : i64,
       bs = 2 : i64, gs = 1 : i64}
      : (!context, tensor<1x!ciphertext>, !evk_map, tensor<4x4xf64>,
         tensor<1x!ciphertext>) -> tensor<1x!ciphertext>
  return %result : tensor<1x!ciphertext>
}

// CHECK: func.func @linear_transform_min_ks
// CHECK: emitc.verbatim "_lt.Evaluate(_lt_cp, {}[0], {}[0], {}, true);"
func.func @linear_transform_min_ks(
    %ctx: !context, %ct: tensor<1x!ciphertext>, %evk: !evk_map,
    %diagonals: tensor<4x4xf64>) -> tensor<1x!ciphertext> {
  %out = tensor.empty() : tensor<1x!ciphertext>
  %result = cheddar.linear_transform %ctx, %ct, %evk, %diagonals, %out
      {diagonal_indices = array<i32: 0, 1, 2, 3>, source_row_indices = array<i32: 0, 1, 2, 3>,
       level = 1 : i64, bs = 2 : i64, gs = 2 : i64, min_ks = true}
      : (!context, tensor<1x!ciphertext>, !evk_map, tensor<4x4xf64>,
         tensor<1x!ciphertext>) -> tensor<1x!ciphertext>
  return %result : tensor<1x!ciphertext>
}

// A width-4 message repeats every 2 * 4 words per prime, so lwe-to-cheddar
// records log_pt_size_per_prime = 3 and Cyclops keeps one period instead of a
// full ring-degree plaintext. scale-snu CHEDDAR takes pre_rotation in that
// position, so the arguments appear only when the attribute is present.
// CHECK: func.func @linear_transform_compact_pt
// CHECK: emitc.verbatim "LinearTransform<word> _lt(_lt_cp, _lt_matrix, 1, {}->param_.GetScale(1), 2, 1, 3, PlaintextCacheConfig(), KeyMode::kInherit, PlaintextMode::kCompiled, 4);"
func.func @linear_transform_compact_pt(
    %ctx: !context, %ct: tensor<1x!ciphertext>, %evk: !evk_map,
    %diagonals: tensor<4x4xf64>) -> tensor<1x!ciphertext> {
  %out = tensor.empty() : tensor<1x!ciphertext>
  %result = cheddar.linear_transform %ctx, %ct, %evk, %diagonals, %out
      {diagonal_indices = array<i32: 0, 1>, source_row_indices = array<i32: 1, 3>, level = 1 : i64,
       bs = 2 : i64, gs = 1 : i64, log_pt_size_per_prime = 3 : i64}
      : (!context, tensor<1x!ciphertext>, !evk_map, tensor<4x4xf64>,
         tensor<1x!ciphertext>) -> tensor<1x!ciphertext>
  return %result : tensor<1x!ciphertext>
}

// CHECK: func.func @prepare_linear_transform
// CHECK: emitc.verbatim "{} = std::make_shared<LinearTransform<word>>(_lt_cp, _lt_matrix, 1, {}->param_.GetScale(1), 2, 1);"
func.func @prepare_linear_transform(
    %ctx: !context, %diagonals: tensor<2x4xf64>) -> tensor<!linear_transform> {
  %out = tensor.empty() : tensor<!linear_transform>
  %result = cheddar.prepare_linear_transform %ctx, %diagonals, %out
      {diagonal_indices = array<i32: 0, 1>, width = 4 : i64, level = 1 : i64,
       bs = 2 : i64, gs = 1 : i64}
      : (!context, tensor<2x4xf64>, tensor<!linear_transform>)
      -> tensor<!linear_transform>
  return %result : tensor<!linear_transform>
}

// CHECK: func.func @prepare_linear_transform_compact_pt
// CHECK: emitc.verbatim "{} = std::make_shared<LinearTransform<word>>(_lt_cp, _lt_matrix, 1, {}->param_.GetScale(1), 2, 1, 3, PlaintextCacheConfig(), KeyMode::kInherit, PlaintextMode::kCompiled, 4);"
func.func @prepare_linear_transform_compact_pt(
    %ctx: !context, %diagonals: tensor<2x4xf64>) -> tensor<!linear_transform> {
  %out = tensor.empty() : tensor<!linear_transform>
  %result = cheddar.prepare_linear_transform %ctx, %diagonals, %out
      {diagonal_indices = array<i32: 0, 1>, width = 4 : i64, level = 1 : i64,
       bs = 2 : i64, gs = 1 : i64, log_pt_size_per_prime = 3 : i64}
      : (!context, tensor<2x4xf64>, tensor<!linear_transform>)
      -> tensor<!linear_transform>
  return %result : tensor<!linear_transform>
}

// CHECK: func.func @apply_prepared_linear_transform
// CHECK: emitc.verbatim "{}->Evaluate(_lt_cp, {}[0], {}[0], {});"
func.func @apply_prepared_linear_transform(
    %ctx: !context, %ct: tensor<1x!ciphertext>, %evk: !evk_map,
    %transform: tensor<!linear_transform>) -> tensor<1x!ciphertext> {
  %out = tensor.empty() : tensor<1x!ciphertext>
  %result = cheddar.apply_prepared_linear_transform
      %ctx, %ct, %evk, %transform, %out
      : (!context, tensor<1x!ciphertext>, !evk_map,
         tensor<!linear_transform>, tensor<1x!ciphertext>)
      -> tensor<1x!ciphertext>
  return %result : tensor<1x!ciphertext>
}

// CHECK: func.func @apply_prepared_linear_transform_min_ks
// CHECK: emitc.verbatim "{}->Evaluate(_lt_cp, {}[0], {}[0], {}, true);"
func.func @apply_prepared_linear_transform_min_ks(
    %ctx: !context, %ct: tensor<1x!ciphertext>, %evk: !evk_map,
    %transform: tensor<!linear_transform>) -> tensor<1x!ciphertext> {
  %out = tensor.empty() : tensor<1x!ciphertext>
  %result = cheddar.apply_prepared_linear_transform
      %ctx, %ct, %evk, %transform, %out {min_ks = true}
      : (!context, tensor<1x!ciphertext>, !evk_map,
         tensor<!linear_transform>, tensor<1x!ciphertext>)
      -> tensor<1x!ciphertext>
  return %result : tensor<1x!ciphertext>
}

{-#
  dialect_resources: {
    builtin: {
      weights: "0x040000000000803f000000400000404000008040"
    }
  }
#-}
