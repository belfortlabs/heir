
!ciphertext = !cheddar.ciphertext
!plaintext = !cheddar.plaintext
!context = !cheddar.context
!encoder = !cheddar.encoder
!user_interface = !cheddar.user_interface
!evk_map = !cheddar.evk_map
!linear_transform = !cheddar.linear_transform

// Rotation and conjugation read the key off the EvkMap with the secret handle
// and the parameters from the context: Cyclops has no level-blind getter, so
// this is the only shape that compiles against it.
func.func @hrot_static(%ctx: !context, %evk: !evk_map,
                       %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.hrot %ctx, %evk, %ct, %d0
      {level = 4 : i64, static_distance = 5 : i64}
      : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>)
      -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

func.func @hrot_dynamic(%ctx: !context, %evk: !evk_map,
                        %ct: tensor<!ciphertext>, %d: index)
    -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.hrot %ctx, %evk, %ct, %d0, %d {level = 4 : i64}
      : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>,
         index) -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

func.func @hrot_add(%ctx: !context, %evk: !evk_map,
                    %a: tensor<!ciphertext>, %b: tensor<!ciphertext>)
    -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.hrot_add %ctx, %evk, %a, %b, %d0
      {distance = 5 : i64, level = 4 : i64}
      : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>,
         tensor<!ciphertext>) -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

func.func @hconj(%ctx: !context, %evk: !evk_map,
                 %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.hconj %ctx, %evk, %ct, %d0
      : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>)
      -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

func.func @hconj_add(%ctx: !context, %evk: !evk_map,
                     %a: tensor<!ciphertext>, %b: tensor<!ciphertext>)
    -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.hconj_add %ctx, %evk, %a, %b, %d0
      : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>,
         tensor<!ciphertext>) -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

// Encode tags the fresh plaintext before Encrypt, which rejects an untagged
// one, and reads the message as real doubles.
func.func @encode_encrypt(%ctx: !context, %enc: !encoder,
                          %msg: tensor<4xf64>, %ui: !user_interface)
    -> tensor<!ciphertext> {
  %dp = tensor.empty() : tensor<!plaintext>
  %pt = cheddar.encode %ctx, %enc, %msg, %dp
      {level = 5 : i64, logScale = 37 : i64, useSlotsApi}
      : (!context, !encoder, tensor<4xf64>, tensor<!plaintext>)
      -> tensor<!plaintext>
  %dc = tensor.empty() : tensor<!ciphertext>
  %ct = cheddar.encrypt %ui, %pt, %dc
      : (!user_interface, tensor<!plaintext>, tensor<!ciphertext>)
      -> tensor<!ciphertext>
  return %ct : tensor<!ciphertext>
}

// EvalPoly takes a MultKeySelector over the map, not a fixed key.
func.func @eval_poly(%ctx: !context, %ct: tensor<!ciphertext>, %evk: !evk_map)
    -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.eval_poly %ctx, %ct, %evk, %d0
      {coefficients = [1.0 : f64, 2.0 : f64, 3.0 : f64],
       levelConsumption = 2 : i64, selectMultKeyAtUseLevel}
      : (!context, tensor<!ciphertext>, !evk_map, tensor<!ciphertext>)
      -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

// bs = gs = 0 hands the baby/giant split to Cyclops' planner, and the compact
// plaintext period follows in the position scale-snu gives to pre_rotation.
func.func @linear_transform(%ctx: !context, %ct: tensor<!ciphertext>,
                            %evk: !evk_map, %diagonals: tensor<2x8xf32>)
    -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.linear_transform %ctx, %ct, %evk, %diagonals, %d0
      {diagonal_indices = array<i32: 0, 1>, level = 1 : i64,
       bs = 0 : i64, gs = 0 : i64, log_pt_size_per_prime = 4 : i64}
      : (!context, tensor<!ciphertext>, !evk_map, tensor<2x8xf32>,
         tensor<!ciphertext>) -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

func.func @prepare_linear_transform(%ctx: !context,
                                    %diagonals: tensor<2x8xf32>)
    -> tensor<!linear_transform> {
  %d0 = tensor.empty() : tensor<!linear_transform>
  %0 = cheddar.prepare_linear_transform %ctx, %diagonals, %d0
      {diagonal_indices = array<i32: 0, 1>, width = 8 : i64, level = 1 : i64,
       bs = 0 : i64, gs = 0 : i64, log_pt_size_per_prime = 4 : i64}
      : (!context, tensor<2x8xf32>, tensor<!linear_transform>)
      -> tensor<!linear_transform>
  return %0 : tensor<!linear_transform>
}

func.func @apply_prepared_linear_transform(%ctx: !context,
                                           %ct: tensor<!ciphertext>,
                                           %evk: !evk_map,
                                           %lt: tensor<!linear_transform>)
    -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.apply_prepared_linear_transform %ctx, %ct, %evk, %lt, %d0
      : (!context, tensor<!ciphertext>, !evk_map, tensor<!linear_transform>,
         tensor<!ciphertext>) -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}
