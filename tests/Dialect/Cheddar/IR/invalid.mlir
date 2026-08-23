// RUN: heir-opt --split-input-file --verify-diagnostics %s

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!evk_map = !cheddar.evk_map
!ui = !cheddar.user_interface

// (0, 0) is the "let the runtime plan the split" encoding. A single zero names
// no grid at all.
func.func @half_planned_grid(%ctx: !context, %ct: tensor<!ciphertext>, %evk: !evk_map, %diagonals: tensor<2x8xf32>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  // expected-error@below {{bs and gs must be positive, or both zero to let the runtime plan the split}}
  %0 = cheddar.linear_transform %ctx, %ct, %evk, %diagonals, %d0 {diagonal_indices = array<i32: 0, 1>, level = 1 : i64, bs = 0 : i64, gs = 2 : i64} : (!context, tensor<!ciphertext>, !evk_map, tensor<2x8xf32>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

// -----

!ciphertext = !cheddar.ciphertext
!context = !cheddar.context
!evk_map = !cheddar.evk_map

// min_ks generates only the two progression-stride keys, which needs a known
// baby/giant split.
func.func @planned_grid_with_min_ks(%ctx: !context, %ct: tensor<!ciphertext>, %evk: !evk_map, %diagonals: tensor<2x8xf32>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  // expected-error@below {{min_ks needs a known baby/giant split}}
  %0 = cheddar.linear_transform %ctx, %ct, %evk, %diagonals, %d0 {diagonal_indices = array<i32: 0, 1>, level = 1 : i64, bs = 0 : i64, gs = 0 : i64, min_ks = true} : (!context, tensor<!ciphertext>, !evk_map, tensor<2x8xf32>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

// -----

!context = !cheddar.context
!ui = !cheddar.user_interface

func.func @transform_keys_bad_width(%ctx: tensor<!context>, %ui: tensor<!ui>) -> tensor<!ui> {
  // expected-error@below {{width must be a positive power of two}}
  %0 = cheddar.prepare_linear_transform_keys %ctx, %ui {diagonal_indices = array<i32: 0, 3>, width = 6 : i64, level = 1 : i64} : (tensor<!context>, tensor<!ui>) -> tensor<!ui>
  return %0 : tensor<!ui>
}

// -----

!context = !cheddar.context
!ui = !cheddar.user_interface

func.func @transform_keys_duplicate_diagonal(%ctx: tensor<!context>, %ui: tensor<!ui>) -> tensor<!ui> {
  // expected-error@below {{duplicate normalized diagonal index 3}}
  %0 = cheddar.prepare_linear_transform_keys %ctx, %ui {diagonal_indices = array<i32: 3, 11>, width = 8 : i64, level = 1 : i64} : (tensor<!context>, tensor<!ui>) -> tensor<!ui>
  return %0 : tensor<!ui>
}
