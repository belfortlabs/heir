// RUN: heir-opt --split-input-file --verify-diagnostics %s

func.func @test_diagonals_not_shaped(%arg0: tensor<4xf32>, %diagonals: f32) -> tensor<4xf32> {
  // expected-error@below {{diagonals must have a shaped type}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0>
  } : tensor<4xf32>, f32 -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @test_diagonals_not_2d(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %diagonals = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  // expected-error@below {{diagonals must be a 2D tensor}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0>
  } : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @test_input_not_1d_or_2d(%arg0: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
  %diagonals = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  // expected-error@below {{input must be 1D or 2D ranked tensor}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<1x2x3xf32>, tensor<2x3xf32> -> tensor<1x2x3xf32>
  return %0 : tensor<1x2x3xf32>
}

// -----

func.func @test_slot_size_mismatch(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %diagonals = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  // expected-error@below {{input slot size (2) is smaller than diagonals slot size (3)}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<2xf32>, tensor<2x3xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

func.func @test_diagonals_indices_mismatch(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %diagonals = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
  // expected-error@below {{number of diagonals (2) must match number of diagonal indices (1)}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0>
  } : tensor<4xf32>, tensor<2x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @test_batch_dim_not_1(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %diagonals = arith.constant dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>
  // expected-error@below {{input tensor batch dimension (first dimension) must be 1}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0>
  } : tensor<2x4xf32>, tensor<1x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @test_prepare_diagonals_indices_mismatch() -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0> {
  %diagonals = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
  // expected-error@below {{number of diagonals (2) must match number of diagonal indices (1)}}
  %0 = kernel.prepare_linear_transform %diagonals {
    diagonal_indices = array<i64: 0>
  } : tensor<2x4xf32> -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
  return %0 : !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
}

// -----

func.func @test_prepare_source_row_out_of_bounds() -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0> {
  %diagonals = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
  // expected-error@below {{source row index 2 is out of bounds for 2 diagonal rows}}
  %0 = kernel.prepare_linear_transform %diagonals {
    diagonal_indices = array<i64: 0>, source_row_indices = array<i64: 2>
  } : tensor<2x4xf32> -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
  return %0 : !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
}

// -----

func.func @test_prepare_slots_too_small() -> !kernel.prepared_linear_transform<level = 0, slots = 2, log_bsgs_ratio = 0> {
  %diagonals = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
  // expected-error@below {{diagonals slot size (4) exceeds the prepared slot count (2)}}
  %0 = kernel.prepare_linear_transform %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<2x4xf32> -> !kernel.prepared_linear_transform<level = 0, slots = 2, log_bsgs_ratio = 0>
  return %0 : !kernel.prepared_linear_transform<level = 0, slots = 2, log_bsgs_ratio = 0>
}

// -----

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain>

// The ciphertext sits at level 0 (its chain's current); a transform
// prepared for level 1 must be rejected.
func.func @test_apply_level_mismatch(%ct: !ct) -> !ct {
  %diagonals = arith.constant dense<1.0> : tensor<2x512xf64>
  %lt = kernel.prepare_linear_transform %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<2x512xf64> -> !kernel.prepared_linear_transform<level = 1, slots = 512, log_bsgs_ratio = 0>
  // expected-error@below {{input ciphertext level (0) does not match the prepared transform level (1)}}
  %0 = kernel.apply_linear_transform %ct, %lt : !ct, !kernel.prepared_linear_transform<level = 1, slots = 512, log_bsgs_ratio = 0> -> !ct
  return %0 : !ct
}

// -----

func.func @test_max_pool_type_mismatch(%arg0: tensor<1x1024xf32>) -> tensor<1x1024xf64> {
  // expected-error@below {{input and output types must match}}
  %0 = kernel.max_pool %arg0 {num_slots = 1024 : i64, input_length = 1024 : i64, window_size = 4 : i64, stride = 4 : i64, value_bound = 0.5 : f64} : tensor<1x1024xf32> -> tensor<1x1024xf64>
  return %0 : tensor<1x1024xf64>
}

// -----

func.func @test_max_pool_two_ciphertexts(%arg0: tensor<2x1024xf32>) -> tensor<2x1024xf32> {
  // expected-error@below {{input must be a single ciphertext}}
  %0 = kernel.max_pool %arg0 {num_slots = 1024 : i64, input_length = 1024 : i64, window_size = 4 : i64, stride = 4 : i64, value_bound = 0.5 : f64} : tensor<2x1024xf32> -> tensor<2x1024xf32>
  return %0 : tensor<2x1024xf32>
}

// -----

func.func @test_max_pool_slots_not_power_of_two(%arg0: tensor<1x24xf32>) -> tensor<1x24xf32> {
  // expected-error@below {{num_slots must be a power of two, but got 24}}
  %0 = kernel.max_pool %arg0 {num_slots = 24 : i64, input_length = 24 : i64, window_size = 4 : i64, stride = 4 : i64, value_bound = 0.5 : f64} : tensor<1x24xf32> -> tensor<1x24xf32>
  return %0 : tensor<1x24xf32>
}

// -----

func.func @test_max_pool_slot_dimension_mismatch(%arg0: tensor<1x1024xf32>) -> tensor<1x1024xf32> {
  // expected-error@below {{the slot dimension (1024) must equal num_slots (2048)}}
  %0 = kernel.max_pool %arg0 {num_slots = 2048 : i64, input_length = 1024 : i64, window_size = 4 : i64, stride = 4 : i64, value_bound = 0.5 : f64} : tensor<1x1024xf32> -> tensor<1x1024xf32>
  return %0 : tensor<1x1024xf32>
}

// -----

func.func @test_max_pool_zero_window(%arg0: tensor<1x1024xf32>) -> tensor<1x1024xf32> {
  // expected-error@below {{window_size, stride and dilation must be at least 1}}
  %0 = kernel.max_pool %arg0 {num_slots = 1024 : i64, input_length = 1024 : i64, window_size = 0 : i64, stride = 4 : i64, value_bound = 0.5 : f64} : tensor<1x1024xf32> -> tensor<1x1024xf32>
  return %0 : tensor<1x1024xf32>
}

// -----

func.func @test_max_pool_input_too_long(%arg0: tensor<1x1024xf32>) -> tensor<1x1024xf32> {
  // expected-error@below {{input_length must lie in [4, 1024], but got 1025}}
  %0 = kernel.max_pool %arg0 {num_slots = 1024 : i64, input_length = 1025 : i64, window_size = 4 : i64, stride = 4 : i64, value_bound = 0.5 : f64} : tensor<1x1024xf32> -> tensor<1x1024xf32>
  return %0 : tensor<1x1024xf32>
}

// -----

func.func @test_max_pool_input_shorter_than_window(%arg0: tensor<1x1024xf32>) -> tensor<1x1024xf32> {
  // expected-error@below {{input_length must lie in [7, 1024], but got 6}}
  %0 = kernel.max_pool %arg0 {num_slots = 1024 : i64, input_length = 6 : i64, window_size = 4 : i64, stride = 4 : i64, dilation = 2 : i64, value_bound = 0.5 : f64} : tensor<1x1024xf32> -> tensor<1x1024xf32>
  return %0 : tensor<1x1024xf32>
}

// -----

func.func @test_max_pool_overlapping_dilated_windows(%arg0: tensor<1x1024xf32>) -> tensor<1x1024xf32> {
  // expected-error@below {{dilated windows must not overlap, but stride 4 is less than the window span 7}}
  %0 = kernel.max_pool %arg0 {num_slots = 1024 : i64, input_length = 1000 : i64, window_size = 4 : i64, stride = 4 : i64, dilation = 2 : i64, value_bound = 0.5 : f64} : tensor<1x1024xf32> -> tensor<1x1024xf32>
  return %0 : tensor<1x1024xf32>
}
