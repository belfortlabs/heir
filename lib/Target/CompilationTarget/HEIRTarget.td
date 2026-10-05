#ifndef LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_
#define LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_

class CompilationTarget<string name> {
  string backendName = name;
  int bootstrapLevelsConsumed = 0;
  int has_kernel_chebyshev = 0;
  int has_kernel_linear_transform = 0;
  // Whether the backend can prepare a linear transform once (encode its
  // diagonals into plaintexts) and apply it to many ciphertexts, via
  // kernel.prepare_linear_transform / kernel.apply_linear_transform.
  int has_prepared_linear_transform = 0;
  int supports_single_diagonal_prepared_linear_transform = 1;
  // Whether ciphertext-plaintext operations require equal modulus-chain levels.
  int requires_matching_ciphertext_plaintext_levels = 0;
  int can_emit_adjust_scale = 1;
  // Whether the backend evaluates a whole max pooling layer in one call, via
  // kernel.max_pool.
  int has_kernel_max_pool = 0;
}

#endif // LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_
