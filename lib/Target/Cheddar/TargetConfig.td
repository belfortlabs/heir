include "lib/Target/CompilationTarget/HEIRTarget.td"

def Cheddar : CompilationTarget<"cheddar"> {
  // scale-snu/cheddar's bootstrap circuit uses a separate extension of the
  // compute modulus chain, so a bootstrap consumes no compute levels.
  let bootstrapLevelsConsumed = 0;

  let has_kernel_linear_transform = 1;
  let has_prepared_linear_transform = 1;
  let supports_single_diagonal_prepared_linear_transform = 0;
  let has_kernel_chebyshev = 1;

  // Cheddar rejects ciphertext-plaintext operations with different prime sets.
  let requires_matching_ciphertext_plaintext_levels = 1;

  // scale-snu/cheddar exposes one canonical scale per level and has no
  // operation for a same-level scalar scale correction.
  let can_emit_adjust_scale = 0;

  // Cheddar's NTT supports ring degrees up to 2^16 (NTTHandler's
  // max_log_degree_), and its parameter sets end at logN = 16.
  let max_ring_degree = 65536;
}
