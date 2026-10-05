// RUN: heir-opt --lwe-to-cheddar=use-cyclops-runtime=true %s | FileCheck %s

// A window of 2 over 32 of 64 slots drops 4 levels: two league
// multiplications and two compaction stages.

#enc = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#chain_in = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64, 35184372088833 : i64, 35184371826689 : i64, 35184371761153 : i64>, current = 4>
#chain_out = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64, 35184372088833 : i64, 35184371826689 : i64, 35184371761153 : i64>, current = 0>
#rf = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_in = !rns.rns<!mod_arith.int<36028797018652673 : i64>, !mod_arith.int<35184372121601 : i64>, !mod_arith.int<35184372088833 : i64>, !mod_arith.int<35184371826689 : i64>, !mod_arith.int<35184371761153 : i64>>
!rns_out = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#rr_in = #polynomial.ring<coefficientType = !rns_in, polynomialModulus = <1 + x**1024>>
#rr_out = #polynomial.ring<coefficientType = !rns_out, polynomialModulus = <1 + x**1024>>
#cs_in = #lwe.ciphertext_space<ring = #rr_in, encryption_type = mix>
#cs_out = #lwe.ciphertext_space<ring = #rr_out, encryption_type = mix>
!ct_in = !lwe.lwe_ciphertext<plaintext_space = <ring = #rf, encoding = #enc>, ciphertext_space = #cs_in, key = #key, modulus_chain = #chain_in>
!ct_out = !lwe.lwe_ciphertext<plaintext_space = <ring = #rf, encoding = #enc>, ciphertext_space = #cs_out, key = #key, modulus_chain = #chain_out>

module attributes {backend.cheddar, ckks.schemeParam = #ckks.scheme_param<logN = 10, Q = [36028797018652673, 35184372121601, 35184372088833, 35184371826689, 35184371761153], P = [1152921504606994433], logDefaultScale = 45, encryptionTechnique = extended>, scheme.ckks} {
  // The pool bootstraps internally, so its function takes the boot context.
  // CHECK: func.func @max_pool(%[[bootCtx:[^:]*]]: !boot_context {{.*}}, %[[evk:[^:]*]]: !evk_map
  // CHECK: cheddar.max_pool %[[bootCtx]], %{{.*}}, %[[evk]], %{{.*}} {ceil_mode = false, dilation = 1 : i64, input_length = 32 : i64, level = 4 : i64, levelConsumption = 4 : i64, num_slots = 64 : i64, stride = 2 : i64, value_bound = 5.000000e-01 : f64, window_size = 2 : i64}
  // CHECK-SAME: !boot_context
  func.func @max_pool(%arg0: tensor<1x!ct_in>) -> tensor<1x!ct_out> {
    %0 = kernel.max_pool %arg0 {num_slots = 64 : i64, input_length = 32 : i64, window_size = 2 : i64, stride = 2 : i64, value_bound = 0.5 : f64} : tensor<1x!ct_in> -> tensor<1x!ct_out>
    return %0 : tensor<1x!ct_out>
  }
}
