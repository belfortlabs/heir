// Compiled as C++ (see BUILD): every math op the cheddar EmitC lowering supports.

func.func @test_absf(%x: f32) -> f32 {
  %0 = math.absf %x : f32
  return %0 : f32
}

func.func @test_acos(%x: f32) -> f32 {
  %0 = math.acos %x : f32
  return %0 : f32
}

func.func @test_acosh(%x: f32) -> f32 {
  %0 = math.acosh %x : f32
  return %0 : f32
}

func.func @test_asin(%x: f32) -> f32 {
  %0 = math.asin %x : f32
  return %0 : f32
}

func.func @test_asinh(%x: f32) -> f32 {
  %0 = math.asinh %x : f32
  return %0 : f32
}

func.func @test_atan(%x: f32) -> f32 {
  %0 = math.atan %x : f32
  return %0 : f32
}

func.func @test_atanh(%x: f32) -> f32 {
  %0 = math.atanh %x : f32
  return %0 : f32
}

func.func @test_cbrt(%x: f32) -> f32 {
  %0 = math.cbrt %x : f32
  return %0 : f32
}

func.func @test_ceil(%x: f32) -> f32 {
  %0 = math.ceil %x : f32
  return %0 : f32
}

func.func @test_cos(%x: f32) -> f32 {
  %0 = math.cos %x : f32
  return %0 : f32
}

func.func @test_cosh(%x: f32) -> f32 {
  %0 = math.cosh %x : f32
  return %0 : f32
}

func.func @test_erf(%x: f32) -> f32 {
  %0 = math.erf %x : f32
  return %0 : f32
}

func.func @test_erfc(%x: f32) -> f32 {
  %0 = math.erfc %x : f32
  return %0 : f32
}

func.func @test_exp(%x: f32) -> f32 {
  %0 = math.exp %x : f32
  return %0 : f32
}

func.func @test_exp2(%x: f32) -> f32 {
  %0 = math.exp2 %x : f32
  return %0 : f32
}

func.func @test_expm1(%x: f32) -> f32 {
  %0 = math.expm1 %x : f32
  return %0 : f32
}

func.func @test_floor(%x: f32) -> f32 {
  %0 = math.floor %x : f32
  return %0 : f32
}

func.func @test_log(%x: f32) -> f32 {
  %0 = math.log %x : f32
  return %0 : f32
}

func.func @test_log10(%x: f32) -> f32 {
  %0 = math.log10 %x : f32
  return %0 : f32
}

func.func @test_log1p(%x: f32) -> f32 {
  %0 = math.log1p %x : f32
  return %0 : f32
}

func.func @test_log2(%x: f32) -> f32 {
  %0 = math.log2 %x : f32
  return %0 : f32
}

func.func @test_round(%x: f32) -> f32 {
  %0 = math.round %x : f32
  return %0 : f32
}

func.func @test_roundeven(%x: f32) -> f32 {
  %0 = math.roundeven %x : f32
  return %0 : f32
}

func.func @test_rsqrt(%x: f32) -> f32 {
  %0 = math.rsqrt %x : f32
  return %0 : f32
}

func.func @test_sin(%x: f32) -> f32 {
  %0 = math.sin %x : f32
  return %0 : f32
}

func.func @test_sinh(%x: f32) -> f32 {
  %0 = math.sinh %x : f32
  return %0 : f32
}

func.func @test_sqrt(%x: f32) -> f32 {
  %0 = math.sqrt %x : f32
  return %0 : f32
}

func.func @test_tan(%x: f32) -> f32 {
  %0 = math.tan %x : f32
  return %0 : f32
}

func.func @test_tanh(%x: f32) -> f32 {
  %0 = math.tanh %x : f32
  return %0 : f32
}

func.func @test_trunc(%x: f32) -> f32 {
  %0 = math.trunc %x : f32
  return %0 : f32
}

func.func @test_atan2(%x: f32, %y: f32) -> f32 {
  %0 = math.atan2 %x, %y : f32
  return %0 : f32
}

func.func @test_copysign(%x: f32, %y: f32) -> f32 {
  %0 = math.copysign %x, %y : f32
  return %0 : f32
}

func.func @test_powf(%x: f32, %y: f32) -> f32 {
  %0 = math.powf %x, %y : f32
  return %0 : f32
}

func.func @test_fma(%x: f64, %y: f64, %z: f64) -> f64 {
  %0 = math.fma %x, %y, %z : f64
  return %0 : f64
}

func.func @test_isnan(%x: f32) -> i1 {
  %0 = math.isnan %x : f32
  return %0 : i1
}

func.func @test_isinf(%x: f32) -> i1 {
  %0 = math.isinf %x : f32
  return %0 : i1
}

func.func @test_isfinite(%x: f32) -> i1 {
  %0 = math.isfinite %x : f32
  return %0 : i1
}

func.func @test_isnormal(%x: f32) -> i1 {
  %0 = math.isnormal %x : f32
  return %0 : i1
}

func.func @test_sincos(%x: f32) -> (f32, f32) {
  %0, %1 = math.sincos %x : f32
  return %0, %1 : f32, f32
}

func.func @test_clampf(%x: f32, %lo: f32, %hi: f32) -> f32 {
  %0 = math.clampf %x to [%lo, %hi] : f32
  return %0 : f32
}

func.func @test_fpowi(%x: f32, %n: i32) -> f32 {
  %0 = math.fpowi %x, %n : f32, i32
  return %0 : f32
}

func.func @test_fpowi_constant(%x: f32) -> f32 {
  %c3 = arith.constant 3 : i32
  %0 = math.fpowi %x, %c3 : f32, i32
  return %0 : f32
}

func.func @test_ctlz(%x: i32) -> i32 {
  %0 = math.ctlz %x : i32
  return %0 : i32
}
