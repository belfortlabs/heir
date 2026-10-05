// RUN: heir-opt --cheddar-to-emitc %s | FileCheck --implicit-check-not=math. %s
// RUN: heir-opt --cheddar-to-emitc %s | heir-translate --mlir-to-cpp | FileCheck --check-prefix=CPP %s

// Every math op except absi, ipowi, ctpop and cttz reaches C++: MathToEmitC
// lowers the ones with a <cmath> function, and math-expand-ops and arith-expand
// rewrite rsqrt, clampf, fpowi and ctlz into ops that lower.
// tests/Conversions/CheddarToEmitC/compile builds the result.

// CHECK: func.func @log
// CHECK: emitc.call_opaque "std::log"
// CHECK: func.func @roundeven
// CHECK: emitc.call_opaque "std::nearbyint"
// CHECK: func.func @rsqrt
// CHECK: emitc.call_opaque "std::sqrt"
// CHECK: emitc.div
// CHECK: func.func @isnan
// CHECK: emitc.call_opaque "std::isnan"{{.*}} -> i1
// CHECK: func.func @sincos
// CHECK: emitc.call_opaque "std::sin"
// CHECK: emitc.call_opaque "std::cos"
// CHECK: func.func @fpowi(
// CHECK: emitc.call_opaque "std::pow"

// CPP: #include <cmath>
// CPP: std::tanh(

func.func @absf(%x: f32) -> f32 {
  %0 = math.absf %x : f32
  return %0 : f32
}

func.func @acos(%x: f32) -> f32 {
  %0 = math.acos %x : f32
  return %0 : f32
}

func.func @acosh(%x: f32) -> f32 {
  %0 = math.acosh %x : f32
  return %0 : f32
}

func.func @asin(%x: f32) -> f32 {
  %0 = math.asin %x : f32
  return %0 : f32
}

func.func @asinh(%x: f32) -> f32 {
  %0 = math.asinh %x : f32
  return %0 : f32
}

func.func @atan(%x: f32) -> f32 {
  %0 = math.atan %x : f32
  return %0 : f32
}

func.func @atanh(%x: f32) -> f32 {
  %0 = math.atanh %x : f32
  return %0 : f32
}

func.func @cbrt(%x: f32) -> f32 {
  %0 = math.cbrt %x : f32
  return %0 : f32
}

func.func @ceil(%x: f32) -> f32 {
  %0 = math.ceil %x : f32
  return %0 : f32
}

func.func @cos(%x: f32) -> f32 {
  %0 = math.cos %x : f32
  return %0 : f32
}

func.func @cosh(%x: f32) -> f32 {
  %0 = math.cosh %x : f32
  return %0 : f32
}

func.func @erf(%x: f32) -> f32 {
  %0 = math.erf %x : f32
  return %0 : f32
}

func.func @erfc(%x: f32) -> f32 {
  %0 = math.erfc %x : f32
  return %0 : f32
}

func.func @exp(%x: f32) -> f32 {
  %0 = math.exp %x : f32
  return %0 : f32
}

func.func @exp2(%x: f32) -> f32 {
  %0 = math.exp2 %x : f32
  return %0 : f32
}

func.func @expm1(%x: f32) -> f32 {
  %0 = math.expm1 %x : f32
  return %0 : f32
}

func.func @floor(%x: f32) -> f32 {
  %0 = math.floor %x : f32
  return %0 : f32
}

func.func @log(%x: f32) -> f32 {
  %0 = math.log %x : f32
  return %0 : f32
}

func.func @log10(%x: f32) -> f32 {
  %0 = math.log10 %x : f32
  return %0 : f32
}

func.func @log1p(%x: f32) -> f32 {
  %0 = math.log1p %x : f32
  return %0 : f32
}

func.func @log2(%x: f32) -> f32 {
  %0 = math.log2 %x : f32
  return %0 : f32
}

func.func @round(%x: f32) -> f32 {
  %0 = math.round %x : f32
  return %0 : f32
}

func.func @roundeven(%x: f32) -> f32 {
  %0 = math.roundeven %x : f32
  return %0 : f32
}

func.func @rsqrt(%x: f32) -> f32 {
  %0 = math.rsqrt %x : f32
  return %0 : f32
}

func.func @sin(%x: f32) -> f32 {
  %0 = math.sin %x : f32
  return %0 : f32
}

func.func @sinh(%x: f32) -> f32 {
  %0 = math.sinh %x : f32
  return %0 : f32
}

func.func @sqrt(%x: f32) -> f32 {
  %0 = math.sqrt %x : f32
  return %0 : f32
}

func.func @tan(%x: f32) -> f32 {
  %0 = math.tan %x : f32
  return %0 : f32
}

func.func @tanh(%x: f32) -> f32 {
  %0 = math.tanh %x : f32
  return %0 : f32
}

func.func @trunc(%x: f32) -> f32 {
  %0 = math.trunc %x : f32
  return %0 : f32
}

func.func @atan2(%x: f32, %y: f32) -> f32 {
  %0 = math.atan2 %x, %y : f32
  return %0 : f32
}

func.func @copysign(%x: f32, %y: f32) -> f32 {
  %0 = math.copysign %x, %y : f32
  return %0 : f32
}

func.func @powf(%x: f32, %y: f32) -> f32 {
  %0 = math.powf %x, %y : f32
  return %0 : f32
}

func.func @fma(%x: f64, %y: f64, %z: f64) -> f64 {
  %0 = math.fma %x, %y, %z : f64
  return %0 : f64
}

func.func @isnan(%x: f32) -> i1 {
  %0 = math.isnan %x : f32
  return %0 : i1
}

func.func @isinf(%x: f32) -> i1 {
  %0 = math.isinf %x : f32
  return %0 : i1
}

func.func @isfinite(%x: f32) -> i1 {
  %0 = math.isfinite %x : f32
  return %0 : i1
}

func.func @isnormal(%x: f32) -> i1 {
  %0 = math.isnormal %x : f32
  return %0 : i1
}

func.func @sincos(%x: f32) -> (f32, f32) {
  %0, %1 = math.sincos %x : f32
  return %0, %1 : f32, f32
}

func.func @clampf(%x: f32, %lo: f32, %hi: f32) -> f32 {
  %0 = math.clampf %x to [%lo, %hi] : f32
  return %0 : f32
}

func.func @fpowi(%x: f32, %n: i32) -> f32 {
  %0 = math.fpowi %x, %n : f32, i32
  return %0 : f32
}

func.func @fpowi_constant(%x: f32) -> f32 {
  %c3 = arith.constant 3 : i32
  %0 = math.fpowi %x, %c3 : f32, i32
  return %0 : f32
}

func.func @ctlz(%x: i32) -> i32 {
  %0 = math.ctlz %x : i32
  return %0 : i32
}
