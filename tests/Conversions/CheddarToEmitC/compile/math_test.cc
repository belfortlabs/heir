// Runs the C++ that --cheddar-to-emitc emits for math.mlir and checks it
// computes the math op each function wraps.

#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

#include "gtest/gtest.h"  // from @googletest

// Emitted by math.mlir; see BUILD.
float test_absf(float);
float test_acos(float);
float test_acosh(float);
float test_asin(float);
float test_asinh(float);
float test_atan(float);
float test_atanh(float);
float test_cbrt(float);
float test_ceil(float);
float test_cos(float);
float test_cosh(float);
float test_erf(float);
float test_erfc(float);
float test_exp(float);
float test_exp2(float);
float test_expm1(float);
float test_floor(float);
float test_log(float);
float test_log10(float);
float test_log1p(float);
float test_log2(float);
float test_round(float);
float test_roundeven(float);
float test_rsqrt(float);
float test_sin(float);
float test_sinh(float);
float test_sqrt(float);
float test_tan(float);
float test_tanh(float);
float test_trunc(float);
float test_atan2(float, float);
float test_copysign(float, float);
float test_powf(float, float);
double test_fma(double, double, double);
bool test_isnan(float);
bool test_isinf(float);
bool test_isfinite(float);
bool test_isnormal(float);
std::tuple<float, float> test_sincos(float);
float test_clampf(float, float, float);
float test_fpowi(float, int32_t);
float test_fpowi_constant(float);
int32_t test_ctlz(int32_t);

namespace {

constexpr float kTol = 1e-6f;

TEST(CheddarEmitCMath, Rsqrt) {
  EXPECT_FLOAT_EQ(test_rsqrt(16.0f), 0.25f);
  EXPECT_FLOAT_EQ(test_rsqrt(0.25f), 2.0f);
}

TEST(CheddarEmitCMath, Clampf) {
  EXPECT_FLOAT_EQ(test_clampf(5.0f, -1.0f, 2.0f), 2.0f);
  EXPECT_FLOAT_EQ(test_clampf(-5.0f, -1.0f, 2.0f), -1.0f);
  EXPECT_FLOAT_EQ(test_clampf(0.5f, -1.0f, 2.0f), 0.5f);
}

TEST(CheddarEmitCMath, Fpowi) {
  EXPECT_FLOAT_EQ(test_fpowi(2.0f, 10), 1024.0f);
  EXPECT_FLOAT_EQ(test_fpowi(2.0f, -2), 0.25f);
  EXPECT_FLOAT_EQ(test_fpowi_constant(-3.0f), -27.0f);
}

TEST(CheddarEmitCMath, Ctlz) {
  EXPECT_EQ(test_ctlz(1), 31);
  EXPECT_EQ(test_ctlz(0x00010000), 15);
  EXPECT_EQ(test_ctlz(0), 32);
  EXPECT_EQ(test_ctlz(-1), 0);
}

TEST(CheddarEmitCMath, RoundingOps) {
  EXPECT_FLOAT_EQ(test_roundeven(2.5f), 2.0f);
  EXPECT_FLOAT_EQ(test_roundeven(3.5f), 4.0f);
  EXPECT_FLOAT_EQ(test_roundeven(-2.5f), -2.0f);
  EXPECT_FLOAT_EQ(test_round(2.5f), 3.0f);
  EXPECT_FLOAT_EQ(test_trunc(-2.7f), -2.0f);
  EXPECT_FLOAT_EQ(test_floor(-2.5f), -3.0f);
  EXPECT_FLOAT_EQ(test_ceil(-2.5f), -2.0f);
}

TEST(CheddarEmitCMath, Classification) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const float subnormal = std::numeric_limits<float>::denorm_min();
  EXPECT_TRUE(test_isnan(nan));
  EXPECT_FALSE(test_isnan(1.0f));
  EXPECT_TRUE(test_isinf(-inf));
  EXPECT_FALSE(test_isinf(1.0f));
  EXPECT_TRUE(test_isfinite(1.0f));
  EXPECT_FALSE(test_isfinite(inf));
  EXPECT_TRUE(test_isnormal(1.0f));
  EXPECT_FALSE(test_isnormal(subnormal));
}

TEST(CheddarEmitCMath, Sincos) {
  auto [s, c] = test_sincos(0.5f);
  EXPECT_NEAR(s, 0.479425539f, kTol);
  EXPECT_NEAR(c, 0.877582562f, kTol);
}

// Known values, so a function wired to the wrong <cmath> call (log vs log10,
// sinh vs cosh, ...) fails.
TEST(CheddarEmitCMath, DirectCalls) {
  EXPECT_FLOAT_EQ(test_absf(-1.5f), 1.5f);
  EXPECT_NEAR(test_acos(0.5f), 1.04719755f, kTol);
  EXPECT_NEAR(test_acosh(2.0f), 1.31695790f, kTol);
  EXPECT_NEAR(test_asin(0.5f), 0.523598776f, kTol);
  EXPECT_NEAR(test_asinh(1.0f), 0.881373587f, kTol);
  EXPECT_NEAR(test_atan(1.0f), 0.785398163f, kTol);
  EXPECT_NEAR(test_atanh(0.5f), 0.549306144f, kTol);
  EXPECT_FLOAT_EQ(test_cbrt(-27.0f), -3.0f);
  EXPECT_NEAR(test_cos(0.0f), 1.0f, kTol);
  EXPECT_NEAR(test_cosh(1.0f), 1.54308063f, kTol);
  EXPECT_NEAR(test_erf(0.5f), 0.520499878f, kTol);
  EXPECT_NEAR(test_erfc(0.5f), 0.479500122f, kTol);
  EXPECT_NEAR(test_exp(1.0f), 2.71828183f, kTol);
  EXPECT_FLOAT_EQ(test_exp2(10.0f), 1024.0f);
  EXPECT_NEAR(test_expm1(1e-3f), 1.00050017e-3f, 1e-9f);
  EXPECT_NEAR(test_log(2.71828183f), 1.0f, kTol);
  EXPECT_FLOAT_EQ(test_log10(1000.0f), 3.0f);
  EXPECT_NEAR(test_log1p(1e-3f), 9.99500333e-4f, 1e-9f);
  EXPECT_FLOAT_EQ(test_log2(1024.0f), 10.0f);
  EXPECT_NEAR(test_sin(0.5f), 0.479425539f, kTol);
  EXPECT_NEAR(test_sinh(1.0f), 1.17520119f, kTol);
  EXPECT_FLOAT_EQ(test_sqrt(16.0f), 4.0f);
  EXPECT_NEAR(test_tan(0.5f), 0.546302490f, kTol);
  EXPECT_NEAR(test_tanh(0.5f), 0.462117157f, kTol);
  EXPECT_NEAR(test_atan2(1.0f, -1.0f), 2.35619449f, kTol);
  EXPECT_FLOAT_EQ(test_copysign(2.0f, -0.0f), -2.0f);
  EXPECT_FLOAT_EQ(test_powf(2.0f, 0.5f), std::sqrt(2.0f));
  EXPECT_DOUBLE_EQ(test_fma(2.0, 3.0, 4.0), 10.0);
}

}  // namespace
