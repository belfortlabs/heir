#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

using Ct = StridedMemRefType<float, 2>;
template <std::size_t N>
using Tensor = StridedMemRefType<float, N>;

extern "C" {
void _mlir_ciface_conv_padded(Ct* res, Ct* x, Tensor<4>* w, Tensor<1>* bias);
void _mlir_ciface_conv_padded__encrypt__arg0(Ct* res, Tensor<4>* arg);
void _mlir_ciface_conv_padded__decrypt__result0(Tensor<4>* res, Ct* arg);

void _mlir_ciface_conv_strided(Ct* res, Ct* x, Tensor<4>* w);
void _mlir_ciface_conv_strided__encrypt__arg0(Ct* res, Tensor<4>* arg);
void _mlir_ciface_conv_strided__decrypt__result0(Tensor<4>* res, Ct* arg);

void _mlir_ciface_matvec(Ct* res, Ct* x, Tensor<2>* m);
void _mlir_ciface_matvec__encrypt__arg0(Ct* res, Tensor<1>* arg);
void _mlir_ciface_matvec__decrypt__result0(Tensor<1>* res, Ct* arg);

void _mlir_ciface_conv_dilated(Ct* res, Ct* x, Tensor<4>* w);
void _mlir_ciface_conv_dilated__encrypt__arg0(Ct* res, Tensor<4>* arg);
void _mlir_ciface_conv_dilated__decrypt__result0(Tensor<4>* res, Ct* arg);

void _mlir_ciface_conv_depthwise(Ct* res, Ct* x, Tensor<3>* w);
void _mlir_ciface_conv_depthwise__encrypt__arg0(Ct* res, Tensor<4>* arg);
void _mlir_ciface_conv_depthwise__decrypt__result0(Tensor<4>* res, Ct* arg);

void _mlir_ciface_sum_pool(Ct* res, Ct* x);
void _mlir_ciface_sum_pool__encrypt__arg0(Ct* res, Tensor<4>* arg);
void _mlir_ciface_sum_pool__decrypt__result0(Tensor<4>* res, Ct* arg);

void _mlir_ciface_conv_grouped(Ct* res, Ct* x, Tensor<5>* w);
void _mlir_ciface_conv_grouped__encrypt__arg0(Ct* res, Tensor<4>* arg);
void _mlir_ciface_conv_grouped__decrypt__result0(Tensor<4>* res, Ct* arg);
}

namespace {

std::vector<float> randomValues(std::mt19937& rng, int64_t size) {
  std::uniform_real_distribution<float> dist(-1.0, 1.0);
  std::vector<float> values(size);
  for (float& v : values) v = dist(rng);
  return values;
}

// A row-major view of `values` with the given shape.
template <std::size_t N>
Tensor<N> view(std::vector<float>& values, const int64_t (&shape)[N]) {
  int64_t strides[N];
  int64_t stride = 1;
  for (int64_t i = N - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return Tensor<N>(values.data(), values.data(), 0, shape, strides);
}

template <std::size_t N>
void expectNear(const Tensor<N>& actual, const std::vector<float>& expected) {
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual.data[actual.offset + i], expected[i], 1e-3)
        << "at element " << i;
  }
}

// Reference 2-D convolution in NCHW/FCHW form with zero padding.
std::vector<float> conv2d(const std::vector<float>& x,
                          const std::vector<float>& w, int64_t channels,
                          int64_t filters, int64_t size, int64_t kernel,
                          int64_t stride, int64_t pad) {
  int64_t outSize = (size + 2 * pad - kernel) / stride + 1;
  std::vector<float> y(filters * outSize * outSize, 0.0f);
  for (int64_t f = 0; f < filters; ++f)
    for (int64_t i = 0; i < outSize; ++i)
      for (int64_t j = 0; j < outSize; ++j)
        for (int64_t c = 0; c < channels; ++c)
          for (int64_t u = 0; u < kernel; ++u)
            for (int64_t v = 0; v < kernel; ++v) {
              int64_t r = stride * i + u - pad;
              int64_t s = stride * j + v - pad;
              if (r < 0 || r >= size || s < 0 || s >= size) continue;
              y[(f * outSize + i) * outSize + j] +=
                  x[(c * size + r) * size + s] *
                  w[((f * channels + c) * kernel + u) * kernel + v];
            }
  return y;
}

TEST(LinearKernelTest, ConvPadded) {
  std::mt19937 rng(1);
  std::vector<float> x = randomValues(rng, 4 * 8 * 8);
  std::vector<float> w = randomValues(rng, 4 * 4 * 3 * 3);
  std::vector<float> bias = randomValues(rng, 4);
  std::vector<float> expected = conv2d(x, w, 4, 4, 8, 3, 1, 1);
  for (int64_t f = 0; f < 4; ++f)
    for (int64_t i = 0; i < 64; ++i) expected[f * 64 + i] += bias[f];

  Tensor<4> xRef = view<4>(x, {1, 4, 8, 8});
  Tensor<4> wRef = view<4>(w, {4, 4, 3, 3});
  Tensor<1> biasRef = view<1>(bias, {4});
  Ct enc, res;
  Tensor<4> out;
  _mlir_ciface_conv_padded__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_conv_padded(&res, &enc, &wRef, &biasRef);
  _mlir_ciface_conv_padded__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelTest, ConvStrided) {
  std::mt19937 rng(2);
  std::vector<float> x = randomValues(rng, 2 * 8 * 8);
  std::vector<float> w = randomValues(rng, 4 * 2 * 3 * 3);
  std::vector<float> expected = conv2d(x, w, 2, 4, 8, 3, 2, 1);

  Tensor<4> xRef = view<4>(x, {1, 2, 8, 8});
  Tensor<4> wRef = view<4>(w, {4, 2, 3, 3});
  Ct enc, res;
  Tensor<4> out;
  _mlir_ciface_conv_strided__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_conv_strided(&res, &enc, &wRef);
  _mlir_ciface_conv_strided__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelTest, ConvDilated) {
  std::mt19937 rng(7);
  std::vector<float> x = randomValues(rng, 2 * 8 * 8);
  std::vector<float> w = randomValues(rng, 3 * 2 * 3 * 3);
  // Dilation 2, no padding: 8 - 2 * (3 - 1) = 4 outputs per axis.
  std::vector<float> expected(3 * 4 * 4, 0.0f);
  for (int64_t f = 0; f < 3; ++f)
    for (int64_t i = 0; i < 4; ++i)
      for (int64_t j = 0; j < 4; ++j)
        for (int64_t c = 0; c < 2; ++c)
          for (int64_t u = 0; u < 3; ++u)
            for (int64_t v = 0; v < 3; ++v)
              expected[(f * 4 + i) * 4 + j] +=
                  x[(c * 8 + i + 2 * u) * 8 + j + 2 * v] *
                  w[((f * 2 + c) * 3 + u) * 3 + v];

  Tensor<4> xRef = view<4>(x, {1, 2, 8, 8});
  Tensor<4> wRef = view<4>(w, {3, 2, 3, 3});
  Ct enc, res;
  Tensor<4> out;
  _mlir_ciface_conv_dilated__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_conv_dilated(&res, &enc, &wRef);
  _mlir_ciface_conv_dilated__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelTest, ConvDepthwise) {
  std::mt19937 rng(8);
  std::vector<float> x = randomValues(rng, 4 * 8 * 8);
  std::vector<float> w = randomValues(rng, 4 * 3 * 3);
  // Stride 2, no padding: (8 - 3) / 2 + 1 = 3 outputs per axis.
  std::vector<float> expected(4 * 3 * 3, 0.0f);
  for (int64_t c = 0; c < 4; ++c)
    for (int64_t i = 0; i < 3; ++i)
      for (int64_t j = 0; j < 3; ++j)
        for (int64_t u = 0; u < 3; ++u)
          for (int64_t v = 0; v < 3; ++v)
            expected[(c * 3 + i) * 3 + j] +=
                x[(c * 8 + 2 * i + u) * 8 + 2 * j + v] * w[(c * 3 + u) * 3 + v];

  Tensor<4> xRef = view<4>(x, {1, 4, 8, 8});
  Tensor<3> wRef = view<3>(w, {4, 3, 3});
  Ct enc, res;
  Tensor<4> out;
  _mlir_ciface_conv_depthwise__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_conv_depthwise(&res, &enc, &wRef);
  _mlir_ciface_conv_depthwise__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelTest, Matvec) {
  std::mt19937 rng(3);
  std::vector<float> x = randomValues(rng, 16);
  std::vector<float> m = randomValues(rng, 10 * 16);
  std::vector<float> expected(10, 0.0f);
  for (int64_t i = 0; i < 10; ++i)
    for (int64_t j = 0; j < 16; ++j) expected[i] += m[i * 16 + j] * x[j];

  Tensor<1> xRef = view<1>(x, {16});
  Tensor<2> mRef = view<2>(m, {10, 16});
  Ct enc, res;
  Tensor<1> out;
  _mlir_ciface_matvec__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_matvec(&res, &enc, &mRef);
  _mlir_ciface_matvec__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelTest, SumPool) {
  std::mt19937 rng(4);
  std::vector<float> x = randomValues(rng, 2 * 8 * 8);
  std::vector<float> expected(2 * 4 * 4, 0.0f);
  for (int64_t c = 0; c < 2; ++c)
    for (int64_t i = 0; i < 4; ++i)
      for (int64_t j = 0; j < 4; ++j)
        for (int64_t u = 0; u < 2; ++u)
          for (int64_t v = 0; v < 2; ++v)
            expected[(c * 4 + i) * 4 + j] +=
                x[(c * 8 + 2 * i + u) * 8 + 2 * j + v];

  Tensor<4> xRef = view<4>(x, {1, 2, 8, 8});
  Ct enc, res;
  Tensor<4> out;
  _mlir_ciface_sum_pool__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_sum_pool(&res, &enc);
  _mlir_ciface_sum_pool__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelTest, ConvGrouped) {
  std::mt19937 rng(9);
  std::vector<float> x = randomValues(rng, 4 * 6 * 6);
  std::vector<float> w = randomValues(rng, 2 * 3 * 2 * 3 * 3);
  // Two groups of two channels, three filters each, padding 1.
  std::vector<float> expected(6 * 6 * 6, 0.0f);
  for (int64_t g = 0; g < 2; ++g)
    for (int64_t f = 0; f < 3; ++f)
      for (int64_t i = 0; i < 6; ++i)
        for (int64_t j = 0; j < 6; ++j)
          for (int64_t c = 0; c < 2; ++c)
            for (int64_t u = 0; u < 3; ++u)
              for (int64_t v = 0; v < 3; ++v) {
                int64_t r = i + u - 1;
                int64_t t = j + v - 1;
                if (r < 0 || r >= 6 || t < 0 || t >= 6) continue;
                expected[((g * 3 + f) * 6 + i) * 6 + j] +=
                    x[((g * 2 + c) * 6 + r) * 6 + t] *
                    w[(((g * 3 + f) * 2 + c) * 3 + u) * 3 + v];
              }

  Tensor<4> xRef = view<4>(x, {1, 4, 6, 6});
  Tensor<5> wRef = view<5>(w, {2, 3, 2, 3, 3});
  Ct enc, res;
  Tensor<4> out;
  _mlir_ciface_conv_grouped__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_conv_grouped(&res, &enc, &wRef);
  _mlir_ciface_conv_grouped__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

}  // namespace
