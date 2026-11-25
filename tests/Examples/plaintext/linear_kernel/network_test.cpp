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
void _mlir_ciface_resblock(Ct* res, Ct* x, Tensor<4>* w1, Tensor<4>* w2,
                           Tensor<4>* ws, Tensor<2>* m);
void _mlir_ciface_resblock__encrypt__arg0(Ct* res, Tensor<4>* arg);
void _mlir_ciface_resblock__decrypt__result0(Tensor<1>* res, Ct* arg);

void _mlir_ciface_join(Ct* res, Ct* x, Ct* y, Tensor<4>* w);
void _mlir_ciface_join__encrypt__arg0(Ct* res, Tensor<4>* arg);
void _mlir_ciface_join__encrypt__arg1(Ct* res, Tensor<4>* arg);
void _mlir_ciface_join__decrypt__result0(Tensor<4>* res, Ct* arg);
}

namespace {

std::vector<float> randomValues(std::mt19937& rng, int64_t size) {
  std::uniform_real_distribution<float> dist(-1.0, 1.0);
  std::vector<float> values(size);
  for (float& v : values) v = dist(rng);
  return values;
}

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

TEST(LinearKernelNetworkTest, ResidualBlock) {
  std::mt19937 rng(5);
  std::vector<float> x = randomValues(rng, 2 * 8 * 8);
  std::vector<float> w1 = randomValues(rng, 4 * 2 * 3 * 3);
  std::vector<float> w2 = randomValues(rng, 4 * 4 * 3 * 3);
  std::vector<float> ws = randomValues(rng, 4 * 4);
  std::vector<float> m = randomValues(rng, 10 * 64);

  std::vector<float> a1 = conv2d(x, w1, 2, 4, 8, 3, 1, 1);
  for (float& v : a1) v = v * v;
  std::vector<float> h2 = conv2d(a1, w2, 4, 4, 8, 3, 2, 1);
  std::vector<float> shortcut = conv2d(a1, ws, 4, 4, 8, 1, 2, 0);
  std::vector<float> expected(10, 0.0f);
  for (int64_t i = 0; i < 10; ++i)
    for (int64_t j = 0; j < 64; ++j)
      expected[i] += m[i * 64 + j] * 0.5f * (h2[j] + shortcut[j]);

  Tensor<4> xRef = view<4>(x, {1, 2, 8, 8});
  Tensor<4> w1Ref = view<4>(w1, {4, 2, 3, 3});
  Tensor<4> w2Ref = view<4>(w2, {4, 4, 3, 3});
  Tensor<4> wsRef = view<4>(ws, {4, 4, 1, 1});
  Tensor<2> mRef = view<2>(m, {10, 64});
  Ct enc, res;
  Tensor<1> out;
  _mlir_ciface_resblock__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_resblock(&res, &enc, &w1Ref, &w2Ref, &wsRef, &mRef);
  _mlir_ciface_resblock__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelNetworkTest, JoinOfDifferentLayouts) {
  std::mt19937 rng(6);
  std::vector<float> x = randomValues(rng, 2 * 8 * 8);
  std::vector<float> y = randomValues(rng, 4 * 4 * 4);
  std::vector<float> w = randomValues(rng, 4 * 2 * 2 * 2);
  std::vector<float> expected = conv2d(x, w, 2, 4, 8, 2, 2, 0);
  for (size_t i = 0; i < expected.size(); ++i) expected[i] += y[i];

  Tensor<4> xRef = view<4>(x, {1, 2, 8, 8});
  Tensor<4> yRef = view<4>(y, {1, 4, 4, 4});
  Tensor<4> wRef = view<4>(w, {4, 2, 2, 2});
  Ct encX, encY, res;
  Tensor<4> out;
  _mlir_ciface_join__encrypt__arg0(&encX, &xRef);
  _mlir_ciface_join__encrypt__arg1(&encY, &yRef);
  _mlir_ciface_join(&res, &encX, &encY, &wRef);
  _mlir_ciface_join__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

}  // namespace
