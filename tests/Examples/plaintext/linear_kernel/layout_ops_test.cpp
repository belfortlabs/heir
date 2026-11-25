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
void _mlir_ciface_transpose_matvec(Ct* res, Ct* x, Tensor<2>* m);
void _mlir_ciface_transpose_matvec__encrypt__arg0(Ct* res, Tensor<3>* arg);
void _mlir_ciface_transpose_matvec__decrypt__result0(Tensor<1>* res, Ct* arg);

void _mlir_ciface_slice_rows(Ct* res, Ct* x);
void _mlir_ciface_slice_rows__encrypt__arg0(Ct* res, Tensor<3>* arg);
void _mlir_ciface_slice_rows__decrypt__result0(Tensor<2>* res, Ct* arg);

void _mlir_ciface_broadcast_add(Ct* res, Ct* x, Ct* y);
void _mlir_ciface_broadcast_add__encrypt__arg0(Ct* res, Tensor<1>* arg);
void _mlir_ciface_broadcast_add__encrypt__arg1(Ct* res, Tensor<2>* arg);
void _mlir_ciface_broadcast_add__decrypt__result0(Tensor<2>* res, Ct* arg);

void _mlir_ciface_concat(Ct* res, Ct* x, Ct* y);
void _mlir_ciface_concat__encrypt__arg0(Ct* res, Tensor<4>* arg);
void _mlir_ciface_concat__encrypt__arg1(Ct* res, Tensor<4>* arg);
void _mlir_ciface_concat__decrypt__result0(Tensor<4>* res, Ct* arg);

void _mlir_ciface_global_sum(Ct* res, Ct* x);
void _mlir_ciface_global_sum__encrypt__arg0(Ct* res, Tensor<4>* arg);
void _mlir_ciface_global_sum__decrypt__result0(Tensor<2>* res, Ct* arg);

void _mlir_ciface_sum_pool_multi(Ct* res, Ct* x);
void _mlir_ciface_sum_pool_multi__encrypt__arg0(Ct* res, Tensor<4>* arg);
void _mlir_ciface_sum_pool_multi__decrypt__result0(Tensor<4>* res, Ct* arg);

void _mlir_ciface_channel_scale(Ct* res, Ct* x);
void _mlir_ciface_channel_scale__encrypt__arg0(Ct* res, Tensor<4>* arg);
void _mlir_ciface_channel_scale__decrypt__result0(Tensor<4>* res, Ct* arg);

void _mlir_ciface_avg_pool_conv(Ct* res, Ct* x, Tensor<4>* w);
void _mlir_ciface_avg_pool_conv__encrypt__arg0(Ct* res, Tensor<4>* arg);
void _mlir_ciface_avg_pool_conv__decrypt__result0(Tensor<4>* res, Ct* arg);

void _mlir_ciface_periodic_chain(Ct* res, Ct* x, Tensor<2>* m1, Tensor<2>* m2);
void _mlir_ciface_periodic_chain__encrypt__arg0(Ct* res, Tensor<1>* arg);
void _mlir_ciface_periodic_chain__decrypt__result0(Tensor<3>* res, Ct* arg);

void _mlir_ciface_conv_clear_init(Ct* res, Ct* x, Tensor<1>* mean,
                                  Tensor<1>* scale, Tensor<4>* w1,
                                  Tensor<4>* w2);
void _mlir_ciface_conv_clear_init__encrypt__arg0(Ct* res, Tensor<4>* arg);
void _mlir_ciface_conv_clear_init__decrypt__result0(Tensor<4>* res, Ct* arg);
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

TEST(LinearKernelLayoutTest, TransposeMatvec) {
  std::mt19937 rng(11);
  std::vector<float> x = randomValues(rng, 2 * 3 * 4);
  std::vector<float> m = randomValues(rng, 5 * 6);
  // flat[(k * 2 + i) * 3 + j] = x[i][j][k], and the slice keeps flat[4 * s].
  std::vector<float> flat(24);
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j)
      for (int64_t k = 0; k < 4; ++k)
        flat[(k * 2 + i) * 3 + j] = x[(i * 3 + j) * 4 + k];
  std::vector<float> expected(5, 0.0f);
  for (int64_t r = 0; r < 5; ++r)
    for (int64_t s = 0; s < 6; ++s) expected[r] += m[r * 6 + s] * flat[4 * s];

  Tensor<3> xRef = view<3>(x, {2, 3, 4});
  Tensor<2> mRef = view<2>(m, {5, 6});
  Ct enc, res;
  Tensor<1> out;
  _mlir_ciface_transpose_matvec__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_transpose_matvec(&res, &enc, &mRef);
  _mlir_ciface_transpose_matvec__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelLayoutTest, SliceRows) {
  std::mt19937 rng(12);
  std::vector<float> x = randomValues(rng, 8 * 8);
  std::vector<float> expected;
  for (int64_t i = 0; i < 4; ++i)
    for (int64_t j = 0; j < 8; ++j) expected.push_back(x[(1 + 2 * i) * 8 + j]);

  Tensor<3> xRef = view<3>(x, {1, 8, 8});
  Ct enc, res;
  Tensor<2> out;
  _mlir_ciface_slice_rows__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_slice_rows(&res, &enc);
  _mlir_ciface_slice_rows__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelLayoutTest, BroadcastAdd) {
  std::mt19937 rng(13);
  std::vector<float> x = randomValues(rng, 4);
  std::vector<float> y = randomValues(rng, 4 * 8);
  std::vector<float> expected;
  for (int64_t i = 0; i < 4; ++i)
    for (int64_t j = 0; j < 8; ++j) expected.push_back(x[i] + y[i * 8 + j]);

  Tensor<1> xRef = view<1>(x, {4});
  Tensor<2> yRef = view<2>(y, {4, 8});
  Ct encX, encY, res;
  Tensor<2> out;
  _mlir_ciface_broadcast_add__encrypt__arg0(&encX, &xRef);
  _mlir_ciface_broadcast_add__encrypt__arg1(&encY, &yRef);
  _mlir_ciface_broadcast_add(&res, &encX, &encY);
  _mlir_ciface_broadcast_add__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelLayoutTest, Concat) {
  std::mt19937 rng(14);
  std::vector<float> x = randomValues(rng, 2 * 4 * 4);
  std::vector<float> y = randomValues(rng, 3 * 4 * 4);
  std::vector<float> expected = x;
  expected.insert(expected.end(), y.begin(), y.end());

  Tensor<4> xRef = view<4>(x, {1, 2, 4, 4});
  Tensor<4> yRef = view<4>(y, {1, 3, 4, 4});
  Ct encX, encY, res;
  Tensor<4> out;
  _mlir_ciface_concat__encrypt__arg0(&encX, &xRef);
  _mlir_ciface_concat__encrypt__arg1(&encY, &yRef);
  _mlir_ciface_concat(&res, &encX, &encY);
  _mlir_ciface_concat__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

using BatchNormFn = void (*)(Ct*, Ct*, Tensor<1>*, Tensor<1>*, Tensor<4>*,
                             Tensor<4>*);
using EncryptFn = void (*)(Ct*, Tensor<4>*);
using DecryptFn = void (*)(Tensor<4>*, Ct*);

// A 2x2 convolution with stride 2, a normalization with cleartext mean and
// scale, and a 3x3 convolution with zero padding.
void checkBatchNorm(BatchNormFn run, EncryptFn encrypt, DecryptFn decrypt) {
  std::mt19937 rng(15);
  std::vector<float> x = randomValues(rng, 2 * 8 * 8);
  std::vector<float> mean = randomValues(rng, 3);
  std::vector<float> scale = randomValues(rng, 3);
  std::vector<float> w1 = randomValues(rng, 3 * 2 * 2 * 2);
  std::vector<float> w2 = randomValues(rng, 2 * 3 * 3 * 3);
  // The 2x2 convolution with stride 2, then the normalization.
  std::vector<float> mid(3 * 4 * 4, 0.0f);
  for (int64_t f = 0; f < 3; ++f)
    for (int64_t h = 0; h < 4; ++h)
      for (int64_t w = 0; w < 4; ++w) {
        float sum = 0;
        for (int64_t c = 0; c < 2; ++c)
          for (int64_t u = 0; u < 2; ++u)
            for (int64_t v = 0; v < 2; ++v)
              sum += w1[((f * 2 + c) * 2 + u) * 2 + v] *
                     x[(c * 8 + 2 * h + u) * 8 + 2 * w + v];
        mid[(f * 4 + h) * 4 + w] = (sum - mean[f]) * scale[f];
      }
  // The 3x3 convolution with one row and column of zero padding around.
  std::vector<float> expected(2 * 4 * 4, 0.0f);
  for (int64_t f = 0; f < 2; ++f)
    for (int64_t h = 0; h < 4; ++h)
      for (int64_t w = 0; w < 4; ++w)
        for (int64_t c = 0; c < 3; ++c)
          for (int64_t u = 0; u < 3; ++u)
            for (int64_t v = 0; v < 3; ++v) {
              int64_t row = h + u - 1, col = w + v - 1;
              if (row < 0 || row >= 4 || col < 0 || col >= 4) continue;
              expected[(f * 4 + h) * 4 + w] +=
                  w2[((f * 3 + c) * 3 + u) * 3 + v] *
                  mid[(c * 4 + row) * 4 + col];
            }

  Tensor<4> xRef = view<4>(x, {1, 2, 8, 8});
  Tensor<1> meanRef = view<1>(mean, {3});
  Tensor<1> scaleRef = view<1>(scale, {3});
  Tensor<4> w1Ref = view<4>(w1, {3, 2, 2, 2});
  Tensor<4> w2Ref = view<4>(w2, {2, 3, 3, 3});
  Ct enc, res;
  Tensor<4> out;
  encrypt(&enc, &xRef);
  run(&res, &enc, &meanRef, &scaleRef, &w1Ref, &w2Ref);
  decrypt(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelLayoutTest, ConvClearInit) {
  checkBatchNorm(_mlir_ciface_conv_clear_init,
                 _mlir_ciface_conv_clear_init__encrypt__arg0,
                 _mlir_ciface_conv_clear_init__decrypt__result0);
}

TEST(LinearKernelLayoutTest, GlobalSum) {
  std::mt19937 rng(16);
  std::vector<float> x = randomValues(rng, 20 * 8 * 8);
  std::vector<float> expected(20, 0.0f);
  for (int64_t c = 0; c < 20; ++c)
    for (int64_t i = 0; i < 64; ++i) expected[c] += x[c * 64 + i];

  Tensor<4> xRef = view<4>(x, {1, 20, 8, 8});
  Ct enc, res;
  Tensor<2> out;
  _mlir_ciface_global_sum__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_global_sum(&res, &enc);
  _mlir_ciface_global_sum__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelLayoutTest, SumPoolMulti) {
  std::mt19937 rng(17);
  std::vector<float> x = randomValues(rng, 30 * 7 * 6);
  std::vector<float> expected;
  for (int64_t c = 0; c < 30; ++c)
    for (int64_t h = 0; h < 3; ++h)
      for (int64_t w = 0; w < 3; ++w) {
        float sum = 0;
        for (int64_t u = 0; u < 3; ++u)
          for (int64_t v = 0; v < 2; ++v)
            sum += x[(c * 7 + 2 * h + u) * 6 + 2 * w + v];
        expected.push_back(sum);
      }

  Tensor<4> xRef = view<4>(x, {1, 30, 7, 6});
  Ct enc, res;
  Tensor<4> out;
  _mlir_ciface_sum_pool_multi__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_sum_pool_multi(&res, &enc);
  _mlir_ciface_sum_pool_multi__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelLayoutTest, ChannelScale) {
  std::mt19937 rng(18);
  std::vector<float> x = randomValues(rng, 20 * 8 * 8);
  std::vector<float> expected(x.size());
  for (int64_t c = 0; c < 20; ++c) {
    float sum = 0;
    for (int64_t i = 0; i < 64; ++i) sum += x[c * 64 + i];
    for (int64_t i = 0; i < 64; ++i) expected[c * 64 + i] = x[c * 64 + i] * sum;
  }

  Tensor<4> xRef = view<4>(x, {1, 20, 8, 8});
  Ct enc, res;
  Tensor<4> out;
  _mlir_ciface_channel_scale__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_channel_scale(&res, &enc);
  _mlir_ciface_channel_scale__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelLayoutTest, AvgPoolConv) {
  std::mt19937 rng(19);
  std::vector<float> x = randomValues(rng, 2 * 8 * 8);
  std::vector<float> w = randomValues(rng, 3 * 2);
  std::vector<float> avg(2 * 4 * 4, 0.0f);
  for (int64_t c = 0; c < 2; ++c)
    for (int64_t h = 0; h < 4; ++h)
      for (int64_t v = 0; v < 4; ++v)
        for (int64_t a = 0; a < 2; ++a)
          for (int64_t b = 0; b < 2; ++b)
            avg[(c * 4 + h) * 4 + v] +=
                x[(c * 8 + 2 * h + a) * 8 + 2 * v + b] / 4.0f;
  std::vector<float> expected(3 * 4 * 4, 0.0f);
  for (int64_t f = 0; f < 3; ++f)
    for (int64_t c = 0; c < 2; ++c)
      for (int64_t i = 0; i < 16; ++i)
        expected[f * 16 + i] += w[f * 2 + c] * avg[c * 16 + i];

  Tensor<4> xRef = view<4>(x, {1, 2, 8, 8});
  Tensor<4> wRef = view<4>(w, {3, 2, 1, 1});
  Ct enc, res;
  Tensor<4> out;
  _mlir_ciface_avg_pool_conv__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_avg_pool_conv(&res, &enc, &wRef);
  _mlir_ciface_avg_pool_conv__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

TEST(LinearKernelLayoutTest, PeriodicChain) {
  std::mt19937 rng(20);
  std::vector<float> x = randomValues(rng, 16);
  std::vector<float> m1 = randomValues(rng, 8 * 16);
  std::vector<float> m2 = randomValues(rng, 16 * 8);
  std::vector<float> a(8, 0.0f), b(16, 0.0f);
  for (int64_t r = 0; r < 8; ++r)
    for (int64_t c = 0; c < 16; ++c) a[r] += m1[r * 16 + c] * x[c];
  for (int64_t r = 0; r < 16; ++r)
    for (int64_t c = 0; c < 8; ++c) b[r] += m2[r * 8 + c] * a[c];
  std::vector<float> expected;
  for (int64_t r = 0; r < 16; ++r)
    for (int64_t i = 0; i < 16; ++i) expected.push_back(b[r]);

  Tensor<1> xRef = view<1>(x, {16});
  Tensor<2> m1Ref = view<2>(m1, {8, 16});
  Tensor<2> m2Ref = view<2>(m2, {16, 8});
  Ct enc, res;
  Tensor<3> out;
  _mlir_ciface_periodic_chain__encrypt__arg0(&enc, &xRef);
  _mlir_ciface_periodic_chain(&res, &enc, &m1Ref, &m2Ref);
  _mlir_ciface_periodic_chain__decrypt__result0(&out, &res);
  expectNear(out, expected);
}

}  // namespace
