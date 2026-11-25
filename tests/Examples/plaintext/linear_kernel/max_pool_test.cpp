#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "tests/llvm_runner/memref_types.h"

template <std::size_t N>
using Tensor = StridedMemRefType<float, N>;

extern "C" {
void _mlir_ciface_max_pool(Tensor<4>* res, Tensor<4>* x);
}

namespace {

// The decomposition is run without the homomorphic pipeline, so its maxima
// are exact. The input is negative everywhere, which a zero padded max pool
// would get wrong at the border.
TEST(DecomposeMaxPoolTest, MatchesReference) {
  std::mt19937 rng(21);
  std::uniform_real_distribution<float> dist(-2.0, -1.0);
  std::vector<float> x(2 * 8 * 8);
  for (float& v : x) v = dist(rng);

  std::vector<float> expected;
  for (int64_t c = 0; c < 2; ++c)
    for (int64_t i = 0; i < 4; ++i)
      for (int64_t j = 0; j < 4; ++j) {
        float best = -std::numeric_limits<float>::infinity();
        for (int64_t u = 0; u < 3; ++u)
          for (int64_t v = 0; v < 3; ++v) {
            int64_t r = 2 * i + u - 1;
            int64_t s = 2 * j + v - 1;
            if (r < 0 || r >= 8 || s < 0 || s >= 8) continue;
            best = std::max(best, x[(c * 8 + r) * 8 + s]);
          }
        expected.push_back(best);
      }

  int64_t shape[4] = {1, 2, 8, 8};
  int64_t strides[4] = {128, 64, 8, 1};
  Tensor<4> xRef(x.data(), x.data(), 0, shape, strides);
  Tensor<4> out;
  _mlir_ciface_max_pool(&out, &xRef);
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(out.data[out.offset + i], expected[i], 1e-5)
        << "at element " << i;
  }
}

}  // namespace
