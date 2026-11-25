#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/EvalVisitor.h"
#include "lib/Kernel/LinearTransformImplementation.h"
#include "lib/Kernel/RotationCountVisitor.h"
#include "lib/Utils/Layout/LinearSpec.h"
#include "lib/Utils/Layout/SparseBsgs.h"

namespace mlir {
namespace heir {
namespace kernel {
namespace {

using Matrix = std::vector<std::vector<int>>;

std::vector<int> rotateLeft(const std::vector<int>& x, int64_t shift) {
  int64_t n = x.size();
  std::vector<int> result(n);
  for (int64_t i = 0; i < n; ++i) result[i] = x[((i + shift) % n + n) % n];
  return result;
}

Matrix randomMatrix(std::mt19937& rng, int64_t rows, int64_t cols) {
  std::uniform_int_distribution<int> dist(-9, 9);
  Matrix m(rows, std::vector<int>(cols));
  for (auto& row : m) {
    for (int& v : row) v = dist(rng);
  }
  return m;
}

std::vector<DiagonalKey> randomKeys(std::mt19937& rng, int64_t n, int64_t numIn,
                                    int64_t numOut) {
  std::set<DiagonalKey> keySet;
  int64_t count = 1 + rng() % (2 * n);
  for (int64_t i = 0; i < count; ++i) {
    keySet.insert({static_cast<int64_t>(rng() % numOut),
                   static_cast<int64_t>(rng() % numIn),
                   static_cast<int64_t>(rng() % n)});
  }
  return {keySet.begin(), keySet.end()};
}

Matrix naive(const std::vector<DiagonalKey>& keys, const Matrix& x,
             const Matrix& diagonals, int64_t numOut) {
  int64_t n = x[0].size();
  Matrix y(numOut, std::vector<int>(n, 0));
  for (size_t k = 0; k < keys.size(); ++k) {
    std::vector<int> rotated = rotateLeft(x[keys[k].ctIn], keys[k].shift);
    for (int64_t i = 0; i < n; ++i) {
      y[keys[k].ctOut][i] += diagonals[k][i] * rotated[i];
    }
  }
  return y;
}

Matrix run(const SparseBsgsPlan& plan, const Matrix& x, const Matrix& diagonals,
           int64_t numOut, const SparseBsgsOptions& options) {
  int64_t n = x[0].size();
  auto dag = implementSparseBsgs(plan, LiteralValue(x), LiteralValue(diagonals),
                                 numOut, DagType::intTensor(32, {n}), options);
  return std::get<Matrix>(evalKernel(dag)[0].get());
}

TEST(LinearTransformImplementationTest, RandomBlockTransforms) {
  std::mt19937 rng(11);
  for (int trial = 0; trial < 96; ++trial) {
    int64_t n = int64_t{8} << (trial % 3);
    int64_t numIn = 1 + trial % 3;
    int64_t numOut = 1 + (trial / 3) % 3;
    std::vector<DiagonalKey> keys = randomKeys(rng, n, numIn, numOut);
    Matrix x = randomMatrix(rng, numIn, n);
    Matrix diagonals = randomMatrix(rng, keys.size(), n);
    Matrix expected = naive(keys, x, diagonals, numOut);
    SparseBsgsPlan plan = planSparseBsgs(keys, n);

    // The packing helper's form of the diagonals: each rotated right by the
    // giant part of its shift.
    Matrix preRotated = diagonals;
    for (size_t r = 0; r < plan.row.size(); ++r) {
      preRotated[plan.row[r]] =
          rotateLeft(diagonals[plan.row[r]], -plan.giant[r]);
    }

    for (bool unroll : {false, true}) {
      SparseBsgsOptions options;
      options.unroll = unroll;
      EXPECT_EQ(run(plan, x, diagonals, numOut, options), expected)
          << "trial " << trial << " unroll " << unroll;
      options.diagonalsPreRotated = true;
      EXPECT_EQ(run(plan, x, preRotated, numOut, options), expected)
          << "trial " << trial << " unroll " << unroll << " pre-rotated";
    }
  }
}

TEST(LinearTransformImplementationTest, UnwrittenOutputRowsAreZero) {
  int64_t n = 8;
  std::vector<DiagonalKey> keys = {{2, 0, 3}, {2, 1, 5}};
  std::mt19937 rng(3);
  Matrix x = randomMatrix(rng, 2, n);
  Matrix diagonals = randomMatrix(rng, keys.size(), n);
  Matrix expected = naive(keys, x, diagonals, 3);
  SparseBsgsPlan plan = planSparseBsgs(keys, n);
  for (bool unroll : {false, true}) {
    SparseBsgsOptions options;
    options.unroll = unroll;
    EXPECT_EQ(run(plan, x, diagonals, 3, options), expected);
  }
}

TEST(LinearTransformImplementationTest, UnrolledRotationCountMatchesPlan) {
  // The 144 diagonals of a 16 -> 16 channel 3x3 convolution on 32x32 images.
  int64_t n = 16384;
  std::vector<DiagonalKey> keys;
  for (int64_t a = 0; a < 16; ++a) {
    for (int64_t u = -1; u <= 1; ++u) {
      for (int64_t v = -1; v <= 1; ++v) {
        keys.push_back({0, 0, ((1024 * a + 32 * u + v) % n + n) % n});
      }
    }
  }
  SparseBsgsPlan plan = planSparseBsgs(keys, n);
  SymbolicValue x({1, n}, /*isSecret=*/true);
  SymbolicValue diagonals({static_cast<int64_t>(keys.size()), n},
                          /*isSecret=*/false);
  SparseBsgsOptions options;
  options.unroll = true;
  options.diagonalsPreRotated = true;
  auto dag = implementSparseBsgs(plan, x, diagonals, 1,
                                 DagType::intTensor(32, {n}), options);
  RotationCountVisitor counter;
  EXPECT_EQ(counter.process(dag), plan.numRotations());
  EXPECT_EQ(plan.numRotations(), 23);
}

TEST(LinearTransformImplementationTest, UniformBoxSum) {
  // A 3x2 window sum on an 8x8 grid: strides 8 and 1.
  int64_t n = 64;
  std::mt19937 rng(5);
  std::vector<int> x = randomMatrix(rng, 1, n)[0];
  std::vector<int> expected(n, 0);
  for (int64_t u = 0; u < 3; ++u) {
    for (int64_t v = 0; v < 2; ++v) {
      std::vector<int> rotated = rotateLeft(x, 8 * u + v);
      for (int64_t i = 0; i < n; ++i) expected[i] += rotated[i];
    }
  }
  auto dag = implementUniformBoxSum(LiteralValue(x), {{8, 3}, {1, 2}});
  EXPECT_EQ(std::get<std::vector<int>>(evalKernel(dag)[0].get()), expected);
}

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
