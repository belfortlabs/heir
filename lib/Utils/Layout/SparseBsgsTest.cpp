#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/LinearSpec.h"
#include "lib/Utils/Layout/SparseBsgs.h"

namespace mlir {
namespace heir {
namespace {

using Ct = std::vector<double>;

Ct rotateLeft(const Ct& x, int64_t shift) {
  int64_t n = x.size();
  Ct result(n);
  for (int64_t i = 0; i < n; ++i) result[i] = x[(i + shift) % n];
  return result;
}

// Evaluates the plan exactly as the emitted loop does.
std::vector<Ct> runPlan(const SparseBsgsPlan& plan,
                        const std::vector<Ct>& diagonals,
                        const std::vector<Ct>& x, int64_t numOut) {
  int64_t n = x[0].size();
  std::vector<Ct> baby;
  for (size_t t = 0; t < plan.babyCt.size(); ++t) {
    baby.push_back(rotateLeft(x[plan.babyCt[t]], plan.babyShift[t]));
  }
  std::vector<Ct> y(numOut, Ct(n, 0.0));
  Ct part(n, 0.0);
  for (size_t r = 0; r < plan.row.size(); ++r) {
    // The packing helper stores the diagonal rotated right by its giant part.
    Ct stored = rotateLeft(diagonals[plan.row[r]], n - plan.giant[r] % n);
    for (int64_t i = 0; i < n; ++i) {
      part[i] += stored[i] * baby[plan.babyOf[r]][i];
    }
    if (plan.isLast[r]) {
      Ct rotated = rotateLeft(part, plan.giant[r]);
      for (int64_t i = 0; i < n; ++i) y[plan.out[r]][i] += rotated[i];
      part.assign(n, 0.0);
    }
  }
  return y;
}

TEST(SparseBsgsTest, RandomBlockTransforms) {
  std::mt19937 rng(7);
  std::normal_distribution<double> dist;
  for (int trial = 0; trial < 200; ++trial) {
    int64_t n = int64_t{8} << (trial % 4);
    int64_t numIn = 1 + trial % 3;
    int64_t numOut = 1 + (trial / 3) % 3;
    std::set<DiagonalKey> keySet;
    int64_t count = 1 + rng() % (2 * n);
    for (int64_t i = 0; i < count; ++i) {
      keySet.insert({static_cast<int64_t>(rng() % numOut),
                     static_cast<int64_t>(rng() % numIn),
                     static_cast<int64_t>(rng() % n)});
    }
    std::vector<DiagonalKey> keys(keySet.begin(), keySet.end());
    std::vector<Ct> diagonals(keys.size(), Ct(n));
    for (Ct& d : diagonals) {
      for (double& v : d) v = dist(rng);
    }
    std::vector<Ct> x(numIn, Ct(n));
    for (Ct& ct : x) {
      for (double& v : ct) v = dist(rng);
    }

    std::vector<Ct> expected(numOut, Ct(n, 0.0));
    for (size_t k = 0; k < keys.size(); ++k) {
      Ct rotated = rotateLeft(x[keys[k].ctIn], keys[k].shift);
      for (int64_t i = 0; i < n; ++i) {
        expected[keys[k].ctOut][i] += diagonals[k][i] * rotated[i];
      }
    }

    SparseBsgsPlan plan = planSparseBsgs(keys, n);
    std::vector<Ct> actual = runPlan(plan, diagonals, x, numOut);
    double maxError = 0.0;
    for (int64_t o = 0; o < numOut; ++o) {
      for (int64_t i = 0; i < n; ++i) {
        maxError = std::max(maxError, std::abs(actual[o][i] - expected[o][i]));
      }
    }
    EXPECT_LT(maxError, 1e-9);

    // Never worse than one rotation per nonzero (input, shift) pair.
    std::set<std::pair<int64_t, int64_t>> naive;
    for (const DiagonalKey& key : keys) {
      if (key.shift != 0) naive.insert({key.ctIn, key.shift});
    }
    EXPECT_LE(plan.numRotations(), static_cast<int64_t>(naive.size()));
  }
}

TEST(SparseBsgsTest, ConvDiagonalsNeedFewRotations) {
  // The 144 diagonals of a 16 -> 16 channel 3x3 convolution on 32x32 images
  // in one 16384-slot ciphertext: shifts 1024 a + 32 u + v.
  std::vector<DiagonalKey> keys;
  for (int64_t a = 0; a < 16; ++a) {
    for (int64_t u = -1; u <= 1; ++u) {
      for (int64_t v = -1; v <= 1; ++v) {
        keys.push_back(
            {0, 0, ((1024 * a + 32 * u + v) % 16384 + 16384) % 16384});
      }
    }
  }
  SparseBsgsPlan plan = planSparseBsgs(keys, 16384);
  EXPECT_EQ(plan.numRotations(), 23);
}

}  // namespace
}  // namespace heir
}  // namespace mlir
