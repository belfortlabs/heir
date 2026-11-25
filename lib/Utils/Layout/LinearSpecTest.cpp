#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/LinearSpec.h"
#include "lib/Utils/Layout/StridedLayout.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using llvm::ArrayRef;
using Coeff = std::function<double(ArrayRef<int64_t>)>;

struct Op {
  LinearSpec spec;
  Coeff coeff;
};

template <typename Fn>
void forEachPoint(const std::vector<int64_t>& extents, Fn fn) {
  for (int64_t extent : extents) {
    if (extent <= 0) return;
  }
  std::vector<int64_t> d(extents.size(), 0);
  while (true) {
    fn(d);
    int64_t k = static_cast<int64_t>(extents.size()) - 1;
    while (k >= 0 && ++d[k] == extents[k]) d[k--] = 0;
    if (k < 0) return;
  }
}

int64_t flatten(const std::vector<int64_t>& idx,
                const std::vector<int64_t>& shape) {
  int64_t flat = 0;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    flat = flat * shape[axis] + idx[axis];
  }
  return flat;
}

std::vector<int64_t> applyMap(const std::vector<std::vector<int64_t>>& map,
                              const std::vector<int64_t>& offset,
                              const std::vector<int64_t>& d) {
  std::vector<int64_t> idx(map.size());
  for (size_t axis = 0; axis < map.size(); ++axis) {
    idx[axis] = offset[axis];
    for (size_t loop = 0; loop < d.size(); ++loop) {
      idx[axis] += map[axis][loop] * d[loop];
    }
  }
  return idx;
}

// The op evaluated directly on row-major data.
std::vector<double> naive(const Op& op, const std::vector<double>& x) {
  int64_t size = 1;
  for (int64_t extent : op.spec.outShape) size *= extent;
  std::vector<double> y(size, 0.0);
  forEachPoint(op.spec.loops, [&](const std::vector<int64_t>& d) {
    std::vector<int64_t> in = applyMap(op.spec.inMap, op.spec.inOffset, d);
    for (size_t axis = 0; axis < in.size(); ++axis) {
      if (in[axis] < 0 || in[axis] >= op.spec.inShape[axis]) return;
    }
    std::vector<int64_t> out = applyMap(op.spec.outMap, op.spec.outOffset, d);
    y[flatten(out, op.spec.outShape)] +=
        op.coeff(d) * x[flatten(in, op.spec.inShape)];
  });
  return y;
}

class Harness {
 public:
  explicit Harness(unsigned seed) : rng_(seed) {}

  double random() { return dist_(rng_); }

  std::vector<double> randomWeights(int64_t count) {
    std::vector<double> w(count);
    for (double& v : w) v = random();
    return w;
  }

  // Packs x into ciphertexts. Slots that hold no element get large random
  // values, to check that the transform never reads them.
  std::vector<std::vector<double>> encode(const std::vector<double>& x,
                                          const StridedLayout& layout) {
    int64_t n = layout.numSlots;
    std::vector<std::vector<double>> cts(layout.numCiphertexts(),
                                         std::vector<double>(n));
    for (auto& ct : cts) {
      for (double& v : ct) v = 100.0 * random();
    }
    forEachPoint(layout.shape, [&](const std::vector<int64_t>& idx) {
      int64_t p = layout.position(idx);
      double value = x[flatten(idx, layout.shape)];
      if (layout.period == 0) {
        cts[p / n][p % n] = value;
        return;
      }
      for (int64_t m = 0; m < n / layout.period; ++m) {
        cts[0][(p + m * layout.period) % n] = value;
      }
    });
    return cts;
  }

  // Runs the op through keys, packed diagonals and rotations, and compares
  // with the direct evaluation. Returns the layout of the result.
  StridedLayout check(const std::string& name, const Op& op,
                      const StridedLayout& in, const StridedLayout& out) {
    SCOPED_TRACE(name);
    EXPECT_TRUE(out.isInjective());
    int64_t n = out.numSlots;
    int64_t inSize = 1;
    for (int64_t extent : op.spec.inShape) inSize *= extent;
    std::vector<double> x = randomWeights(inSize);
    std::vector<double> expected = naive(op, x);

    std::vector<DiagonalKey> keys = computeDiagonalKeys(op.spec, in, out);
    EXPECT_EQ(keys, computeDiagonalKeysByEnumeration(op.spec, in, out));
    std::vector<std::vector<double>> diagonals =
        packDiagonals(op.spec, in, out, keys, op.coeff);

    std::vector<std::vector<double>> input = encode(x, in);
    std::vector<std::vector<double>> result(out.numCiphertexts(),
                                            std::vector<double>(n, 0.0));
    for (size_t k = 0; k < keys.size(); ++k) {
      const DiagonalKey& key = keys[k];
      EXPECT_LT(key.ctOut, static_cast<int64_t>(result.size()));
      EXPECT_LT(key.ctIn, static_cast<int64_t>(input.size()));
      for (int64_t slot = 0; slot < n; ++slot) {
        result[key.ctOut][slot] +=
            diagonals[k][slot] * input[key.ctIn][(slot + key.shift) % n];
      }
    }
    if (out.period != 0) {
      std::vector<double> folded(n, 0.0);
      for (int64_t m = 0; m < n / out.period; ++m) {
        for (int64_t slot = 0; slot < n; ++slot) {
          folded[(slot + m * out.period) % n] += result[0][slot];
        }
      }
      result[0] = folded;
    }

    double maxError = 0.0;
    forEachPoint(out.shape, [&](const std::vector<int64_t>& idx) {
      int64_t p = out.position(idx);
      double want = expected[flatten(idx, out.shape)];
      int64_t copies = out.period == 0 ? 1 : n / out.period;
      for (int64_t m = 0; m < copies; ++m) {
        int64_t q = p + m * out.period;
        maxError = std::max(maxError, std::abs(result[q / n][q % n] - want));
      }
    });
    EXPECT_LT(maxError, 1e-9);
    lastKeyCount = keys.size();
    std::set<int64_t> shifts;
    for (const DiagonalKey& key : keys) shifts.insert(key.shift);
    lastShiftCount = shifts.size();
    return out;
  }

  StridedLayout check(const std::string& name, const Op& op,
                      const StridedLayout& in) {
    return check(name, op, in, inducedOutputLayout(op.spec, in));
  }

  // Runs a unit-coefficient op as rotations and additions of the input
  // followed by the rest of the op, and compares with the direct evaluation.
  // Returns false if the op has no such form in this layout.
  bool checkBoxSum(const std::string& name, const Op& op,
                   const StridedLayout& in) {
    SCOPED_TRACE(name);
    std::optional<BoxSumFactor> factor = factorBoxSum(op.spec, in);
    if (!factor) return false;
    int64_t n = in.numSlots;
    StridedLayout out = inducedOutputLayout(op.spec, in);
    // The copies of a periodic result are not modeled here.
    if (out.period != 0) out = StridedLayout::rowMajor(op.spec.outShape, n);
    int64_t inSize = 1;
    for (int64_t extent : op.spec.inShape) inSize *= extent;
    std::vector<double> x = randomWeights(inSize);
    std::vector<double> expected = naive(op, x);

    std::vector<std::vector<double>> summed = encode(x, in);
    for (const auto& [stride, count] : factor->axes) {
      for (std::vector<double>& ct : summed) {
        std::vector<double> sum(n, 0.0);
        for (int64_t i = 0; i < count; ++i) {
          for (int64_t slot = 0; slot < n; ++slot) {
            sum[slot] += ct[(slot + i * stride) % n];
          }
        }
        ct = sum;
      }
    }

    std::vector<DiagonalKey> keys = computeDiagonalKeys(factor->rest, in, out);
    lastRestIsRelabel = true;
    for (const DiagonalKey& key : keys) {
      lastRestIsRelabel &= key.shift == 0 && key.ctOut == key.ctIn;
    }
    std::vector<std::vector<double>> diagonals = packDiagonals(
        factor->rest, in, out, keys, [](ArrayRef<int64_t>) { return 1.0; });
    std::vector<std::vector<double>> result(out.numCiphertexts(),
                                            std::vector<double>(n, 0.0));
    for (size_t k = 0; k < keys.size(); ++k) {
      const DiagonalKey& key = keys[k];
      for (int64_t slot = 0; slot < n; ++slot) {
        result[key.ctOut][slot] +=
            diagonals[k][slot] * summed[key.ctIn][(slot + key.shift) % n];
      }
    }
    double maxError = 0.0;
    forEachPoint(out.shape, [&](const std::vector<int64_t>& idx) {
      int64_t p = out.position(idx);
      maxError = std::max(
          maxError,
          std::abs(result[p / n][p % n] - expected[flatten(idx, out.shape)]));
    });
    EXPECT_LT(maxError, 1e-9);
    return true;
  }

  // out[c] = sum over h, w of in[c, h, w]
  Op globalSum(int64_t c, int64_t h, int64_t w) {
    Op op;
    op.spec.loops = {c, h, w};
    op.spec.inShape = {c, h, w};
    op.spec.inMap = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    op.spec.inOffset = {0, 0, 0};
    op.spec.outShape = {c};
    op.spec.outMap = {{1, 0, 0}};
    op.spec.outOffset = {0};
    op.coeff = [](ArrayRef<int64_t>) { return 1.0; };
    return op;
  }

  bool lastRestIsRelabel = false;

  // w: (Co, Ci / groups, kh, kw) row-major; tensors are (C, H, W).
  Op conv2d(int64_t co, int64_t ci, int64_t k, int64_t h, int64_t w,
            int64_t stride, int64_t pad, int64_t dil = 1, int64_t groups = 1) {
    int64_t cig = ci / groups;
    int64_t cog = co / groups;
    int64_t ho = (h + 2 * pad - dil * (k - 1) - 1) / stride + 1;
    int64_t wo = (w + 2 * pad - dil * (k - 1) - 1) / stride + 1;
    std::vector<double> weights = randomWeights(co * cig * k * k);
    Op op;
    // loops: group, cog, cig, kh, kw, ho, wo
    op.spec.loops = {groups, cog, cig, k, k, ho, wo};
    op.spec.inShape = {ci, h, w};
    op.spec.inMap = {{cig, 0, 1, 0, 0, 0, 0},
                     {0, 0, 0, dil, 0, stride, 0},
                     {0, 0, 0, 0, dil, 0, stride}};
    op.spec.inOffset = {0, -pad, -pad};
    op.spec.outShape = {co, ho, wo};
    op.spec.outMap = {
        {cog, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 0, 1}};
    op.spec.outOffset = {0, 0, 0};
    op.coeff = [=](ArrayRef<int64_t> d) {
      return weights[(((d[0] * cog + d[1]) * cig + d[2]) * k + d[3]) * k +
                     d[4]];
    };
    return op;
  }

  // The channel is one loop, as in linalg.depthwise_conv_2d_nchw_chw.
  Op depthwise(int64_t c, int64_t k, int64_t h, int64_t w, int64_t stride,
               int64_t pad, std::optional<double> constant = std::nullopt) {
    int64_t ho = (h + 2 * pad - k) / stride + 1;
    int64_t wo = (w + 2 * pad - k) / stride + 1;
    std::vector<double> weights = randomWeights(c * k * k);
    Op op;
    op.spec.loops = {c, k, k, ho, wo};
    op.spec.inShape = {c, h, w};
    op.spec.inMap = {
        {1, 0, 0, 0, 0}, {0, 1, 0, stride, 0}, {0, 0, 1, 0, stride}};
    op.spec.inOffset = {0, -pad, -pad};
    op.spec.outShape = {c, ho, wo};
    op.spec.outMap = {{1, 0, 0, 0, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 0, 1}};
    op.spec.outOffset = {0, 0, 0};
    op.coeff = [=](ArrayRef<int64_t> d) {
      return constant ? *constant : weights[(d[0] * k + d[1]) * k + d[2]];
    };
    return op;
  }

  Op sumPool(int64_t c, int64_t h, int64_t w, int64_t k, int64_t stride,
             double scale) {
    return depthwise(c, k, h, w, stride, 0, scale);
  }

  // y = W x, reading x through its (C, H, W) layout.
  Op fullyConnected(int64_t rows, int64_t c, int64_t h, int64_t w) {
    std::vector<double> weights = randomWeights(rows * c * h * w);
    Op op;
    op.spec.loops = {rows, c, h, w};
    op.spec.inShape = {c, h, w};
    op.spec.inMap = {{0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    op.spec.inOffset = {0, 0, 0};
    op.spec.outShape = {rows};
    op.spec.outMap = {{1, 0, 0, 0}};
    op.spec.outOffset = {0};
    op.coeff = [=](ArrayRef<int64_t> d) {
      return weights[((d[0] * c + d[1]) * h + d[2]) * w + d[3]];
    };
    return op;
  }

  Op matvec(int64_t rows, int64_t cols) {
    std::vector<double> weights = randomWeights(rows * cols);
    Op op;
    op.spec.loops = {rows, cols};
    op.spec.inShape = {cols};
    op.spec.inMap = {{0, 1}};
    op.spec.inOffset = {0};
    op.spec.outShape = {rows};
    op.spec.outMap = {{1, 0}};
    op.spec.outOffset = {0};
    op.coeff = [=](ArrayRef<int64_t> d) { return weights[d[0] * cols + d[1]]; };
    return op;
  }

  // out[c, h, w] = in[c, 0, 0]
  Op broadcast(int64_t c, int64_t h, int64_t w) {
    Op op;
    op.spec.loops = {c, h, w};
    op.spec.inShape = {c, 1, 1};
    op.spec.inMap = {{1, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    op.spec.inOffset = {0, 0, 0};
    op.spec.outShape = {c, h, w};
    op.spec.outMap = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    op.spec.outOffset = {0, 0, 0};
    op.coeff = [](ArrayRef<int64_t>) { return 1.0; };
    return op;
  }

  Op identity(const std::vector<int64_t>& shape) {
    Op op;
    op.spec.loops = shape;
    op.spec.inShape = shape;
    op.spec.outShape = shape;
    for (size_t axis = 0; axis < shape.size(); ++axis) {
      std::vector<int64_t> row(shape.size(), 0);
      row[axis] = 1;
      op.spec.inMap.push_back(row);
      op.spec.outMap.push_back(row);
      op.spec.inOffset.push_back(0);
      op.spec.outOffset.push_back(0);
    }
    op.coeff = [](ArrayRef<int64_t>) { return 1.0; };
    return op;
  }

  size_t lastKeyCount = 0;
  size_t lastShiftCount = 0;

 private:
  std::mt19937 rng_;
  std::normal_distribution<double> dist_{0.0, 1.0};
};

TEST(LinearSpecTest, ConvChainOnOneCiphertext) {
  Harness t(0);
  int64_t n = 512;
  StridedLayout l0 = StridedLayout::rowMajor({3, 8, 8}, n);
  StridedLayout l1 =
      t.check("conv 3->8 k3 p1", t.conv2d(8, 3, 3, 8, 8, 1, 1), l0);
  EXPECT_EQ(t.lastKeyCount, 72u);
  StridedLayout l2 =
      t.check("conv 8->16 k3 s2 p1", t.conv2d(16, 8, 3, 8, 8, 2, 1), l1);
  EXPECT_EQ(t.lastKeyCount, 128u);
  StridedLayout shortcut =
      t.check("conv 8->16 k1 s2", t.conv2d(16, 8, 1, 8, 8, 2, 0), l1);
  // Both operands of the residual addition get the same layout.
  EXPECT_TRUE(l2 == shortcut);
  StridedLayout l3 = t.check("conv 16->16 k3 p1 on gap 2",
                             t.conv2d(16, 16, 3, 4, 4, 1, 1), l2);
  EXPECT_TRUE(l3 == l2);
  StridedLayout l4 = t.check("depthwise k3 s2 p1 on gap 2",
                             t.depthwise(16, 3, 4, 4, 2, 1), l3);
  t.check("conv 16->64 k1 after depthwise", t.conv2d(64, 16, 1, 2, 2, 1, 0),
          l4);
  t.check("grouped conv g4", t.conv2d(16, 16, 3, 4, 4, 1, 1, 1, 4), l3);
  t.check("dilated conv d2", t.conv2d(8, 8, 3, 8, 8, 1, 2, 2), l1);
  StridedLayout pooled =
      t.check("avg pool 2x2 s2", t.sumPool(8, 8, 8, 2, 2, 0.25), l1);
  t.check("conv after pool", t.conv2d(8, 8, 3, 4, 4, 1, 1), pooled);
}

TEST(LinearSpecTest, PoolAndFullyConnected) {
  Harness t(1);
  int64_t n = 512;
  StridedLayout l0 = StridedLayout::rowMajor({8, 8, 8}, n);
  StridedLayout gap2 =
      t.check("conv 8->16 k3 s2 p1", t.conv2d(16, 8, 3, 8, 8, 2, 1), l0);
  StridedLayout pooled =
      t.check("global pool", t.sumPool(16, 4, 4, 4, 4, 1.0 / 16), gap2);
  StridedLayout fc =
      t.check("fc 16->10", t.fullyConnected(10, 16, 1, 1), pooled);
  // A result with no axis on the input's grid is periodic, which bounds the
  // number of diagonals by the period.
  EXPECT_EQ(fc.period, 16);
  EXPECT_LE(t.lastKeyCount, 16u);
  t.check("fc 10->7, periodic in and out", t.matvec(7, 10), fc);
  t.check("fc 10->40, wider out", t.matvec(40, 10), fc);
  t.check("fc 256->10 from gap 2", t.fullyConnected(10, 16, 4, 4), gap2);
  t.check("broadcast", t.broadcast(16, 4, 4), pooled);
}

TEST(LinearSpecTest, OddSizesAndUnpaddedConvs) {
  Harness t(2);
  int64_t n = 512;
  t.check("conv k3 s2 p1 on 7x7", t.conv2d(8, 4, 3, 7, 7, 2, 1),
          StridedLayout::rowMajor({4, 7, 7}, n));
  StridedLayout l0 = StridedLayout::rowMajor({2, 12, 12}, n);
  StridedLayout l1 =
      t.check("valid conv k5", t.conv2d(3, 2, 5, 12, 12, 1, 0), l0);
  // The result stays on the 12-wide grid of the input: one diagonal per tap
  // and channel pair.
  EXPECT_EQ(t.lastKeyCount, 100u);
  t.check("valid conv k3 on 8x8", t.conv2d(3, 3, 3, 8, 8, 1, 0), l1);
}

TEST(LinearSpecTest, PlanesLargerThanACiphertext) {
  Harness t(3);
  int64_t n = 128;
  // Planes of 144 elements at stride 256, two ciphertexts apart.
  StridedLayout l0 = StridedLayout::aligned({3, 12, 12}, n);
  EXPECT_EQ(l0.numCiphertexts(), 6);
  StridedLayout l1 =
      t.check("conv 3->6 k3 p1", t.conv2d(6, 3, 3, 12, 12, 1, 1), l0);
  EXPECT_EQ(l1.numCiphertexts(), 12);
  // Every channel pair has the same nine shifts.
  EXPECT_EQ(t.lastShiftCount, 9u);
  StridedLayout l2 =
      t.check("conv 6->12 k3 s2 p1", t.conv2d(12, 6, 3, 12, 12, 2, 1), l1);
  t.check("depthwise k3 p1 on gap 2", t.depthwise(12, 3, 6, 6, 1, 1), l2);
  t.check("fc 432->5", t.fullyConnected(5, 12, 6, 6), l2);
}

TEST(LinearSpecTest, DilatedConvAcrossCiphertexts) {
  Harness t(6);
  int64_t n = 128;
  StridedLayout l0 = StridedLayout::aligned({5, 11, 9}, n);
  EXPECT_EQ(l0.numCiphertexts(), 5);
  t.check("conv k3 dilation 4 p4", t.conv2d(5, 5, 3, 11, 9, 1, 4, 4), l0);
  // One plane per ciphertext: a shift per tap, in every ciphertext pair.
  EXPECT_EQ(t.lastShiftCount, 9u);
  EXPECT_LE(t.lastKeyCount, 9u * 5u * 5u);
}

TEST(LinearSpecTest, DilatedConvAtModelSize) {
  Harness t(7);
  // Planes of 4040 elements at stride 4096, four to a ciphertext.
  StridedLayout l0 = StridedLayout::aligned({45, 101, 40}, 16384);
  EXPECT_EQ(l0.numCiphertexts(), 12);
  Op op = t.conv2d(45, 45, 3, 101, 40, 1, 16, 16);
  StridedLayout out = inducedOutputLayout(op.spec, l0);
  EXPECT_EQ(out, l0);
  std::vector<DiagonalKey> keys = computeDiagonalKeys(op.spec, l0, out);
  std::set<int64_t> shifts;
  for (const DiagonalKey& key : keys) shifts.insert(key.shift);
  // Nine taps times four plane offsets within a ciphertext.
  EXPECT_EQ(shifts.size(), 36u);
  EXPECT_LE(keys.size(), 36u * 12u * 12u);
  // Row-major planes at stride 4040 need a shift per channel offset.
  StridedLayout dense = StridedLayout::rowMajor({45, 101, 40}, 16384);
  EXPECT_EQ(computeDiagonalKeys(op.spec, dense, dense).size(), 9680u);
}

TEST(LinearSpecTest, BoxSums) {
  Harness t(8);
  int64_t n = 128;
  // Planes of 30 elements at stride 32, four to a ciphertext.
  StridedLayout planes = StridedLayout::aligned({10, 5, 6}, n);
  EXPECT_EQ(planes.numCiphertexts(), 3);
  EXPECT_TRUE(t.checkBoxSum("global sum", t.globalSum(10, 5, 6), planes));
  // Each sum is left at the first position of its plane.
  EXPECT_TRUE(t.lastRestIsRelabel);
  EXPECT_TRUE(t.checkBoxSum("pool k2 s2", t.sumPool(10, 4, 6, 2, 2, 1.0),
                            StridedLayout::aligned({10, 4, 6}, n)));
  EXPECT_TRUE(
      t.checkBoxSum("pool k3 s1", t.sumPool(10, 5, 6, 3, 1, 1.0), planes));

  // Row-major planes of 30 elements cross ciphertext boundaries.
  EXPECT_FALSE(t.checkBoxSum("unaligned", t.globalSum(10, 5, 6),
                             StridedLayout::rowMajor({10, 5, 6}, n)));
  // A padded pool reads zeros that no rotation provides.
  EXPECT_FALSE(
      t.checkBoxSum("padded", t.depthwise(10, 3, 5, 6, 1, 1, 1.0), planes));
  // A gapped input: the strided convolution leaves its result on every
  // second position of the input grid.
  StridedLayout gapped = inducedOutputLayout(
      t.conv2d(4, 4, 3, 8, 8, 2, 1).spec, StridedLayout::aligned({4, 8, 8}, n));
  EXPECT_TRUE(
      t.checkBoxSum("global sum on gaps", t.globalSum(4, 4, 4), gapped));
}

TEST(LinearSpecTest, AnyLayoutPairIsCorrect) {
  Harness t(4);
  int64_t n = 512;
  StridedLayout l0 = StridedLayout::rowMajor({8, 8, 8}, n);
  t.check("conv into row-major", t.conv2d(8, 8, 3, 8, 8, 1, 1), l0,
          StridedLayout::rowMajor({8, 8, 8}, n));
  t.check("strided conv into row-major", t.conv2d(16, 8, 3, 8, 8, 2, 1), l0,
          StridedLayout::rowMajor({16, 4, 4}, n));
  // Misaligned layouts cost diagonals, not correctness.
  EXPECT_EQ(t.lastKeyCount, 512u);
  StridedLayout gap2 =
      inducedOutputLayout(t.conv2d(16, 8, 3, 8, 8, 2, 1).spec, l0);
  t.check("layout conversion", t.identity({16, 4, 4}), gap2,
          StridedLayout::rowMajor({16, 4, 4}, n));
}

TEST(LinearSpecTest, RandomConvs) {
  std::mt19937 rng(5);
  auto pick = [&](std::vector<int64_t> values) {
    return values[std::uniform_int_distribution<size_t>(
        0, values.size() - 1)(rng)];
  };
  for (int trial = 0; trial < 60; ++trial) {
    Harness t(100 + trial);
    int64_t n = pick({64, 128, 256});
    int64_t h = pick({5, 6, 7, 8, 9});
    int64_t w = pick({5, 6, 8, 10});
    int64_t ci = pick({1, 2, 3, 4});
    StridedLayout layout = StridedLayout::rowMajor({ci, h, w}, n);
    for (int layer = 0; layer < 3; ++layer) {
      int64_t k = pick({1, 2, 3});
      int64_t stride = pick({1, 1, 2});
      int64_t pad = pick({0, 1, 2});
      int64_t dil = pick({1, 1, 2, 3});
      if (h + 2 * pad < dil * (k - 1) + 1 || w + 2 * pad < dil * (k - 1) + 1) {
        break;
      }
      int64_t groups = (ci % 2 == 0) ? pick({1, 2}) : 1;
      int64_t co = groups * pick({1, 2, 3});
      Op op = t.conv2d(co, ci, k, h, w, stride, pad, dil, groups);
      layout = t.check(
          "trial " + std::to_string(trial) + " layer " + std::to_string(layer),
          op, layout);
      ci = co;
      h = op.spec.outShape[1];
      w = op.spec.outShape[2];
    }
  }
}

}  // namespace
}  // namespace heir
}  // namespace mlir
