#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/StridedLayout.h"
#include "lib/Utils/Layout/Utils.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace {

using Point = std::vector<int64_t>;

// All (index, (ct, slot)) pairs of the layout, from the position formula.
std::vector<std::pair<Point, Point>> pointsOf(const StridedLayout& layout) {
  std::vector<std::pair<Point, Point>> points;
  int64_t n = layout.numSlots;
  Point idx(layout.shape.size(), 0);
  while (true) {
    int64_t p = layout.position(idx);
    if (layout.period == 0) {
      points.push_back({idx, {p / n, p % n}});
    } else {
      for (int64_t m = 0; m < n / layout.period; ++m) {
        points.push_back({idx, {0, (p + m * layout.period) % n}});
      }
    }
    int64_t axis = static_cast<int64_t>(idx.size()) - 1;
    while (axis >= 0 && ++idx[axis] == layout.shape[axis]) idx[axis--] = 0;
    if (axis < 0) break;
  }
  std::sort(points.begin(), points.end());
  return points;
}

void expectRelationMatches(const StridedLayout& layout) {
  auto relation = getIntegerRelationFromIslStr(layout.toIslString());
  ASSERT_TRUE(succeeded(relation)) << layout.toIslString();
  PointPairCollector collector(layout.shape.size(), 2);
  enumeratePoints(relation.value(), collector);
  std::sort(collector.points.begin(), collector.points.end());
  EXPECT_EQ(collector.points, pointsOf(layout)) << layout.toIslString();
}

TEST(StridedLayoutTest, RowMajor) {
  StridedLayout layout = StridedLayout::rowMajor({3, 4, 5}, 32);
  EXPECT_EQ(layout.position({2, 3, 4}), 59);
  EXPECT_EQ(layout.numCiphertexts(), 2);
  EXPECT_EQ(layout.affineStride(0), 20);
  EXPECT_EQ(layout.affineStride(2), 1);
  EXPECT_TRUE(layout.isInjective());
  expectRelationMatches(layout);
}

TEST(StridedLayoutTest, PeriodicVector) {
  StridedLayout layout = StridedLayout::rowMajor({5}, 32, 8);
  EXPECT_EQ(layout.numCiphertexts(), 1);
  EXPECT_TRUE(layout.isInjective());
  expectRelationMatches(layout);
}

TEST(StridedLayoutTest, ChannelMultiplexed) {
  // 8 channels of 4x4 on an 8x8 grid with gap 2: channels fill the 2x2 gaps,
  // then a second plane.
  StridedLayout layout;
  layout.shape = {8, 4, 4};
  layout.numSlots = 64;
  layout.digits = {
      {1, 1, 0, 16}, {2, 1, 0, 2}, {0, 1, 2, 1}, {0, 2, 2, 8}, {0, 4, 0, 64}};
  EXPECT_EQ(layout.position({0, 0, 0}), 0);
  EXPECT_EQ(layout.position({1, 0, 0}), 1);
  EXPECT_EQ(layout.position({2, 0, 0}), 8);
  EXPECT_EQ(layout.position({4, 0, 0}), 64);
  EXPECT_EQ(layout.position({0, 1, 1}), 18);
  EXPECT_EQ(layout.numCiphertexts(), 2);
  EXPECT_FALSE(layout.affineStride(0).has_value());
  EXPECT_EQ(layout.affineStride(1), 16);
  EXPECT_TRUE(layout.isInjective());
  expectRelationMatches(layout);
}

TEST(StridedLayoutTest, CollidingDigitsAreNotInjective) {
  StridedLayout layout;
  layout.shape = {4, 4};
  layout.numSlots = 16;
  layout.digits = {{0, 1, 0, 2}, {1, 1, 0, 1}};
  EXPECT_FALSE(layout.isInjective());
}

// Calls fn(idx) for every index of the shape.
template <typename Fn>
void forEachIndex(const std::vector<int64_t>& shape, Fn fn) {
  Point idx(shape.size(), 0);
  while (true) {
    fn(idx);
    int64_t axis = static_cast<int64_t>(idx.size()) - 1;
    while (axis >= 0 && ++idx[axis] == shape[axis]) idx[axis--] = 0;
    if (axis < 0) return;
  }
}

int64_t rowMajorIndex(const Point& idx, const std::vector<int64_t>& shape) {
  int64_t result = 0;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    result = result * shape[axis] + idx[axis];
  }
  return result;
}

StridedLayout channelMultiplexed() {
  StridedLayout layout;
  layout.shape = {8, 4, 4};
  layout.numSlots = 64;
  layout.digits = {
      {1, 1, 0, 16}, {2, 1, 0, 2}, {0, 1, 2, 1}, {0, 2, 2, 8}, {0, 4, 0, 64}};
  return layout;
}

TEST(StridedLayoutTest, NormalizeMergesDigitChains) {
  StridedLayout layout;
  layout.shape = {60};
  layout.numSlots = 64;
  layout.digits = {{0, 1, 5, 1}, {0, 5, 4, 5}, {0, 20, 0, 20}};
  StridedLayout normalized = layout;
  normalized.normalize();
  EXPECT_EQ(normalized.affineStride(0), 1);
  forEachIndex(layout.shape, [&](const Point& idx) {
    EXPECT_EQ(normalized.position(idx), layout.position(idx));
  });
}

TEST(StridedLayoutTest, Transpose) {
  StridedLayout layout = channelMultiplexed();
  StridedLayout transposed = layout.transpose({1, 2, 0});
  EXPECT_EQ(transposed.shape, (std::vector<int64_t>{4, 4, 8}));
  forEachIndex(transposed.shape, [&](const Point& idx) {
    EXPECT_EQ(transposed.position(idx),
              layout.position({idx[2], idx[0], idx[1]}));
  });
}

TEST(StridedLayoutTest, CollapseAndExpandRoundTrip) {
  for (const StridedLayout& layout :
       {StridedLayout::rowMajor({3, 4, 5}, 32), channelMultiplexed()}) {
    for (const std::vector<std::vector<int64_t>>& groups :
         std::vector<std::vector<std::vector<int64_t>>>{
             {{0, 1, 2}}, {{0}, {1, 2}}, {{0, 1}, {2}}}) {
      auto collapsed = layout.collapse(groups);
      ASSERT_TRUE(collapsed.has_value());
      // Each collapsed axis is the row-major index of its group.
      forEachIndex(layout.shape, [&](const Point& idx) {
        Point collapsedIdx;
        for (const std::vector<int64_t>& group : groups) {
          Point part;
          std::vector<int64_t> partShape;
          for (int64_t axis : group) {
            part.push_back(idx[axis]);
            partShape.push_back(layout.shape[axis]);
          }
          collapsedIdx.push_back(rowMajorIndex(part, partShape));
        }
        EXPECT_EQ(collapsed->position(collapsedIdx), layout.position(idx));
      });

      auto expanded = collapsed->expand(groups, layout.shape);
      ASSERT_TRUE(expanded.has_value());
      forEachIndex(layout.shape, [&](const Point& idx) {
        EXPECT_EQ(expanded->position(idx), layout.position(idx));
      });
    }
  }
}

TEST(StridedLayoutTest, UnitDimReshapesAlwaysStayInTheFamily) {
  // Six channels placed two per gap and then in planes: (c mod 2) and c / 2.
  // The digit c / 2 has no reduction and 6 is not a multiple of 4, so only
  // the fact that the added axis has extent 1 makes these reshapes exact.
  StridedLayout layout;
  layout.shape = {6, 4, 4};
  layout.numSlots = 256;
  layout.digits = {{0, 1, 2, 1}, {0, 2, 0, 64}, {1, 1, 0, 16}, {2, 1, 0, 2}};

  auto expanded = layout.expand({{0, 1}, {2}, {3}}, {1, 6, 4, 4});
  ASSERT_TRUE(expanded.has_value());
  forEachIndex(layout.shape, [&](const Point& idx) {
    EXPECT_EQ(expanded->position({0, idx[0], idx[1], idx[2]}),
              layout.position(idx));
  });

  auto collapsed = expanded->collapse({{0, 1}, {2}, {3}});
  ASSERT_TRUE(collapsed.has_value());
  EXPECT_EQ(*collapsed, layout);
}

TEST(StridedLayoutTest, CollapseOfRowMajorIsRowMajor) {
  auto collapsed = StridedLayout::rowMajor({3, 4, 5}, 64).collapse({{0, 1, 2}});
  ASSERT_TRUE(collapsed.has_value());
  EXPECT_EQ(*collapsed, StridedLayout::rowMajor({60}, 64));
}

TEST(StridedLayoutTest, ExpandOutsideTheFamily) {
  // A vector whose index is split as (i mod 3) and (i / 3) cannot be viewed
  // as a 3x4 tensor: the low digit would straddle both new axes.
  StridedLayout layout;
  layout.shape = {12};
  layout.numSlots = 64;
  layout.digits = {{0, 1, 3, 1}, {0, 3, 0, 16}};
  EXPECT_FALSE(layout.expand({{0, 1}}, {3, 4}).has_value());
  EXPECT_TRUE(layout.expand({{0, 1}}, {4, 3}).has_value());
}

TEST(StridedLayoutTest, Slice) {
  StridedLayout layout = channelMultiplexed();
  std::vector<int64_t> offsets = {2, 1, 0};
  std::vector<int64_t> sizes = {2, 2, 2};
  std::vector<int64_t> steps = {1, 1, 2};
  auto sliced = layout.slice(offsets, sizes, steps);
  ASSERT_TRUE(sliced.has_value());
  forEachIndex(sizes, [&](const Point& idx) {
    EXPECT_EQ(sliced->position(idx),
              layout.position({offsets[0] + steps[0] * idx[0],
                               offsets[1] + steps[1] * idx[1],
                               offsets[2] + steps[2] * idx[2]}));
  });

  // Fixing the channel moves its position into the offset.
  auto channel = layout.slice({5, 0, 0}, {1, 4, 4}, {1, 1, 1});
  ASSERT_TRUE(channel.has_value());
  EXPECT_EQ(channel->offset, layout.position({5, 0, 0}));
  EXPECT_EQ(channel->affineStride(1), 16);

  // A channel range that starts inside a gap cell leaves the family.
  EXPECT_FALSE(layout.slice({1, 0, 0}, {4, 4, 4}, {1, 1, 1}).has_value());
}

}  // namespace
}  // namespace heir
}  // namespace mlir
