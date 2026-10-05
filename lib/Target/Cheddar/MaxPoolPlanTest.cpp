#include <optional>
#include <string>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Target/Cheddar/MaxPoolPlan.h"

namespace mlir {
namespace heir {
namespace cheddar {
namespace {

// The expected values come from Cyclops' MaxPool planner at commit e8efd441
// (branch aikata/maxpool). A value bound of 0.5 needs no scaling stage.

TEST(MaxPoolPlanTest, AlignedPowerOfTwoWindowsSkipTheGather) {
  // 64 channels of 76 slots each (L = 75, padded to a multiple of k = 2).
  MaxPoolShape shape{8192, 64 * 76, 2, 2, 1, false, 0.5};
  std::optional<MaxPoolDepths> depths = getMaxPoolDepths(shape);
  ASSERT_TRUE(depths.has_value());
  EXPECT_EQ(depths->gather, 0);
  EXPECT_EQ(depths->league, 2);
  EXPECT_EQ(depths->compaction, 2);
  EXPECT_EQ(depths->total(), 4);
  EXPECT_EQ(depths->minimumInputLevel, 4);
}

TEST(MaxPoolPlanTest, CeilModeKeepsThePartialWindow) {
  std::optional<MaxPoolDepths> depths =
      getMaxPoolDepths({4096, 1000, 3, 3, 1, true, 0.5});
  ASSERT_TRUE(depths.has_value());
  EXPECT_EQ(depths->gather, 2);
  EXPECT_EQ(depths->total(), 6);
}

TEST(MaxPoolPlanTest, OverlappingWindowsSplitIntoChains) {
  // 2998 windows of 4 padded slots need three chains of 1024 windows.
  std::optional<MaxPoolDepths> depths =
      getMaxPoolDepths({4096, 3000, 3, 1, 1, true, 0.5});
  ASSERT_TRUE(depths.has_value());
  EXPECT_EQ(depths->total(), 6);
}

TEST(MaxPoolPlanTest, DilationNeedsAThirdGatherStage) {
  std::optional<MaxPoolDepths> depths =
      getMaxPoolDepths({4096, 1000, 5, 5, 2, true, 0.5});
  ASSERT_TRUE(depths.has_value());
  EXPECT_EQ(depths->gather, 3);
  EXPECT_EQ(depths->total(), 7);
  // A window of 5 pads to 8 slots: the league needs 3 levels.
  EXPECT_EQ(depths->minimumInputLevel, 7);
}

TEST(MaxPoolPlanTest, LargeWindowsNeedMoreInputLevelsThanTheyDrop) {
  // A window of 16 has no gather; the league needs 7 levels but drops 2.
  std::optional<MaxPoolDepths> depths =
      getMaxPoolDepths({4096, 4096, 16, 16, 1, false, 0.5});
  ASSERT_TRUE(depths.has_value());
  EXPECT_EQ(depths->gather, 0);
  EXPECT_EQ(depths->total(), 4);
  EXPECT_EQ(depths->minimumInputLevel, 7);
}

TEST(MaxPoolPlanTest, WindowOverAllSlotsSkipsTheMask) {
  std::optional<MaxPoolDepths> depths =
      getMaxPoolDepths({4096, 4096, 4096, 4096, 1, true, 0.5});
  ASSERT_TRUE(depths.has_value());
  EXPECT_EQ(depths->league, 1);
  EXPECT_EQ(depths->total(), 1);
}

TEST(MaxPoolPlanTest, WindowOfOneIsFree) {
  std::optional<MaxPoolDepths> depths =
      getMaxPoolDepths({4096, 100, 1, 1, 1, true, 0.5});
  ASSERT_TRUE(depths.has_value());
  EXPECT_EQ(depths->total(), 0);
}

TEST(MaxPoolPlanTest, ScalingFoldsIntoExistingStages) {
  // The compaction stages carry the scaling; without gather stages, the
  // scaling into the comparison domain costs one level.
  std::optional<MaxPoolDepths> depths =
      getMaxPoolDepths({8192, 64 * 76, 2, 2, 1, false, 4.0});
  ASSERT_TRUE(depths.has_value());
  EXPECT_EQ(depths->gather, 1);
  EXPECT_EQ(depths->compaction, 2);
  EXPECT_EQ(depths->total(), 5);
  // The ceil-mode pool has gather stages, so its scaling is free.
  depths = getMaxPoolDepths({4096, 1000, 3, 3, 1, true, 4.0});
  ASSERT_TRUE(depths.has_value());
  EXPECT_EQ(depths->gather, 2);
  EXPECT_EQ(depths->total(), 6);
}

TEST(MaxPoolPlanTest, WindowOfOneScalesTwice) {
  // No gather, no league and no compaction stages: both scalings cost a level.
  std::optional<MaxPoolDepths> depths =
      getMaxPoolDepths({4096, 100, 1, 1, 1, true, 2.0});
  ASSERT_TRUE(depths.has_value());
  EXPECT_EQ(depths->gather, 1);
  EXPECT_EQ(depths->compaction, 1);
  EXPECT_EQ(depths->total(), 2);
}

TEST(MaxPoolPlanTest, RejectedShapesGiveNoPlan) {
  // These overlapping dilated windows collide in the staged gather.
  std::string error;
  EXPECT_FALSE(
      getMaxPoolDepths({4096, 1000, 3, 2, 2, true, 0.5}, &error).has_value());
  EXPECT_NE(error.find("gather collides"), std::string::npos) << error;
  // A slot count that does not fit Cyclops' int.
  EXPECT_FALSE(
      getMaxPoolDepths({int64_t{1} << 40, 1000, 2, 2, 1, false}).has_value());
}

}  // namespace
}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
