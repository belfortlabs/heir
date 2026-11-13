#include <cstdint>
#include <optional>
#include <string>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Target/Cheddar/CyclopsPlanner.h"
#include "lib/Target/Cheddar/MaxPoolPlan.h"

namespace mlir {
namespace heir {
namespace cheddar {
namespace {

// The expected values come from Cyclops' MaxPool planner on branch
// mdgrs/maxpool-c-api (aikata/maxpool at 26b6f443, with the MaxPool functions
// of the C API). A value bound of 0.5 needs no scaling stage.

// The tests need libcyclops_planner with the MaxPool functions on the library
// path, e.g. bazel test --test_env=LD_LIBRARY_PATH=/path/to/dir.
class MaxPoolPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string& error = getCyclopsPlanner().maxPoolError;
    if (!error.empty()) GTEST_SKIP() << "no Cyclops MaxPool planner: " << error;
  }
};

MaxPoolRequirements expectRequirements(const MaxPoolShape& shape) {
  std::string error;
  std::optional<MaxPoolRequirements> requirements =
      getMaxPoolRequirements(shape, &error);
  EXPECT_TRUE(requirements.has_value()) << error;
  return requirements.value_or(MaxPoolRequirements{-1, -1, -1});
}

TEST_F(MaxPoolPlanTest, AlignedPowerOfTwoWindowsSkipTheGather) {
  // 64 channels of 76 slots each (L = 75, padded to a multiple of k = 2): the
  // league and 2 compaction stages.
  MaxPoolRequirements r =
      expectRequirements({8192, 64 * 76, 2, 2, 1, false, 0.5});
  EXPECT_EQ(r.levelDrop, 3);
  EXPECT_EQ(r.minimumInputLevel, 3);
  EXPECT_EQ(r.minimumBootstrapEndLevel, 11);
}

TEST_F(MaxPoolPlanTest, SmallWindowsDropTheSameFromEachInputLevel) {
  // The comparisons of a window below 8 slots run on a leveled-down copy, so
  // a high input level keeps its levels (requirement R2).
  MaxPoolRequirements r = expectRequirements({4096, 4096, 2, 2, 1, false, 0.5});
  EXPECT_EQ(r.levelDrop, 3);
  EXPECT_EQ(r.minimumInputLevel, 3);
}

TEST_F(MaxPoolPlanTest, CeilModeKeepsThePartialWindow) {
  MaxPoolRequirements r = expectRequirements({4096, 1000, 3, 3, 1, true, 0.5});
  EXPECT_EQ(r.levelDrop, 5);
  EXPECT_EQ(r.minimumInputLevel, 5);
}

TEST_F(MaxPoolPlanTest, OverlappingWindowsSplitIntoChains) {
  MaxPoolRequirements r = expectRequirements({4096, 3000, 3, 1, 1, true, 0.5});
  EXPECT_EQ(r.levelDrop, 5);
}

TEST_F(MaxPoolPlanTest, DilationNeedsNoExtraGatherStage) {
  // The fewest-stage selection (Cyclops 1e9f46d6) gathers the dilated windows
  // in as many stages as the undilated ones.
  MaxPoolRequirements r = expectRequirements({4096, 1000, 5, 5, 2, true, 0.5});
  EXPECT_EQ(r.levelDrop, 5);
  EXPECT_EQ(r.minimumInputLevel, 5);
}

TEST_F(MaxPoolPlanTest, LargeWindowsNeedMoreInputLevelsThanTheyDrop) {
  // A window of 16 has no gather; the league needs 7 levels but drops 1, and
  // a bootstrap end level of 13.
  MaxPoolRequirements r =
      expectRequirements({4096, 4096, 16, 16, 1, false, 0.5});
  EXPECT_EQ(r.levelDrop, 3);
  EXPECT_EQ(r.minimumInputLevel, 7);
  EXPECT_EQ(r.minimumBootstrapEndLevel, 13);
}

TEST_F(MaxPoolPlanTest, FoldCopiesTakeTheLargestDropOverTheEndLevels) {
  // Cyclops can fold these 16 windows with a different copy count at each end
  // level (requirement R17), which changes the gather stages, so HEIR takes
  // the largest values over the end levels.
  MaxPoolRequirements r =
      expectRequirements({4096, 1024, 64, 64, 1, false, 0.5});
  EXPECT_EQ(r.levelDrop, 4);
  EXPECT_EQ(r.minimumInputLevel, 9);
  EXPECT_EQ(r.minimumBootstrapEndLevel, 13);
}

TEST_F(MaxPoolPlanTest, WindowOverAllSlotsNeedsADeepBootstrap) {
  MaxPoolRequirements r =
      expectRequirements({4096, 4096, 4096, 4096, 1, true, 0.5});
  EXPECT_EQ(r.levelDrop, 2);
  EXPECT_EQ(r.minimumBootstrapEndLevel, 15);
}

TEST_F(MaxPoolPlanTest, WindowOfOneIsFree) {
  MaxPoolRequirements r = expectRequirements({4096, 100, 1, 1, 1, true, 0.5});
  EXPECT_EQ(r.levelDrop, 0);
  EXPECT_EQ(r.minimumInputLevel, 0);
}

TEST_F(MaxPoolPlanTest, ScalingFoldsIntoExistingStages) {
  // The compaction stages carry the scaling; without gather stages, the
  // scaling into the comparison domain costs one level.
  EXPECT_EQ(expectRequirements({8192, 64 * 76, 2, 2, 1, false, 4.0}).levelDrop,
            4);
  // The ceil-mode pool has gather stages, so its scaling is free.
  EXPECT_EQ(expectRequirements({4096, 1000, 3, 3, 1, true, 4.0}).levelDrop, 5);
}

TEST_F(MaxPoolPlanTest, WindowOfOneScalesTwice) {
  // No gather, no league and no compaction stages: both scalings cost a level.
  EXPECT_EQ(expectRequirements({4096, 100, 1, 1, 1, true, 2.0}).levelDrop, 2);
}

TEST_F(MaxPoolPlanTest, FewerSlotsThanTheBootstrapMinimum) {
  // The levels of a 64-slot ciphertext. Its keys need the 256-slot bootstrap
  // transform (requirement R6), but its level plan does not.
  MaxPoolRequirements r = expectRequirements({64, 64, 2, 2, 1, false, 0.5});
  EXPECT_EQ(r.levelDrop, 2);
  EXPECT_EQ(r.minimumBootstrapEndLevel, 11);
}

TEST_F(MaxPoolPlanTest, RejectedShapesGiveNoPlan) {
  // A window that does not fit the input.
  std::string error;
  EXPECT_FALSE(getMaxPoolRequirements({4096, 1, 2, 2, 1, true, 0.5}, &error)
                   .has_value());
  EXPECT_NE(error.find("bad input_length"), std::string::npos) << error;
  // A slot count that does not fit Cyclops' int.
  EXPECT_FALSE(getMaxPoolRequirements({int64_t{1} << 40, 1000, 2, 2, 1, false})
                   .has_value());
}

}  // namespace
}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
