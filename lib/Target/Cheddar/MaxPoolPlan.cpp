#include "lib/Target/Cheddar/MaxPoolPlan.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "extension/max/MaxPoolPlanner.h"      // from @cyclops
#include "extension/poly/SignApproximation.h"  // from @cyclops
#include "lib/Target/Cheddar/MaxPoolCyclopsConfig.h"
#include "lib/Utils/MaxPoolUtils.h"

namespace mlir {
namespace heir {
namespace cheddar {

namespace {

struct CachedRequirements {
  std::optional<MaxPoolRequirements> requirements;
  std::string error;
};

// Cyclops selects the fold copies of a pool for each bootstrap end level, so
// the stages, the level drop and the minimum input level can change with the
// end level (requirement R17 in cyclops_maxpool_requirements.md). HEIR selects
// the end level after its level analysis, so it takes the largest values over
// the end levels that Cyclops' own search covers, and pins the output level.
// Above that range, each fold round fits, so the selection does not change.
constexpr int kHighestEndLevel = 32;

CachedRequirements computeRequirements(const MaxPoolShape& shape) {
  std::optional<cyclops::MaxPoolConfig> config = toCyclopsMaxPoolConfig(shape);
  if (!config)
    return {std::nullopt,
            "a field does not fit a 32-bit int, or value_bound is not "
            "positive"};
  // The planner library reports a rejected shape with an exception.
  try {
    const cyclops::CompositePolynomial comparison =
        cyclops::BuildMaxSelectorLowDepthSignApproximation();
    int minimumEnd =
        cyclops::MaxPoolMinimumBootstrapEndLevel(*config, comparison);
    if (minimumEnd < 0)
      return {std::nullopt, "no bootstrap end level up to 32 is enough"};
    MaxPoolRequirements requirements{0, 0, minimumEnd};
    for (int end = minimumEnd; end <= kHighestEndLevel; ++end) {
      cyclops::MaxPoolPlan plan = cyclops::PlanMaxPoolLayout(*config, end);
      int minimumInput =
          cyclops::MaxPoolMinimumInputLevel(plan, end, comparison);
      if (minimumInput < 0 ||
          !cyclops::MaxPoolTryPlanLevels(plan, minimumInput, end, comparison))
        return {std::nullopt,
                "no input level is enough at bootstrap end level " +
                    std::to_string(end)};
      requirements.levelDrop = std::max(
          requirements.levelDrop, minimumInput - plan.natural_output_level);
      requirements.minimumInputLevel =
          std::max(requirements.minimumInputLevel, minimumInput);
    }
    // An input comes from a bootstrap or an encryption, at most at the end
    // level.
    requirements.minimumBootstrapEndLevel = std::max(
        requirements.minimumBootstrapEndLevel, requirements.minimumInputLevel);
    return {requirements, ""};
  } catch (const std::exception& error) {
    return {std::nullopt, error.what()};
  }
}

}  // namespace

std::optional<MaxPoolRequirements> getMaxPoolRequirements(
    const MaxPoolShape& shape, std::string* error) {
  // The level analyses ask for the same pool many times, and each planner
  // query builds the selection maps over all slots, for each end level.
  using Key =
      std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, bool, double>;
  static std::mutex mutex;
  static std::map<Key, CachedRequirements> cache;
  Key key{shape.numSlots, shape.inputLength, shape.windowSize, shape.stride,
          shape.dilation, shape.ceilMode,    shape.valueBound};
  std::unique_lock<std::mutex> lock(mutex);
  auto it = cache.find(key);
  if (it == cache.end()) {
    lock.unlock();
    CachedRequirements computed = computeRequirements(shape);
    lock.lock();
    it = cache.try_emplace(key, std::move(computed)).first;
  }
  if (!it->second.requirements && error) *error = it->second.error;
  return it->second.requirements;
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
