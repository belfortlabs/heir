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

#include "extension/max/MaxPoolPlanner.h"  // from @cyclops
#include "lib/Target/Cheddar/MaxPoolCyclopsConfig.h"
#include "lib/Utils/MaxPoolUtils.h"

namespace mlir {
namespace heir {
namespace cheddar {

namespace {

struct CachedDepths {
  std::optional<MaxPoolDepths> depths;
  std::string error;
};

// An input level and a bootstrap end level above the needs of any pool.
constexpr int kHighLevel = 64;

CachedDepths computeDepths(const MaxPoolShape& shape) {
  std::optional<cyclops::MaxPoolConfig> config = toCyclopsMaxPoolConfig(shape);
  if (!config)
    return {std::nullopt,
            "a field does not fit a 32-bit int, or value_bound is not "
            "positive"};
  // The planner library reports a rejected shape with an exception.
  try {
    // The small-window league level cap lowers the league input level, but
    // not the levels below it, so the differences below do not depend on it.
    cyclops::MaxPoolPlan plan =
        cyclops::PlanMaxPool(*config, kHighLevel, kHighLevel);
    MaxPoolDepths depths{std::ranges::max(plan.gather_stages),
                         plan.league_level - plan.league_output_level,
                         std::ranges::max(plan.compaction_stages),
                         /*minimumInputLevel=*/-1};
    // The output level is at least zero, so the pool needs at least the
    // levels that it drops.
    for (int level = depths.total(); level <= kHighLevel; ++level) {
      try {
        cyclops::PlanMaxPool(*config, level, kHighLevel);
        depths.minimumInputLevel = level;
        break;
      } catch (const std::exception&) {
      }
    }
    if (depths.minimumInputLevel < 0)
      return {std::nullopt, "no input level up to 64 is enough"};
    return {depths, ""};
  } catch (const std::exception& error) {
    return {std::nullopt, error.what()};
  }
}

}  // namespace

std::optional<MaxPoolDepths> getMaxPoolDepths(const MaxPoolShape& shape,
                                              std::string* error) {
  // The level analyses ask for the same pool many times, and each PlanMaxPool
  // call builds the selection maps over all slots.
  using Key =
      std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, bool, double>;
  static std::mutex mutex;
  static std::map<Key, CachedDepths> cache;
  Key key{shape.numSlots, shape.inputLength, shape.windowSize, shape.stride,
          shape.dilation, shape.ceilMode,    shape.valueBound};
  std::unique_lock<std::mutex> lock(mutex);
  auto it = cache.find(key);
  if (it == cache.end()) {
    lock.unlock();
    CachedDepths computed = computeDepths(shape);
    lock.lock();
    it = cache.try_emplace(key, std::move(computed)).first;
  }
  if (!it->second.depths && error) *error = it->second.error;
  return it->second.depths;
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
