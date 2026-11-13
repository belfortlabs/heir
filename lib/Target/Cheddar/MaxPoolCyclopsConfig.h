#ifndef LIB_TARGET_CHEDDAR_MAXPOOLCYCLOPSCONFIG_H_
#define LIB_TARGET_CHEDDAR_MAXPOOLCYCLOPSCONFIG_H_

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#include "lib/Target/Cheddar/CyclopsPlanner.h"
#include "lib/Utils/MaxPoolUtils.h"

namespace mlir {
namespace heir {
namespace cheddar {

// The Cyclops MaxPoolConfig of a kernel.max_pool: compact output and a padding
// value of -valueBound. Returns std::nullopt when a field does not fit Cyclops'
// int or valueBound is not positive. The emitted C++ in CheddarToEmitC sets the
// same fields. `planner` must have the MaxPool functions.
inline std::optional<cyclops_max_pool_config> toCyclopsMaxPoolConfig(
    const CyclopsPlanner& planner, const MaxPoolShape& shape) {
  constexpr int64_t kMax = std::numeric_limits<int>::max();
  for (int64_t value : {shape.numSlots, shape.inputLength, shape.windowSize,
                        shape.stride, shape.dilation}) {
    if (value < 1 || value > kMax) return std::nullopt;
  }
  if (!std::isfinite(shape.valueBound) || !(shape.valueBound > 0.0))
    return std::nullopt;
  cyclops_max_pool_config config = planner.cyclops_default_max_pool_config(
      static_cast<int>(shape.numSlots), static_cast<int>(shape.inputLength),
      static_cast<int>(shape.windowSize), static_cast<int>(shape.stride));
  config.dilation = static_cast<int>(shape.dilation);
  config.value_bound = shape.valueBound;
  config.pad_value = -shape.valueBound;
  config.compact_output = 1;
  config.ceil_mode = shape.ceilMode ? 1 : 0;
  return config;
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_CHEDDAR_MAXPOOLCYCLOPSCONFIG_H_
