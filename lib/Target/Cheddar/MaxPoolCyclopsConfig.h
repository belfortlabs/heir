#ifndef LIB_TARGET_CHEDDAR_MAXPOOLCYCLOPSCONFIG_H_
#define LIB_TARGET_CHEDDAR_MAXPOOLCYCLOPSCONFIG_H_

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#include "extension/max/MaxPoolPlanner.h"  // from @cyclops
#include "lib/Utils/MaxPoolUtils.h"

namespace mlir {
namespace heir {
namespace cheddar {

// The Cyclops MaxPoolConfig of a kernel.max_pool: compact output and a padding
// value of -valueBound. Returns std::nullopt when a field does not fit Cyclops'
// int or valueBound is not positive. The emitted C++ in CheddarToEmitC sets the
// same fields.
inline std::optional<cyclops::MaxPoolConfig> toCyclopsMaxPoolConfig(
    const MaxPoolShape& shape) {
  constexpr int64_t kMax = std::numeric_limits<int>::max();
  for (int64_t value : {shape.numSlots, shape.inputLength, shape.windowSize,
                        shape.stride, shape.dilation}) {
    if (value < 1 || value > kMax) return std::nullopt;
  }
  if (!std::isfinite(shape.valueBound) || !(shape.valueBound > 0.0))
    return std::nullopt;
  cyclops::MaxPoolConfig config{
      static_cast<int>(shape.numSlots), static_cast<int>(shape.inputLength),
      static_cast<int>(shape.windowSize), static_cast<int>(shape.stride),
      static_cast<int>(shape.dilation)};
  config.value_bound = shape.valueBound;
  config.pad_value = -shape.valueBound;
  config.compact_output = true;
  config.ceil_mode = shape.ceilMode;
  return config;
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_CHEDDAR_MAXPOOLCYCLOPSCONFIG_H_
