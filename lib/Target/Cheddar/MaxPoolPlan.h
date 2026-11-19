#ifndef LIB_TARGET_CHEDDAR_MAXPOOLPLAN_H_
#define LIB_TARGET_CHEDDAR_MAXPOOLPLAN_H_

#include <optional>
#include <string>

#include "lib/Utils/MaxPoolUtils.h"

namespace mlir {
namespace heir {
namespace cheddar {

// What one call of Cyclops' MaxPool<word>::EvaluateMax needs from HEIR's
// level plan, with the default comparison polynomial
// (BuildMaxSelectorLowDepthSignApproximation).
struct MaxPoolRequirements {
  // input level - output level. The generated code pins the output level to
  // input level - levelDrop (MaxPoolConfig::output_level), so the drop is the
  // same for each valid input level and each valid bootstrap end level.
  int levelDrop;
  // The lowest input level, for each valid bootstrap end level.
  int minimumInputLevel;
  // The lowest bootstrap end level. Each higher end level is also valid.
  int minimumBootstrapEndLevel;
};

// The requirements of the pool, from Cyclops' MaxPool planner. Returns
// std::nullopt when Cyclops rejects the shape, and then sets `error` to the
// reason if it is not null. The results are cached per shape.
std::optional<MaxPoolRequirements> getMaxPoolRequirements(
    const MaxPoolShape& shape, std::string* error = nullptr);

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_CHEDDAR_MAXPOOLPLAN_H_
