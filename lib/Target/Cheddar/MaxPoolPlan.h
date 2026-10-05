#ifndef LIB_TARGET_CHEDDAR_MAXPOOLPLAN_H_
#define LIB_TARGET_CHEDDAR_MAXPOOLPLAN_H_

#include <optional>
#include <string>

#include "lib/Utils/MaxPoolUtils.h"

namespace mlir {
namespace heir {
namespace cheddar {

// The levels each phase of MaxPool<word>::EvaluateMax drops from its input.
struct MaxPoolDepths {
  // Selection stages of the deepest chain's gather, one level each. A pool
  // without gather stages has one scaling stage when valueBound != 0.5.
  int gather;
  // HierarchicalLeague's final multiplications: by the selector, and by a
  // window mask unless the window spans the whole ciphertext.
  int league;
  // Selection stages of the deepest chain's compaction, one level each, with
  // the same scaling stage rule as `gather`.
  int compaction;

  // The lowest input level that PlanMaxPool accepts.
  int minimumInputLevel;

  int total() const { return gather + league + compaction; }
};

// The depths of the pool, from Cyclops' PlanMaxPool, for a pool without the
// small-window league level cap (requirement R2 in
// cyclops_maxpool_requirements.md). Returns std::nullopt when Cyclops rejects
// the shape, and then sets `error` to the reason if it is not null. The
// results are cached per shape.
std::optional<MaxPoolDepths> getMaxPoolDepths(const MaxPoolShape& shape,
                                              std::string* error = nullptr);

// The lowest bootstrap end level that HierarchicalLeague::Compile accepts with
// the default comparison polynomial (BuildMaxSelectorLowDepthSignApproximation,
// 9 levels): its minimum input level plus 2.
//
// TODO: take this from the planner once it exports it (requirement R3b in
// cyclops_maxpool_requirements.md).
inline int getMaxPoolMinimumBootstrapEndLevel() { return 11; }

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_CHEDDAR_MAXPOOLPLAN_H_
