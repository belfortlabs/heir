#include "lib/Utils/Layout/SparseBsgs.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "lib/Utils/Layout/LinearSpec.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"  // from @llvm-project

namespace mlir {
namespace heir {

namespace {

// (rotations, groups) of the plan with the given giant step.
std::pair<int64_t, int64_t> costOf(llvm::ArrayRef<DiagonalKey> keys,
                                   int64_t giantStep) {
  std::set<std::pair<int64_t, int64_t>> babies;
  std::set<std::pair<int64_t, int64_t>> groups;
  int64_t rotations = 0;
  for (const DiagonalKey& key : keys) {
    int64_t baby = key.shift % giantStep;
    int64_t giant = key.shift - baby;
    if (babies.insert({key.ctIn, baby}).second && baby != 0) ++rotations;
    if (groups.insert({key.ctOut, giant}).second && giant != 0) ++rotations;
  }
  return {rotations, static_cast<int64_t>(groups.size())};
}

}  // namespace

int64_t SparseBsgsPlan::numRotations() const {
  int64_t rotations = 0;
  for (int64_t shift : babyShift) rotations += shift != 0;
  for (size_t r = 0; r < row.size(); ++r) {
    rotations += isLast[r] && giant[r] != 0;
  }
  return rotations;
}

SparseBsgsPlan planSparseBsgs(llvm::ArrayRef<DiagonalKey> keys,
                              int64_t numSlots) {
  SparseBsgsPlan plan;
  std::pair<int64_t, int64_t> best = costOf(keys, 1);
  for (int64_t giantStep = 2; giantStep <= numSlots; giantStep *= 2) {
    std::pair<int64_t, int64_t> cost = costOf(keys, giantStep);
    if (cost < best) {
      best = cost;
      plan.giantStep = giantStep;
    }
  }

  std::map<std::pair<int64_t, int64_t>, int64_t> babyIndex;
  for (const DiagonalKey& key : keys) {
    babyIndex.insert({{key.ctIn, key.shift % plan.giantStep}, 0});
  }
  for (auto& [baby, index] : babyIndex) {
    index = plan.babyCt.size();
    plan.babyCt.push_back(baby.first);
    plan.babyShift.push_back(baby.second);
  }

  std::vector<int64_t> order(keys.size());
  std::iota(order.begin(), order.end(), 0);
  auto groupOf = [&](int64_t k) {
    int64_t baby = keys[k].shift % plan.giantStep;
    return std::make_tuple(keys[k].ctOut, keys[k].shift - baby);
  };
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return groupOf(a) < groupOf(b);
  });
  for (size_t r = 0; r < order.size(); ++r) {
    int64_t k = order[r];
    int64_t baby = keys[k].shift % plan.giantStep;
    plan.row.push_back(k);
    plan.babyOf.push_back(babyIndex.at({keys[k].ctIn, baby}));
    plan.giant.push_back(keys[k].shift - baby);
    plan.out.push_back(keys[k].ctOut);
    plan.isLast.push_back(r + 1 == order.size() ||
                          groupOf(order[r + 1]) != groupOf(k));
  }
  return plan;
}

}  // namespace heir
}  // namespace mlir
