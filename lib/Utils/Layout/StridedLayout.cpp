#include "lib/Utils/Layout/StridedLayout.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/ArrayRef.h"  // from @llvm-project

namespace mlir {
namespace heir {

namespace {

int64_t floorDiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

int64_t digitValue(const StridedDigit& digit, int64_t index) {
  int64_t value = floorDiv(index, digit.div);
  if (digit.mod != 0) value = ((value % digit.mod) + digit.mod) % digit.mod;
  return value;
}

// The contribution of one axis to the position.
int64_t axisTerm(const StridedLayout& layout, int64_t axis, int64_t index) {
  int64_t term = 0;
  for (const StridedDigit& digit : layout.digits) {
    if (digit.axis == axis) term += digit.stride * digitValue(digit, index);
  }
  return term;
}

// Calls fn(idx) for every index of the shape, in row-major order.
template <typename Fn>
void forEachIndex(llvm::ArrayRef<int64_t> shape, Fn fn) {
  for (int64_t extent : shape) {
    if (extent <= 0) return;
  }
  std::vector<int64_t> idx(shape.size(), 0);
  while (true) {
    fn(idx);
    int64_t axis = static_cast<int64_t>(shape.size()) - 1;
    while (axis >= 0 && ++idx[axis] == shape[axis]) idx[axis--] = 0;
    if (axis < 0) return;
  }
}

std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> sortedDigits(
    const StridedLayout& layout) {
  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> result;
  result.reserve(layout.digits.size());
  for (const StridedDigit& digit : layout.digits) {
    result.emplace_back(digit.axis, digit.div, digit.mod, digit.stride);
  }
  std::sort(result.begin(), result.end());
  return result;
}

}  // namespace

int64_t StridedLayout::position(llvm::ArrayRef<int64_t> idx) const {
  int64_t result = offset;
  for (const StridedDigit& digit : digits) {
    result += digit.stride * digitValue(digit, idx[digit.axis]);
  }
  return result;
}

int64_t StridedLayout::maxPosition() const {
  // The axes contribute independently, so the maximum is the sum of the
  // per-axis maxima.
  int64_t result = offset;
  for (int64_t axis = 0; axis < static_cast<int64_t>(shape.size()); ++axis) {
    int64_t best = axisTerm(*this, axis, 0);
    for (int64_t index = 1; index < shape[axis]; ++index) {
      best = std::max(best, axisTerm(*this, axis, index));
    }
    result += best;
  }
  return result;
}

int64_t StridedLayout::numElements() const {
  int64_t result = 1;
  for (int64_t extent : shape) result *= extent;
  return result;
}

std::optional<int64_t> StridedLayout::affineStride(int64_t axis) const {
  std::optional<int64_t> stride = 0;
  int64_t count = 0;
  for (const StridedDigit& digit : digits) {
    if (digit.axis != axis) continue;
    ++count;
    if (digit.div != 1 || digit.mod != 0) return std::nullopt;
    stride = digit.stride;
  }
  if (count > 1) return std::nullopt;
  return stride;
}

bool StridedLayout::isInjective() const {
  std::unordered_set<int64_t> seen;
  seen.reserve(numElements());
  bool injective = true;
  forEachIndex(shape, [&](const std::vector<int64_t>& idx) {
    if (!injective) return;
    int64_t p = position(idx);
    if (period != 0) p = ((p % period) + period) % period;
    injective = seen.insert(p).second;
  });
  return injective;
}

std::string StridedLayout::toIslString() const {
  std::string vars;
  std::string bounds;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    std::string name = "i" + std::to_string(axis);
    vars += (axis == 0 ? "" : ", ") + name;
    bounds += "0 <= " + name + " < " + std::to_string(shape[axis]) + " and ";
  }
  std::string pos = std::to_string(offset);
  for (const StridedDigit& digit : digits) {
    std::string value = "i" + std::to_string(digit.axis);
    if (digit.div != 1) {
      value = "floor((" + value + ")/" + std::to_string(digit.div) + ")";
    }
    if (digit.mod != 0) {
      value = "((" + value + ") mod " + std::to_string(digit.mod) + ")";
    }
    pos += " + " + std::to_string(digit.stride) + " * " + value;
  }
  std::string n = std::to_string(numSlots);
  std::string result = "{ [" + vars + "] -> [ct, slot] : " + bounds;
  if (period != 0) {
    result += "ct = 0 and 0 <= slot < " + n + " and (slot - (" + pos +
              ")) mod " + std::to_string(period) + " = 0 }";
  } else {
    result += "0 <= slot < " + n + " and " + n + " * ct + slot = " + pos + " }";
  }
  return result;
}

bool StridedLayout::operator==(const StridedLayout& other) const {
  return shape == other.shape && numSlots == other.numSlots &&
         offset == other.offset && period == other.period &&
         sortedDigits(*this) == sortedDigits(other);
}

void StridedLayout::normalize() {
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<StridedDigit> kept;
    for (StridedDigit digit : digits) {
      int64_t extent = shape[digit.axis];
      // Largest value of index / div within the shape.
      int64_t maxQuotient = (extent - 1) / digit.div;
      if (digit.stride == 0 || maxQuotient == 0) {
        changed = true;
        continue;
      }
      if (digit.mod != 0 && maxQuotient < digit.mod) {
        digit.mod = 0;
        changed = true;
      }
      kept.push_back(digit);
    }
    digits = std::move(kept);

    // (i / d) mod (m * m2) = (i / d) mod m + m * ((i / (d * m)) mod m2).
    for (size_t lo = 0; lo < digits.size() && !changed; ++lo) {
      if (digits[lo].mod == 0) continue;
      for (size_t hi = 0; hi < digits.size(); ++hi) {
        if (hi == lo || digits[hi].axis != digits[lo].axis ||
            digits[hi].div != digits[lo].div * digits[lo].mod ||
            digits[hi].stride != digits[lo].stride * digits[lo].mod) {
          continue;
        }
        digits[lo].mod =
            digits[hi].mod == 0 ? 0 : digits[lo].mod * digits[hi].mod;
        digits.erase(digits.begin() + hi);
        changed = true;
        break;
      }
    }
  }
}

StridedLayout StridedLayout::transpose(llvm::ArrayRef<int64_t> perm) const {
  StridedLayout result = *this;
  std::vector<int64_t> newAxisOf(shape.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    result.shape[i] = shape[perm[i]];
    newAxisOf[perm[i]] = i;
  }
  for (StridedDigit& digit : result.digits) digit.axis = newAxisOf[digit.axis];
  return result;
}

std::optional<StridedLayout> StridedLayout::collapse(
    llvm::ArrayRef<std::vector<int64_t>> reassociation) const {
  StridedLayout normalized = *this;
  normalized.normalize();

  StridedLayout result;
  result.numSlots = numSlots;
  result.offset = offset;
  result.period = period;
  for (size_t group = 0; group < reassociation.size(); ++group) {
    const std::vector<int64_t>& axes = reassociation[group];
    // The collapsed index is sum of idx[axis] * weight[axis], so
    // idx[axis] = (collapsed / weight) mod extent, and for the leading axis
    // of the group the reduction is not needed.
    int64_t weight = 1;
    for (int64_t k = static_cast<int64_t>(axes.size()) - 1; k >= 0; --k) {
      int64_t axis = axes[k];
      int64_t extent = shape[axis];
      for (const StridedDigit& digit : normalized.digits) {
        if (digit.axis != axis) continue;
        int64_t newAxis = group;
        // No reduction is needed if every axis in front of this one in the
        // group has extent 1.
        bool leading = true;
        for (int64_t j = 0; j < k; ++j) leading &= shape[axes[j]] == 1;
        if (leading) {
          result.digits.push_back(
              {newAxis, weight * digit.div, digit.mod, digit.stride});
          continue;
        }
        if (extent % digit.div != 0) return std::nullopt;
        int64_t range = extent / digit.div;
        if (digit.mod != 0 && range % digit.mod != 0) return std::nullopt;
        result.digits.push_back({newAxis, weight * digit.div,
                                 digit.mod == 0 ? range : digit.mod,
                                 digit.stride});
      }
      weight *= extent;
    }
    result.shape.push_back(weight);
  }
  result.normalize();
  return result;
}

std::optional<StridedLayout> StridedLayout::expand(
    llvm::ArrayRef<std::vector<int64_t>> reassociation,
    llvm::ArrayRef<int64_t> expandedShape) const {
  StridedLayout normalized = *this;
  normalized.normalize();

  StridedLayout result;
  result.numSlots = numSlots;
  result.offset = offset;
  result.period = period;
  result.shape.assign(expandedShape.begin(), expandedShape.end());
  for (const StridedDigit& digit : normalized.digits) {
    const std::vector<int64_t>& axes = reassociation[digit.axis];
    // weights[k] is the weight of axes[k] in the collapsed index.
    std::vector<int64_t> weights(axes.size(), 1);
    for (int64_t k = static_cast<int64_t>(axes.size()) - 2; k >= 0; --k) {
      weights[k] = weights[k + 1] * expandedShape[axes[k + 1]];
    }

    bool placed = false;
    for (size_t k = 0; k < axes.size() && !placed; ++k) {
      int64_t extent = expandedShape[axes[k]];
      if (digit.div % weights[k] != 0) continue;
      int64_t div = digit.div / weights[k];
      // The digit reads within one new axis. Without a reduction that holds
      // if every axis in front of it in the group has extent 1.
      bool leading = true;
      for (size_t j = 0; j < k; ++j) leading &= expandedShape[axes[j]] == 1;
      if ((digit.mod == 0 && leading) ||
          (digit.mod != 0 && extent % (div * digit.mod) == 0)) {
        result.digits.push_back({axes[k], div, digit.mod, digit.stride});
        placed = true;
        continue;
      }
      if (div != 1) continue;
      // The digit reads the row-major index of the new axes first..k.
      size_t first = 0;
      if (digit.mod != 0) {
        int64_t span = 1;
        first = k + 1;
        while (first > 0 && span < digit.mod) {
          --first;
          span *= expandedShape[axes[first]];
        }
        if (span != digit.mod) continue;
      }
      for (size_t c = first; c <= k; ++c) {
        result.digits.push_back(
            {axes[c], 1, 0, digit.stride * (weights[c] / weights[k])});
      }
      placed = true;
    }
    if (!placed) return std::nullopt;
  }
  result.normalize();
  return result;
}

std::optional<StridedLayout> StridedLayout::slice(
    llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> sizes,
    llvm::ArrayRef<int64_t> steps) const {
  StridedLayout normalized = *this;
  normalized.normalize();

  StridedLayout result;
  result.numSlots = numSlots;
  result.offset = offset;
  result.period = period;
  result.shape.assign(sizes.begin(), sizes.end());
  for (const StridedDigit& digit : normalized.digits) {
    int64_t start = offsets[digit.axis];
    int64_t step = steps[digit.axis];
    int64_t last = step * (sizes[digit.axis] - 1);
    int64_t base = start / digit.div;
    int64_t remainder = start % digit.div;

    // The quotient (start + step * i) / div over the slice is one of
    //   base                      (the slice stays within one multiple of div)
    //   base + (step / div) * i   (div divides start and step)
    //   base + i / div            (div divides start, and the step is one)
    if (remainder + last < digit.div) {
      result.offset += digit.stride * digitValue(digit, start);
      continue;
    }
    if (remainder != 0) return std::nullopt;
    StridedDigit quotient;
    if (step == 1) {
      quotient = {digit.axis, digit.div, 0, digit.stride};
    } else if (step % digit.div == 0) {
      quotient = {digit.axis, 1, 0, digit.stride * (step / digit.div)};
    } else {
      return std::nullopt;
    }

    if (digit.mod == 0) {
      result.offset += digit.stride * base;
      result.digits.push_back(quotient);
      continue;
    }
    int64_t first = base % digit.mod;
    if (first + last / digit.div < digit.mod) {
      // The reduction never applies within the slice.
      result.offset += digit.stride * first;
      result.digits.push_back(quotient);
    } else if (first == 0 && step == 1) {
      result.digits.push_back(digit);
    } else {
      return std::nullopt;
    }
  }
  result.normalize();
  return result;
}

StridedLayout StridedLayout::rowMajor(llvm::ArrayRef<int64_t> shape,
                                      int64_t numSlots, int64_t period) {
  StridedLayout layout;
  layout.shape.assign(shape.begin(), shape.end());
  layout.numSlots = numSlots;
  layout.period = period;
  int64_t stride = 1;
  for (int64_t axis = static_cast<int64_t>(shape.size()) - 1; axis >= 0;
       --axis) {
    if (shape[axis] > 1) layout.digits.push_back({axis, 1, 0, stride});
    stride *= shape[axis];
  }
  assert(period == 0 || (numSlots % period == 0 && stride <= period));
  return layout;
}

}  // namespace heir
}  // namespace mlir
