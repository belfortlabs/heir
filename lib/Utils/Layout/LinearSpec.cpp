#include "lib/Utils/Layout/LinearSpec.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "lib/Utils/Layout/StridedLayout.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project

namespace mlir {
namespace heir {

namespace {

using llvm::ArrayRef;

int64_t floorDiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

int64_t ceilDiv(int64_t a, int64_t b) { return -floorDiv(-a, b); }

int64_t floorMod(int64_t a, int64_t b) { return a - floorDiv(a, b) * b; }

std::vector<int64_t> applyMap(const std::vector<std::vector<int64_t>>& map,
                              const std::vector<int64_t>& offset,
                              ArrayRef<int64_t> d) {
  std::vector<int64_t> idx(map.size());
  for (size_t axis = 0; axis < map.size(); ++axis) {
    int64_t value = offset[axis];
    for (size_t loop = 0; loop < d.size(); ++loop) {
      value += map[axis][loop] * d[loop];
    }
    idx[axis] = value;
  }
  return idx;
}

// Calls fn(d) for every point of the box, in row-major order. `dims` selects
// the loops to step; the other entries of `d` are left as they are.
template <typename Fn>
void forEachPoint(ArrayRef<int64_t> extents, ArrayRef<int64_t> dims,
                  std::vector<int64_t>& d, Fn fn) {
  for (int64_t dim : dims) {
    if (extents[dim] <= 0) return;
    d[dim] = 0;
  }
  while (true) {
    fn();
    int64_t k = static_cast<int64_t>(dims.size()) - 1;
    while (k >= 0 && ++d[dims[k]] == extents[dims[k]]) d[dims[k--]] = 0;
    if (k < 0) return;
  }
}

using ShiftModulus = ShiftRule;

ShiftModulus getShiftModulus(const StridedLayout& in,
                             const StridedLayout& out) {
  return getShiftRule(in, out);
}

DiagonalPlacement place(int64_t inPos, int64_t outPos, ShiftModulus modulus,
                        int64_t numSlots) {
  assert(inPos >= 0 && outPos >= 0 && "positions must be non-negative");
  int64_t shift = floorMod(inPos - outPos, modulus.modulus);
  if (modulus.writeFollowsInput) {
    return {{0, inPos / numSlots, shift}, floorMod(inPos - shift, numSlots)};
  }
  return {{outPos / numSlots, inPos / numSlots, shift}, outPos % numSlots};
}

// The loops on which the shift does not depend, with the coefficient of the
// output position on each.
std::vector<std::pair<int64_t, int64_t>> getAlignedLoops(
    const LinearSpec& spec, const StridedLayout& in, const StridedLayout& out) {
  size_t numLoops = spec.loops.size();
  std::vector<int64_t> coeff(numLoops, 0);
  std::vector<bool> aligned(numLoops, false);
  for (size_t loop = 0; loop < numLoops; ++loop) {
    bool affine = true;
    int64_t coeffIn = 0;
    int64_t coeffOut = 0;
    for (size_t axis = 0; axis < spec.inMap.size(); ++axis) {
      if (spec.inMap[axis][loop] == 0) continue;
      std::optional<int64_t> stride = in.affineStride(axis);
      affine &= stride.has_value();
      coeffIn += stride.value_or(0) * spec.inMap[axis][loop];
    }
    for (size_t axis = 0; axis < spec.outMap.size(); ++axis) {
      if (spec.outMap[axis][loop] == 0) continue;
      std::optional<int64_t> stride = out.affineStride(axis);
      affine &= stride.has_value();
      coeffOut += stride.value_or(0) * spec.outMap[axis][loop];
    }
    if (affine && coeffIn == coeffOut && coeffOut > 0) {
      aligned[loop] = true;
      coeff[loop] = coeffOut;
    }
  }
  // The bounds of each input axis must give an interval for one aligned loop.
  for (size_t axis = 0; axis < spec.inMap.size(); ++axis) {
    int64_t first = -1;
    for (size_t loop = 0; loop < numLoops; ++loop) {
      if (!aligned[loop] || spec.inMap[axis][loop] == 0) continue;
      if (first < 0) {
        first = loop;
      } else {
        aligned[loop] = false;
      }
    }
    if (first >= 0 && spec.inMap[axis][first] < 0) aligned[first] = false;
  }
  std::vector<std::pair<int64_t, int64_t>> result;
  for (size_t loop = 0; loop < numLoops; ++loop) {
    if (aligned[loop]) result.emplace_back(loop, coeff[loop]);
  }
  // Largest coefficient first, so the last one has the most points per
  // ciphertext boundary.
  std::stable_sort(
      result.begin(), result.end(),
      [](const auto& a, const auto& b) { return a.second > b.second; });
  return result;
}

class KeySet {
 public:
  void insert(const DiagonalKey& key) {
    assert(key.ctOut >= 0 && key.ctOut < (1 << 20) && key.ctIn >= 0 &&
           key.ctIn < (1 << 20) && key.shift >= 0 && key.shift < (1 << 24));
    uint64_t packed = (static_cast<uint64_t>(key.ctOut) << 44) |
                      (static_cast<uint64_t>(key.ctIn) << 24) |
                      static_cast<uint64_t>(key.shift);
    if (packed == last_) return;
    last_ = packed;
    keys_.insert(packed);
  }

  std::vector<DiagonalKey> sorted() const {
    std::vector<DiagonalKey> result;
    result.reserve(keys_.size());
    for (uint64_t packed : keys_) {
      result.push_back({static_cast<int64_t>(packed >> 44),
                        static_cast<int64_t>((packed >> 24) & 0xFFFFF),
                        static_cast<int64_t>(packed & 0xFFFFFF)});
    }
    std::sort(result.begin(), result.end());
    return result;
  }

 private:
  std::unordered_set<uint64_t> keys_;
  uint64_t last_ = ~uint64_t{0};
};

}  // namespace

ShiftRule getShiftRule(const StridedLayout& in, const StridedLayout& out) {
  if (out.period != 0 && (in.period == 0 || out.period <= in.period)) {
    return {out.period, true};
  }
  if (in.period != 0) return {in.period, false};
  return {out.numSlots, false};
}

int64_t LinearSpec::numIterations() const {
  int64_t result = 1;
  for (int64_t extent : loops) result *= extent;
  return result;
}

std::optional<DiagonalPlacement> placeIteration(const LinearSpec& spec,
                                                const StridedLayout& in,
                                                const StridedLayout& out,
                                                ArrayRef<int64_t> d) {
  std::vector<int64_t> inIdx = applyMap(spec.inMap, spec.inOffset, d);
  for (size_t axis = 0; axis < inIdx.size(); ++axis) {
    if (inIdx[axis] < 0 || inIdx[axis] >= spec.inShape[axis]) {
      return std::nullopt;
    }
  }
  std::vector<int64_t> outIdx = applyMap(spec.outMap, spec.outOffset, d);
  return place(in.position(inIdx), out.position(outIdx),
               getShiftModulus(in, out), out.numSlots);
}

std::vector<DiagonalKey> computeDiagonalKeysByEnumeration(
    const LinearSpec& spec, const StridedLayout& in, const StridedLayout& out) {
  KeySet keys;
  std::vector<int64_t> allLoops(spec.loops.size());
  for (size_t loop = 0; loop < allLoops.size(); ++loop) allLoops[loop] = loop;
  std::vector<int64_t> d(spec.loops.size(), 0);
  forEachPoint(spec.loops, allLoops, d, [&]() {
    if (auto placement = placeIteration(spec, in, out, d)) {
      keys.insert(placement->key);
    }
  });
  return keys.sorted();
}

std::vector<DiagonalKey> computeDiagonalKeys(const LinearSpec& spec,
                                             const StridedLayout& in,
                                             const StridedLayout& out) {
  int64_t numSlots = out.numSlots;
  ShiftModulus modulus = getShiftModulus(in, out);
  // With a periodic layout the ciphertext of a term is not a function of the
  // output position alone. Such tensors fit one ciphertext, so the domain is
  // small.
  if (modulus.modulus != numSlots || modulus.writeFollowsInput) {
    return computeDiagonalKeysByEnumeration(spec, in, out);
  }

  std::vector<std::pair<int64_t, int64_t>> aligned =
      getAlignedLoops(spec, in, out);
  std::vector<bool> isAligned(spec.loops.size(), false);
  for (const auto& [loop, coeff] : aligned) isAligned[loop] = true;
  std::vector<int64_t> rest;
  for (size_t loop = 0; loop < spec.loops.size(); ++loop) {
    if (!isAligned[loop]) rest.push_back(loop);
  }
  for (int64_t extent : spec.loops) {
    if (extent <= 0) return {};
  }

  KeySet keys;
  size_t numAligned = aligned.size();
  std::vector<int64_t> lo(numAligned);
  std::vector<int64_t> hi(numAligned);
  std::vector<int64_t> d(spec.loops.size(), 0);

  forEachPoint(spec.loops, rest, d, [&]() {
    // Positions with the aligned loops at 0. An input index may be out of
    // bounds here; the position formula is affine on those axes.
    std::vector<int64_t> inIdx = applyMap(spec.inMap, spec.inOffset, d);
    std::vector<int64_t> outIdx = applyMap(spec.outMap, spec.outOffset, d);
    for (size_t k = 0; k < numAligned; ++k) {
      lo[k] = 0;
      hi[k] = spec.loops[aligned[k].first] - 1;
    }
    for (size_t axis = 0; axis < inIdx.size(); ++axis) {
      int64_t which = -1;
      for (size_t k = 0; k < numAligned; ++k) {
        if (spec.inMap[axis][aligned[k].first] != 0) which = k;
      }
      if (which < 0) {
        if (inIdx[axis] < 0 || inIdx[axis] >= spec.inShape[axis]) return;
        continue;
      }
      int64_t step = spec.inMap[axis][aligned[which].first];
      lo[which] = std::max(lo[which], ceilDiv(-inIdx[axis], step));
      hi[which] = std::min(
          hi[which], floorDiv(spec.inShape[axis] - 1 - inIdx[axis], step));
    }
    for (size_t k = 0; k < numAligned; ++k) {
      if (lo[k] > hi[k]) return;
    }
    int64_t outPos = out.position(outIdx);
    int64_t delta = in.position(inIdx) - outPos;
    int64_t shift = floorMod(delta, numSlots);
    auto emit = [&](int64_t p) {
      assert(p >= 0 && p + delta >= 0 && "positions must be non-negative");
      keys.insert({p / numSlots, (p + delta) / numSlots, shift});
    };
    if (numAligned == 0) {
      emit(outPos);
      return;
    }

    // Step the outer aligned loops; along the innermost one the output
    // position is an arithmetic progression, and the key changes only where
    // the output or the input position crosses a multiple of numSlots.
    int64_t step = aligned[numAligned - 1].second;
    int64_t count = hi[numAligned - 1] - lo[numAligned - 1] + 1;
    std::vector<int64_t> v(lo.begin(), lo.end() - 1);
    while (true) {
      int64_t start = outPos + step * lo[numAligned - 1];
      for (size_t k = 0; k + 1 < numAligned; ++k) {
        start += aligned[k].second * v[k];
      }
      int64_t last = start + (count - 1) * step;
      emit(start);
      for (int64_t base : {start, start + delta}) {
        for (int64_t multiple = floorDiv(base, numSlots) + 1;; ++multiple) {
          int64_t i = ceilDiv(multiple * numSlots - base, step);
          if (i > count - 1) break;
          emit(start + i * step);
        }
      }
      (void)last;
      int64_t k = static_cast<int64_t>(numAligned) - 2;
      while (k >= 0 && ++v[k] > hi[k]) {
        v[k] = lo[k];
        --k;
      }
      if (k < 0) break;
    }
  });
  return keys.sorted();
}

std::vector<std::vector<double>> packDiagonals(
    const LinearSpec& spec, const StridedLayout& in, const StridedLayout& out,
    ArrayRef<DiagonalKey> keys,
    llvm::function_ref<double(ArrayRef<int64_t>)> coeff) {
  std::map<DiagonalKey, size_t> rowOf;
  for (size_t row = 0; row < keys.size(); ++row) rowOf[keys[row]] = row;
  std::vector<std::vector<double>> diagonals(
      keys.size(), std::vector<double>(out.numSlots, 0.0));
  std::vector<int64_t> allLoops(spec.loops.size());
  for (size_t loop = 0; loop < allLoops.size(); ++loop) allLoops[loop] = loop;
  std::vector<int64_t> d(spec.loops.size(), 0);
  forEachPoint(spec.loops, allLoops, d, [&]() {
    auto placement = placeIteration(spec, in, out, d);
    if (!placement) return;
    auto it = rowOf.find(placement->key);
    assert(it != rowOf.end() && "iteration has a diagonal outside the key set");
    diagonals[it->second][placement->slot] += coeff(d);
  });
  return diagonals;
}

namespace {

struct FreeDigit {
  int64_t stride;
  int64_t radix;
};

StridedLayout inducedOutputLayoutImpl(const LinearSpec& spec,
                                      const StridedLayout& in,
                                      bool keepIdenticalAxes) {
  int64_t numSlots = in.numSlots;
  size_t numLoops = spec.loops.size();
  StridedLayout result;
  result.shape = spec.outShape;
  result.numSlots = numSlots;
  result.offset = in.offset;

  // (stride, extent) of the positions taken by output axes that stay on the
  // input's grid.
  std::vector<std::pair<int64_t, int64_t>> occupied;
  std::vector<int64_t> freeAxes;
  std::vector<bool> pairedInAxis(spec.inShape.size(), false);

  for (size_t axis = 0; axis < spec.outShape.size(); ++axis) {
    int64_t extent = spec.outShape[axis];
    if (extent == 1) continue;
    bool placed = false;
    // The output index must be a single loop variable.
    int64_t loop = -1;
    int64_t numVars = 0;
    for (size_t l = 0; l < numLoops; ++l) {
      if (spec.outMap[axis][l] != 0) {
        ++numVars;
        loop = l;
      }
    }
    if (numVars == 1 && spec.outMap[axis][loop] == 1 &&
        spec.outOffset[axis] == 0) {
      // ... that steps through exactly one input axis.
      int64_t inAxis = -1;
      int64_t numAxes = 0;
      for (size_t a = 0; a < spec.inMap.size(); ++a) {
        if (spec.inMap[a][loop] != 0) {
          ++numAxes;
          inAxis = a;
        }
      }
      if (numAxes == 1) {
        int64_t step = spec.inMap[inAxis][loop];
        std::optional<int64_t> stride = in.affineStride(inAxis);
        bool onlyVar = true;
        for (size_t l = 0; l < numLoops; ++l) {
          if (static_cast<int64_t>(l) != loop && spec.inMap[inAxis][l] != 0) {
            onlyVar = false;
          }
        }
        if (stride.has_value() && *stride > 0 && step > 0) {
          result.digits.push_back(
              {static_cast<int64_t>(axis), 1, 0, step * *stride});
          occupied.emplace_back(step * *stride, extent);
          pairedInAxis[inAxis] = true;
          placed = true;
        } else if (keepIdenticalAxes && step == 1 && onlyVar &&
                   spec.inOffset[inAxis] == 0) {
          // The same index on both sides: keep the input's digits.
          for (const StridedDigit& digit : in.digits) {
            if (digit.axis != inAxis) continue;
            result.digits.push_back({static_cast<int64_t>(axis), digit.div,
                                     digit.mod, digit.stride});
            occupied.emplace_back(
                digit.stride,
                digit.mod != 0 ? digit.mod : ceilDiv(extent, digit.div));
          }
          pairedInAxis[inAxis] = true;
          placed = true;
        }
      }
    }
    if (!placed) freeAxes.push_back(axis);
  }

  // Free positions between the occupied strides.
  std::sort(occupied.begin(), occupied.end());
  std::vector<FreeDigit> freeDigits;
  int64_t cursor = 1;
  for (const auto& [stride, extent] : occupied) {
    if (stride >= 2 * cursor) freeDigits.push_back({cursor, stride / cursor});
    cursor = std::max(cursor, stride * extent);
  }
  // New planes sit above everything occupied. Reuse the stride of the input's
  // planes when there is one, so that planes of both tensors line up.
  int64_t top = cursor;
  bool foundPlane = false;
  for (const StridedDigit& digit : in.digits) {
    if (pairedInAxis[digit.axis] || digit.stride < cursor) continue;
    if (!foundPlane || digit.stride < top) top = digit.stride;
    foundPlane = true;
  }

  size_t nextFree = 0;
  for (auto it = freeAxes.rbegin(); it != freeAxes.rend(); ++it) {
    int64_t axis = *it;
    int64_t extent = spec.outShape[axis];
    int64_t div = 1;
    while (nextFree < freeDigits.size() && div < extent) {
      const FreeDigit& digit = freeDigits[nextFree++];
      result.digits.push_back({axis, div, digit.radix, digit.stride});
      div *= digit.radix;
    }
    if (div < extent) {
      result.digits.push_back({axis, div, 0, top});
      top *= ceilDiv(extent, div);
    }
  }

  int64_t total = result.numElements();
  if (occupied.empty() && total <= numSlots) {
    int64_t period = 1;
    while (period < total) period *= 2;
    return StridedLayout::rowMajor(spec.outShape, numSlots, period);
  }
  if (!result.isInjective()) {
    return StridedLayout::rowMajor(spec.outShape, numSlots);
  }
  return result;
}

}  // namespace

StridedLayout inducedOutputLayout(const LinearSpec& spec,
                                  const StridedLayout& in) {
  StridedLayout inPlace = inducedOutputLayoutImpl(spec, in, true);
  if (inPlace.period != 0 ||
      inPlace.numElements() >= (inPlace.maxPosition() + 1) / 2) {
    return inPlace;
  }
  // Keeping axes in place leaves the result sparse: let them fill the gaps.
  StridedLayout compact = inducedOutputLayoutImpl(spec, in, false);
  if (compact.numCiphertexts() < inPlace.numCiphertexts() ||
      inPlace.numCiphertexts() == 1) {
    return compact;
  }
  return inPlace;
}

}  // namespace heir
}  // namespace mlir
