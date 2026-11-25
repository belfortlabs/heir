#ifndef LIB_UTILS_LAYOUT_LINEARSPEC_H_
#define LIB_UTILS_LAYOUT_LINEARSPEC_H_

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "lib/Utils/Layout/StridedLayout.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project

namespace mlir {
namespace heir {

/// An operation that is linear in one tensor:
///
///   out[outMap * d + outOffset] += coeff(d) * in[inMap * d + inOffset]
///
/// for every point `d` of the box `[0, loops[0]) x ...`. Reads outside the
/// input's shape are zero, which is how zero padding is expressed. The maps
/// are integer matrices with one row per tensor axis and one column per loop.
///
/// Convolutions of every kind, sum pools, matrix-vector products with a
/// cleartext matrix, broadcasts and pure moves all have this form; they differ
/// only in the maps and in `coeff`.
struct LinearSpec {
  std::vector<int64_t> loops;
  std::vector<int64_t> inShape;
  std::vector<std::vector<int64_t>> inMap;
  std::vector<int64_t> inOffset;
  std::vector<int64_t> outShape;
  std::vector<std::vector<int64_t>> outMap;
  std::vector<int64_t> outOffset;

  int64_t numIterations() const;
};

/// One generalized diagonal of the transform between two laid-out tensors:
/// output ciphertext `ctOut` receives `diagonal * rotateLeft(input[ctIn],
/// shift)`.
struct DiagonalKey {
  int64_t ctOut;
  int64_t ctIn;
  int64_t shift;

  bool operator==(const DiagonalKey& other) const {
    return std::tie(ctOut, ctIn, shift) ==
           std::tie(other.ctOut, other.ctIn, other.shift);
  }
  bool operator<(const DiagonalKey& other) const {
    return std::tie(ctOut, ctIn, shift) <
           std::tie(other.ctOut, other.ctIn, other.shift);
  }
};

/// Where the term of one iteration goes: slot `slot` of the diagonal `key`.
struct DiagonalPlacement {
  DiagonalKey key;
  int64_t slot;
};

/// How shifts are reduced for a pair of layouts. With `writeFollowsInput`, a
/// term is written to the copy of the output element that sits under the input
/// element, all terms go to output ciphertext 0, and the caller must sum the
/// `numSlots / modulus` copies of the result.
struct ShiftRule {
  int64_t modulus;
  bool writeFollowsInput;
};

ShiftRule getShiftRule(const StridedLayout& in, const StridedLayout& out);

/// The placement of iteration `d`, or nullopt if it reads outside the input.
///
/// If the output layout is periodic, shifts are reduced modulo its period and
/// each term is written to the output copy that sits under the input element;
/// the caller then sums the copies of the result (a rotate-and-reduce over the
/// period). If only the input layout is periodic, shifts are reduced modulo
/// the input's period and the term reads the input copy above the output
/// slot.
std::optional<DiagonalPlacement> placeIteration(const LinearSpec& spec,
                                                const StridedLayout& in,
                                                const StridedLayout& out,
                                                llvm::ArrayRef<int64_t> d);

/// The sorted set of diagonals of the transform.
///
/// The shift does not depend on loops whose position coefficients agree in the
/// two layouts (the spatial loops of a convolution between layouts on the same
/// grid). Those loops are not enumerated; the ciphertext boundaries they cross
/// are found arithmetically. The cost is the number of points of the
/// remaining loops, times the extents of all but one of the skipped loops.
std::vector<DiagonalKey> computeDiagonalKeys(const LinearSpec& spec,
                                             const StridedLayout& in,
                                             const StridedLayout& out);

/// The same set, found by visiting every iteration. For tests.
std::vector<DiagonalKey> computeDiagonalKeysByEnumeration(
    const LinearSpec& spec, const StridedLayout& in, const StridedLayout& out);

/// The diagonals as dense rows of `numSlots` values, in the order of `keys`.
/// This is the reference for the packing loop that is emitted into the IR.
std::vector<std::vector<double>> packDiagonals(
    const LinearSpec& spec, const StridedLayout& in, const StridedLayout& out,
    llvm::ArrayRef<DiagonalKey> keys,
    llvm::function_ref<double(llvm::ArrayRef<int64_t>)> coeff);

/// A unit-coefficient spec split into a sum over a box of rotations and the
/// rest of the operation.
struct BoxSumFactor {
  /// One `(stride, count)` per summed loop. The sum over the loop is the sum
  /// of the `count` left rotations of the input by multiples of `stride`.
  std::vector<std::pair<int64_t, int64_t>> axes;
  /// The spec with the summed loops fixed at zero. Applied to the summed
  /// input, it gives the result of the original spec.
  LinearSpec rest;
};

/// Splits off the loops of `spec` that only sum input elements: loops that do
/// not occur in the output index and along which the input position grows by
/// a constant stride. Every coefficient of `spec` must be one. Returns nullopt
/// if there is no such loop, if the spec reads outside the input, or if a
/// summed window crosses a ciphertext boundary; in the last two cases a
/// rotation would bring in a slot that is not the element the spec reads.
///
/// The summed input holds a window sum at the position of each window's
/// first element and partial sums elsewhere.
std::optional<BoxSumFactor> factorBoxSum(const LinearSpec& spec,
                                         const StridedLayout& in);

/// The mirror image of `factorBoxSum`: splits off the loops of `spec` that
/// only copy an input element to several output positions, which are loops
/// that do not occur in the input index and along which the output position
/// grows by a constant stride. `rest` writes the first copy of each element;
/// the sum of the `count` right rotations of its result by multiples of
/// `stride` completes the copies, provided the result of `rest` is zero in
/// every other slot. Returns nullopt if there is no such loop or if the
/// copies of an element cross a ciphertext boundary.
std::optional<BoxSumFactor> factorBoxCopy(const LinearSpec& spec,
                                          const StridedLayout& out);

/// How a spec is evaluated between two layouts: sums of rotations of the
/// input, then generalized diagonals, then sums of rotations of the result.
struct TransformPlan {
  /// `(stride, count)` of the left-rotation sums applied to the input.
  std::vector<std::pair<int64_t, int64_t>> sums;
  /// The spec that the diagonals evaluate, and their keys.
  LinearSpec spec;
  std::vector<DiagonalKey> keys;
  /// True if the diagonals would only select slots of the summed input in
  /// place and the caller allows other values in the slots that hold no
  /// element. They are then skipped: the result is the summed input, whose
  /// unused slots hold partial sums.
  bool skipDiagonals = false;
  /// `(stride, count)` of the right-rotation sums applied to the result.
  std::vector<std::pair<int64_t, int64_t>> copies;

  /// The number of plaintext multiplications of the plan.
  int64_t numDiagonals() const {
    return skipDiagonals ? 0 : static_cast<int64_t>(keys.size());
  }
};

/// Plans the evaluation of `spec`. Sums and copies are split off only if
/// every coefficient is one and neither layout is periodic.
///
/// Unused slots of a tensor are zero unless `allowUnusedSlotValues` is set.
/// Set it only if every reader of the result ignores them, as linear
/// transforms do. Values in unused slots are not bounded by the range of the
/// data, so a polynomial approximation applied to them can overflow.
TransformPlan planTransform(const LinearSpec& spec, bool unitCoefficients,
                            const StridedLayout& in, const StridedLayout& out,
                            bool allowUnusedSlotValues);

/// A layout for the result that keeps the transform's diagonal count low.
///
/// Output axes that step through an input axis stay on the input's grid, with
/// the input stride multiplied by the step. The other output axes are placed
/// in the positions left free between those strides, and then in new planes
/// above them. If keeping an axis in place would leave the result less than
/// half full, it is placed in free positions instead. A result with no axis on
/// the input's grid that fits one ciphertext is row-major and periodic.
StridedLayout inducedOutputLayout(const LinearSpec& spec,
                                  const StridedLayout& in);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_LINEARSPEC_H_
