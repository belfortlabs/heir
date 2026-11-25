#ifndef LIB_UTILS_LAYOUT_LINEARSPEC_H_
#define LIB_UTILS_LAYOUT_LINEARSPEC_H_

#include <cstdint>
#include <optional>
#include <tuple>
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
