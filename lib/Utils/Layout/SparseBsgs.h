#ifndef LIB_UTILS_LAYOUT_SPARSEBSGS_H_
#define LIB_UTILS_LAYOUT_SPARSEBSGS_H_

#include <cstdint>
#include <vector>

#include "lib/Utils/Layout/LinearSpec.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"  // from @llvm-project

namespace mlir {
namespace heir {

/// A baby-step giant-step evaluation order for a set of diagonals
///
///   y[ctOut] = sum over keys of diagonal * rotateLeft(x[ctIn], shift).
///
/// Each shift is split as `giant + baby` with `baby = shift mod giantStep`.
/// The input ciphertexts are rotated once per distinct `(ctIn, baby)`. Each
/// diagonal is stored rotated right by its giant part, the products of one
/// `(ctOut, giant)` group are summed, and the sum is rotated left by `giant`:
///
///   baby[t]  = rotateLeft(x[babyCt[t]], babyShift[t])
///   part     = sum over the rows r of a group of
///                rotateRight(diagonal[row[r]], giant[r]) * baby[babyOf[r]]
///   y[out[r]] += rotateLeft(part, giant[r])      at the group's last row
///
/// Rows are ordered by `(ctOut, giant)`, so the groups are contiguous.
struct SparseBsgsPlan {
  int64_t giantStep = 1;

  // One entry per distinct (input ciphertext, baby shift).
  std::vector<int64_t> babyCt;
  std::vector<int64_t> babyShift;

  // One entry per diagonal, in evaluation order.
  std::vector<int64_t> row;     // index into the key list
  std::vector<int64_t> babyOf;  // index into the baby steps
  std::vector<int64_t> giant;   // giant part of the shift
  std::vector<int64_t> out;     // output ciphertext
  std::vector<bool> isLast;     // last row of its (out, giant) group

  /// Ciphertext rotations the plan executes.
  int64_t numRotations() const;
};

/// The plan with the fewest ciphertext rotations among power-of-two giant
/// steps up to `numSlots`; ties go to the plan with fewer groups.
SparseBsgsPlan planSparseBsgs(llvm::ArrayRef<DiagonalKey> keys,
                              int64_t numSlots);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_SPARSEBSGS_H_
