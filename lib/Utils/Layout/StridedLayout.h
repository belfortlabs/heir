#ifndef LIB_UTILS_LAYOUT_STRIDEDLAYOUT_H_
#define LIB_UTILS_LAYOUT_STRIDEDLAYOUT_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "llvm/include/llvm/ADT/ArrayRef.h"  // from @llvm-project

namespace mlir {
namespace heir {

/// One term of a strided layout's position: `stride * ((idx[axis] / div) mod
/// mod)`. A `mod` of 0 means no reduction, so a digit with `div == 1` and
/// `mod == 0` makes the position affine in that axis.
struct StridedDigit {
  int64_t axis;
  int64_t div;
  int64_t mod;
  int64_t stride;

  bool operator==(const StridedDigit& other) const {
    return axis == other.axis && div == other.div && mod == other.mod &&
           stride == other.stride;
  }
};

/// A layout that places the element at index `idx` of a tensor at the global
/// position
///
///   p(idx) = offset + sum over digits of stride * ((idx[axis] / div) mod mod)
///
/// which is slot `p mod numSlots` of ciphertext `p / numSlots`. With a nonzero
/// `period`, the tensor occupies one ciphertext and the element is also at
/// every position `p + m * period`; `period` must divide `numSlots`.
///
/// Slots that hold no element are unspecified, but they hold values in the
/// range of the data: zeros, what slotwise operations make of zeros, or
/// elements of a tensor this one was cut from. A transform that would leave
/// anything else there is applied only where every reader ignores those slots.
///
/// This family contains row-major layouts, channel-multiplexed (gapped)
/// layouts of strided convolution chains, and cyclically replicated vectors.
/// The position of an element is evaluated in constant time, without
/// enumerating a relation.
struct StridedLayout {
  std::vector<int64_t> shape;
  std::vector<StridedDigit> digits;
  int64_t numSlots = 0;
  int64_t offset = 0;
  int64_t period = 0;

  /// The global position of the element at `idx`. Indices outside the shape
  /// are allowed and extend the formula.
  int64_t position(llvm::ArrayRef<int64_t> idx) const;

  /// The largest position of any element (of the first copy, if periodic).
  int64_t maxPosition() const;

  int64_t numCiphertexts() const { return maxPosition() / numSlots + 1; }

  int64_t numElements() const;

  /// The stride of `axis` if the position is affine in its index, 0 if the
  /// position does not depend on it, and nullopt otherwise.
  std::optional<int64_t> affineStride(int64_t axis) const;

  /// Whether distinct elements have distinct positions (modulo the period, if
  /// any). Visits every element.
  bool isInjective() const;

  /// The layout as an isl basic map `{ [i0, ...] -> [ct, slot] : ... }`, the
  /// form stored by `tensor_ext.layout`.
  std::string toIslString() const;

  /// Digits are compared as a set.
  bool operator==(const StridedLayout& other) const;

  /// Rewrites the digits to an equivalent shorter form: chains of digits that
  /// together read a contiguous range of an index are merged, reductions that
  /// never apply within the shape are dropped, and so are digits that are
  /// always zero within the shape. Positions of indices inside the shape are
  /// unchanged.
  void normalize();

  // The transforms below give the layout of a view of the same data, so that
  // layout-only operations cost nothing: the element at index `idx` of the
  // result sits where the corresponding element of this tensor sits. Those
  // that can leave the family return nullopt when they do.

  /// The layout of the tensor with axes permuted: axis `i` of the result is
  /// axis `perm[i]` of this tensor.
  StridedLayout transpose(llvm::ArrayRef<int64_t> perm) const;

  /// The layout after `tensor.collapse_shape`: each group of consecutive axes
  /// becomes one row-major axis.
  std::optional<StridedLayout> collapse(
      llvm::ArrayRef<std::vector<int64_t>> reassociation) const;

  /// The layout after `tensor.expand_shape`: axis `g` of this tensor becomes
  /// the axes `reassociation[g]` of the result, which has shape
  /// `expandedShape`.
  std::optional<StridedLayout> expand(
      llvm::ArrayRef<std::vector<int64_t>> reassociation,
      llvm::ArrayRef<int64_t> expandedShape) const;

  /// The layout of the slice whose element `i` is element
  /// `offsets + steps * i` of this tensor, with shape `sizes`.
  std::optional<StridedLayout> slice(llvm::ArrayRef<int64_t> offsets,
                                     llvm::ArrayRef<int64_t> sizes,
                                     llvm::ArrayRef<int64_t> steps) const;

  /// Row-major order, with stride 1 on the last axis.
  static StridedLayout rowMajor(llvm::ArrayRef<int64_t> shape, int64_t numSlots,
                                int64_t period = 0);

  /// Row-major, except that a tensor that needs several ciphertexts gets a
  /// power of two as the stride of its outermost axis of extent above one.
  /// The slices along that axis (the channel planes of an image tensor) then
  /// start at the same slot in every ciphertext, or at a ciphertext boundary,
  /// so the shift between two planes takes few values modulo the slot count.
  static StridedLayout aligned(llvm::ArrayRef<int64_t> shape, int64_t numSlots);
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_STRIDEDLAYOUT_H_
