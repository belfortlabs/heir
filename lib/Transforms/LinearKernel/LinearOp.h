#ifndef LIB_TRANSFORMS_LINEARKERNEL_LINEAROP_H_
#define LIB_TRANSFORMS_LINEARKERNEL_LINEAROP_H_

#include <functional>
#include <optional>

#include "lib/Utils/Layout/LinearSpec.h"
#include "lib/Utils/Layout/StridedLayout.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project

namespace mlir {
namespace heir {

/// An operation read as a LinearSpec over one of its tensor operands.
struct LinearOp {
  LinearSpec spec;

  /// The tensor the spec reads. This is the operand itself, or, when the
  /// operand is a zero `tensor.pad` (possibly behind reshapes that
  /// `isViewReshape` accepts), the tensor that was padded. The padding is then
  /// part of the spec: reads outside `source` are zero.
  Value source;

  /// The value the result is added to, or null when there is none or it is
  /// known to be zero.
  Value init;

  /// Emits the coefficient of one iteration, given the loop indices as
  /// `index` values. Null when every coefficient is one.
  std::function<Value(OpBuilder&, Location, ValueRange)> coefficient;
};

/// Reads `op` as a linear operation in operand `operandNumber`. Every other
/// operand is treated as cleartext.
///
/// Accepted are structured linalg operations with static shapes whose body is
/// `out + f(x)` or `f(x)`, where `f` is built from the operand's element by
/// multiplication with and division by cleartext values, negation, and sums
/// of such terms; `tensor.pad` with a zero padding value; and
/// `tensor.collapse_shape`, `tensor.expand_shape` and `tensor.extract_slice`
/// with static shapes, which move every element unchanged.
///
/// `tensor.insert_slice` is linear in both operands, and its result is the
/// sum of the two readings: operand 0 is copied into the box, and operand 1
/// is copied everywhere else, which is an identity with a 0/1 coefficient.
std::optional<LinearOp> matchLinearOp(Operation* op, unsigned operandNumber);

/// The layout of the result of an operation that only renames or selects
/// elements of its first operand (a reshape, a slice, a transpose or any
/// other linalg operation that permutes axes), such that no element has to
/// move: the result is the operand's ciphertexts read through the returned
/// layout. Nullopt if the operation is of another kind or the layout of the
/// result has no strided form.
std::optional<StridedLayout> relabeledLayout(Operation* op,
                                             const StridedLayout& in);

/// The element-for-element copy of a tensor of the given shape. This is the
/// spec of a layout conversion.
LinearOp identityLinearOp(Value source);

/// Follows `value` back through reshapes that `isViewReshape` accepts to a
/// zero `tensor.pad`. Returns the pad if there is one.
Operation* findFoldablePad(Value value);

/// True if the indices of a reshape's operand are integer combinations of the
/// indices of its result, so that a linear operation reading the result reads
/// the operand through an affine map: any `tensor.expand_shape`, and a
/// `tensor.collapse_shape` that only removes unit dimensions.
bool isViewReshape(Operation* op);

/// True if a `tensor.pad` has static bounds and a zero padding value.
bool isStaticZeroPad(Operation* op);

/// True if a cleartext tensor is zero or has no defined contents: a zero
/// constant, a `tensor.empty`, or a `linalg.fill` with zero.
bool isZeroTensor(Value value);

/// True if the tensor a `tensor.insert_slice` inserts into is zero at every
/// position of the box: it is a zero tensor, or a chain of earlier insertions
/// into one whose boxes are disjoint from this one.
bool isZeroWhereInserted(tensor::InsertSliceOp insert);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LINEARKERNEL_LINEAROP_H_
