#ifndef LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_STRIDEDPATTERNS_H_
#define LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_STRIDEDPATTERNS_H_

#include <cstdint>

#include "lib/Utils/ContextAwareTypeConversion.h"
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"     // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project

namespace mlir {
namespace heir {

/// Marks an operation as already being in ciphertext semantics.
void markStridedConverted(Operation* op);

/// Conversion patterns for values in strided layouts:
///
/// - A linalg operation that is linear in its secret operand, a zero
///   `tensor.pad`, a reshape or a slice becomes a loop nest that packs the
///   operation's diagonals followed by one `kernel.block_linear_transform`.
///   When the result's layout is the operand's layout seen through the
///   operation, it is replaced by its operand instead.
/// - `tensor.insert_slice` is such a transform of the inserted tensor, added
///   to the tensor inserted into.
/// - `tensor_ext.convert_layout` is a transform with the identity as the
///   operation.
///
/// A transform with more than `maxDiagonals` diagonals is an error.
void populateStridedLayoutPatterns(RewritePatternSet& patterns,
                                   const ContextAwareTypeConverter& converter,
                                   MLIRContext* context, int64_t maxDiagonals);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_CONVERTTOCIPHERTEXTSEMANTICS_STRIDEDPATTERNS_H_
