#ifndef LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_H_
#define LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_H_

#include <optional>

// IWYU pragma: begin_keep
#include "lib/Utils/Layout/StridedLayout.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"   // from @llvm-project
#include "mlir/include/mlir/IR/TensorEncoding.h"     // from @llvm-project
// IWYU pragma: end_keep

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h.inc"

namespace mlir {
namespace heir {
namespace tensor_ext {

/// The relation of a layout attribute of either form (`tensor_ext.layout` or
/// `tensor_ext.strided_layout`), or nullopt if `attr` is neither.
std::optional<presburger::IntegerRelation> getLayoutRelation(Attribute attr);

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_H_
