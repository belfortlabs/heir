#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"

#include <cassert>
#include <cstdlib>
#include <string>

#include "lib/Utils/Layout/IslConversion.h"
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"        // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"         // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

// ISL
#include "include/isl/ctx.h"         // from @isl
#include "include/isl/map.h"         // from @isl
#include "include/isl/map_type.h"    // from @isl
#include "include/isl/space_type.h"  // from @isl

namespace mlir {
namespace heir {
namespace tensor_ext {

using presburger::IntegerPolyhedron;
using presburger::IntegerRelation;
using presburger::VarKind;

LogicalResult LayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 StringAttr layoutStr) {
  auto result = getIntegerRelationFromIslStr(layoutStr.getValue().str());
  if (failed(result)) {
    return emitError() << "Failed to parse the layout string (ISL): "
                       << layoutStr;
  }
  // Success if you can parse the ISL string and convert it.
  return success();
}

LogicalResult KernelInfoAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI64ArrayAttr resultShape,
    int64_t gapFactor) {
  if (!resultShape) {
    return emitError() << "expected a resultShape";
  }
  // A gap of 1 means no multiplexing; 0 or less describes no packing at all.
  if (gapFactor < 1) {
    return emitError() << "expected a gapFactor of at least 1, got "
                       << gapFactor;
  }
  return success();
}

LogicalResult ConvPackingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    RankedTensorType matrixDataType, int64_t padding, bool interchangeRows,
    int64_t absorbedMatrixWidth) {
  // The conv kernels pack a rank-3 (N, C, W) or rank-4 (N, C, H, W) operand,
  // and read the spatial extents by index, so no other rank is meaningful.
  if (matrixDataType.getRank() != 3 && matrixDataType.getRank() != 4) {
    return emitError() << "expected a rank-3 or rank-4 matrixDataType, got "
                       << matrixDataType;
  }
  if (padding < 0) {
    return emitError() << "expected a non-negative padding, got " << padding;
  }
  if (absorbedMatrixWidth < 0) {
    return emitError() << "expected a non-negative absorbedMatrixWidth, got "
                       << absorbedMatrixWidth;
  }
  return success();
}

IntegerRelation LayoutAttr::getIntegerRelation() const {
  auto result = getIntegerRelationFromIslStr(getLayoutStr());
  assert(succeeded(result) && "Failed to parse the layout string");
  return result.value();
}

LayoutAttr LayoutAttr::getFromIntegerRelation(::mlir::MLIRContext* context,
                                              const IntegerRelation& relation) {
  isl_ctx* ctx = isl_ctx_alloc();
  isl_basic_map* bmap = convertRelationToBasicMap(relation, ctx);

  bmap = isl_basic_map_set_dim_name(bmap, isl_dim_out, 0, "ct");
  bmap = isl_basic_map_set_dim_name(bmap, isl_dim_out, 1, "slot");

  char* resultStr = isl_basic_map_to_str(bmap);
  std::string layoutStr(resultStr);
  free(resultStr);
  isl_basic_map_free(bmap);
  isl_ctx_free(ctx);
  return LayoutAttr::get(context, layoutStr);
}

LayoutAttr LayoutAttr::composeLayouts(::mlir::ArrayAttr arrayAttr,
                                      ::mlir::MLIRContext* context) {
  auto layoutAttrs = arrayAttr.getAsRange<LayoutAttr>();
  auto it = layoutAttrs.begin();
  IntegerRelation composedRel = (*it).getIntegerRelation();
  ++it;
  for (; it != layoutAttrs.end(); ++it) {
    composedRel.compose((*it).getIntegerRelation());
  }
  return LayoutAttr::getFromIntegerRelation(context, composedRel);
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
