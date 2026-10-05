#include "lib/Utils/MaxPoolUtils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "llvm/include/llvm/ADT/APFloat.h"               // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {

bool isNegativeInfinity(Value value) {
  APFloat constant(0.0);
  return value && matchPattern(value, m_ConstantFloat(&constant)) &&
         constant.isInfinity() && constant.isNegative();
}

bool isNegativeInfinityTensor(Value init) {
  if (auto fillOp = init.getDefiningOp<linalg::FillOp>())
    return isNegativeInfinity(fillOp.getInputs()[0]);
  SplatElementsAttr splat;
  if (!matchPattern(init, m_Constant(&splat))) return false;
  auto floatAttr = dyn_cast<FloatAttr>(splat.getSplatValue<Attribute>());
  return floatAttr && floatAttr.getValue().isInfinity() &&
         floatAttr.getValue().isNegative();
}

FailureOr<SupportedMaxPool> getSupportedMaxPool(linalg::PoolingNcwMaxOp op,
                                                std::string& reason) {
  auto inputType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
  auto windowType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
  auto outputType = dyn_cast<RankedTensorType>(op.getOutputs()[0].getType());
  if (!inputType || !windowType || !outputType || !inputType.hasStaticShape() ||
      !windowType.hasStaticShape() || !outputType.hasStaticShape()) {
    reason = "requires static shapes";
    return failure();
  }
  if (inputType.getDimSize(0) != 1) {
    reason = "requires a batch size of 1";
    return failure();
  }

  int64_t windowSize = windowType.getDimSize(0);
  int64_t stride = op.getStrides().getValues<int64_t>()[0];
  int64_t dilation = op.getDilations().getValues<int64_t>()[0];
  if (stride != windowSize || dilation != 1) {
    reason = "requires a stride equal to the window size and a dilation of 1";
    return failure();
  }

  int64_t length = inputType.getDimSize(2);
  int64_t outputLength = outputType.getDimSize(2);
  if (length != outputLength * windowSize) {
    reason =
        "requires an input length that the windows tile exactly; run "
        "--linalg-canonicalizations to pad or trim the input";
    return failure();
  }

  if (!isNegativeInfinityTensor(op.getOutputs()[0])) {
    reason = "requires a -inf init";
    return failure();
  }

  auto lowerAttr = op->getAttrOfType<FloatAttr>("domain_lower");
  auto upperAttr = op->getAttrOfType<FloatAttr>("domain_upper");
  if (!lowerAttr || !upperAttr) {
    reason = "requires float domain_lower/domain_upper attributes";
    return failure();
  }
  double lower = lowerAttr.getValueAsDouble();
  double upper = upperAttr.getValueAsDouble();
  double valueBound = std::max(std::abs(lower), std::abs(upper));
  if (!(upper > lower) || !std::isfinite(valueBound)) {
    reason = "requires a finite domain with domain_lower < domain_upper";
    return failure();
  }

  return SupportedMaxPool{inputType.getDimSize(1), length, outputLength,
                          windowSize, valueBound};
}

}  // namespace heir
}  // namespace mlir
