#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>

#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/StridedLayout.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"        // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"         // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"           // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

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

Attribute StridedLayoutAttr::parse(AsmParser& parser, Type type) {
  SmallVector<int64_t> shape;
  SmallVector<int64_t> digits;
  int64_t slots = 0;
  int64_t offset = 0;
  int64_t period = 0;
  SMLoc loc = parser.getCurrentLocation();

  auto parseDigit = [&]() -> ParseResult {
    size_t before = digits.size();
    SMLoc digitLoc = parser.getCurrentLocation();
    if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, [&]() {
          return parser.parseInteger(digits.emplace_back());
        }))
      return failure();
    if (digits.size() - before != 4) {
      return parser.emitError(digitLoc,
                              "expected a digit (axis, div, mod, stride)");
    }
    return success();
  };

  if (parser.parseLess() || parser.parseKeyword("shape") ||
      parser.parseEqual() ||
      parser.parseCommaSeparatedList(
          AsmParser::Delimiter::Square,
          [&]() { return parser.parseInteger(shape.emplace_back()); }) ||
      parser.parseComma() || parser.parseKeyword("slots") ||
      parser.parseEqual() || parser.parseInteger(slots) ||
      parser.parseComma() || parser.parseKeyword("digits") ||
      parser.parseEqual() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseDigit))
    return {};

  while (succeeded(parser.parseOptionalComma())) {
    StringRef keyword;
    SMLoc keywordLoc = parser.getCurrentLocation();
    if (parser.parseKeyword(&keyword) || parser.parseEqual()) return {};
    if (keyword == "offset") {
      if (parser.parseInteger(offset)) return {};
    } else if (keyword == "period") {
      if (parser.parseInteger(period)) return {};
    } else {
      parser.emitError(keywordLoc, "expected 'offset' or 'period'");
      return {};
    }
  }
  if (parser.parseGreater()) return {};

  return parser.getChecked<StridedLayoutAttr>(
      loc, parser.getContext(), ArrayRef<int64_t>(shape),
      ArrayRef<int64_t>(digits), slots, offset, period);
}

void StridedLayoutAttr::print(AsmPrinter& printer) const {
  printer << "<shape = [";
  llvm::interleaveComma(getShape(), printer);
  printer << "], slots = " << getSlots() << ", digits = [";
  ArrayRef<int64_t> digits = getDigits();
  for (size_t i = 0; i + 3 < digits.size(); i += 4) {
    if (i != 0) printer << ", ";
    printer << "(" << digits[i] << ", " << digits[i + 1] << ", "
            << digits[i + 2] << ", " << digits[i + 3] << ")";
  }
  printer << "]";
  if (getOffset() != 0) printer << ", offset = " << getOffset();
  if (getPeriod() != 0) printer << ", period = " << getPeriod();
  printer << ">";
}

LogicalResult StridedLayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape,
    ArrayRef<int64_t> digits, int64_t slots, int64_t offset, int64_t period) {
  if (slots <= 0) return emitError() << "slots must be positive";
  if (offset < 0) return emitError() << "offset must be nonnegative";
  for (int64_t extent : shape) {
    if (extent <= 0) return emitError() << "shape extents must be positive";
  }
  if (digits.size() % 4 != 0) {
    return emitError() << "digits must have four entries per digit";
  }
  for (size_t i = 0; i < digits.size(); i += 4) {
    int64_t axis = digits[i];
    if (axis < 0 || axis >= static_cast<int64_t>(shape.size())) {
      return emitError() << "digit axis " << axis << " is out of bounds for "
                         << shape.size() << " axes";
    }
    if (digits[i + 1] <= 0) return emitError() << "digit div must be positive";
    if (digits[i + 2] < 0) {
      return emitError() << "digit mod must be nonnegative";
    }
    if (digits[i + 3] < 0) {
      return emitError() << "digit stride must be nonnegative";
    }
  }
  if (period < 0 || (period != 0 && slots % period != 0)) {
    return emitError() << "period must divide slots";
  }
  if (period != 0) {
    StridedLayout layout;
    layout.shape.assign(shape.begin(), shape.end());
    for (size_t i = 0; i < digits.size(); i += 4) {
      layout.digits.push_back(
          {digits[i], digits[i + 1], digits[i + 2], digits[i + 3]});
    }
    layout.numSlots = slots;
    if (layout.maxPosition() >= period) {
      return emitError() << "a periodic layout must fit within one period";
    }
  }
  return success();
}

StridedLayout StridedLayoutAttr::getStridedLayout() const {
  StridedLayout layout;
  layout.shape.assign(getShape().begin(), getShape().end());
  ArrayRef<int64_t> digits = getDigits();
  for (size_t i = 0; i < digits.size(); i += 4) {
    layout.digits.push_back(
        {digits[i], digits[i + 1], digits[i + 2], digits[i + 3]});
  }
  layout.numSlots = getSlots();
  layout.offset = getOffset();
  layout.period = getPeriod();
  return layout;
}

StridedLayoutAttr StridedLayoutAttr::getFromStridedLayout(
    ::mlir::MLIRContext* context, const StridedLayout& layout) {
  // Equal layouts must give the same attribute, so the digits are stored in
  // their merged form and in a fixed order.
  StridedLayout canonical = layout;
  canonical.normalize();
  llvm::sort(canonical.digits,
             [](const StridedDigit& a, const StridedDigit& b) {
               return std::tie(a.axis, a.div, a.mod, a.stride) <
                      std::tie(b.axis, b.div, b.mod, b.stride);
             });
  SmallVector<int64_t> digits;
  for (const StridedDigit& digit : canonical.digits) {
    digits.append({digit.axis, digit.div, digit.mod, digit.stride});
  }
  return StridedLayoutAttr::get(context, layout.shape, digits, layout.numSlots,
                                layout.offset, layout.period);
}

IntegerRelation StridedLayoutAttr::getIntegerRelation() const {
  auto result = getIntegerRelationFromIslStr(getStridedLayout().toIslString());
  assert(succeeded(result) && "Failed to convert the strided layout");
  return result.value();
}

std::optional<IntegerRelation> getLayoutRelation(Attribute attr) {
  if (auto layout = dyn_cast_if_present<LayoutAttr>(attr)) {
    return layout.getIntegerRelation();
  }
  if (auto strided = dyn_cast_if_present<StridedLayoutAttr>(attr)) {
    return strided.getIntegerRelation();
  }
  return std::nullopt;
}

bool usesStridedLayouts(Operation* module) {
  bool found = false;
  module->walk([&](Operation* op) {
    if (found) return WalkResult::interrupt();
    auto func = dyn_cast<FunctionOpInterface>(op);
    if (!func) return WalkResult::advance();
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      if (isa_and_nonnull<StridedLayoutAttr>(
              func.getArgAttr(i, TensorExtDialect::kLayoutAttrName))) {
        found = true;
      }
    }
    return WalkResult::advance();
  });
  return found;
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
