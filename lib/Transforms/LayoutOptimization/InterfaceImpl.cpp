#include "lib/Transforms/LayoutOptimization/InterfaceImpl.h"

#include <cassert>
#include <cstdint>
#include <vector>

#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Interface/HoistingInterfaces.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Transforms/LayoutOptimization/Hoisting.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Layout/Hoisting.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

#define DEBUG_TYPE "interface-impl"

namespace mlir {
namespace heir {

using tensor_ext::ConvertLayoutOp;
using tensor_ext::LayoutAttr;
static auto& kLayoutAttrName = tensor_ext::TensorExtDialect::kLayoutAttrName;
using presburger::IntegerRelation;
using presburger::PresburgerSpace;
using presburger::VarKind;

namespace {

template <typename OpTy>
struct DoNothingHoistingImpl
    : public LayoutConversionHoistableOpInterface::ExternalModel<
          DoNothingHoistingImpl<OpTy>, OpTy> {
  std::vector<::mlir::heir::Hoister> getHoisters(
      Operation* op, tensor_ext::ConvertLayoutOp convertLayoutOp) const {
    return {createTrivialHoister(op)};
  }
};

struct CollapseShapeHoistingImpl
    : public LayoutConversionHoistableOpInterface::ExternalModel<
          CollapseShapeHoistingImpl, tensor::CollapseShapeOp> {
  std::vector<Hoister> getHoisters(
      Operation* op, tensor_ext::ConvertLayoutOp convertLayoutOp) const {
    std::vector<Hoister> hoisters;
    tensor::CollapseShapeOp collapseShapeOp = cast<tensor::CollapseShapeOp>(op);

    // Hoist layout conversion through trivial rank-reducing collapse_shape
    // operations.

    if (!op->hasAttr(tensor_ext::TensorExtDialect::kLayoutAttrName)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Layout attribute not found on op " << *op << "\n");
      return hoisters;
    }

    hoisters.push_back(createCollapseShapeHoister(collapseShapeOp));

    return hoisters;
  }
};

struct PadHoistingImpl
    : public LayoutConversionHoistableOpInterface::ExternalModel<
          PadHoistingImpl, tensor::PadOp> {
  std::vector<Hoister> getHoisters(
      Operation* op, tensor_ext::ConvertLayoutOp convertLayoutOp) const {
    std::vector<Hoister> hoisters;
    tensor::PadOp padOp = cast<tensor::PadOp>(op);

    for (int64_t val : padOp.getStaticLow()) {
      if (ShapedType::isDynamic(val)) {
        return hoisters;
      }
    }
    for (int64_t val : padOp.getStaticHigh()) {
      if (ShapedType::isDynamic(val)) {
        return hoisters;
      }
    }

    hoisters.push_back(createPadHoister(padOp));
    return hoisters;
  }
};

struct MatmulHoistingImpl
    : public LayoutConversionHoistableOpInterface::ExternalModel<
          MatmulHoistingImpl, linalg::MatmulOp> {
  std::vector<Hoister> getHoisters(
      Operation* op, tensor_ext::ConvertLayoutOp convertLayoutOp) const {
    std::vector<Hoister> hoisters;

    auto kernel = op->getAttrOfType<secret::KernelAttr>(
        secret::SecretDialect::kKernelAttrName);
    if (!kernel) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Kernel attribute not found on op " << *op << "\n");
      return hoisters;
    }

    if (!op->hasAttr(tensor_ext::TensorExtDialect::kLayoutAttrName)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Layout attribute not found on op " << *op << "\n");
      return hoisters;
    }

    // TODO(#2385): try hoisting layout conversion through bicyclic matmul
    return hoisters;
  }
};

}  // namespace

Hoister createTrivialHoister(Operation* op) {
  return [op](ConvertLayoutOp convertLayoutOp) -> llvm::FailureOr<HoistResult> {
    HoistResult result;
    Attribute outputLayout = convertLayoutOp.getToLayout();
    result.convertLayoutOp = convertLayoutOp;
    result.newInputLayouts =
        SmallVector<Attribute>(op->getNumOperands(), outputLayout);
    result.newKernel = KernelName::Trivial;
    result.newOutputLayout = outputLayout;
    return result;
  };
}

static bool isTrivialRankReduction(tensor::CollapseShapeOp op) {
  auto srcType = op.getSrcType();
  for (const auto& reassociation : op.getReassociationIndices()) {
    int64_t numNonUnit = 0;
    for (auto i : reassociation) {
      if (srcType.getDimSize(i) != 1) {
        numNonUnit++;
      }
    }
    if (numNonUnit > 1) return false;
  }
  return true;
}

static LayoutAttr hoistLayoutThroughCollapseShape(LayoutAttr attr,
                                                  tensor::CollapseShapeOp op) {
  auto rel = attr.getIntegerRelation();
  auto srcType = op.getSrcType();
  auto reassociationIndices = op.getReassociationIndices();

  IntegerRelation mapRel(PresburgerSpace::getRelationSpace(
      srcType.getRank(), op.getResultType().getRank(), 0, 0));

  for (auto [j, reassociation] : llvm::enumerate(reassociationIndices)) {
    int64_t nonUnitIdx = -1;
    for (auto i : reassociation) {
      if (srcType.getDimSize(i) != 1) {
        nonUnitIdx = i;
        break;
      }
    }
    if (nonUnitIdx == -1) nonUnitIdx = reassociation[0];

    for (auto i : reassociation) {
      SmallVector<int64_t> eq(mapRel.getNumCols(), 0);
      if (i == (int64_t)nonUnitIdx) {
        // i_i - j_j = 0
        eq[mapRel.getVarKindOffset(VarKind::Domain) + i] = 1;
        eq[mapRel.getVarKindOffset(VarKind::Range) + j] = -1;
      } else {
        // i_i = 0
        eq[mapRel.getVarKindOffset(VarKind::Domain) + i] = 1;
      }
      mapRel.addEquality(eq);
    }
  }

  mapRel.compose(rel);
  return LayoutAttr::getFromIntegerRelation(op.getContext(), mapRel);
}

Hoister createCollapseShapeHoister(tensor::CollapseShapeOp op) {
  return [op](ConvertLayoutOp convertLayoutOp) -> llvm::FailureOr<HoistResult> {
    if (!isTrivialRankReduction(op)) return failure();

    auto fromLayout = dyn_cast<LayoutAttr>(convertLayoutOp.getFromLayout());
    auto toLayout = dyn_cast<LayoutAttr>(convertLayoutOp.getToLayout());
    if (!fromLayout || !toLayout) return failure();

    HoistResult result;
    result.convertLayoutOp = convertLayoutOp;
    result.newOutputLayout = toLayout;
    result.newKernel = KernelName::Trivial;

    auto hoistedLayout = hoistLayoutThroughCollapseShape(toLayout, op);
    result.newInputLayouts = SmallVector<Attribute>{hoistedLayout};

    return result;
  };
}

static LayoutAttr hoistLayoutThroughPad(LayoutAttr attr, tensor::PadOp op) {
  SmallVector<int64_t> lowPadding = llvm::to_vector(op.getStaticLow());
  RankedTensorType paddedType = op.getResultType();
  RankedTensorType unpaddedType = op.getSourceType();
  IntegerRelation paddingRel =
      getPaddingRelation(paddedType, unpaddedType, lowPadding);
  paddingRel.inverse();
  IntegerRelation toRel = attr.getIntegerRelation();
  paddingRel.compose(toRel);
  return LayoutAttr::getFromIntegerRelation(op.getContext(), paddingRel);
}

Hoister createPadHoister(tensor::PadOp op) {
  return [op](ConvertLayoutOp convertLayoutOp) -> llvm::FailureOr<HoistResult> {
    auto fromLayout = dyn_cast<LayoutAttr>(convertLayoutOp.getFromLayout());
    auto toLayout = dyn_cast<LayoutAttr>(convertLayoutOp.getToLayout());
    if (!fromLayout || !toLayout) return failure();

    HoistResult result;
    result.convertLayoutOp = convertLayoutOp;
    result.newOutputLayout = toLayout;
    result.newKernel = KernelName::Trivial;

    auto hoistedLayout = hoistLayoutThroughPad(toLayout, op);
    result.newInputLayouts = SmallVector<Attribute>{hoistedLayout};

    return result;
  };
}

void registerLayoutConversionHoistableInterface(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, arith::ArithDialect* dialect) {
    arith::AddFOp::attachInterface<DoNothingHoistingImpl<arith::AddFOp>>(*ctx);
    arith::AddIOp::attachInterface<DoNothingHoistingImpl<arith::AddIOp>>(*ctx);
    arith::MulFOp::attachInterface<DoNothingHoistingImpl<arith::MulFOp>>(*ctx);
    arith::MulIOp::attachInterface<DoNothingHoistingImpl<arith::MulIOp>>(*ctx);
    arith::SubFOp::attachInterface<DoNothingHoistingImpl<arith::SubFOp>>(*ctx);
    arith::SubIOp::attachInterface<DoNothingHoistingImpl<arith::SubIOp>>(*ctx);
  });
  registry.addExtension(+[](MLIRContext* ctx, tensor::TensorDialect* dialect) {
    tensor::CollapseShapeOp::attachInterface<CollapseShapeHoistingImpl>(*ctx);
    tensor::PadOp::attachInterface<PadHoistingImpl>(*ctx);
  });
  registry.addExtension(+[](MLIRContext* ctx, linalg::LinalgDialect* dialect) {
    linalg::MatmulOp::attachInterface<MatmulHoistingImpl>(*ctx);
  });
}

}  // namespace heir
}  // namespace mlir
