#include "lib/Transforms/LayoutPropagation/LayoutPropagation.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Dialect/TensorExt/Transforms/Patterns.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Transforms/LayoutPropagation/Utils.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Layout/Hoisting.h"
#include "lib/Utils/Layout/IslConversion.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"       // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"       // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"      // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"             // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"        // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "layout-propagation"

namespace mlir {
namespace heir {

using linalg::BatchMatmulOp;
using linalg::BroadcastOp;
using linalg::DotOp;
using linalg::MatmulOp;
using linalg::ReduceOp;
using linalg::TransposeOp;
using presburger::IntegerRelation;
using secret::GenericOp;
using secret::SecretType;
using secret::YieldOp;
using tensor::CollapseShapeOp;
using tensor::ExpandShapeOp;
using tensor::InsertOp;
using tensor_ext::AssignLayoutOp;
using tensor_ext::BroadcastedReduceOp;
using tensor_ext::ConvertLayoutOp;
using tensor_ext::LayoutAttr;

#define GEN_PASS_DEF_LAYOUTPROPAGATION
#include "lib/Transforms/LayoutPropagation/LayoutPropagation.h.inc"

namespace {

// The result of a compatibility check for the layouts of an op's operands (cf.
// hasCompatibleArgumentLayouts). If the check fails, the presence of a
// diagnostic signals that the failure is unrecoverable and should cause the
// pass to fail. If the diagnostic is nullopt, then the failure can be
// recovered by rectifyIncompatibleOperandLayouts.
struct CompatibilityResult {
  bool compatible;
  std::optional<InFlightDiagnostic> diag;
};

void visitDebugInfo(Operation* op, DataFlowSolver* solver) {
  LLVM_DEBUG({
    llvm::dbgs() << "Visiting: " << op->getName() << "\n";
    llvm::dbgs() << "Operand secretness:\n";
    for (OpOperand& opOperand : op->getOpOperands()) {
      bool secretness = isSecret(opOperand.get(), solver);
      llvm::dbgs() << "- Operand " << opOperand.getOperandNumber() << ": "
                   << secretness << "\n";
    }
  });
}

void debugAssignLayout(Value value, Attribute layout) {
  LLVM_DEBUG(llvm::dbgs() << "Assigning layout " << layout << " to value "
                          << value << "\n");
}

std::pair<Value, LayoutAttr> convertToLayout(
    MLIRContext* ctx, mlir::IRRewriter& builder, Operation* op, Value value,
    LayoutAttr oldLayout, const IntegerRelation& newRelation,
    ArrayRef<int64_t> domainSchedule = {}) {
  builder.setInsertionPoint(op);
  LayoutAttr layoutAttr = LayoutAttr::getFromIntegerRelation(ctx, newRelation);
  ConvertLayoutOp convertLayoutOp = ConvertLayoutOp::create(
      builder, op->getLoc(), value, oldLayout, layoutAttr,
      builder.getDenseI64ArrayAttr(domainSchedule));
  convertLayoutOp->setAttr(tensor_ext::TensorExtDialect::kLayoutAttrName,
                           layoutAttr);
  Value toReplace = convertLayoutOp.getResult();
  builder.replaceUsesWithIf(value, toReplace, [&](OpOperand& operand) {
    return operand.getOwner() == op;
  });
  return std::make_pair(toReplace, layoutAttr);
}

// Return a copy of the kernel info associated with the value and update the
// result shape to the new result shape. If the value does not have a kernel
// info, return an empty Attribute.

Attribute cloneKernelInfoWithResultShape(Value value,
                                         ArrayRef<int64_t> resultShape) {
  auto kernelInfo = findAttributeAssociatedWith(value, kKernelInfoAttrName);
  if (succeeded(kernelInfo)) {
    auto info = getKernelInfo(kernelInfo.value()).value();
    info.resultShape = llvm::to_vector(resultShape);
    return makeKernelInfoAttr(value.getContext(), info);
  }
  return Attribute{};
}

}  // namespace

struct LayoutPropagation : impl::LayoutPropagationBase<LayoutPropagation> {
  using LayoutPropagationBase::LayoutPropagationBase;

  // Top level visit method handling common logic and dispatching to specific
  // visitOperation overloads.
  LogicalResult visitOperation(Operation* op);

  // Op-specific transfer functions
  LogicalResult visitOperation(CollapseShapeOp op);
  LogicalResult visitOperation(ExpandShapeOp op);
  LogicalResult visitOperation(BroadcastOp op);
  LogicalResult visitOperation(TransposeOp op);
  LogicalResult visitOperation(GenericOp op);
  LogicalResult visitOperation(ReduceOp op);
  LogicalResult visitOperation(BroadcastedReduceOp op);
  LogicalResult visitOperation(MatmulOp op);
  LogicalResult visitOperation(BatchMatmulOp op);
  LogicalResult visitOperation(DotOp op);
  LogicalResult visitOperation(YieldOp op);
  LogicalResult visitOperation(affine::AffineForOp op);
  LogicalResult visitOperation(func::FuncOp op);
  LogicalResult visitOperation(func::ReturnOp op);
  LogicalResult visitOperation(tensor::ExtractOp op);
  LogicalResult visitOperation(tensor::InsertOp op);
  LogicalResult visitOperation(tensor::InsertSliceOp op);
  LogicalResult visitOperation(tensor::ExtractSliceOp op);
  LogicalResult visitOperation(tensor::PadOp op);

  // Determine if the operation arguments have compatible layouts for the
  // given op. If the check fails, the CompatibilityResult::compatible field
  // is false. If there is also a diagnostic populated in the result, the
  // failure is unrecoverable.
  CompatibilityResult hasCompatibleArgumentLayouts(Operation* op);

  // Op-specific compatibility functions
  CompatibilityResult hasCompatibleArgumentLayouts(DotOp op);
  CompatibilityResult hasCompatibleArgumentLayouts(ReduceOp op);
  CompatibilityResult hasCompatibleArgumentLayouts(MatmulOp op);
  CompatibilityResult hasCompatibleArgumentLayouts(BatchMatmulOp op);
  CompatibilityResult hasCompatibleArgumentLayouts(tensor::InsertSliceOp op);

  // Insert conversion ops to rectify incompatible operand layouts
  void rectifyIncompatibleOperandLayouts(Operation* op);

  // Op-specific overrides
  void rectifyIncompatibleOperandLayouts(DotOp op);
  void rectifyIncompatibleOperandLayouts(ReduceOp op);
  void rectifyIncompatibleOperandLayouts(tensor::InsertOp op);
  void rectifyIncompatibleOperandLayouts(tensor::InsertSliceOp op);

  // Return the default layout for a given type
  FailureOr<LayoutAttr> defaultLayoutForType(Type type);
  FailureOr<LayoutAttr> defaultLayoutForScalarType(Type type);

  // Create an assign_layout op for the given value, and return the resulting
  // op. The given builder should have its insertion point set before calling.
  FailureOr<AssignLayoutOp> assignDefaultLayoutForOpOperand(
      Operation* op, Value operand, IRRewriter& builder);

  // Helper to pass layouts through generic ops
  void passLayoutThroughOp(Operation* op);

  // Add an op attribute denoting the layouts of the op results. Assumes the
  // assignedLayouts map contains the layout for the result SSA values already.
  void setResultLayoutAttr(Operation* op, Attribute kernelInfoAttr = {});

  void runOnOperation() override;

  LayoutAttr getComposedLayoutAttr(Value v) {
    Attribute attr = assignedLayouts.at(v);
    if (auto layoutAttr = dyn_cast<LayoutAttr>(attr)) {
      return layoutAttr;
    }
    if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
      auto layoutAttrs = arrayAttr.getAsRange<LayoutAttr>();
      auto it = layoutAttrs.begin();
      presburger::IntegerRelation composedRel = (*it).getIntegerRelation();
      ++it;
      for (; it != layoutAttrs.end(); ++it) {
        composedRel.compose((*it).getIntegerRelation());
      }
      return LayoutAttr::getFromIntegerRelation(&getContext(), composedRel);
    }
    llvm_unreachable("unexpected attribute type in assignedLayouts");
  }

  DenseMap<Value, Attribute> assignedLayouts;
  DataFlowSolver* solver;
};

FailureOr<AssignLayoutOp> LayoutPropagation::assignDefaultLayoutForOpOperand(
    Operation* op, Value operand, IRRewriter& builder) {
  FailureOr<LayoutAttr> layout = failure();
  auto existingLayout = findAttributeAssociatedWith(
      operand, tensor_ext::TensorExtDialect::kLayoutAttrName);
  if (succeeded(existingLayout)) {
    if (auto layoutAttr = dyn_cast<LayoutAttr>(*existingLayout)) {
      layout = layoutAttr;
    }
  }

  if (failed(layout)) {
    layout = defaultLayoutForType(operand.getType());
  }
  if (failed(layout)) {
    return failure();
  }
  LayoutAttr layoutAttr = layout.value();
  AssignLayoutOp assignLayoutOp =
      AssignLayoutOp::create(builder, op->getLoc(), operand, layoutAttr);
  setAttributeAssociatedWith(assignLayoutOp.getResult(),
                             tensor_ext::TensorExtDialect::kLayoutAttrName,
                             layoutAttr);
  Value toReplace = assignLayoutOp.getResult();
  // This may create duplicate layout assignment ops, and we expect CSE
  // to later clean them up. Otherwise we risk replacing a use of the
  // cleartext value in some other context.
  builder.replaceUsesWithIf(operand, toReplace, [&](OpOperand& otherOperand) {
    return otherOperand.getOwner() == op;
  });
  debugAssignLayout(toReplace, layoutAttr);
  assignedLayouts.insert({toReplace, layoutAttr});

  return assignLayoutOp;
}

LogicalResult LayoutPropagation::visitOperation(Operation* op) {
  visitDebugInfo(op, solver);

  if (!isa<func::FuncOp, func::ReturnOp, GenericOp, YieldOp, AssignLayoutOp>(
          op) &&
      !isSecret(op->getOperands(), solver) &&
      !isSecret(op->getResults(), solver)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping op " << op->getName()
               << " because its operands and results are non-secret, or it is "
                  "in a special allowlist of ops to ignore\n");
    return success();
  }

  // Convolutions, pools and products of a secret vector with a cleartext
  // matrix have a kernel only over strided layouts.
  if (isa<linalg::MatvecOp, linalg::VecmatOp, linalg::ConvolutionOpInterface>(
          op)) {
    return op->emitOpError()
           << "is only supported with strided layouts, and "
              "strided-layout-propagation did not apply to this module; run "
              "that pass on its own to see which operation it does not "
              "support";
  }

  // If an operand has no layout, it may for example be produced as a plaintext
  // constant, such as a zero-valued tensor for the initializer of a reduction.
  // In this case, we insert a layout assignment.
  for (OpOperand& opOperand : op->getOpOperands()) {
    Value operand = opOperand.get();
    if (!assignedLayouts.contains(operand)) {
      // Some operations may function properly with operands that have no
      // layout, e.g., a tensor_ext.rotate op doesn't need a layout for the
      // shift operand, and a tensor.insert op has a kernel that can work with a
      // cleartext scalar operand (and is, in fact, more efficient than when the
      // scalar operand is packed).
      if (auto layoutReqIface =
              dyn_cast<OperandLayoutRequirementOpInterface>(op)) {
        bool secretness = isSecret(opOperand.get(), solver);
        if (!layoutReqIface.operandRequiresLayout(opOperand.getOperandNumber(),
                                                  secretness)) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "OperandLayoutRequirementOpInterface ensures us that operand "
              << opOperand.getOperandNumber() << " of op " << op->getName()
              << " does not need a layout.\n");
          continue;
        }
      }

      LLVM_DEBUG(llvm::dbgs() << "operand has no layout: " << operand << "\n");
      mlir::IRRewriter builder(&getContext());
      builder.setInsertionPoint(op);
      if (failed(assignDefaultLayoutForOpOperand(op, operand, builder)))
        return op->emitError()
               << "Failed to assign default layout to operand " << operand;
    }
  }

  auto [compatible, diag] = hasCompatibleArgumentLayouts(op);
  if (!compatible) {
    LLVM_DEBUG(llvm::dbgs() << "Argument layouts for " << op->getName()
                            << " are incompatible\n");
    if (diag.has_value()) {
      // An InFlightDiagnostic casts to a failure()
      return diag.value();
    }
    rectifyIncompatibleOperandLayouts(op);
  }

  return TypeSwitch<Operation*, LogicalResult>(op)
      // func ops
      .Case<func::FuncOp, func::ReturnOp>(
          [&](auto op) { return visitOperation(op); })
      // secret ops
      .Case<GenericOp, YieldOp>([&](auto op) { return visitOperation(op); })
      // linalg ops
      .Case<DotOp, ReduceOp, BroadcastOp, TransposeOp, MatmulOp, BatchMatmulOp>(
          [&](auto op) { return visitOperation(op); })
      // affine ops
      .Case<affine::AffineForOp>([&](auto op) { return visitOperation(op); })
      // tensor ops
      .Case<tensor::ExtractOp, tensor::InsertOp, tensor::InsertSliceOp,
            tensor::ExtractSliceOp, tensor::PadOp, CollapseShapeOp,
            ExpandShapeOp, BroadcastedReduceOp>(
          [&](auto op) { return visitOperation(op); })
      // AddI, AddF, mgmt.* all pass the layout through unchanged.
      .Default([&](Operation* op) {
        passLayoutThroughOp(op);
        return success();
      });
}

LogicalResult LayoutPropagation::visitOperation(func::FuncOp op) {
  // Set a default value for each secret argument
  for (Value arg : op.getArguments()) {
    if (!isSecret(arg, solver)) {
      // Cleartext arguments don't get layouts, they are later given
      // assign_layout ops and materialized to plaintexts server-side.
      continue;
    }
    FailureOr<LayoutAttr> layout = failure();
    auto existingLayout = findAttributeAssociatedWith(
        arg, tensor_ext::TensorExtDialect::kLayoutAttrName);
    if (succeeded(existingLayout)) {
      if (auto layoutAttr = dyn_cast<LayoutAttr>(*existingLayout)) {
        layout = layoutAttr;
      }
    }
    if (failed(layout)) {
      layout = defaultLayoutForType(arg.getType());
    }
    if (failed(layout)) {
      return op->emitOpError()
             << "Failed to assign default layout to func argument " << arg;
    }
    debugAssignLayout(arg, layout.value());
    assignedLayouts.insert({arg, layout.value()});
    setAttributeAssociatedWith(
        arg, tensor_ext::TensorExtDialect::kLayoutAttrName, layout.value());

    // Set a default kernel info for each secret argument.
    if (auto tensorType = dyn_cast<RankedTensorType>(
            cast<SecretType>(arg.getType()).getValueType())) {
      setAttributeAssociatedWith(
          arg, kKernelInfoAttrName,
          makeKernelInfoAttr(
              arg.getContext(),
              KernelInfo{.resultShape = llvm::to_vector(tensorType.getShape()),
                         .gapFactor = 1}));
    }
  }

  // Func result attrs are handled by the ReturnOp
  return success();
}

LogicalResult LayoutPropagation::visitOperation(func::ReturnOp op) {
  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  for (OpOperand& operand : op->getOpOperands()) {
    if (!assignedLayouts.contains(operand.get())) {
      if (isSecret(operand.get(), solver)) {
        return op->emitError("secret return value has no assigned layout");
      }

      // It needs no layout.
      continue;
    }
    Attribute layout = assignedLayouts.at(operand.get());
    func.setResultAttr(operand.getOperandNumber(),
                       tensor_ext::TensorExtDialect::kLayoutAttrName, layout);
  }
  return success();
}

LogicalResult LayoutPropagation::visitOperation(GenericOp op) {
  // Every block argument has the same layout as its corresponding operand.
  for (OpOperand& operand : op->getOpOperands()) {
    if (!assignedLayouts.contains(operand.get())) {
      // Assume it is not a tensor type and doesn't need a layout.
      continue;
    }
    Attribute layout = assignedLayouts.at(operand.get());
    BlockArgument blockArg =
        op.getRegion().getArgument(operand.getOperandNumber());
    assignedLayouts.insert({blockArg, layout});
    setAttributeAssociatedWith(
        blockArg, tensor_ext::TensorExtDialect::kLayoutAttrName, layout);
    debugAssignLayout(blockArg, layout);
    // Pass through kernel info
    auto kernelInfo =
        findAttributeAssociatedWith(operand.get(), kKernelInfoAttrName);
    if (succeeded(kernelInfo)) {
      setAttributeAssociatedWith(blockArg, kKernelInfoAttrName,
                                 kernelInfo.value());
    }
  }
  // The layout of the result of the generic op is handled when the YieldOp is
  // visited.
  return success();
}

LogicalResult LayoutPropagation::visitOperation(YieldOp op) {
  // The results of the generic op has the same layouts as the yielded values
  GenericOp generic = op->getParentOfType<GenericOp>();
  for (OpOperand& operand : op->getOpOperands()) {
    Type operandType = operand.get().getType();
    if (!assignedLayouts.contains(operand.get())) {
      // If it's a tensor type, it may be something like a tensor.empty()
      // that would not be assigned a layout earlier in the walk, because
      // it does not depend on any secret information. In this case, use the
      // default layout.
      LLVM_DEBUG(llvm::dbgs() << "No layout assigned to operand "
                              << operand.get() << ", using default layout\n");
      if (isa<RankedTensorType>(operandType)) {
        FailureOr<LayoutAttr> layout = defaultLayoutForType(operandType);
        if (failed(layout)) {
          return failure();
        }
        LayoutAttr layoutAttr = layout.value();
        debugAssignLayout(operand.get(), layoutAttr);
        assignedLayouts.insert({operand.get(), layoutAttr});
      } else {
        // Assume it is not a tensor type and doesn't need a layout.
        continue;
      }
    }
    Attribute layout = assignedLayouts.at(operand.get());
    Value result = generic.getResult(operand.getOperandNumber());
    assignedLayouts.insert({result, layout});
    debugAssignLayout(result, layout);
    setAttributeAssociatedWith(
        result, tensor_ext::TensorExtDialect::kLayoutAttrName, layout);
  }
  return success();
}

LogicalResult LayoutPropagation::visitOperation(CollapseShapeOp op) {
  // Reassociate the dimensions to collapse. For e.g. if we collapse d0 x d1 x
  // d2 to d' then we need to construct the row-major relation from (d0, d1, d2)
  // -> d'. Then the relation on the collapsed tensor will be the composition
  // from d' -> (d0, d1, d2) -> (ct, slot).
  auto rowMajorRelation = getCollapsedRelation(
      op.getSrcType(), op.getResultType(), op.getReassociationIndices());

  auto tensor = op.getSrc();
  LayoutAttr inputLayout = getComposedLayoutAttr(tensor);
  IntegerRelation relation = inputLayout.getIntegerRelation();

  relation.applyDomain(rowMajorRelation);
  MLIRContext* ctx = &getContext();
  LayoutAttr resultLayoutAttr =
      LayoutAttr::getFromIntegerRelation(ctx, relation);
  Attribute kernelInfoAttr =
      cloneKernelInfoWithResultShape(tensor, op.getResultType().getShape());

  assignedLayouts.insert({op.getResult(), resultLayoutAttr});
  setResultLayoutAttr(op, kernelInfoAttr);
  debugAssignLayout(op.getResult(), resultLayoutAttr);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(ExpandShapeOp op) {
  // Only support rank-reduced types for now, i.e., where the expanded shape
  // only adds static dimensions of size 1.
  SliceVerificationResult res =
      isRankReducedType(op.getResultType(), op.getSrcType());
  if (res != SliceVerificationResult::Success)
    return op->emitError(
        "Only rank-reduced types are supported for ExpandShapeOp");

  auto tensor = op.getSrc();
  LayoutAttr inputLayout = getComposedLayoutAttr(tensor);
  IntegerRelation relation = inputLayout.getIntegerRelation();
  IntegerRelation expandedRelation = expandDimensions(
      relation, op.getResultType(), op.getReassociationIndices());

  MLIRContext* ctx = &getContext();
  LayoutAttr resultLayoutAttr =
      LayoutAttr::getFromIntegerRelation(ctx, expandedRelation);
  Attribute kernelInfoAttr =
      cloneKernelInfoWithResultShape(tensor, op.getResultType().getShape());

  assignedLayouts.insert({op.getResult(), resultLayoutAttr});
  setResultLayoutAttr(op, kernelInfoAttr);
  debugAssignLayout(op.getResult(), resultLayoutAttr);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(BroadcastOp op) {
  auto input = op.getInput();
  LayoutAttr inputLayout = getComposedLayoutAttr(input);
  Value result = op->getResult(0);
  auto inputType = cast<RankedTensorType>(input.getType());
  auto resultType = cast<RankedTensorType>(result.getType());
  MLIRContext* ctx = &getContext();

  // If the input is in cyclic layout and the result can be packed in a
  // cyclic layout, the result is assigned the cyclic layout (which
  // lowers to zero-cost). Otherwise, assign the row-major default layout.
  LayoutAttr resultLayoutAttr;
  IntegerRelation cyclicResultRel =
      getCyclicLayoutRelation(resultType, minSlotCount);
  if (isRelationCyclic(inputType, minSlotCount,
                       inputLayout.getIntegerRelation()) &&
      isRelationCyclic(resultType, minSlotCount, cyclicResultRel)) {
    resultLayoutAttr = LayoutAttr::getFromIntegerRelation(ctx, cyclicResultRel);
  } else {
    resultLayoutAttr = LayoutAttr::getFromIntegerRelation(
        ctx, getRowMajorLayoutRelation(resultType, minSlotCount));
  }

  Attribute kernelInfoAttr =
      cloneKernelInfoWithResultShape(input, resultType.getShape());
  assignedLayouts.insert({result, resultLayoutAttr});
  setResultLayoutAttr(op, kernelInfoAttr);
  debugAssignLayout(result, resultLayoutAttr);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(TransposeOp op) {
  // The result layout is the input relation with its domain variables
  // permuted: result index (j_0, ..., j_r) reads the slot of input index
  // (j_perm[0], ..., j_perm[r]), so the packed data is unchanged and the
  // transpose materializes as a retype. Consumers needing a materialized
  // layout get a convert_layout from the ordinary rectification path.
  auto input = op.getInput();
  LayoutAttr inputLayout = getComposedLayoutAttr(input);
  IntegerRelation relation = getTransposedRelation(
      inputLayout.getIntegerRelation(), op.getPermutation());

  Value result = op->getResult(0);
  auto resultType = cast<RankedTensorType>(result.getType());
  MLIRContext* ctx = &getContext();
  LayoutAttr resultLayoutAttr =
      LayoutAttr::getFromIntegerRelation(ctx, relation);
  Attribute kernelInfoAttr =
      cloneKernelInfoWithResultShape(input, resultType.getShape());
  assignedLayouts.insert({result, resultLayoutAttr});
  setResultLayoutAttr(op, kernelInfoAttr);
  debugAssignLayout(result, resultLayoutAttr);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(DotOp op) {
  LLVM_DEBUG(llvm::dbgs() << "visitOperation(DotOp): " << op << "\n");
  Value result = op.getResult(0);
  FailureOr<LayoutAttr> layout = defaultLayoutForType(result.getType());
  if (failed(layout)) {
    return failure();
  }
  Attribute kernelInfoAttr = cloneKernelInfoWithResultShape(
      op.getInputs().front(),
      cast<RankedTensorType>(result.getType()).getShape());
  assignedLayouts.insert({result, layout.value()});
  setResultLayoutAttr(op, kernelInfoAttr);
  debugAssignLayout(result, layout.value());

  // Add secret.kernel attribute for Dot
  MLIRContext* ctx = &getContext();
  auto kernelAttr =
      secret::KernelAttr::get(ctx, KernelName::Dot, /*force=*/false);
  op->setAttr(secret::SecretDialect::kKernelAttrName, kernelAttr);

  return success();
}

LogicalResult LayoutPropagation::visitOperation(BatchMatmulOp op) {
  MLIRContext* ctx = &getContext();
  mlir::IRRewriter builder(ctx);

  Value lhs = op.getOperand(0);
  Value rhs = op.getOperand(1);
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  auto result = op->getResult(0);

  bool secretLhs = isSecret(lhs, solver);
  bool secretRhs = isSecret(rhs, solver);

  int64_t hDim = lhsType.getDimSize(0);
  int64_t mDim = lhsType.getDimSize(1);
  int64_t nDim = lhsType.getDimSize(2);
  int64_t pDim = rhsType.getDimSize(2);
  bool batchEqual = (hDim == rhsType.getDimSize(0));

  bool lhsCoprime = std::gcd(hDim, mDim) == 1 && std::gcd(hDim, nDim) == 1 &&
                    std::gcd(mDim, nDim) == 1;
  bool rhsCoprime = std::gcd(hDim, nDim) == 1 && std::gcd(hDim, pDim) == 1 &&
                    std::gcd(nDim, pDim) == 1;
  bool outputCoprime = std::gcd(hDim, mDim) == 1 && std::gcd(hDim, pDim) == 1 &&
                       std::gcd(mDim, pDim) == 1;

  LLVM_DEBUG(llvm::dbgs() << "lhs=" << lhs << ";\nrhs=" << rhs << "\n");

  if (!batchEqual || !outputCoprime) {
    return builder.notifyMatchFailure(
        op, "batch matmul shape requirements for tricyclic layout are not met");
  }

  // Tricyclic ct-ct batch matmul.
  if (secretLhs && secretRhs && lhsCoprime && rhsCoprime) {
    LayoutAttr lhsLayout = getComposedLayoutAttr(lhs);
    if (!isRelationTricyclic(lhsType, minSlotCount,
                             lhsLayout.getIntegerRelation())) {
      auto [toReplace, newInputMatrixLayoutAttr] =
          convertToLayout(ctx, builder, op, lhs, lhsLayout,
                          getTricyclicLayoutRelation(lhsType, minSlotCount));
      debugAssignLayout(toReplace, newInputMatrixLayoutAttr);
      assignedLayouts.insert({toReplace, newInputMatrixLayoutAttr});
    }

    LayoutAttr rhsLayout = getComposedLayoutAttr(rhs);
    if (!isRelationTricyclic(rhsType, minSlotCount,
                             rhsLayout.getIntegerRelation())) {
      auto [toReplace, newFilterMatrixLayoutAttr] =
          convertToLayout(ctx, builder, op, rhs, rhsLayout,
                          getTricyclicLayoutRelation(rhsType, minSlotCount));
      debugAssignLayout(toReplace, newFilterMatrixLayoutAttr);
      assignedLayouts.insert({toReplace, newFilterMatrixLayoutAttr});
    }

    RankedTensorType outputType = cast<RankedTensorType>(result.getType());
    IntegerRelation outputLayoutResult =
        getTricyclicLayoutRelation(outputType, minSlotCount);
    LayoutAttr outputLayoutAttr =
        LayoutAttr::getFromIntegerRelation(ctx, outputLayoutResult);

    assignedLayouts.insert({result, outputLayoutAttr});
    setResultLayoutAttr(op);
    debugAssignLayout(result, outputLayoutAttr);

    auto kernelAttr = secret::KernelAttr::get(
        ctx, KernelName::BatchMatmulTricyclic, /*force=*/false);
    op->setAttr(secret::SecretDialect::kKernelAttrName, kernelAttr);
    return success();
  }

  // Tricyclic ct-pt / pt-ct batch matmul with generalized-diagonal-layout
  // plaintext: the secret operand keeps its tricyclic packing, the weight
  // packs as encode-time diagonals, and the BSGS reach is period * (steps - 1).
  if (secretLhs != secretRhs) {
    Value secretOperand;
    Value weightOperand;
    RankedTensorType secretType;
    RankedTensorType weightType;
    int64_t ctStride;
    int64_t contractionDim;
    bool secretCoprime;

    if (secretLhs) {
      // ct-pt: LHS is ciphertext (h x m x n), RHS is weight (h x n x p).
      secretOperand = lhs;
      weightOperand = rhs;
      secretType = lhsType;
      weightType = rhsType;
      ctStride = mDim;
      contractionDim = 1;
      secretCoprime = lhsCoprime;
    } else {
      // pt-ct: LHS is weight (h x m x n), RHS is ciphertext (h x n x p).
      secretOperand = rhs;
      weightOperand = lhs;
      secretType = rhsType;
      weightType = lhsType;
      ctStride = pDim;
      contractionDim = 2;
      secretCoprime = rhsCoprime;
    }

    if (secretCoprime) {
      RankedTensorType bmmOutputType = cast<RankedTensorType>(result.getType());
      int64_t period = hDim * ctStride;
      int64_t reach = period * (nDim - 1);
      if (reach + bmmOutputType.getNumElements() > minSlotCount) {
        return builder.notifyMatchFailure(
            op, "slot count budget exceeded for batch matmul diagonal reach");
      }

      LayoutAttr secretLayout = getComposedLayoutAttr(secretOperand);
      IntegerRelation secretTricyclic =
          getTricyclicLayoutRelation(secretType, minSlotCount);
      if (!isRelationTricyclic(secretType, minSlotCount,
                               secretLayout.getIntegerRelation())) {
        auto [toReplace, newSecretLayoutAttr] = convertToLayout(
            ctx, builder, op, secretOperand, secretLayout, secretTricyclic);
        debugAssignLayout(toReplace, newSecretLayoutAttr);
        assignedLayouts.insert({toReplace, newSecretLayoutAttr});
      }

      auto assignOrUpdateLayout = [&](Value val, const IntegerRelation& rel) {
        LayoutAttr layout = LayoutAttr::getFromIntegerRelation(ctx, rel);
        if (auto assignOp = val.getDefiningOp<AssignLayoutOp>();
            assignOp && assignOp->hasOneUse()) {
          assignOp.setLayoutAttr(layout);
          assignedLayouts[val] = layout;
          setAttributeAssociatedWith(
              val, tensor_ext::TensorExtDialect::kLayoutAttrName, layout);
          debugAssignLayout(val, layout);
          return val;
        }
        builder.setInsertionPoint(op);
        AssignLayoutOp assignOp =
            AssignLayoutOp::create(builder, op->getLoc(), val, layout);
        setAttributeAssociatedWith(
            assignOp.getResult(), tensor_ext::TensorExtDialect::kLayoutAttrName,
            layout);
        Value toReplace = assignOp.getResult();
        builder.replaceUsesWithIf(val, toReplace, [&](OpOperand& other) {
          return other.getOwner() == op;
        });
        assignedLayouts.insert({toReplace, layout});
        debugAssignLayout(toReplace, layout);
        return toReplace;
      };

      // Pack plaintext weight into diagonals walking the contraction dimension.
      IntegerRelation diagRelation = getTricyclicDiagonalRelation(
          weightType, contractionDim, ctStride, minSlotCount);
      LayoutAttr weightLayout = getComposedLayoutAttr(weightOperand);
      LayoutAttr diagLayoutAttr =
          LayoutAttr::getFromIntegerRelation(ctx, diagRelation);
      if (weightLayout != diagLayoutAttr) {
        assignOrUpdateLayout(weightOperand, diagRelation);
      }

      LayoutAttr outputLayoutAttr = LayoutAttr::getFromIntegerRelation(
          ctx, getTricyclicLayoutRelation(bmmOutputType, minSlotCount));
      assignedLayouts.insert({result, outputLayoutAttr});
      debugAssignLayout(result, outputLayoutAttr);

      // The kernel adds the accumulator (init) directly to the output via SIMD
      // addition. Align the init layout with the result layout at compile time
      // so cleartext constants (zeros/bias) re-pack freely without conversions.
      Value init = op.getOutputs()[0];
      LayoutAttr initLayout = getComposedLayoutAttr(init);
      if (initLayout != outputLayoutAttr) {
        assignOrUpdateLayout(init, outputLayoutAttr.getIntegerRelation());
      }

      setResultLayoutAttr(op);
      auto kernelAttr = secret::KernelAttr::get(
          ctx, KernelName::BatchMatmulTricyclicDiagonal, /*force=*/false);
      op->setAttr(secret::SecretDialect::kKernelAttrName, kernelAttr);
      return success();
    }
  }

  return builder.notifyMatchFailure(
      op, "batch matmul shape requirements for tricyclic layout are not met");
}

LogicalResult LayoutPropagation::visitOperation(MatmulOp op) {
  MLIRContext* ctx = &getContext();
  mlir::IRRewriter builder(ctx);

  Value lhs = op.getOperand(0);
  Value rhs = op.getOperand(1);
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  auto result = op->getResult(0);

  bool inputSecret = isSecret(lhs, solver);
  bool filterSecret = isSecret(rhs, solver);
  int64_t mDim = lhsType.getDimSize(0);
  int64_t nDim = lhsType.getDimSize(1);
  int64_t pDim = rhsType.getDimSize(1);
  bool lhsCoprime = std::gcd(mDim, nDim) == 1;
  bool rhsCoprime = std::gcd(nDim, pDim) == 1;
  bool outputCoprime = std::gcd(mDim, pDim) == 1;

  LLVM_DEBUG(llvm::dbgs() << "lhs=" << lhs << ";\nrhs=" << rhs << "\n");

  // Bicyclic ct-ct matmul.
  if (inputSecret && filterSecret && lhsCoprime && rhsCoprime &&
      outputCoprime) {
    LayoutAttr lhsLayout = getComposedLayoutAttr(lhs);
    if (!isRelationBicyclic(lhsType, minSlotCount,
                            lhsLayout.getIntegerRelation())) {
      auto [toReplace, newInputMatrixLayoutAttr] =
          convertToLayout(ctx, builder, op, lhs, lhsLayout,
                          getBicyclicLayoutRelation(lhsType, minSlotCount));
      debugAssignLayout(toReplace, newInputMatrixLayoutAttr);
      assignedLayouts.insert({toReplace, newInputMatrixLayoutAttr});
    }

    LayoutAttr rhsLayout = getComposedLayoutAttr(rhs);
    if (!isRelationBicyclic(rhsType, minSlotCount,
                            rhsLayout.getIntegerRelation())) {
      auto [toReplace, newFilterMatrixLayoutAttr] =
          convertToLayout(ctx, builder, op, rhs, rhsLayout,
                          getBicyclicLayoutRelation(rhsType, minSlotCount));
      debugAssignLayout(toReplace, newFilterMatrixLayoutAttr);
      assignedLayouts.insert({toReplace, newFilterMatrixLayoutAttr});
    }

    RankedTensorType outputType = cast<RankedTensorType>(result.getType());
    IntegerRelation outputLayoutResult =
        getBicyclicLayoutRelation(outputType, minSlotCount);
    LayoutAttr outputLayoutAttr =
        LayoutAttr::getFromIntegerRelation(ctx, outputLayoutResult);
    auto kernelInfoAttr =
        cloneKernelInfoWithResultShape(lhs, outputType.getShape());

    assignedLayouts.insert({result, outputLayoutAttr});
    setResultLayoutAttr(op, kernelInfoAttr);
    debugAssignLayout(result, outputLayoutAttr);

    auto kernelAttr = secret::KernelAttr::get(ctx, KernelName::MatmulBicyclic,
                                              /*force=*/false);
    op->setAttr(secret::SecretDialect::kKernelAttrName, kernelAttr);
    return success();
  }

  // Bicyclic ct-pt / pt-ct matmul with generalized-diagonal-layout
  // plaintext: the secret operand keeps its bicyclic packing, the weight
  // packs as encode-time diagonals, and the BSGS reach is period * (steps - 1)
  // with period the secret operand's packed row (or column) count.
  if (outputCoprime && ((inputSecret && !filterSecret && lhsCoprime) ||
                        (!inputSecret && filterSecret && rhsCoprime))) {
    Value secretOperand = inputSecret ? lhs : rhs;
    RankedTensorType secretType = inputSecret ? lhsType : rhsType;
    Value weight = inputSecret ? rhs : lhs;
    RankedTensorType weightType = inputSecret ? rhsType : lhsType;

    LayoutAttr secretLayout = getComposedLayoutAttr(secretOperand);
    if (!isRelationBicyclic(secretType, minSlotCount,
                            secretLayout.getIntegerRelation())) {
      auto [toReplace, newSecretLayoutAttr] =
          convertToLayout(ctx, builder, op, secretOperand, secretLayout,
                          getBicyclicLayoutRelation(secretType, minSlotCount));
      debugAssignLayout(toReplace, newSecretLayoutAttr);
      assignedLayouts.insert({toReplace, newSecretLayoutAttr});
    }

    // The weight packs as the generalized-diagonal matrix; a conversion
    // on a constant folds into its assign_layout.
    IntegerRelation diagRelation =
        inputSecret
            ? getBicyclicDiagonalRelation(weightType,
                                          /*contractionDim=*/0,
                                          /*stride=*/mDim, minSlotCount)
            : getBicyclicDiagonalRelation(weightType,
                                          /*contractionDim=*/1,
                                          /*stride=*/pDim, minSlotCount);
    auto assignOrUpdateLayout = [&](Value val, const IntegerRelation& rel) {
      LayoutAttr layout = LayoutAttr::getFromIntegerRelation(ctx, rel);
      if (auto assignOp = val.getDefiningOp<AssignLayoutOp>();
          assignOp && assignOp->hasOneUse()) {
        assignOp.setLayoutAttr(layout);
        assignedLayouts[val] = layout;
        setAttributeAssociatedWith(
            val, tensor_ext::TensorExtDialect::kLayoutAttrName, layout);
        debugAssignLayout(val, layout);
        return val;
      }
      builder.setInsertionPoint(op);
      AssignLayoutOp assignOp =
          AssignLayoutOp::create(builder, op->getLoc(), val, layout);
      setAttributeAssociatedWith(assignOp.getResult(),
                                 tensor_ext::TensorExtDialect::kLayoutAttrName,
                                 layout);
      Value toReplace = assignOp.getResult();
      builder.replaceUsesWithIf(val, toReplace, [&](OpOperand& other) {
        return other.getOwner() == op;
      });
      assignedLayouts.insert({toReplace, layout});
      debugAssignLayout(toReplace, layout);
      return toReplace;
    };

    LayoutAttr weightLayout = getComposedLayoutAttr(weight);
    LayoutAttr diagLayoutAttr =
        LayoutAttr::getFromIntegerRelation(ctx, diagRelation);
    if (weightLayout != diagLayoutAttr) {
      assignOrUpdateLayout(weight, diagRelation);
    }

    RankedTensorType outputType = cast<RankedTensorType>(result.getType());
    LayoutAttr outputLayoutAttr = LayoutAttr::getFromIntegerRelation(
        ctx, getBicyclicLayoutRelation(outputType, minSlotCount));
    auto kernelInfoAttr =
        cloneKernelInfoWithResultShape(secretOperand, outputType.getShape());

    assignedLayouts.insert({result, outputLayoutAttr});
    debugAssignLayout(result, outputLayoutAttr);

    // The kernel adds the accumulator (init) directly to the output via SIMD
    // addition. Align the init layout with the result layout at compile time
    // so cleartext constants (zeros/bias) re-pack freely without conversions.
    Value init = op.getOutputs()[0];
    LayoutAttr initLayout = getComposedLayoutAttr(init);
    if (initLayout != outputLayoutAttr) {
      assignOrUpdateLayout(init, outputLayoutAttr.getIntegerRelation());
    }

    setResultLayoutAttr(op, kernelInfoAttr);

    auto kernelAttr = secret::KernelAttr::get(
        ctx, KernelName::MatmulBicyclicDiagonal, /*force=*/false);
    op->setAttr(secret::SecretDialect::kKernelAttrName, kernelAttr);
    return success();
  }

  return op->emitOpError()
         << "has no kernel for these operands: the bicyclic kernels need "
            "pairwise coprime dimensions, and a product with a cleartext "
            "matrix is otherwise only supported with strided layouts";
}

LogicalResult LayoutPropagation::visitOperation(ReduceOp op) {
  MLIRContext* ctx = &getContext();
  mlir::IRRewriter builder(ctx);
  builder.setInsertionPoint(op);

  for (const auto& [tensor, result] :
       llvm::zip(op.getInputs(), op.getResults())) {
    LayoutAttr thisLayout = getComposedLayoutAttr(tensor);

    // enforce row-major layout
    RankedTensorType thisType = cast<RankedTensorType>(tensor.getType());
    if (!isRelationRowMajor(thisType, minSlotCount,
                            thisLayout.getIntegerRelation())) {
      LLVM_DEBUG(llvm::dbgs() << "ReduceOp tensor is not row major");
      auto [toReplace, newLayoutAttr] =
          convertToLayout(ctx, builder, op, tensor, thisLayout,
                          getRowMajorLayoutRelation(thisType, minSlotCount));
      debugAssignLayout(toReplace, newLayoutAttr);
      assignedLayouts.insert({toReplace, newLayoutAttr});
      thisLayout = newLayoutAttr;
    }

    // drop dimension on output
    LayoutAttr resultLayout =
        convertLayoutForReduce(thisLayout, op.getDimensions());
    assignedLayouts.insert({result, resultLayout});
    debugAssignLayout(result, resultLayout);
  }
  setResultLayoutAttr(op);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(BroadcastedReduceOp op) {
  MLIRContext* ctx = &getContext();
  mlir::IRRewriter builder(ctx);
  builder.setInsertionPoint(op);

  Value tensor = op.getTensor();
  LayoutAttr thisLayout = getComposedLayoutAttr(tensor);

  // enforce row-major layout
  RankedTensorType thisType = cast<RankedTensorType>(tensor.getType());
  if (!isRelationRowMajor(thisType, minSlotCount,
                          thisLayout.getIntegerRelation())) {
    LLVM_DEBUG(llvm::dbgs() << "BroadcastedReduceOp tensor is not row major\n");
    auto [toReplace, newLayoutAttr] =
        convertToLayout(ctx, builder, op, tensor, thisLayout,
                        getRowMajorLayoutRelation(thisType, minSlotCount));
    debugAssignLayout(toReplace, newLayoutAttr);
    assignedLayouts.insert({toReplace, newLayoutAttr});
    thisLayout = newLayoutAttr;
  }

  // propagate layout to output (layout-preserving)
  Value result = op.getOutput();
  assignedLayouts.insert({result, thisLayout});
  debugAssignLayout(result, thisLayout);

  setResultLayoutAttr(op);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(affine::AffineForOp op) {
  // Transfer the layout of the inits to the region iter args
  for (const auto& [init, iterArg, result] :
       llvm::zip(op.getInits(), op.getRegionIterArgs(), op.getResults())) {
    LayoutAttr layout;

    // The init may not have a layout if it is a non-secret initial value. In
    // that case we assign it a default layout.
    if (!assignedLayouts.contains(init)) {
      mlir::IRRewriter builder(&getContext());
      builder.setInsertionPoint(op);
      auto res = assignDefaultLayoutForOpOperand(op, init, builder);
      if (failed(res)) {
        return op->emitError()
               << "Failed to assign default layout to init " << init;
      }
      layout = cast<LayoutAttr>(res.value().getLayout());
    } else {
      layout = getComposedLayoutAttr(init);
    }
    assignedLayouts.insert({iterArg, layout});
    setAttributeAssociatedWith(
        iterArg, tensor_ext::TensorExtDialect::kLayoutAttrName, layout);
    debugAssignLayout(iterArg, layout);

    // The result of an AffineForOp also has the same type as its iter args.
    assignedLayouts.insert({result, layout});
    debugAssignLayout(result, layout);
  }

  setResultLayoutAttr(op);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(tensor::ExtractOp op) {
  std::string relationStr = "{ [] -> [ct, slot] : ct = 0 and slot = 0 }";
  LayoutAttr scalarLayout = LayoutAttr::get(op.getContext(), relationStr);
  LLVM_DEBUG(llvm::dbgs() << "Assigning scalar layout " << scalarLayout
                          << " to value " << op.getResult() << "\n");
  Value result = op.getResult();
  auto kernelInfoAttr = cloneKernelInfoWithResultShape(op.getTensor(), {});
  assignedLayouts.insert({result, scalarLayout});
  debugAssignLayout(result, scalarLayout);
  setResultLayoutAttr(op, kernelInfoAttr);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(tensor::InsertOp op) {
  // The layout of the result is the same as the layout of the input tensor.
  // The scalar input is handled by the generic logic in visitOperation,
  // which will assign a default scalar layout if it is secret and has no
  // layout.
  LayoutAttr tensorLayout = getComposedLayoutAttr(op.getDest());
  auto kernelInfoAttr = cloneKernelInfoWithResultShape(
      op.getScalar(), op.getDest().getType().getShape());
  Value result = op.getResult();
  assignedLayouts.insert({result, tensorLayout});
  debugAssignLayout(result, tensorLayout);
  setResultLayoutAttr(op, kernelInfoAttr);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(tensor::InsertSliceOp op) {
  // Set the result layout to be the same as the dest tensor.
  if (!assignedLayouts.contains(op.getDest())) {
    return op->emitError() << "Destination tensor has no assigned layout";
  }
  LayoutAttr destLayout = getComposedLayoutAttr(op.getDest());
  auto kernelInfoAttr = cloneKernelInfoWithResultShape(
      op.getSource(), op.getDest().getType().getShape());

  Value result = op.getResult();
  assignedLayouts.insert({result, destLayout});
  debugAssignLayout(result, destLayout);
  setResultLayoutAttr(op, kernelInfoAttr);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(tensor::PadOp op) {
  if (!assignedLayouts.contains(op.getSource())) {
    return op->emitError() << "Source tensor has no assigned layout";
  }

  // Extract static low and high paddings. If dynamic bounds are present, emit
  // error.
  bool hasDynamicBounds = false;
  for (int64_t val : op.getStaticLow()) {
    if (ShapedType::isDynamic(val)) {
      hasDynamicBounds = true;
    }
  }
  for (int64_t val : op.getStaticHigh()) {
    if (ShapedType::isDynamic(val)) {
      hasDynamicBounds = true;
    }
  }
  if (hasDynamicBounds || !op.getLow().empty() || !op.getHigh().empty()) {
    return op->emitError()
           << "Only static padding is supported for layout propagation";
  }

  Value padValue = op.getConstantPaddingValue();
  bool isZeroPad = padValue && (matchPattern(padValue, m_AnyZeroFloat()) ||
                                matchPattern(padValue, m_Zero()));
  if (!isZeroPad) {
    return op->emitError()
           << "layout propagation only supports zero-padding tensor.pad";
  }

  RankedTensorType paddedType = op.getResultType();
  SmallVector<int64_t> lowPadding = llvm::to_vector(op.getStaticLow());
  IntegerRelation sourceLayout =
      getComposedLayoutAttr(op.getSource()).getIntegerRelation();

  RankedTensorType unpaddedType = op.getSourceType();
  IntegerRelation paddingRel =
      getPaddingRelation(paddedType, unpaddedType, lowPadding);

  paddingRel.compose(sourceLayout);

  Attribute kernelInfoAttr =
      cloneKernelInfoWithResultShape(op.getSource(), paddedType.getShape());

  LayoutAttr outputLayout =
      LayoutAttr::getFromIntegerRelation(op.getContext(), paddingRel);
  assignedLayouts.insert({op.getResult(), outputLayout});
  debugAssignLayout(op.getResult(), outputLayout);
  setResultLayoutAttr(op, kernelInfoAttr);
  return success();
}

LogicalResult LayoutPropagation::visitOperation(tensor::ExtractSliceOp op) {
  // Assign the induced layout from extracting a slice from the source tensor.
  if (!assignedLayouts.contains(op.getSource())) {
    return op->emitError() << "Source tensor has no assigned layout";
  }
  IntegerRelation sourceLayout =
      getComposedLayoutAttr(op.getSource()).getIntegerRelation();

  FailureOr<IntegerRelation> maybeSliceExtractionLayout =
      getSliceExtractionRelation(op.getSourceType(), op.getResultType(),
                                 SmallVector<int64_t>(op.getStaticOffsets()),
                                 SmallVector<int64_t>(op.getStaticSizes()),
                                 SmallVector<int64_t>(op.getStaticStrides()));
  if (failed(maybeSliceExtractionLayout)) {
    return failure();
  }
  IntegerRelation sliceExtractionLayout = maybeSliceExtractionLayout.value();

  // Compose the inverted slice extraction layout with the source layout to
  // get the result slice layout.
  sliceExtractionLayout.inverse();
  sliceExtractionLayout.compose(sourceLayout);
  // If the slice extracted was not at offset zero, then the resulting slice may
  // be indexed at a non-zero ciphertext. For example, imagine extracting a
  // slice out of the second ciphertext. Then computing the inverse of the slice
  // extraction layout and composing that with the source relation would mean
  // that the slice would map to the second ciphertext. But a slice extracted
  // from a tensor.extract_slice op is always indexed starting from zero.
  // Reindexing the the resulting relation to start from ciphertext zero.
  auto ctVarOffset =
      sliceExtractionLayout.getVarKindOffset(presburger::VarKind::Range);
  auto ctLowerBound = sliceExtractionLayout.getConstantBound64(
      presburger::BoundType::LB, ctVarOffset);
  if (!ctLowerBound) {
    return op.emitError() << "failed to get constant bound on ciphertext index";
  }
  auto zeroIndexedSliceLayout =
      shiftVar(sliceExtractionLayout, ctVarOffset, -ctLowerBound.value());

  Attribute kernelInfoAttr = cloneKernelInfoWithResultShape(
      op.getSource(), op.getResultType().getShape());
  LayoutAttr outputLayout = LayoutAttr::getFromIntegerRelation(
      op.getContext(), zeroIndexedSliceLayout);
  assignedLayouts.insert({op.getResult(), outputLayout});
  debugAssignLayout(op.getResult(), outputLayout);
  setResultLayoutAttr(op, kernelInfoAttr);
  return success();
}

CompatibilityResult LayoutPropagation::hasCompatibleArgumentLayouts(
    Operation* op) {
  LLVM_DEBUG(llvm::dbgs() << "Checking for compatible argument layouts for: "
                          << op->getName() << "\n");
  return TypeSwitch<Operation*, CompatibilityResult>(op)
      // Trivially true ops
      .Case<func::FuncOp, GenericOp, YieldOp, affine::AffineForOp,
            affine::AffineYieldOp, ConvertLayoutOp>(
          [&](auto op) { return CompatibilityResult{true, std::nullopt}; })
      // Ops with special rules
      .Case<DotOp, ReduceOp, MatmulOp, BatchMatmulOp, tensor::InsertSliceOp>(
          [&](auto op) { return hasCompatibleArgumentLayouts(op); })
      // By default, assume operands must all have the same layout.
      .Default([&](Operation* op) {
        std::optional<LayoutAttr> firstFoundLayout;
        // All operands require a layout unless the op implements
        // OperandLayoutRequirementOpInterface, which allows it to specify which
        // subset of operands require layouts.
        auto layoutReqIface = dyn_cast<OperandLayoutRequirementOpInterface>(op);

        for (auto& opOperand : op->getOpOperands()) {
          bool secretness = isSecret(opOperand.get(), solver);
          if (layoutReqIface && !layoutReqIface.operandRequiresLayout(
                                    opOperand.getOperandNumber(), secretness)) {
            continue;
          }

          if (!assignedLayouts.contains(opOperand.get())) {
            // If the operand has no layout, we can't propagate layout
            // information to the result.
            return CompatibilityResult{
                false, op->emitError("operand has no assigned layout")};
          }
          LayoutAttr layout = getComposedLayoutAttr(opOperand.get());

          if (!firstFoundLayout.has_value()) firstFoundLayout = layout;
          if (layout != firstFoundLayout.value()) {
            return CompatibilityResult{false, std::nullopt};
          }
        }

        return CompatibilityResult{true, std::nullopt};
      });
}

CompatibilityResult LayoutPropagation::hasCompatibleArgumentLayouts(DotOp op) {
  LayoutAttr lhsLayout = getComposedLayoutAttr(op.getOperand(0));
  LayoutAttr rhsLayout = getComposedLayoutAttr(op.getOperand(1));

  if (lhsLayout != rhsLayout) {
    return {false, std::nullopt};
  }

  // Outs operand (operand 2) should be 0D.
  LayoutAttr outsLayout = getComposedLayoutAttr(op.getOperand(2));
  if (outsLayout.getIntegerRelation().getNumDomainVars() != 0) {
    return {false, std::nullopt};
  }

  return {true, std::nullopt};
}

CompatibilityResult LayoutPropagation::hasCompatibleArgumentLayouts(
    ReduceOp op) {
  // The arguments of a ReduceOp are the tensor(s) to reduce and the
  // initializer values for the reduction.
  for (const auto& [input, init] : llvm::zip(op.getInputs(), op.getInits())) {
    if (!assignedLayouts.contains(input)) {
      return {false, op->emitError("input tensor has no assigned layout")};
    }
    if (!assignedLayouts.contains(init)) {
      return {false,
              op->emitError("initializer tensor has no assigned layout")};
    }

    LayoutAttr inputLayout = getComposedLayoutAttr(input);
    LayoutAttr initLayout = getComposedLayoutAttr(init);
    LayoutAttr reducedInputLayout =
        convertLayoutForReduce(inputLayout, op.getDimensions());

    if (reducedInputLayout != initLayout) {
      return {false, std::nullopt};
    }
  }

  return {true, std::nullopt};
}

CompatibilityResult LayoutPropagation::hasCompatibleArgumentLayouts(
    MatmulOp op) {
  Value lhs = op.getOperand(0);
  Value rhs = op.getOperand(1);

  if (!assignedLayouts.contains(lhs)) {
    return {false, op->emitError("LHS operand has no assigned layout")};
  }
  if (!assignedLayouts.contains(rhs)) {
    return {false, op->emitError("RHS operand has no assigned layout")};
  }

  return {true, std::nullopt};
}

CompatibilityResult LayoutPropagation::hasCompatibleArgumentLayouts(
    BatchMatmulOp op) {
  Value lhs = op.getOperand(0);
  Value rhs = op.getOperand(1);

  if (!assignedLayouts.contains(lhs)) {
    return {false, op->emitError("LHS operand has no assigned layout")};
  }
  if (!assignedLayouts.contains(rhs)) {
    return {false, op->emitError("RHS operand has no assigned layout")};
  }

  return {true, std::nullopt};
}

CompatibilityResult LayoutPropagation::hasCompatibleArgumentLayouts(
    tensor::InsertSliceOp op) {
  // The arguments of a tensor::InsertSliceOp are the tensors to insert and the
  // tensor to insert into.
  auto insert = op.getSource();
  auto dest = op.getDest();

  if (!assignedLayouts.contains(insert)) {
    return {false, op->emitError("input tensor has no assigned layout")};
  }
  if (!assignedLayouts.contains(dest)) {
    return {false, op->emitError("destination tensor has no assigned layout")};
  }

  LayoutAttr insertLayout = getComposedLayoutAttr(insert);
  LayoutAttr destLayout = getComposedLayoutAttr(dest);
  auto destRelation = destLayout.getIntegerRelation();
  auto compatibleDestRelation = pushSliceLayoutThroughInsertSlice(
      SmallVector<int64_t>(op.getStaticSizes()), op.getResultType().getShape(),
      insertLayout.getIntegerRelation());
  assert(succeeded(compatibleDestRelation) && "expected to infer slice layout");

  // We compare layout attributes, which is a strict equality check on the
  // attribute string. This is more strict than comparing the IntegerRelations,
  // but equality checking on the relations is expensive.
  if (LayoutAttr::getFromIntegerRelation(
          op.getContext(), compatibleDestRelation.value()) != destLayout) {
    return {false, std::nullopt};
  }

  return {true, std::nullopt};
}

void LayoutPropagation::rectifyIncompatibleOperandLayouts(Operation* op) {
  LLVM_DEBUG({
    auto diag = op->emitRemark() << "Inserting layout conversion op due to "
                                    "disagreeing operand layouts";
    auto& note = diag.attachNote();
    for (auto operand : op->getOperands()) {
      LayoutAttr operandLayout;
      if (assignedLayouts.contains(operand))
        operandLayout = getComposedLayoutAttr(operand);
      note << "\n- Operand: " << operand << "; Layout: " << operandLayout;
    }
  });

  TypeSwitch<Operation*>(op)
      // These ops shouldn't rectify operand layouts
      .Case<func::FuncOp, func::ReturnOp, secret::GenericOp, secret::YieldOp,
            ConvertLayoutOp>([&](auto op) { return; })
      // Ops with special rules
      .Case<DotOp, ReduceOp, tensor::InsertOp, tensor::InsertSliceOp>(
          [&](auto op) { return rectifyIncompatibleOperandLayouts(op); })
      .Default([&](Operation* op) {
        // Prefer an operand with a secret ciphertext layout over a cleartext
        // constant: converting a cleartext operand is free because
        // FoldConvertLayoutIntoAssignLayout folds the conversion into its
        // assign_layout at compile time, whereas converting a secret ciphertext
        // incurs expensive homomorphic rotations and shift networks.
        mlir::IRRewriter builder(&getContext());
        auto isSecretOperand = [&](Value v) {
          return assignedLayouts.contains(v) && isSecret(v, solver);
        };
        auto it = llvm::find_if(op->getOperands(), isSecretOperand);
        if (it == op->getOperands().end()) {
          it = llvm::find_if(op->getOperands(), [this](Value v) {
            return assignedLayouts.contains(v);
          });
        }
        LayoutAttr targetLayout;
        if (it != op->getOperands().end()) {
          targetLayout = getComposedLayoutAttr(*it);
        } else if (op->getNumOperands() > 0) {
          FailureOr<LayoutAttr> defaultLayout =
              defaultLayoutForType(op->getOperand(0).getType());
          if (failed(defaultLayout)) return;
          targetLayout = *defaultLayout;
        } else {
          return;
        }

        for (auto& opOperand : op->getOpOperands()) {
          if (!assignedLayouts.contains(opOperand.get())) continue;
          LayoutAttr sourceLayout = getComposedLayoutAttr(opOperand.get());

          if (sourceLayout != targetLayout) {
            if (auto assignOp = opOperand.get().getDefiningOp<AssignLayoutOp>();
                assignOp && assignOp->hasOneUse()) {
              assignOp.setLayoutAttr(targetLayout);
              assignedLayouts[opOperand.get()] = targetLayout;
              setAttributeAssociatedWith(
                  opOperand.get(),
                  tensor_ext::TensorExtDialect::kLayoutAttrName, targetLayout);
              debugAssignLayout(opOperand.get(), targetLayout);
              continue;
            }
            // If the relations are provably equal, re-annotate in place without
            // emitting a redundant conversion op.
            if (opOperand.get().hasOneUse() &&
                targetLayout.getIntegerRelation().getSpace().isCompatible(
                    sourceLayout.getIntegerRelation().getSpace())) {
              if (isRelationEqual(targetLayout.getIntegerRelation(),
                                  sourceLayout.getIntegerRelation())) {
                assignedLayouts[opOperand.get()] = targetLayout;
                setAttributeAssociatedWith(
                    opOperand.get(),
                    tensor_ext::TensorExtDialect::kLayoutAttrName,
                    targetLayout);
                debugAssignLayout(opOperand.get(), targetLayout);
                continue;
              }
            }
            builder.setInsertionPoint(op);
            ConvertLayoutOp convertOp =
                ConvertLayoutOp::create(builder, op->getLoc(), opOperand.get(),
                                        sourceLayout, targetLayout);

            auto* lattice = solver->getOrCreateState<SecretnessLattice>(
                convertOp.getResult());
            lattice->getValue().setSecretness(
                isSecret(opOperand.get(), solver));

            // Layout of the result is the same as the target layout of the
            // conversion. Mostly this is done for consistency: all ops have
            // an attribute describing the layout of their results.
            OpBuilder builder(&getContext());
            assignedLayouts.insert({convertOp.getResult(), targetLayout});
            setResultLayoutAttr(convertOp,
                                findAttributeAssociatedWith(opOperand.get(),
                                                            kKernelInfoAttrName)
                                    .value_or(Attribute()));
            op->setOperand(opOperand.getOperandNumber(), convertOp.getResult());
          }
        }
      });
}

void LayoutPropagation::rectifyIncompatibleOperandLayouts(DotOp op) {
  mlir::IRRewriter builder(&getContext());
  builder.setInsertionPoint(op);

  LayoutAttr lhsLayout = getComposedLayoutAttr(op.getOperand(0));
  LayoutAttr rhsLayout = getComposedLayoutAttr(op.getOperand(1));

  if (lhsLayout != rhsLayout) {
    // Convert RHS to match LHS layout
    ConvertLayoutOp convertOp = ConvertLayoutOp::create(
        builder, op->getLoc(), op.getOperand(1), rhsLayout, lhsLayout);
    auto* lattice =
        solver->getOrCreateState<SecretnessLattice>(convertOp.getResult());
    lattice->getValue().setSecretness(isSecret(op.getOperand(1), solver));

    builder.replaceUsesWithIf(
        op.getOperand(1), convertOp.getResult(),
        [&](OpOperand& operand) { return operand.getOwner() == op; });
    assignedLayouts.insert({convertOp.getResult(), lhsLayout});
  }

  // Ensure Outs is 0D.
  LayoutAttr outsLayout = getComposedLayoutAttr(op.getOperand(2));
  if (outsLayout.getIntegerRelation().getNumDomainVars() != 0) {
    FailureOr<LayoutAttr> default0D =
        defaultLayoutForType(op.getOperand(2).getType());
    if (succeeded(default0D)) {
      ConvertLayoutOp convertOp =
          ConvertLayoutOp::create(builder, op->getLoc(), op.getOperand(2),
                                  outsLayout, default0D.value());
      auto* lattice =
          solver->getOrCreateState<SecretnessLattice>(convertOp.getResult());
      lattice->getValue().setSecretness(isSecret(op.getOperand(2), solver));

      builder.replaceUsesWithIf(
          op.getOperand(2), convertOp.getResult(),
          [&](OpOperand& operand) { return operand.getOwner() == op; });
      assignedLayouts.insert({convertOp.getResult(), default0D.value()});
    }
  }
}

void LayoutPropagation::rectifyIncompatibleOperandLayouts(ReduceOp op) {
  mlir::IRRewriter builder(&getContext());
  builder.setInsertionPoint(op);

  for (const auto& [input, init] : llvm::zip(op.getInputs(), op.getInits())) {
    LayoutAttr inputLayout = getComposedLayoutAttr(input);
    LayoutAttr initLayout = getComposedLayoutAttr(init);
    LayoutAttr reducedInputLayout =
        convertLayoutForReduce(inputLayout, op.getDimensions());

    if (reducedInputLayout != initLayout) {
      ConvertLayoutOp convertOp = ConvertLayoutOp::create(
          builder, op->getLoc(), init, initLayout, reducedInputLayout);
      Value toReplace = convertOp.getResult();
      builder.replaceUsesWithIf(init, toReplace, [&](OpOperand& operand) {
        return operand.getOwner() == op;
      });
      assignedLayouts.insert({toReplace, reducedInputLayout});
      setResultLayoutAttr(convertOp);
    }
  }
}

void LayoutPropagation::rectifyIncompatibleOperandLayouts(
    tensor::InsertSliceOp op) {
  // Update the dest tensor to align with the source tensor slice.
  LayoutAttr sliceLayout = getComposedLayoutAttr(op.getSource());
  auto maybeNewLayout = pushSliceLayoutThroughInsertSlice(
      SmallVector<int64_t>(op.getStaticSizes()), op.getResultType().getShape(),
      sliceLayout.getIntegerRelation());
  assert(succeeded(maybeNewLayout) && "expected to infer slice layout");
  auto newLayoutAttr = LayoutAttr::getFromIntegerRelation(
      op.getContext(), maybeNewLayout.value());

  if (!assignedLayouts.contains(op.getDest())) {
    assignedLayouts.insert({op.getDest(), newLayoutAttr});
    return;
  }

  mlir::IRRewriter builder(&getContext());
  builder.setInsertionPoint(op);
  LayoutAttr destLayout = getComposedLayoutAttr(op.getDest());
  auto convertLayoutOp = ConvertLayoutOp::create(
      builder, op->getLoc(), op.getDest(), destLayout, newLayoutAttr);
  Value toReplace = convertLayoutOp.getResult();
  builder.replaceUsesWithIf(op.getDest(), toReplace, [&](OpOperand& operand) {
    return operand.getOwner() == op;
  });
  assignedLayouts.insert({toReplace, newLayoutAttr});
  setResultLayoutAttr(convertLayoutOp);
}

void LayoutPropagation::rectifyIncompatibleOperandLayouts(tensor::InsertOp op) {
  // The scalar input's layout must be made to match the tensor input's layout.
  // Ideally we could be smarter here, for example to take a scalar layout
  // which is a single slot in a single ciphertext, and insert it into the
  // correct slot of the single ciphertext of the dest tensor via a mask and
  // rotate. But this kernel is not expected to be used with ciphertext inputs,
  // so we instead incur the cost of a layout conversion before the insert.
  mlir::IRRewriter builder(&getContext());
  builder.setInsertionPoint(op);

  LayoutAttr destLayout = getComposedLayoutAttr(op.getDest());
  LayoutAttr scalarLayout = getComposedLayoutAttr(op.getScalar());
  IntegerRelation destRel = destLayout.getIntegerRelation();
  std::optional<int64_t> destNumCts = destRel.getConstantBound64(
      presburger::BoundType::UB,
      // First range var is the ciphertext index, always has lower bound of
      // zero
      destRel.getVarKindOffset(presburger::VarKind::Range));
  assert(destNumCts.has_value());

  std::string newScalarLayoutStr = llvm::formatv(
      "{ [] -> [ct, slot] : 0 <= ct <= {0} and 0 <= slot <= {1} }",
      destNumCts.value(), minSlotCount - 1);
  LayoutAttr newScalarLayout =
      LayoutAttr::get(op.getContext(), newScalarLayoutStr);

  ConvertLayoutOp convertOp = ConvertLayoutOp::create(
      builder, op->getLoc(), op.getScalar(), scalarLayout, newScalarLayout);
  Value toReplace = convertOp.getResult();
  builder.replaceUsesWithIf(op.getScalar(), toReplace, [&](OpOperand& operand) {
    return operand.getOwner() == op;
  });
  assignedLayouts.insert({toReplace, newScalarLayout});
  setResultLayoutAttr(convertOp);
}

void LayoutPropagation::passLayoutThroughOp(Operation* op) {
  // All inputs have the same layout, so just propagate it to all results
  LayoutAttr layout = getComposedLayoutAttr(op->getOperand(0));
  for (Value result : op->getResults()) {
    assignedLayouts.insert({result, layout});
    debugAssignLayout(result, layout);
  }
  setResultLayoutAttr(op);
}

FailureOr<LayoutAttr> LayoutPropagation::defaultLayoutForScalarType(
    Type scalarType) {
  auto* ctx = scalarType.getContext();
  OpBuilder builder(ctx);

  // Only support Int/Index/Float scalars,
  // fail for any other random type that might make it here.
  if (!isa<FloatType, IntegerType, IndexType>(scalarType)) {
    return failure();
  }

  std::string relationStr = llvm::formatv(
      "{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= {0} }", minSlotCount - 1);
  return LayoutAttr::get(ctx, relationStr);
}

FailureOr<LayoutAttr> LayoutPropagation::defaultLayoutForType(Type type) {
  Type ty = type;
  if (SecretType secretType = dyn_cast<SecretType>(type)) {
    ty = secretType.getValueType();
  }

  RankedTensorType tensorType = dyn_cast<RankedTensorType>(ty);
  if (!tensorType) {
    LLVM_DEBUG(llvm::dbgs() << "defaultLayoutForType: not a tensor, calling "
                               "defaultLayoutForScalarType\n");
    return defaultLayoutForScalarType(ty);
  }

  LLVM_DEBUG(llvm::dbgs() << "defaultLayoutForType: tensor rank="
                          << tensorType.getRank() << "\n");
  if (tensorType.getRank() == 0) {
    LLVM_DEBUG(llvm::dbgs() << "defaultLayoutForType: rank 0 tensor, calling "
                               "defaultLayoutForScalarType\n");
    return defaultLayoutForScalarType(tensorType.getElementType());
  }

  // By default, each tensor is laid out in row-major order. The slots will be
  // repeated each N elements where N is the next power of two of the total
  // number of data elements.
  LLVM_DEBUG(llvm::dbgs() << "getting row-major layout map for type="
                          << tensorType << "\n");
  IntegerRelation relation =
      getRowMajorLayoutRelation(tensorType, minSlotCount);
  return LayoutAttr::getFromIntegerRelation(tensorType.getContext(), relation);
}

void LayoutPropagation::setResultLayoutAttr(Operation* op,
                                            Attribute kernelInfo) {
  // If no kernelInfo attribute is present, check if any of the operands have
  // one.
  if (!kernelInfo && op->getNumOperands() > 0) {
    for (Value operand : op->getOperands()) {
      auto info = findAttributeAssociatedWith(operand, kKernelInfoAttrName);
      if (succeeded(info)) {
        kernelInfo = info.value();
        break;
      }
    }
  }
  if (kernelInfo) {
    for (Value result : op->getResults()) {
      setAttributeAssociatedWith(result, kKernelInfoAttrName, kernelInfo);
    }
  }

  OpBuilder builder(&getContext());
  SmallVector<Attribute> resultLayouts = llvm::map_to_vector(
      op->getResults(),
      [&](Value result) -> Attribute { return assignedLayouts.at(result); });

  if (op->getNumResults() == 1) {
    op->setAttr(tensor_ext::TensorExtDialect::kLayoutAttrName,
                resultLayouts.front());
    return;
  }
  op->setAttr(tensor_ext::TensorExtDialect::kLayoutAttrName,
              builder.getArrayAttr(resultLayouts));
}

void LayoutPropagation::runOnOperation() {
  // `strided-layout-propagation` has assigned the layouts of this module.
  if (tensor_ext::usesStridedLayouts(getOperation())) return;

  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  if (failed(solver.initializeAndRun(getOperation()))) {
    getOperation()->emitOpError() << "Failed to run secretness analysis.\n";
    signalPassFailure();
    return;
  }
  this->solver = &solver;

  LLVM_DEBUG(llvm::dbgs() << "Running layout propagation on operation: "
                          << getOperation()->getName() << "\n");
  WalkResult result =
      getOperation()->walk<WalkOrder::PreOrder>([&](Operation* op) {
        LogicalResult result = visitOperation(op);
        if (failed(result)) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

  if (result.wasInterrupted()) {
    signalPassFailure();
  }

  RewritePatternSet cleanupPatterns(&getContext());
  cleanupPatterns.add<tensor_ext::FoldConvertLayoutIntoAssignLayoutPattern>(
      &getContext());
  (void)applyPatternsGreedily(getOperation(), std::move(cleanupPatterns));
};

}  // namespace heir
}  // namespace mlir
