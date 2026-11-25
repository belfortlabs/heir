#include "lib/Transforms/LinearKernel/StridedLayoutPropagation.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/LinearKernel/LinearOp.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Layout/LinearSpec.h"
#include "lib/Utils/Layout/StridedLayout.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"  // from @llvm-project

#define DEBUG_TYPE "strided-layout-propagation"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_STRIDEDLAYOUTPROPAGATION
#include "lib/Transforms/LinearKernel/StridedLayoutPropagation.h.inc"

namespace {

using tensor_ext::AssignLayoutOp;
using tensor_ext::ConvertLayoutOp;
using tensor_ext::StridedLayoutAttr;

constexpr StringRef kLayoutAttrName =
    tensor_ext::TensorExtDialect::kLayoutAttrName;

std::optional<ArrayRef<int64_t>> staticShape(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.hasStaticShape()) return std::nullopt;
  return tensorType.getShape();
}

}  // namespace

struct StridedLayoutPropagation
    : impl::StridedLayoutPropagationBase<StridedLayoutPropagation> {
  using StridedLayoutPropagationBase::StridedLayoutPropagationBase;

  void runOnOperation() override {
    WalkResult result =
        getOperation()->walk<WalkOrder::PreOrder>([&](Operation* op) {
          return failed(visit(op)) ? WalkResult::interrupt()
                                   : WalkResult::advance();
        });
    if (result.wasInterrupted()) signalPassFailure();
  }

 private:
  // Layouts of secret values, and of cleartext values that were packed for a
  // slotwise operation.
  llvm::DenseMap<Value, StridedLayout> layouts;
  llvm::DenseSet<Value> cleartexts;

  bool isSecret(Value value) const {
    return layouts.contains(value) && !cleartexts.contains(value);
  }

  void setLayout(Value value, const StridedLayout& layout) {
    layouts[value] = layout;
    setAttributeAssociatedWith(
        value, kLayoutAttrName,
        StridedLayoutAttr::getFromStridedLayout(&getContext(), layout));
  }

  StridedLayoutAttr attrOf(const StridedLayout& layout) {
    return StridedLayoutAttr::getFromStridedLayout(&getContext(), layout);
  }

  // A value whose layout describes a tensor of another shape is a padded view
  // of that tensor.
  bool isView(Value value) const {
    auto shape = staticShape(value.getType());
    auto it = layouts.find(value);
    return shape && it != layouts.end() &&
           ArrayRef<int64_t>(it->second.shape) != *shape;
  }

  // True if every user of a padded value reads it as a linear operation, so
  // the padding never has to exist in a ciphertext.
  bool isOnlyReadLinearly(Value value) {
    for (OpOperand& use : value.getUses()) {
      Operation* user = use.getOwner();
      if (isViewReshape(user)) {
        if (!isOnlyReadLinearly(user->getResult(0))) return false;
        continue;
      }
      auto linalgOp = dyn_cast<linalg::LinalgOp>(user);
      if (!linalgOp || !linalgOp.isDpsInput(&use) ||
          !matchLinearOp(user, use.getOperandNumber())) {
        return false;
      }
    }
    return true;
  }

  // Replaces operand `operand` of `op` by a copy of it in `layout`.
  void convertOperand(Operation* op, OpOperand& operand,
                      const StridedLayout& layout) {
    OpBuilder builder(op);
    Value value = operand.get();
    Value replacement;
    if (isSecret(value)) {
      if (layouts[value] == layout) return;
      replacement =
          ConvertLayoutOp::create(builder, op->getLoc(), value,
                                  attrOf(layouts[value]), attrOf(layout))
              .getResult();
    } else {
      replacement =
          AssignLayoutOp::create(builder, op->getLoc(), value, attrOf(layout))
              .getResult();
      cleartexts.insert(replacement);
    }
    setLayout(replacement, layout);
    operand.set(replacement);
  }

  LogicalResult visit(Operation* op) {
    if (auto func = dyn_cast<func::FuncOp>(op)) return visitFunc(func);
    if (auto generic = dyn_cast<secret::GenericOp>(op)) {
      for (OpOperand& operand : generic->getOpOperands()) {
        auto it = layouts.find(operand.get());
        if (it == layouts.end()) continue;
        setLayout(generic.getRegion().getArgument(operand.getOperandNumber()),
                  StridedLayout(it->second));
      }
      return success();
    }
    if (auto yield = dyn_cast<secret::YieldOp>(op)) {
      auto generic = cast<secret::GenericOp>(yield->getParentOp());
      for (OpOperand& operand : yield->getOpOperands()) {
        if (!isSecret(operand.get()) || isView(operand.get())) {
          return op->emitOpError() << "result " << operand.getOperandNumber()
                                   << " of the secret region has no layout";
        }
        setLayout(generic.getResult(operand.getOperandNumber()),
                  StridedLayout(layouts[operand.get()]));
      }
      return success();
    }
    if (auto ret = dyn_cast<func::ReturnOp>(op)) {
      auto func = cast<func::FuncOp>(ret->getParentOp());
      for (OpOperand& operand : ret->getOpOperands()) {
        auto it = layouts.find(operand.get());
        if (it == layouts.end()) continue;
        func.setResultAttr(operand.getOperandNumber(), kLayoutAttrName,
                           attrOf(it->second));
      }
      return success();
    }

    // Everything else is handled only directly inside a secret region, and
    // only if it touches a secret value.
    if (!isa_and_nonnull<secret::GenericOp>(op->getParentOp())) {
      return success();
    }
    if (llvm::none_of(op->getOperands(),
                      [&](Value v) { return isSecret(v); })) {
      return success();
    }

    if (isa<tensor::PadOp>(op) && isStaticZeroPad(op) &&
        isOnlyReadLinearly(op->getResult(0))) {
      setLayout(op->getResult(0), StridedLayout(layouts[op->getOperand(0)]));
      return success();
    }
    if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op) &&
        isView(op->getOperand(0))) {
      // A reshape of a padded view; `isOnlyReadLinearly` has checked that it
      // is one that `isViewReshape` accepts.
      setLayout(op->getResult(0), StridedLayout(layouts[op->getOperand(0)]));
      return success();
    }
    if (auto insert = dyn_cast<tensor::InsertSliceOp>(op)) {
      return visitInsertSlice(insert);
    }
    // Operations that only rename or select elements cost nothing when the
    // layout of the result is again strided.
    if (Value source = op->getOperand(0);
        isSecret(source) && !isView(source) && op->getNumResults() == 1) {
      if (std::optional<StridedLayout> out =
              relabeledLayout(op, layouts[source])) {
        setLayout(op->getResult(0), *out);
        return success();
      }
    }
    if (isa<tensor::PadOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp,
            tensor::ExtractSliceOp>(op) ||
        isa<linalg::LinalgOp>(op)) {
      return visitLinear(op);
    }
    return visitSlotwise(op);
  }

  LogicalResult visitFunc(func::FuncOp func) {
    if (func.isDeclaration()) return success();
    for (BlockArgument arg : func.getArguments()) {
      auto secretType = dyn_cast<secret::SecretType>(arg.getType());
      if (!secretType) continue;
      auto shape = staticShape(secretType.getValueType());
      if (!shape || shape->empty()) {
        return func.emitError()
               << "argument " << arg.getArgNumber()
               << " must be a secret tensor of static shape and nonzero rank";
      }
      StridedLayout layout = StridedLayout::rowMajor(*shape, minSlotCount);
      layouts[arg] = layout;
      func.setArgAttr(arg.getArgNumber(), kLayoutAttrName, attrOf(layout));
    }
    return success();
  }

  LogicalResult visitLinear(Operation* op) {
    // The secret operand among the inputs.
    int64_t secretOperand = -1;
    unsigned numInputs = 1;
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      numInputs = linalgOp.getNumDpsInputs();
    }
    for (unsigned i = 0; i < numInputs; ++i) {
      if (!isSecret(op->getOperand(i))) continue;
      if (secretOperand >= 0) {
        return op->emitOpError()
               << "has more than one secret input; only operations that are "
                  "linear in one secret tensor are supported";
      }
      secretOperand = i;
    }
    if (secretOperand < 0) {
      return op->emitOpError() << "has a secret output operand but no secret "
                                  "input, which is not supported";
    }
    std::optional<LinearOp> linear = matchLinearOp(op, secretOperand);
    if (!linear) {
      return op->emitOpError()
             << "is not a linear operation in its secret operand with "
                "cleartext coefficients";
    }

    const StridedLayout& in = layouts[linear->source];
    StridedLayout out = inducedOutputLayout(linear->spec, in);
    LLVM_DEBUG(llvm::dbgs() << "linear op " << op->getName() << ": "
                            << computeDiagonalKeys(linear->spec, in, out).size()
                            << " diagonals\n");

    if (linear->init) {
      OpOperand* initOperand = cast<linalg::LinalgOp>(op).getDpsInitOperand(0);
      convertOperand(op, *initOperand, out);
    }
    setLayout(op->getResult(0), out);
    return success();
  }

  // The inserted tensor is moved into its box by a transform. The result
  // keeps the layout of the tensor inserted into, or, when that tensor is
  // empty, gets a layout that follows the inserted tensor.
  LogicalResult visitInsertSlice(tensor::InsertSliceOp op) {
    Value source = op.getSource();
    Value dest = op.getDest();
    std::optional<LinearOp> linear = matchLinearOp(op, 0);
    if (!linear) {
      return op.emitOpError() << "requires static shapes and bounds";
    }
    if (!isSecret(source) || isView(source) || isView(dest) ||
        (!isSecret(dest) && !isZeroTensor(dest))) {
      return op.emitOpError()
             << "must insert a secret tensor into a secret or empty tensor";
    }
    setLayout(op.getResult(),
              isSecret(dest)
                  ? StridedLayout(layouts[dest])
                  : inducedOutputLayout(linear->spec, layouts[source]));
    return success();
  }

  LogicalResult visitSlotwise(Operation* op) {
    if (op->getNumRegions() != 0 || op->getNumResults() != 1) {
      return op->emitOpError() << "is not supported on secret tensors";
    }
    auto shape = staticShape(op->getResult(0).getType());
    if (!shape) {
      return op->emitOpError() << "is not supported on secret tensors";
    }
    const StridedLayout* target = nullptr;
    for (Value operand : op->getOperands()) {
      auto operandShape = staticShape(operand.getType());
      if (!operandShape) continue;
      if (*operandShape != *shape) {
        return op->emitOpError() << "is not supported on secret tensors: its "
                                    "operands and result differ in shape";
      }
      if (isView(operand)) {
        return op->emitOpError() << "reads a padded tensor slot by slot";
      }
    }
    // Operands in different layouts are joined in the layout that is cheapest
    // to convert the others to, counted in diagonals.
    int64_t bestCost = -1;
    for (Value candidate : op->getOperands()) {
      if (!isSecret(candidate) || !staticShape(candidate.getType())) continue;
      const StridedLayout& layout = layouts[candidate];
      int64_t cost = 0;
      for (Value other : op->getOperands()) {
        if (!isSecret(other) || !staticShape(other.getType()) ||
            layouts[other] == layout) {
          continue;
        }
        cost += computeDiagonalKeys(identityLinearOp(other).spec,
                                    layouts[other], layout)
                    .size();
      }
      if (bestCost < 0 || cost < bestCost) {
        bestCost = cost;
        target = &layouts[candidate];
      }
    }
    if (!target) {
      return op->emitOpError() << "is not supported on secret tensors";
    }
    StridedLayout layout = *target;
    for (OpOperand& operand : op->getOpOperands()) {
      if (staticShape(operand.get().getType())) {
        convertOperand(op, operand, layout);
      }
    }
    setLayout(op->getResult(0), layout);
    return success();
  }
};

}  // namespace heir
}  // namespace mlir
