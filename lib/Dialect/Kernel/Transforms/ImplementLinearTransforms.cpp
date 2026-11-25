#include "lib/Dialect/Kernel/Transforms/ImplementLinearTransforms.h"

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "lib/Dialect/Kernel/IR/KernelOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/IRMaterializingVisitor.h"
#include "lib/Kernel/LinearTransformImplementation.h"
#include "lib/Kernel/Utils.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "lib/Utils/Layout/LinearSpec.h"
#include "lib/Utils/Layout/SparseBsgs.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

#define GEN_PASS_DEF_IMPLEMENTLINEARTRANSFORMS
#include "lib/Dialect/Kernel/Transforms/Passes.h.inc"

namespace {

// One ciphertext row of a tensor<n x N>, as a tensor<1 x N>.
Value extractRow(ImplicitLocOpBuilder& b, Value tensor, int64_t row,
                 RankedTensorType rowType) {
  SmallVector<OpFoldResult> offsets = {b.getIndexAttr(row), b.getIndexAttr(0)};
  SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1),
                                     b.getIndexAttr(rowType.getDimSize(1))};
  SmallVector<OpFoldResult> strides = {b.getIndexAttr(1), b.getIndexAttr(1)};
  return tensor::ExtractSliceOp::create(b, rowType, tensor, offsets, sizes,
                                        strides);
}

Value insertRow(ImplicitLocOpBuilder& b, Value rowValue, Value dest,
                int64_t row, RankedTensorType rowType) {
  SmallVector<OpFoldResult> offsets = {b.getIndexAttr(row), b.getIndexAttr(0)};
  SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1),
                                     b.getIndexAttr(rowType.getDimSize(1))};
  SmallVector<OpFoldResult> strides = {b.getIndexAttr(1), b.getIndexAttr(1)};
  return tensor::InsertSliceOp::create(b, rowValue, dest, offsets, sizes,
                                       strides);
}

// Splits the op into one kernel.linear_transform per (output, input) block,
// for backends that evaluate a single-ciphertext transform natively.
Value splitIntoBlocks(BlockLinearTransformOp op, ImplicitLocOpBuilder& b,
                      RankedTensorType rowType) {
  ArrayRef<int64_t> outputCiphertexts = op.getOutputCiphertexts();
  ArrayRef<int64_t> inputCiphertexts = op.getInputCiphertexts();
  ArrayRef<int64_t> shifts = op.getShifts();

  struct Block {
    SmallVector<int64_t> rows;
    SmallVector<int64_t> shifts;
  };
  std::map<std::pair<int64_t, int64_t>, Block> blocks;
  for (size_t k = 0; k < shifts.size(); ++k) {
    Block& block = blocks[{outputCiphertexts[k], inputCiphertexts[k]}];
    block.rows.push_back(k);
    block.shifts.push_back(shifts[k]);
  }

  bool isFloat = isa<mlir::FloatType>(rowType.getElementType());
  int64_t numOut = op.getOutput().getType().getDimSize(0);
  std::vector<Value> outputs(numOut, Value());
  std::map<int64_t, Value> inputRows;
  for (const auto& [block, contents] : blocks) {
    auto [ctOut, ctIn] = block;
    if (!inputRows.contains(ctIn)) {
      inputRows[ctIn] = extractRow(b, op.getInput(), ctIn, rowType);
    }
    Value transformed = LinearTransformOp::create(
        b, rowType, inputRows[ctIn], op.getDiagonals(),
        b.getDenseI64ArrayAttr(contents.shifts),
        b.getDenseI64ArrayAttr(contents.rows), /*bsgs_ratio=*/nullptr);
    if (!outputs[ctOut]) {
      outputs[ctOut] = transformed;
    } else if (isFloat) {
      outputs[ctOut] = arith::AddFOp::create(b, outputs[ctOut], transformed);
    } else {
      outputs[ctOut] = arith::AddIOp::create(b, outputs[ctOut], transformed);
    }
  }

  Value result = tensor::EmptyOp::create(b, op.getOutput().getType().getShape(),
                                         rowType.getElementType());
  Value zeroRow;
  for (int64_t o = 0; o < numOut; ++o) {
    Value rowValue = outputs[o];
    if (!rowValue) {
      if (!zeroRow) {
        zeroRow = arith::ConstantOp::create(b, rowType, b.getZeroAttr(rowType));
      }
      rowValue = zeroRow;
    }
    result = insertRow(b, rowValue, result, o, rowType);
  }
  return result;
}

// Lowers the op to a baby-step giant-step evaluation over all blocks.
Value implementAsBsgs(BlockLinearTransformOp op, ImplicitLocOpBuilder& b,
                      RankedTensorType rowType, bool unroll) {
  ArrayRef<int64_t> outputCiphertexts = op.getOutputCiphertexts();
  ArrayRef<int64_t> inputCiphertexts = op.getInputCiphertexts();
  ArrayRef<int64_t> shifts = op.getShifts();
  std::vector<DiagonalKey> keys;
  keys.reserve(shifts.size());
  for (size_t k = 0; k < shifts.size(); ++k) {
    keys.push_back({outputCiphertexts[k], inputCiphertexts[k], shifts[k]});
  }

  SparseBsgsPlan plan = planSparseBsgs(keys, rowType.getDimSize(1));
  SparseBsgsOptions options;
  options.unroll = unroll;
  auto dag = implementSparseBsgs(plan, SSAValue(op.getInput()),
                                 SSAValue(op.getDiagonals()),
                                 op.getOutput().getType().getDimSize(0),
                                 mlirTypeToDagType(rowType), options);
  IRMaterializingVisitor visitor(rowType);
  return visitor.process(dag, b)[0];
}

struct ImplementLinearTransforms
    : impl::ImplementLinearTransformsBase<ImplementLinearTransforms> {
  using ImplementLinearTransformsBase::ImplementLinearTransformsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // getTargetConfig reports an error on a module without a backend, which
    // is a valid input here: it gets the generic lowering.
    bool hasBackend = llvm::any_of(module->getAttrs(), [](NamedAttribute attr) {
      return isa<UnitAttr>(attr.getValue()) &&
             attr.getName().getValue().starts_with("backend.");
    });
    bool native = false;
    if (hasBackend) {
      auto target = getTargetConfig(module);
      native = succeeded(target) && target->hasKernelLinearTransform;
    }

    SmallVector<BlockLinearTransformOp> ops;
    module->walk([&](BlockLinearTransformOp op) { ops.push_back(op); });
    for (BlockLinearTransformOp op : ops) {
      RankedTensorType inputType = op.getInput().getType();
      auto rowType = RankedTensorType::get({1, inputType.getDimSize(1)},
                                           inputType.getElementType());
      IRRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(op);
      ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      Value result = native ? splitIntoBlocks(op, b, rowType)
                            : implementAsBsgs(op, b, rowType, unroll);
      rewriter.replaceOp(op, result);
    }
  }
};

}  // namespace

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
