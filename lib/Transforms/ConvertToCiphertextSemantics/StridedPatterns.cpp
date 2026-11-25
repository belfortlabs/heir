#include "lib/Transforms/ConvertToCiphertextSemantics/StridedPatterns.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "lib/Dialect/Kernel/IR/KernelOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/AssignLayout.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"
#include "lib/Transforms/LinearKernel/LinearOp.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/ContextAwareTypeConversion.h"
#include "lib/Utils/Layout/LinearSpec.h"
#include "lib/Utils/Layout/StridedLayout.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project

namespace mlir {
namespace heir {

namespace {

using tensor_ext::StridedLayoutAttr;

constexpr StringRef kLayoutAttrName =
    tensor_ext::TensorExtDialect::kLayoutAttrName;

// The largest lookup table from (output ciphertext, input ciphertext, shift)
// to a diagonal's row that the packing loop may use.
constexpr int64_t kMaxRowTableSize = int64_t{1} << 26;

Value indexConstant(OpBuilder& b, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(b, loc, value);
}

// `row . ivs + constant` as an `index` value.
Value emitAffine(OpBuilder& b, Location loc, const std::vector<int64_t>& row,
                 int64_t constant, ValueRange ivs) {
  Value result = indexConstant(b, loc, constant);
  for (size_t loop = 0; loop < row.size(); ++loop) {
    if (row[loop] == 0) continue;
    Value term = ivs[loop];
    if (row[loop] != 1) {
      term =
          arith::MulIOp::create(b, loc, term, indexConstant(b, loc, row[loop]));
    }
    result = arith::AddIOp::create(b, loc, result, term);
  }
  return result;
}

// `value mod modulus` with a result in `[0, modulus)` for negative values.
Value emitFloorMod(OpBuilder& b, Location loc, Value value, int64_t modulus) {
  Value m = indexConstant(b, loc, modulus);
  Value rem = arith::RemSIOp::create(b, loc, value, m);
  return arith::RemSIOp::create(b, loc, arith::AddIOp::create(b, loc, rem, m),
                                m);
}

// True if `row . d + constant` can leave `[0, extent)` for `d` in the box.
bool canLeaveBounds(const std::vector<int64_t>& row, int64_t constant,
                    const std::vector<int64_t>& loops, int64_t extent) {
  int64_t lo = constant;
  int64_t hi = constant;
  for (size_t loop = 0; loop < row.size(); ++loop) {
    int64_t reach = row[loop] * (loops[loop] - 1);
    (reach < 0 ? lo : hi) += reach;
  }
  return lo < 0 || hi >= extent;
}

// Emits the diagonals of a transform as a `tensor<numKeys x numSlots>`: a
// loop nest over the iterations of the spec that adds each coefficient into
// its place. This mirrors `packDiagonals`.
FailureOr<Value> emitDiagonals(OpBuilder& b, Location loc,
                               const LinearOp& linear, const StridedLayout& in,
                               const StridedLayout& out,
                               ArrayRef<DiagonalKey> keys, Type elementType) {
  const LinearSpec& spec = linear.spec;
  ShiftRule rule = getShiftRule(in, out);
  int64_t numSlots = out.numSlots;
  int64_t numIn = in.numCiphertexts();
  int64_t numOut = rule.writeFollowsInput ? 1 : out.numCiphertexts();

  // rowOf[(ctOut * numIn + ctIn) * modulus + shift] is the diagonal's row.
  int64_t tableSize = numOut * numIn * rule.modulus;
  if (tableSize > kMaxRowTableSize) {
    return emitError(loc) << "linear transform between " << numIn << " and "
                          << numOut
                          << " ciphertexts needs a diagonal lookup table of "
                          << tableSize << " entries, which is not supported";
  }
  std::vector<int32_t> rowOf(tableSize, -1);
  for (size_t row = 0; row < keys.size(); ++row) {
    rowOf[(keys[row].ctOut * numIn + keys[row].ctIn) * rule.modulus +
          keys[row].shift] = row;
  }
  auto tableType = RankedTensorType::get({tableSize}, b.getI32Type());
  Value table = arith::ConstantOp::create(
      b, loc, DenseIntElementsAttr::get(tableType, ArrayRef<int32_t>(rowOf)));

  auto diagonalsType = RankedTensorType::get(
      {static_cast<int64_t>(keys.size()), numSlots}, elementType);
  // Filled at run time: a constant of this size would be stored in the
  // compiled module once per transform.
  Value zeros = linalg::FillOp::create(
                    b, loc,
                    ValueRange{arith::ConstantOp::create(
                        b, loc, b.getZeroAttr(elementType))},
                    ValueRange{tensor::EmptyOp::create(
                        b, loc, diagonalsType.getShape(), elementType)})
                    .getResult(0);

  SmallVector<Value> lbs, ubs, steps;
  for (int64_t extent : spec.loops) {
    lbs.push_back(indexConstant(b, loc, 0));
    ubs.push_back(indexConstant(b, loc, extent));
    steps.push_back(indexConstant(b, loc, 1));
  }

  auto addTerm = [&](OpBuilder& b, Location loc, ValueRange ivs,
                     ArrayRef<Value> inIdx, Value diagonals) -> Value {
    SmallVector<Value> outIdx;
    for (size_t axis = 0; axis < spec.outMap.size(); ++axis) {
      outIdx.push_back(
          emitAffine(b, loc, spec.outMap[axis], spec.outOffset[axis], ivs));
    }
    Value inPos = emitStridedPosition(b, loc, in, inIdx);
    Value outPos = emitStridedPosition(b, loc, out, outIdx);
    Value slots = indexConstant(b, loc, numSlots);
    Value shift = emitFloorMod(
        b, loc, arith::SubIOp::create(b, loc, inPos, outPos), rule.modulus);
    Value ctIn = arith::DivUIOp::create(b, loc, inPos, slots);
    Value block = ctIn;
    Value slot;
    if (rule.writeFollowsInput) {
      slot = emitFloorMod(b, loc, arith::SubIOp::create(b, loc, inPos, shift),
                          numSlots);
    } else {
      Value ctOut = arith::DivUIOp::create(b, loc, outPos, slots);
      block = arith::AddIOp::create(
          b, loc,
          arith::MulIOp::create(b, loc, ctOut, indexConstant(b, loc, numIn)),
          ctIn);
      slot = arith::RemUIOp::create(b, loc, outPos, slots);
    }
    Value tableIndex = arith::AddIOp::create(
        b, loc,
        arith::MulIOp::create(b, loc, block,
                              indexConstant(b, loc, rule.modulus)),
        shift);
    Value row = arith::IndexCastOp::create(
        b, loc, b.getIndexType(),
        tensor::ExtractOp::create(b, loc, table, ValueRange{tableIndex}));

    Value old =
        tensor::ExtractOp::create(b, loc, diagonals, ValueRange{row, slot});
    Value sum;
    if (linear.coefficient) {
      sum = makeAppropriatelyTypedAddOp(b, loc, old,
                                        linear.coefficient(b, loc, ivs))
                ->getResult(0);
    } else if (isa<FloatType>(elementType)) {
      sum = arith::AddFOp::create(
          b, loc, old,
          arith::ConstantOp::create(b, loc, b.getFloatAttr(elementType, 1.0)));
    } else {
      sum = arith::AddIOp::create(
          b, loc, old,
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(elementType, 1)));
    }
    return tensor::InsertOp::create(b, loc, sum, diagonals,
                                    ValueRange{row, slot});
  };

  scf::LoopNest nest = scf::buildLoopNest(
      b, loc, lbs, ubs, steps, {zeros},
      [&](OpBuilder& b, Location loc, ValueRange ivs,
          ValueRange args) -> scf::ValueVector {
        SmallVector<Value> inIdx;
        Value inBounds;
        for (size_t axis = 0; axis < spec.inMap.size(); ++axis) {
          Value idx =
              emitAffine(b, loc, spec.inMap[axis], spec.inOffset[axis], ivs);
          inIdx.push_back(idx);
          if (!canLeaveBounds(spec.inMap[axis], spec.inOffset[axis], spec.loops,
                              spec.inShape[axis])) {
            continue;
          }
          Value check = arith::AndIOp::create(
              b, loc,
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, idx,
                                    indexConstant(b, loc, 0)),
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, idx,
                                    indexConstant(b, loc, spec.inShape[axis])));
          inBounds = inBounds
                         ? Value(arith::AndIOp::create(b, loc, inBounds, check))
                         : check;
        }
        if (!inBounds) return {addTerm(b, loc, ivs, inIdx, args[0])};
        // Reads outside the input are zero and contribute nothing.
        auto ifOp = scf::IfOp::create(
            b, loc, inBounds,
            [&](OpBuilder& b, Location loc) {
              scf::YieldOp::create(b, loc,
                                   addTerm(b, loc, ivs, inIdx, args[0]));
            },
            [&](OpBuilder& b, Location loc) {
              scf::YieldOp::create(b, loc, args[0]);
            });
        return {ifOp.getResult(0)};
      });
  return nest.results.front();
}

// Emits the transform of `input`, a ciphertext-semantic tensor in layout
// `in`, into layout `out`.
FailureOr<Value> emitLinearTransform(OpBuilder& b, Location loc,
                                     const LinearOp& linear,
                                     const StridedLayout& in,
                                     const StridedLayout& out, Value input,
                                     int64_t maxDiagonals) {
  Type elementType = cast<RankedTensorType>(input.getType()).getElementType();
  int64_t numSlots = out.numSlots;
  auto resultType =
      RankedTensorType::get({out.numCiphertexts(), numSlots}, elementType);

  std::vector<DiagonalKey> keys = computeDiagonalKeys(linear.spec, in, out);
  if (keys.empty()) {
    return Value(arith::ConstantOp::create(b, loc, b.getZeroAttr(resultType)));
  }
  if (static_cast<int64_t>(keys.size()) > maxDiagonals) {
    return emitError(loc) << "linear transform needs " << keys.size()
                          << " diagonals, more than the limit of "
                          << maxDiagonals << "; input layout "
                          << StridedLayoutAttr::getFromStridedLayout(
                                 b.getContext(), in)
                          << ", output layout "
                          << StridedLayoutAttr::getFromStridedLayout(
                                 b.getContext(), out);
  }

  FailureOr<Value> diagonals =
      emitDiagonals(b, loc, linear, in, out, keys, elementType);
  if (failed(diagonals)) return failure();

  SmallVector<int64_t> outputCiphertexts, inputCiphertexts, shifts;
  for (const DiagonalKey& key : keys) {
    outputCiphertexts.push_back(key.ctOut);
    inputCiphertexts.push_back(key.ctIn);
    shifts.push_back(key.shift);
  }
  Value result = kernel::BlockLinearTransformOp::create(
      b, loc, resultType, input, *diagonals, outputCiphertexts,
      inputCiphertexts, shifts);

  // With a periodic output every term was written to one copy of its output
  // element. Summing the rotations by multiples of the period completes all
  // copies.
  ShiftRule rule = getShiftRule(in, out);
  if (rule.writeFollowsInput) {
    int64_t copies = numSlots / rule.modulus;
    if (copies * rule.modulus != numSlots || (copies & (copies - 1)) != 0) {
      return emitError(loc) << "output period " << rule.modulus
                            << " must divide the slot count " << numSlots
                            << " into a power of two";
    }
    for (int64_t step = 1; step < copies; step *= 2) {
      // The shift has the type the transform's own rotations use, so that
      // rotations can be merged.
      Value shift = arith::ConstantOp::create(
          b, loc, b.getI64IntegerAttr(step * rule.modulus));
      Value rotated =
          tensor_ext::RotateOp::create(b, loc, result.getType(), result, shift);
      result =
          makeAppropriatelyTypedAddOp(b, loc, result, rotated)->getResult(0);
    }
  }
  return result;
}

class StridedPattern : public ContextAwareConversionPattern {
 public:
  StridedPattern(const ContextAwareTypeConverter& converter,
                 MLIRContext* context, int64_t maxDiagonals)
      : ContextAwareConversionPattern(converter,
                                      RewritePattern::MatchAnyOpTypeTag(),
                                      /*benefit=*/2, context),
        maxDiagonals(maxDiagonals) {}

 protected:
  static std::optional<StridedLayout> asLayout(Attribute attr) {
    auto strided = dyn_cast_or_null<StridedLayoutAttr>(attr);
    if (!strided) return std::nullopt;
    return strided.getStridedLayout();
  }

  // The layout of an already converted value. The original operands of an
  // operation cannot be asked: they may belong to a region that the
  // conversion has already replaced.
  std::optional<StridedLayout> layoutOf(Value converted) const {
    FailureOr<Attribute> attr =
        getTypeConverter()->getContextualAttr(converted);
    if (failed(attr)) return std::nullopt;
    return asLayout(*attr);
  }

  std::optional<StridedLayout> resultLayout(Operation* op) const {
    return asLayout(op->getAttr(kLayoutAttrName));
  }

  // A result whose layout describes a tensor of another shape is a padded
  // view of that tensor: it is the same ciphertexts.
  bool isView(Operation* op, const StridedLayout& layout) const {
    return cast<RankedTensorType>(op->getResult(0).getType()).getShape() !=
           ArrayRef<int64_t>(layout.shape);
  }

  // Replaces `op` by `result`, which gets the layout `op` had, and marks the
  // operations created since `before` as converted.
  void finish(Operation* op, Value result, Operation* before,
              ContextAwareConversionPatternRewriter& rewriter) const {
    Attribute layout = op->getAttr(kLayoutAttrName);
    Operation* created =
        before ? before->getNextNode() : &op->getBlock()->front();
    for (; created != op; created = created->getNextNode()) {
      created->walk([](Operation* nested) { markStridedConverted(nested); });
    }
    if (Operation* def = result.getDefiningOp();
        def && def->getNumResults() == 1) {
      def->setAttr(kLayoutAttrName, layout);
    }
    rewriter.replaceOp(op, result);
  }

  int64_t maxDiagonals;
};

// Operations on one secret tensor: linalg operations, pads, reshapes and
// slices.
//
// If the result's layout is the operand's layout seen through the operation
// (a folded pad, or a reshape, slice or transpose that stays strided), no
// element moves and the result is the operand's ciphertexts. Otherwise the
// operation is one linear transform.
struct ConvertStridedTensorOp : public StridedPattern {
  using StridedPattern::StridedPattern;

  bool isRelabel(Operation* op, ArrayRef<Value> operands,
                 const StridedLayout& out) const {
    if (isView(op, out)) return true;
    std::optional<StridedLayout> in = layoutOf(operands[0]);
    if (!in) return false;
    std::optional<StridedLayout> relabeled = relabeledLayout(op, *in);
    return relabeled && *relabeled == out;
  }

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    if (!isa<tensor::PadOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp,
             tensor::ExtractSliceOp>(op) &&
        !isa<linalg::LinalgOp>(op)) {
      return failure();
    }
    std::optional<StridedLayout> out = resultLayout(op);
    if (!out) return failure();

    if (isRelabel(op, operands, *out)) {
      // A cast to the same type carries the new layout for the users; it is
      // folded away after the conversion.
      auto castOp = UnrealizedConversionCastOp::create(
          rewriter, op->getLoc(), operands[0].getType(), operands[0]);
      castOp->setAttr(kLayoutAttrName, op->getAttr(kLayoutAttrName));
      markStridedConverted(castOp);
      rewriter.replaceOp(op, castOp.getResult(0));
      return success();
    }

    unsigned numInputs = 1;
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      numInputs = linalgOp.getNumDpsInputs();
    }
    std::optional<LinearOp> linear;
    unsigned secretOperand = 0;
    // The operand's converted value is the source's ciphertexts and carries
    // the source's layout, also when a pad was folded into this operation.
    std::optional<StridedLayout> in;
    for (; secretOperand < numInputs; ++secretOperand) {
      in = layoutOf(operands[secretOperand]);
      if (!in) continue;
      linear = matchLinearOp(op, secretOperand);
      break;
    }
    if (!linear || !in) return failure();

    Operation* before = op->getPrevNode();
    rewriter.setInsertionPoint(op);
    FailureOr<Value> result =
        emitLinearTransform(rewriter, op->getLoc(), *linear, *in, *out,
                            operands[secretOperand], maxDiagonals);
    if (failed(result)) return failure();
    Value value = *result;
    if (linear->init) {
      unsigned initNumber =
          cast<linalg::LinalgOp>(op).getDpsInitOperand(0)->getOperandNumber();
      value = makeAppropriatelyTypedAddOp(rewriter, op->getLoc(), value,
                                          operands[initNumber])
                  ->getResult(0);
    }
    finish(op, value, before, rewriter);
    return success();
  }
};

// An insertion is the inserted tensor moved into its box, plus the tensor
// inserted into with the box cleared. Clearing is skipped when the box is
// known to be zero already, which is the case for the chains of insertions
// that a concatenation decomposes into.
struct ConvertStridedInsertSlice : public StridedPattern {
  using StridedPattern::StridedPattern;

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    auto insert = dyn_cast<tensor::InsertSliceOp>(op);
    if (!insert) return failure();
    std::optional<StridedLayout> out = resultLayout(op);
    std::optional<StridedLayout> in = layoutOf(operands[0]);
    std::optional<LinearOp> linear = matchLinearOp(op, 0);
    if (!out || !in || !linear) return failure();

    Operation* before = op->getPrevNode();
    rewriter.setInsertionPoint(op);
    FailureOr<Value> moved = emitLinearTransform(
        rewriter, op->getLoc(), *linear, *in, *out, operands[0], maxDiagonals);
    if (failed(moved)) return failure();
    Value result = *moved;

    if (!isZeroTensor(insert.getDest())) {
      Value kept = operands[1];
      if (!isZeroWhereInserted(insert)) {
        FailureOr<Value> cleared =
            emitLinearTransform(rewriter, op->getLoc(), *matchLinearOp(op, 1),
                                *out, *out, operands[1], maxDiagonals);
        if (failed(cleared)) return failure();
        kept = *cleared;
      }
      result = makeAppropriatelyTypedAddOp(rewriter, op->getLoc(), result, kept)
                   ->getResult(0);
    }
    finish(op, result, before, rewriter);
    return success();
  }
};

struct ConvertStridedConvertLayout : public StridedPattern {
  using StridedPattern::StridedPattern;

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ContextAwareConversionPatternRewriter& rewriter) const override {
    auto convert = dyn_cast<tensor_ext::ConvertLayoutOp>(op);
    if (!convert) return failure();
    auto from = dyn_cast<StridedLayoutAttr>(convert.getFromLayout());
    auto to = dyn_cast<StridedLayoutAttr>(convert.getToLayout());
    if (!from || !to) return failure();

    Operation* before = op->getPrevNode();
    rewriter.setInsertionPoint(op);
    FailureOr<Value> result = emitLinearTransform(
        rewriter, op->getLoc(), identityLinearOp(convert.getValue()),
        from.getStridedLayout(), to.getStridedLayout(), operands[0],
        maxDiagonals);
    if (failed(result)) return failure();
    op->setAttr(kLayoutAttrName, to);
    finish(op, *result, before, rewriter);
    return success();
  }
};

}  // namespace

void markStridedConverted(Operation* op) {
  op->setAttr("tensor_ext.layout_materialized",
              UnitAttr::get(op->getContext()));
}

bool usesStridedLayouts(Operation* module) {
  bool found = false;
  module->walk([&](Operation* op) {
    if (found) return WalkResult::interrupt();
    auto func = dyn_cast<FunctionOpInterface>(op);
    if (!func) return WalkResult::advance();
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      if (isa_and_nonnull<StridedLayoutAttr>(
              func.getArgAttr(i, kLayoutAttrName))) {
        found = true;
      }
    }
    return WalkResult::advance();
  });
  return found;
}

void populateStridedLayoutPatterns(RewritePatternSet& patterns,
                                   const ContextAwareTypeConverter& converter,
                                   MLIRContext* context, int64_t maxDiagonals) {
  patterns.add<ConvertStridedTensorOp, ConvertStridedInsertSlice,
               ConvertStridedConvertLayout>(converter, context, maxDiagonals);
}

}  // namespace heir
}  // namespace mlir
