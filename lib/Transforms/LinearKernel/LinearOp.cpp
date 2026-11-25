#include "lib/Transforms/LinearKernel/LinearOp.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "lib/Utils/Layout/LinearSpec.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallBitVector.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExprVisitor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"           // from @llvm-project

namespace mlir {
namespace heir {

namespace {

using IntMatrix = std::vector<std::vector<int64_t>>;

// The results of an affine map as an integer matrix over its dims and a
// constant column. Fails for maps with symbols, `mod` or `div`.
bool flattenMap(AffineMap map, IntMatrix& rows, std::vector<int64_t>& consts) {
  if (map.getNumSymbols() != 0) return false;
  unsigned numDims = map.getNumDims();
  for (AffineExpr expr : map.getResults()) {
    SimpleAffineExprFlattener flattener(numDims, 0);
    if (failed(flattener.walkPostOrder(expr))) return false;
    if (flattener.numLocals != 0) return false;
    const auto& flat = flattener.operandExprStack.back();
    rows.emplace_back(flat.begin(), flat.begin() + numDims);
    consts.push_back(flat.back());
  }
  return true;
}

bool isZeroConstant(Value value) {
  return matchPattern(value, m_Zero()) || matchPattern(value, m_AnyZeroFloat());
}

// How the indices of a value relate to the indices of the tensor it is a
// padded view of: `sourceIndex = matrix * valueIndex + offset`.
struct View {
  Value source;
  IntMatrix matrix;
  std::vector<int64_t> offset;
};

// For a reshape that only adds or removes unit dimensions: the fine axis that
// carries each coarse axis.
SmallVector<int64_t> carrierAxes(ArrayRef<ReassociationIndices> groups,
                                 ArrayRef<int64_t> fineShape) {
  SmallVector<int64_t> carrier;
  for (const ReassociationIndices& group : groups) {
    int64_t axis = group.front();
    for (int64_t fine : group) {
      if (fineShape[fine] != 1) axis = fine;
    }
    carrier.push_back(axis);
  }
  return carrier;
}

View resolveView(Value value) {
  size_t rank = cast<RankedTensorType>(value.getType()).getRank();
  View view;
  view.source = value;
  view.offset.assign(rank, 0);
  for (size_t axis = 0; axis < rank; ++axis) {
    view.matrix.emplace_back(rank, 0);
    view.matrix.back()[axis] = 1;
  }

  Operation* pad = findFoldablePad(value);
  if (!pad) return view;

  // `view.matrix` has one row per axis of `cursor` and one column per axis
  // of `value`.
  Value cursor = value;
  while (cursor != pad->getResult(0)) {
    Operation* def = cursor.getDefiningOp();
    IntMatrix next;
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(def)) {
      // An index of the operand is the row-major combination of the indices
      // of its group.
      ArrayRef<int64_t> fineShape = expand.getResultType().getShape();
      for (const ReassociationIndices& group :
           expand.getReassociationIndices()) {
        std::vector<int64_t> row(rank, 0);
        int64_t weight = 1;
        for (auto it = group.rbegin(); it != group.rend(); ++it) {
          for (size_t col = 0; col < rank; ++col) {
            row[col] += weight * view.matrix[*it][col];
          }
          weight *= fineShape[*it];
        }
        next.push_back(row);
      }
      cursor = expand.getSrc();
    } else {
      // Only unit dimensions are removed: each index of the operand is the
      // index of the one axis that carries its group, or zero.
      auto collapse = cast<tensor::CollapseShapeOp>(def);
      ArrayRef<int64_t> fineShape = collapse.getSrcType().getShape();
      SmallVector<ReassociationIndices, 4> groups =
          collapse.getReassociationIndices();
      SmallVector<int64_t> carrier = carrierAxes(groups, fineShape);
      next.assign(fineShape.size(), std::vector<int64_t>(rank, 0));
      for (size_t coarse = 0; coarse < carrier.size(); ++coarse) {
        next[carrier[coarse]] = view.matrix[coarse];
      }
      cursor = collapse.getSrc();
    }
    view.matrix = std::move(next);
  }

  auto padOp = cast<tensor::PadOp>(pad);
  view.offset.clear();
  for (int64_t low : padOp.getStaticLow()) view.offset.push_back(-low);
  view.source = padOp.getSource();
  return view;
}

// The body of a linalg operation, read as `out + f(x)` or `f(x)`.
class BodyAnalysis {
 public:
  BodyAnalysis(linalg::LinalgOp op, unsigned operandNumber)
      : op(op),
        body(op.getBlock()),
        secretArg(body->getArgument(operandNumber)),
        outArg(body->getArgument(op.getNumDpsInputs())) {}

  // Returns false if the body does not have the accepted form.
  bool analyze() {
    Value yielded = body->getTerminator()->getOperand(0);
    linear = yielded;
    if (Operation* def = definedInBody(yielded);
        def && isa<arith::AddFOp, arith::AddIOp>(def)) {
      Value lhs = def->getOperand(0);
      Value rhs = def->getOperand(1);
      if (lhs == outArg && !dependsOn(rhs, outArg)) {
        linear = rhs;
        accumulates = true;
      } else if (rhs == outArg && !dependsOn(lhs, outArg)) {
        linear = lhs;
        accumulates = true;
      }
    }
    return !dependsOn(linear, outArg) && isLinear(linear);
  }

  bool accumulates = false;

  // True if `f` is the identity, so every coefficient is one.
  bool isIdentity() const { return linear == secretArg; }

  // Emits the coefficient of `f` at the given loop indices.
  Value emit(OpBuilder& b, Location loc, ValueRange ivs,
             const std::vector<IntMatrix>& operandMaps,
             const std::vector<std::vector<int64_t>>& operandConsts) const {
    Emitter emitter{*this, b, loc, ivs, operandMaps, operandConsts, {}};
    Value result = emitter.linearPart(linear);
    if (!result) result = emitter.one(linear.getType());
    return result;
  }

 private:
  Operation* definedInBody(Value value) const {
    Operation* def = value.getDefiningOp();
    return def && def->getBlock() == body ? def : nullptr;
  }

  bool dependsOn(Value value, Value arg) const {
    if (value == arg) return true;
    Operation* def = definedInBody(value);
    if (!def) return false;
    return llvm::any_of(def->getOperands(),
                        [&](Value operand) { return dependsOn(operand, arg); });
  }

  bool isPublic(Value value) const {
    return !dependsOn(value, secretArg) && !dependsOn(value, outArg);
  }

  bool isLinear(Value value) const {
    if (value == secretArg) return true;
    Operation* def = definedInBody(value);
    if (!def) return false;
    if (isa<arith::MulFOp, arith::MulIOp>(def)) {
      Value lhs = def->getOperand(0);
      Value rhs = def->getOperand(1);
      return (isLinear(lhs) && isPublic(rhs)) ||
             (isLinear(rhs) && isPublic(lhs));
    }
    if (isa<arith::DivFOp>(def)) {
      return isLinear(def->getOperand(0)) && isPublic(def->getOperand(1));
    }
    if (isa<arith::NegFOp>(def)) return isLinear(def->getOperand(0));
    if (isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp>(def)) {
      return isLinear(def->getOperand(0)) && isLinear(def->getOperand(1));
    }
    return false;
  }

  struct Emitter {
    const BodyAnalysis& analysis;
    OpBuilder& b;
    Location loc;
    ValueRange ivs;
    const std::vector<IntMatrix>& operandMaps;
    const std::vector<std::vector<int64_t>>& operandConsts;
    llvm::DenseMap<Value, Value> publicValues;

    Value one(Type type) {
      if (isa<FloatType>(type)) {
        return arith::ConstantOp::create(b, loc, b.getFloatAttr(type, 1.0));
      }
      return arith::ConstantOp::create(b, loc, b.getIntegerAttr(type, 1));
    }

    Value index(const std::vector<int64_t>& row, int64_t constant) {
      Value result = arith::ConstantIndexOp::create(b, loc, constant);
      for (size_t loop = 0; loop < row.size(); ++loop) {
        if (row[loop] == 0) continue;
        Value term = ivs[loop];
        if (row[loop] != 1) {
          term = arith::MulIOp::create(
              b, loc, term, arith::ConstantIndexOp::create(b, loc, row[loop]));
        }
        result = arith::AddIOp::create(b, loc, result, term);
      }
      return result;
    }

    Value publicValue(Value value) {
      if (auto it = publicValues.find(value); it != publicValues.end()) {
        return it->second;
      }
      Value result;
      if (auto arg = dyn_cast<BlockArgument>(value);
          arg && arg.getOwner() == analysis.body) {
        unsigned operand = arg.getArgNumber();
        Value tensor = analysis.op->getOperand(operand);
        if (!isa<ShapedType>(tensor.getType())) {
          result = tensor;
        } else {
          SmallVector<Value> indices;
          for (size_t axis = 0; axis < operandMaps[operand].size(); ++axis) {
            indices.push_back(index(operandMaps[operand][axis],
                                    operandConsts[operand][axis]));
          }
          result = tensor::ExtractOp::create(b, loc, tensor, indices);
        }
      } else if (Operation* def = analysis.definedInBody(value)) {
        if (auto indexOp = dyn_cast<linalg::IndexOp>(def)) {
          result = ivs[indexOp.getDim()];
        } else {
          Operation* clone = b.clone(*def);
          for (OpOperand& operand : clone->getOpOperands()) {
            operand.set(publicValue(operand.get()));
          }
          result = clone->getResult(0);
        }
      } else {
        // Defined outside the linalg operation.
        result = value;
      }
      publicValues[value] = result;
      return result;
    }

    // The coefficient of the secret element in `value`. Null stands for one.
    Value linearPart(Value value) {
      if (value == analysis.secretArg) return Value();
      Operation* def = analysis.definedInBody(value);
      Value lhs = def->getOperand(0);
      if (isa<arith::NegFOp>(def)) {
        Value inner = linearPart(lhs);
        return arith::NegFOp::create(b, loc,
                                     inner ? inner : one(value.getType()));
      }
      Value rhs = def->getOperand(1);
      if (isa<arith::MulFOp, arith::MulIOp>(def)) {
        bool lhsLinear = !analysis.isPublic(lhs);
        Value inner = linearPart(lhsLinear ? lhs : rhs);
        Value factor = publicValue(lhsLinear ? rhs : lhs);
        if (!inner) return factor;
        return isa<arith::MulFOp>(def)
                   ? Value(arith::MulFOp::create(b, loc, inner, factor))
                   : Value(arith::MulIOp::create(b, loc, inner, factor));
      }
      Value left = linearPart(lhs);
      if (!left) left = one(value.getType());
      if (isa<arith::DivFOp>(def)) {
        return arith::DivFOp::create(b, loc, left, publicValue(rhs));
      }
      Value right = linearPart(rhs);
      if (!right) right = one(value.getType());
      if (isa<arith::AddFOp>(def)) {
        return arith::AddFOp::create(b, loc, left, right);
      }
      if (isa<arith::AddIOp>(def)) {
        return arith::AddIOp::create(b, loc, left, right);
      }
      if (isa<arith::SubFOp>(def)) {
        return arith::SubFOp::create(b, loc, left, right);
      }
      return arith::SubIOp::create(b, loc, left, right);
    }
  };

  linalg::LinalgOp op;
  Block* body;
  Value secretArg;
  Value outArg;
  Value linear;
};

std::optional<LinearOp> matchPad(tensor::PadOp pad) {
  if (!isStaticZeroPad(pad)) return std::nullopt;
  LinearOp result;
  result.source = pad.getSource();
  ArrayRef<int64_t> shape = pad.getSourceType().getShape();
  size_t rank = shape.size();
  result.spec.loops.assign(shape.begin(), shape.end());
  result.spec.inShape.assign(shape.begin(), shape.end());
  ArrayRef<int64_t> outShape = pad.getResultType().getShape();
  result.spec.outShape.assign(outShape.begin(), outShape.end());
  result.spec.inOffset.assign(rank, 0);
  result.spec.outOffset.assign(pad.getStaticLow().begin(),
                               pad.getStaticLow().end());
  for (size_t axis = 0; axis < rank; ++axis) {
    std::vector<int64_t> row(rank, 0);
    row[axis] = 1;
    result.spec.inMap.push_back(row);
    result.spec.outMap.push_back(row);
  }
  return result;
}

// One row per axis of the coarse side of a reshape: the row-major weights of
// the fine axes in its group.
IntMatrix reshapeRows(ArrayRef<ReassociationIndices> groups,
                      ArrayRef<int64_t> fineShape) {
  IntMatrix rows;
  for (const ReassociationIndices& group : groups) {
    std::vector<int64_t> row(fineShape.size(), 0);
    int64_t weight = 1;
    for (auto it = group.rbegin(); it != group.rend(); ++it) {
      row[*it] = weight;
      weight *= fineShape[*it];
    }
    rows.push_back(row);
  }
  return rows;
}

IntMatrix identityRows(size_t rank) {
  IntMatrix rows;
  for (size_t axis = 0; axis < rank; ++axis) {
    std::vector<int64_t> row(rank, 0);
    row[axis] = 1;
    rows.push_back(row);
  }
  return rows;
}

void setShape(std::vector<int64_t>& target, ArrayRef<int64_t> shape) {
  target.assign(shape.begin(), shape.end());
}

// A reshape as a move of every element: the loops run over the fine shape.
std::optional<LinearOp> matchReshape(Operation* op) {
  LinearOp result;
  result.source = op->getOperand(0);
  auto sourceType = dyn_cast<RankedTensorType>(result.source.getType());
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
      !resultType.hasStaticShape()) {
    return std::nullopt;
  }
  setShape(result.spec.inShape, sourceType.getShape());
  setShape(result.spec.outShape, resultType.getShape());
  if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op)) {
    setShape(result.spec.loops, sourceType.getShape());
    result.spec.inMap = identityRows(sourceType.getRank());
    result.spec.outMap =
        reshapeRows(collapse.getReassociationIndices(), sourceType.getShape());
  } else {
    auto expand = cast<tensor::ExpandShapeOp>(op);
    setShape(result.spec.loops, resultType.getShape());
    result.spec.inMap =
        reshapeRows(expand.getReassociationIndices(), resultType.getShape());
    result.spec.outMap = identityRows(resultType.getRank());
  }
  result.spec.inOffset.assign(sourceType.getRank(), 0);
  result.spec.outOffset.assign(resultType.getRank(), 0);
  return result;
}

bool isStaticSlice(tensor::ExtractSliceOp slice) {
  auto hasDynamic = [](ArrayRef<int64_t> values) {
    return llvm::any_of(values, ShapedType::isDynamic);
  };
  return !hasDynamic(slice.getStaticOffsets()) &&
         !hasDynamic(slice.getStaticSizes()) &&
         !hasDynamic(slice.getStaticStrides()) &&
         slice.getSourceType().hasStaticShape() &&
         slice.getResultType().getRank() > 0;
}

// For a slice that drops unit dimensions: the source axes that each result
// axis stands for, as a reassociation of the full-rank slice.
std::vector<std::vector<int64_t>> droppedDimGroups(
    tensor::ExtractSliceOp slice) {
  llvm::SmallBitVector dropped = slice.getDroppedDims();
  std::vector<std::vector<int64_t>> groups;
  std::vector<int64_t> pending;
  for (int64_t axis = 0; axis < slice.getSourceType().getRank(); ++axis) {
    pending.push_back(axis);
    if (dropped.test(axis)) continue;
    groups.push_back(pending);
    pending.clear();
  }
  // Dropped axes behind the last kept one join its group.
  if (!groups.empty()) {
    groups.back().insert(groups.back().end(), pending.begin(), pending.end());
  }
  return groups;
}

std::optional<LinearOp> matchSlice(tensor::ExtractSliceOp slice) {
  if (!isStaticSlice(slice)) return std::nullopt;
  LinearOp result;
  result.source = slice.getSource();
  size_t rank = slice.getSourceType().getRank();
  setShape(result.spec.inShape, slice.getSourceType().getShape());
  setShape(result.spec.outShape, slice.getResultType().getShape());
  setShape(result.spec.loops, slice.getStaticSizes());
  setShape(result.spec.inOffset, slice.getStaticOffsets());
  result.spec.inMap = identityRows(rank);
  for (size_t axis = 0; axis < rank; ++axis) {
    result.spec.inMap[axis][axis] = slice.getStaticStrides()[axis];
  }
  llvm::SmallBitVector dropped = slice.getDroppedDims();
  IntMatrix rows = identityRows(rank);
  for (size_t axis = 0; axis < rank; ++axis) {
    if (!dropped.test(axis)) result.spec.outMap.push_back(rows[axis]);
  }
  result.spec.outOffset.assign(result.spec.outMap.size(), 0);
  return result;
}

bool isStaticInsert(tensor::InsertSliceOp insert) {
  auto hasDynamic = [](ArrayRef<int64_t> values) {
    return llvm::any_of(values, ShapedType::isDynamic);
  };
  return !hasDynamic(insert.getStaticOffsets()) &&
         !hasDynamic(insert.getStaticSizes()) &&
         !hasDynamic(insert.getStaticStrides()) &&
         insert.getSourceType().hasStaticShape() &&
         insert.getDestType().hasStaticShape();
}

// The inserted tensor: a copy into a box of the result.
std::optional<LinearOp> matchInsertedSource(tensor::InsertSliceOp insert) {
  if (!isStaticInsert(insert)) return std::nullopt;
  LinearOp result;
  result.source = insert.getSource();
  size_t rank = insert.getDestType().getRank();
  setShape(result.spec.inShape, insert.getSourceType().getShape());
  setShape(result.spec.outShape, insert.getDestType().getShape());
  setShape(result.spec.loops, insert.getStaticSizes());
  setShape(result.spec.outOffset, insert.getStaticOffsets());
  result.spec.outMap = identityRows(rank);
  for (size_t axis = 0; axis < rank; ++axis) {
    result.spec.outMap[axis][axis] = insert.getStaticStrides()[axis];
  }
  // The source may lack unit axes of the box.
  llvm::SmallBitVector dropped = insert.getDroppedDims();
  IntMatrix rows = identityRows(rank);
  for (size_t axis = 0; axis < rank; ++axis) {
    if (!dropped.test(axis)) result.spec.inMap.push_back(rows[axis]);
  }
  result.spec.inOffset.assign(result.spec.inMap.size(), 0);
  return result;
}

// The tensor inserted into: a copy of everything outside the box.
std::optional<LinearOp> matchInsertedDest(tensor::InsertSliceOp insert) {
  if (!isStaticInsert(insert)) return std::nullopt;
  LinearOp result = identityLinearOp(insert.getDest());
  std::vector<int64_t> offsets(insert.getStaticOffsets());
  std::vector<int64_t> sizes(insert.getStaticSizes());
  std::vector<int64_t> strides(insert.getStaticStrides());
  Type elementType = insert.getDestType().getElementType();
  result.coefficient = [=](OpBuilder& b, Location loc, ValueRange ivs) {
    Value inBox;
    for (size_t axis = 0; axis < offsets.size(); ++axis) {
      Value rel = arith::SubIOp::create(
          b, loc, ivs[axis],
          arith::ConstantIndexOp::create(b, loc, offsets[axis]));
      Value stride = arith::ConstantIndexOp::create(b, loc, strides[axis]);
      Value zero = arith::ConstantIndexOp::create(b, loc, 0);
      Value end = arith::ConstantIndexOp::create(
          b, loc, (sizes[axis] - 1) * strides[axis]);
      Value hit = arith::AndIOp::create(
          b, loc,
          arith::AndIOp::create(
              b, loc,
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, rel,
                                    zero),
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sle, rel,
                                    end)),
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                arith::RemSIOp::create(b, loc, rel, stride),
                                zero));
      inBox = inBox ? Value(arith::AndIOp::create(b, loc, inBox, hit)) : hit;
    }
    Value keep = arith::ConstantOp::create(b, loc, b.getOneAttr(elementType));
    Value drop = arith::ConstantOp::create(b, loc, b.getZeroAttr(elementType));
    return Value(arith::SelectOp::create(b, loc, inBox, drop, keep));
  };
  return result;
}

// If `linear` only renames axes, the permutation: axis `i` of the result is
// axis `perm[i]` of the source.
std::optional<std::vector<int64_t>> asPermutation(const LinearOp& linear) {
  const LinearSpec& spec = linear.spec;
  size_t rank = spec.loops.size();
  if (linear.coefficient || linear.init || spec.inMap.size() != rank ||
      spec.outMap.size() != rank) {
    return std::nullopt;
  }
  // The single loop a row reads with coefficient one, or -1.
  auto loopOf = [&](const std::vector<int64_t>& row) -> int64_t {
    int64_t loop = -1;
    for (size_t l = 0; l < rank; ++l) {
      if (row[l] == 0) continue;
      if (row[l] != 1 || loop >= 0) return -1;
      loop = l;
    }
    return loop;
  };
  std::vector<int64_t> inAxisOfLoop(rank, -1);
  for (size_t axis = 0; axis < rank; ++axis) {
    int64_t loop = loopOf(spec.inMap[axis]);
    if (loop < 0 || inAxisOfLoop[loop] >= 0 || spec.inOffset[axis] != 0) {
      return std::nullopt;
    }
    inAxisOfLoop[loop] = axis;
  }
  std::vector<int64_t> perm;
  std::vector<bool> seen(rank, false);
  for (size_t axis = 0; axis < rank; ++axis) {
    int64_t loop = loopOf(spec.outMap[axis]);
    if (loop < 0 || seen[loop] || spec.outOffset[axis] != 0) {
      return std::nullopt;
    }
    seen[loop] = true;
    perm.push_back(inAxisOfLoop[loop]);
  }
  return perm;
}

}  // namespace

bool isZeroWhereInserted(tensor::InsertSliceOp insert) {
  if (!isStaticInsert(insert)) return false;
  // The first and last index of the box along each axis.
  auto bounds = [](tensor::InsertSliceOp op, size_t axis) {
    int64_t first = op.getStaticOffsets()[axis];
    return std::make_pair(first, first + (op.getStaticSizes()[axis] - 1) *
                                             op.getStaticStrides()[axis]);
  };
  Value dest = insert.getDest();
  while (!isZeroTensor(dest)) {
    auto earlier = dest.getDefiningOp<tensor::InsertSliceOp>();
    if (!earlier || !isStaticInsert(earlier)) return false;
    bool disjoint = false;
    for (size_t axis = 0; axis < insert.getStaticOffsets().size(); ++axis) {
      auto [lo, hi] = bounds(insert, axis);
      auto [earlierLo, earlierHi] = bounds(earlier, axis);
      if (hi < earlierLo || earlierHi < lo) disjoint = true;
    }
    if (!disjoint) return false;
    dest = earlier.getDest();
  }
  return true;
}

std::optional<StridedLayout> relabeledLayout(Operation* op,
                                             const StridedLayout& in) {
  auto toVectors = [](ArrayRef<ReassociationIndices> groups) {
    std::vector<std::vector<int64_t>> result;
    for (const ReassociationIndices& group : groups) {
      result.emplace_back(group.begin(), group.end());
    }
    return result;
  };
  if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op)) {
    return in.collapse(toVectors(collapse.getReassociationIndices()));
  }
  if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
    if (!expand.getResultType().hasStaticShape()) return std::nullopt;
    return in.expand(toVectors(expand.getReassociationIndices()),
                     expand.getResultType().getShape());
  }
  if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op)) {
    if (!isStaticSlice(slice)) return std::nullopt;
    std::optional<StridedLayout> sliced =
        in.slice(slice.getStaticOffsets(), slice.getStaticSizes(),
                 slice.getStaticStrides());
    if (!sliced || slice.getDroppedDims().none()) return sliced;
    return sliced->collapse(droppedDimGroups(slice));
  }
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || linalgOp.getNumDpsInputs() != 1) return std::nullopt;
  std::optional<LinearOp> linear = matchLinearOp(op, 0);
  if (!linear || linear->source != op->getOperand(0)) return std::nullopt;
  std::optional<std::vector<int64_t>> perm = asPermutation(*linear);
  if (!perm) return std::nullopt;
  return in.transpose(*perm);
}

bool isZeroTensor(Value value) {
  if (isZeroConstant(value)) return true;
  if (value.getDefiningOp<tensor::EmptyOp>()) return true;
  if (auto fill = value.getDefiningOp<linalg::FillOp>()) {
    return isZeroConstant(fill.getInputs()[0]);
  }
  return false;
}

bool isStaticZeroPad(Operation* op) {
  auto pad = dyn_cast_or_null<tensor::PadOp>(op);
  if (!pad || !pad.getLow().empty() || !pad.getHigh().empty()) return false;
  if (!pad.getSourceType().hasStaticShape()) return false;
  if (llvm::any_of(pad.getStaticLow(), [](int64_t v) { return v < 0; }) ||
      llvm::any_of(pad.getStaticHigh(), [](int64_t v) { return v < 0; })) {
    return false;
  }
  Value padValue = pad.getConstantPaddingValue();
  return padValue && isZeroConstant(padValue);
}

bool isViewReshape(Operation* op) {
  if (auto expand = dyn_cast_or_null<tensor::ExpandShapeOp>(op)) {
    return expand.getResultType().hasStaticShape();
  }
  auto collapse = dyn_cast_or_null<tensor::CollapseShapeOp>(op);
  if (!collapse) return false;
  ArrayRef<int64_t> fineShape = collapse.getSrcType().getShape();
  for (const ReassociationIndices& group : collapse.getReassociationIndices()) {
    int64_t numNonUnit = llvm::count_if(
        group, [&](int64_t axis) { return fineShape[axis] != 1; });
    if (numNonUnit > 1) return false;
  }
  return true;
}

Operation* findFoldablePad(Value value) {
  Value cursor = value;
  while (Operation* def = cursor.getDefiningOp()) {
    if (isStaticZeroPad(def)) return def;
    if (!isViewReshape(def)) return nullptr;
    cursor = def->getOperand(0);
  }
  return nullptr;
}

LinearOp identityLinearOp(Value source) {
  LinearOp result;
  result.source = source;
  ArrayRef<int64_t> shape = cast<RankedTensorType>(source.getType()).getShape();
  size_t rank = shape.size();
  result.spec.loops.assign(shape.begin(), shape.end());
  result.spec.inShape = result.spec.loops;
  result.spec.outShape = result.spec.loops;
  result.spec.inOffset.assign(rank, 0);
  result.spec.outOffset.assign(rank, 0);
  for (size_t axis = 0; axis < rank; ++axis) {
    std::vector<int64_t> row(rank, 0);
    row[axis] = 1;
    result.spec.inMap.push_back(row);
    result.spec.outMap.push_back(row);
  }
  return result;
}

std::optional<LinearOp> matchLinearOp(Operation* op, unsigned operandNumber) {
  if (auto pad = dyn_cast<tensor::PadOp>(op)) {
    return operandNumber == 0 ? matchPad(pad) : std::nullopt;
  }
  if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op)) {
    return operandNumber == 0 ? matchReshape(op) : std::nullopt;
  }
  if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op)) {
    return operandNumber == 0 ? matchSlice(slice) : std::nullopt;
  }
  if (auto insert = dyn_cast<tensor::InsertSliceOp>(op)) {
    if (operandNumber == 0) return matchInsertedSource(insert);
    return operandNumber == 1 ? matchInsertedDest(insert) : std::nullopt;
  }
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || linalgOp.getNumDpsInits() != 1 ||
      operandNumber >= static_cast<unsigned>(linalgOp.getNumDpsInputs()) ||
      linalgOp.hasDynamicShape() || !linalgOp.hasPureTensorSemantics()) {
    return std::nullopt;
  }
  if (!isa<RankedTensorType>(op->getOperand(operandNumber).getType())) {
    return std::nullopt;
  }

  // Indexing maps of every operand, in operand order.
  std::vector<IntMatrix> maps(op->getNumOperands());
  std::vector<std::vector<int64_t>> consts(op->getNumOperands());
  for (OpOperand& operand : op->getOpOperands()) {
    if (!flattenMap(linalgOp.getMatchingIndexingMap(&operand),
                    maps[operand.getOperandNumber()],
                    consts[operand.getOperandNumber()])) {
      return std::nullopt;
    }
  }

  auto analysis = std::make_shared<BodyAnalysis>(linalgOp, operandNumber);
  if (!analysis->analyze()) return std::nullopt;

  LinearOp result;
  SmallVector<int64_t, 4> loops = linalgOp.getStaticLoopRanges();
  result.spec.loops.assign(loops.begin(), loops.end());
  size_t numLoops = loops.size();

  unsigned initNumber = linalgOp.getDpsInitOperand(0)->getOperandNumber();
  Value initValue = op->getOperand(initNumber);
  ArrayRef<int64_t> outShape =
      cast<RankedTensorType>(initValue.getType()).getShape();
  result.spec.outShape.assign(outShape.begin(), outShape.end());
  result.spec.outMap = maps[initNumber];
  result.spec.outOffset = consts[initNumber];

  if (analysis->accumulates) {
    if (!isZeroTensor(initValue)) result.init = initValue;
  } else {
    // Without accumulation every loop must select its own output element.
    for (size_t loop = 0; loop < numLoops; ++loop) {
      bool used = llvm::any_of(
          result.spec.outMap,
          [&](const std::vector<int64_t>& row) { return row[loop] != 0; });
      if (!used && loops[loop] != 1) return std::nullopt;
    }
  }

  // The operand's indices, mapped onto the indices of the tensor it views.
  View view = resolveView(op->getOperand(operandNumber));
  result.source = view.source;
  ArrayRef<int64_t> sourceShape =
      cast<RankedTensorType>(view.source.getType()).getShape();
  result.spec.inShape.assign(sourceShape.begin(), sourceShape.end());
  result.spec.inMap.assign(sourceShape.size(),
                           std::vector<int64_t>(numLoops, 0));
  result.spec.inOffset = view.offset;
  for (size_t sourceAxis = 0; sourceAxis < sourceShape.size(); ++sourceAxis) {
    for (size_t axis = 0; axis < view.matrix[sourceAxis].size(); ++axis) {
      int64_t weight = view.matrix[sourceAxis][axis];
      if (weight == 0) continue;
      for (size_t loop = 0; loop < numLoops; ++loop) {
        result.spec.inMap[sourceAxis][loop] +=
            weight * maps[operandNumber][axis][loop];
      }
      result.spec.inOffset[sourceAxis] += weight * consts[operandNumber][axis];
    }
  }

  if (!analysis->isIdentity()) {
    result.coefficient = [analysis, maps, consts](OpBuilder& b, Location loc,
                                                  ValueRange ivs) {
      return analysis->emit(b, loc, ivs, maps, consts);
    };
  }
  return result;
}

}  // namespace heir
}  // namespace mlir
