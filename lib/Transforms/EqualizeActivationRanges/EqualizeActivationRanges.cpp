#include "lib/Transforms/EqualizeActivationRanges/EqualizeActivationRanges.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Debug/IR/DebugOps.h"
#include "llvm/include/llvm/ADT/APFloat.h"                 // from @llvm-project
#include "llvm/include/llvm/ADT/DenseMap.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SetVector.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AsmState.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectResourceBlobManager.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"      // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"          // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Transforms/RegionUtils.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_EQUALIZEACTIVATIONRANGES
#include "lib/Transforms/EqualizeActivationRanges/EqualizeActivationRanges.h.inc"

namespace {

constexpr StringLiteral kDomainLower = "domain_lower";
constexpr StringLiteral kDomainUpper = "domain_upper";
constexpr StringLiteral kDebugScale = "debug.scale";

bool isFloatValue(Value value) {
  return isa<FloatType>(getElementTypeOrSelf(value.getType()));
}

APFloat scaleFloat(const APFloat& value, double factor) {
  bool losesInfo;
  APFloat wide = value;
  wide.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
  APFloat result(wide.convertToDouble() * factor);
  result.convert(value.getSemantics(), APFloat::rmNearestTiesToEven,
                 &losesInfo);
  return result;
}

template <typename T>
bool hasResourceData(DenseResourceElementsAttr resource) {
  return resource.getData().size() ==
         resource.getType().getNumElements() * sizeof(T);
}

template <typename T>
Attribute scaleResource(DenseResourceElementsAttr resource, double factor) {
  ArrayRef<char> raw = resource.getData();
  ArrayRef<T> values(reinterpret_cast<const T*>(raw.data()),
                     resource.getType().getNumElements());
  SmallVector<T> result;
  result.reserve(values.size());
  for (T value : values) result.push_back(static_cast<T>(value * factor));
  return DenseResourceElementsAttr::get(
      resource.getType(), "range_equalized",
      HeapAsmResourceBlob::allocateAndCopyInferAlign<T>(result));
}

bool isScalableAttr(Attribute attr) {
  if (isa<FloatAttr, DenseFPElementsAttr>(attr)) return true;
  auto resource = dyn_cast<DenseResourceElementsAttr>(attr);
  if (!resource) return false;
  Type elementType = resource.getType().getElementType();
  if (elementType.isF32()) return hasResourceData<float>(resource);
  if (elementType.isF64()) return hasResourceData<double>(resource);
  return false;
}

TypedAttr scaleAttr(Attribute attr, double factor) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return FloatAttr::get(floatAttr.getType(),
                          scaleFloat(floatAttr.getValue(), factor));
  if (auto dense = dyn_cast<DenseFPElementsAttr>(attr))
    return cast<TypedAttr>(
        dense.mapValues(dense.getElementType(), [&](const APFloat& value) {
          return scaleFloat(value, factor).bitcastToAPInt();
        }));
  auto resource = cast<DenseResourceElementsAttr>(attr);
  if (resource.getType().getElementType().isF32())
    return cast<TypedAttr>(scaleResource<float>(resource, factor));
  return cast<TypedAttr>(scaleResource<double>(resource, factor));
}

// Whether `value`, a constant, can be rebuilt multiplied by a factor. Covers
// the constants torch-mlir emits for weights and biases: dense or resource
// constants, possibly transposed, broadcast, reshaped or filled into a tensor.
bool isScalable(Value value) {
  Operation* def = value.getDefiningOp();
  if (!def) return false;
  return llvm::TypeSwitch<Operation*, bool>(def)
      .Case<arith::ConstantOp>(
          [](arith::ConstantOp op) { return isScalableAttr(op.getValue()); })
      .Case<tensor::EmptyOp>([](auto) { return true; })
      .Case<linalg::FillOp, linalg::BroadcastOp, linalg::TransposeOp,
            tensor::ExpandShapeOp>(
          [](Operation* op) { return isScalable(op->getOperand(0)); })
      .Default([](Operation*) { return false; });
}

bool isZero(Value value) {
  if (matchPattern(value, m_AnyZeroFloat())) return true;
  auto fill = value.getDefiningOp<linalg::FillOp>();
  return fill && matchPattern(fill.getInputs()[0], m_AnyZeroFloat());
}

// Builds `value * factor` from new constants, for a `value` that isScalable.
Value scaleConstant(OpBuilder& builder, Value value, double factor) {
  Operation* def = value.getDefiningOp();
  if (isa<tensor::EmptyOp>(def) || isZero(value)) return value;
  if (auto constant = dyn_cast<arith::ConstantOp>(def)) {
    builder.setInsertionPointAfter(def);
    return arith::ConstantOp::create(builder, def->getLoc(),
                                     scaleAttr(constant.getValue(), factor));
  }
  // The scaled operand is the fill value or the source of a shape op.
  Value scaledInput = scaleConstant(builder, def->getOperand(0), factor);
  builder.setInsertionPointAfter(def);
  Operation* clone = builder.clone(*def);
  clone->setOperand(0, scaledInput);
  return clone->getResult(cast<OpResult>(value).getResultNumber());
}

// Whether the body of `op` is select(x > 0, x, 0), the ReLU that torch-mlir
// emits and whose domain the composite-sign approximation reads.
bool isRelu(linalg::LinalgOp op) {
  if (op.getNumDpsInputs() != 1 || op->getNumResults() != 1) return false;
  Block* body = op.getBlock();
  Value input = body->getArgument(0);
  auto select =
      body->getTerminator()->getOperand(0).getDefiningOp<arith::SelectOp>();
  if (!select) return false;
  auto cmp = select.getCondition().getDefiningOp<arith::CmpFOp>();
  if (!cmp) return false;
  arith::CmpFPredicate predicate = cmp.getPredicate();
  if (predicate != arith::CmpFPredicate::UGT &&
      predicate != arith::CmpFPredicate::UGE)
    return false;
  return cmp.getLhs() == input && select.getTrueValue() == input &&
         matchPattern(cmp.getRhs(), m_AnyZeroFloat()) &&
         matchPattern(select.getFalseValue(), m_AnyZeroFloat());
}

// How a value in the body of a linalg op scales when the op's secret operands
// are divided by a common factor.
enum class Kind {
  Zero,    // A zero constant.
  Fixed,   // A value that does not depend on the scaled operands.
  Scaled,  // A value that is divided by the factor.
  Bool,    // A comparison whose outcome the factor does not change.
};

// Checks that the body of `op` commutes with dividing its secret operands by a
// common positive factor. Public operands that are added to a scaled value, or
// compared with one by max or min, have to be divided by the factor too, and
// are returned in `divided`. Other public operands, such as a multiplier, stay
// as they are.
LogicalResult analyzeBody(linalg::LinalgOp op,
                          llvm::function_ref<bool(Value)> isScaled,
                          SmallVectorImpl<OpOperand*>& divided) {
  DenseMap<Value, Kind> kinds;
  DenseMap<Value, OpOperand*> publicArgs;
  for (OpOperand& operand : op->getOpOperands()) {
    if (!op.isDpsInput(&operand) && !op.isDpsInit(&operand)) continue;
    BlockArgument arg = op.getMatchingBlockArgument(&operand);
    if (isScaled(operand.get())) {
      kinds[arg] = Kind::Scaled;
    } else {
      kinds[arg] = Kind::Fixed;
      publicArgs[arg] = &operand;
    }
  }

  auto kindOf = [&](Value value) -> std::optional<Kind> {
    auto it = kinds.find(value);
    if (it != kinds.end()) return it->second;
    if (matchPattern(value, m_AnyZeroFloat())) return Kind::Zero;
    if (matchPattern(value, m_Constant())) return Kind::Fixed;
    return std::nullopt;
  };
  auto isScaledOrZero = [](std::optional<Kind> kind) {
    return kind == Kind::Scaled || kind == Kind::Zero;
  };

  // Public operands combined additively with a scaled value, and public
  // operands used in any other way.
  DenseSet<Value> addedArgs;
  DenseSet<Value> fixedArgs;

  // The kind of `lhs <op> rhs` for an op that commutes with a common factor
  // of both operands: add, sub, max and min.
  auto additive = [&](Value lhs, Value rhs) -> std::optional<Kind> {
    std::optional<Kind> lhsKind = kindOf(lhs);
    std::optional<Kind> rhsKind = kindOf(rhs);
    if (!lhsKind || !rhsKind || lhsKind == Kind::Bool || rhsKind == Kind::Bool)
      return std::nullopt;
    if (lhsKind != Kind::Scaled && rhsKind != Kind::Scaled) {
      for (Value value : {lhs, rhs})
        if (publicArgs.contains(value)) fixedArgs.insert(value);
      if (lhsKind == Kind::Zero && rhsKind == Kind::Zero) return Kind::Zero;
      return Kind::Fixed;
    }
    std::pair<Value, Kind> sides[] = {{lhs, *lhsKind}, {rhs, *rhsKind}};
    for (auto [value, kind] : sides) {
      if (kind != Kind::Fixed) continue;
      if (!publicArgs.contains(value)) return std::nullopt;
      addedArgs.insert(value);
    }
    return Kind::Scaled;
  };

  Block* body = op.getBlock();
  for (Operation& inner : body->without_terminator()) {
    if (inner.getNumResults() != 1) return failure();
    std::optional<Kind> kind;
    if (isa<arith::AddFOp, arith::SubFOp, arith::MaximumFOp, arith::MinimumFOp>(
            inner)) {
      kind = additive(inner.getOperand(0), inner.getOperand(1));
    } else {
      for (Value value : inner.getOperands())
        if (publicArgs.contains(value)) fixedArgs.insert(value);
      kind = llvm::TypeSwitch<Operation*, std::optional<Kind>>(&inner)
                 .Case<arith::MulFOp>([&](auto) -> std::optional<Kind> {
                   std::optional<Kind> lhs = kindOf(inner.getOperand(0));
                   std::optional<Kind> rhs = kindOf(inner.getOperand(1));
                   if (!lhs || !rhs || lhs == Kind::Bool || rhs == Kind::Bool ||
                       (lhs == Kind::Scaled && rhs == Kind::Scaled))
                     return std::nullopt;
                   if (lhs == Kind::Zero || rhs == Kind::Zero)
                     return Kind::Zero;
                   if (lhs == Kind::Scaled || rhs == Kind::Scaled)
                     return Kind::Scaled;
                   return Kind::Fixed;
                 })
                 .Case<arith::DivFOp>([&](auto) -> std::optional<Kind> {
                   std::optional<Kind> lhs = kindOf(inner.getOperand(0));
                   std::optional<Kind> rhs = kindOf(inner.getOperand(1));
                   if (rhs != Kind::Fixed) return std::nullopt;
                   if (lhs == Kind::Bool) return std::nullopt;
                   return lhs;
                 })
                 // Comparing values that share a positive factor, or one such
                 // value with zero, does not depend on the factor.
                 .Case<arith::CmpFOp>([&](auto) -> std::optional<Kind> {
                   if (isScaledOrZero(kindOf(inner.getOperand(0))) &&
                       isScaledOrZero(kindOf(inner.getOperand(1))))
                     return Kind::Bool;
                   return std::nullopt;
                 })
                 .Case<arith::SelectOp>([&](auto) -> std::optional<Kind> {
                   std::optional<Kind> trueKind = kindOf(inner.getOperand(1));
                   std::optional<Kind> falseKind = kindOf(inner.getOperand(2));
                   if (kindOf(inner.getOperand(0)) != Kind::Bool ||
                       !isScaledOrZero(trueKind) || !isScaledOrZero(falseKind))
                     return std::nullopt;
                   if (trueKind == Kind::Zero && falseKind == Kind::Zero)
                     return Kind::Zero;
                   return Kind::Scaled;
                 })
                 .Default([](Operation*) { return std::nullopt; });
    }
    if (!kind) return failure();
    kinds[inner.getResult(0)] = *kind;
  }

  for (Value yielded : body->getTerminator()->getOperands())
    if (!isScaledOrZero(kindOf(yielded))) return failure();
  for (Value arg : addedArgs) {
    if (fixedArgs.contains(arg)) return failure();
    divided.push_back(publicArgs[arg]);
  }
  return success();
}

class Analysis {
 public:
  Analysis(func::FuncOp func, const DataFlowSolver& solver) : solver(solver) {
    // Function arguments and results keep factor 1.
    for (Value arg : func.getArguments())
      if (isScaled(arg)) pinned.push_back(arg);
    for (Operation& op : func.getBody().front()) analyzeOp(&op);
  }

  // Chooses a factor for each class with annotated ReLUs.
  void chooseFactors(double target, double margin) {
    for (const Domain& domain : domains) {
      if (!domain.isRelu) continue;
      auto lower =
          dyn_cast_or_null<FloatAttr>(domain.op->getAttr(kDomainLower));
      auto upper =
          dyn_cast_or_null<FloatAttr>(domain.op->getAttr(kDomainUpper));
      if (!lower || !upper) continue;
      double lo = lower.getValueAsDouble();
      double hi = upper.getValueAsDouble();
      if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi)) continue;
      // The domain is the calibrated range widened by `margin` about its
      // centre.
      double centre = (lo + hi) / 2;
      double maxAbs = std::abs(centre) + (hi - lo) / 2 / margin;
      double desired = std::max(1.0, margin * maxAbs / target);
      double& factor =
          factors.try_emplace(find(domain.value), 1.0).first->second;
      factor = std::max(factor, desired);
    }
    for (Value value : pinned) factors.erase(find(value));
  }

  // Folds the factors into the constants, domain attributes and validations.
  void rewrite(OpBuilder& builder) {
    llvm::SetVector<Operation*> replaced;
    auto replace = [&](OpOperand* operand, double factor) {
      if (factor == 1.0) return;
      Value scaled = scaleConstant(builder, operand->get(), factor);
      if (scaled == operand->get()) return;
      replaced.insert(operand->get().getDefiningOp());
      operand->set(scaled);
    };
    for (auto [operand, value] : divided)
      replace(operand, 1.0 / factorOf(value));
    for (const Weight& weight : weights)
      replace(weight.operand, factorOf(weight.input) / factorOf(weight.result));

    for (const Domain& domain : domains) {
      double factor = factorOf(domain.value);
      if (factor == 1.0) continue;
      for (StringLiteral name : {kDomainLower, kDomainUpper}) {
        auto bound = dyn_cast_or_null<FloatAttr>(domain.op->getAttr(name));
        if (!bound) continue;
        domain.op->setAttr(
            name,
            FloatAttr::get(bound.getType(), bound.getValueAsDouble() / factor));
      }
    }

    for (auto [validate, value] : validates) {
      double factor = factorOf(value);
      if (factor != 1.0)
        validate->setAttr(kDebugScale, builder.getF64FloatAttr(factor));
    }

    // Drop the original constants that are no longer used.
    while (!replaced.empty()) {
      Operation* op = replaced.pop_back_val();
      if (!isOpTriviallyDead(op)) continue;
      for (Value operand : op->getOperands())
        if (Operation* def = operand.getDefiningOp()) replaced.insert(def);
      op->erase();
    }
  }

 private:
  struct Weight {
    OpOperand* operand;
    Value input;
    Value result;
  };

  // An op with domain attributes, with a value of the class of its input.
  struct Domain {
    Operation* op;
    Value value;
    // Only ReLU domains choose the factor of their class.
    bool isRelu;
  };

  bool isScaled(Value value) const {
    return isFloatValue(value) && isSecret(value, &solver);
  }

  Value find(Value value) {
    Value root = value;
    for (auto it = parent.find(root); it != parent.end();
         it = parent.find(root))
      root = it->second;
    while (value != root) {
      Value& next = parent[value];
      value = std::exchange(next, root);
    }
    return root;
  }

  void unite(Value lhs, Value rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs != rhs) parent[rhs] = lhs;
  }

  double factorOf(Value value) {
    auto it = factors.find(find(value));
    return it == factors.end() ? 1.0 : it->second;
  }

  // Puts the scaled `operands` in the class of `result`, and records the
  // public ones to be divided by its factor. Fails if one of them is not a
  // constant that can be rebuilt.
  LogicalResult shareFactor(ArrayRef<OpOperand*> operands, Value result) {
    for (OpOperand* operand : operands)
      if (!isScaled(operand->get()) && !isScalable(operand->get()))
        return failure();
    for (OpOperand* operand : operands) {
      if (isScaled(operand->get()))
        unite(operand->get(), result);
      else
        divided.emplace_back(operand, result);
    }
    return success();
  }

  // A contraction or convolution of a scaled input with constant weights
  // moves from the factor of its input to the factor of its result by
  // rescaling the weights, and divides the accumulator it starts from.
  LogicalResult analyzeContraction(linalg::LinalgOp op) {
    if (!linalg::isaContractionOpInterface(op) &&
        !linalg::isaConvolutionOpInterface(op))
      return failure();
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1) return failure();
    OpOperand* input = op.getDpsInputOperand(0);
    OpOperand* weight = op.getDpsInputOperand(1);
    if (!isScaled(input->get())) std::swap(input, weight);
    if (!isScaled(input->get()) || isScaled(weight->get()) ||
        !isScalable(weight->get()))
      return failure();
    // Pooling ops match the convolution interface but never read the filter.
    if (op.getMatchingBlockArgument(weight).use_empty()) return failure();
    Value result = op->getResult(0);
    if (failed(shareFactor({op.getDpsInitOperand(0)}, result)))
      return failure();
    weights.push_back({weight, input->get(), result});
    return success();
  }

  LogicalResult analyzeLinalg(linalg::LinalgOp op) {
    if (op->getNumResults() == 0) return failure();
    if (succeeded(analyzeContraction(op))) return success();

    SmallVector<OpOperand*> divided;
    if (failed(analyzeBody(
            op, [&](Value value) { return isScaled(value); }, divided)))
      return failure();
    SmallVector<OpOperand*> operands(divided);
    for (OpOperand& operand : op->getOpOperands())
      if (isScaled(operand.get())) operands.push_back(&operand);
    Value result = op->getResult(0);
    if (failed(shareFactor(operands, result))) return failure();
    for (Value other : op->getResults()) unite(other, result);

    // The calibrated domain of an elementwise activation. The exporter
    // attaches it to the generic; it is moved into the body later.
    if (op->hasAttr(kDomainLower) || op->hasAttr(kDomainUpper))
      domains.push_back({op, result, isRelu(op)});
    return success();
  }

  LogicalResult analyzeTensorOp(Operation* op) {
    if (op->getNumResults() != 1) return failure();
    Value result = op->getResult(0);
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<tensor::CollapseShapeOp, tensor::ExpandShapeOp,
              tensor::ExtractSliceOp>([&](Operation* reshape) {
          return shareFactor({&reshape->getOpOperand(0)}, result);
        })
        .Case<tensor::PadOp>([&](tensor::PadOp pad) {
          Value padding = pad.getConstantPaddingValue();
          if (!padding || !matchPattern(padding, m_AnyZeroFloat()))
            return failure();
          return shareFactor({&pad.getSourceMutable()}, result);
        })
        .Default([](Operation*) { return failure(); });
  }

  void analyzeOp(Operation* op) {
    auto isScaledValue = [&](Value value) { return isScaled(value); };
    // The secretness analysis ignores values captured by a region, so they
    // are checked here.
    llvm::SetVector<Value> captured;
    for (Region& region : op->getRegions())
      getUsedValuesDefinedAbove(region, captured);
    if (!llvm::any_of(op->getOperands(), isScaledValue) &&
        !llvm::any_of(op->getResults(), isScaledValue) &&
        !llvm::any_of(captured, isScaledValue))
      return;
    // debug.validate checkpoints report the scaled value x / s; the factor is
    // recorded on the op.
    if (auto validate = dyn_cast<debug::ValidateOp>(op)) {
      validates.emplace_back(validate.getOperation(), validate.getInput());
      return;
    }

    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (succeeded(analyzeLinalg(linalgOp))) return;
    } else if (succeeded(analyzeTensorOp(op))) {
      return;
    }

    // Any other op sees the original values.
    for (Value value : op->getOperands())
      if (isScaled(value)) pinned.push_back(value);
    for (Value value : op->getResults())
      if (isScaled(value)) pinned.push_back(value);
    for (Value value : captured)
      if (isScaled(value)) pinned.push_back(value);
  }

  const DataFlowSolver& solver;
  DenseMap<Value, Value> parent;
  DenseMap<Value, double> factors;
  SmallVector<Value> pinned;
  // Constant operands to divide by the factor of the class of the value.
  SmallVector<std::pair<OpOperand*, Value>> divided;
  SmallVector<Weight> weights;
  SmallVector<Domain> domains;
  SmallVector<std::pair<Operation*, Value>> validates;
};

}  // namespace

struct EqualizeActivationRanges
    : impl::EqualizeActivationRangesBase<EqualizeActivationRanges> {
  using EqualizeActivationRangesBase::EqualizeActivationRangesBase;

  void runOnOperation() override {
    if (!(target > 0) || !(margin >= 1)) {
      getOperation()->emitError()
          << "equalize-activation-ranges needs target > 0 and margin >= 1";
      return signalPassFailure();
    }
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run SecretnessAnalysis.\n";
      return signalPassFailure();
    }

    OpBuilder builder(&getContext());
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      // The analysis only visits the ops of the entry block.
      if (!func.getBody().hasOneBlock()) continue;
      Analysis analysis(func, solver);
      analysis.chooseFactors(target, margin);
      analysis.rewrite(builder);
    }
  }
};

}  // namespace heir
}  // namespace mlir
