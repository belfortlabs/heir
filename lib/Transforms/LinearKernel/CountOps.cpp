#include "lib/Transforms/LinearKernel/CountOps.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>

#include "llvm/include/llvm/ADT/DenseMap.h"         // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"          // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_COUNTOPS
#include "lib/Transforms/LinearKernel/CountOps.h.inc"

namespace {

struct Counts {
  std::map<std::string, uint64_t> ops;
  std::set<int64_t> shifts;
};

// The number of scalar operations an elementwise operation on a tensor of
// non-builtin elements stands for.
uint64_t elementMultiplicity(Operation* op) {
  if (op->getNumResults() == 0) return 1;
  auto shaped = dyn_cast<ShapedType>(op->getResult(0).getType());
  if (!shaped || !shaped.hasStaticShape()) return 1;
  if (isa<BuiltinDialect>(shaped.getElementType().getDialect())) return 1;
  return shaped.getNumElements();
}

std::optional<uint64_t> tripCount(Operation* op) {
  if (auto affineFor = dyn_cast<affine::AffineForOp>(op)) {
    return affine::getConstantTripCount(affineFor);
  }
  if (auto scfFor = dyn_cast<scf::ForOp>(op)) {
    auto lb = getConstantIntValue(scfFor.getLowerBound());
    auto ub = getConstantIntValue(scfFor.getUpperBound());
    auto step = getConstantIntValue(scfFor.getStep());
    if (!lb || !ub || !step || *step <= 0) return std::nullopt;
    if (*ub <= *lb) return 0;
    return (*ub - *lb + *step - 1) / *step;
  }
  return std::nullopt;
}

struct Counter {
  ModuleOp module;
  llvm::DenseMap<Operation*, Counts> perFunction;

  const Counts& countFunction(func::FuncOp func) {
    auto it = perFunction.find(func.getOperation());
    if (it != perFunction.end()) return it->second;
    Counts counts;
    for (Region& region : func->getRegions()) countRegion(region, 1, counts);
    return perFunction[func.getOperation()] = std::move(counts);
  }

  void countRegion(Region& region, uint64_t weight, Counts& counts) {
    for (Block& block : region)
      for (Operation& op : block) countOp(&op, weight, counts);
  }

  void countOp(Operation* op, uint64_t weight, Counts& counts) {
    if (isa<affine::AffineForOp, scf::ForOp>(op)) {
      auto trips = tripCount(op);
      if (!trips) counts.ops["unknown-loop"] += weight;
      for (Region& region : op->getRegions())
        countRegion(region, weight * trips.value_or(1), counts);
      return;
    }
    if (auto call = dyn_cast<func::CallOp>(op)) {
      auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
      if (callee && !callee.isExternal()) {
        const Counts& calleeCounts = countFunction(callee);
        for (const auto& [name, count] : calleeCounts.ops)
          counts.ops[name] += weight * count;
        counts.shifts.insert(calleeCounts.shifts.begin(),
                             calleeCounts.shifts.end());
      }
      counts.ops["func.call"] += weight;
      return;
    }
    std::string name = op->getName().getStringRef().str();
    counts.ops[name] += weight * elementMultiplicity(op);
    if (name == "ckks.rotate" || name == "bgv.rotate_cols" ||
        name == "tensor_ext.rotate") {
      IntegerAttr shift;
      for (NamedAttribute attr : op->getAttrs())
        if (auto intAttr = dyn_cast<IntegerAttr>(attr.getValue()))
          shift = intAttr;
      if (auto props = op->getPropertiesAsAttribute())
        if (auto dict = dyn_cast<DictionaryAttr>(props))
          for (NamedAttribute attr : dict)
            if (auto intAttr = dyn_cast<IntegerAttr>(attr.getValue()))
              shift = intAttr;
      std::optional<int64_t> constantShift;
      if (shift) constantShift = shift.getValue().getSExtValue();
      // A shift operand that is a constant is as good as a static one.
      for (Value operand : op->getOperands().drop_front())
        if (auto value = getConstantIntValue(operand)) constantShift = value;
      if (constantShift)
        counts.shifts.insert(*constantShift);
      else
        counts.ops[name + "(dynamic shift)"] += weight;
    }
    if (name == "kernel.eval_chebyshev") {
      if (auto coefficients = op->getAttrOfType<ArrayAttr>("coefficients")) {
        counts.ops[name + "(degree " + std::to_string(coefficients.size() - 1) +
                   ")"] += weight * elementMultiplicity(op);
      }
    }
    if (isa<linalg::LinalgOp>(op)) return;
    for (Region& region : op->getRegions()) countRegion(region, weight, counts);
  }
};

}  // namespace

struct CountOps : impl::CountOpsBase<CountOps> {
  using CountOpsBase::CountOpsBase;

  void runOnOperation() override {
    Counter counter{getOperation()};
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      if (func.isExternal()) continue;
      const Counts& counts = counter.countFunction(func);
      for (const auto& [name, count] : counts.ops)
        llvm::errs() << "count-ops\t" << func.getSymName() << "\t" << name
                     << "\t" << count << "\n";
      llvm::errs() << "count-ops\t" << func.getSymName()
                   << "\tdistinct-shifts\t" << counts.shifts.size() << "\n";
    }
    markAllAnalysesPreserved();
  }
};

}  // namespace heir
}  // namespace mlir
