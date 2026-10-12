#include "lib/Target/Cheddar/MaxPoolInterfaces.h"

#include <cstdint>
#include <optional>
#include <string>

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Kernel/IR/KernelDialect.h"
#include "lib/Dialect/Kernel/IR/KernelOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Target/Cheddar/MaxPoolPlan.h"
#include "lib/Utils/MaxPoolUtils.h"
#include "llvm/include/llvm/ADT/Twine.h"              // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

namespace {

// The verifier of kernel.max_pool accepts only shapes that the planner
// accepts, so a rejection here is a disagreement between the two.
MaxPoolRequirements getRequirements(kernel::MaxPoolOp op) {
  std::string error;
  std::optional<MaxPoolRequirements> requirements = getMaxPoolRequirements(
      {static_cast<int64_t>(op.getNumSlots()),
       static_cast<int64_t>(op.getInputLength()),
       static_cast<int64_t>(op.getWindowSize()),
       static_cast<int64_t>(op.getStride()),
       static_cast<int64_t>(op.getDilation()), op.getCeilMode(),
       op.getValueBound().convertToDouble()},
      &error);
  if (!requirements)
    llvm::report_fatal_error(
        llvm::Twine("the Cyclops max pool planner rejects a kernel.max_pool "
                    "that the verifier accepts: ") +
        error);
  return *requirements;
}

struct KernelMaxPoolReducesLevelModel
    : public ReducesLevelOpInterface::ExternalModel<
          KernelMaxPoolReducesLevelModel, kernel::MaxPoolOp> {
  int getLevelsToDrop(Operation* op) const {
    return getRequirements(cast<kernel::MaxPoolOp>(op)).levelDrop;
  }

  int getRequiredInputLevels(Operation* op) const {
    return getRequirements(cast<kernel::MaxPoolOp>(op)).minimumInputLevel;
  }

  SmallVector<OpOperand*> getOperandsToReduce(
      Operation* op, const DataFlowSolver* solver) const {
    return {&op->getOpOperand(0)};
  }
};

struct KernelMaxPoolBootstrapsInternallyModel
    : public BootstrapsInternallyOpInterface::ExternalModel<
          KernelMaxPoolBootstrapsInternallyModel, kernel::MaxPoolOp> {
  int getMinimumBootstrapEndLevel(Operation* op) const {
    return getRequirements(cast<kernel::MaxPoolOp>(op))
        .minimumBootstrapEndLevel;
  }
};

// A max pool lowers to kernel.max_pool, whose level cost depends on the slot
// count. Layout propagation records the result layout on the op, and the slot
// count is the bound of that layout's slot variable.
struct LinalgMaxPoolReducesLevelModel
    : public ReducesLevelOpInterface::ExternalModel<
          LinalgMaxPoolReducesLevelModel, linalg::PoolingNcwMaxOp> {
  static std::optional<MaxPoolRequirements> getRequirements(Operation* op) {
    std::string reason;
    FailureOr<SupportedMaxPool> pool =
        getSupportedMaxPool(cast<linalg::PoolingNcwMaxOp>(op), reason);
    auto layout = op->getAttrOfType<tensor_ext::LayoutAttr>(
        tensor_ext::TensorExtDialect::kLayoutAttrName);
    if (failed(pool) || !layout) return std::nullopt;
    presburger::IntegerRelation relation = layout.getIntegerRelation();
    std::optional<int64_t> lastSlot = relation.getConstantBound64(
        presburger::BoundType::UB,
        relation.getVarKindOffset(presburger::VarKind::Range) + 1);
    if (!lastSlot) return std::nullopt;
    return getMaxPoolRequirements(pool->getKernelShape(*lastSlot + 1));
  }

  // Before layout propagation, count one level, as for the other linalg ops.
  int getLevelsToDrop(Operation* op) const {
    std::optional<MaxPoolRequirements> requirements = getRequirements(op);
    return requirements ? requirements->levelDrop : 1;
  }

  int getRequiredInputLevels(Operation* op) const {
    std::optional<MaxPoolRequirements> requirements = getRequirements(op);
    return requirements ? requirements->minimumInputLevel : 1;
  }

  SmallVector<OpOperand*> getOperandsToReduce(
      Operation* op, const DataFlowSolver* solver) const {
    return {&op->getOpOperand(0)};
  }
};

}  // namespace

void registerMaxPoolInterfaceExternalModels(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, kernel::KernelDialect* dialect) {
    kernel::MaxPoolOp::attachInterface<KernelMaxPoolReducesLevelModel,
                                       KernelMaxPoolBootstrapsInternallyModel>(
        *ctx);
  });
  registry.addExtension(+[](MLIRContext* ctx, linalg::LinalgDialect* dialect) {
    linalg::PoolingNcwMaxOp::attachInterface<LinalgMaxPoolReducesLevelModel>(
        *ctx);
  });
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
