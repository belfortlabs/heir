#include "lib/Analysis/PendingRescaleAnalysis/PendingRescaleAnalysis.h"

#include "lib/Analysis/Utils.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "pending-rescale-analysis"

namespace mlir {
namespace heir {

LogicalResult PendingRescaleAnalysis::visitOperation(
    Operation* op, ArrayRef<const PendingRescaleLattice*> operands,
    ArrayRef<PendingRescaleLattice*> results) {
  auto propagate = [&](Value value, const PendingRescaleState& state) {
    auto* lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(state);
    propagateIfChanged(lattice, changed);
  };

  if (auto genericOp = dyn_cast<secret::GenericOp>(op)) {
    for (BlockArgument arg : genericOp.getBody()->getArguments()) {
      propagate(arg, PendingRescaleState(inputsPending));
    }
    return success();
  }

  // condition on result secretness
  SmallVector<OpResult> secretResults;
  getSecretResults(op, secretResults);
  if (secretResults.empty()) {
    return success();
  }

  PendingRescaleState resultState(false);
  if (isa<ResetsMulDepthOpInterface>(op)) {
    // A refreshed value is in the state of an input.
    resultState = PendingRescaleState(inputsPending);
  } else if (isa<mgmt::ModReduceOp>(op)) {
    resultState = PendingRescaleState(false);
  } else if (isa<IncreasesMulDepthOpInterface>(op)) {
    // The operands are mod reduced before the op where needed, so the result
    // only depends on whether the op rescales it.
    resultState = PendingRescaleState(!isa<ReducesLevelOpInterface>(op));
  } else {
    SmallVector<OpOperand*> secretOperands;
    getSecretOperands(op, secretOperands);
    for (auto* operand : secretOperands) {
      resultState = PendingRescaleState::join(
          resultState, getLatticeElement(operand->get())->getValue());
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "transferForward: " << op->getName() << " = "
                          << resultState << "\n");
  for (auto result : secretResults) {
    propagate(result, resultState);
  }
  return success();
}

void PendingRescaleAnalysis::visitExternalCall(
    CallOpInterface call,
    ArrayRef<const PendingRescaleLattice*> argumentLattices,
    ArrayRef<PendingRescaleLattice*> resultLattices) {
  ::mlir::heir::visitExternalCall<PendingRescaleState, PendingRescaleLattice>(
      call, argumentLattices, resultLattices,
      [this](AnalysisState* state, ChangeResult changed) {
        this->propagateIfChanged(state, changed);
      });
}

bool hasPendingRescale(Value value, DataFlowSolver* solver) {
  auto* lattice = solver->lookupState<PendingRescaleLattice>(value);
  return lattice && lattice->getValue().isPending();
}

}  // namespace heir
}  // namespace mlir
