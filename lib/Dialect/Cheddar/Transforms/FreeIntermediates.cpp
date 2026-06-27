#include "lib/Dialect/Cheddar/Transforms/FreeIntermediates.h"

#include <iterator>
#include <utility>

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir::heir::cheddar {

#define GEN_PASS_DEF_CHEDDARFREEINTERMEDIATES
#include "lib/Dialect/Cheddar/Transforms/FreeIntermediates.h.inc"

namespace {

// Is this a memref of a move-only CHEDDAR payload (owns GPU memory)?
static bool isPayloadMemRef(Type t) {
  if (!isa<MemRefType>(t)) return false;
  return isa<CiphertextType, PlaintextType>(
      cast<MemRefType>(t).getElementType());
}

struct CheddarFreeIntermediates
    : impl::CheddarFreeIntermediatesBase<CheddarFreeIntermediates> {
  void runOnOperation() override {
    // (buffer, op-after-which-to-free). Collected during analysis; deallocs are
    // inserted only afterwards so we never invalidate block ordering /
    // iterators mid-analysis (which would make this quadratic).
    SmallVector<std::pair<Value, Operation *>> toFree;

    // Process each block independently with a single linear forward sweep, so
    // last-use is found by op position without any O(n) isBeforeInBlock calls.
    getOperation()->walk([&](Block *block) {
      // Payload buffers allocated in *this* block (the only freeable locals).
      DenseSet<Value> localAllocs;
      bool alreadyFreed = false;
      for (Operation &op : *block) {
        if (auto a = dyn_cast<memref::AllocOp>(&op)) {
          Value buf = a.getMemref();
          if (isPayloadMemRef(buf.getType())) localAllocs.insert(buf);
        }
        if (isa<memref::DeallocOp>(op)) alreadyFreed = true;
      }
      if (localAllocs.empty()) return;

      // Last top-level op in this block that (transitively) uses each local
      // alloc. Walking each top-level op's subtree once attributes nested uses
      // (e.g. inside an scf.for body) to the enclosing block-level op.
      DenseMap<Value, Operation *> lastUse;
      for (Operation &top : *block) {
        top.walk([&](Operation *inner) {
          for (Value operand : inner->getOperands())
            if (localAllocs.contains(operand)) lastUse[operand] = &top;

          // Split-preprocessing stores an encoded payload into persistent
          // storage as bufferized `%v = memref.load %localAlloc[] ;
          // memref.store %v, %storage[...]` (the rank-0 tensor.extract -> store
          // from PreprocessingToCheddar). The store's operands are %v and
          // %storage, NOT %localAlloc, so the operand sweep above would treat
          // the memref.load as the alloc's last use and free it BEFORE the
          // store -- leaving an EMPTY payload (NPInfo 0,0,0) in storage. Treat
          // the store into a non-local payload memref as a use of the loaded
          // local alloc so its dealloc lands after the store.
          if (auto store = dyn_cast<memref::StoreOp>(inner)) {
            Value dst = store.getMemref();
            if (!localAllocs.contains(dst) && isPayloadMemRef(dst.getType())) {
              if (auto load =
                      store.getValueToStore().getDefiningOp<memref::LoadOp>()) {
                Value src = load.getMemref();
                if (localAllocs.contains(src)) lastUse[src] = &top;
              }
            }
          }
        });
      }

      for (Value buf : localAllocs) {
        auto it = lastUse.find(buf);
        if (it == lastUse.end()) continue;  // unused -> leave to scope cleanup
        toFree.push_back({buf, it->second});
      }
      // If any explicit dealloc already exists in the block, skip those buffers
      // (a prior dealloc pass ran); cheap guard since it's rare here.
      if (alreadyFreed) {
        for (auto &pair : toFree)
          for (Operation *u : pair.first.getUsers())
            if (isa<memref::DeallocOp>(u)) {
              pair.second = nullptr;
              break;
            }
      }
    });

    for (auto &[buf, after] : toFree) {
      if (!after) continue;
      OpBuilder b(after->getBlock(), std::next(after->getIterator()));
      memref::DeallocOp::create(b, buf.getLoc(), buf);
    }
  }
};

}  // namespace
}  // namespace mlir::heir::cheddar
