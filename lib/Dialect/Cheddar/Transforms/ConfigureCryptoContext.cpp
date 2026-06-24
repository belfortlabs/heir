#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h"

#include <algorithm>
#include <string>

#include "lib/Analysis/RotationAnalysis/RotationAnalysis.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarOps.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "lib/Utils/TransformUtils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"         // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"        // from @llvm-project

namespace mlir::heir::cheddar {

#define GEN_PASS_DEF_CHEDDARCONFIGURECRYPTOCONTEXT
#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h.inc"

namespace {

// Build a `<entry>__configure() -> (context, user_interface)` function in
// destination-passing tensor form:
//   %p   = cheddar.make_parameter ...
//   %ctx = cheddar.create_context %p, %ctx_init
//   %ui0 = cheddar.create_user_interface %ctx, %ui_init
//   %ui1 = cheddar.prepare_rot_key %ui0 {distance, maxLevel}   // per distance
//   return %ctx, %uiN
// Params come from the CKKS scheme attrs (logN/scale/Q/P); the rotation
// distances come from the shared RotationAnalysis (the same way the lattigo and
// openfhe backends discover which keys to generate). The two tensor results
// become owning context/user_interface out-params after one-shot-bufferize +
// buffer-results-to-out-params, and the cheddar-to-emitc boundary re-types them
// to owning smart-pointer references.
void buildConfigureFunc(ModuleOp moduleOp, func::FuncOp entry, int64_t logN,
                        int64_t logScale, DenseI64ArrayAttr Q,
                        DenseI64ArrayAttr P,
                        ArrayRef<int64_t> rotationIndices) {
  MLIRContext *ctx = moduleOp.getContext();
  int64_t maxLevel = Q.size() - 1;

  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(moduleOp.getBody());
  Location loc = entry.getLoc();

  auto ctxTensor = RankedTensorType::get({}, ContextType::get(ctx));
  auto uiTensor = RankedTensorType::get({}, UserInterfaceType::get(ctx));
  auto funcType = FunctionType::get(ctx, {}, {ctxTensor, uiTensor});
  // Discovered by name convention (`<entry>__configure`), like the lattigo
  // backend -- no marker attribute needed.
  std::string name = (entry.getSymName() + "__configure").str();
  auto configFunc = func::FuncOp::create(builder, loc, name, funcType);
  configFunc.setPublic();

  Block *bodyBlock = configFunc.addEntryBlock();
  builder.setInsertionPointToStart(bodyBlock);

  auto i64 = [&](int64_t v) { return builder.getI64IntegerAttr(v); };
  Value params = MakeParameterOp::create(builder, loc, ParameterType::get(ctx),
                                         i64(logN), i64(logScale), Q, P)
                     .getParams();
  // DPS destination buffers; bufferization hoists these allocs to the
  // out-params.
  Value ctxInit = bufferization::AllocTensorOp::create(builder, loc, ctxTensor,
                                                       ValueRange{})
                      .getResult();
  Value context = CreateContextOp::create(builder, loc, TypeRange{ctxTensor},
                                          ValueRange{params, ctxInit})
                      ->getResult(0);
  Value uiInit =
      bufferization::AllocTensorOp::create(builder, loc, uiTensor, ValueRange{})
          .getResult();
  Value ui = CreateUserInterfaceOp::create(builder, loc, TypeRange{uiTensor},
                                           ValueRange{context, uiInit})
                 ->getResult(0);
  for (int64_t d : rotationIndices)
    ui = PrepareRotKeyOp::create(builder, loc, TypeRange{uiTensor}, ui, i64(d),
                                 i64(maxLevel))
             ->getResult(0);
  func::ReturnOp::create(builder, loc, ValueRange{context, ui});
}

}  // namespace

struct CheddarConfigureCryptoContext
    : public impl::CheddarConfigureCryptoContextBase<
          CheddarConfigureCryptoContext> {
  using CheddarConfigureCryptoContextBase::CheddarConfigureCryptoContextBase;

  void runOnOperation() override {
    auto moduleOp = cast<ModuleOp>(getOperation());
    MLIRContext *ctx = &getContext();

    // RotationAnalysis requires -sccp to have propagated constants so the
    // rotation indices are statically detectable (mirrors the lattigo/openfhe
    // configure passes).
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createSCCPPass());
    pipeline.addPass(createCanonicalizerPass());
    (void)runPipeline(pipeline, moduleOp);

    auto schemeParamAttr = moduleOp->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);

    if (schemeParamAttr) {
      int64_t logN = schemeParamAttr.getLogN();
      int64_t logDefaultScale = schemeParamAttr.getLogDefaultScale();
      // CKKS SchemeParam already types Q/P as DenseI64ArrayAttr.
      DenseI64ArrayAttr Q = schemeParamAttr.getQ();
      DenseI64ArrayAttr P = schemeParamAttr.getP();

      moduleOp->setAttr("cheddar.logN",
                        IntegerAttr::get(IntegerType::get(ctx, 64), logN));
      moduleOp->setAttr(
          "cheddar.logDefaultScale",
          IntegerAttr::get(IntegerType::get(ctx, 64), logDefaultScale));

      // CHEDDAR derives its RNS limbs from the CKKS Q/P moduli directly;
      // shallow computations must already carry enough primes via scheme
      // management.
      if (Q) moduleOp->setAttr("cheddar.Q", Q);
      if (P) moduleOp->setAttr("cheddar.P", P);

      // Generate the <entry>__configure function so the backend harness can set
      // up the context/keys with one call.
      auto entry = detectEntryFunction(moduleOp, entryFunction);
      if (entry && Q && P) {
        // Discover the rotation keys to generate, the same way the other
        // backends do (RotationAnalysis over the program's rotation ops).
        RotationAnalysis rotationAnalysis;
        if (failed(rotationAnalysis.run(moduleOp))) {
          entry->emitOpError("failed to compute static rotation indices");
          signalPassFailure();
          return;
        }
        const auto &indexSet = rotationAnalysis.getRotationIndices();
        SmallVector<int64_t> rotationIndices(indexSet.begin(), indexSet.end());
        llvm::sort(rotationIndices);  // deterministic key-prep order
        buildConfigureFunc(moduleOp, entry, logN, logDefaultScale, Q, P,
                           rotationIndices);
      }

      // Remove the CKKS scheme param attribute — consumed
      moduleOp->removeAttr(ckks::CKKSDialect::kSchemeParamAttrName);
    }

    // Remove scheme.ckks marker attribute
    moduleOp->removeAttr("scheme.ckks");
  }
};

}  // namespace mlir::heir::cheddar
