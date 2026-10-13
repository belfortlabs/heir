#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/RangeAnalysis/RangeAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Parameters/CKKS/Params.h"
#include "lib/Parameters/Cheddar/ParameterFile.h"
#include "lib/Parameters/RLWEParams.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "lib/Utils/LogArithmetic.h"
#include "llvm/include/llvm/ADT/SmallVector.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/Twine.h"                   // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "llvm/include/llvm/Support/DebugLog.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Error.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project
#include "mlir/include/mlir/Support/WalkResult.h"          // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Transforms/GenerateParam/GenerateParam.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"        // from @llvm-project
// IWYU pragma: end_keep

#define DEBUG_TYPE "generate-param-ckks"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_GENERATEPARAMCKKS
#include "lib/Transforms/GenerateParam/GenerateParam.h.inc"

namespace {
bool containsBootstrap(Operation* op) {
  auto result = op->walk([&](Operation* walkOp) {
    if (isa<ResetsMulDepthOpInterface>(walkOp)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

// scale-snu/cheddar's BootContext uses one explicit chain containing both the
// compute levels and the bootstrap circuit: CoeffToSlot, eight EvalMod levels,
// and SlotToCoeff. The bootstrap returns to the original compute maximum, so
// these levels extend the generated chain without changing HEIR's level model.
constexpr int kCheddarBootNumCts = 4;
constexpr int kCheddarBootNumStc = 2;
constexpr int kCheddarBootEvalModLevels = 8;
constexpr int kCheddarBootArithmeticLevels =
    kCheddarBootNumCts + kCheddarBootEvalModLevels;
constexpr int kCheddarBootModBits = 50;
constexpr int kCheddarBootOverhead =
    kCheddarBootNumCts + kCheddarBootNumStc + kCheddarBootEvalModLevels;

constexpr int kDefaultScalingModBits = 45;

// The largest prime of a model chain modeling a parameter file: a model
// prime is one 64-bit modulus, and HEIR's own chains use primes of at most
// 60 bits.
constexpr int kMaxModelPrimeBits = 60;

// The bootstrap message ratio for a parameter file that does not state one:
// log2(q0 / scale) less two bits. A smaller ratio puts large messages at the
// edge of EvalMod's approximation domain, a larger one leaves too little
// room above the noise floor; two bits below the headroom balanced the two
// on scale-snu chains.
constexpr double kLogMessageRatioMargin = 2.0;

DenseI64ArrayAttr primesAttr(MLIRContext* context,
                             const std::vector<uint64_t>& primes) {
  SmallVector<int64_t> values(primes.begin(), primes.end());
  return DenseI64ArrayAttr::get(context, values);
}

std::optional<int64_t> optionalField(int value) {
  if (value < 0) return std::nullopt;
  return value;
}

}  // namespace

struct GenerateParamCKKS : impl::GenerateParamCKKSBase<GenerateParamCKKS> {
  using GenerateParamCKKSBase::GenerateParamCKKSBase;

  // The slot count the layouts are packed for: the min-slot-count option, or
  // the count layout-propagation recorded on the module.
  int64_t slotCount = 0;

  // In CKKS, the modulus for L0 should be larger than the
  // scaling modulus, however, the number of extra bits is often
  // empirically chosen. We use RangeAnalysis to find the
  // maximum number of extra bits needed for the L0 modulus.
  // TODO(#2754): improve this analysis
  std::optional<int> getExtraBitsForLevel0() {
    LDBG() << "Using range analysis to determine extra bits for level 0";
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    // RangeAnalysis depends on SecretnessAnalysis
    solver.load<SecretnessAnalysis>();
    // For double input in range [-1, 1], we use Log2Arithmetic::of(1) to
    // represent it.
    solver.load<RangeAnalysis>(Log2Arithmetic::of(inputRange));
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
    }

    std::optional<double> extraBits;

    getOperation()->walk([&](Operation* op) {
      for (auto result : op->getResults()) {
        if (mgmt::shouldHaveMgmtAttribute(result, &solver) &&
            getLevelFromMgmtAttr(result) == 0) {
          auto range = getRange(result, &solver);
          if (range.has_value()) {
            auto resultExtraBits = range->getLog2Value();
            if (!extraBits.has_value() || resultExtraBits > extraBits.value()) {
              extraBits = resultExtraBits;
            }
          }
        }
      }
    });

    if (!extraBits.has_value()) {
      return std::nullopt;
    }
    // 2 more bits for cushion
    int level0ModBits = ceil(extraBits.value()) + 2;
    LDBG() << "Decided on " << level0ModBits << " bits for level 0";
    return level0ModBits;
  }

  // Takes the modulus chain from a CHEDDAR parameter file. The passes that
  // read ckks.scheme_param after this one assume one prime per level:
  // secret-to-ckks builds the ciphertext type's RNS ring from one modulus per
  // level, and populate-scale and validate-scale read each level's log2 size.
  // A file's level can span several primes, so ckks.scheme_param gets one
  // model prime per level instead, sized to the file's modulus growth at that
  // level (its level-0 modulus, capped at kMaxModelPrimeBits, for level 0).
  // The file's own primes and level layout go into cheddar.parameter_set,
  // which the Cheddar backend reads.
  LogicalResult importCheddarParameters(int computeMaxLevel,
                                        bool hasBootstrap) {
    Operation* module = getOperation();
    MLIRContext* context = &getContext();
    if (!moduleIsCheddar(module))
      return module->emitError(
          "cheddar-parameter-file requires a Cheddar backend module "
          "(annotate-module backend=cheddar)");
    llvm::Expected<cheddar::ParameterFile> file =
        cheddar::ParameterFile::load(cheddarParameterFile);
    if (!file) return module->emitError() << llvm::toString(file.takeError());
    if (scalingModBits != 0 && scalingModBits != file->logDefaultScale)
      return module->emitError()
             << "scaling-mod-bits=" << static_cast<int>(scalingModBits)
             << " conflicts with the " << file->logDefaultScale
             << "-bit scale of " << cheddarParameterFile;

    int minLogDegree = 0;
    while ((int64_t{1} << minLogDegree) < 2 * static_cast<int64_t>(slotCount))
      ++minLogDegree;
    const cheddar::RingProfile* profile =
        file->selectProfile(minLogDegree, computeMaxLevel, hasBootstrap);
    if (!profile) {
      InFlightDiagnostic diagnostic = module->emitError();
      diagnostic << "no ring profile in " << cheddarParameterFile << " holds "
                 << computeMaxLevel << " levels"
                 << (hasBootstrap ? " plus the bootstrap chain" : "")
                 << " at logN >= " << minLogDegree << "; the file offers";
      for (const cheddar::RingProfile& candidate : file->profiles) {
        diagnostic << " logN " << candidate.logDegree << ": "
                   << candidate.defaultEncryptionLevel << " levels";
        if (file->hasBootstrap(candidate))
          diagnostic << " (" << file->topLevel(candidate, true)
                     << " with bootstrapping)";
        diagnostic << ",";
      }
      return diagnostic;
    }
    // A bootstrap lands at a fixed level of the file's chain, and HEIR's
    // level model puts bootstrap outputs at the program's top level. A
    // program whose deepest stretch between bootstraps is shorter than the
    // residual chain is placed at the top of the chain so its bootstraps land
    // where the runtime's do, and its lowest levels go unused. Without a
    // bootstrap the program computes on the chain's lowest levels, up to its
    // own depth. Either way the top is pinned so that later level annotation
    // keeps the program where the chain was sized.
    if (hasBootstrap) computeMaxLevel = file->topLevel(*profile, true);
    OpBuilder builder(context);
    module->setAttr(mgmt::MgmtDialect::kTopLevelAttrName,
                    builder.getI64IntegerAttr(computeMaxLevel));
    OpPassManager annotate("builtin.module");
    annotate.addPass(mgmt::createAnnotateMgmt());
    if (failed(runPipeline(annotate, module))) return failure();
    // A program that does not bootstrap is given the levels it uses where the
    // runtime accepts a prefix of the chain (see ParameterFile::prefixChains).
    std::vector<cheddar::LevelLayout> layout = file->levels;
    if (!hasBootstrap && file->prefixChains) layout.resize(computeMaxLevel + 1);
    int64_t defaultEncryptionLevel =
        hasBootstrap ? profile->defaultEncryptionLevel : computeMaxLevel;
    LDBG() << "Selected ring profile logN=" << profile->logDegree << " with "
           << layout.size() << " levels";

    // The file fixes the level-0 headroom; report when the program's values
    // at level 0 are known to need more, the way an explicit first-mod-bits
    // choice is checked.
    double headroom = file->log2Modulus(layout[0]) - file->logDefaultScale;
    if (std::optional<int> extraBits = getExtraBitsForLevel0();
        extraBits && *extraBits > headroom)
      module->emitWarning()
          << "Range analysis indicates that level 0 must be larger than the "
             "scaling modulus by at least "
          << *extraBits << " bits, but " << cheddarParameterFile << " leaves "
          << headroom << " bits";

    int ringDim = 1 << profile->logDegree;
    std::vector<int64_t> existing;
    std::vector<int64_t> qi;
    std::vector<int64_t> pi;
    std::vector<double> logqi;
    std::vector<double> logpi;
    // findPrime returns a larger prime when no NTT-friendly prime of the
    // requested size exists, which would model a growth the file does not
    // have, so a model prime must come out at the size it models.
    auto modelPrime = [&](int bits,
                          const llvm::Twine& what) -> FailureOr<int64_t> {
      if (bits >= 1 && bits <= kMaxModelPrimeBits) {
        int64_t prime = findPrime(bits, ringDim, existing);
        if (std::llround(std::log2(prime)) == bits) {
          existing.push_back(prime);
          return prime;
        }
      }
      module->emitError() << cheddarParameterFile << ": " << what << " is "
                          << bits << " bits, which no model prime at logN "
                          << profile->logDegree << " can have";
      return failure();
    };
    for (size_t level = 0; level < layout.size(); ++level) {
      int bits = std::llround(file->log2ModulusGrowth(layout, level));
      // Nothing rescales by the level-0 modulus: the model reads its size
      // only as the room above the scale, so a smaller model prime keeps
      // that bound conservative. A 32-bit chain's level 0 can hold more
      // bits than one model prime.
      if (level == 0) bits = std::min(bits, kMaxModelPrimeBits);
      FailureOr<int64_t> prime =
          modelPrime(bits, "the modulus growth at level " + llvm::Twine(level));
      if (failed(prime)) return failure();
      qi.push_back(*prime);
      logqi.push_back(std::log2(*prime));
    }
    for (uint64_t aux : file->auxPrimes) {
      int bits = std::llround(std::log2(static_cast<double>(aux)));
      FailureOr<int64_t> prime =
          modelPrime(bits, "auxiliary prime " + llvm::Twine(aux));
      if (failed(prime)) return failure();
      pi.push_back(*prime);
      logpi.push_back(std::log2(*prime));
    }
    int dnum =
        static_cast<int>(std::ceil(static_cast<double>(qi.size()) / pi.size()));
    ckks::SchemeParam schemeParam(
        RLWESchemeParam(ringDim, layout.size() - 1, logqi, qi, dnum, logpi, pi,
                        usePublicKey, /*encryptionTechniqueExtended=*/true),
        file->logDefaultScale);
    LDBG() << "Scheme Param (model chain):\n" << schemeParam;

    module->setAttr(kRequestedSlotCountAttrName,
                    builder.getI64IntegerAttr(slotCount));
    module->setAttr(kActualSlotCountAttrName,
                    builder.getI64IntegerAttr(ringDim / 2));
    module->setAttr(ckks::CKKSDialect::kSchemeParamAttrName,
                    ckks::SchemeParamAttr::get(
                        context, profile->logDegree,
                        DenseI64ArrayAttr::get(context, ArrayRef(qi)),
                        DenseI64ArrayAttr::get(context, ArrayRef(pi)),
                        file->logDefaultScale,
                        usePublicKey ? ckks::CKKSEncryptionType::pk
                                     : ckks::CKKSEncryptionType::sk,
                        ckks::CKKSEncryptionTechnique::extended));

    SmallVector<int64_t> levelConfig;
    for (const cheddar::LevelLayout& level : layout) {
      levelConfig.push_back(level.numMain);
      levelConfig.push_back(level.numTerminal);
    }
    DenseI64ArrayAttr additionalBase;
    if (file->additionalBase != cheddar::LevelLayout{})
      additionalBase = builder.getDenseI64ArrayAttr(
          {file->additionalBase.numMain, file->additionalBase.numTerminal});
    // Rings that do not bootstrap keep a single secret: the sparse companion
    // takes the dense weight.
    int64_t sparseHammingWeight =
        hasBootstrap ? file->boot->sparseHammingWeight : profile->hammingWeight;
    module->setAttr(
        cheddar::kParameterSetAttrName,
        cheddar::ParameterSetAttr::get(
            context, profile->logDegree, file->logDefaultScale,
            primesAttr(context, file->mainPrimes),
            primesAttr(context, file->auxPrimes),
            file->terminalPrimes.empty()
                ? DenseI64ArrayAttr()
                : primesAttr(context, file->terminalPrimes),
            builder.getDenseI64ArrayAttr(levelConfig), file->wordBits(),
            defaultEncryptionLevel, additionalBase,
            optionalField(file->defaultNumAux),
            file->levelSpecificKs ? builder.getBoolAttr(*file->levelSpecificKs)
                                  : BoolAttr(),
            optionalField(file->maxKeySwitchAux),
            profile->maxLogPq > 0.0 ? builder.getF64FloatAttr(profile->maxLogPq)
                                    : FloatAttr(),
            profile->hammingWeight, sparseHammingWeight));
    if (!hasBootstrap) return success();

    const cheddar::BootstrapConfig& boot = *file->boot;
    int64_t logMessageRatio = boot.logMessageRatio;
    if (logMessageRatio <= 0) {
      logMessageRatio =
          static_cast<int64_t>(std::floor(headroom - kLogMessageRatioMargin));
      if (logMessageRatio <= 0)
        return module->emitError()
               << cheddarParameterFile << " leaves " << headroom
               << " bits between q0 and the scale, too few for a bootstrap";
    }
    cheddar::EvalModAttr evalMod;
    if (boot.evalMod.present) {
      const cheddar::EvalModConfig& config = boot.evalMod;
      evalMod = cheddar::EvalModAttr::get(
          context,
          config.type.empty() ? StringAttr()
                              : builder.getStringAttr(config.type),
          optionalField(config.degree), optionalField(config.interval),
          optionalField(config.logIntervalReduction),
          optionalField(config.invDegree),
          config.invType.empty() ? StringAttr()
                                 : builder.getStringAttr(config.invType),
          config.invInterval < 0.0
              ? FloatAttr()
              : builder.getF64FloatAttr(config.invInterval));
    }
    module->setAttr(cheddar::kBootstrapConfigAttrName,
                    cheddar::BootstrapConfigAttr::get(
                        context, boot.numCtsLevels, boot.numStcLevels,
                        boot.numEvalModLevels, logMessageRatio, evalMod));
    return success();
  }

  void runOnOperation() override {
    slotCount = getLayoutSlotCount(getOperation(), minSlotCount);
    LDBG() << "Starting generate-param-ckks pass";

    std::optional<int> maxLevel = getMaxLevel(getOperation());
    LDBG() << "Max level identified as " << maxLevel;

    if (auto schemeParamAttr =
            getOperation()->getAttrOfType<ckks::SchemeParamAttr>(
                ckks::CKKSDialect::kSchemeParamAttrName)) {
      if (auto module = dyn_cast<ModuleOp>(getOperation());
          module && failed(verifyRingDegree(
                        module, int64_t{1} << schemeParamAttr.getLogN()))) {
        signalPassFailure();
        return;
      }
      // TODO: put this in validate-noise once CKKS noise model is in
      auto schemeParam = ckks::getSchemeParamFromAttr(schemeParamAttr);
      if (schemeParam.getLevel() < maxLevel.value_or(0)) {
        getOperation()->emitOpError()
            << "The level in the scheme param is smaller than the max level.\n";
        signalPassFailure();
        return;
      }
      return;
    }

    bool hasBootstrap = containsBootstrap(getOperation());
    if (!cheddarParameterFile.empty()) {
      if (failed(importCheddarParameters(maxLevel.value_or(0), hasBootstrap)))
        signalPassFailure();
      return;
    }

    if (scalingModBits == 0) scalingModBits = kDefaultScalingModBits;

    if (firstModBits == 0 || validateFirstModBits) {
      auto extraBits = getExtraBitsForLevel0();
      if (!extraBits.has_value()) {
        emitError(getOperation()->getLoc())
            << "Cannot generate CKKS parameters without first modulus bits "
               "or extra bits for level 0.\n";
        signalPassFailure();
        return;
      }

      if (firstModBits == 0) {
        firstModBits = scalingModBits + extraBits.value();
        LDBG() << "First modulus bits not specified, using " << firstModBits
               << " bits.";
      } else if (extraBits.has_value() &&
                 firstModBits - scalingModBits < extraBits.value()) {
        emitWarning(getOperation()->getLoc())
            << "Range Analysis indicate that the first modulus must be larger "
               "than the scaling modulus by at least "
            << extraBits.value() << " bits.\n";
      }
    }
    LDBG() << "First modulus finalized as having " << firstModBits << " bits";

    // The data occupies minSlotCount slots regardless of how large the ring
    // has to be, so the layouts' packing width is recorded before any bump
    // below. Widening it would desync the packed layouts from the ciphertexts.
    int64_t requestedSlotCount = slotCount;

    bool cheddarTarget = moduleIsCheddar(getOperation());

    // Lattigo and scale-snu/cheddar use the extended-encryption CKKS
    // parameter path.
    if (moduleIsLattigo(getOperation()) || cheddarTarget) {
      encryptionTechniqueExtended = true;
      LDBG() << "For lattigo/cheddar, fixing extended encryption technique";

      // The supported Lattigo and scale-snu/cheddar bootstrap configurations
      // require LogN >= 14. Since ringDim is twice minSlotCount, enforce the
      // corresponding 8192-slot floor.
      if (hasBootstrap) {
        if (slotCount < 8192) {
          LDBG() << "Bootstrapping detected, bumping minSlotCount from "
                 << slotCount << " to 8192";
          slotCount = 8192;
        }
      }
    }

    int computeMaxLevel = maxLevel.value_or(0);
    bool cheddarBootstrap = cheddarTarget && hasBootstrap;
    int generatedMaxLevel =
        computeMaxLevel + (cheddarBootstrap ? kCheddarBootOverhead : 0);

    auto schemeParam = ckks::SchemeParam::getConcreteSchemeParam(
        firstModBits, scalingModBits, generatedMaxLevel, slotCount,
        usePublicKey, encryptionTechniqueExtended, reducedError,
        cheddarBootstrap ? kCheddarBootArithmeticLevels : 0,
        cheddarBootstrap ? std::max<int>(scalingModBits, kCheddarBootModBits)
                         : 0);

    LDBG() << "Scheme Param:\n" << schemeParam;
    if (auto module = dyn_cast<ModuleOp>(getOperation());
        module && failed(verifyRingDegree(module, schemeParam.getRingDim())))
      return signalPassFailure();

    auto* context = &getContext();
    OpBuilder builder(context);
    if (cheddarBootstrap) {
      getOperation()->setAttr("cheddar.boot.num_cts",
                              builder.getI64IntegerAttr(kCheddarBootNumCts));
      getOperation()->setAttr("cheddar.boot.num_stc",
                              builder.getI64IntegerAttr(kCheddarBootNumStc));
    }
    getOperation()->setAttr(kRequestedSlotCountAttrName,
                            builder.getI64IntegerAttr(requestedSlotCount));
    getOperation()->setAttr(
        kActualSlotCountAttrName,
        builder.getI64IntegerAttr(schemeParam.getRingDim() / 2));

    // annotate ckks::SchemeParamAttr to ModuleOp
    getOperation()->setAttr(
        ckks::CKKSDialect::kSchemeParamAttrName,
        ckks::SchemeParamAttr::get(
            context, log2(schemeParam.getRingDim()),
            DenseI64ArrayAttr::get(context, ArrayRef(schemeParam.getQi())),
            DenseI64ArrayAttr::get(context, ArrayRef(schemeParam.getPi())),
            schemeParam.getLogDefaultScale(),
            usePublicKey ? ckks::CKKSEncryptionType::pk
                         : ckks::CKKSEncryptionType::sk,
            encryptionTechniqueExtended
                ? ckks::CKKSEncryptionTechnique::extended
                : ckks::CKKSEncryptionTechnique::standard));
  }
};

}  // namespace heir
}  // namespace mlir
