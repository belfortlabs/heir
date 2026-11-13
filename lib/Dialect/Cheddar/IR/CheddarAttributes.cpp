#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

LogicalResult EvalModAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, StringAttr type,
    std::optional<int64_t> degree, std::optional<int64_t> interval,
    std::optional<int64_t> logIntervalReduction,
    std::optional<int64_t> invDegree, StringAttr invType,
    FloatAttr invInterval) {
  if (type && !llvm::is_contained(kEvalModTypes, type.getValue()))
    return emitError() << "unknown eval_mod type '" << type.getValue()
                       << "'; expected one of cos_hk, cos_hk_even, "
                          "cos_cheby, sin_cheby, exp_complex";
  if (degree && *degree <= 0)
    return emitError() << "eval_mod degree must be positive";
  if (interval && *interval <= 0)
    return emitError() << "eval_mod interval must be positive";
  if (logIntervalReduction && *logIntervalReduction < 0)
    return emitError() << "eval_mod logIntervalReduction must be non-negative";
  if (invDegree && (*invDegree < 0 || (*invDegree > 0 && *invDegree % 2 == 0)))
    return emitError() << "eval_mod invDegree must be zero or odd";
  if (invType && !llvm::is_contained(kEvalModInvTypes, invType.getValue()))
    return emitError() << "unknown eval_mod invType '" << invType.getValue()
                       << "'; expected taylor or cheby";
  if (invInterval && invInterval.getValueAsDouble() < 0.0)
    return emitError() << "eval_mod invInterval must be non-negative";
  return success();
}

LogicalResult BootstrapConfigAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, int64_t numCtsLevels,
    int64_t numStcLevels, int64_t numEvalModLevels,
    std::optional<int64_t> logMessageRatio, EvalModAttr evalMod) {
  if (numCtsLevels < 0 || numStcLevels < 0 || numEvalModLevels < 0)
    return emitError() << "bootstrap level counts must be non-negative";
  if (logMessageRatio && *logMessageRatio <= 0)
    return emitError() << "bootstrap logMessageRatio must be positive";
  return success();
}

LogicalResult ParameterSetAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, int64_t logN,
    int64_t logScale, DenseI64ArrayAttr mainPrimes, DenseI64ArrayAttr auxPrimes,
    DenseI64ArrayAttr terminalPrimes, DenseI64ArrayAttr levelConfig,
    std::optional<int64_t> wordBits,
    std::optional<int64_t> defaultEncryptionLevel,
    DenseI64ArrayAttr additionalBase, std::optional<int64_t> defaultNumAux,
    BoolAttr levelSpecificKs, std::optional<int64_t> maxKeySwitchAux,
    FloatAttr maxLogPq, std::optional<int64_t> denseHammingWeight,
    std::optional<int64_t> sparseHammingWeight) {
  if (logN <= 0) return emitError() << "logN must be positive";
  if (logScale < 0 || logScale >= 64)
    return emitError() << "logScale must be in [0, 63]";
  if (mainPrimes.empty()) return emitError() << "needs at least one main prime";
  if (wordBits && *wordBits != 32 && *wordBits != 64)
    return emitError() << "wordBits must be 32 or 64";
  int64_t bits = wordBits.value_or(64);
  for (DenseI64ArrayAttr primes : {mainPrimes, auxPrimes, terminalPrimes}) {
    if (!primes) continue;
    for (int64_t prime : primes.asArrayRef()) {
      // The runtime keeps one spare bit below the word width.
      if (prime <= 0 || (bits == 32 && prime >= (int64_t{1} << 31)))
        return emitError() << "prime " << prime << " does not fit "
                           << (bits - 1) << " bits";
    }
  }
  int64_t numTerminal = terminalPrimes ? terminalPrimes.size() : 0;
  int64_t numLevels = mainPrimes.size();
  if (levelConfig) {
    if (levelConfig.empty() || levelConfig.size() % 2 != 0)
      return emitError()
             << "levelConfig must hold (numMain, numTerminal) pairs";
    numLevels = levelConfig.size() / 2;
    std::pair<int64_t, int64_t> previous{0, 0};
    for (int64_t level = 0; level < numLevels; ++level) {
      std::pair<int64_t, int64_t> layout{levelConfig[2 * level],
                                         levelConfig[2 * level + 1]};
      if (layout.first < 0 || layout.first > mainPrimes.size() ||
          layout.second < 0 || layout.second > numTerminal)
        return emitError() << "level " << level << " uses " << layout.first
                           << " main and " << layout.second
                           << " terminal primes, but " << mainPrimes.size()
                           << " and " << numTerminal << " are given";
      int64_t total = layout.first + layout.second;
      int64_t previousTotal = previous.first + previous.second;
      if (total < previousTotal ||
          (total == previousTotal && layout.first <= previous.first))
        return emitError() << "level " << level
                           << " does not have more primes than the level "
                              "below";
      previous = layout;
    }
  } else if (numTerminal != 0) {
    return emitError() << "terminal primes need a levelConfig that places them";
  }
  if (defaultEncryptionLevel &&
      (*defaultEncryptionLevel < 0 || *defaultEncryptionLevel >= numLevels))
    return emitError() << "defaultEncryptionLevel " << *defaultEncryptionLevel
                       << " is outside the " << numLevels << "-level chain";
  if (additionalBase) {
    if (additionalBase.size() != 2)
      return emitError() << "additionalBase must be a (numMain, numTerminal) "
                            "pair";
    if (additionalBase[0] < 0 || additionalBase[1] < 0)
      return emitError() << "additionalBase counts must be non-negative";
  }
  if (defaultNumAux &&
      (*defaultNumAux < 1 || *defaultNumAux > auxPrimes.size()))
    return emitError() << "defaultNumAux must be in [1, number of aux primes]";
  if (maxKeySwitchAux && *maxKeySwitchAux < 1)
    return emitError() << "maxKeySwitchAux must be positive";
  if (maxLogPq && maxLogPq.getValueAsDouble() <= 0.0)
    return emitError() << "maxLogPq must be positive";
  if (denseHammingWeight && *denseHammingWeight <= 0)
    return emitError() << "denseHammingWeight must be positive";
  if (sparseHammingWeight && *sparseHammingWeight <= 0)
    return emitError() << "sparseHammingWeight must be positive";
  if (denseHammingWeight && sparseHammingWeight &&
      *sparseHammingWeight > *denseHammingWeight)
    return emitError() << "sparseHammingWeight exceeds denseHammingWeight";
  return success();
}

int64_t ParameterSetAttr::getWordBitsOrDefault() const {
  return getWordBits().value_or(64);
}

llvm::SmallVector<std::pair<int64_t, int64_t>> ParameterSetAttr::getLevelPairs()
    const {
  llvm::SmallVector<std::pair<int64_t, int64_t>> pairs;
  if (DenseI64ArrayAttr config = getLevelConfig()) {
    for (int64_t i = 0; i + 1 < config.size(); i += 2)
      pairs.emplace_back(config[i], config[i + 1]);
    return pairs;
  }
  for (int64_t level = 0; level < getMainPrimes().size(); ++level)
    pairs.emplace_back(level + 1, 0);
  return pairs;
}

int64_t ParameterSetAttr::getMaxLevel() const {
  if (DenseI64ArrayAttr config = getLevelConfig()) return config.size() / 2 - 1;
  return getMainPrimes().size() - 1;
}

int64_t ParameterSetAttr::getDefaultEncryptionLevelOrDefault() const {
  return getDefaultEncryptionLevel().value_or(getMaxLevel());
}

std::pair<int64_t, int64_t> ParameterSetAttr::getAdditionalBasePair() const {
  if (DenseI64ArrayAttr base = getAdditionalBase()) return {base[0], base[1]};
  return {0, 0};
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
