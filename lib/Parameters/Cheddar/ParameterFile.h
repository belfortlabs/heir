#ifndef LIB_PARAMETERS_CHEDDAR_PARAMETERFILE_H_
#define LIB_PARAMETERS_CHEDDAR_PARAMETERFILE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "llvm/include/llvm/ADT/ArrayRef.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Error.h"  // from @llvm-project

// A CHEDDAR parameter file: the modulus chain a CHEDDAR runtime is built
// from, in the format of CHEDDAR's bootstrapping parameters
// (`parameters/*.json` in scale-snu/cheddar-fhe, read by its unit-test
// Testbed).
//
// The modulus chain is a list of levels, each built from a count of main
// primes and a count of terminal primes, so consecutive levels need not be
// prefixes of each other; a 32-bit word chain spends two or three primes per
// level. HEIR keeps its own one-modulus-per-level model, so this mirror only
// has to answer what that model and the backend need: which ring a program
// fits, the level layout of that ring, and the log2 growth of the modulus
// between levels.

namespace mlir {
namespace heir {
namespace cheddar {

// The primes a level is built from: the first `numMain` main primes and the
// first `numTerminal` terminal primes.
struct LevelLayout {
  int numMain = 0;
  int numTerminal = 0;
  bool operator==(const LevelLayout&) const = default;
};

// How the bootstrap circuit uses the levels at the top of the largest ring's
// chain: CoeffToSlot from the top, then EvalMod, then SlotToCoeff, landing at
// `endLevel`.
struct BootstrapConfig {
  int numCtsLevels = 0;
  int numStcLevels = 0;
  int numEvalModLevels = 0;
  // The level bootstrap outputs sit at, and so the top of the levels a
  // bootstrapping program computes on.
  int endLevel = 0;
  int sparseHammingWeight = 0;
  // log2(q0 / scale) the EvalMod approximation is set up for; -1 when the
  // file does not say.
  int logMessageRatio = -1;
};

// One ring the chain can be used in.
struct RingProfile {
  int logDegree = 0;
  // The runtime Parameter's default encryption level: the highest level a
  // program that does not bootstrap may use. For the bootstrapping ring,
  // CHEDDAR's files put it at the start of SlotToCoeff.
  int defaultEncryptionLevel = 0;
  int hammingWeight = 0;
};

class ParameterFile {
 public:
  // Parses the JSON text of a parameter file. `//` line comments are
  // allowed.
  static llvm::Expected<ParameterFile> parse(llvm::StringRef json,
                                             llvm::StringRef name = "<json>");
  static llvm::Expected<ParameterFile> load(llvm::StringRef path);

  int logDefaultScale = 0;
  std::vector<uint64_t> mainPrimes;
  std::vector<uint64_t> terminalPrimes;
  std::vector<uint64_t> auxPrimes;
  // The largest ring's whole chain, level 0 first, including the levels the
  // bootstrap circuit runs on; smaller rings use a prefix of it.
  std::vector<LevelLayout> levels;
  LevelLayout additionalBase;
  // Only the largest ring bootstraps.
  std::optional<BootstrapConfig> boot;
  // Ascending log degrees.
  std::vector<RingProfile> profiles;

  // The narrowest word the primes fit in: 32 when every prime is below 2^31,
  // otherwise 64 (the runtimes keep one spare bit).
  int wordBits() const;

  // Whether `profile` carries the bootstrap chain: only the largest ring does.
  bool hasBootstrap(const RingProfile& profile) const;

  // The highest level a program may compute on in `profile`: the bootstrap's
  // end level for a bootstrapping program, the default encryption level
  // otherwise.
  int topLevel(const RingProfile& profile, bool bootstraps) const;

  // The smallest ring with at least `minLogDegree`, a top level of at least
  // `requiredLevels` and, when `needsBootstrap`, the bootstrap chain; null
  // when no profile fits.
  const RingProfile* selectProfile(int minLogDegree, int requiredLevels,
                                   bool needsBootstrap) const;

  // log2 of the product of the primes a level is built from.
  double log2Modulus(const LevelLayout& layout) const;

  // log2 of the ratio between the moduli of `level` and `level - 1` (what
  // the runtime divides by when it rescales from `level`), computed from the
  // primes each level adds and drops. Level 0 has no lower level: its whole
  // modulus is returned.
  double log2ModulusGrowth(llvm::ArrayRef<LevelLayout> layout, int level) const;
};

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_PARAMETERS_CHEDDAR_PARAMETERFILE_H_
