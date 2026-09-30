#include "lib/Parameters/Cheddar/ParameterFile.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/ArrayRef.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/Twine.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Error.h"         // from @llvm-project
#include "llvm/include/llvm/Support/JSON.h"          // from @llvm-project
#include "llvm/include/llvm/Support/MemoryBuffer.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

namespace {

// The runtimes' level ordering: a level is above another when it has more
// primes in total, or the same total with more main primes (CompareNPPair).
bool isAbove(const LevelLayout& upper, const LevelLayout& lower) {
  int upperTotal = upper.numMain + upper.numTerminal;
  int lowerTotal = lower.numMain + lower.numTerminal;
  return lowerTotal < upperTotal ||
         (lowerTotal == upperTotal && lower.numMain < upper.numMain);
}

// Drops `//` comments outside string literals, which generated parameter
// files may start with.
std::string stripLineComments(llvm::StringRef text) {
  std::string result;
  result.reserve(text.size());
  bool inString = false;
  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    if (inString) {
      result += c;
      if (c == '\\' && i + 1 < text.size()) result += text[++i];
      if (c == '"') inString = false;
      continue;
    }
    if (c == '"') {
      inString = true;
      result += c;
      continue;
    }
    if (c == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      while (i < text.size() && text[i] != '\n') ++i;
      if (i < text.size()) result += '\n';
      continue;
    }
    result += c;
  }
  return result;
}

llvm::Error error(const llvm::Twine& message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

// Typed field access with errors that name the file and the field.
class Reader {
 public:
  explicit Reader(llvm::StringRef name) : name(name) {}

  llvm::Error fail(const llvm::Twine& message) const {
    return error(name + ": " + message);
  }

  llvm::Expected<const llvm::json::Object*> object(
      const llvm::json::Object& in, llvm::StringRef key,
      llvm::StringRef owner) const {
    const llvm::json::Object* value = in.getObject(key);
    if (!value) return fail(owner + " " + key + " must be an object");
    return value;
  }

  llvm::Expected<const llvm::json::Array*> array(const llvm::json::Object& in,
                                                 llvm::StringRef key,
                                                 llvm::StringRef owner) const {
    const llvm::json::Array* value = in.getArray(key);
    if (!value) return fail(owner + " " + key + " must be an array");
    return value;
  }

  llvm::Expected<int> integer(const llvm::json::Object& in, llvm::StringRef key,
                              llvm::StringRef owner) const {
    std::optional<int64_t> value = in.getInteger(key);
    if (!value) return fail(owner + " " + key + " must be an integer");
    if (*value < INT32_MIN || *value > INT32_MAX)
      return fail(owner + " " + key + " is out of range");
    return static_cast<int>(*value);
  }

  llvm::Expected<double> number(const llvm::json::Object& in,
                                llvm::StringRef key,
                                llvm::StringRef owner) const {
    std::optional<double> value = in.getNumber(key);
    if (!value) return fail(owner + " " + key + " must be a number");
    return *value;
  }

  llvm::Error primes(const llvm::json::Object& in, llvm::StringRef key,
                     llvm::StringRef owner, std::vector<uint64_t>& out) const {
    auto values = array(in, key, owner);
    if (!values) return values.takeError();
    for (const llvm::json::Value& value : **values) {
      std::optional<int64_t> prime = value.getAsInteger();
      if (!prime || *prime <= 0)
        return fail(owner + " " + key + " must hold positive integers");
      out.push_back(static_cast<uint64_t>(*prime));
    }
    return llvm::Error::success();
  }

  // A `[num_main, num_terminal]` pair.
  llvm::Expected<LevelLayout> pair(const llvm::json::Value& value,
                                   llvm::StringRef key,
                                   llvm::StringRef owner) const {
    const llvm::json::Array* pair = value.getAsArray();
    if (!pair || pair->size() != 2 || !(*pair)[0].getAsInteger() ||
        !(*pair)[1].getAsInteger())
      return fail(owner + " " + key +
                  " must hold [num_main, num_terminal] pairs");
    return LevelLayout{static_cast<int>(*(*pair)[0].getAsInteger()),
                       static_cast<int>(*(*pair)[1].getAsInteger())};
  }

  llvm::Error levels(const llvm::json::Object& in, llvm::StringRef key,
                     llvm::StringRef owner,
                     std::vector<LevelLayout>& out) const {
    auto values = array(in, key, owner);
    if (!values) return values.takeError();
    for (const llvm::json::Value& value : **values) {
      auto level = pair(value, key, owner);
      if (!level) return level.takeError();
      out.push_back(*level);
    }
    return llvm::Error::success();
  }

 private:
  llvm::StringRef name;
};

llvm::Error checkLevels(const Reader& reader,
                        llvm::ArrayRef<LevelLayout> levels,
                        const LevelLayout& below, int numMain, int numTerminal,
                        llvm::StringRef owner) {
  LevelLayout previous = below;
  for (auto [index, level] : llvm::enumerate(levels)) {
    if (level.numMain < 0 || level.numMain > numMain)
      return reader.fail(owner + " level " + llvm::Twine(index) + " uses " +
                         llvm::Twine(level.numMain) + " main primes, but " +
                         llvm::Twine(numMain) + " are given");
    if (level.numTerminal < 0 || level.numTerminal > numTerminal)
      return reader.fail(owner + " level " + llvm::Twine(index) + " uses " +
                         llvm::Twine(level.numTerminal) +
                         " terminal primes, but " + llvm::Twine(numTerminal) +
                         " are given");
    if (!isAbove(level, previous))
      return reader.fail(owner + " level " + llvm::Twine(index) +
                         " does not have more primes than the level below");
    previous = level;
  }
  return llvm::Error::success();
}

}  // namespace

llvm::Expected<ParameterFile> ParameterFile::load(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return error("cannot read CHEDDAR parameter file " + path + ": " +
                 buffer.getError().message());
  return parse((*buffer)->getBuffer(), path);
}

namespace {

// The prime pools and the level layout of the chain.
llvm::Error parseChain(const Reader& reader, const llvm::json::Object& chain,
                       llvm::StringRef owner, ParameterFile& file) {
  auto logScale = reader.integer(chain, "log_default_scale", owner);
  if (!logScale) return logScale.takeError();
  if (*logScale < 0 || *logScale >= 64)
    return reader.fail(owner + " log_default_scale must be in [0, 63]");
  file.logDefaultScale = *logScale;

  if (llvm::Error e =
          reader.primes(chain, "main_primes", owner, file.mainPrimes))
    return e;
  if (chain.get("terminal_primes"))
    if (llvm::Error e =
            reader.primes(chain, "terminal_primes", owner, file.terminalPrimes))
      return e;
  if (llvm::Error e =
          reader.primes(chain, "auxiliary_primes", owner, file.auxPrimes))
    return e;
  if (file.mainPrimes.empty() || file.auxPrimes.empty())
    return reader.fail(owner + " needs main and auxiliary primes");

  if (llvm::Error e = reader.levels(chain, "level_config", owner, file.levels))
    return e;
  if (file.levels.empty())
    return reader.fail(owner + " level_config must not be empty");
  if (llvm::Error e = checkLevels(reader, file.levels, LevelLayout{},
                                  file.mainPrimes.size(),
                                  file.terminalPrimes.size(), owner))
    return e;

  if (const llvm::json::Value* value = chain.get("additional_base")) {
    auto base = reader.pair(*value, "additional_base", owner);
    if (!base) return base.takeError();
    file.additionalBase = *base;
    const LevelLayout& bottom = file.levels.front();
    if (base->numMain < 0 || base->numTerminal < 0 ||
        base->numMain > bottom.numMain ||
        base->numTerminal > bottom.numTerminal)
      return reader.fail(owner + " additional_base exceeds level 0");
  }
  return llvm::Error::success();
}

// CHEDDAR's bootstrapping parameters: one ring, whose default encryption
// level is where SlotToCoeff starts (BootParameter::GetStCStartLevel), with
// CoeffToSlot and EvalMod on the levels above it.
llvm::Expected<ParameterFile> parseCheddar(const Reader& reader,
                                           const llvm::json::Object& root) {
  ParameterFile file;
  if (llvm::Error e = parseChain(reader, root, "parameter file", file))
    return std::move(e);
  RingProfile ring;
  for (auto [key, field] :
       {std::pair{"log_degree", &ring.logDegree},
        std::pair{"default_encryption_level", &ring.defaultEncryptionLevel},
        std::pair{"dense_hamming_weight", &ring.hammingWeight}}) {
    auto value = reader.integer(root, key, "parameter file");
    if (!value) return value.takeError();
    *field = *value;
  }
  int maxLevel = static_cast<int>(file.levels.size()) - 1;
  if (ring.logDegree <= 0 || ring.hammingWeight <= 0)
    return reader.fail(
        "parameter file log_degree and dense_hamming_weight must be positive");
  if (ring.defaultEncryptionLevel < 0 || ring.defaultEncryptionLevel > maxLevel)
    return reader.fail(
        "parameter file default_encryption_level is outside level_config");
  file.profiles.push_back(ring);

  if (!root.getBoolean("boot").value_or(false)) return file;
  BootstrapConfig boot;
  for (auto [key, field] :
       {std::pair{"num_cts_levels", &boot.numCtsLevels},
        std::pair{"num_stc_levels", &boot.numStcLevels},
        std::pair{"sparse_hamming_weight", &boot.sparseHammingWeight}}) {
    auto value = reader.integer(root, key, "parameter file");
    if (!value) return value.takeError();
    if (*value < 0)
      return reader.fail(llvm::Twine("parameter file ") + key +
                         " must be non-negative");
    *field = *value;
  }
  boot.numEvalModLevels =
      maxLevel - ring.defaultEncryptionLevel - boot.numCtsLevels;
  boot.endLevel = ring.defaultEncryptionLevel - boot.numStcLevels;
  if (boot.numEvalModLevels < 0 || boot.endLevel < 0)
    return reader.fail(
        "parameter file level_config cannot hold the bootstrap: " +
        llvm::Twine(boot.numCtsLevels) + " CtS levels above and " +
        llvm::Twine(boot.numStcLevels) +
        " StC levels below default_encryption_level " +
        llvm::Twine(ring.defaultEncryptionLevel) + " of " +
        llvm::Twine(maxLevel));
  if (boot.sparseHammingWeight > ring.hammingWeight)
    return reader.fail(
        "parameter file sparse_hamming_weight exceeds dense_hamming_weight");
  file.boot = boot;
  return file;
}

}  // namespace

llvm::Expected<ParameterFile> ParameterFile::parse(llvm::StringRef json,
                                                   llvm::StringRef name) {
  Reader reader(name);
  llvm::Expected<llvm::json::Value> document =
      llvm::json::parse(stripLineComments(json));
  if (!document)
    return error(name + ": " + llvm::toString(document.takeError()));
  const llvm::json::Object* root = document->getAsObject();
  if (!root) return reader.fail("parameter file must be a JSON object");
  return parseCheddar(reader, *root);
}

int ParameterFile::wordBits() const {
  uint64_t maxPrime = 0;
  for (const std::vector<uint64_t>* primes :
       {&mainPrimes, &terminalPrimes, &auxPrimes})
    for (uint64_t prime : *primes) maxPrime = std::max(maxPrime, prime);
  return maxPrime < (uint64_t{1} << 31) ? 32 : 64;
}

bool ParameterFile::hasBootstrap(const RingProfile& profile) const {
  return boot.has_value() && profile.logDegree == profiles.back().logDegree;
}

int ParameterFile::topLevel(const RingProfile& profile, bool bootstraps) const {
  return bootstraps ? boot->endLevel : profile.defaultEncryptionLevel;
}

const RingProfile* ParameterFile::selectProfile(int minLogDegree,
                                                int requiredLevels,
                                                bool needsBootstrap) const {
  for (const RingProfile& profile : profiles) {
    if (profile.logDegree < minLogDegree) continue;
    if (needsBootstrap && !hasBootstrap(profile)) continue;
    if (topLevel(profile, needsBootstrap) < requiredLevels) continue;
    return &profile;
  }
  return nullptr;
}

double ParameterFile::log2Modulus(const LevelLayout& layout) const {
  double result = 0.0;
  for (int i = 0; i < layout.numMain; ++i)
    result += std::log2(static_cast<double>(mainPrimes[i]));
  for (int i = 0; i < layout.numTerminal; ++i)
    result += std::log2(static_cast<double>(terminalPrimes[i]));
  return result;
}

double ParameterFile::log2ModulusGrowth(llvm::ArrayRef<LevelLayout> layout,
                                        int level) const {
  if (level == 0) return log2Modulus(layout[0]);
  const LevelLayout& upper = layout[level];
  const LevelLayout& lower = layout[level - 1];
  auto log2Range = [](llvm::ArrayRef<uint64_t> primes, int begin, int count) {
    double result = 0.0;
    for (int i = 0; i < count; ++i)
      result += std::log2(static_cast<double>(primes[begin + i]));
    return result;
  };
  double result = 0.0;
  int terminalDiff = upper.numTerminal - lower.numTerminal;
  if (terminalDiff >= 0)
    result += log2Range(terminalPrimes, lower.numTerminal, terminalDiff);
  else
    result -= log2Range(terminalPrimes, upper.numTerminal, -terminalDiff);
  int mainDiff = upper.numMain - lower.numMain;
  if (mainDiff >= 0)
    result += log2Range(mainPrimes, lower.numMain, mainDiff);
  else
    result -= log2Range(mainPrimes, upper.numMain, -mainDiff);
  return result;
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
