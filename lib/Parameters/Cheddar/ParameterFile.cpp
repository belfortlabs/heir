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

// The ring degrees Cyclops' NTT supports (NTTHandler::min/max_log_degree_).
constexpr int kCyclopsMinLogDegree = 12;
constexpr int kCyclopsMaxLogDegree = 16;

// The runtimes' level ordering: a level is above another when it has more
// primes in total, or the same total with more main primes (CompareNPPair).
bool isAbove(const LevelLayout& upper, const LevelLayout& lower) {
  int upperTotal = upper.numMain + upper.numTerminal;
  int lowerTotal = lower.numMain + lower.numTerminal;
  return lowerTotal < upperTotal ||
         (lowerTotal == upperTotal && lower.numMain < upper.numMain);
}

// Drops `//` comments outside string literals: the generated files record the
// primegen command in comment lines before the JSON document.
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

llvm::Error parseEvalMod(const Reader& reader, const llvm::json::Object& in,
                         BootstrapConfig& boot) {
  EvalModConfig& out = boot.evalMod;
  out.present = true;
  llvm::StringRef owner = "boot eval_mod";
  auto parseInt = [&](llvm::StringRef key, int& field,
                      int minimum) -> llvm::Error {
    if (!in.get(key)) return llvm::Error::success();
    std::optional<int64_t> value = in.getInteger(key);
    if (!value) return reader.fail(owner + " " + key + " must be an integer");
    if (*value < minimum)
      return reader.fail(owner + " " + key + " must be at least " +
                         llvm::Twine(minimum));
    field = static_cast<int>(*value);
    return llvm::Error::success();
  };
  if (in.get("type")) {
    std::optional<llvm::StringRef> type = in.getString("type");
    if (!type ||
        (*type != "cos_hk" && *type != "cos_hk_even" && *type != "cos_cheby" &&
         *type != "sin_cheby" && *type != "exp_complex"))
      return reader.fail(owner +
                         " type must be cos_hk, cos_hk_even, cos_cheby, "
                         "sin_cheby or exp_complex");
    out.type = type->str();
  }
  if (llvm::Error e = parseInt("degree", out.degree, 1)) return e;
  if (llvm::Error e = parseInt("interval", out.interval, 1)) return e;
  if (llvm::Error e =
          parseInt("log_interval_reduction", out.logIntervalReduction, 0))
    return e;
  if (llvm::Error e = parseInt("inv_degree", out.invDegree, 0)) return e;
  if (in.get("inv_type")) {
    std::optional<llvm::StringRef> type = in.getString("inv_type");
    if (!type || (*type != "taylor" && *type != "cheby"))
      return reader.fail(owner + " inv_type must be taylor or cheby");
    out.invType = type->str();
  }
  if (in.get("inv_interval")) {
    std::optional<double> value = in.getNumber("inv_interval");
    if (!value || *value < 0.0)
      return reader.fail(owner + " inv_interval must be a non-negative number");
    out.invInterval = *value;
  }
  if (llvm::Error e = parseInt("log_message_ratio", boot.logMessageRatio, 1))
    return e;
  if (out.invDegree > 0 && out.invDegree % 2 == 0)
    return reader.fail(owner + " inv_degree must be odd");
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

// Pools and chain fields both formats spell the same way.
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

// A Cyclops multi-profile parameter set: a residual chain shared by all rings
// (each ring's default encryption level cuts it), the bootstrap levels above
// it for the largest ring, and Cyclops' key-switching policy.
llvm::Expected<ParameterFile> parseCyclops(const Reader& reader,
                                           const llvm::json::Object& root) {
  auto schema = reader.object(root, "schema", "parameter file");
  if (!schema) return schema.takeError();
  std::optional<llvm::StringRef> schemaName = (*schema)->getString("name");
  if (!schemaName || *schemaName != "cyclops.multi_profile_parameter_set")
    return reader.fail(
        "unsupported schema name; expected "
        "cyclops.multi_profile_parameter_set");
  if ((*schema)->getInteger("major").value_or(0) != 1)
    return reader.fail("unsupported schema major version; expected 1");

  auto chainObject = reader.object(root, "ring_chain", "parameter file");
  if (!chainObject) return chainObject.takeError();
  const llvm::json::Object& chain = **chainObject;
  ParameterFile file;
  if (llvm::Error e = parseChain(reader, chain, "ring_chain", file))
    return std::move(e);
  // The residual chain; the bootstrap levels follow it in `file.levels`.
  const int numResidualLevels = static_cast<int>(file.levels.size());
  // Cyclops' loader enables level-specific key switching unless the file
  // turns it off.
  file.levelSpecificKs = true;
  file.prefixChains = true;

  if (chain.get("default_num_aux")) {
    auto value = reader.integer(chain, "default_num_aux", "ring_chain");
    if (!value) return value.takeError();
    if (*value < 1 || *value > static_cast<int>(file.auxPrimes.size()))
      return reader.fail(
          "ring_chain default_num_aux must be in [1, number of auxiliary "
          "primes]");
    file.defaultNumAux = *value;
  }
  if (chain.get("level_specific_ks")) {
    std::optional<bool> value = chain.getBoolean("level_specific_ks");
    if (!value)
      return reader.fail("ring_chain level_specific_ks must be a boolean");
    file.levelSpecificKs = *value;
  }
  if (chain.get("max_key_switch_aux")) {
    auto value = reader.integer(chain, "max_key_switch_aux", "ring_chain");
    if (!value) return value.takeError();
    if (*value < 1)
      return reader.fail("ring_chain max_key_switch_aux must be positive");
    file.maxKeySwitchAux = *value;
  }

  if (chain.get("boot")) {
    auto bootObject = reader.object(chain, "boot", "ring_chain");
    if (!bootObject) return bootObject.takeError();
    BootstrapConfig boot;
    std::vector<LevelLayout> bootLevels;
    if (llvm::Error e = reader.levels(**bootObject, "level_config",
                                      "ring_chain boot", bootLevels))
      return std::move(e);
    if (llvm::Error e = checkLevels(
            reader, bootLevels, file.levels.back(), file.mainPrimes.size(),
            file.terminalPrimes.size(), "ring_chain boot"))
      return std::move(e);
    for (auto [key, field] :
         {std::pair{"sparse_hamming_weight", &boot.sparseHammingWeight},
          std::pair{"num_cts_levels", &boot.numCtsLevels},
          std::pair{"num_stc_levels", &boot.numStcLevels},
          std::pair{"num_eval_mod_levels", &boot.numEvalModLevels}}) {
      auto value = reader.integer(**bootObject, key, "ring_chain boot");
      if (!value) return value.takeError();
      if (*value < 0)
        return reader.fail(llvm::Twine("ring_chain boot ") + key +
                           " must be non-negative");
      *field = *value;
    }
    if (boot.numCtsLevels + boot.numEvalModLevels + boot.numStcLevels !=
        static_cast<int>(bootLevels.size()))
      return reader.fail(
          "ring_chain boot level_config has " + llvm::Twine(bootLevels.size()) +
          " levels, but num_cts_levels + num_eval_mod_levels + num_stc_levels "
          "is " +
          llvm::Twine(boot.numCtsLevels + boot.numEvalModLevels +
                      boot.numStcLevels));
    if ((*bootObject)->get("eval_mod")) {
      auto evalMod = reader.object(**bootObject, "eval_mod", "ring_chain boot");
      if (!evalMod) return evalMod.takeError();
      if (llvm::Error e = parseEvalMod(reader, **evalMod, boot))
        return std::move(e);
    }
    // The bootstrap lands at the top of the residual chain.
    boot.endLevel = numResidualLevels - 1;
    file.levels.insert(file.levels.end(), bootLevels.begin(), bootLevels.end());
    file.boot = std::move(boot);
  }

  auto profiles = reader.array(chain, "profiles", "ring_chain");
  if (!profiles) return profiles.takeError();
  if ((*profiles)->empty()) return reader.fail("ring_chain profiles is empty");
  for (auto [index, value] : llvm::enumerate(**profiles)) {
    const llvm::json::Object* profile = value.getAsObject();
    if (!profile) return reader.fail("ring_chain profiles must hold objects");
    if (profile->get("ring_type"))
      return reader.fail("profiles describe standard rings only");
    RingProfile ring;
    auto logDegree = reader.integer(*profile, "log_degree", "profile");
    if (!logDegree) return logDegree.takeError();
    ring.logDegree = *logDegree;
    auto level =
        reader.integer(*profile, "default_encryption_level", "profile");
    if (!level) return level.takeError();
    ring.defaultEncryptionLevel = *level;
    auto weight = reader.integer(*profile, "hamming_weight", "profile");
    if (!weight) return weight.takeError();
    ring.hammingWeight = *weight;
    auto budget = reader.number(*profile, "max_log_pq", "profile");
    if (!budget) return budget.takeError();
    ring.maxLogPq = *budget;

    int expectedLogDegree =
        index == 0 ? ring.logDegree : file.profiles.back().logDegree + 1;
    if (ring.logDegree != expectedLogDegree ||
        ring.logDegree < kCyclopsMinLogDegree ||
        ring.logDegree > kCyclopsMaxLogDegree)
      return reader.fail(
          "ring_chain profiles must form a contiguous chain of log degrees "
          "within [12, 16]");
    if (ring.defaultEncryptionLevel < 0 ||
        ring.defaultEncryptionLevel >= numResidualLevels)
      return reader.fail(
          "profile default_encryption_level is outside ring_chain "
          "level_config");
    if (index > 0 && ring.defaultEncryptionLevel <
                         file.profiles.back().defaultEncryptionLevel)
      return reader.fail(
          "profile default_encryption_level values must not decrease with "
          "ring degree");
    if (ring.hammingWeight <= 0 || ring.maxLogPq <= 0.0)
      return reader.fail(
          "profile hamming_weight and max_log_pq must be positive");
    file.profiles.push_back(ring);
  }
  if (file.profiles.back().logDegree != kCyclopsMaxLogDegree)
    return reader.fail("ring_chain profiles must end at log degree 16");
  if (file.profiles.back().defaultEncryptionLevel != numResidualLevels - 1)
    return reader.fail(
        "the largest profile must consume the complete ring_chain "
        "level_config");
  if (file.boot &&
      file.boot->sparseHammingWeight > file.profiles.back().hammingWeight)
    return reader.fail(
        "boot sparse_hamming_weight exceeds the largest ring's hamming_weight");
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
  if (root->get("schema")) return parseCyclops(reader, *root);
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
