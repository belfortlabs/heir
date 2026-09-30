#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Parameters/Cheddar/ParameterFile.h"
#include "llvm/include/llvm/Support/Error.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {
namespace {

ParameterFile parseOrDie(const std::string& text) {
  llvm::Expected<ParameterFile> file = ParameterFile::parse(text, "test");
  if (!file) {
    ADD_FAILURE() << llvm::toString(file.takeError());
    return ParameterFile();
  }
  return std::move(*file);
}

// CHEDDAR's single-ring bootstrapping parameters: the default encryption
// level (4) is where SlotToCoeff starts, so the bootstrap lands one level
// below it, and CoeffToSlot and EvalMod take the two levels above it.
constexpr const char* kCheddarFile = R"({
  "log_degree": 16,
  "log_default_scale": 35,
  "boot": true,
  "dense_hamming_weight": 32768,
  "sparse_hamming_weight": 32,
  "num_cts_levels": 1,
  "num_stc_levels": 1,
  "terminal_primes": [30539777, 32899073],
  "main_primes": [969146369, 1172439041, 958136321, 1070727169, 1115815937, 1062862849],
  "auxiliary_primes": [964558849, 974258177],
  "default_encryption_level": 4,
  "level_config": [[0, 2], [2, 1], [4, 0], [3, 2], [5, 1], [6, 1], [6, 2]],
  "additional_base": [0, 0]
})";

TEST(CheddarParameterFileTest, ParsesChain) {
  ParameterFile file = parseOrDie(kCheddarFile);
  EXPECT_EQ(file.logDefaultScale, 35);
  EXPECT_EQ(file.wordBits(), 32);
  ASSERT_EQ(file.levels.size(), 7u);
  ASSERT_EQ(file.profiles.size(), 1u);
  EXPECT_EQ(file.profiles[0].logDegree, 16);
  EXPECT_EQ(file.profiles[0].defaultEncryptionLevel, 4);
  EXPECT_EQ(file.profiles[0].hammingWeight, 32768);
  ASSERT_TRUE(file.boot.has_value());
  EXPECT_EQ(file.boot->numCtsLevels, 1);
  EXPECT_EQ(file.boot->numStcLevels, 1);
  EXPECT_EQ(file.boot->numEvalModLevels, 1);
  EXPECT_EQ(file.boot->endLevel, 3);
  EXPECT_EQ(file.boot->sparseHammingWeight, 32);
  EXPECT_EQ(file.boot->logMessageRatio, -1);
}

TEST(CheddarParameterFileTest, SelectsLevels) {
  ParameterFile file = parseOrDie(kCheddarFile);
  const RingProfile& ring = file.profiles[0];
  EXPECT_EQ(file.topLevel(ring, /*bootstraps=*/false), 4);
  EXPECT_EQ(file.topLevel(ring, /*bootstraps=*/true), 3);
  EXPECT_EQ(file.selectProfile(0, 4, false), &ring);
  EXPECT_EQ(file.selectProfile(0, 4, true), nullptr);
}

TEST(CheddarParameterFileTest, RejectsMalformedFiles) {
  auto expectError = [](const std::string& text, const char* fragment) {
    llvm::Expected<ParameterFile> file = ParameterFile::parse(text, "test");
    ASSERT_FALSE(static_cast<bool>(file));
    std::string message = llvm::toString(file.takeError());
    EXPECT_NE(message.find(fragment), std::string::npos) << message;
  };
  std::string text = kCheddarFile;
  expectError(text.replace(text.find("\"num_cts_levels\": 1"), 19,
                           "\"num_cts_levels\": 3"),
              "cannot hold the bootstrap");
  text = kCheddarFile;
  expectError(text.replace(text.find("\"default_encryption_level\": 4"), 29,
                           "\"default_encryption_level\": 9"),
              "default_encryption_level is outside level_config");
  text = kCheddarFile;
  expectError(text.replace(text.find("[6, 2]"), 6, "[5, 0]"),
              "more primes than the level below");
}

}  // namespace
}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
