// The CPU client of the max pool end-to-end test. It generates the keys and
// encrypts the input, waits for the server's result, decrypts it, and compares
// it with a cleartext max pool. See README.md.

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>

#include "exchange.h"
#include "max_pool_client.h"

namespace api = heir::generated::max_pool::client;
using namespace heir::examples::max_pool;

namespace {

// The shape of max_pool.mlir: MaxPool1d(8, ceil_mode=True) over 4 channels of
// 61 values in [-3, 5], so the last window of each channel has 5 values.
constexpr int kChannels = 4;
constexpr int kLength = 61;
constexpr int kWindow = 8;
constexpr int kWindows = (kLength + kWindow - 1) / kWindow;

// Cyclops' MaxPool test accepts an error of 0.01 in its comparison domain
// [-0.5, 0.5]. HEIR sets value_bound = max(|-3|, |5|) = 5, so that error is
// 0.01 * 2 * 5 in the units of the model.
constexpr double kTolerance = 0.1;

// Each window has one winner, at least 0.25 above the other values: the
// separation that Cyclops' test uses (5% of value_bound). The last window of
// each channel holds only negative values, so a padding value above the real
// values gives a wrong maximum.
api::Input0 makeInput() {
  std::mt19937 gen(20261002);
  api::Input0 input{};
  for (int c = 0; c < kChannels; ++c) {
    for (int w = 0; w < kWindows; ++w) {
      int begin = w * kWindow, end = std::min(begin + kWindow, kLength);
      bool negative = w == kWindows - 1;
      std::uniform_real_distribution<float> background(-3.0f,
                                                       negative ? -2.5f : 4.0f);
      std::uniform_real_distribution<float> top(negative ? -2.2f : 4.5f,
                                                negative ? -2.0f : 4.99f);
      int winner = std::uniform_int_distribution<int>(begin, end - 1)(gen);
      for (int i = begin; i < end; ++i)
        input[0][c][i] = i == winner ? top(gen) : background(gen);
    }
  }
  return input;
}

// A ceil-mode max pool: the last window reads only the values that exist.
api::Output0 maxPool(const api::Input0& input) {
  api::Output0 output{};
  for (int c = 0; c < kChannels; ++c) {
    for (int w = 0; w < kWindows; ++w) {
      int begin = w * kWindow, end = std::min(begin + kWindow, kLength);
      output[0][c][w] = *std::max_element(input[0][c].begin() + begin,
                                          input[0][c].begin() + end);
    }
  }
  return output;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <exchange directory>\n";
    return 2;
  }
  std::filesystem::path directory = argv[1];

  auto context = api::Setup();
  auto keys = api::KeyGen(context);
  std::stringstream keyWire;
  heir::cyclops::writeKeys(context->param_, keys.storage->GetEvkMap(), keyWire);
  writeFileAtomically(directory / "keys.bin", keyWire.str());

  const api::Input0 input = makeInput();
  api::CleartextInputs inputs{input};
  auto encrypted = api::Encrypt(*context, keys.secret_key, inputs);
  std::stringstream inputWire;
  heir::cyclops::writeValues(context->param_, encrypted, inputWire);
  writeFileAtomically(directory / "inputs.bin", inputWire.str());
  std::cout << "client: wrote the keys and the encrypted input\n";

  std::stringstream outputWire(waitForFile(directory / "outputs.bin"));
  auto outputs = heir::cyclops::readValues<api::EncryptedOutputs>(
      context->param_, outputWire);
  api::Output0 result = api::Decrypt(*context, keys.secret_key, outputs);

  api::Output0 expected = maxPool(input);
  double worstError = 0.0;
  int failures = 0;
  for (int c = 0; c < kChannels; ++c) {
    for (int w = 0; w < kWindows; ++w) {
      double error = std::abs(result[0][c][w] - expected[0][c][w]);
      worstError = std::max(worstError, error);
      if (!(error <= kTolerance)) {
        ++failures;
        std::cerr << "channel " << c << " window " << w << ": got "
                  << result[0][c][w] << ", expected " << expected[0][c][w]
                  << "\n";
      }
    }
  }
  std::cout << "client: worst error " << worstError << " (tolerance "
            << kTolerance << ")\n";
  if (failures != 0) {
    std::cerr << failures << " of " << kChannels * kWindows
              << " maxima are wrong\n";
    return 1;
  }
  std::cout << "max pool end-to-end test passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "client: " << error.what() << "\n";
  return 1;
}
