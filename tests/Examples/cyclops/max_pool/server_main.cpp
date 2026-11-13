// The GPU server of the max pool end-to-end test. It prepares the context,
// waits for the client's keys and encrypted input, evaluates the max pool, and
// writes the encrypted result. It holds no secret key. See README.md.

#include <cuda_runtime.h>

#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>

#include "exchange.h"
#include "max_pool_server.h"

namespace api = heir::generated::max_pool::server;
using namespace heir::examples::max_pool;

int main(int argc, char** argv) try {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <exchange directory>\n";
    return 2;
  }
  std::filesystem::path directory = argv[1];

  // The context and the preprocessing need no client data, so they run while
  // the client generates its keys.
  auto context = api::Setup();
  // The model has no external weights, so no resource file is read.
  auto prepared = api::Preprocess(*context, directory.string());

  std::stringstream keyWire(waitForFile(directory / "keys.bin"));
  auto keys = heir::cyclops::readKeys(context->param_, keyWire);
  std::stringstream inputWire(waitForFile(directory / "inputs.bin"));
  auto inputs = heir::cyclops::readValues<api::EncryptedInputs>(context->param_,
                                                                inputWire);

  auto outputs = api::Evaluate(*context, &keys, prepared, inputs, nullptr);
  cudaDeviceSynchronize();

  std::stringstream outputWire;
  heir::cyclops::writeValues(context->param_, outputs, outputWire);
  writeFileAtomically(directory / "outputs.bin", outputWire.str());
  std::cout << "server: wrote the encrypted result\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "server: " << error.what() << "\n";
  return 1;
}
