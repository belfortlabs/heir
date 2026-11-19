#ifndef TESTS_EXAMPLES_CYCLOPS_MAX_POOL_EXCHANGE_H_
#define TESTS_EXAMPLES_CYCLOPS_MAX_POOL_EXCHANGE_H_

// The client and the server of the max pool end-to-end test run as separate
// processes, because the CPU and GPU runtimes define the same Cyclops symbols.
// They exchange keys, inputs and outputs as files in a shared directory.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>

namespace heir::examples::max_pool {

// Writes `bytes` to a temporary file and renames it, so that the other
// process never reads a partial file.
inline void writeFileAtomically(const std::filesystem::path& path,
                                const std::string& bytes) {
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("cannot write " + temporary.string());
  }
  std::filesystem::rename(temporary, path);
}

// The wait limit in seconds, from MAX_POOL_E2E_TIMEOUT_S (default one hour).
// The server prepares the bootstrap context, which can take minutes.
inline std::chrono::seconds exchangeTimeout() {
  const char* value = std::getenv("MAX_POOL_E2E_TIMEOUT_S");
  return std::chrono::seconds(value ? std::atoi(value) : 3600);
}

// Waits until `path` exists, then returns its contents.
inline std::string waitForFile(const std::filesystem::path& path) {
  auto deadline = std::chrono::steady_clock::now() + exchangeTimeout();
  while (!std::filesystem::exists(path)) {
    if (std::chrono::steady_clock::now() > deadline)
      throw std::runtime_error("timed out waiting for " + path.string());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

}  // namespace heir::examples::max_pool

#endif  // TESTS_EXAMPLES_CYCLOPS_MAX_POOL_EXCHANGE_H_
