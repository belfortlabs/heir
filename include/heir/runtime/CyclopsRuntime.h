#ifndef INCLUDE_HEIR_RUNTIME_CYCLOPSRUNTIME_H_
#define INCLUDE_HEIR_RUNTIME_CYCLOPSRUNTIME_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/string.hpp>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <istream>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/EvkRequest.h"
#include "serialize/Transfer.h"

namespace heir::cyclops {
using Word = std::uint64_t;
using Parameter = ::cyclops::Parameter<Word>;
using Ciphertext = ::cyclops::Ciphertext<Word>;
using Plaintext = ::cyclops::Plaintext<Word>;
using EvaluationKeys = ::cyclops::EvkMap<Word>;
using EvaluationKeyRequest = ::cyclops::EvkRequest;

inline void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename T, typename Fn>
void forEachLeaf(T&& value, Fn&& fn) {
  using V = std::remove_cvref_t<T>;
  if constexpr (requires {
                  value.begin();
                  value.end();
                }) {
    for (auto&& element : value) forEachLeaf(element, fn);
  } else if constexpr (requires { std::tuple_size<V>::value; }) {
    std::apply([&](auto&&... fields) { (forEachLeaf(fields, fn), ...); },
               value);
  } else {
    fn(value);
  }
}

template <typename Aggregate>
void writeValues(const Parameter& parameter, const Aggregate& values,
                 std::ostream& out) {
  cereal::PortableBinaryOutputArchive ar(out);
  forEachLeaf(values, [&](const auto& value) {
    using T = std::remove_cvref_t<decltype(value)>;
    if constexpr (std::is_same_v<T, Ciphertext>) {
      ar(::cyclops::SendCiphertextCereal(parameter, value));
    } else if constexpr (std::is_same_v<T, Plaintext>) {
      ar(::cyclops::SendPlaintextCereal(parameter, value,
                                        parameter.NPToLevel(value.GetNP())));
    } else {
      static_assert(std::is_arithmetic_v<T>, "unsupported input/output leaf");
      ar(value);
    }
  });
}

template <typename Aggregate>
Aggregate readValues(const Parameter& parameter, std::istream& in) {
  cereal::PortableBinaryInputArchive ar(in);
  Aggregate result{};
  forEachLeaf(result, [&](auto& value) {
    using T = std::remove_cvref_t<decltype(value)>;
    if constexpr (std::is_same_v<T, Ciphertext>) {
      std::string bytes;
      ar(bytes);
      auto received = ::cyclops::ReceiveCiphertextCereal(parameter, bytes);
      require(!received.boosted_deflation, "unsupported ciphertext encoding");
      value = std::move(received.ct);
    } else if constexpr (std::is_same_v<T, Plaintext>) {
      std::string bytes;
      ar(bytes);
      auto received = ::cyclops::ReceivePlaintextCereal(parameter, bytes);
      value = std::move(received.value);
    } else {
      static_assert(std::is_arithmetic_v<T>, "unsupported input/output leaf");
      ar(value);
    }
  });
  return result;
}

inline void writeKeys(const Parameter& parameter, const EvaluationKeys& keys,
                      std::ostream& out) {
  cereal::PortableBinaryOutputArchive ar(out);
  ar(static_cast<std::uint32_t>(keys.size()));
  for (const auto& [index, key] : keys) {
    bool sparse = index.idx == EvaluationKeys::kSparseToDenseKeyIndex;
    ar(sparse ? ::cyclops::SendSparseToDenseKeyCereal(parameter, key)
              : ::cyclops::SendEvaluationKeyCereal(parameter, key, index.idx));
  }
}
inline EvaluationKeys readKeys(const Parameter& parameter, std::istream& in) {
  cereal::PortableBinaryInputArchive ar(in);
  std::uint32_t count;
  ar(count);
  EvaluationKeys result;
  for (std::uint32_t i = 0; i < count; ++i) {
    // Decode one key at a time: Cyclops' aggregate API otherwise retains
    // serialized bytes for the entire evaluation-key map.
    std::vector<std::string> bytes(1);
    ar(bytes.front());
    auto key = ::cyclops::ReceiveEvkMapCereal<Word>(parameter, bytes);
    for (auto& [index, value] : key)
      result.insert_or_assign(index, std::move(value));
  }
  return result;
}

// Encrypted intermediate values for debugging. The evaluator passes them to
// the callback for client-side decryption; it does not need the secret key.
// Ciphertext pointers are valid only for the duration of the callback.
struct Checkpoint {
  const char* name;
  const char* metadata;
  std::vector<const Ciphertext*> ciphertexts;
};
using DebugSink = std::function<void(const Checkpoint&)>;

template <typename T>
void emitCheckpoint(const DebugSink* sink, const T& value, const char* name,
                    const char* metadata) {
  if (!sink || !*sink) return;
  Checkpoint checkpoint{name, metadata, {}};
  forEachLeaf(value,
              [&](const auto& ct) { checkpoint.ciphertexts.push_back(&ct); });
  (*sink)(checkpoint);
}
inline void emitCheckpoint(const DebugSink* sink, const Ciphertext* values,
                           std::size_t count, const char* name,
                           const char* metadata) {
  emitCheckpoint(sink, std::span(values, count), name, metadata);
}
}  // namespace heir::cyclops
#endif  // INCLUDE_HEIR_RUNTIME_CYCLOPSRUNTIME_H_
