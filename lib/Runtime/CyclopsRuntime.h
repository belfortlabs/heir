#ifndef LIB_RUNTIME_CYCLOPSRUNTIME_H_
#define LIB_RUNTIME_CYCLOPSRUNTIME_H_

#include <array>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/string.hpp>
#include <complex>
#include <cstdint>
#include <functional>
#include <istream>
#include <ostream>
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

inline void writeRequest(const EvaluationKeyRequest& request,
                         std::ostream& out) {
  cereal::PortableBinaryOutputArchive ar(out);
  auto write = [&](const auto& entries) {
    ar(static_cast<std::uint32_t>(entries.size()));
    for (const auto& [key, count] : entries) {
      if constexpr (requires { key.rot_idx; }) ar(key.rot_idx);
      ar(key.level, static_cast<std::int32_t>(key.key_mode),
         key.required_num_aux, count);
    }
  };
  write(request.AllRequests());
  write(request.ConjugationRequests());
  write(request.MultiplicationRequests());
  write(request.RotatedMultiplicationRequests());
}
inline EvaluationKeyRequest readRequest(std::istream& in) {
  cereal::PortableBinaryInputArchive ar(in);
  EvaluationKeyRequest result;
  for (int kind = 0; kind < 4; ++kind) {
    std::uint32_t size;
    ar(size);
    for (std::uint32_t i = 0; i < size; ++i) {
      std::int32_t rotation = 0, level, mode, aux, count;
      if (kind == 0 || kind == 3) ar(rotation);
      ar(level, mode, aux, count);
      require(mode >= 0 && mode <= 2 && level >= 0 && aux >= -1 && count > 0,
              "invalid evaluation-key request");
      auto keyMode = static_cast<::cyclops::KeyMode>(mode);
      for (int n = 0; n < count; ++n) {
        if (kind == 0) result.AddRequest(rotation, level, keyMode, aux);
        if (kind == 1) result.RequestConjugationKey(level, keyMode, aux);
        if (kind == 2) result.RequestMultiplicationKey(level, keyMode, aux);
        if (kind == 3)
          result.RequestRotatedMultiplicationKey(rotation, level, keyMode, aux);
      }
    }
  }
  return result;
}

// Gather only prepared transforms. Other prepared fields are plaintexts.
template <typename T>
void addPreparedRequests(const T& value, EvaluationKeyRequest& request) {
  forEachLeaf(value, [&](const auto& field) {
    if constexpr (requires { field->AddRequiredRotations(request); }) {
      require(bool(field), "uninitialized prepared transform");
      field->AddRequiredRotations(request);
    }
  });
}

// Ciphertexts are borrowed for the duration of the callback. The consumer
// chooses whether to serialize them; no secret key is needed by the evaluator.
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
}  // namespace heir::cyclops
#endif  // LIB_RUNTIME_CYCLOPSRUNTIME_H_
