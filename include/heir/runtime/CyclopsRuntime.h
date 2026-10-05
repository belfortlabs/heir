#ifndef INCLUDE_HEIR_RUNTIME_CYCLOPSRUNTIME_H_
#define INCLUDE_HEIR_RUNTIME_CYCLOPSRUNTIME_H_

#include <algorithm>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/string.hpp>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <istream>
#include <map>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "core/GaloisKeyExecution.h"
#include "core/GaloisKeyPlan.h"
#include "serialize/Transfer.h"

// Serialization and debugging helpers shared by the generated Cyclops client
// and server. Everything is generic over the runtime's word type: the
// generated code fixes it (`using word = std::uint32_t;` or
// `std::uint64_t;`), and these helpers deduce it from the Parameter they are
// handed.
namespace heir::cyclops {
template <typename Word>
using Parameter = ::cyclops::Parameter<Word>;
template <typename Word>
using Ciphertext = ::cyclops::Ciphertext<Word>;
template <typename Word>
using Plaintext = ::cyclops::Plaintext<Word>;
template <typename Word>
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

template <typename Word, typename Aggregate>
void writeValues(const Parameter<Word>& parameter, const Aggregate& values,
                 std::ostream& out) {
  cereal::PortableBinaryOutputArchive ar(out);
  forEachLeaf(values, [&](const auto& value) {
    using T = std::remove_cvref_t<decltype(value)>;
    if constexpr (std::is_same_v<T, Ciphertext<Word>>) {
      ar(::cyclops::SendCiphertextCereal(parameter, value));
    } else if constexpr (std::is_same_v<T, Plaintext<Word>>) {
      ar(::cyclops::SendPlaintextCereal(parameter, value,
                                        parameter.NPToLevel(value.GetNP())));
    } else {
      static_assert(std::is_arithmetic_v<T>, "unsupported input/output leaf");
      ar(value);
    }
  });
}

template <typename Aggregate, typename Word>
Aggregate readValues(const Parameter<Word>& parameter, std::istream& in) {
  cereal::PortableBinaryInputArchive ar(in);
  Aggregate result{};
  forEachLeaf(result, [&](auto& value) {
    using T = std::remove_cvref_t<decltype(value)>;
    if constexpr (std::is_same_v<T, Ciphertext<Word>>) {
      std::string bytes;
      ar(bytes);
      auto received = ::cyclops::ReceiveCiphertextCereal(parameter, bytes);
      require(!received.boosted_deflation, "unsupported ciphertext encoding");
      value = std::move(received.ct);
    } else if constexpr (std::is_same_v<T, Plaintext<Word>>) {
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

template <typename Word>
void writeKeys(const Parameter<Word>& parameter,
               const EvaluationKeys<Word>& keys, std::ostream& out) {
  cereal::PortableBinaryOutputArchive ar(out);
  ar(static_cast<std::uint32_t>(keys.size()));
  for (const auto& [index, key] : keys) {
    bool sparse = index.idx == EvaluationKeys<Word>::kSparseToDenseKeyIndex;
    ar(sparse ? ::cyclops::SendSparseToDenseKeyCereal(parameter, key)
              : ::cyclops::SendEvaluationKeyCereal(parameter, key, index.idx));
  }
}
template <typename Word>
EvaluationKeys<Word> readKeys(const Parameter<Word>& parameter,
                              std::istream& in) {
  cereal::PortableBinaryInputArchive ar(in);
  std::uint32_t count;
  ar(count);
  EvaluationKeys<Word> result;
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

// Hierarchical Galois keys: instead of every rotation key, the client uploads a
// few seed keys per key-switching layout, and the server derives the rotation
// keys from them. Both sides plan from the same EvaluationKeyRequest, so the
// plans are never sent.

// A request with every key except the rotation keys, for a KeyGen whose
// rotation keys the server derives.
inline EvaluationKeyRequest withoutRotationKeys(
    const EvaluationKeyRequest& request) {
  EvaluationKeyRequest result;
  for (const auto& [key, count] : request.ConjugationRequests())
    for (int i = 0; i < count; ++i)
      result.RequestConjugationKey(key.level, key.key_mode,
                                   key.required_num_aux);
  for (const auto& [key, count] : request.MultiplicationRequests())
    for (int i = 0; i < count; ++i)
      result.RequestMultiplicationKey(key.level, key.key_mode,
                                      key.required_num_aux);
  for (const auto& [key, count] : request.RotatedMultiplicationRequests())
    for (int i = 0; i < count; ++i)
      result.RequestRotatedMultiplicationKey(
          key.rot_idx, key.level, key.key_mode, key.required_num_aux);
  return result;
}

// (num_main, num_ter, num_aux) of a key-switching layout.
using KeySwitchShape = std::tuple<int, int, int>;

// The rotation keys of a request, grouped by the key-switching layout
// UserInterface::PrepareRotationKey would build each at: a request whose
// level resolves to the default key gets that rotation's default key, as wide
// as its widest such request, and every other request its level's exact key.
// PrepareRotationKey also drops an exact key a default or wider key serves as
// well; the derivation keeps it, so the evaluator may find a better fit.
template <typename Word>
std::map<KeySwitchShape, std::vector<int>> galoisKeyGroups(
    const Parameter<Word>& parameter, const EvaluationKeyRequest& request) {
  const ::cyclops::SecretId secret = parameter.NativeSecretId();
  std::map<int, int> defaultMain;
  std::map<KeySwitchShape, std::vector<int>> groups;
  for (const auto& [key, count] : request.AllRequests()) {
    if (key.rot_idx == 0) continue;
    require(key.rot_idx > 0, "galoisKeyGroups: negative rotation index");
    const auto config = parameter.ResolveKeySwitchConfig(
        key.level, key.key_mode, /*is_relin=*/false, secret);
    require(key.required_num_aux < 0 || config.num_aux == key.required_num_aux,
            "galoisKeyGroups: planned key aux count does not match request "
            "constraint");
    if (config.use_default) {
      auto [it, inserted] =
          defaultMain.try_emplace(key.rot_idx, config.num_main);
      if (!inserted) it->second = std::max(it->second, config.num_main);
      continue;
    }
    groups[{config.num_main, config.num_ter, config.num_aux}].push_back(
        key.rot_idx);
  }
  for (const auto& [rotation, numMain] : defaultMain) {
    const auto config =
        parameter.DefaultKeySwitchConfigForMain(numMain, secret);
    groups[{config.num_main, config.num_ter, config.num_aux}].push_back(
        rotation);
  }
  for (auto& [shape, rotations] : groups) {
    std::sort(rotations.begin(), rotations.end());
    rotations.erase(std::unique(rotations.begin(), rotations.end()),
                    rotations.end());
  }
  return groups;
}

// Plans in the parameter's ring only: a Parameter alone has no 2N profile.
template <typename Word>
::cyclops::GaloisKeyPlan<Word> planGaloisKeys(
    const Parameter<Word>& parameter, const KeySwitchShape& shape,
    const std::vector<int>& rotations) {
  const auto [numMain, numTer, numAux] = shape;
  return ::cyclops::PlanGaloisKeys<Word>(
      parameter, nullptr,
      ::cyclops::KeySwitchLayout<Word>(
          parameter, ::cyclops::NPInfo(numMain, numTer, numAux)),
      rotations);
}

// Client: one Galois key upload per layout of the request's rotation keys.
// `client` is the UserInterface of the KeyPair from KeyGen(..., false).
template <typename UserInterface, typename Word>
void writeGaloisKeys(const UserInterface& client,
                     const Parameter<Word>& parameter,
                     const EvaluationKeyRequest& request, std::ostream& out) {
  cereal::PortableBinaryOutputArchive ar(out);
  const auto groups = galoisKeyGroups(parameter, request);
  ar(static_cast<std::uint32_t>(groups.size()));
  for (const auto& [shape, rotations] : groups)
    ar(::cyclops::SendGaloisKeyUpload<Word>(
        ::cyclops::WireSerializer::kCereal,
        client.GenerateGaloisKeyUpload(
            planGaloisKeys(parameter, shape, rotations))));
}

// Server: derive the request's rotation keys from the client's uploads into
// `keys`, where the evaluator looks them up. Each upload is checked against
// the server's own plan before any key is derived from it.
template <typename Word>
void readGaloisKeys(const Parameter<Word>& parameter,
                    const EvaluationKeyRequest& request, std::istream& in,
                    EvaluationKeys<Word>& keys) {
  cereal::PortableBinaryInputArchive ar(in);
  const auto groups = galoisKeyGroups(parameter, request);
  std::uint32_t count;
  ar(count);
  require(count == groups.size(),
          "readGaloisKeys: the upload does not match the planned layouts");
  for (const auto& [shape, rotations] : groups) {
    const auto plan = planGaloisKeys(parameter, shape, rotations);
    std::string bytes;
    ar(bytes);
    auto cold = ::cyclops::DeriveGaloisColdStorage(
        plan, ::cyclops::ReceiveGaloisKeyUpload(plan, bytes));
    ::cyclops::DeriveGaloisHotKeys(plan, std::move(cold), parameter, keys);
  }
}

// Encrypted intermediate values for debugging. The evaluator passes them to
// the callback for client-side decryption; it does not need the secret key.
// Ciphertext pointers are valid only for the duration of the callback.
template <typename Word>
struct Checkpoint {
  const char* name;
  const char* metadata;
  std::vector<const Ciphertext<Word>*> ciphertexts;
};
template <typename Word>
using DebugSink = std::function<void(const Checkpoint<Word>&)>;

template <typename Word, typename T>
void emitCheckpoint(const DebugSink<Word>* sink, const T& value,
                    const char* name, const char* metadata) {
  if (!sink || !*sink) return;
  Checkpoint<Word> checkpoint{name, metadata, {}};
  forEachLeaf(value,
              [&](const auto& ct) { checkpoint.ciphertexts.push_back(&ct); });
  (*sink)(checkpoint);
}
template <typename Word>
void emitCheckpoint(const DebugSink<Word>* sink, const Ciphertext<Word>* values,
                    std::size_t count, const char* name, const char* metadata) {
  emitCheckpoint(sink, std::span(values, count), name, metadata);
}
}  // namespace heir::cyclops
#endif  // INCLUDE_HEIR_RUNTIME_CYCLOPSRUNTIME_H_
