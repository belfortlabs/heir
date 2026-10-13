#include <cmath>
#include <iostream>
#include <sstream>

#include "split_client.h"
namespace api = heir::generated::split::client;
int main() {
  auto context = api::Setup();
  auto keys = api::KeyGen(context);
  api::CleartextInputs input{api::Input0{0.125f, -0.25f, 0.5f, 0.75f}};
  auto encrypted = api::Encrypt(*context, keys.secret_key, input);
  std::stringstream ciphertextWire;
  heir::cyclops::writeValues(context->param_, encrypted, ciphertextWire);
  auto output = heir::cyclops::readValues<api::EncryptedOutputs>(
      context->param_, ciphertextWire);
  auto clear = api::Decrypt(*context, keys.secret_key, output);
  for (int i = 0; i < 4; ++i)
    if (!(std::abs(clear[i] - std::get<0>(input)[i]) <= 1e-4)) return 1;
  // A seed reproduces the secret key, whether exported from an unseeded
  // KeyGen or passed to two KeyGens.
  auto decryptsWith = [&](api::KeyPair& encryptKeys,
                          api::KeyPair& decryptKeys) {
    auto restored =
        api::Decrypt(*context, decryptKeys.secret_key,
                     api::Encrypt(*context, encryptKeys.secret_key, input));
    for (int i = 0; i < 4; ++i)
      if (!(std::abs(restored[i] - std::get<0>(input)[i]) <= 1e-4))
        return false;
    return true;
  };
  auto restoredKeys = api::KeyGen(context, keys.storage->GetSecretSeed());
  if (!decryptsWith(keys, restoredKeys)) return 1;
  ::cyclops::prng::Seed seed;
  for (unsigned i = 0; i < seed.bytes.size(); ++i) seed.bytes[i] = i;
  auto firstKeys = api::KeyGen(context, seed);
  auto secondKeys = api::KeyGen(context, seed);
  if (!decryptsWith(firstKeys, secondKeys)) return 1;
  // Without its rotation keys, KeyGen leaves them to the Galois key upload,
  // from which the server derives every planned rotation key.
  auto request = api::GetKeyRequest();
  auto galoisKeys = api::KeyGen(context, std::nullopt, false);
  std::stringstream galoisWire;
  heir::cyclops::writeGaloisKeys(*galoisKeys.storage, context->param_, request,
                                 galoisWire);
  api::EvaluationKeys derived;
  heir::cyclops::readGaloisKeys(context->param_, request, galoisWire, derived);
  // Layouts whose Galois plan would upload more keep their keys in KeyGen.
  for (auto& [index, value] : galoisKeys.storage->MutableEvkMap())
    derived.insert_or_assign(index, std::move(value));
  for (const auto& [key, count] : request.AllRequests())
    if (key.rot_idx != 0)
      (void)derived.GetRotationKey(key.rot_idx, context->NativeSecretId(),
                                   context->param_, key.level, key.key_mode,
                                   key.required_num_aux);
  // Model preprocessing can produce arrays larger than Clang's fold limit.
  std::array<float, 403> values;
  for (unsigned i = 0; i < values.size(); ++i) values[i] = i / 8.0f;
  std::stringstream valuesWire;
  heir::cyclops::writeValues(context->param_, values, valuesWire);
  if (heir::cyclops::readValues<decltype(values)>(context->param_,
                                                  valuesWire) != values)
    return 1;
  std::cout << "CPU client ciphertext serialization and decrypt round "
               "trip passed\n";
}
