#include <cmath>
#include <iostream>
#include <sstream>

#include "split_client.h"
namespace api = heir::generated::split::client;
int main() {
  auto context = api::Setup();
  auto keys = api::KeyGen(context, {});
  api::CleartextInputs input{api::Input0{0.125f, -0.25f, 0.5f, 0.75f}};
  auto encrypted = api::Encrypt(*context, keys.secret_key, input);
  std::stringstream ciphertextWire;
  heir::cyclops::writeValues(context->param_, encrypted, ciphertextWire);
  auto output = heir::cyclops::readValues<api::EncryptedOutputs>(
      context->param_, ciphertextWire);
  auto clear = api::Decrypt(*context, keys.secret_key, output);
  for (int i = 0; i < 4; ++i)
    if (!(std::abs(clear[i] - std::get<0>(input)[i]) <= 1e-4)) return 1;
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
