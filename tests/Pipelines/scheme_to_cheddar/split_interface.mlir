// RUN: heir-opt %s --annotate-module="backend=cheddar scheme=ckks" --mlir-to-ckks="min-slot-count=4096" --scheme-to-cheddar="entry-function=split runtime=cyclops" --cheddar-to-emitc --cheddar-emitc-entry-interface="runtime=cyclops" > %t
// RUN: heir-translate %t --mlir-to-cpp --file-id=client_header | FileCheck %s --check-prefix=CLIENT-H --implicit-check-not=BootContext --implicit-check-not=core/Context.h --implicit-check-not=extension/ --implicit-check-not=cuda
// RUN: heir-translate %t --mlir-to-cpp --file-id=client_source | FileCheck %s --check-prefix=CLIENT-CPP --implicit-check-not=BootContext --implicit-check-not=" Context<word>::Create" --implicit-check-not=__server_setup --implicit-check-not=cuda
// RUN: heir-translate %t --mlir-to-cpp --file-id=server_header | FileCheck %s --check-prefix=SERVER-H --implicit-check-not=UserInterface --implicit-check-not=SecretKey --implicit-check-not=KeyGen
// RUN: heir-translate %t --mlir-to-cpp --file-id=server_source | FileCheck %s --check-prefix=SERVER-CPP --implicit-check-not=UserInterface --implicit-check-not=__keygen --implicit-check-not=__decrypt
// REQUIRES: cyclops-planner

// CLIENT-H: #include <optional>
// CLIENT-H: #include "core/ClientContext.h"
// CLIENT-H: namespace heir::generated::split::client
// CLIENT-H: using Context = ::cyclops::ClientContext<word>;
// CLIENT-H: EvaluationKeyRequest GetKeyRequest()
// CLIENT-H: KeyPair KeyGen(const std::shared_ptr<Context>&, const std::optional<::cyclops::prng::Seed>&, bool);
// CLIENT-H-NEXT: inline KeyPair KeyGen(const std::shared_ptr<Context>& ctx, const std::optional<::cyclops::prng::Seed>& seed) { return KeyGen(ctx, seed, true); }
// CLIENT-H-NEXT: inline KeyPair KeyGen(const std::shared_ptr<Context>& ctx) { return KeyGen(ctx, std::nullopt, true); }
// CLIENT-CPP: #include "split_client.h"
// CLIENT-CPP: ClientContext<word>::Create
// CLIENT-CPP: void split__keygen(const std::shared_ptr<ClientContext<word>>& [[CTX:v[0-9]+]], const std::optional<::cyclops::prng::Seed>& [[SEED:v[0-9]+]], std::unique_ptr<UserInterface<word>>& [[UI:v[0-9]+]]) {
// CLIENT-CPP-NEXT: [[UI]] = std::make_unique<UserInterface<word>>([[CTX]], true, ::cyclops::prng::Backend::kShake128, true, [[SEED]]);
// CLIENT-CPP: namespace heir::generated::split::client
// CLIENT-CPP: EvaluationKeyRequest GetKeyRequest
// CLIENT-CPP: KeyPair KeyGen(const std::shared_ptr<Context>& [[KCTX:v[0-9]+]], const std::optional<::cyclops::prng::Seed>& [[KSEED:v[0-9]+]], bool [[KROT:v[0-9]+]]) {
// CLIENT-CPP: split__keygen([[KCTX]], [[KSEED]], {{v[0-9]+}}.storage);
// CLIENT-CPP: PrepareRotationKey([[KROT]] ? GetKeyRequest() : ::heir::cyclops::withoutRotationKeys(GetKeyRequest()),
// SERVER-H: namespace heir::generated::split::server
// SERVER-CPP: #include "split_server.h"
// SERVER-CPP: Context<word>::Create

func.func @split(%input: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
  %result = arith.mulf %input, %input : tensor<4xf32>
  return %result : tensor<4xf32>
}
