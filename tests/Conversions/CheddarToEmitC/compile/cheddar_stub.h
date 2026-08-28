// Header-only stub of the CHEDDAR C++ API, used to *compile* (not run) the
// C++ that `cheddar-to-emitc` + `heir-translate --mlir-to-cpp` produce, with
// no GPU/CUDA toolchain. This is a CI-runnable guard that the emitted code
// honours CHEDDAR's move/const contract -- the kind of bug that FileCheck
// (which only inspects emitted text) cannot catch.
//
// The move/const semantics below mirror the real library (verified against
// CHEDDAR's include/core headers); only these properties matter here, so the
// method bodies are empty and the data layout is omitted:
//
//   * Ciphertext/Plaintext/Constant/EvaluationKey -- move-only (copy deleted)
//     *with* move-assignment, default-constructible.   (core/Container.h)
//   * EvkMap -- move-only, copy deleted, *no* move-assignment. (core/EvkMap.h)
//   * Context::MadUnsafe(Ct& res, ...) mutates `res` in place, so `res` is a
//     non-const reference.                              (core/Context.h:377)
//   * UserInterface::Get*Key() / GetEvkMap() return `const&`. (UserInterface.h)
//
// `HEIR_CYCLOPS_STUB` switches the declarations that differ between the two
// runtimes the emitter targets. Belfort's Cyclops fork, after its client/server
// split, keys every evaluation key by the secret it was built for -- so the
// UserInterface's secret-blind Get*Key() getters are gone and the Context takes
// the whole EvkMap instead -- and moved the encoder's slot path onto real
// vectors as EncodeSlots/DecodeSlots.
//
// Kept deliberately narrow: setup is covered by conversion and opt-in real
// CHEDDAR compile tests. Getter ops remain unsupported by this lowering.

#ifndef TESTS_CONVERSIONS_CHEDDARTOEMITC_COMPILE_CHEDDAR_STUB_H_
#define TESTS_CONVERSIONS_CHEDDARTOEMITC_COMPILE_CHEDDAR_STUB_H_

#include <complex>
#include <initializer_list>
#include <map>
#include <memory>
#include <vector>

namespace cheddar {

using Complex = std::complex<double>;

template <typename word>
class Context;
template <typename word>
struct Ciphertext;
template <typename word>
struct Plaintext;

#ifdef HEIR_CYCLOPS_STUB
// Handle of the secret a container is encrypted under / a key was built for.
// Every Cyclops key lookup takes one.                          (core/Type.h)
struct SecretId {
  int value = -1;
};

// Which key-switching variant a lookup accepts; every Cyclops key getter
// defaults it.                                                 (core/Type.h)
enum class KeyMode { kInherit };
#endif

// Number-of-primes info a ciphertext carries; the EvalPoly emitter maps it back
// to a level via Parameter::NPToLevel.
struct NPInfo {
  int num_main_ = 0;
};

// Minimal parameter stub: the EvalPoly emitter reads the level/q-product via
// NPToLevel / GetRescalePrimeProd (mirroring EvalMod), and cheddar.make_parameter
// builds one from the CKKS modulus chain.
template <typename word>
struct Parameter {
  Parameter(int log_n, double scale, int default_encryption_level,
            const std::vector<std::pair<int, int>>& level_config,
            const std::vector<word>& main_primes,
            const std::vector<word>& aux_primes);
  void SetDenseHammingWeight(int weight);
  void SetSparseHammingWeight(int weight);
  double GetScale(int level) const;
  double GetRescalePrimeProd(int level) const;
  int NPToLevel(NPInfo np) const;
  int max_level_ = 0;
};

// The set of rotations a prepared transform needs keys for; the bootstrap
// precompute fills one and key generation consumes it.    (core/EvkRequest.h)
struct EvkRequest {};

// Whether bootstrapping drops the imaginary part at the end.
enum class BootVariant { kNormal, kImaginaryRemoving };

// Bootstrapping's own parameters, derived from the chain top and the CtS/StC
// level budgets.                                    (extension/BootParameter.h)
struct BootParameter {
  BootParameter(int max_level, int num_cts_levels, int num_stc_levels,
                int log_message_ratio = 5);
};

// The encoder exposes the per-level canonical scale (the EvalPoly emitter reads
// input/target scales via `encoder.GetScale(level)`) and the slot-domain
// message path. Cyclops names its entry points after the domain they read from
// / write to and lets no std::complex cross the API.        (core/Encode.h)
template <typename word>
struct Encoder {
  double GetScale(int level) const;
#ifdef HEIR_CYCLOPS_STUB
  void EncodeSlots(Plaintext<word>& ptxt, int level, double scale,
                   const std::vector<double>& real, int num_aux = 0,
                   int slot_period = 0) const;
  void DecodeSlots(std::vector<double>& real,
                   const Plaintext<word>& ptxt) const;
#else
  void Encode(Plaintext<word>& ptxt, int level, double scale,
              const std::vector<Complex>& message) const;
  void Decode(std::vector<Complex>& message,
              const Plaintext<word>& ptxt) const;
#endif
};

// Move-only payload types with full move support (default + move-ctor +
// move-assign; copy deleted).
template <typename word>
struct Ciphertext {
  Ciphertext() = default;
  Ciphertext(Ciphertext&&) = default;
  Ciphertext& operator=(Ciphertext&&) = default;
  Ciphertext(const Ciphertext&) = delete;
  Ciphertext& operator=(const Ciphertext&) = delete;
  // The EvalPoly emitter reads the actual level/scale off the input ct.
  double GetScale() const;
  NPInfo GetNP() const;
};
template <typename word>
struct Plaintext {
  Plaintext() = default;
  Plaintext(Plaintext&&) = default;
  Plaintext& operator=(Plaintext&&) = default;
  Plaintext(const Plaintext&) = delete;
  Plaintext& operator=(const Plaintext&) = delete;
};
template <typename word>
struct Constant {
  Constant() = default;
  Constant(Constant&&) = default;
  Constant& operator=(Constant&&) = default;
  Constant(const Constant&) = delete;
  Constant& operator=(const Constant&) = delete;
};
template <typename word>
struct EvaluationKey {
  EvaluationKey() = default;
  EvaluationKey(EvaluationKey&&) = default;
  EvaluationKey& operator=(EvaluationKey&&) = default;
  EvaluationKey(const EvaluationKey&) = delete;
  EvaluationKey& operator=(const EvaluationKey&) = delete;
};

// Move-only, and -- unlike the payload types -- has *no* move-assignment and
// is not default-constructible (the real EvkMap inherits std::unordered_map
// and declares only a move ctor). This is what makes the value+assign getter
// shape uncompilable, so the stub preserves it.
template <typename word>
struct EvkMap {
  EvkMap(EvkMap&&) = default;
  EvkMap(const EvkMap&) = delete;
  EvkMap& operator=(const EvkMap&) = delete;

#ifdef HEIR_CYCLOPS_STUB
  // Every lookup names the secret the key must have been built for; the
  // rotation/conjugation getters additionally take the level, which is why the
  // emitter hands the Context the whole map instead of a key.
  const EvaluationKey<word>& GetRotationKey(int rot_idx, SecretId secret,
                                            const Parameter<word>& param,
                                            int level, KeyMode key_mode) const;
  const EvaluationKey<word>& GetConjugationKey(SecretId secret) const;
  const EvaluationKey<word>& GetMultiplicationKey(SecretId secret) const;
#else
  const EvaluationKey<word>& GetRotationKey(int) const;
  const EvaluationKey<word>& GetConjugationKey() const;
  const EvaluationKey<word>& GetMultiplicationKey() const;
#endif
};

class StripedMatrix {
 public:
  StripedMatrix(int rows, int columns);
  std::vector<Complex>& operator[](int diagonal);
};

template <typename word>
class LinearTransform {
 public:
  LinearTransform(std::shared_ptr<const Context<word>> context,
                  const StripedMatrix& matrix, int level, double scale, int bs,
                  int gs);
#ifdef HEIR_CYCLOPS_STUB
  void Evaluate(std::shared_ptr<const Context<word>> context,
                Ciphertext<word>& result, const Ciphertext<word>& input,
                const EvkMap<word>& evk_map) const;
#else
  void Evaluate(std::shared_ptr<const Context<word>> context,
                Ciphertext<word>& result, const Ciphertext<word>& input,
                const EvkMap<word>& evk_map, bool min_ks = false) const;
#endif
};

template <typename word>
class UserInterface {
 public:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

  explicit UserInterface(std::shared_ptr<Context<word>> context);

#ifdef HEIR_CYCLOPS_STUB
  // Key generation names the secret the keys are built for.
  void PrepareRotationKey(int rot_idx, SecretId secret, int max_level = -1,
                          bool force = false,
                          KeyMode key_mode = KeyMode::kInherit);
  void PrepareRotationKey(const EvkRequest& request, SecretId secret,
                          bool force = false);
#else
  void PrepareRotationKey(int rot_idx, int max_level = -1, bool force = false);
  void PrepareRotationKey(const EvkRequest& request, bool force = false);
#endif

  void Encrypt(Ct& res, const Pt& a) const;
  void Decrypt(Pt& res, const Ct& a) const;

#ifdef HEIR_CYCLOPS_STUB
  // The secret-blind rotation/conjugation/multiplication getters are gone: a
  // key is qualified by its secret (and, for rotations, by the levels it
  // covers), so lookups go through the map.
  const Evk& GetRotationKey(int rot_idx, int level, SecretId secret,
                            KeyMode key_mode = KeyMode::kInherit) const;
#else
  const Evk& GetRotationKey(int rot_idx) const;
  const Evk& GetConjugationKey() const;
  const Evk& GetMultiplicationKey() const;
#endif
  const EvkMap<word>& GetEvkMap() const;
};

template <typename word>
class Context {
 public:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Const = Constant<word>;
  using Evk = EvaluationKey<word>;

  // Overloaded ct/pt/const arithmetic -- the emitter dispatches the dialect's
  // *_plain / *_const ops to the base name and relies on C++ overloading.
  void Copy(Ct& res, const Ct& input) const;
  void Add(Ct& res, const Ct& a, const Ct& b) const;
  void Add(Ct& res, const Ct& a, const Pt& b) const;
  void Add(Ct& res, const Ct& a, const Const& b) const;
  void Sub(Ct& res, const Ct& a, const Ct& b) const;
  void Sub(Ct& res, const Ct& a, const Pt& b) const;
  void Mult(Ct& res, const Ct& a, const Ct& b) const;
  void Mult(Ct& res, const Ct& a, const Pt& b) const;
  void Mult(Ct& res, const Ct& a, const Const& b) const;

  void Neg(Ct& res, const Ct& a) const;
  void Rescale(Ct& res, const Ct& a) const;
  void Relinearize(Ct& res, const Ct& a, const Evk& key) const;
  void RelinearizeRescale(Ct& res, const Ct& a, const Evk& key) const;
  void LevelDown(Ct& res, const Ct& a, int target_level) const;

  void HMult(Ct& res, const Ct& a, const Ct& b, const Evk& mult_key,
             bool rescale) const;
  void HRot(Ct& res, const Ct& a, const Evk& rot_key, int rot_dist) const;
  void HRotAdd(Ct& res, const Ct& a, const Ct& b, const Evk& rot_key,
               int rot_dist) const;
  void HConj(Ct& res, const Ct& a, const Evk& conj_key) const;
  void HConjAdd(Ct& res, const Ct& a, const Ct& b, const Evk& conj_key) const;

#ifdef HEIR_CYCLOPS_STUB
  // Cyclops' preferred overloads: resolve the key from the map at the
  // ciphertext's own secret and level (best fit), because a caller-supplied key
  // that does not cover the ciphertext's primes is a hard error there.
  void HRot(Ct& res, const Ct& a, const EvkMap<word>& evk_map, int rot_dist,
            KeyMode key_mode = KeyMode::kInherit) const;
  void HRotAdd(Ct& res, const Ct& a, const Ct& b, const EvkMap<word>& evk_map,
               int rot_dist, KeyMode key_mode = KeyMode::kInherit) const;
  void HConj(Ct& res, const Ct& a, const EvkMap<word>& evk_map) const;
  void HConjAdd(Ct& res, const Ct& a, const Ct& b,
                const EvkMap<word>& evk_map) const;

  // The handle of the program's secret, which every key request names.
  SecretId BootSecretId() const;
#endif

  // In-place multiply-accumulate: `res` is mutated, so it is a *non-const*
  // reference. This is the crux of the mad_unsafe finding.
  void MadUnsafe(Ct& res, const Ct& a, const Const& b) const;

  static std::shared_ptr<Context<word>> Create(const Parameter<word>& param);

  // The EvalPoly emitter reads the canonical scale from here.
  Parameter<word> param_;
};

// ConstContextPtr is a non-owning shared_ptr aliased onto the raw Context*.
template <typename word>
using ConstContextPtr = std::shared_ptr<const Context<word>>;

#ifdef HEIR_CYCLOPS_STUB
// Cyclops' EvalPoly resolves the multiplication key per ciphertext, through a
// selector that wraps either a fixed key or the whole map.
// (extension/poly/EvalPoly.h)
template <typename word>
class MultKeySelector {
 public:
  // NOLINTNEXTLINE(google-explicit-constructor) -- matches the real class.
  MultKeySelector(const EvaluationKey<word>& key);
  explicit MultKeySelector(const EvkMap<word>& evk_map,
                           KeyMode key_mode = KeyMode::kInherit);
};
#endif

// CHEDDAR's EvalPoly extension: also a class -- construct from coefficients,
// Compile(), then Evaluate() with the multiplication key.
template <typename word>
class EvalPoly {
 public:
  EvalPoly(const std::vector<double>& coefficients, int input_level,
           double input_scale, double target_scale, bool chebyshev = false);
  void Compile(ConstContextPtr<word> context);
  void Evaluate(ConstContextPtr<word> context, Ciphertext<word>& res,
                const Ciphertext<word>& input,
#ifdef HEIR_CYCLOPS_STUB
                const MultKeySelector<word>& mult_key) const;
#else
                const EvaluationKey<word>& mult_key) const;
#endif
};

// Boot lives on the derived BootContext (extension/BootContext.h), not Context.
// cheddar.boot takes a !cheddar.boot_context, lowered to BootContext<word>*, so
// the emitter calls `ctx->Boot(...)` directly; the stub mirrors that hierarchy.
template <typename word>
class BootContext : public Context<word> {
 public:
  using Ct = Ciphertext<word>;
  static std::shared_ptr<BootContext<word>> Create(
      const Parameter<word>& param, const BootParameter& boot_param);
  void Boot(Ct& res, const Ct& a, const EvkMap<word>& evk_map) const;

  // The one-time bootstrap precompute. Cyclops renamed the CtS/StC pair after
  // the transform it prepares; the arguments are unchanged.
  void PrepareEvalMod();
#ifdef HEIR_CYCLOPS_STUB
  void PrepareHomomorphicDFT(int num_slots,
                             BootVariant variant = BootVariant::kNormal);
#else
  void PrepareEvalSpecialFFT(int num_slots, BootVariant variant);
#endif
  void AddRequiredRotations(EvkRequest& request, int num_slots) const;
};

}  // namespace cheddar

#endif  // TESTS_CONVERSIONS_CHEDDARTOEMITC_COMPILE_CHEDDAR_STUB_H_
