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
// `HEIR_CYCLOPS_STUB` switches the fork-specific half of the surface to
// Cyclops.
//
//   * Cyclops has no level-blind key getter; every lookup goes through the
//     level-aware EvkMap overloads, which the emitter calls for both forks.
//     Compiling the Cyclops kernels against a stub that still declared the
//     getters would prove nothing, so they are compiled out here.
//   * Every Cyclops evaluation key is indexed by the secret it matches, so
//     lookups and key preparation take a SecretId.
//   * Cyclops containers start untagged and Encrypt rejects an untagged
//     plaintext, hence Plaintext::SetSecretId.
//   * Cyclops' LinearTransform::Evaluate has no min_ks flag, and its
//     constructor takes the compact plaintext period where scale-snu takes
//     pre_rotation.
//   * Cyclops' EvalPoly states the polynomial's parity in the constructor.
//   * Cyclops names the homomorphic DFT preparation PrepareHomomorphicDFT,
//     scale-snu PrepareEvalSpecialFFT.
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
struct Plaintext;

#ifdef HEIR_CYCLOPS_STUB
// A handle into the parameter set's secret descriptor list. Every container is
// tagged with one and every evaluation key is indexed by one. (core/Type.h)
struct SecretId {
  int value = -1;
};

enum class KeyMode { kInherit, kDefault, kLevelSpecific };
// kShapeOnly records the BSGS index structure and skips plaintext encoding.
// (extension/linalg/Hoist.h)
enum class PlaintextMode { kCompiled, kShapeOnly };
struct PlaintextCacheConfig {};
// The polynomial's declared parity, which Cyclops' EvalPoly takes and trusts.
// (extension/poly/Approximation.h)
enum class PolynomialParity { kFull, kEven, kOdd };
#endif

// A batch of rotation-key requests; bootstrap setup and the transforms fill it
// and the UserInterface turns it into keys. (core/EvkRequest.h)
class EvkRequest {
 public:
  EvkRequest() = default;
};

// Which bootstrapping variant the homomorphic DFT is prepared for.
enum class BootVariant { kNormal, kImaginaryRemoving };

// Number-of-primes info a ciphertext carries; the EvalPoly emitter maps it back
// to a level via Parameter::NPToLevel.
struct NPInfo {
  int num_main_ = 0;
};

// The secret a key was built for, and the key-mode selector: what the emitted
// rotation/conjugation lookups pass when they read keys off the EvkMap with the
// secret handle and the parameters taken from the context. Cyclops declares
// both itself, above.
#ifndef HEIR_CYCLOPS_STUB
struct SecretId {
  int value = 0;
};
enum class KeyMode { kInherit };
#endif

// Minimal parameter stub: the EvalPoly emitter reads the level/q-product via
// NPToLevel / GetRescalePrimeProd (mirroring EvalMod).
template <typename word>
struct Parameter {
  double GetScale(int level) const;
  double GetRescalePrimeProd(int level) const;
  int NPToLevel(NPInfo np) const;
};

// The encoder exposes the per-level canonical scale (the EvalPoly emitter reads
// input/target scales via `encoder.GetScale(level)`).
template <typename word>
struct Encoder {
  double GetScale(int level) const;
#ifdef HEIR_CYCLOPS_STUB
  // Cyclops' slots API works in real doubles; scale-snu's Encode/Decode take
  // Complex. Only the Cyclops pair is exercised here (kernels.mlir has no
  // encode op).
  void EncodeSlots(Plaintext<word>& pt, int level, double scale,
                   const std::vector<double>& message) const;
  void DecodeSlots(std::vector<double>& message,
                   const Plaintext<word>& pt) const;
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
#ifdef HEIR_CYCLOPS_STUB
  void SetSecretId(SecretId secret);
#endif
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

  // The real map's lookups, which the emitter calls directly so an evaluating
  // process needs no UserInterface.
  const EvaluationKey<word>& GetRotationKey(int rot_idx, SecretId secret,
                                            const Parameter<word>& param,
                                            int level, KeyMode key_mode) const;
  const EvaluationKey<word>& GetConjugationKey(SecretId secret) const;
  const EvaluationKey<word>& GetMultiplicationKey(SecretId secret) const;
#ifndef HEIR_CYCLOPS_STUB
  // The level-blind getters, which only scale-snu has. The EvalPoly emitter
  // still reads the multiplication key through one.
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
#ifdef HEIR_CYCLOPS_STUB
  // bs = gs = 0 asks Cyclops' own planner for the split. The trailing
  // arguments are the compact plaintext period and its cache/key/plaintext
  // modes; scale-snu takes pre_rotation in that position instead.
  LinearTransform(std::shared_ptr<const Context<word>> context,
                  const StripedMatrix& matrix, int level, double scale, int bs,
                  int gs, int log_pt_size_per_prime = -1,
                  PlaintextCacheConfig cache_config = {},
                  KeyMode key_mode = KeyMode::kInherit,
                  PlaintextMode plaintext_mode = PlaintextMode::kCompiled,
                  int msg_slot_period = 0);
  void AddRequiredRotations(EvkRequest& request) const;
  void Evaluate(std::shared_ptr<const Context<word>> context,
                Ciphertext<word>& result, const Ciphertext<word>& input,
                const EvkMap<word>& evk_map) const;
#else
  LinearTransform(std::shared_ptr<const Context<word>> context,
                  const StripedMatrix& matrix, int level, double scale, int bs,
                  int gs);
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

  void Encrypt(Ct& res, const Pt& a) const;
  void Decrypt(Pt& res, const Ct& a) const;

#ifndef HEIR_CYCLOPS_STUB
  const Evk& GetRotationKey(int rot_idx) const;
  const Evk& GetConjugationKey() const;
  const Evk& GetMultiplicationKey() const;
#else
  void PrepareRotationKey(int rot_idx, SecretId secret, int max_level = -1,
                          bool force = false,
                          KeyMode key_mode = KeyMode::kInherit);
  void PrepareRotationKey(const EvkRequest& request, SecretId secret,
                          bool force = false);
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


  // In-place multiply-accumulate: `res` is mutated, so it is a *non-const*
  // reference. This is the crux of the mad_unsafe finding.
  void MadUnsafe(Ct& res, const Ct& a, const Const& b) const;

  // The EvalPoly emitter reads the canonical scale from here; the key lookups
  // read the parameters and the boot secret's handle.
  Parameter<word> param_;
  SecretId BootSecretId() const;
};

// ConstContextPtr is a non-owning shared_ptr aliased onto the raw Context*.
template <typename word>
using ConstContextPtr = std::shared_ptr<const Context<word>>;

// CHEDDAR's EvalPoly extension: also a class -- construct from coefficients,
// Compile(), then Evaluate() with the multiplication key.
#ifdef HEIR_CYCLOPS_STUB
// Selects the multiplication key per relinearization: from a fixed key, or
// from an EvkMap at the ciphertext's own level. (extension/poly/EvalPoly.h)
template <typename word>
class MultKeySelector {
 public:
  MultKeySelector(const EvaluationKey<word>& key);
  explicit MultKeySelector(const EvkMap<word>& evk_map,
                           KeyMode key_mode = KeyMode::kInherit);
};
#endif

template <typename word>
class EvalPoly {
 public:
#ifdef HEIR_CYCLOPS_STUB
  // Cyclops takes the declared parity second and has no overload without it.
  EvalPoly(const std::vector<double>& coefficients, PolynomialParity parity,
           int input_level, double input_scale, double target_scale,
           bool chebyshev = false);
#else
  EvalPoly(const std::vector<double>& coefficients, int input_level,
           double input_scale, double target_scale, bool chebyshev = false);
#endif
  void Compile(ConstContextPtr<word> context);
#ifdef HEIR_CYCLOPS_STUB
  void Evaluate(ConstContextPtr<word> context, Ciphertext<word>& res,
                const Ciphertext<word>& input,
                const MultKeySelector<word>& mult_key) const;
#else
  void Evaluate(ConstContextPtr<word> context, Ciphertext<word>& res,
                const Ciphertext<word>& input,
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
  void Boot(Ct& res, const Ct& a, const EvkMap<word>& evk_map) const;

  void PrepareEvalMod();
#ifdef HEIR_CYCLOPS_STUB
  void PrepareHomomorphicDFT(int num_slots,
                             BootVariant variant = BootVariant::kNormal);
#else
  void PrepareEvalSpecialFFT(int num_slots, BootVariant variant);
#endif
  void AddRequiredRotations(EvkRequest& req, int num_slots) const;
};

}  // namespace cheddar

#endif  // TESTS_CONVERSIONS_CHEDDARTOEMITC_COMPILE_CHEDDAR_STUB_H_
