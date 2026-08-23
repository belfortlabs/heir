// Header-only stub of the CHEDDAR API with the real library's move/const
// semantics (move-only payloads with move-assignment; EvkMap move-only without
// move-assignment; MadUnsafe mutates `res`; getters return const&). Used to
// compile, not run, the emitted C++ without CUDA.
// `HEIR_CYCLOPS_STUB` switches the fork-specific half of the surface to
// Cyclops: secret-indexed key lookups, tagged plaintexts, no min_ks flag on
// LinearTransform::Evaluate, parity on EvalPoly, PrepareHomomorphicDFT.

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

// The encoder exposes the per-level canonical scale.
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
  // Containers carry their ring; encode targets take it from the context.
  void MatchRing(const Plaintext& other);
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

// Move-only, no move-assignment, not default-constructible (like the real
// EvkMap, which inherits std::unordered_map and declares only a move ctor).
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

  // In-place: `res` is a non-const reference.
  void MadUnsafe(Ct& res, const Ct& a, const Const& b) const;

  Parameter<word> param_;
  Encoder<word> encoder_;
#ifdef HEIR_CYCLOPS_STUB
  SecretId BootSecretId() const;
  Plaintext<word> NewPlaintext() const;
#endif
};

// ConstContextPtr is a non-owning shared_ptr aliased onto the raw Context*.
template <typename word>
using ConstContextPtr = std::shared_ptr<const Context<word>>;

// CHEDDAR's EvalPoly extension: construct from coefficients, Compile(), then
// Evaluate() with the multiplication key.
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

// Boot lives on BootContext, not Context.
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
