# HEIR requirements for Cyclops `MaxPool`

HEIR lowers a PyTorch `MaxPool1d` layer to one call of
`MaxPool<word>::EvaluateMax` (branch `aikata/maxpool`). HEIR plans levels and
evaluation keys at compile time, without a GPU. This file lists what HEIR needs
from Cyclops for that.

Updated on 2026-10-02 for commit e8efd441 ("MaxPool: CPU key planner, range
normalization in the gather weights"). Thank you for R1 and R4.

Updated on 2026-10-05, still for e8efd441: HEIR now gets its levels from
`PlanMaxPool`, R6 is confirmed as a limit, and R10 to R16 are new. HEIR ran
`MaxPool1d(8)` and `MaxPool1d(2)` end to end on an RTX 5090, with worst errors
of 0.00035 and 0.00021.

## Summary

| Id | Requirement | Status at e8efd441 |
|----|-------------|--------------------|
| R1 | Move the stage functions to the CPU planner library. | Done. |
| R2 | An option that disables the small-window league cap. | **Open. Blocks HEIR when the cap has an effect.** |
| R3a | The level drop and the minimum input level. | Done: HEIR uses `PlanMaxPool`. One question. |
| R3b | The minimum bootstrap end level. | **Open. Blocks HEIR for windows of 16 or more.** |
| R4 | A key planner for `MaxPool`. | Done. HEIR adapts to the new signature. |
| R5 | Confirm that slots after `input_length` may hold data. | Open question. |
| R6 | Sparse ciphertexts and fewer than 256 slots. | **Open. Blocks HEIR below 256 slots.** |
| R7 | Clarify the dilation rule. | Open question. |
| R8 | Planner build dependency on `libtommath`. | HEIR solves it. Request: make it optional. |
| R9 | The `value_bound` rule. | Resolved: HEIR passes the bound. |
| R10 | The minimum input level and the level drop in `MaxPoolPlan`. | New request. |
| R11 | Planning without the selection maps. | New request. |
| R12 | Validation without exceptions. | New request. |
| R13 | A default `pad_value` of `-value_bound`. | New request. |
| R14 | Several sequences in one call. | New request. |
| R15 | `Compile` without the input scale. | New request. |
| R16 | A stable secret-id API. | New request. |

## How HEIR uses `MaxPool`

- `total_slots` is the slot count of the ciphertext (`input.GetNumSlots()`).
- `compact_output = true`, so the maximum of window `w` is in slot `w`.
- HEIR does not normalize the values. It sets
  `value_bound = max(|domain_lower|, |domain_upper|)` from the domain of the
  pool, and `pad_value = -value_bound`. See R9.
- In the first version, `stride == window_size`, `dilation == 1`, and
  `input_length` is a multiple of the stride. HEIR puts all channels in one
  sequence, so no window crosses a channel boundary.
- HEIR now constructs and compiles the `MaxPool` object in each call of the
  generated function, and then calls `EvaluateMax`. HEIR wants to compile it
  once, in its preprocessing step (see R15).
- HEIR uses the default comparison,
  `BuildMaxSelectorLowDepthSignApproximation()`, and
  `BootVariant::kImaginaryRemoving`.

## R1: Stage functions in the planner library

**Status: done.** `extension/max/MaxPoolPlanner.h` has the configuration, the
stage functions and `PlanMaxPool`. The planner library builds them. HEIR does
not need `ChainWindows` or `WindowSpan`, so they can stay private.

The HEIR prototype of this move (branch `mdgrs/maxpool-planner`) is obsolete.
We will delete it.

## R2: An option that disables the small-window league cap

**Status: open.** This blocks HEIR for a window of 2 to 7 slots, which
includes the common `MaxPool1d(2)`, when the cap has an effect. The cap has an
effect when the input level is more than the selector depth + 1 (10) above the
gather stages, for example for a pool directly after a bootstrap.

**Problem.** `PlanMaxPool` (`src/extension/max/MaxPoolPlanner.cpp`) sets the
league level as follows:

```cpp
plan.league_level = padded > 1 && padded < 8
    ? std::min(bootstrap_level, comparison.GetTotalLevelConsumption() + 1)
    : bootstrap_level;
// ... then, for each chain:
plan.league_level = std::min(plan.league_level, input_level - plan.gather_stages.back());
```

So for a padded window of fewer than 8 slots, the output level is:

```
output = min(end, selector depth + 1, input - gather) - league - compaction
```

With the default selector (9 levels), this is `min(end, 10, input - gather)`.
The output level thus depends on the absolute bootstrap end level.

HEIR counts levels as levels consumed. It knows the absolute end level only
after it generates the parameters, and that happens after the level analysis.
So HEIR cannot model the cap. A wrong level stops the program at run time,
because `Context::Add` asserts equal levels.

The cap applies in all HEIR programs. The league needs an end level of at
least 11 (R3b), and 11 is more than 10. So the cap lowers each input at the end
level.

**Request.** Add a field to `MaxPoolConfig`, for example:

```cpp
// When false, the league starts at the input level less the gather stages, and
// input_level - PlanMaxPool(...).output_level depends only on the configuration.
bool cap_league_level = true;
```

`PlanMaxPool`, `MaxPool::Compile` and `AddMaxPoolRequiredKeys` must all obey
the field. HEIR will set it to `false`. The cap is only a speed optimization: it
makes the comparisons cheaper when the input level is high. The name of the
field is only a proposal. HEIR will use the name that you select.

**HEIR status.** The key planner call and the generated code set
`cap_league_level = false` only when the field exists (`if constexpr`), so both
compile against e8efd441. Key planning calls `PlanMaxPool` with the input level
and the bootstrap end level of the program, and compares the level drop with
HEIR's plan. Key planning stops with an error only when the two differ, that
is, when the cap has an effect. The generated code also checks at run time that
`GetOutputLevel()` is the level that HEIR planned. A `MaxPool1d(2)` with its
input at level 9 passed end to end on a GPU.

## R3a: The level drop and the minimum input level

**Status: done.** HEIR gets these values from `PlanMaxPool` and no longer
copies the rules. HEIR calls `PlanMaxPool` at input level 64 and bootstrap
level 64. The cap changes only the league input level, so these values do not
depend on it:

- **Level drop:** the largest `gather_stages` entry, plus
  `league_level - league_output_level`, plus the largest `compaction_stages`
  entry.
- **Minimum input level:** the lowest `input_level` for which `PlanMaxPool`
  does not throw. HEIR searches up from the level drop. See R10.

**Question.** With the cap disabled (R2), please confirm these two points:

1. `input_level - plan.output_level` is the same for each valid input level and
   each valid bootstrap level.
2. When `PlanMaxPool` accepts an input level, `MaxPool::Compile` also accepts
   it. We read that the `level >= 1` check in `CompileSelections`
   (`MaxPool.cpp:46`) follows from the checks in `PlanMaxPool`. Is this
   correct?

## R3b: The minimum bootstrap end level

**Status: open.**

**Problem.** HEIR selects the bootstrap end level at compile time, during
parameter generation. Cyclops checks that level only at run time, in
`HierarchicalLeague::Compile`. If the level is too low, the compiled program
stops at run time with an assert, for example "HierarchicalLeague: insufficient
levels for the running selector".

`Compile` has several conditions on `end`:

| Check | Line in `HierarchicalLeague.cpp` |
|-------|----------------------------------|
| `end >= round_cost_ + 2` | 376 |
| `indicator_level >= 1` for each round | 400 |
| `selector_level_ >= 0`, 1 level lower for each round | 393-403 |
| `plan->output_level >= 2` between rounds | 462 |
| each comparison stage fits one bootstrap | `MaxCommon.h:214` |

There are also implicit conditions. `Compile` builds masks and constants at
levels that it calculates in the function, for example
`CompileBleach(context, end - 3, kBleachTouchUp)` (line 428). The `Round`
constructor (line 221) also calculates its own levels.

HEIR now uses a fixed minimum of 11. That value is from line 376 only. From the
explicit checks above, we calculate these values:

| Padded window | League rounds | Minimum end level |
|---------------|---------------|-------------------|
| 2 to 8 | 0 | 11 |
| 16 | 1 (2 candidates) | 12 |
| 64 | 1 (8 candidates) | 12 |
| 256 | 2 (8, then 4) | 13 |

We did not run these shapes. A HEIR copy of these rules can drift from the
Cyclops code without warning, and it can miss an implicit condition.

**Request.** Add this CPU-only function to `MaxPoolPlanner.h`:

```cpp
// The lowest bootstrap end level for which MaxPool::Compile accepts this
// configuration and comparison, for each valid input level. Each higher end
// level is also accepted.
int MaxPoolMinimumBootstrapEndLevel(
    const MaxPoolConfig& config,
    const CompositePolynomial& comparison = BuildMaxSelectorLowDepthSignApproximation());
```

**Recommended form.** Please do for the league what e8efd441 does for the pool
with `MaxPoolPlan`. Move each level that `HierarchicalLeague::Compile` uses into
a CPU-only plan:

```cpp
struct HierarchicalLeaguePlan
{
    int round_cost, selector_level, output_level;
    int pair_level, weighted_level;     // windows of 8 slots or more
    std::vector<int> round_mask_levels; // one for each round
    // ... each level that Compile uses
};

// Throws when Compile would reject the arguments.
HierarchicalLeaguePlan PlanHierarchicalLeague(int total_slots, int window_size, int input_level,
                                              int end, const CompositePolynomial& comparison);
```

`Compile` then reads its levels from this plan. `MaxPoolMinimumBootstrapEndLevel`
is a search for the lowest `end` that `PlanMaxPool` and `PlanHierarchicalLeague`
accept. The planner and the run-time code then cannot disagree.

**Simple form.** If the plan is too much work now, a check function is
sufficient. This sketch shows the explicit checks only. We did not compile it.

```cpp
bool HierarchicalLeagueEndLevelValid(int total_slots, int window_size, int end,
                                     const CompositePolynomial& comparison)
{
    using namespace max_detail;
    if (window_size == 1)
        return true;
    for (const auto& stage : comparison.stages)          // MaxCommon.h:214
        if (stage.GetLevelConsumption() > end)
            return false;
    const int round_cost = ComparisonMinimumInputLevel(comparison, end);
    if (end < round_cost + 2)                            // line 376
        return false;
    int selector = window_size < 8 ? end - 2 * kBleachTouchUp - (window_size - 2)
                                   : end - kWarmupSelectorCost;
    const auto rounds = LeagueRounds(window_size);
    for (size_t r = 0; r < rounds.size(); ++r)
    {
        const int indicator = LeagueIndicatorLevel(end, rounds[r].first);
        if (indicator < 1)                               // line 400
            return false;
        if (r + 1 < rounds.size() && indicator < 2)      // line 462
            return false;
        selector = std::min(selector, indicator) - 1;
    }
    return selector >= 0;                                // line 403
}
```

**Questions.**

1. Do the implicit conditions ever need a higher end level than the explicit
   checks?
2. Is the set of valid end levels closed upward? `ComparisonMinimumInputLevel`
   returns the full depth when `depth <= end`, and the first stage when it does
   not. So a low end level can pass the check at line 376, and a higher one can
   fail it. HEIR assumes that each end level above the minimum is valid. If
   this is not true, please export the check function in place of a minimum.

**Test proposal.** Next to `MaxPoolPlanner.KeysAndLevelsMatchCompiledMaxPool`,
check for each configuration that `Compile` passes at
`MaxPoolMinimumBootstrapEndLevel(config)` and fails one level below it.

**HEIR status.** `MaxPoolOp::getMinimumBootstrapEndLevel` returns the fixed 11.
`annotate-mgmt` raises the base level so that the end level is at least this
value. When the function exists, HEIR calls it with the configuration of each
pool.

## R4: Key planner

**Status: done.** `AddMaxPoolRequiredKeys` is in `MaxPoolPlanner.h`. Its
signature differs from our proposal: it has a `BootVariant` argument, and
`comparison` comes before `key_mode`. HEIR will call it as follows:

```cpp
cyclops::AddMaxPoolRequiredKeys(request, params, bootstrap, config, input_level,
                                cyclops::BootVariant::kImaginaryRemoving);
```

No action from Cyclops is necessary.

## R5: Data in the slots after `input_length`

**Status: open question.**

The header still says: "Input slots at or beyond input_length must be zero."
In e8efd441, the empty gather slots read `pad_value`.

HEIR's default layout repeats the data with a period of
`nextPow2(input_length)`. So the slots after `input_length` hold copies of the
input values, which are in `[-value_bound, value_bound]`.

We read the code as follows:

- With gather stages, the selection maps read only slots below
  `input_length`. Each other slot gets `pad_value`.
- Without gather stages, a slot at or after `input_length` belongs to a window
  with index `NumWindows()` or higher, because `input_length` is a multiple of
  the stride. The compaction drops those windows.

**Question.** Is a value in the comparison domain safe in those slots, also for
the bootstraps inside the league? If yes, can the header say so?

## R6: Sparse ciphertexts and fewer than 256 slots

**Status: open. Blocks HEIR for ciphertexts with fewer than 256 slots.**

HEIR ciphertexts have fewer slots than N/2: the slot count is the program's
requested slot count, for example 8192 with N = 2^16. HEIR prepares the
bootstrap context for `max(requested slot count, 256)` slots. So for a small
program, the prepared slot count can be more than the slot count of the
ciphertext.

`AddMaxPoolRequiredKeys` requests the bootstrap keys for `config.total_slots`
(through `AddHierarchicalLeagueRequiredKeys`). With `total_slots = 64`,
`AddBootstrapRequiredRotations` throws "Currently only high number of slots are
supported" (`BootKeyPlanner.cpp:58`), because it requires at least 256 slots.
So no max pool on a ciphertext with fewer than 256 slots can get its keys.

**Request.** Let `MaxPool` bootstrap at the slot count of the boot context when
the ciphertext has fewer slots, and plan those keys. If this is not possible,
document a minimum of 256 slots in `MaxPoolPlanner.h`, so that HEIR can reject
the shape early.

**Question.** Does `MaxPool` (with its internal bootstraps) work on a
ciphertext with `total_slots = input.GetNumSlots()` of 256 or more, when the
bootstrap context was prepared for more slots? Are the bootstrap keys for
`total_slots` the correct keys in that case?

## R7: The dilation rule

**Status: open question.**

`MaxPoolPlanner.h` says: "Dilation > 1 with overlapping window spans
(stride < dilation*(window_size-1)+1) is rejected at construction."

The code rejects only shapes whose gather stages collide
(`StagesFromPositions`). For example, `window_size = 4, stride = 4,
dilation = 2` has overlapping spans, and `ValidateMaxPoolConfig` accepts it.

**Question.** Which rule is correct? HEIR now uses the stricter rule from the
header.

The `kernel.max_pool` verifier cannot call the planner, because the HEIR Kernel
dialect does not depend on Cyclops. So the verifier copies the rules of
`ValidateMaxPoolConfig`, and also the rules of R5 and R7. A clear statement in
`MaxPoolPlanner.h` keeps that copy correct.

## R8: `libtommath` in the planner build

**Status: HEIR solves it. Request: make `libtommath` optional.**
`aikata/maxpool` is based on e997f4b9, so the planner needs `libtommath`. HEIR
already builds `libtommath` for its CHEDDAR backend, and gives it to the
planner build (`bazel/cyclops/cyclops.BUILD`). This needs two workarounds in
HEIR: an include directory for `<tommath.h>` in the CMake call, and `-O2` for
`libtommath`, because its random-source code links only after optimization.

**Request.** Make `libtommath` optional for the planner, for example with a
CMake option that leaves out the `BigInt` primality checks. HEIR then removes
both workarounds.

## R9: The `value_bound` rule

**Status: resolved.** HEIR sets `value_bound` to `max(|domain_lower|,
|domain_upper|)`, from the domain that the model gives for the pool, and
`pad_value = -value_bound`. HEIR no longer normalizes the input. A pool without
gather stages (or without compaction stages) then costs one more level for
each scaling that Cyclops cannot fold, and HEIR's level plan counts it through
`MaxPoolGatherMaps` and `MaxPoolCompactionMaps`.

For a ceil-mode pool, torch-mlir pads the input with `-inf`. HEIR replaces that
padding with `domain_lower`, which is in `[-value_bound, value_bound]` and is
not more than any real value.

## R10: The minimum input level and the level drop in `MaxPoolPlan`

**Status: new request.**

HEIR needs two values for each pool: the level drop and the lowest input level.
`MaxPoolPlan` gives neither directly. HEIR takes the largest entry of each
per-chain stage list, and it calls `PlanMaxPool` again for each input level from
the level drop up, until it does not throw.

**Request.** Add these fields to `MaxPoolPlan`:

```cpp
int minimum_input_level = -1; // the lowest input_level that PlanMaxPool accepts
int level_drop = -1;          // input_level - output_level without the league cap
```

## R11: Planning without the selection maps

**Status: new request.**

`PlanMaxPool` builds every gather and compaction map over all slots, for each
chain, only to count the stages. One HEIR depth query (with the R10 search)
takes up to 23 ms at 32768 slots. HEIR's level analyses ask for the same pool
many times, so HEIR caches the result for each shape.

**Request.** Add functions that count the stages without the maps, for example
`MaxPoolNumGatherStages(config, chain)` and
`MaxPoolNumCompactionStages(config, chain)`, and use them in `PlanMaxPool`.
HEIR can then remove its cache.

## R12: Validation without exceptions

**Status: new request.**

The planner reports an invalid configuration only with an exception. HEIR
builds its planner wrappers with `-fexceptions` and catches the exception,
although the rest of HEIR builds without exceptions.

**Request.** Add a validation function that returns the reason, for example:

```cpp
// std::nullopt when ValidateMaxPoolConfig accepts the configuration.
std::optional<std::string> MaxPoolConfigError(const MaxPoolConfig& config);
```

A similar non-throwing form of `PlanMaxPool` is also useful.

## R13: A default `pad_value` of `-value_bound`

**Status: new request.**

HEIR sets the same fields of `MaxPoolConfig` in three places: the planner
wrapper, the key planner and the generated C++. One of them is always
`pad_value = -value_bound`. The generated C++ sets the fields one at a time,
because the HEIR code generator cannot emit a brace initializer.

**Request.** Make `-value_bound` the default `pad_value`, for example with a
sentinel default that means "-value_bound". A factory function that takes all
fields as arguments is also good, for example
`MaxPoolConfig::Make(total_slots, input_length, window_size, stride, dilation,
value_bound, ceil_mode)`.

## R14: Several sequences in one call

**Status: new request.** This is the largest possible simplification for HEIR.

HEIR puts all channels of a pool end to end in one sequence. A window must not
cross into the next channel. So for a ceil-mode pool, HEIR first pads each
channel to a multiple of the window size, and fills the padding with the lower
bound of the domain: a zero pad plus a plaintext constant. For a floor-mode
pool, HEIR first slices off the values that no window reads. The padding also
costs a layout conversion before the pool.

**Request.** Let `MaxPoolConfig` describe several sequences, for example with
`num_sequences` and `sequence_stride` (the distance in slots between the starts
of two sequences). Each sequence then has its own windows, its own ceil-mode
tail and its own padding with `pad_value`. HEIR then passes `ceil_mode`
directly, and removes its padding and trimming patterns.

## R15: `Compile` without the input scale

**Status: new request.**

`MaxPool` takes the input scale in its constructor, so HEIR builds and compiles
the object in the generated function, from the scale of the ciphertext. Then
`Compile` (selection transforms, league masks and constants) runs on each call.
HEIR has a preprocessing step, where it already prepares linear transforms
once.

**Request.** Let `MaxPool` use the canonical scale of the input level when the
caller gives no scale, or take the scale in `EvaluateMax`. HEIR can then
compile the object once, in its preprocessing step. Please also confirm that a
compiled `MaxPool` can evaluate many inputs.

## R16: A stable secret-id API

**Status: new request.**

Cyclops commit 8de7c716 renamed `BootSecretId()` to `NativeSecretId()` on
`ClientContext` and `Parameter`. HEIR emits `BootSecretId()`, so its generated
code does not compile against e8efd441 without a change. For the end-to-end
runs, we changed the name in the generated files by hand.

**Request.** Keep `BootSecretId()` as a deprecated alias of `NativeSecretId()`
for some releases, so that HEIR can move to the new name without a break.
