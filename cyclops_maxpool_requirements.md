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

Updated on 2026-10-07 for commit 1716ac30 ("hoisting the shared mod-ups"), the
tip of `aikata/maxpool`. This commit adds 2D pooling and idle-slot folding.
HEIR still pins e8efd441 and did not build against 1716ac30. The status of
each requirement below is from a read of the 1716ac30 source. R7 is answered.
R17 is new: the folding makes the level drop depend on the bootstrap end level.

Updated on 2026-10-08 for commit 0e5dde7b ("clang-format"), the tip of
`aikata/maxpool`, after 468bdaef ("MaxPool: minimum input and bootstrap end
level queries, output level pin"). Thank you for R3b and R10. HEIR still pins
e8efd441. We built the CPU planner library at 0e5dde7b and called it from a
test program. The numbers in this update come from that program. We did not
run these shapes on a GPU. R2 is still open and still blocks HEIR. R17 is
still open, but the new output level pin gives HEIR a workaround.

Updated on 2026-10-09, still for 0e5dde7b, the tip of `aikata/maxpool`. We
propose a patch for R2 and R6 on branch `mdgrs/maxpool-heir-fixes`, which
starts at 0e5dde7b. See "Proposed patch for R2 and R6" below. HEIR now pins
0e5dde7b, uses `MaxPoolMinimumBootstrapEndLevel`, `MaxPoolMinimumInputLevel`
and `MaxPoolTryPlanLevels`, and uses the output level pin as the workaround
for R17.

## Terms

- **Input level:** the level of the ciphertext that HEIR gives to `MaxPool`.
- **Output level:** the level of the result, `GetOutputLevel()`.
- **Level drop:** `input level - output level`.
- **Bootstrap end level** (or **end level**):
  `BootParameter::GetEndLevel()`, the level of a ciphertext immediately after a
  bootstrap. It is a property of the parameters, not of the ciphertext. The
  league bootstraps internally, and its comparisons run at levels below the
  end level. So the end level must be high enough for the comparisons, for
  each input level.

HEIR needs the level drop to be a constant for each configuration: one number,
for each valid input level and each valid end level. HEIR also needs the
minimum end level, because HEIR selects the parameters after its level
analysis.

## The main request

Most of the open items have one cause: HEIR cannot get, on the CPU and without
an exception, the three numbers that it needs for each configuration. We
propose one query that replaces R2, R3b, R10, R11, R12 and R17:

```cpp
struct MaxPoolRequirements
{
    int level_drop;                  // input_level - output_level, for each valid input and end level
    int minimum_input_level;
    int minimum_bootstrap_end_level; // each higher end level is also valid
};

// std::nullopt and a reason when the configuration is not supported. Does not throw.
std::optional<MaxPoolRequirements> GetMaxPoolRequirements(
    const MaxPoolConfig& config,
    const CompositePolynomial& comparison = BuildMaxSelectorLowDepthSignApproximation(),
    std::string* error = nullptr);
```

For this query to exist, `level_drop` must not depend on the end level. Today
two features make it depend on the end level: the small-window league cap (R2)
and the selection of the fold copies (R17). HEIR needs a configuration field
that makes the level drop constant. HEIR then removes its fixed minimum end
level of 11, its search for the minimum input level, its `PlanMaxPool` call at
level 64, its cache and its `-fexceptions` build flag.

**Status at 0e5dde7b: partly done.** 468bdaef adds
`MaxPoolMinimumBootstrapEndLevel`, `MaxPoolMinimumInputLevel`, the
non-throwing `MaxPoolTryPlanLevels`, and `MaxPoolConfig::output_level`. HEIR
can use the first two in place of its fixed 11 and its search for the minimum
input level.

The output level pin is useful for HEIR. HEIR plans a constant level drop `D`
for each pool. The generated code can then set
`output_level = input_level - D`, and the output level is always the planned
level. But the pin is a free level-down: it can only lower the output below the
natural output level. It cannot raise the output. So the pin solves the
problem only when `D` is at least the natural level drop for each valid input
level and end level:

- For R17, a largest natural drop exists. HEIR can find it with a search over
  the end levels, and then pin to it. See R17.
- For R2, no largest natural drop exists, because the drop grows with the input
  level. So R2 still blocks HEIR. See R2.

## Summary

| Id | Requirement | Status at 0e5dde7b |
|----|-------------|--------------------|
| R1 | Move the stage functions to the CPU planner library. | Done. |
| R2 | An option that disables the small-window league cap. | **Patch proposed:** the cap applies only to the comparison copy. |
| R3a | The level drop and the minimum input level. | Question 2 is answered. Question 1 waits for R2 and R17. |
| R3b | The minimum bootstrap end level. | Done: `MaxPoolMinimumBootstrapEndLevel`. |
| R4 | A key planner for `MaxPool`. | Done. |
| R5 | Confirm that slots after `input_length` may hold data. | Open question. |
| R6 | Sparse ciphertexts and fewer than 256 slots. | **Patch proposed:** the league plans the bootstrap keys of 256 slots. |
| R7 | Clarify the dilation rule. | Answered: overlapping dilated windows use fold copies. |
| R8 | Planner build dependency on `libtommath`. | HEIR solves it. Request: make it optional. |
| R9 | The `value_bound` rule. | Resolved: HEIR passes the bound. |
| R10 | The minimum input level and the level drop in `MaxPoolPlan`. | Done: `min_input_level` and `natural_output_level`. |
| R11 | Planning without the selection maps. | Open. One query takes up to 186 ms. |
| R12 | Validation without exceptions. | Partly done: `MaxPoolTryPlanLevels`. A configuration that is not valid still throws. |
| R13 | A default `pad_value` of `-value_bound`. | Open. |
| R14 | Several sequences in one call. | Open. One question about `MaxPool2d`. |
| R15 | `Compile` without the input scale. | Open. |
| R16 | A stable secret-id API. | Resolved: HEIR emits `NativeSecretId()`. |
| R17 | The fold copies must not depend on the end level. | Open. HEIR uses the workaround with the output level pin. |

## Proposed patch for R2 and R6

Branch `mdgrs/maxpool-heir-fixes`, from 0e5dde7b. It is not merged.

**R2.** The league reads the values at `input_level - gather` for each window
size. `Schedule` in `MaxPoolPlanner.cpp` no longer caps the league level. For
a window below 8 slots, `HierarchicalLeagueSmallWindowLevel` now returns
`min(input_level, bootstrap_level, comparison depth + 1)`, and
`EvaluateSelector` levels down a copy of the input to that level for the
comparisons. The comparisons thus run at the same level as before (10 with
the default selector), and cost the same. The values keep their level, and
the selector multiplies them at the league level. So the level drop is
`gather + 1 + compaction` for each input level. The compiled league, its
masks and its keys follow the same function, so the planner and the runtime
agree. The rotation keys of the window sums and the compaction are at higher
levels than before, so they are larger.

**R6.** `AddHierarchicalLeagueRequiredKeys` requests the bootstrap keys for
`max(total_slots, kMinBootstrapNumSlots)` slots. `kMinBootstrapNumSlots` (256)
is new in `BootKeyPlanner.h`. At run time, `BootContext` already bootstraps a
sparser ciphertext with the transform of a larger prepared slot count, and
gives the result the input's slot count. So the runtime needed no change. The
caller must prepare the boot context for `max(total_slots, 256)` slots, as
HEIR does.

**Tests** (`test/MaxPoolTest.cpp`, all on an RTX 5090):

- `MaxPoolPlanner.LevelDropDoesNotDependOnTheInputLevel`: for 8 shapes, each
  end level from the minimum to 24 and each input level, the drop is constant.
- `MaxPoolEncrypted.SmallWindowsKeepTheInputLevel`: `MaxPool1d(2)`, `(3)` and
  `(4)` from end level 15 and from the minimum input level. From level 15,
  `MaxPool1d(2)` now gives level 12; at 0e5dde7b it gave level 7.
- `MaxPoolEncrypted.FewerSlotsThanTheBootstrapMinimum`: pools on 64 and 128
  slots, with only the planner's keys in a fresh testbed. Worst error 2.3e-5.
- All other tests in `maxpool_test` and `hierarchical_league_test` pass.

A note for the key parity test: for a sparse slot count,
`AddBootstrapRequiredRotations` requests the split-format key with
`KeyMode::kInherit`, and `BootContext` requests it with the resolved mode. Both
give the same key, but `ExpectSameKeyRequests` reports a difference. So the
new test runs with the planner's keys instead of comparing the requests.

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

**Status: patch proposed on 2026-10-09** (see "Proposed patch for R2 and
R6"). The patch keeps the cap for the comparisons only, so a new field is not
necessary. The text below describes the problem at 0e5dde7b.

**Status: open at 0e5dde7b.** This blocks HEIR for an unfolded window of 2 to
7 slots, which includes the common `MaxPool1d(2)`, when the cap has an effect.
The cap has an effect when the input level is more than the selector depth + 1
(10) above the gather stages, for example for a pool directly after a
bootstrap. The output level pin of 468bdaef does not solve this problem (see
"Results at 0e5dde7b" below).

**Problem.** At 0e5dde7b, `Schedule` (`src/extension/max/MaxPoolPlanner.cpp`
line 380) sets the league level as follows:

```cpp
s.league = plan.padded > 1 && plan.padded < 8 && plan.copies == 1
               ? std::min(bootstrap_level, comparison.GetTotalLevelConsumption() + 1)
               : bootstrap_level;
s.league = std::min(s.league, input_level - StageDepth(plan.gather_maps));
```

The cap applies only to unfolded windows (`copies == 1`). For a folded window,
the copy count depends on the end level (R17).

So for an unfolded padded window of fewer than 8 slots, the output level is:

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

**Results at 0e5dde7b.** We called `MaxPoolTryPlanLevels` for
`MaxPool1d(2)` on 4096 slots with `input_length = 4096`. The plan has 1 gather
stage, 2 compaction stages, a minimum input level of 4 and a minimum end level
of 11. For each end level, we tried each input level from 4 to the end level:

| End level | Level drop over the valid input levels | Highest output level |
|-----------|----------------------------------------|----------------------|
| 11 | 4 | 7 |
| 12 | 4 to 5 | 7 |
| 16 | 4 to 9 | 7 |
| 24 | 4 to 17 | 7 |

`MaxPool1d(4)` on 4096 slots, and `MaxPool1d(2)` on 512 of 4096 slots, give the
same table. `MaxPool1d(3)` on 999 of 4096 slots gives drops of 5 to 17 and a
highest output level of 7. Padded windows of 8 or more do not have the cap,
and their drop is constant (4 for `MaxPool1d(8)`, `MaxPool1d(16)` and
`MaxPool1d(64)` on 4096 full slots).

The pin cannot remove this dependency, because it can only lower the output.
At end level 16 we pinned `output_level = input_level - D`:

| Input level | D = 3 | D = 4 | D = 6 | D = 8 |
|-------------|-------|-------|-------|-------|
| 8 | rejected | accepted | accepted | accepted |
| 12 | rejected | rejected | accepted | accepted |
| 16 | rejected | rejected | rejected | rejected |

For a constant `D`, an input level above `10 + D - league - compaction` is
always rejected (above `10 + D - 3` for these shapes). HEIR selects the end
level after its level analysis, so HEIR cannot put an upper limit on the input
level. A pin to a
fixed absolute level, for example `output_level = 7`, makes the output level
the same for each input level. But HEIR counts levels down from the end level,
so a fixed absolute level is a different HEIR level for each end level. HEIR
does not know the end level when it plans the pool.

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

**HEIR status (2026-10-09).** HEIR no longer sets `cap_league_level`. It pins
the output level to its planned drop, and key planning checks the pinned pool
with `MaxPoolTryPlanLevels`. With the patch, a `MaxPool1d(2)` on 64 slots with
its input at level 11 and no gather stage plans its keys and runs end to end.
Without the patch, key planning stops with an error for such a pool.

**HEIR status (2026-10-05).** The key planner call and the generated code set
`cap_league_level = false` only when the field exists (`if constexpr`), so both
compile against e8efd441. Key planning calls `PlanMaxPool` with the input level
and the bootstrap end level of the program, and compares the level drop with
HEIR's plan. Key planning stops with an error only when the two differ, that
is, when the cap has an effect. The generated code also checks at run time that
`GetOutputLevel()` is the level that HEIR planned. A `MaxPool1d(2)` with its
input at level 9 passed end to end on a GPU.

With the field, HEIR can also use the pin for R17. HEIR then sets both
`cap_league_level = false` and `output_level = input_level - D`.

## R3a: The level drop and the minimum input level

**Status at 0e5dde7b: Question 1 waits for R2 and R17.** R10 gives the
minimum input level directly. With the pin, HEIR can also pin the output to its
planned level drop when R2 is solved. Question 1 is then the question that
remains for this item.

At e8efd441 this was done. HEIR gets these values
from `PlanMaxPool` and does not copy the rules. HEIR calls `PlanMaxPool` at
input level 64 and bootstrap level 64. At e8efd441 the cap changed only the
league input level, so these values did not depend on it. At 1716ac30 the
fold copies depend on the bootstrap level, so the values at level 64 can differ
from the values at the end level of the program (R17). HEIR calculates:

- **Level drop:** the largest number of gather stages of a chain, plus
  `league_level - league_output_level`, plus the largest number of compaction
  stages of a chain.
- **Minimum input level:** the lowest `input_level` for which `PlanMaxPool`
  does not throw. HEIR searches up from the level drop. See R10.

**Question 1 (open).** With the cap disabled (R2) and the copies fixed (R17),
please confirm that `input_level - plan.output_level` is the same for each
valid input level and each valid bootstrap level.

**Question 2 (answered at 1716ac30).** We asked: when `PlanMaxPool` accepts an
input level, does `MaxPool::Compile` also accept it? At 1716ac30,
`MaxPool::Compile` calls `PlanMaxPool` with the end level of the boot context,
and then asserts that each phase agrees with the plan (`MaxPool.cpp:172`, 189
and 195). So the selection stages follow the plan. Only the league checks in
`HierarchicalLeague::Compile` can still reject an accepted plan (R3b).

## R3b: The minimum bootstrap end level

**Status: done at 468bdaef.** Thank you. `MaxPoolPlanner.h` now has:

```cpp
int MaxPoolMinimumBootstrapEndLevel(
    const MaxPoolConfig& config,
    const CompositePolynomial& comparison = BuildMaxSelectorLowDepthSignApproximation());
```

It is the "simple form" of our request: `MaxPoolTryPlanLevels` calls the new
`HierarchicalLeagueSupportsBootstrapLevel`, and the function searches down
from end level 32 to the first end level that fails. So the set of valid end
levels is closed upward by construction (our Question 2). The new GPU test
`MaxPoolEncrypted.RunsAtEveryEndLevelThePlannerAccepts` runs the pool with
parameter sets of 7 to 17 levels, at each end level that the planner accepts.

**Results at 0e5dde7b.** The planner gives these minimum end levels:

| Shape (4096 slots) | Minimum end level |
|--------------------|-------------------|
| `MaxPool1d(2)`, `MaxPool1d(4)`, `MaxPool1d(8)`, full slots | 11 |
| `MaxPool1d(16)`, `MaxPool1d(64)`, full slots | 13 |
| `MaxPool1d(64)` on 64 to 512 values (folded, 8 copies) | 12 |
| `MaxPool1d(9)`, `MaxPool1d(16)`, `MaxPool1d(32)` on few values (folded) | 11 |

For the full-slot shapes, we tried each end level from the minimum to 24, and
each input level from the minimum input level to the end level. The planner
accepted all of them. So the minimum end level does not depend on the
input level.

**Remaining risk (no action needed now).** `HierarchicalLeagueSupportsBootstrapLevel`
is a second copy of the checks in `HierarchicalLeague::Compile` (lines 442,
476, 480 and 552 at 0e5dde7b). `Compile` does not call it. So the two copies
can drift. The "recommended form" of our earlier request, a
`HierarchicalLeaguePlan` that `Compile` reads, removes this risk. A test that
`Compile` fails one level below `MaxPoolMinimumBootstrapEndLevel` also shows
that the minimum is not too high.

**HEIR status.** Done: HEIR calls `MaxPoolMinimumBootstrapEndLevel` for each
pool, in place of its fixed 11.

## R4: Key planner

**Status: done.** `AddMaxPoolRequiredKeys` is in `MaxPoolPlanner.h`. Its
signature differs from our proposal: it has a `BootVariant` argument, and
`comparison` comes before `key_mode`. HEIR will call it as follows:

```cpp
cyclops::AddMaxPoolRequiredKeys(request, params, bootstrap, config, input_level,
                                cyclops::BootVariant::kImaginaryRemoving);
```

No action from Cyclops is necessary. The signature is the same at 1716ac30.

## R5: Data in the slots after `input_length`

**Status: open question at 0e5dde7b.**

At 0e5dde7b the `MaxPool.h` header (line 5) still says: "Input slots at or beyond
input_length must be zero." Since e8efd441, the empty gather slots read
`pad_value` (`MaxPoolPlan::pad_slots` at 1716ac30).

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

**Status: patch proposed on 2026-10-09** (see "Proposed patch for R2 and
R6"). The patch is the first option of the request below: the league plans the
keys of the 256-slot bootstrap. The questions are answered by the new GPU test:
yes, for 64 and 128 slots with a boot context prepared for 256 slots.

**Status: open at 0e5dde7b. Blocks HEIR for ciphertexts with fewer than 256
slots.** `BootKeyPlanner.cpp:60` still requires 256 slots or more.

HEIR ciphertexts have fewer slots than N/2: the slot count is the program's
requested slot count, for example 8192 with N = 2^16. HEIR prepares the
bootstrap context for `max(requested slot count, 256)` slots. So for a small
program, the prepared slot count can be more than the slot count of the
ciphertext.

`AddMaxPoolRequiredKeys` requests the bootstrap keys for `config.total_slots`
(through `AddHierarchicalLeagueRequiredKeys`). With `total_slots = 64`,
`AddBootstrapRequiredRotations` throws "Currently only high number of slots are
supported" (`BootKeyPlanner.cpp:60` at 1716ac30), because it requires at least
256 slots.
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

**Status: answered at 1716ac30.**

At e8efd441, `MaxPoolPlanner.h` said that overlapping dilated windows are
rejected, but the code rejected only shapes whose gather stages collide.

At 1716ac30, `ValidateMaxPoolConfig` does not reject overlapping dilated
windows. `MaxPoolRequiredCopies` selects the smallest power-of-two copy count
that gives a collision-free gather. It throws only when no copy count up to
`total_slots / P` is collision-free. The header now says this. Thank you.

HEIR supports only `dilation == 1` in the first version, so this change does
not affect HEIR now. A shape that needs 2 or more copies uses more slots for
each window, and it can need more chains.

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

**Status: done at 468bdaef.** Thank you. `MaxPoolPlan` now has
`min_input_level` and `natural_output_level`, and `MaxPoolMinimumInputLevel`
gives the minimum input level from the configuration. HEIR can remove its
search through exceptions. The level drop is `input_level -
natural_output_level`, but it is constant only after R2 and R17. At 4096 slots,
the minimum input level is 4 for `MaxPool1d(2)`, `MaxPool1d(4)` and
`MaxPool1d(8)` on full slots, and 8 for `MaxPool1d(16)` and `MaxPool1d(64)`.

The text below is the original request.

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

**Status: open at 0e5dde7b.** "The main request" covers this item.

At 0e5dde7b we measured the new queries at 32768 slots, with full slots:

| Shape | `MaxPoolMinimumBootstrapEndLevel` | `MaxPoolMinimumInputLevel` |
|-------|-----------------------------------|----------------------------|
| `MaxPool1d(2)` | 68 ms | 3 ms |
| `MaxPool1d(8)` | 100 ms | 4 ms |
| `MaxPool1d(64)` | 186 ms | 10 ms |

The end-level search builds a new layout, with all the maps, for each end
level from 32 down. So HEIR must keep its cache. This is not a blocker.

`PlanMaxPool` builds every gather and compaction map over all slots, for each
chain, only to count the stages. At 1716ac30, `MaxPoolFoldCopies` also builds
the gather stages once for each copy count that it tries, so a query does more
work than at e8efd441. We did not measure it. At e8efd441, one HEIR depth
query (with the R10 search) took up to 23 ms at 32768 slots. HEIR's level
analyses ask for the same pool many times, so HEIR caches the result for each
shape.

**Request.** Add functions that count the stages without the maps, for example
`MaxPoolNumGatherStages(config, chain)` and
`MaxPoolNumCompactionStages(config, chain)`, and use them in `PlanMaxPool`.
HEIR can then remove its cache.

## R12: Validation without exceptions

**Status: partly done at 468bdaef.** `MaxPoolTryPlanLevels` returns `false`
and does not throw when the levels do not fit. But `PlanMaxPoolLayout`,
`MaxPoolMinimumInputLevel` and `MaxPoolMinimumBootstrapEndLevel` still call
`ValidateMaxPoolConfig`, which throws for a configuration that is not valid.
For example, `MaxPoolMinimumBootstrapEndLevel` on
`NonOverlapping(32768, 1, 4)` throws "MaxPool: bad input_length". So HEIR
keeps `-fexceptions` for its planner wrappers. The request below is still
open.

1716ac30 adds a C interface to the planner
(`planner/include/cyclops_planner.h`). Its functions return an error code and a message, and do not throw. This is
the form that HEIR needs. But the C interface does not cover `MaxPool` yet. A
`MaxPool` function in it, for example for the query in "The main request",
also solves this item.

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

**Status: open at 0e5dde7b.** The default `pad_value` is still `-0.5`, in
`MaxPoolConfig` and in `MaxPool2dConfig`.

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

**Status: open at 0e5dde7b.** This is the largest possible simplification for
HEIR.

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

**Question.** 1716ac30 adds `MaxPool2d`, which has a channel axis. Can HEIR
express a 1-D pool over C channels as a `MaxPool2d` with `height = 1` and
`kernel_height = 1`? We see three limits:

- `MaxPool2dConfig` has no `ceil_mode`, only symmetric `padding`.
- The output uses the Orion layout, not a compact layout.
- The vertical pass possibly costs levels, also with `kernel_height = 1`.

At 0e5dde7b, `MaxPool2d` can run one pass over all kernel taps
(`MaxPool2dStrategy::kSingle`), so the third limit is possibly gone. The first
two limits are still in `MaxPool2dConfig`. So HEIR still needs R14 for
`MaxPool1d`.

## R15: `Compile` without the input scale

**Status: open at 0e5dde7b.** Both constructors of `MaxPool` (`MaxPool.h`
lines 68 and 71) and the constructor of `MaxPool2d` still take the input
scale.

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

**Status: resolved on the HEIR side.**

Cyclops commit 8de7c716 renamed `BootSecretId()` to `NativeSecretId()` on
`ClientContext` and `Parameter`. HEIR now emits `NativeSecretId()`, so its
generated code needs Cyclops 8de7c716 or later. HEIR pins e8efd441.

**Request.** Before a future rename of a public API that HEIR emits, please
keep the old name as a deprecated alias for some releases, so that HEIR can
move to the new name without a break.

## R17: The fold copies must not depend on the end level

**Status: open at 0e5dde7b. HEIR has a workaround with the output level pin.**

**Problem.** `PlanMaxPoolLayout` selects the copy count with
`MaxPoolFoldCopies(config, bootstrap_level)` (`MaxPoolPlanner.cpp` line 272).
A higher copy count is selected only when
`LeagueIndicatorLevel(bootstrap_level, candidates) >= 1` for each round. The
copy count changes the gather and compaction maps, so the number of stages, and
it changes whether the small-window cap of R2 applies (only when
`copies == 1`). So the level drop and the minimum input level of one
configuration can be different at two end levels.

HEIR has a circular dependency here. HEIR calculates the level drop of each
pool first, and it selects the parameters, and so the end level, from those
drops.

**Results at 0e5dde7b.** We tested `MaxPool1d(k)` for k in {2, 3, 4, 5, 8, 9,
16, 32, 64}, at 4096 and 32768 slots, with input lengths of k, 2k, 4k and up
to the slot count, and each end level from the minimum to 32. In 48 shapes, the
copy count changes with the end level. Some examples at 4096 slots:

| Shape | Copies |
|-------|--------|
| `MaxPool1d(16)` on 16 to 256 values | 4 at end level 11 to 13, 16 from 14 |
| `MaxPool1d(32)` on 32 to 128 values | 4 at end level 11 to 16, 32 from 17 |
| `MaxPool1d(64)` on 64 to 512 values | 8 at end level 12 to 13, 4 from 14 |
| `MaxPool1d(64)` on 1024 values | 1 at end level 13, 4 from 14 |

The copy count is not monotonic in the end level: for `MaxPool1d(64)` on 64
values, it goes from 8 down to 4.

In 46 of the 48 shapes, the level drop and the minimum input level do not
change. In 2 shapes they change: `MaxPool1d(64)` on 1024 of 4096 slots, and on
8192 of 32768 slots:

| End level | Copies | Gather stages | Level drop | Minimum input level |
|-----------|--------|---------------|------------|---------------------|
| 13 | 1 | 1 | 4 | 8 |
| 14 or more | 4 | 3 | 6 | 10 |

The fold of small windows (padded window below 8) did not change in our tests.
These windows stay at one copy, so the cap of R2 always applies to them.

**HEIR workaround.** HEIR can call `PlanMaxPoolLayout` and
`MaxPoolTryPlanLevels` for each end level from the minimum end level to 32,
and take the largest level drop and the largest minimum input level. The
generated code then pins `output_level = input_level - D`. For the shape in
the table, the pin with `D = 6` is accepted at each end level from 13 to 20
and each input level from 10 to the end level. At end level 13 the pin is a
free level-down of 2. This workaround costs one level for these shapes at end
level 13, and a search over the end levels in HEIR.

**Request.** Make the copy count a function of the configuration only. Two
options are good for HEIR:

1. A configuration field that sets the copy count, or that disables the fold
   (copy count `MaxPoolRequiredCopies(config)`). This can be the same field as
   in R2.
2. Select the copy count at the minimum end level of the configuration (R3b),
   and use the same count at each higher end level.

With either option, HEIR removes its search over the end levels. With option
2, `MaxPoolMinimumBootstrapEndLevel` must agree with the selected copy count.

**HEIR status.** HEIR uses the workaround above: it takes the largest level
drop and the largest minimum input level over the end levels from the minimum
to 32, and pins the output level in the key planner and in the generated code.
A request for option 1 or 2 is still open.

## Architecture review for the next session

Saved on 2026-10-05 for the session on 2026-10-06. These are proposed changes,
not completed work or approval to implement them.

The main proposal is one complete CPU plan in Cyclops. HEIR consumes a small
requirements summary. Keep the Cyclops adapter in `lib/Target/Cheddar`.

- Export `level_consumption`, `minimum_input_level`, and
  `bootstrap_end_level_floor` together. Define the floor as a threshold above
  which all higher bootstrap end levels satisfy the algorithm's level requirements.
- Keep two steps: analyze the configuration before parameter selection, then
  assign exact levels after parameter selection. Share the rules between both steps.
- Include every intermediate level in the concrete plan: league, selector,
  masks, and selection stages. Key generation and GPU compilation consume this plan.
- Make the execution policy explicit. HEIR needs a constant level drop for
  valid input levels. Remove conditional cap detection after the pinned Cyclops
  revision supports that policy.
- Replace HEIR's search through exceptions and its fixed bootstrap level of
  11 with the requirements query. The earlier standalone formula is only a
  proposal; it does not verify every runtime requirement.
- Keep generic IR verification separate from Cyclops support checks. Validate
  target support before level analysis. Report a compiler diagnostic for an
  unsupported configuration instead of `report_fatal_error`.
- Keep any cached analysis within one compilation. Invalidate it when the
  configuration changes.
- Use a separate output buffer until Cyclops supports input/output aliasing
  across multiple chains.
- Later, move GPU compilation into server preprocessing. Retain the compiled
  object with its context, configuration, level, scale, and policy.

Suggested implementation order:

1. Extract the complete CPU level plan from the Cyclops runtime checks.
2. Make key generation and GPU compilation consume the plan.
3. Export the requirements and remove HEIR's level search and fixed threshold.
4. Move GPU compilation into preprocessing after the correctness tests pass.

Tests must compare planner acceptance with runtime compilation at the selected
boundaries. Execute with exactly the planned keys. Include pools with multiple chains.

Resume from `lib/Target/Cheddar/MaxPoolPlan.cpp` and
`lib/Target/Cheddar/MaxPoolInterfaces.cpp`. The Cyclops checkout is
`/volume2/mdesgros/cyclops-dev/review`. Check both checkout revisions before work.
