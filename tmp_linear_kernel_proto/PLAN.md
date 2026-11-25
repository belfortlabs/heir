# Plan: one linear-transform kernel over strided layouts

Status: draft, 2026-11-24. Base: google/heir main (7a97bfc19). Prototype and
corpus numbers are in this directory (`lt.py`, `test_ops.py`, `test_net.py`,
`corpus_sim.py`, `corpus_16384.tsv`). Line references are to main unless stated.

## 1. Scope

In scope: every structured op that is linear in one secret operand with
cleartext coefficients (all convs, depthwise and grouped convs, sum/avg pools,
matvec, vecmat, ciphertext-plaintext matmul, cleartext scale and bias, secret
broadcasts, layout conversions), elementwise ops, and a decomposition for
everything else. Target: the plaintext backend first, with the 38-model sweep
and a numeric check as the measure.

Out of scope for this plan: CKKS levels and bootstrapping, rolled loops on CKKS
backends other than through a native transform, layout search,
ciphertext-ciphertext matmul (the bicyclic and tricyclic kernels stay as they
are).

No dependency on the Rotom dialect.

## 2. Design summary

**Layout.**
`p(idx) = offset + sum over digits of stride * ((idx[axis] / div) mod m)`,
ciphertext `p div N`, slot `p mod N`, optional period `P` (single ciphertext,
`P | N`). Unused slots are unspecified. A strided layout converts to the
existing relation form, so passes that only understand `tensor_ext.layout`
relations keep working.

**Spec.** `out[outMap(d)] += coeff(d) * in[inMap(d)]` over a box of loop
extents, with reads outside `in` equal to zero. `inMap`/`outMap` are the op's
indexing maps; a zero `tensor.pad` on the input is folded in as a negative
offset.

**Kernel.**

1. Keys `{(ctOut, ctIn, shift)}` at compile time: loop dims with equal position
   coefficients on both sides are skipped, the rest are enumerated, and
   ciphertext boundary crossings along the skipped dims are found
   arithmetically.
1. Diagonals at run time: a rolled loop nest over the op's domain scatters
   `coeff(d)` into `tensor<K x N>`, outlined as a preprocessing helper.
1. Evaluation: `kernel.linear_transform` with per-row input and output
   ciphertext attributes, lowered natively or to a sparse BSGS. A periodic
   output takes shifts mod `P` and adds a rotate-and-reduce. A spec with a
   constant coefficient and a box of shifts uses a rotate-and-sum network with
   no multiplications.

**Layout rule.** Output dims paired with an input dim keep the input's grid
(stride times step). Remaining output dims fill gaps, then new planes. Channels
are re-multiplexed when keeping them in place would leave the result less than
half full. An output with no paired dims gets row-major with period
`nextPow2(size)`.

**Everything else.** Elementwise ops apply slotwise when operand layouts agree;
a disagreeing operand gets an identity spec through the same kernel. A generic
with several secret operands under different maps, or a nonlinear reduction, is
split at the tensor level into access specs plus slotwise ops, with reduction
dims unrolled.

## 3. Work packages

Each package is one PR-sized unit with its own tests. Days are rough.

### WP1. Strided layout utility and attribute (3 d)

- `lib/Utils/Layout/StridedLayout.{h,cpp}`: the struct, `position`,
  `numCiphertexts`, `affineStride(axis)`, injectivity check, `toRelation()` (ISL
  string for `tensor_ext.layout`), transforms for transpose, collapse/expand
  along digit boundaries, unit-stride and strided slices, and
  `rowMajor(shape, N, period)`.
- `#tensor_ext.strided_layout<...>` attribute next to `TensorExt_LayoutAttr`
  (`lib/Dialect/TensorExt/IR/TensorExtAttributes.td:19`), stored under the same
  `kLayoutAttrName`. A free function `getLayoutRelation(Attribute)` replaces
  direct `getIntegerRelation()` calls at the consumers, so both forms are
  accepted.
- Tests: gtest comparing `position` with points of `toRelation()` on small
  shapes; round-trip lit test for the attribute.

### WP2. Spec, keys and reference packing (4 d)

- `lib/Kernel/LinearSpec.h`: `LinearSpec` (loop extents, two `AffineMap`s plus
  offsets, input and output shapes).
- `computeDiagonalKeys(spec, inLayout, outLayout)`: port of `fast_keys` in
  `lt.py`, including the period cases and the per-dimension validity intervals.
- `packDiagonals(spec, layouts, keys, coeffFn)`: plain C++ reference, used by
  tests and by constant folding of small cases.
- `inducedOutputLayout(spec, inLayout)`: port of `induced_layout`, with the
  density rule.
- Tests: gtest port of `test_ops.py` (26 cases) and a fuzz test over random loop
  nests, maps and layouts that checks `keys == brute force` and
  `evaluate(pack) == naive loops` with random values in unused slots. The random
  spec generator of `GenericKernelFuzzTest.cpp` on the old stack is a starting
  point.

### WP3. Block linear transform op and its lowering (4 d)

- `kernel.linear_transform` (`lib/Dialect/Kernel/IR/KernelOps.td:55`): optional
  `input_ciphertexts` and `output_ciphertexts` arrays parallel to
  `diagonal_indices`; the verifier (`KernelOps.cpp:308`) accepts a 2D
  ciphertext-semantic input with more than one row only with these attributes.
  `PrepareLinearTransforms.cpp` and `SecretToCKKS.cpp` reject the block form.
- `lib/Kernel/KernelImplementation.h`:
  `implementSparseBlockBsgs(keys, x, diags, unroll)` on the ArithmeticDag (baby
  steps shared per input ciphertext, giant step chosen by rotation count over
  powers of two, diagonals pre-rotated in the packing helper), built from the
  `ForLoopNode`/`ExtractNode`/`InsertNode` pieces of the existing rolled BSGS
  (`:315`). `implementUniformBoxSum` for the constant-coefficient case.
- New pass `implement-linear-transforms` after `implement-rotate-and-reduce`
  (`ArithmeticPipelineRegistration.cpp`, in
  `mlirToSecretArithmeticPipelineBuilder`): rolled on targets without
  `hasKernelLinearTransform`, split into one-block ops on Lattigo.
- Tests: DAG fuzz test with `EvalVisitor` against a dense reference, for every
  giant step; lit tests for op and pass.

WP1, WP2 and WP3 are independent of each other.

Status 2026-11-24: WP1-3 are written and pass under bazel. Differences from the
text above: the block transform is a separate op,
`kernel.block_linear_transform`, because `kernel.linear_transform` is mapped
elementwise over tensors of ciphertexts and a block mode on the same op would
contradict that; so no guards were needed in `PrepareLinearTransforms.cpp` or
`SecretToCKKS.cpp`. The DAG emission is in
`lib/Kernel/LinearTransformImplementation.h`, and `LinearSpec` is in
`lib/Utils/Layout`. The existing consumers of `tensor_ext.layout` are not yet
switched to `getLayoutRelation`; that moves to WP4.

### WP4. Pipeline integration behind a flag (6 d)

Depends on WP1-3. Flag `linear-kernel` on layout propagation and the conversion
pass, off by default, so existing tests are unaffected.

- `LinearSpec::fromLinalgOp`: uses `getIndexingMapsArray`, iterator types and
  the body. Accepts a body that is linear in one secret block argument with a
  public coefficient (multiplication or division by public values, negation,
  casts, sums of linear terms) and an add or no reduction. The coefficient is
  recorded as a list of cleartext operand reads and constants.
- Layout propagation (`LayoutPropagation.cpp:476`): one visit for any `LinalgOp`
  that yields a spec, calling `inducedOutputLayout`; it replaces the per-op
  visits when the flag is on. Cleartext operands of such ops get no layout.
  Function arguments get `rowMajor` on the tensor's own grid.
- Conversion pass (`ConvertToCiphertextSemantics.cpp`): one pattern that
  computes keys, emits the packing helper (an `scf.for` nest with the body's
  public part cloned, outlined like `_assign_layout_<hash>` so preprocessing
  hoists it), the transform op, and the period fold. Keys above a budget
  (default 65,536 per op) are an error naming the op, both layouts and the
  count.
- `assign_layout` and `unpack` for strided layouts as closed-form loops, without
  ISL code generation (`AssignLayout.cpp:398`).
- Elementwise ops and `tensor.pad` (folded into the consumer, or a one-diagonal
  mask when the consumer is not linear).
- Numeric check: a runner that compares plaintext-backend output with the same
  module lowered through upstream linalg-to-LLVM, on random inputs.
- Tests: lit tests for conv (stride 1 and 2, padding, dilation), depthwise,
  grouped, pool, matvec; plaintext e2e tests for a strided residual block.

Status 2026-11-24 (in progress). Differences from the text above:

- Propagation is a separate pass, `strided-layout-propagation`
  (`lib/Transforms/LinearKernel/`), selected by the pipeline option
  `linear-kernel=true`. `layout-propagation` stores `LayoutAttr` relations in
  every visit and compatibility check, so a flag inside it would have meant a
  second branch at each of those sites. The new pass uses no ISL.
- `matchLinearOp` (`LinearOp.h`) reads any structured linalg op, or a zero
  `tensor.pad`, as a `LinearSpec` plus a coefficient emitter. A zero pad whose
  users are all linear ops is folded into them, also through reshapes that only
  add or remove unit dims (the form DropUnitDims leaves behind): the pad and
  those reshapes keep the source's layout attribute, and the consumer's spec
  gets a negative input offset.
- The conversion pass is shared. When a function argument carries a strided
  layout it registers only the strided patterns
  (`ConvertToCiphertextSemantics/StridedPatterns.cpp`). The diagonal packing
  loop nest is emitted inline in the secret region, not outlined.
- Diagonal rows are found at run time through a constant lookup table of
  `numOut * numIn * modulus` entries, capped at 2^26. WP6 needs a two-level
  table or a search for many-ciphertext tensors.
- `assign_layout` and `unpack` for strided layouts are closed-form loops in
  `AssignLayout.cpp`; the client interface uses them unchanged.
- `tensor.extract_slice` with static bounds (the stride-2 shortcut of ResNet-32)
  is handled already, as a relabeling through `StridedLayout::slice`.
- The rotate-merging canonicalization patterns in `TensorExtCanonicalization.td`
  added two shift amounts without checking that their types agree, which
  produced invalid IR once `index` and `i64` shifts met. They now require equal
  types.
- Numeric check: `check.sh` + `gen_driver.py` in this directory. The reference
  is the same pipeline cut before layout propagation, followed by
  `secret-forget-secrets` and `heir-polynomial-to-llvm`. It is a script run on
  the build box, not a bazel test.

Milestone A result (16,384 slots, plaintext backend, fastbuild heir-opt):

| Model         | heir-opt | max abs error vs reference                                                     |
| ------------- | -------- | ------------------------------------------------------------------------------ |
| BLeNet5       | 10 s     | 3e-8                                                                           |
| BCryptoNets   | 5 s      | 3e-8                                                                           |
| CLeNetRelu44K | 9 s      | 4e-8                                                                           |
| CLeNetRelu62K | 10 s     | 3e-8                                                                           |
| CLeNetTanh62K | 9 s      | 3e-8                                                                           |
| CMnistMLP269K | 5 s      | 1e-7                                                                           |
| BResNet8      | 15 s     | 7e-8                                                                           |
| BResNet32     | 42 s     | 2e-8                                                                           |
| CResNet20     | 25 s     | 3e-8                                                                           |
| BWRN16_1      | 22 s     | no verdict: the reference output is non-finite for random batch-norm arguments |

None of the canonicalization PRs on the old stack (3549, 3551, 3553, 3554) was
needed.

Open in WP4: lit coverage of the conversion output beyond matvec; the packing
loops are not outlined into preprocessing helpers; grouped convolution is
untested.

**Milestone A:** with the flag on, the 6 LeNet/MLP/CryptoNets models and
BResNet8, BResNet32, BWRN16_1 and CResNet20 compile in under 2 minutes each and
pass the numeric check. Whether the canonicalization fixes on the old stack (PRs
3549, 3551, 3553, 3554) are prerequisites is determined here.

### WP5. Remaining op surface (5 d)

- Layout-only ops through `StridedLayout` transforms: transpose,
  collapse/expand, extract_slice. Fallback when the result leaves the family:
  identity spec into `rowMajor`.
- Joins: when operand layouts differ, convert the cheaper side with an identity
  spec; the cost is the key count.
- Cleartext broadcast operands (bias, scale): packed by the same helper with a
  one-diagonal spec. Secret broadcasts (squeeze-excite) and
  `insert_slice`/concat as identity specs.
- Decomposition pass for generics that yield no spec: split per secret operand,
  unroll nonlinear reduction dims (max pool as taps plus a max tree, using the
  existing ReLU-based max).
- Remove the flag's restriction to the supported set: unsupported ops report a
  diagnostic naming the op and the reason.

**Milestone B:** all 15 models whose activations fit in at most 5 ciphertexts
(`features.tsv` rows up to CResNet18Cifar) pass the numeric check.

**WP5 status (2026-11-24): done, uncommitted. Milestone B reached.**

What was built:

- `matchLinearOp` also accepts `tensor.collapse_shape`, `tensor.expand_shape`,
  static `tensor.extract_slice` (with dropped unit dimensions) and both operands
  of `tensor.insert_slice`. `tensor.concat` needs no handling of its own:
  `ApplyFolders` decomposes it into `insert_slice` chains before layout
  propagation.
- `relabeledLayout(op, layout)`: reshapes, slices, transposes and any one-input
  linalg op that permutes axes keep the operand's ciphertexts when the result
  has a strided form. Otherwise the op falls through to `visitLinear` and
  becomes one transform into the induced layout. The conversion has a single
  pattern for both (`ConvertStridedTensorOp`) and one for `insert_slice`.
- A zero pad folds into a linear consumer through any `expand_shape` and through
  unit-dim collapses (`isViewReshape`, matrix-valued `View`). This is what
  grouped convolutions need:
  `pad -> expand_shape (split groups) -> conv_2d_ngchw_gfchw`.
- Joins: `visitSlotwise` converts every secret operand to the operand layout
  with the lowest total diagonal count.
- `decompose-max-pool` (new pass, first in `linalgPreprocessingBuilder`):
  `linalg.pooling_nchw_max` becomes one tap generic per window position, a mask
  for taps that read padding, and a balanced tree of `b + maximumf(a - b, 0)`
  with the domain attributes of the pooling op.
- The zero diagonals tensor of a transform is a run-time `linalg.fill`, not a
  constant (constants made a 2.5 GB object file for CResNet18Cifar).

Not built, because no corpus model reached it: a cleartext affine term inside
one generic body (`x * scale + bias` with `bias` not the `outs` operand) and the
general split of a generic with several secret operands. Both report
`is not a linear operation`.

Milestone B result (16,384 slots, plaintext backend, fastbuild heir-opt). The
five new models and the WP4 models all match the reference; BWRN16_1 now has a
verdict.

| Model                                                                            | heir-opt | secret input range | max abs error vs reference | max abs reference |
| -------------------------------------------------------------------------------- | -------- | ------------------ | -------------------------- | ----------------- |
| BLeNet5, BCryptoNets, CLeNetRelu44K, CLeNetRelu62K, CLeNetTanh62K, CMnistMLP269K | 5-10 s   | \[-1, 1\]          | 3e-8 to 7e-8               | 0.05-0.21         |
| BResNet8                                                                         | 15 s     | \[-1, 1\]          | 2e-8                       | 0.15              |
| BResNet32                                                                        | 42 s     | \[-1, 1\]          | 2e-8                       | 0.16              |
| CResNet20                                                                        | 25 s     | \[-1, 1\]          | 3e-8                       | 0.23              |
| BWRN16_1                                                                         | 22 s     | \[-1, 1\]          | 4e-8                       | 0.12              |
| BDSCNN                                                                           | 16 s     | \[-1, 1\]          | 3e-8                       | 0.15              |
| BMobileNetV1_025                                                                 | 34 s     | \[-1, 1\]          | 3e-8                       | 0.06              |
| BTestEfficientNet                                                                | 31 s     | \[-1, 1\]          | 9e-8                       | 0.09              |
| CResNet18Cifar                                                                   | 66 s     | \[-0.1, 0.1\]      | 9e-8                       | 0.27              |
| CRepVGGA0                                                                        | 170 s    | \[-0.1, 0.1\]      | 1.5e-4                     | 20.6              |
| AResNet18_32 (not in Milestone B)                                                | 45 s     | \[-0.1, 0.1\]      | 1.5e-7                     | 0.17              |
| ARegNetY400MF_32 (not in Milestone B)                                            | 161 s    | \[-1, 1\]          | 1.2e-6                     | 0.04              |

The heir-opt times of the first ten models are the WP4 measurements. For
CRepVGGA0 the reference moves by 1e-3 for a 1e-6 relative change of the input,
so its error of 1.5e-4 is rounding.

Tests: 58 targeted tests and the full suite (1088 of 1088) pass on the final
state.

Beyond Milestone B: AResNet18_32 (max pool) and ARegNetY400MF_32 (grouped
convolutions, squeeze-excite) compile; see the table.

Changes to the numeric check (`check.sh`, `gen_driver.py`):

- The compiled functions write into their cleartext argument buffers, so the
  driver restores every argument before each call. Without this the second call
  saw other batch-norm statistics, which looked like a mismatch of about 1 % on
  BWRN16_1 and CRepVGGA0.
- Batch-norm statistics are drawn as mean in \[-0.1, 0.1\] and variance in
  \[0.8, 1.2\].
- With inputs in \[-1, 1\] the polynomial activations of the deeper models
  overflow in the reference itself. `check.sh` retries with inputs in \[-0.1,
  0.1\], and the driver reports how far the reference moves for a 1e-6 relative
  input change.

Open after WP5:

- `linalg-fuse-linear-ops` (upstream, on main) folds `conv(x, w) * s` into
  `conv(x, w * s)` without scaling a non-zero `outs` operand, so
  `(conv(x, w) - mean) * s` loses the factor on the mean. It fires when the
  scale traces back to a function argument or constant through broadcasts only;
  the corpus models compute the scale from the variance and are not affected.
  The reference of the numeric check shares this pass, so the check cannot see
  it. By reading the code, `c - conv(x)` and `c / conv(x)` are also fused as if
  the linalg op were on the left.
- `compare-to-sign-rewrite` asserts on tensor-typed `arith.cmpf` (upstream). It
  stops AGhostNet_32, AMobileNetV2_32, AMobileNetV3Small_32 and
  AMobileNetV3Large_32 before they reach the linear kernel.
- CRepVGGA0 and CRepVGGA1 take 170 s and 175 s in heir-opt, of which 62 s is
  `LinalgCanonicalizations` and 50-60 s the conversion. The 2-minute mark is
  missed.
- Lowering the larger models to an object file takes 23 minutes each
  (`mlir-translate` and `llc` on 300 MB of LLVM IR). This is the check harness
  and the size of the packing loops, and belongs to the outlining item left open
  in WP4.
- BHonkRes15 (12 ciphertexts) needs more than 65,536 diagonals for its first
  reshape: WP6.

### WP6. Multi-ciphertext scale (5 d)

- Pre-pass choosing the input grid so that every layer's `extent * gap` fits
  (avoids the odd-size regrid).
- Slotwise ops and polynomial activations over `tensor<n x N>`; check that later
  passes (secret-distribute-generic, bufferization) stay linear in IR size on
  rolled loops.
- Diagonal memory on the plaintext backend: packing immediately before use and
  freed after, so peak is one layer's `K x N`.
- Sweep at 16,384 and 32,768 slots with per-pass time, peak memory and the
  numeric check.

**Milestone C:** at least 30 of 38 pass, each under 10 minutes and 60 GB; every
remaining failure is listed by op.

**WP6 status (2026-11-25): done, Milestone C reached.**

What was built:

- **Aligned planes.** A tensor that needs several ciphertexts gets a power of
  two as the stride of its outermost axis (`StridedLayout::aligned`), and
  `inducedOutputLayout` rounds the stride of new planes the same way. Channel
  planes then start at the same slot of every ciphertext, so the shift between
  two channels takes `N / stride` values instead of one per channel offset.
  BHonkRes15's 45-to-45-channel convolutions went from 801 distinct shifts to
  36\. This replaces the planned input-grid pre-pass.
- **Rotation-only sums and copies** (`planTransform`, `factorBoxSum`,
  `factorBoxCopy` in `LinearSpec`). A unit-coefficient operation whose windows
  are boxes inside one ciphertext is a logarithmic number of rotations and
  additions with no plaintext multiplication: sum pools and reductions sum the
  input first, broadcasts copy the result afterwards. A global pool that needed
  48,480 diagonals needs none.
- **Join partner layouts.** A linear result that is only combined slot by slot
  with another secret tensor is produced in that tensor's layout when that costs
  no more diagonals than converting afterwards. A squeeze-excite scale is then
  copied over the planes it multiplies by rotations; before, the conversion at
  that join cost up to 200,704 diagonals.
- **Dilations and pools stay native.** `linalg-canonicalizations` has an option
  `densify-convolutions`, off for the linear kernel: dilated convolutions are
  not rewritten to filters with zeros (a dilation of 16 made a 33x33 filter,
  821,953 diagonals), and sum pools are not rewritten to convolutions over all
  channel pairs.
- **Clamps.** `activation-canonicalizations` rewrites `select(a > c, a, c)` and
  `select(a < c, a, c)` with any ordered or unordered predicate to
  `maximumf`/`minimumf`. ReLU6 and hard-sigmoid reached
  `compare-to-sign-rewrite` before, which asserts on tensor comparisons.
- **Divisions by cleartext** become multiplications by the reciprocal in
  `strided-layout-propagation`. The packed divisor is zero in unused slots.

Bugs found by the numeric check and fixed:

- A transform from a periodic tensor to one with a longer period wrote only the
  first copy of its result. Readers that take another copy got zeros. This was
  present since WP4 and is why squeeze-excite models were off by up to 0.7 %;
  the old threshold of 1e-3 absolute hid it for small outputs.
- A rotation-only sum leaves partial sums in unused slots. They are now only
  left there when every reader is a linear operation (`isOnlyReadLinearly`); a
  polynomial activation applied to them overflowed.

Milestone C result (16,384 slots, plaintext backend, fastbuild heir-opt):

All 38 models compile and match the reference. heir-opt takes 6-201 s and at
most 2.5 GB; the limits were 10 minutes and 60 GB.

| Model                                                  | Ciphertexts | heir-opt  | Peak memory | Max abs error  | Max abs reference |
| ------------------------------------------------------ | ----------- | --------- | ----------- | -------------- | ----------------- |
| 6 LeNet/MLP/CryptoNets models                          | 1           | 6-32 s    | 0.1 GB      | 2e-8 to 7e-8   | 0.05-0.21         |
| BResNet8, BResNet32, CResNet20, BWRN16_1               | 1           | 24-41 s   | 0.1 GB      | 1e-8 to 4e-8   | 0.12-0.23         |
| BDSCNN, BMobileNetV1_025, BTestEfficientNet            | 1-3         | 31-45 s   | 0.1 GB      | 3e-8           | 0.06-0.15         |
| BHonkRes15                                             | 12          | 43 s      | 0.2 GB      | 3e-7           | 0.33              |
| BBCResNet8                                             | 16          | 69 s      | 0.4 GB      | 4e-8           | 0.10              |
| CResNet18Cifar                                         | 4           | 64 s      | 0.3 GB      | 9e-8           | 0.27              |
| CRepVGGA0                                              | 3           | 74 s      | 0.3 GB      | 1.5e-4         | 20.6              |
| CRepVGGA1                                              | 4           | 86 s      | 0.4 GB      | 3.6e28         | 1.6e33            |
| AResNet18_32                                           | 1           | 51 s      | 0.3 GB      | 1.5e-7         | 0.17              |
| AMNASNet_32, AMobileNetV2_32, AMobileNetV4ConvSmall_32 | 1-2         | 59-69 s   | 0.2-0.5 GB  | 4e-8 to 2.5e-6 | 0.04-1.5          |
| AConvNeXtAtto_32                                       | 1           | 100 s     | 0.6 GB      | 6e-7           | 0.42              |
| ARegNetY400MF_32                                       | 1           | 153 s     | 1.4 GB      | 2e-8           | 0.04              |
| AEfficientNetB0_32                                     | 2           | 141 s     | 1.4 GB      | 4e-20          | 3.8e-14           |
| AGhostNet_32 (domain 8)                                | 1           | 145 s     | 1.0 GB      | 1.5e-7         | 0.25              |
| AMobileNetV3Small_32, AMobileNetV3Large_32 (domain 16) | 1           | 102-118 s | 1.0-1.2 GB  | 1e-6 to 2e-6   | 1.2-2.3           |
| AMNASNet                                               | 37          | 86 s      | 0.8 GB      | 2e-7           | 0.22              |
| AMobileNetV2                                           | 74          | 91 s      | 1.3 GB      | 1e-7           | 0.07              |
| AMobileNetV4ConvSmall                                  | 25          | 80 s      | 0.6 GB      | 8e-8           | 0.08              |
| AConvNeXtAtto                                          | 31          | 118 s     | 0.8 GB      | 3e-7           | 0.45              |
| AResNet18 (domain 16)                                  | 49          | 123 s     | 1.2 GB      | 4e-6           | 4.3               |
| AMobileNetV3Small (domain 16)                          | 16          | 117 s     | 1.1 GB      | 3e-6           | 2.2               |
| AMobileNetV3Large (domain 16)                          | 49          | 143 s     | 1.6 GB      | 4e-6           | 3.5               |
| AGhostNet (domain 8)                                   | 37          | 189 s     | 1.7 GB      | 7e-7           | 0.61              |
| ARegNetY400MF                                          | 37          | 190 s     | 2.0 GB      | 4e-8           | 0.06              |
| AEfficientNetB0                                        | 109         | 201 s     | 2.5 GB      | 3e-20          | 1.7e-14           |

Ciphertext counts are those of the row-major corpus statistics; aligned planes
need up to 30 % more. "Domain N" marks a corpus variant in which the
compare-and-select activations carry the approximation domain \[-N, N\]; without
it the reference of these seven models overflows. Three results say less than
the others: the outputs of CRepVGGA1 are of the order 1e33 and those of the two
EfficientNetB0 models of the order 1e-14, so the polynomial activations are far
outside their domain or the signal has died out. They agree with the reference
to a relative 2e-5 and 2e-6.

Also checked at 256 slots, where every plane spans several ciphertexts: 10 small
models and a minimal squeeze-excite model match.

At 32,768 slots all 38 models compile (6-248 s, at most 1.8 GB); that sweep was
run before the last two fixes and without a numeric check.

Tests: the full suite passes on the final state (1091 of 1091).

Changes to the numeric check: `pcheck.sh` runs models in parallel; a match is an
error of at most 1e-4 of the largest output; the driver also measures how far
the compiled result moves for a 1e-6 input change; `add_domains.py` writes
corpus variants with approximation domains for models whose reference overflows
on the default domain \[-1, 1\] (hard-swish, hard-sigmoid, ReLU at 224x224).

Open after WP6:

- The join in AGhostNet between a gapped and a compact layout of a
  `[24, 56, 56]` tensor costs 60,048 diagonals. It compiles and matches, but a
  layout choice that avoids the conversion would be much cheaper.
- Packing loops are still emitted inline per transform. Lowering a full-size
  model to an object file takes 15-30 minutes in the check harness.
- No numeric check was run at 32,768 slots; all 38 models compile there.
- `linalg-fuse-linear-ops` and `compare-to-sign-rewrite` still have the upstream
  defects listed under WP5.

### WP7. Make it the default and remove the old paths (4 d)

- Flag on by default; delete the conv Toeplitz relations
  (`lib/Utils/Layout/Convolution.cpp`), the conv and matvec Halevi-Shoup
  patterns and their layout-propagation visits, and the
  `VecmatDiagonal`/`MatmulDiagonal` kernel names that have no conversion
  pattern.
- Existing Lattigo and OpenFHE e2e tests pass through the new path (unrolled
  sparse BSGS where no native transform exists).
- Close or park google/heir PRs 3536 and 3546.

## 4. Risks

- **Later passes on rolled secret loops.** Loops carrying `tensor<K x N>` and
  `tensor<n x N>` through secret-distribute-generic and bufferization are
  lightly exercised. WP4 finds this early on small models.
- **CKKS backends.** OpenFHE and Cheddar get unrolled BSGS, which is fine for
  CIFAR-size models (about 1k-4k rotations) and too much IR at ImageNet size. A
  rolled form there needs runtime rotation indices and loop-invariant levels;
  not planned here.
- **Level cost.** Every transform, including pools and layout conversions with
  masks, uses one level. The uniform path avoids it for pools.
- **Diagonal volume.** About 480k N-vectors for ResNet-18 at 224. Real backends
  need per-layer preparation or disk-backed storage.
- **Concat and non-nesting reshapes.** Each falls back to an identity spec;
  GhostNet is the model to watch.

## 5. Decisions needed

1. Attribute name and whether it lives in `tensor_ext` (assumed above).
1. Where this lands first: google/heir as a PR stack, or the belfortlabs fork.
1. Whether WP7's deletions are acceptable upstream, given the conv path's
   authorship.
1. Key budget default (65,536 per op assumed).

<!-- mdformat global-off -->
