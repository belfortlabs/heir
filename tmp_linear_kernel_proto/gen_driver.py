"""Writes a C++ driver for one corpus model.

The driver runs the model compiled two ways on the same random arguments and
compares the results: through the plaintext backend with the linear kernel
(encrypt, evaluate, decrypt), and as a plain tensor program lowered to loops
(the same symbols with a `ref_` prefix).

usage: gen_driver.py model.mlir > driver.cpp
"""

import re
import sys

src = open(sys.argv[1]).read()
sig = re.search(
    r'func\.func @(\w+)\((.*?)\) -> tensor<([\dx]*?)x?(f32|f64|i64|i32)>',
    src,
    re.S,
)
fn, args, out_dims, out_ty = (
    sig.group(1),
    sig.group(2),
    sig.group(3),
    sig.group(4),
)
ctype = {'f32': 'float', 'f64': 'double', 'i64': 'int64_t', 'i32': 'int32_t'}

params = []
for m in re.finditer(
    r'%(\w+): tensor<([\dx]*?)x?(f32|f64|i64|i32)>( \{secret\.secret\})?', args
):
  dims = [int(d) for d in m.group(2).split('x') if d]
  params.append((m.group(1), dims, m.group(3), bool(m.group(4))))
secret = [p for p in params if p[3]]
assert len(secret) == 1, 'expected exactly one secret argument'
out_shape = [int(d) for d in out_dims.split('x') if d]


def memref(ty, rank):
  return f'MemRef<{ctype[ty]}, {rank}>'


lines = ["""#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#ifndef VAR_LO
#define VAR_LO 0.8
#define VAR_HI 1.2
#endif
#ifndef MEAN_HI
#define MEAN_HI 0.1
#endif
#ifndef SECRET_HI
#define SECRET_HI 1.0
#endif

template <typename T, int N>
struct MemRef {
  T* basePtr;
  T* data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};
template <typename T>
struct MemRef<T, 0> {
  T* basePtr;
  T* data;
  int64_t offset;
};
using Ct = MemRef<float, 2>;
"""]
out = memref(out_ty, len(out_shape))
fhe_params = ', '.join(
    ('Ct*' if s else memref(t, len(d)) + '*') for _, d, t, s in params
)
ref_params = ', '.join(memref(t, len(d)) + '*' for _, d, t, s in params)
index = params.index(secret[0])
lines.append(f"""extern "C" {{
void _mlir_ciface_{fn}(Ct* res, {fhe_params});
void _mlir_ciface_{fn}__encrypt__arg{index}(Ct* res, {memref(secret[0][2], len(secret[0][1]))}* x);
void _mlir_ciface_{fn}__decrypt__result0({out}* res, Ct* x);
void _mlir_ciface_ref_{fn}({out}* res, {ref_params});
}}

int main() {{
  std::mt19937 rng(7);
  // Cleartext arguments are batch-norm statistics here. A running mean is
  // read through a tensor.expand_shape; a running variance goes through a
  // square root first and has to be positive.
  std::uniform_real_distribution<double> secretDist(-SECRET_HI, SECRET_HI);
  std::uniform_real_distribution<double> meanDist(-MEAN_HI, MEAN_HI);
  std::uniform_real_distribution<double> varDist(VAR_LO, VAR_HI);""")
for name, dims, ty, is_secret in params:
  size = 1
  for d in dims:
    size *= d
  strides, stride = [], 1
  for d in reversed(dims):
    strides.insert(0, stride)
    stride *= d
  is_mean = re.search(r'tensor\.expand_shape %' + name + r'\b', src) is not None
  value = (
      '0'
      if ty.startswith('i')
      else (
          'secretDist(rng)'
          if is_secret
          else 'meanDist(rng)'
          if is_mean
          else 'varDist(rng)'
      )
  )
  lines.append(f'  std::vector<{ctype[ty]}> {name}_master({size});')
  lines.append(f'  for (auto& v : {name}_master) v = {value};')
  lines.append(f'  std::vector<{ctype[ty]}> {name}_data = {name}_master;')
  if dims:
    lines.append(
        f'  {memref(ty, len(dims))} {name}{{{name}_data.data(),'
        f' {name}_data.data(), 0, {{{", ".join(map(str, dims))}}},'
        f' {{{", ".join(map(str, strides))}}}}};'
    )
  else:
    lines.append(
        f'  {memref(ty, 0)} {name}{{{name}_data.data(), {name}_data.data(),'
        ' 0};'
    )
s = secret[0][0]
out_size = 1
for d in out_shape:
  out_size *= d
lines.append(f"""
  // The compiled functions may write into their argument buffers, so every
  // call gets the original values back first.
  auto reset = [&]() {{
{chr(10).join(f"    {p[0]}_data = {p[0]}_master;" for p in params)}
  }};

  {out} expected;
  _mlir_ciface_ref_{fn}(&expected, {", ".join("&" + p[0] for p in params)});

  // How far the reference moves when the secret input changes by one part
  // in a million: an error of that size says nothing about the compiler.
  reset();
  for (auto& v : {s}_data) v *= 1.000001f;
  {out} moved;
  _mlir_ciface_ref_{fn}(&moved, {", ".join("&" + p[0] for p in params)});
  reset();
  double sensitivity = 0;
  for (int64_t i = 0; i < {out_size}; ++i) {{
    sensitivity = std::fmax(sensitivity,
        std::fabs(moved.data[moved.offset + i] - expected.data[expected.offset + i]));
  }}

  auto start = std::chrono::steady_clock::now();
  Ct enc, res;
  {out} actual;
  _mlir_ciface_{fn}__encrypt__arg{index}(&enc, &{s});
  _mlir_ciface_{fn}(&res, {", ".join(("&enc" if p[3] else "&" + p[0]) for p in params)});
  _mlir_ciface_{fn}__decrypt__result0(&actual, &res);
  double seconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start).count();

  // The same for the compiled result. It evaluates the activation
  // polynomials in another order than the reference, and in single precision
  // that evaluation can be far noisier than the reference's.
  reset();
  for (auto& v : {s}_data) v *= 1.000001f;
  Ct encMoved, resMoved;
  {out} actualMoved;
  _mlir_ciface_{fn}__encrypt__arg{index}(&encMoved, &{s});
  _mlir_ciface_{fn}(&resMoved, {", ".join(("&encMoved" if p[3] else "&" + p[0]) for p in params)});
  _mlir_ciface_{fn}__decrypt__result0(&actualMoved, &resMoved);
  double fheSensitivity = 0;
  for (int64_t i = 0; i < {out_size}; ++i) {{
    fheSensitivity = std::fmax(fheSensitivity,
        std::fabs(actualMoved.data[actualMoved.offset + i] - actual.data[actual.offset + i]));
  }}

  double maxError = 0, maxValue = 0;
  int nanRef = 0, nanFhe = 0;
  for (int64_t i = 0; i < {out_size}; ++i) {{
    double e = expected.data[expected.offset + i];
    double a = actual.data[actual.offset + i];
    nanRef += !std::isfinite(e);
    nanFhe += !std::isfinite(a);
    if (!std::isfinite(a) || !std::isfinite(e)) maxError = INFINITY;
    maxError = std::fmax(maxError, std::fabs(e - a));
    maxValue = std::fmax(maxValue, std::fabs(e));
  }}
  // A match is an error of at most 1e-4 of the largest output, or of the
  // order of what a 1e-6 change of the input does to either result.
  double noise = std::fmax(sensitivity, fheSensitivity);
  bool ok = maxError <= 1e-4 * maxValue + 1e-6;
  bool noisy = !ok && std::isfinite(maxError) && maxError <= 20 * noise;
  if (nanRef == {out_size}) {{
    std::printf("NO-REFERENCE: every reference output is non-finite "
                "(%d of them in the compiled result)\\n", nanFhe);
    return 2;
  }}
  std::printf("%s max|ref|=%.4g max|err|=%.3g sens ref/fhe=%.3g/%.3g nonfinite ref/fhe=%d/%d run=%.1fs\\n",
              ok ? "MATCH" : noisy ? "ILL-CONDITIONED" : "MISMATCH", maxValue,
              maxError, sensitivity, fheSensitivity, nanRef, nanFhe, seconds);
  return ok ? 0 : 1;
}}""")
print('\n'.join(lines))
