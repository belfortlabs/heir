"""Writes C++ drivers for the ReLU approximation experiment.

usage: relu_driver.py model.mlir calib > calib_driver.cpp
       relu_driver.py model.mlir eval VARIANT... > eval_driver.cpp

calib: runs `calib_<fn>` (the function after relu-calibration) on N random
inputs and writes the combined statistics (first half maxima, second half
means) to a file.  argv: N out.txt
eval: runs `exact_<fn>` and `<variant>_<fn>` on M other random inputs and
prints the output error of every variant.  argv: M
"""

import re
import sys

src = open(sys.argv[1]).read()
mode = sys.argv[2]
variants = sys.argv[3:]
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
assert len(secret) == 1
out_shape = [int(d) for d in out_dims.split('x') if d]
out_size = 1
for d in out_shape:
  out_size *= d


def memref(ty, rank):
  return f'MemRef<{ctype[ty]}, {rank}>'


out = memref(out_ty, len(out_shape))
plist = ', '.join(memref(t, len(d)) + '*' for _, d, t, s in params)
L = ["""#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
extern "C" {"""]
if mode == 'calib':
  L.append(f'void _mlir_ciface_calib_{fn}(MemRef<float, 1>* res, {plist});')
else:
  for v in ['exact'] + variants:
    L.append(f'void _mlir_ciface_{v}_{fn}({out}* res, {plist});')
L.append("""}

int main(int argc, char** argv) {
  int count = std::atoi(argv[1]);
  // Cleartext arguments (batch-norm statistics) are the same in every run.
  std::mt19937 fixedRng(7);
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
      if ty.startswith('i') or is_secret
      else 'meanDist(fixedRng)'
      if is_mean
      else 'varDist(fixedRng)'
  )
  L.append(f'  std::vector<{ctype[ty]}> {name}_master({size});')
  L.append(f'  for (auto& v : {name}_master) v = {value};')
  L.append(f'  std::vector<{ctype[ty]}> {name}_data = {name}_master;')
  if dims:
    L.append(
        f'  {memref(ty, len(dims))} {name}{{{name}_data.data(),'
        f' {name}_data.data(), 0, {{{", ".join(map(str, dims))}}},'
        f' {{{", ".join(map(str, strides))}}}}};'
    )
  else:
    L.append(
        f'  {memref(ty, 0)} {name}{{{name}_data.data(), {name}_data.data(),'
        ' 0};'
    )
s = secret[0][0]
call_args = ', '.join('&' + p[0] for p in params)
resets = '\n'.join(f'    {p[0]}_data = {p[0]}_master;' for p in params)
L.append(f"""
  // The compiled functions may write into their argument buffers.
  auto reset = [&]() {{
{resets}
  }};
  auto draw = [&](int seed) {{
    std::mt19937 rng(seed);
    for (auto& v : {s}_master) v = secretDist(rng);
  }};
""")
if mode == 'calib':
  L.append(f"""  std::vector<double> stats;
  for (int i = 0; i < count; ++i) {{
    draw(1000 + i);
    reset();
    MemRef<float, 1> res;
    _mlir_ciface_calib_{fn}(&res, {call_args});
    int64_t n = res.sizes[0];
    if (stats.empty()) stats.assign(n, 0.0);
    for (int64_t j = 0; j < n; ++j) {{
      double v = res.data[res.offset + j * res.strides[0]];
      if (j < n / 2) stats[j] = std::fmax(stats[j], v);
      else stats[j] += v / count;
    }}
  }}
  FILE* f = std::fopen(argv[2], "w");
  for (double v : stats) std::fprintf(f, "%.9g\\n", v);
  std::fclose(f);
  std::printf("wrote %zu statistics\\n", stats.size());
  return 0;
}}""")
else:
  names = ', '.join(f'"{v}"' for v in variants)
  cases = '\n'.join(
      f'        case {i}: _mlir_ciface_{v}_{fn}(&actual, {call_args}); break;'
      for i, v in enumerate(variants)
  )
  L.append(f"""  const int V = {len(variants)};
  const char* names[] = {{{names}}};
  std::vector<double> maxErr(V, 0), sumRel(V, 0);
  std::vector<int> agree(V, 0), nonfinite(V, 0);
  double maxExact = 0;
  for (int i = 0; i < count; ++i) {{
    draw(2000 + i);
    reset();
    {out} expected;
    _mlir_ciface_exact_{fn}(&expected, {call_args});
    std::vector<double> e({out_size});
    int best = 0;
    double norm = 0;
    for (int64_t j = 0; j < {out_size}; ++j) {{
      e[j] = expected.data[expected.offset + j];
      if (e[j] > e[best]) best = j;
      norm = std::fmax(norm, std::fabs(e[j]));
    }}
    maxExact = std::fmax(maxExact, norm);
    for (int v = 0; v < V; ++v) {{
      reset();
      {out} actual;
      switch (v) {{
{cases}
      }}
      int bestActual = 0;
      double err = 0;
      bool finite = true;
      for (int64_t j = 0; j < {out_size}; ++j) {{
        double a = actual.data[actual.offset + j];
        if (!std::isfinite(a)) finite = false;
        if (a > actual.data[actual.offset + bestActual]) bestActual = j;
        err = std::fmax(err, std::fabs(a - e[j]));
      }}
      if (!finite) {{
        nonfinite[v]++;
        continue;
      }}
      maxErr[v] = std::fmax(maxErr[v], err);
      sumRel[v] += err / norm / count;
      agree[v] += bestActual == best;
    }}
  }}
  std::printf("max|exact|=%.4g over %d inputs\\n", maxExact, count);
  for (int v = 0; v < V; ++v) {{
    std::printf("%s\\tmaxerr=%.3g\\tmeanrel=%.3g\\targmax=%d/%d\\tnonfinite=%d\\n",
                names[v], maxErr[v], sumRel[v], agree[v], count, nonfinite[v]);
  }}
  return 0;
}}""")
print('\n'.join(L))
