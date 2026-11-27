"""Writes a Go test that runs one compiled corpus model under encryption.

usage: gentest.py <package dir name> <func name> <input.txt> <expected.txt> <lib.go> > <func>_test.go

Cleartext arguments (for example batch norm statistics) are read from args.txt
next to input.txt, as written by ref_dump.py.
"""

import sys

pkg, fn, inp, exp, lib = sys.argv[1:6]
go = fn[0].upper() + fn[1:]
import re

libsrc = open(lib).read()
conf = re.search(
    r'func ' + fn[0].upper() + fn[1:] + r'__configure\(\) \((.*?)\) \{', libsrc
).group(1)
boot = 'bootstrapping.Evaluator' in conf
nboot = len(re.findall(r'\.Bootstrap\(', libsrc))
bt_decl = 'btEvaluator, ' if boot else ''
bt_arg = 'btEvaluator, ' if boot else ''
# Cleartext arguments: the evaluation function takes all of them, in order,
# between the encoder and the ciphertext; the preprocessing function takes
# only the ones it reads.
import os

sig = re.search(r'func ' + go + r'__preprocessed\((.*?)\) \(', libsrc).group(1)
clear_types = re.findall(r'_ (\[\](?:float32|float64|int64|int32))', sig)
secret_idx = re.search(r'func ' + go + r'__encrypt__arg(\d+)\(', libsrc).group(
    1
)
utilsrc = open(lib.replace('_lib.go', '_utils.go')).read()
presig = re.search(
    r'func ' + go + r'__preprocessing\((.*?)\) \(', utilsrc
).group(1)
n_pre = len(re.findall(r'v\d+ \[\]', presig))
argfile = os.path.join(os.path.dirname(inp), 'args.txt')
argvals = (
    [l.split()[2:] for l in open(argfile).read().splitlines() if l.strip()]
    if os.path.exists(argfile)
    else []
)
assert len(argvals) == len(clear_types), (len(argvals), len(clear_types))
arg_decls, eval_args = [], []
for i, (ty, vals) in enumerate(zip(clear_types, argvals)):
  if 'int' in ty:
    vals = [str(int(float(v))) for v in vals]
  arg_decls.append(f'var e2eArg{i} = {ty}{{{", ".join(vals)}}}')
  eval_args.append(f'e2eArg{i}, ')
# The compiler records which entry arguments the preprocessing function takes,
# and in which order, in its heir.interface attribute.
pre_args = []
if n_pre:
  mlir = open(lib.replace('_lib.go', '_heir_opt.mlir')).read()
  line = re.search(r'func\.func @' + fn + r'__preprocessing\(.*', mlir).group(0)
  idx = re.search(r'entry_arg_indices = array<i64: ([\d, ]+)>', line).group(1)
  pre_args = [f', e2eArg{int(i)}' for i in idx.split(',')]
assert len(pre_args) == n_pre, (len(pre_args), n_pre)
arg_decls = '\n'.join(arg_decls)
eval_args = ''.join(eval_args)
pre_args = ''.join(pre_args)
xs = open(inp).read().split()
es = open(exp).read().split()
print(f"""package {fn}

import (
    "math"
    "runtime"
    "testing"
    "time"

    "tests/Examples/lattigo/ckks/{pkg}/{fn}_utils"
)

var e2eInput = []float32{{{", ".join(xs)}}}
var e2eExpected = []float64{{{", ".join(es)}}}
{arg_decls}

func TestE2E(t *testing.T) {{
    start := time.Now()
    {bt_decl}evaluator, params, ecd, enc, dec := {go}__configure()
    t.Logf("E2E keygen %.3fs logN=%d maxLevel=%d logQP=%.0f bootstrapCalls={nboot}", time.Since(start).Seconds(), params.LogN(), params.MaxLevel(), params.LogQP())

    start = time.Now()
    ct0 := {go}__encrypt__arg{secret_idx}(evaluator, params, ecd, enc, e2eInput)
    t.Logf("E2E encrypt %.3fs ciphertexts=%d", time.Since(start).Seconds(), len(ct0))

    start = time.Now()
    linearTransforms, weightPlains := {fn}_utils.{go}__preprocessing(params, ecd{pre_args})
    t.Logf("E2E preprocessing %.3fs transforms=%d plaintexts=%d", time.Since(start).Seconds(), len(linearTransforms), len(weightPlains))

    start = time.Now()
    resultCt := {go}__preprocessed({bt_arg}evaluator, params, ecd, {eval_args}ct0, linearTransforms, weightPlains)
    t.Logf("E2E eval %.3fs", time.Since(start).Seconds())

    start = time.Now()
    result := {go}__decrypt__result0(evaluator, params, ecd, dec, resultCt)
    t.Logf("E2E decrypt %.3fs", time.Since(start).Seconds())

    var mem runtime.MemStats
    runtime.ReadMemStats(&mem)
    t.Logf("E2E mem sys=%.1fGB", float64(mem.Sys)/1e9)

    maxErr, maxRef := 0.0, 0.0
    argRef, argGot := 0, 0
    for i, want := range e2eExpected {{
        got := float64(result[i])
        if math.IsNaN(got) || math.IsInf(got, 0) {{
            t.Errorf("output %d is not finite", i)
        }}
        maxErr = math.Max(maxErr, math.Abs(got-want))
        maxRef = math.Max(maxRef, math.Abs(want))
        if want > e2eExpected[argRef] {{
            argRef = i
        }}
        if got > float64(result[argGot]) {{
            argGot = i
        }}
        t.Logf("E2E out %d got %.6f want %.6f", i, got, want)
    }}
    t.Logf("E2E result maxerr=%.3g maxref=%.3g argmax got/want=%d/%d", maxErr, maxRef, argGot, argRef)
    if maxErr > 0.05*maxRef {{
        t.Errorf("max error %.3g exceeds 5%% of max |reference| %.3g", maxErr, maxRef)
    }}
}}""")
