"""Writes a Go test that runs one compiled corpus model under encryption.

usage: gentest.py <package dir name> <func name> <input.txt> <expected.txt> > <func>_test.go
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

func TestE2E(t *testing.T) {{
    start := time.Now()
    {bt_decl}evaluator, params, ecd, enc, dec := {go}__configure()
    t.Logf("E2E keygen %.3fs logN=%d maxLevel=%d logQP=%.0f bootstrapCalls={nboot}", time.Since(start).Seconds(), params.LogN(), params.MaxLevel(), params.LogQP())

    start = time.Now()
    ct0 := {go}__encrypt__arg0(evaluator, params, ecd, enc, e2eInput)
    t.Logf("E2E encrypt %.3fs ciphertexts=%d", time.Since(start).Seconds(), len(ct0))

    start = time.Now()
    linearTransforms, weightPlains := {fn}_utils.{go}__preprocessing(params, ecd)
    t.Logf("E2E preprocessing %.3fs transforms=%d plaintexts=%d", time.Since(start).Seconds(), len(linearTransforms), len(weightPlains))

    start = time.Now()
    resultCt := {go}__preprocessed({bt_arg}evaluator, params, ecd, ct0, linearTransforms, weightPlains)
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
