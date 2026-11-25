# usage: pcheck.sh <slots> <parallel jobs> models...
# check.sh with several models at a time. Compiles each corpus model with the
# linear kernel and as a plain tensor program, runs both on the same random
# input, and compares.
cd ~/sky_workdir
N=$1; J=$2; shift 2
W=${CHECK_DIR:-~/check_$N}; mkdir -p $W
export Q="--config=remote --output_filter=^\$ --noshow_progress --ui_event_filters=-info,-warning,-stdout,-stderr"
FLAGS="--mlir-to-plaintext-backend=plaintext-size=$N"
# The reference: the same pipeline up to layout propagation, then the tensor
# program is lowered to loops directly.
bazel run $Q //tools:heir-opt -- "$FLAGS" --dump-pass-pipeline - < /dev/null > $W/pipeline.txt 2>&1
python3 - $W/pipeline.txt > $W/ref_pipeline.txt <<'PY'
import re, sys
t = re.search(r'builtin\.module\(.*?\n\)', open(sys.argv[1]).read(), re.S).group(0)
t = t[:t.index('strided-layout-propagation')].rstrip().rstrip(',')
print(t + ')' * (t.count('(') - t.count(')')))
PY
one() {
  m=$1; N=$2; W=$3
  FLAGS="--mlir-to-plaintext-backend=plaintext-size=$N"
  OPT="bazel run $Q //tools:heir-opt --"
  REF=$(cat $W/ref_pipeline.txt)
  src=~/sky_workdir/tmp_linear_kernel_proto/corpus/michiel_$m.mlir
  fn=$(grep -o 'func.func @[A-Za-z0-9_]*' $src | head -1 | cut -d@ -f2)
  d=$W/$m; rm -rf $d; mkdir -p $d
  s=$(date +%s)
  lower() {
    k=$1
    bazel run $Q @llvm-project//mlir:mlir-translate -- --mlir-to-llvmir $d/$k.mlir -o $d/$k.ll || return 1
    if [ $k = ref ]; then
      sed -i "s/@$fn\b/@ref_$fn/g; s/@_mlir_ciface_$fn\b/@_mlir_ciface_ref_$fn/g" $d/ref.ll
    fi
    bazel run $Q @llvm-project//llvm:llc -- -filetype=obj -relocation-model=pic $d/$k.ll -o $d/$k.o || return 1
    rm -f $d/$k.ll
  }
  ( set -e
    ( $OPT "$FLAGS" $src -o $d/fhe.mlir && lower fhe ) > $d/fhe.log 2>&1 &
    fhe=$!
    ( $OPT "--pass-pipeline=$REF" $src -o $d/ref0.mlir &&
      $OPT --secret-forget-secrets --heir-polynomial-to-llvm $d/ref0.mlir -o $d/ref.mlir &&
      lower ref ) > $d/ref.log 2>&1 &
    ref=$!
    wait $fhe; wait $ref
    python3 ~/sky_workdir/tmp_linear_kernel_proto/gen_driver.py $src > $d/driver.cpp
  ) > $d/build.log 2>&1
  rc=$?
  cat $d/fhe.log $d/ref.log >> $d/build.log 2>/dev/null
  c=$(( $(date +%s) - s ))
  if [ $rc -ne 0 ]; then
    { printf "== %-24s BUILD FAILED after %ss\n" $m $c
      grep -o "error: .*\|undefined .*\|Assertion.*" $d/build.log | sed 's/[0-9]\{4,\}/N/g' | sort | uniq -c | head -3 | cut -c1-300
    } > $d/summary
    return
  fi
  # With inputs in [-1, 1] the polynomial activations of the deeper models
  # overflow in the reference itself, so smaller inputs are tried next.
  for defs in "-DSECRET_HI=1.0" "-DSECRET_HI=0.1" "-DSECRET_HI=0.1 -DMEAN_HI=0.02 -DVAR_LO=4 -DVAR_HI=5" "-DSECRET_HI=0.01 -DMEAN_HI=0.02 -DVAR_LO=4 -DVAR_HI=5"; do
    clang++ -std=c++20 -O1 $defs $d/driver.cpp ~/sky_workdir/tests/llvm_runner/memrefCopy.cpp -I ~/sky_workdir $d/fhe.o $d/ref.o -o $d/check >> $d/build.log 2>&1
    verdict=$(timeout 1800 $d/check 2>&1 | tail -1)
    case "$verdict" in NO-REFERENCE*) ;; *) break ;; esac
  done
  printf "== %-24s build=%ss  [%s] %s\n" $m $c "$defs" "$verdict" > $d/summary
}
export -f one
printf "%s\n" "$@" | xargs -P $J -I{} bash -c "one {} $N $W"
for m in "$@"; do cat $W/$m/summary; done
