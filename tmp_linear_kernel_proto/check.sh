# usage: check.sh <slots> models...
# Compiles each corpus model with the linear kernel and as a plain tensor
# program, runs both on a random input, and compares.
cd ~/sky_workdir
N=$1; shift
W=~/check; mkdir -p $W
Q="--config=remote --output_filter=^\$ --noshow_progress --ui_event_filters=-info,-warning,-stdout,-stderr"
OPT="bazel run $Q //tools:heir-opt --"
FLAGS="--mlir-to-plaintext-backend=plaintext-size=$N linear-kernel=true"
# The reference: the same pipeline up to layout propagation, then the tensor
# program is lowered to loops directly.
$OPT "$FLAGS" --dump-pass-pipeline - < /dev/null > $W/pipeline.txt 2>&1
REF=$(python3 - $W/pipeline.txt <<'PY'
import re, sys
t = re.search(r'builtin\.module\(.*?\n\)', open(sys.argv[1]).read(), re.S).group(0)
t = t[:t.index('strided-layout-propagation')].rstrip().rstrip(',')
print(t + ')' * (t.count('(') - t.count(')')))
PY
)
for m in "$@"; do
  src=~/sky_workdir/tmp_linear_kernel_proto/corpus/michiel_$m.mlir
  fn=$(grep -o 'func.func @[A-Za-z0-9_]*' $src | head -1 | cut -d@ -f2)
  d=$W/$m; rm -rf $d; mkdir -p $d
  s=$(date +%s)
  ( set -e
    $OPT "$FLAGS" $src -o $d/fhe.mlir
    $OPT "--pass-pipeline=$REF" $src -o $d/ref0.mlir
    $OPT --secret-forget-secrets --heir-polynomial-to-llvm $d/ref0.mlir -o $d/ref.mlir
    for k in fhe ref; do
      bazel run $Q @llvm-project//mlir:mlir-translate -- --mlir-to-llvmir $d/$k.mlir -o $d/$k.ll
    done
    sed -i "s/@$fn\b/@ref_$fn/g; s/@_mlir_ciface_$fn\b/@_mlir_ciface_ref_$fn/g" $d/ref.ll
    for k in fhe ref; do
      bazel run $Q @llvm-project//llvm:llc -- -filetype=obj -relocation-model=pic $d/$k.ll -o $d/$k.o
    done
    python3 ~/sky_workdir/tmp_linear_kernel_proto/gen_driver.py $src > $d/driver.cpp
  ) > $d/build.log 2>&1
  rc=$?
  c=$(( $(date +%s) - s ))
  if [ $rc -ne 0 ]; then
    printf "== %-16s BUILD FAILED after %ss\n" $m $c
    grep -o "error: .*\|undefined .*\|Assertion.*" $d/build.log | sort | uniq -c | head -4 | cut -c1-300
    continue
  fi
  # With inputs in [-1, 1] the polynomial activations of the deeper models
  # overflow in the reference itself, so smaller inputs are tried next.
  for defs in "-DSECRET_HI=1.0" "-DSECRET_HI=0.1" "-DSECRET_HI=0.1 -DMEAN_HI=0.02 -DVAR_LO=4 -DVAR_HI=5"; do
    clang++ -std=c++20 -O1 $defs $d/driver.cpp ~/sky_workdir/tests/llvm_runner/memrefCopy.cpp -I ~/sky_workdir $d/fhe.o $d/ref.o -o $d/check >> $d/build.log 2>&1
    verdict=$(timeout 900 $d/check 2>&1 | tail -1)
    case "$verdict" in NO-REFERENCE*) ;; *) break ;; esac
  done
  printf "== %-16s build=%ss  [%s] %s\n" $m $c "$defs" "$verdict"
done
