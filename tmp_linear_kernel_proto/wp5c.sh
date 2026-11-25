# Reruns existing check binaries with other input ranges: "<secret hi> <mean hi> <var lo> <var hi>".
models="$@"; for m in $models; do
  d=~/check/$m
  python3 ~/sky_workdir/tmp_linear_kernel_proto/gen_driver.py ~/sky_workdir/tmp_linear_kernel_proto/corpus/michiel_$m.mlir > $d/driver.cpp
  for r in "1.0 0.1 0.8 1.2" "0.1 0.1 0.8 1.2" "0.1 0.02 4 5" "0.1 0.5 0.3 0.6"; do
    set -- $r
    clang++ -std=c++20 -O1 -DSECRET_HI=$1 -DMEAN_HI=$2 -DVAR_LO=$3 -DVAR_HI=$4 $d/driver.cpp ~/sky_workdir/tests/llvm_runner/memrefCopy.cpp -I ~/sky_workdir $d/fhe.o $d/ref.o -o $d/check_c 2>&1 | tail -2
    echo "== $m secret<=$1 |mean|<=$2 var in [$3, $4]: $(timeout 900 $d/check_c 2>&1 | tail -1)"
  done
done
