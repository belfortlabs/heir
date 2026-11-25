# usage: rerun.sh <check dir> <clang defs> models...; regenerates the driver and reruns existing objects.
W=$1; defs=$2; shift 2
for m in "$@"; do
  d=$W/$m
  python3 ~/sky_workdir/tmp_linear_kernel_proto/gen_driver.py ~/sky_workdir/tmp_linear_kernel_proto/corpus/michiel_$m.mlir > $d/driver.cpp
  clang++ -std=c++20 -O1 $defs $d/driver.cpp ~/sky_workdir/tests/llvm_runner/memrefCopy.cpp -I ~/sky_workdir $d/fhe.o $d/ref.o -o $d/check 2>&1 | grep error
  echo "== $m [$defs] $(timeout 1800 $d/check 2>&1 | tail -1)"
done
