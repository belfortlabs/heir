# usage: ref.sh models...   Writes ~/e2e/<model>/{input,expected,plain}.txt from
# the objects of the 16,384-slot numeric check (~/check_16384/<model>).
for m in "$@"; do
  d=~/e2e/$m; mkdir -p $d; c=~/check_16384/$m
  defs=$(sed -n 's/.*\[\(-D[^]]*\)\].*/\1/p' $c/summary)
  python3 ~/sky_workdir/tmp_linear_kernel_proto/e2e/ref_dump.py $c/driver.cpp > $d/driver_dump.cpp
  clang++ -std=c++20 -O1 $defs $d/driver_dump.cpp ~/sky_workdir/tests/llvm_runner/memrefCopy.cpp -I ~/sky_workdir $c/fhe.o $c/ref.o -o $d/refdump > $d/ref_build.log 2>&1 || { echo "$m: build failed"; continue; }
  ( cd $d && ./refdump | tail -1 | cut -c1-160 )
  echo "$m: defs=$defs input=$(wc -l < $d/input.txt) expected=$(tr '\n' ' ' < $d/expected.txt | cut -c1-120)"
done
