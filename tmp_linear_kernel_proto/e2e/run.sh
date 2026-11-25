# usage: TAG=<tag> run.sh <slots> "<options>" Model...
# Builds the package, writes the tests, and runs them; logs in ~/e2e/<tag>/.
N=$1; OPTS=$2; shift 2
tag=${TAG:-$N}; out=~/e2e/$tag; mkdir -p $out
P=tests/Examples/lattigo/ckks/corpus_e2e_$tag
pkg=~/sky_workdir/$P
E=~/sky_workdir/tmp_linear_kernel_proto/e2e
mkdir -p $pkg
rm -f $pkg/*_test.go
TAG=$tag bash $E/mkpkg.sh $N "$OPTS" "$@" > /dev/null
cd ~/sky_workdir
bazel build --config=remote --noshow_progress -k //$P:all > $out/build.log 2>&1
grep -a 'error:\|ERROR' $out/build.log | head -6 | cut -c1-300
for m in "$@"; do
  src=~/sky_workdir/tmp_linear_kernel_proto/corpus/michiel_$m.mlir
  fn=$(grep -o 'func.func @[A-Za-z0-9_]*' $src | head -1 | cut -d@ -f2)
  [ -f ~/e2e/$m/input.txt ] || bash $E/ref.sh $m > /dev/null
  lib=~/sky_workdir/bazel-bin/$P/${fn}_lib.go
  [ -f $lib ] || { echo "== $m: no generated Go code"; continue; }
  python3 $E/gentest.py corpus_e2e_$tag $fn ~/e2e/$m/input.txt ~/e2e/$m/expected.txt $lib > $pkg/${fn}_test.go
done
TAG=$tag bash $E/mkpkg.sh $N "$OPTS" "$@" > /dev/null
for m in "$@"; do
  fn=$(echo $m | tr 'A-Z' 'a-z')
  [ -f $pkg/${fn}_test.go ] || continue
  s=$(date +%s)
  bazel test --config=remote --noshow_progress --test_output=all --test_timeout=${TT:-7200} --test_strategy=standalone --strategy=TestRunner=local --test_arg=-test.v //$P:${fn}_test > $out/$m.log 2>&1
  echo "== $m rc=$? wall=$(( $(date +%s) - s ))s"
  grep -a 'E2E \(keygen\|encrypt\|preprocessing\|eval\|decrypt\|mem\|result\)\|^--- \|panic\|error:\|ERROR' $out/$m.log | sed 's/^ *[a-z0-9_]*_test.go:[0-9]*: //' | head -14 | cut -c1-220
done
