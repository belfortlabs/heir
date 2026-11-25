# usage: sweep.sh <slots> <timeout seconds> models...
# Compiles each corpus model with the linear kernel on the plaintext backend.
cd ~/sky_workdir
N=$1; TO=$2; shift 2
mkdir -p ~/sweep
for m in "$@"; do
  s=$(date +%s.%N)
  bazel run --config=remote --output_filter='^$' --noshow_progress --ui_event_filters=-info,-warning,-stdout,-stderr --run_under="timeout $TO" //tools:heir-opt -- "--mlir-to-plaintext-backend=plaintext-size=$N linear-kernel=true" --mlir-timing ~/sky_workdir/tmp_linear_kernel_proto/corpus/michiel_$m.mlir -o ~/sweep/$m.out.mlir > ~/sweep/$m.log 2>&1
  rc=$?
  e=$(date +%s.%N)
  printf "== %-16s rc=%-3s %6.1fs  out=%s lines\n" $m $rc $(echo "$e - $s" | bc) $(wc -l < ~/sweep/$m.out.mlir 2>/dev/null)
  grep -o "error: .*" ~/sweep/$m.log | sort | uniq -c | head -3 | cut -c1-330
  grep -o "Assertion.*failed" ~/sweep/$m.log | head -1 | cut -c1-300
  grep -E "^\s+[0-9.]+ \([ 0-9.]+%\)" ~/sweep/$m.log | sort -k1 -n -r | sed -n 2,4p | cut -c1-110
done
