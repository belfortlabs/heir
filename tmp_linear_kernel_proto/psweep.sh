# usage: psweep.sh <slots> <timeout seconds> <parallel jobs> models...
# Like sweep.sh, with several models at a time and the peak memory of heir-opt.
cd ~/sky_workdir
N=$1; TO=$2; J=$3; shift 3
D=~/sweep_$N; mkdir -p $D
one() {
  m=$1; N=$2; TO=$3; D=$4
  s=$(date +%s.%N)
  bazel run --config=remote --output_filter='^$' --noshow_progress --ui_event_filters=-info,-warning,-stdout,-stderr --run_under="/usr/bin/time -v timeout $TO" //tools:heir-opt -- "--mlir-to-plaintext-backend=plaintext-size=$N" --mlir-timing ~/sky_workdir/tmp_linear_kernel_proto/corpus/michiel_$m.mlir -o $D/$m.out.mlir > $D/$m.log 2>&1
  e=$(date +%s.%N)
  rc=$(grep -o "Exit status: [0-9]*" $D/$m.log | cut -d' ' -f3)
  rss=$(grep "Maximum resident" $D/$m.log | grep -o "[0-9]*$")
  {
    printf "== %-24s rc=%-3s %6.1fs  %5.1f GB  out=%s lines\n" $m "$rc" $(echo "$e - $s" | bc) $(echo "scale=1; ${rss:-0} / 1048576" | bc) $(wc -l < $D/$m.out.mlir 2>/dev/null)
    grep -o "error: .*" $D/$m.log | sed 's/[0-9]\{3,\}/N/g' | sort | uniq -c | sort -rn | head -3 | cut -c1-330
    grep -o "Assertion.*failed" $D/$m.log | head -1 | cut -c1-300
    grep -E "^\s+[0-9.]+ \([ 0-9.]+%\)" $D/$m.log | sort -k1 -n -r | sed -n 2,3p | cut -c1-110
  } > $D/$m.summary
}
export -f one
printf "%s\n" "$@" | xargs -P $J -I{} bash -c "one {} $N $TO $D"
for m in "$@"; do cat $D/$m.summary; done
