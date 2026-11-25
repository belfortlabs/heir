# usage: timing.sh <model> <slots> <timeout> [extra pipeline options]; prints the slowest passes.
cd ~/sky_workdir
m=$1; N=$2; TO=$3; X=$4
s=$(date +%s)
bazel run --config=remote --output_filter='^$' --noshow_progress --ui_event_filters=-info,-warning,-stdout,-stderr --run_under="timeout $TO" //tools:heir-opt -- "--mlir-to-plaintext-backend=plaintext-size=$N linear-kernel=true $X" --mlir-timing --mlir-timing-display=list ~/sky_workdir/tmp_linear_kernel_proto/corpus/michiel_$m.mlir -o ~/timing_$m_$N.mlir > ~/timing_$m_$N.log 2>&1
echo "== $m N=$N rc=$? $(( $(date +%s) - s ))s"
grep -o "error: .*" ~/timing_$m_$N.log | sort | uniq -c | head -3 | cut -c1-300
grep -E "^\s+[0-9.]+ \([ 0-9.]+%\)" ~/timing_$m_$N.log | sort -k1 -n -r | head -${TOP:-7} | cut -c1-120
