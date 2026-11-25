# usage: ops.sh <backend> <model> [pipeline options]
# Compiles a corpus model to CKKS and prints executed-op counts of its main function.
cd ~/sky_workdir
b=$1; m=$2; shift 2
d=~/ops/$b; mkdir -p $d
s=$(date +%s)
timeout ${T:-3600} bazel run --config=remote --output_filter='^$' --noshow_progress --ui_event_filters=-info,-warning,-stdout,-stderr //tools:heir-opt -- "--annotate-module=backend=$b scheme=ckks" "--torch-linalg-to-ckks=min-slot-count=${N:-16384} $*" --count-ops ~/sky_workdir/tmp_linear_kernel_proto/corpus/michiel_$m.mlir -o /dev/null 2>&1 | grep -a --line-buffered '^count-ops\|error:\|Assertion\|LLVM ERROR' | cut -c1-400 | head -3000 > $d/$m.counts
rc=${PIPESTATUS[0]}
echo "== $m rc=$rc time=$(( $(date +%s) - s ))s errors=$(grep -c 'error:\|Assertion\|LLVM ERROR' $d/$m.counts)" > $d/$m.summary
grep -m3 'error:\|Assertion\|LLVM ERROR' $d/$m.counts | sed 's/^.*error: /error: /' | cut -c1-250 >> $d/$m.summary
cat $d/$m.summary
