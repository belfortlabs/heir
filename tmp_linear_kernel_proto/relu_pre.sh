# usage: pre.sh model   -> ~/relu/<model>/pre.mlir (pipeline cut before polynomial-approximation)
cd ~/sky_workdir
m=$1; d=~/relu/$m; mkdir -p $d
Q="--config=remote --output_filter=^\$ --noshow_progress --ui_event_filters=-info,-warning,-stdout,-stderr"
python3 - > ~/relu/pre_pipeline.txt <<'PY'
import os
t = open(os.path.expanduser('~/check_32768/ref_pipeline.txt')).read()
t = t[:t.index('polynomial-approximation')].rstrip().rstrip(',')
print(t + ')' * (t.count('(') - t.count(')')))
PY
bazel run $Q //tools:heir-opt -- "--pass-pipeline=$(cat ~/relu/pre_pipeline.txt)" ~/sky_workdir/tmp_linear_kernel_proto/corpus/michiel_$m.mlir -o $d/pre.mlir > $d/pre.log 2>&1
echo "exit $?"; grep -m3 "error" $d/pre.log | cut -c1-300
