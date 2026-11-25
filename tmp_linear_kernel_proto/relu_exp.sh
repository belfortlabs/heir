# usage: relu_exp.sh <model> [calibration inputs] [evaluation inputs]
# ReLU approximation experiment on one corpus model at the plain tensor level:
# exact ReLU against polynomial approximations with and without calibrated
# range normalization. Results in ~/relu/<model>/.
# RUNG_ERRORS (deg5:deg15:deg31:deg63:composite) comes from relu_rungs.sh.
# MARGIN (default 1.25) is the factor on the calibrated bounds; the results
# file is results_m<MARGIN>.txt.
cd ~/sky_workdir
m=$1; N=${2:-32}; M=${3:-20}
E=${RUNG_ERRORS:?set RUNG_ERRORS}
MG=${MARGIN:-1.25}
d=~/relu/$m; mkdir -p $d
export Q="--config=remote --output_filter=^\$ --noshow_progress --ui_event_filters=-info,-warning,-stdout,-stderr"
src=~/sky_workdir/tmp_linear_kernel_proto/corpus/michiel_$m.mlir
fn=$(grep -o 'func.func @[A-Za-z0-9_]*' $src | head -1 | cut -d@ -f2)
[ -s $d/pre.mlir ] || bash ~/pre.sh $m > /dev/null
lower() {
  k=$1
  bazel run $Q @llvm-project//mlir:mlir-translate -- --mlir-to-llvmir $d/$k.mlir -o $d/$k.ll || return 1
  sed -i "s/@$fn\b/@${k}_$fn/g; s/@_mlir_ciface_$fn\b/@_mlir_ciface_${k}_$fn/g" $d/$k.ll
  bazel run $Q @llvm-project//llvm:llc -- -filetype=obj -relocation-model=pic $d/$k.ll -o $d/$k.o || return 1
}
TAIL="--polynomial-approximation --lower-polynomial-eval --canonicalize --cse --secret-forget-secrets --heir-polynomial-to-llvm"
build() {
  k=$1; shift
  { bazel run $Q //tools:heir-opt -- "$@" $d/pre.mlir -o $d/$k.mlir && lower $k; } > $d/$k.log 2>&1 || echo "BUILD FAILED $k"
}
build exact --secret-forget-secrets --heir-polynomial-to-llvm &
build calib --secret-forget-secrets --relu-calibration --heir-polynomial-to-llvm &
build A0 $TAIL &
wait
python3 tmp_linear_kernel_proto/relu_driver.py $src calib > $d/calib_driver.cpp
clang++ -std=c++20 -O1 $DEFS $d/calib_driver.cpp tests/llvm_runner/memrefCopy.cpp -I . $d/calib.o -o $d/calib 2> $d/calib_link.log || { echo "calib link failed"; exit 1; }
$d/calib $N $d/calib.txt
C="calibration-file=$d/calib.txt rung-errors=$E margin=$MG"
build A1  "--relu-normalization=$C mode=per-tensor rung=deg5" $TAIL &
build A15 "--relu-normalization=$C mode=per-tensor rung=deg15" $TAIL &
build B   "--relu-normalization=$C mode=per-channel rung=deg5" $TAIL &
build B15 "--relu-normalization=$C mode=per-channel rung=deg15" $TAIL &
build B31 "--relu-normalization=$C mode=per-channel rung=deg31" $TAIL &
build C30 "--relu-normalization=$C mode=per-channel rung=auto tau=0.3" $TAIL &
build C10 "--relu-normalization=$C mode=per-channel rung=auto tau=0.1" $TAIL &
build C03 "--relu-normalization=$C mode=per-channel rung=auto tau=0.03" $TAIL &
build D   "--relu-normalization=$C mode=per-channel rung=composite" $TAIL &
build D0  "--relu-normalization=$C mode=per-tensor rung=composite" $TAIL &
wait
V=""; O=""
for k in A0 A1 A15 B B15 B31 C30 C10 C03 D D0; do [ -s $d/$k.o ] && { V="$V $k"; O="$O $d/$k.o"; }; done
python3 tmp_linear_kernel_proto/relu_driver.py $src eval $V > $d/eval_driver.cpp
clang++ -std=c++20 -O1 $DEFS $d/eval_driver.cpp tests/llvm_runner/memrefCopy.cpp -I . $d/exact.o $O -o $d/eval 2> $d/eval_link.log || { echo "eval link failed"; exit 1; }
{ echo "== $m N=$N M=$M margin=$MG variants:$V"; $d/eval $M
  for k in A1 A15 B B15 B31 C30 C10 C03 D D0; do
    echo "$k rungs: $(grep -a '^relu-normalization' $d/$k.log | cut -f7 | sort | uniq -c | tr '\n' ' ') r(min/med/max): $(grep -a '^relu-normalization' $d/$k.log | cut -f4 | sort -g | awk '{a[NR]=$1} END {printf "%.3g/%.3g/%.3g", a[1], a[int((NR+1)/2)], a[NR]}')"
  done
} > $d/results_m$MG.txt 2>&1
cat $d/results_m$MG.txt
