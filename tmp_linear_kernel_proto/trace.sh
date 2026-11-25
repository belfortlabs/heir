# usage: trace.sh <model> <slots> <timeout>; reports the passes that ran and where time went.
cd ~/sky_workdir
m=$1; N=$2; TO=$3
rm -rf ~/tree; mkdir -p ~/tree
bazel run --config=remote --output_filter='^$' --noshow_progress --ui_event_filters=-info,-warning,-stdout,-stderr --run_under="timeout $TO" //tools:heir-opt -- "--mlir-to-plaintext-backend=plaintext-size=$N linear-kernel=true" --mlir-print-ir-after-all --mlir-print-ir-tree-dir=$HOME/tree --mlir-elide-elementsattrs-if-larger=4 --mlir-elide-resource-strings-if-larger=4 ~/sky_workdir/tmp_linear_kernel_proto/corpus/michiel_$m.mlir -o /dev/null > ~/trace.log 2>&1
echo "rc=$?"
find ~/tree -name '*.mlir' -printf '%T@ %s %f\n' | sort -n | awk 'NR>1{printf "%7.1fs %9d %s\n", $1-prev, $2, $3} {prev=$1}' | sort -rn | head -8
echo "last:"; find ~/tree -name '*.mlir' -printf '%T@ %f\n' | sort -n | tail -3 | cut -d' ' -f2
