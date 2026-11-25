# usage: wp7c.sh <lines> <mlir> <flags...>: dumps the IR before each pass and
# prints the name of the last pass reached and the head of the IR it received.
cd ~/sky_workdir
n=$1; shift
rm -rf /tmp/mlir; mkdir -p /tmp/mlir
bash tmp_linear_kernel_proto/ir.sh "$@" --mlir-print-ir-before-all --mlir-print-ir-tree-dir=/tmp/mlir
grep -n 'Assertion\|LLVM ERROR' ~/ir.err | cut -c1-300 | head -3
d=$(ls -d /tmp/mlir/*/ | head -1)
last=$(ls $d | sort -n | tail -1)
ls $d | sort -n | tail -3
cut -c1-330 "$d/$last" | head -$n
