# usage: dbg.sh <slots> models...; prints the largest linear ops and joins of each model.
cd ~/sky_workdir
N=$1; shift
for m in "$@"; do
  echo "== $m"
  bash tmp_linear_kernel_proto/ir.sh tmp_linear_kernel_proto/corpus/michiel_$m.mlir "--mlir-to-plaintext-backend=plaintext-size=$N" --debug-only=strided-layout-propagation --mlir-timing | cut -c1-600
  grep -o "^linear op [a-z_0-9.]*: [0-9]* diagonals, [0-9]* rotation sums; loops [^]]*\]" ~/ir.err | sort | uniq -c | sort -k5 -n -r | head -6
  grep "^linear op" ~/ir.err | awk '{d+=$4} END {print "total diagonals", d, "in", NR, "linear ops"}'
  grep "^join at" ~/ir.err | awk '{d+=$5} END {print "total join diagonals", d+0, "in", NR, "joins"}'
  grep "^join at" ~/ir.err | sort -k5 -n -r | head -3 | cut -c1-700
  grep -c "^join at" ~/ir.err
done
