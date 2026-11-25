# usage: wp7a.sh <mlir> <flags...>: runs heir-opt and prints assertion and error lines.
cd ~/sky_workdir
bash tmp_linear_kernel_proto/ir.sh "$@"
grep -n 'Assertion\|LLVM ERROR\|UNREACHABLE\|error:' ~/ir.err | grep -v 'Unable to find backend' | cut -c1-600 | head -10
