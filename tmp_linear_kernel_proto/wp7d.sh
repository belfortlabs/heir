# usage: wp7d.sh <mlir> <flags...>: runs heir-opt with a symbolizer and prints the stack of a crash.
cd ~/sky_workdir
sym=$(find ~/.cache/bazel -name llvm-symbolizer -type f 2>/dev/null | head -1)
[ -z "$sym" ] && sym=$(find ~/.cache/bazel -name 'llvm-symbolizer*' 2>/dev/null | head -1)
echo "symbolizer: $sym"
export LLVM_SYMBOLIZER_PATH=$sym
f=$1; shift
bazel build --config=remote //tools:heir-opt > /dev/null 2>&1
LLVM_SYMBOLIZER_PATH=$sym bazel-bin/tools/heir-opt "$@" ~/sky_workdir/$f > ~/ir.out 2> ~/ir.err
echo "exit $?"
grep -n '^ *#[0-9]\|^[0-9]* *heir-opt' ~/ir.err | sed 's/0x[0-9a-f]* //' | cut -c1-260 | head -${LINES_:-22}
