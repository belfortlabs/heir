# usage: ir.sh <mlir relative to workdir> <heir-opt flags...>
# stdout of heir-opt goes to ~/ir.out, stderr (IR dumps, diagnostics) to ~/ir.err.
cd ~/sky_workdir
f=$1; shift
bazel run --config=remote --output_filter='^$' --noshow_progress --ui_event_filters=-info,-warning,-stdout,-stderr //tools:heir-opt -- "$@" ~/sky_workdir/$f > ~/ir.out 2> ~/ir.err
echo "exit $? out=$(wc -l < ~/ir.out) err=$(wc -l < ~/ir.err)"
grep -n 'error:\|Assertion\|LLVM ERROR\|PLEASE submit' ~/ir.err | cut -c1-400 | head -20
