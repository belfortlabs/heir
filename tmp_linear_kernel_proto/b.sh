# Build heir-opt and print compile errors only.
cd ~/sky_workdir
bazel build --config=remote //tools:heir-opt > ~/b.log 2>&1
echo "build exit $?"
grep -n ' error: \|fatal error\|^ERROR' ~/b.log | cut -c1-500 | head -${1:-40}
