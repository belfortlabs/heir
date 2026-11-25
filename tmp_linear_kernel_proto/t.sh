# usage: t.sh <bazel test targets...>; prints failures only.
cd ~/sky_workdir
bazel test --config=remote "$@" --test_output=errors > ~/t.log 2>&1
echo "test exit $?"
grep -n ' error: \|fatal error\|Failure$\|^Expected\|Which is\|Value of\|Actual\|at element\|The difference\|\[  FAILED\|Assertion\|LLVM ERROR\|PASSED \|FAILED \|tests pass\|expected string\|: error:\|^ERROR\|Segmentation\|Aborted\|check-next\|CHECK' ~/t.log | grep -v Wdeprecated | cut -c1-300 | head -${TLINES:-60}
