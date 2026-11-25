# Runs the e2e and regression tests that failed with the linear kernel as default.
cd ~/sky_workdir
bazel test --config=remote -k //tests/Examples/... //tests/Regression/... //tests/Transforms/... //tests/Dialect/... --test_output=errors > ~/t.log 2>&1
grep '^ERROR: /home/ubuntu/sky_workdir' ~/t.log | sed 's/.*sky_workdir\/tests\///; s/\/BUILD:[0-9:]* / /; s/ tests.*//' | sort | uniq -c
grep 'Failed to translate op\|Assertion\|error: \|TEST.*FAILED\|  FAILED in\|TIMEOUT in\|Executed' ~/t.log | grep -v 'builtin.module\|func.func\|scf.for\|Wdeprecated\|affine.for' | sed 's/.*error: //' | cut -c1-200 | sort | uniq -c | sort -rn | head -50
