cd ~/sky_workdir
bazel test --config=remote -k //lib/... //tests/... --test_output=errors > ~/full.log 2>&1
grep -n 'FAILED\|TIMEOUT\|tests pass\|Executed\|Build completed\|ERROR:' ~/full.log | grep -v Wdeprecated | cut -c1-300 | tail -60
