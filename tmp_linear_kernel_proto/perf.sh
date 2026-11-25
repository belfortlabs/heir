# usage: perf.sh <model> <slots> <seconds before sampling>
cd ~/sky_workdir
m=$1; N=$2; W=$3
bazel run --config=remote --output_filter='^$' --noshow_progress --ui_event_filters=-info,-warning,-stdout,-stderr --run_under="timeout 240" //tools:heir-opt -- "--mlir-to-plaintext-backend=plaintext-size=$N" ~/sky_workdir/tmp_linear_kernel_proto/corpus/michiel_$m.mlir -o /dev/null > ~/perfrun.log 2>&1 &
sleep $W
pid=$(pgrep -n -x heir-opt)
echo "pid=$pid rss=$(ps -o rss= -p $pid)"
cd /tmp
sudo perf record -F 199 -p $pid -g -o /tmp/perf.data -- sleep 8 > /tmp/perfrec.log 2>&1
tail -2 /tmp/perfrec.log | cut -c1-200
sudo perf report -i /tmp/perf.data --stdio --no-children --percent-limit 3 --max-stack 45 2> /tmp/perfrep.err | grep -v '^#' | grep -v '^$' | c++filt | cut -c1-190 | head -90
tail -2 /tmp/perfrep.err | cut -c1-200
pkill -x heir-opt
