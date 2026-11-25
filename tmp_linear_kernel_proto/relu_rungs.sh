# Measures the worst-case error of every ReLU approximation rung on [-1, 1]
# (and how it behaves up to 10 % and 25 % outside), as lowered by the pipeline
# and evaluated in single precision. Prints RUNG_ERRORS for relu_exp.sh.
cd ~/sky_workdir
d=~/relu/rungs; mkdir -p $d
Q="--config=remote --output_filter=^\$ --noshow_progress --ui_event_filters=-info,-warning,-stdout,-stderr"
n=8001
for r in deg5 deg15 deg31 deg63 composite; do
  case $r in composite) attr='relu_method = "composite"';; *) attr="relu_method = \"polynomial\", degree = ${r#deg} : i32";; esac
  cat > $d/$r.mlir <<MLIR
func.func @rung(%x: tensor<${n}xf32> {secret.secret}) -> tensor<${n}xf32> {
  %z = arith.constant dense<0.0> : tensor<${n}xf32>
  %r = arith.maximumf %x, %z {$attr, domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : tensor<${n}xf32>
  return %r : tensor<${n}xf32>
}
MLIR
  { bazel run $Q //tools:heir-opt -- --wrap-generic --polynomial-approximation --lower-polynomial-eval --canonicalize --cse --secret-forget-secrets --heir-polynomial-to-llvm $d/$r.mlir -o $d/$r.low.mlir &&
    bazel run $Q @llvm-project//mlir:mlir-translate -- --mlir-to-llvmir $d/$r.low.mlir -o $d/$r.ll &&
    sed -i "s/@rung\b/@${r}_rung/g; s/@_mlir_ciface_rung\b/@_mlir_ciface_${r}_rung/g" $d/$r.ll &&
    bazel run $Q @llvm-project//llvm:llc -- -filetype=obj -relocation-model=pic $d/$r.ll -o $d/$r.o; } > $d/$r.log 2>&1 || echo "FAILED $r"
done
cat > $d/driver.cpp <<CPP
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
template <typename T, int N> struct MemRef { T* basePtr; T* data; int64_t offset; int64_t sizes[N]; int64_t strides[N]; };
using V = MemRef<float, 1>;
extern "C" {
void _mlir_ciface_deg5_rung(V*, V*); void _mlir_ciface_deg15_rung(V*, V*); void _mlir_ciface_deg31_rung(V*, V*);
void _mlir_ciface_deg63_rung(V*, V*); void _mlir_ciface_composite_rung(V*, V*);
}
int main() {
  const int n = $n;
  void (*f[])(V*, V*) = {_mlir_ciface_deg5_rung, _mlir_ciface_deg15_rung, _mlir_ciface_deg31_rung, _mlir_ciface_deg63_rung, _mlir_ciface_composite_rung};
  const char* names[] = {"deg5", "deg15", "deg31", "deg63", "composite"};
  double widths[] = {1.0, 1.1, 1.25};
  double unit[5];
  for (int k = 0; k < 5; ++k) {
    std::printf("%s", names[k]);
    for (double w : widths) {
      std::vector<float> x(n);
      for (int i = 0; i < n; ++i) x[i] = -w + 2 * w * i / (n - 1);
      std::vector<float> copy = x;
      V in{copy.data(), copy.data(), 0, {n}, {1}}, out;
      f[k](&out, &in);
      double err = 0;
      for (int i = 0; i < n; ++i) err = std::fmax(err, std::fabs(out.data[out.offset + i] - std::fmax(x[i], 0.0f)));
      std::printf("\tmaxerr[-%.2f,%.2f]=%.4g", w, w, err);
      if (w == 1.0) unit[k] = err;
    }
    std::printf("\n");
  }
  std::printf("RUNG_ERRORS=%.4g:%.4g:%.4g:%.4g:%.4g\n", unit[0], unit[1], unit[2], unit[3], unit[4]);
}
CPP
clang++ -std=c++20 -O1 $d/driver.cpp tests/llvm_runner/memrefCopy.cpp -I . $d/deg5.o $d/deg15.o $d/deg31.o $d/deg63.o $d/composite.o -o $d/rungs 2> $d/link.log && $d/rungs | tee $d/rungs.txt
grep -a -m2 "error" $d/*.log | cut -c1-300
