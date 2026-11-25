# usage: mkpkg.sh <slots> "<extra torch-linalg-to-ckks options>" Model...
# Creates the bazel package tests/Examples/lattigo/ckks/corpus_e2e_<tag> in ~/sky_workdir
# with one heir_lattigo_lib per model. TAG env var names the package (default: slots).
N=$1; OPTS=$2; shift 2
tag=${TAG:-$N}
pkg=~/sky_workdir/tests/Examples/lattigo/ckks/corpus_e2e_$tag
mkdir -p $pkg
{
cat <<HDR
load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])
HDR
for m in "$@"; do
  src=~/sky_workdir/tmp_linear_kernel_proto/corpus/michiel_$m.mlir
  fn=$(grep -o 'func.func @[A-Za-z0-9_]*' $src | head -1 | cut -d@ -f2)
  cp $src $pkg/$fn.mlir
  cat <<LIB

heir_lattigo_lib(
    name = "$fn",
    go_library_name = "$fn",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=$N $OPTS",
        "--scheme-to-lattigo",
    ],
    mlir_src = "$fn.mlir",
    split_preprocessing = True,
)
LIB
  if [ -f $pkg/${fn}_test.go ]; then
  cat <<TST

go_test(
    name = "${fn}_test",
    size = "enormous",
    srcs = ["${fn}_test.go"],
    embed = [":$fn"],
    deps = [":${fn}_utils"],
)
TST
  fi
done
} > $pkg/BUILD
echo $pkg
