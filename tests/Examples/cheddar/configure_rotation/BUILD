load("@heir//bazel/cheddar:config.bzl", "requires_cheddar")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:heir-translate.bzl", "heir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

heir_opt(
    name = "configure_rotation_emitc",
    src = "configure_rotation.mlir",
    generated_filename = "configure_rotation_emitc.mlir",
    pass_flags = [
        "--cheddar-configure-crypto-context=entry-function=rotate",
        "--cheddar-to-emitc",
    ],
)

heir_translate(
    name = "configure_rotation_cpp_raw",
    src = ":configure_rotation_emitc.mlir",
    generated_filename = "configure_rotation_raw.cc",
    pass_flags = ["--mlir-to-cpp"],
)

genrule(
    name = "configure_rotation_lib_src",
    srcs = [":configure_rotation_raw.cc"],
    outs = ["configure_rotation_lib.cc"],
    cmd = """cat > $@ <<'PRELUDE_EOF'
// AUTO-GENERATED: do not edit. See tests/Examples/cheddar/configure_rotation/BUILD.
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"
using namespace cheddar;
using word = uint64_t;
using Complex = std::complex<double>;
PRELUDE_EOF
cat $(location :configure_rotation_raw.cc) >> $@
""",
)

cc_library(
    name = "configure_rotation_lib",
    srcs = [":configure_rotation_lib_src"],
    target_compatible_with = requires_cheddar(),
    deps = ["@cheddar"],
)

cc_test(
    name = "configure_rotation_test",
    timeout = "long",
    srcs = ["configure_rotation_test.cpp"],
    linkstatic = False,
    tags = [
        "exclusive",
        "notap",
    ],
    target_compatible_with = requires_cheddar(),
    deps = [
        ":configure_rotation_lib",
        "@cheddar",
        "@googletest//:gtest_main",
    ],
)
