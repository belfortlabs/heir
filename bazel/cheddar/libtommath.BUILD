load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "tommath",
    srcs = glob(
        ["*.c"],
        allow_empty = True,
        exclude = [
            "demo/**",
            "etc/**",
            "mtest/**",
        ],
    ),
    hdrs = glob(
        ["*.h"],
        allow_empty = True,
    ),
    # libtommath drops calls to unavailable entropy sources by dead-code
    # elimination, so it needs optimization even in fastbuild. Apple clang does
    # not announce IEEE 754 doubles, without which mp_set_double is left out.
    copts = ["-O2"] + select({
        "@platforms//os:macos": ["-D__STDC_IEC_559__=1"],
        "//conditions:default": [],
    }),
    includes = ["."],
    visibility = ["//visibility:public"],
)
