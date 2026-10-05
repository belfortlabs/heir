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
    # bn_s_mp_rand_platform.c calls the random sources of other platforms
    # behind constant-false MP_HAS() checks, and relies on the optimizer to
    # drop those calls. Without it (fastbuild), a link fails on s_read_*.
    copts = ["-O2"],
    includes = ["."],
    visibility = ["//visibility:public"],
)
