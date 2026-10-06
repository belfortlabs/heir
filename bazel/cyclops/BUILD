load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# The declarations HEIR uses from Cyclops' cyclops_planner.h. heir-opt does not
# link the planner; it loads libcyclops_planner from the library path when it
# plans keys.
cc_library(
    name = "planner",
    hdrs = ["cyclops_planner.h"],
    includes = ["."],
)
