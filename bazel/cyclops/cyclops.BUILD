load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "planner_sources",
    srcs = ["CMakeLists.txt"] + glob([
        "client/include/**",
        "client/src/core/**",
        "include/**",
        "planner/**",
        "src/extension/**",
    ]),
)

# The planner links libtommath statically. find_library would pick up the
# shared library rules_foreign_cc copies next to it, and it does not know
# Bazel's .pic.a name, so name the position-independent archive directly.
genrule(
    name = "tommath_archive",
    srcs = ["@libtommath//:tommath"],
    outs = ["libtommath.a"],
    cmd = "for f in $(locations @libtommath//:tommath); do case $$f in *.pic.a) cp $$f $@; exit 0;; esac; done; " +
          "for f in $(locations @libtommath//:tommath); do case $$f in *.a) cp $$f $@; exit 0;; esac; done; exit 1",
)

# Configure only planner/, never the CUDA server or standalone client project.
# The installed header and binary come from the same Cyclops revision.
cmake(
    name = "planner",
    build_data = ["@llvm//tools:clang++"],
    cache_entries = {
        "BUILD_PLANNER_TEST": "OFF",
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CXX_STANDARD": "20",
        "LIBTOMMATH": "$$EXT_BUILD_ROOT$$/$(execpath :tommath_archive)",
    },
    # In the target configuration, unlike build_data, so the archive is the
    # library deps would link.
    data = [":tommath_archive"],
    env = {
        # tommath.h; the planner's CMake project does not look for the header.
        "CPLUS_INCLUDE_PATH": "$${EXT_BUILD_DEPS}/include",
        # CMake links try-compiles from scratch directories, outside the execroot.
        "LLVM_CLANGXX": "$$EXT_BUILD_ROOT$$/$(execpath @llvm//tools:clang++)",
    },
    generate_args = ["-GNinja"],
    lib_source = ":planner_sources",
    out_shared_libs = select({
        "@platforms//os:macos": ["libcyclops_planner.dylib"],
        "//conditions:default": ["libcyclops_planner.so"],
    }),
    # Keep proprietary sources and planner artifacts off shared remote workers
    # and caches. This action runs on the trusted build host.
    tags = ["no-remote"],
    working_directory = "planner",
    deps = ["@libtommath//:tommath"],
)
