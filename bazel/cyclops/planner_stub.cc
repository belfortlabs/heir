// Stand-in for the Cyclops key planner, so heir-opt builds without the private
// Cyclops source. Every planning call fails; the real libcyclops_planner,
// found on the library path at run time, replaces this library.

#include <cstdlib>
#include <cstring>

#include "cyclops_planner.h"

namespace {

int fail(char** error) {
  if (error)
    *error = strdup(
        "heir-opt was built with the Cyclops planner stub; put the real "
        "libcyclops_planner on the library path, or build with "
        "--//bazel/cyclops:real_planner");
  return -1;
}

}  // namespace

extern "C" {

void cyclops_free_error(char* error) { std::free(error); }

cyclops_params* cyclops_params_create(int, double, int, const int32_t*, size_t,
                                      const uint64_t*, size_t, const uint64_t*,
                                      size_t, const uint64_t*, size_t,
                                      const int32_t[2], int, int, int,
                                      char** error) {
  fail(error);
  return nullptr;
}

int cyclops_params_set_dense_hamming_weight(cyclops_params*, int,
                                            char** error) {
  return fail(error);
}

int cyclops_params_set_sparse_hamming_weight(cyclops_params*, int,
                                             char** error) {
  return fail(error);
}

int cyclops_params_set_max_log_pq(cyclops_params*, double, char** error) {
  return fail(error);
}

int cyclops_params_set_max_key_switch_aux(cyclops_params*, int, char** error) {
  return fail(error);
}

int cyclops_params_set_level_specific_ks(cyclops_params*, int, char** error) {
  return fail(error);
}

void cyclops_params_free(cyclops_params*) {}

cyclops_evk_request* cyclops_evk_request_create(void) { return nullptr; }

int cyclops_evk_request_add_request(cyclops_evk_request*, int, int, int, int,
                                    char** error) {
  return fail(error);
}

int cyclops_evk_request_request_conjugation_key(cyclops_evk_request*, int, int,
                                                int, char** error) {
  return fail(error);
}

int cyclops_evk_request_request_multiplication_key(cyclops_evk_request*, int,
                                                   int, int, char** error) {
  return fail(error);
}

int cyclops_evk_request_request_rotated_multiplication_key(cyclops_evk_request*,
                                                           int, int, int, int,
                                                           char** error) {
  return fail(error);
}

size_t cyclops_evk_request_keys(const cyclops_evk_request*, cyclops_key*,
                                size_t) {
  return 0;
}

void cyclops_evk_request_free(cyclops_evk_request*) {}

int cyclops_add_linear_transform_required_keys(cyclops_evk_request*,
                                               const cyclops_params*, int,
                                               const int32_t*, size_t, int, int,
                                               int, int, char** error) {
  return fail(error);
}

cyclops_mod1 cyclops_default_mod1(void) { return {}; }

int cyclops_add_bootstrap_required_rotations(cyclops_evk_request*,
                                             const cyclops_params*, int, int,
                                             int, const cyclops_mod1*, int, int,
                                             int, char** error) {
  return fail(error);
}

int cyclops_mod1_depth(const cyclops_mod1*, int, int*, char** error) {
  return fail(error);
}

int cyclops_add_bootstrap_non_dft_required_keys(cyclops_evk_request*, int, int,
                                                int, int, const cyclops_mod1*,
                                                int, int, int, char** error) {
  return fail(error);
}

int cyclops_add_mod1_required_keys(cyclops_evk_request*, const cyclops_mod1*,
                                   int, int, int, char** error) {
  return fail(error);
}

}  // extern "C"
