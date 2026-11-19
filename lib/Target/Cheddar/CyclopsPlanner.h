#ifndef LIB_TARGET_CHEDDAR_CYCLOPSPLANNER_H_
#define LIB_TARGET_CHEDDAR_CYCLOPSPLANNER_H_

#include <string>

#include "cyclops_planner.h"  // from @heir//bazel/cyclops:planner

namespace mlir {
namespace heir {
namespace cheddar {

// The functions of Cyclops' planner that HEIR calls. heir-opt does not link the
// proprietary planner: it loads libcyclops_planner from the library path when
// it first needs it.
// clang-format off
#define CYCLOPS_PLANNER_FUNCTIONS(X) \
  X(cyclops_free_error) \
  X(cyclops_params_create) \
  X(cyclops_params_set_dense_hamming_weight) \
  X(cyclops_params_set_sparse_hamming_weight) \
  X(cyclops_params_set_max_log_pq) \
  X(cyclops_params_set_max_key_switch_aux) \
  X(cyclops_params_set_level_specific_ks) \
  X(cyclops_params_free) \
  X(cyclops_evk_request_create) \
  X(cyclops_evk_request_add_request) \
  X(cyclops_evk_request_request_multiplication_key) \
  X(cyclops_evk_request_keys) \
  X(cyclops_evk_request_free) \
  X(cyclops_add_linear_transform_required_keys) \
  X(cyclops_default_mod1) \
  X(cyclops_add_bootstrap_required_rotations) \
  X(cyclops_mod1_depth)

// The MaxPool functions, which only a planner with MaxPool support has. Only a
// program with a max pool needs them.
#define CYCLOPS_PLANNER_MAX_POOL_FUNCTIONS(X) \
  X(cyclops_default_max_pool_config) \
  X(cyclops_max_pool_minimum_input_level) \
  X(cyclops_max_pool_minimum_bootstrap_end_level) \
  X(cyclops_max_pool_plan_levels) \
  X(cyclops_add_max_pool_required_keys)
// clang-format on

struct CyclopsPlanner {
#define DECLARE_FUNCTION(name) decltype(&::name) name = nullptr;
  CYCLOPS_PLANNER_FUNCTIONS(DECLARE_FUNCTION)
  CYCLOPS_PLANNER_MAX_POOL_FUNCTIONS(DECLARE_FUNCTION)
#undef DECLARE_FUNCTION
  // Why loading failed; empty once every function of
  // CYCLOPS_PLANNER_FUNCTIONS is found.
  std::string error;
  // Why the MaxPool functions are not available; empty once every function of
  // CYCLOPS_PLANNER_MAX_POOL_FUNCTIONS is found.
  std::string maxPoolError;
};

// The planner, loaded on the first call.
const CyclopsPlanner& getCyclopsPlanner();

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_CHEDDAR_CYCLOPSPLANNER_H_
