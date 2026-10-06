#ifndef BAZEL_CYCLOPS_CYCLOPS_PLANNER_H_
#define BAZEL_CYCLOPS_CYCLOPS_PLANNER_H_

// The functions of Cyclops' planner/include/cyclops_planner.h (at commit
// c4b4d20487ff855811f3d8de0592ab6871d70ff3), and the constants HEIR uses, for
// loading the planner at run time.
// The constants keep Cyclops' values; the structs keep its layout.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { CYCLOPS_KEY_MODE_INHERIT = 0, CYCLOPS_KEY_MODE_DEFAULT = 1 };
enum { CYCLOPS_BOOT_VARIANT_IMAGINARY_REMOVING = 1 };
enum { CYCLOPS_RING_STANDARD = 0 };
enum {
  CYCLOPS_MOD1_COS_HK = 0,
  CYCLOPS_MOD1_COS_HK_EVEN = 1,
  CYCLOPS_MOD1_SIN_CHEBY = 2,
  CYCLOPS_MOD1_COS_CHEBY = 3,
  CYCLOPS_MOD1_EXP_COMPLEX = 4
};
enum {
  CYCLOPS_MOD1_INV_ARCSINE_TAYLOR = 0,
  CYCLOPS_MOD1_INV_ARCSINE_CHEBY = 1
};

typedef struct {
  int type;
  int degree;
  int interval;
  int log_interval_reduction;
  double scaling;
  int inv_degree;
  int inv_type;
  double inv_interval;
} cyclops_mod1;

typedef struct {
  int64_t family;
  int64_t rotation;
  int64_t level;
  int64_t key_mode;
  int64_t required_num_aux;
} cyclops_key;

typedef struct cyclops_params cyclops_params;
typedef struct cyclops_evk_request cyclops_evk_request;

void cyclops_free_error(char* error);

cyclops_params* cyclops_params_create(
    int log_degree, double base_scale, int default_encryption_level,
    const int32_t* level_config, size_t num_level_config,
    const uint64_t* main_primes, size_t num_main_primes,
    const uint64_t* aux_primes, size_t num_aux_primes,
    const uint64_t* ter_primes, size_t num_ter_primes,
    const int32_t additional_base[2], int default_num_aux, int ring_type,
    int word_bits, char** error);
int cyclops_params_set_dense_hamming_weight(cyclops_params* params, int weight,
                                            char** error);
int cyclops_params_set_sparse_hamming_weight(cyclops_params* params, int weight,
                                             char** error);
int cyclops_params_set_max_log_pq(cyclops_params* params, double max_log_pq,
                                  char** error);
int cyclops_params_set_max_key_switch_aux(cyclops_params* params,
                                          int max_key_switch_aux, char** error);
int cyclops_params_set_level_specific_ks(cyclops_params* params, int enabled,
                                         char** error);
void cyclops_params_free(cyclops_params* params);

cyclops_evk_request* cyclops_evk_request_create(void);
int cyclops_evk_request_add_request(cyclops_evk_request* request, int rot_idx,
                                    int level, int key_mode,
                                    int required_num_aux, char** error);
int cyclops_evk_request_request_conjugation_key(cyclops_evk_request* request,
                                                int level, int key_mode,
                                                int required_num_aux,
                                                char** error);
int cyclops_evk_request_request_multiplication_key(cyclops_evk_request* request,
                                                   int level, int key_mode,
                                                   int required_num_aux,
                                                   char** error);
int cyclops_evk_request_request_rotated_multiplication_key(
    cyclops_evk_request* request, int rot_idx, int level, int key_mode,
    int required_num_aux, char** error);
size_t cyclops_evk_request_keys(const cyclops_evk_request* request,
                                cyclops_key* keys, size_t capacity);
void cyclops_evk_request_free(cyclops_evk_request* request);

int cyclops_add_linear_transform_required_keys(
    cyclops_evk_request* request, const cyclops_params* params, int width,
    const int32_t* diagonal_indices, size_t num_diagonal_indices, int level,
    int hint_num_bs, int hint_num_gs, int key_mode, char** error);
cyclops_mod1 cyclops_default_mod1(void);
int cyclops_add_bootstrap_required_rotations(
    cyclops_evk_request* request, const cyclops_params* params,
    int num_cts_levels, int num_stc_levels, int log_message_ratio,
    const cyclops_mod1* mod1, int num_slots, int variant, int key_mode,
    char** error);
int cyclops_add_bootstrap_non_dft_required_keys(
    cyclops_evk_request* request, int max_level, int num_cts_levels,
    int num_stc_levels, int log_message_ratio, const cyclops_mod1* mod1,
    int num_slots, int max_slots, int key_mode, char** error);
int cyclops_mod1_depth(const cyclops_mod1* mod1, int log_message_ratio,
                       int* depth, char** error);
int cyclops_add_mod1_required_keys(cyclops_evk_request* request,
                                   const cyclops_mod1* mod1, int level,
                                   int log_message_ratio, int key_mode,
                                   char** error);

#ifdef __cplusplus
}
#endif

#endif  // BAZEL_CYCLOPS_CYCLOPS_PLANNER_H_
