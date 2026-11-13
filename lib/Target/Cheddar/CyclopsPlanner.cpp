#include "lib/Target/Cheddar/CyclopsPlanner.h"

#include <dlfcn.h>

#include <string>

namespace mlir {
namespace heir {
namespace cheddar {

const CyclopsPlanner& getCyclopsPlanner() {
  static const CyclopsPlanner loaded = [] {
    CyclopsPlanner result;
#ifdef __APPLE__
    void* library =
        dlopen("@rpath/libcyclops_planner.dylib", RTLD_NOW | RTLD_LOCAL);
#else
    void* library = dlopen("libcyclops_planner.so", RTLD_NOW | RTLD_LOCAL);
#endif
    if (!library) {
      result.error = dlerror();
      result.maxPoolError = result.error;
      return result;
    }
#define LOAD_FUNCTION(name, errorField)                                     \
  result.name = reinterpret_cast<decltype(&::name)>(dlsym(library, #name)); \
  if (!result.name && result.errorField.empty()) {                          \
    result.errorField = dlerror();                                          \
  }
#define LOAD_REQUIRED(name) LOAD_FUNCTION(name, error)
#define LOAD_MAX_POOL(name) LOAD_FUNCTION(name, maxPoolError)
    CYCLOPS_PLANNER_FUNCTIONS(LOAD_REQUIRED)
    CYCLOPS_PLANNER_MAX_POOL_FUNCTIONS(LOAD_MAX_POOL)
#undef LOAD_MAX_POOL
#undef LOAD_REQUIRED
#undef LOAD_FUNCTION
    if (!result.error.empty() && result.maxPoolError.empty())
      result.maxPoolError = result.error;
    return result;
  }();
  return loaded;
}

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir
