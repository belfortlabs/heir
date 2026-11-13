#ifndef LIB_TARGET_CHEDDAR_MAXPOOLINTERFACES_H_
#define LIB_TARGET_CHEDDAR_MAXPOOLINTERFACES_H_

namespace mlir {
class DialectRegistry;

namespace heir {
namespace cheddar {

// Attaches the level model of Cyclops' MaxPool, which comes from the Cyclops
// planner library: ReducesLevelOpInterface and
// BootstrapsInternallyOpInterface to kernel.max_pool, and
// ReducesLevelOpInterface to linalg.pooling_ncw_max. The Kernel dialect only
// promises these interfaces, so that it does not depend on Cyclops.
void registerMaxPoolInterfaceExternalModels(DialectRegistry& registry);

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_CHEDDAR_MAXPOOLINTERFACES_H_
