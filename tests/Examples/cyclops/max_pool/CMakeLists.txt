cmake_minimum_required(VERSION 3.30)
project(heir_cyclops_max_pool LANGUAGES C CXX ASM)
set(CMAKE_CXX_STANDARD 20)
if(NOT CYCLOPS_SOURCE_DIR OR NOT HEIR_BIN_DIR)
  message(FATAL_ERROR "Set CYCLOPS_SOURCE_DIR and HEIR_BIN_DIR")
endif()
# The CPU and GPU runtimes define the same Cyclops symbols, so each side is a
# separate configure: run_max_pool_e2e.sh builds both.
option(MAX_POOL_SERVER "Build the GPU server instead of the CPU client" OFF)
get_filename_component(HEIR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)

if(MAX_POOL_SERVER)
  set(BUILD_UNITTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_BENCH OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  # MaxPool is in the extension library.
  set(ENABLE_EXTENSION ON CACHE BOOL "" FORCE)
  set(ENABLE_NN OFF CACHE BOOL "" FORCE)
  find_package(CUDAToolkit REQUIRED)
  add_subdirectory("${CYCLOPS_SOURCE_DIR}" cyclops)
  # CUDA 13 installs cuda/std under a separate CCCL include root.
  foreach(cuda_include IN LISTS CUDAToolkit_INCLUDE_DIRS)
    if(IS_DIRECTORY "${cuda_include}/cccl")
      target_include_directories(cyclops SYSTEM PUBLIC "${cuda_include}/cccl")
    endif()
  endforeach()
  # Cyclops uses quadmath.h, which Clang does not search for in GCC's private
  # include directory. Keep those headers after Clang's own.
  foreach(link_dir IN LISTS CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES)
    if(EXISTS "${link_dir}/include/quadmath.h")
      target_compile_options(cyclops PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-idirafter${link_dir}/include>")
      break()
    endif()
  endforeach()
  set(side server)
  set(runtime cyclops CUDA::cudart)
else()
  set(BUILD_CLIENT_TEST OFF CACHE BOOL "" FORCE)
  set(BUILD_CLIENT_BENCH OFF CACHE BOOL "" FORCE)
  add_subdirectory("${CYCLOPS_SOURCE_DIR}/client" cyclops)
  set(side client)
  set(runtime cyclops_client)
endif()

set(model "${CMAKE_CURRENT_SOURCE_DIR}/max_pool.mlir")
# The generated file must not share the model's name: CMake resolves a
# relative DEPENDS against the source directory first.
add_custom_command(
  OUTPUT max_pool_emitc.mlir
  COMMAND "${HEIR_BIN_DIR}/heir-opt" "${model}"
    "--annotate-module=backend=cheddar scheme=ckks"
    "--torch-linalg-to-ckks=min-slot-count=4096"
    "--scheme-to-cheddar=entry-function=max_pool runtime=cyclops"
    --cheddar-to-emitc --cheddar-emitc-entry-interface=runtime=cyclops
    -o max_pool_emitc.mlir
  DEPENDS "${model}" "${HEIR_BIN_DIR}/heir-opt")
foreach(pair "header;h" "source;cpp")
  list(GET pair 0 kind)
  list(GET pair 1 suffix)
  add_custom_command(
    OUTPUT "max_pool_${side}.${suffix}"
    COMMAND "${HEIR_BIN_DIR}/heir-translate" max_pool_emitc.mlir --mlir-to-cpp
      "--file-id=${side}_${kind}" -o "max_pool_${side}.${suffix}"
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/max_pool_emitc.mlir"
      "${HEIR_BIN_DIR}/heir-translate")
endforeach()

add_executable(max_pool_${side} ${side}_main.cpp
  "${CMAKE_CURRENT_BINARY_DIR}/max_pool_${side}.h"
  "${CMAKE_CURRENT_BINARY_DIR}/max_pool_${side}.cpp")
# The generated unit only calls the library; optimizing it buys nothing.
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/max_pool_${side}.cpp"
  PROPERTIES COMPILE_OPTIONS "-O0")
target_include_directories(max_pool_${side} PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" "${HEIR_ROOT}/include")
target_link_libraries(max_pool_${side} PRIVATE ${runtime})
if(MAX_POOL_SERVER)
  target_compile_definitions(max_pool_${side} PRIVATE DEFAULT_STREAM=cudaStreamPerThread)
endif()
