cmake_minimum_required(VERSION 3.21)
project(heir_cyclops_client_test LANGUAGES C CXX ASM)
set(CMAKE_CXX_STANDARD 20)
if(NOT CYCLOPS_SOURCE_DIR OR NOT HEIR_BIN_DIR)
  message(FATAL_ERROR "Set CYCLOPS_SOURCE_DIR and HEIR_BIN_DIR")
endif()
get_filename_component(HEIR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(BUILD_CLIENT_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory("${CYCLOPS_SOURCE_DIR}/client" cyclops)
set(fixture "${HEIR_ROOT}/tests/Pipelines/scheme_to_cheddar/split_interface.mlir")
add_custom_command(
  OUTPUT split.mlir
  COMMAND "${HEIR_BIN_DIR}/heir-opt" "${fixture}"
    "--annotate-module=backend=cheddar scheme=ckks"
    "--mlir-to-ckks=min-slot-count=4096"
    "--scheme-to-cheddar=entry-function=split runtime=cyclops"
    --cheddar-to-emitc --cheddar-emitc-entry-interface=runtime=cyclops
    -o split.mlir
  DEPENDS "${fixture}" "${HEIR_BIN_DIR}/heir-opt")
foreach(pair "header;h" "source;cpp")
  list(GET pair 0 kind)
  list(GET pair 1 suffix)
  add_custom_command(
    OUTPUT "split_client.${suffix}"
    COMMAND "${HEIR_BIN_DIR}/heir-translate" split.mlir --mlir-to-cpp
      "--file-id=client_${kind}" -o "split_client.${suffix}"
    DEPENDS split.mlir "${HEIR_BIN_DIR}/heir-translate")
endforeach()
add_executable(client_roundtrip client_roundtrip.cpp split_client.h split_client.cpp)
target_include_directories(client_roundtrip PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" "${HEIR_ROOT}")
target_link_libraries(client_roundtrip PRIVATE cyclops_client)
enable_testing()
add_test(NAME client_roundtrip COMMAND client_roundtrip)
