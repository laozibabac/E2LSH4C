cmake_minimum_required(VERSION 3.20)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" E2LSH_VERSION_FILE)
string(STRIP "${E2LSH_VERSION_FILE}" E2LSH_VERSION_FILE)

project(native-l2-e2lsh-chamfer VERSION ${E2LSH_VERSION_FILE} LANGUAGES CXX)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(FATAL_ERROR "Use GCC or Clang (MinGW-w64 GCC 13 on Windows). MSVC is not supported.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(E2LSH_FAST_MATH "Add -ffast-math" OFF)
option(E2LSH_EARLY_EXIT "Enable NN early-break ablation" OFF)
set(CHAMFER_DIM 784 CACHE STRING "Compile-time DIM for compare_perf (Fashion-MNIST)")

find_package(OpenMP REQUIRED)

if(WIN32)
  add_compile_definitions(_WIN32_WINNT=0x0601 NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

set(VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/neurips2025_chamfer")
set(VENDOR_SRCS
  ${VENDOR_DIR}/quad_tree.cpp
  ${VENDOR_DIR}/balanced_binary_tree.cpp
  ${VENDOR_DIR}/datatype.cpp
  ${VENDOR_DIR}/aspect_ratio_depth.cpp
  ${VENDOR_DIR}/BruteForceNN.cpp
  ${VENDOR_DIR}/data_preprocess.cpp
  ${VENDOR_DIR}/baseline/benchmark.cpp
  ${VENDOR_DIR}/misc.cpp
  ${VENDOR_DIR}/random.cpp
  ${VENDOR_DIR}/point.cpp
  ${VENDOR_DIR}/ANN.cpp
  ${VENDOR_DIR}/LSH.cpp
  ${VENDOR_DIR}/ANN-Oracle.cpp
  ${VENDOR_DIR}/text_embedding.cpp
)

function(e2lsh_apply_flags tgt)
  target_compile_options(${tgt} PRIVATE
    -O3 -march=native -Wall -Wno-deprecated-declarations -Wno-unused-variable
    -fopenmp)
  # Outlier GIST QuadTree walks deep AVL samplers. The 2MB default overflows.
  target_link_options(${tgt} PRIVATE -fopenmp -Wl,--stack,134217728)
  if(E2LSH_FAST_MATH)
    target_compile_options(${tgt} PRIVATE -ffast-math)
  endif()
  if(E2LSH_EARLY_EXIT)
    target_compile_definitions(${tgt} PRIVATE E2LSH_NN_EARLY_EXIT=1)
  endif()
  target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
endfunction()

function(e2lsh_add_vendor name dim)
  add_library(${name} OBJECT ${VENDOR_SRCS})
  target_include_directories(${name} PUBLIC
    ${VENDOR_DIR} ${VENDOR_DIR}/baseline ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(${name} PUBLIC CHAMFER_DIM=${dim})
  e2lsh_apply_flags(${name})
endfunction()

function(e2lsh_add_compare name dim)
  set(vlib vendor_${name})
  e2lsh_add_vendor(${vlib} ${dim})
  add_executable(${name} src/compare_main.cpp $<TARGET_OBJECTS:${vlib}>)
  target_include_directories(${name} PRIVATE
    ${VENDOR_DIR} ${VENDOR_DIR}/baseline ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(${name} PRIVATE CHAMFER_DIM=${dim})
  e2lsh_apply_flags(${name})
endfunction()

# Fashion-MNIST is 784-d and is the default binary.
e2lsh_add_compare(compare_perf ${CHAMFER_DIM})

e2lsh_add_compare(compare_perf_sift 128)
set_target_properties(compare_perf_sift PROPERTIES EXCLUDE_FROM_ALL TRUE)

e2lsh_add_compare(compare_perf_gist 960)
set_target_properties(compare_perf_gist PROPERTIES EXCLUDE_FROM_ALL TRUE)

e2lsh_add_compare(compare_perf_d300 300)
set_target_properties(compare_perf_d300 PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_executable(exact_cache_f32 src/exact_cache_f32.cpp)
target_include_directories(exact_cache_f32 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
e2lsh_apply_flags(exact_cache_f32)
set_target_properties(exact_cache_f32 PROPERTIES EXCLUDE_FROM_ALL TRUE)
