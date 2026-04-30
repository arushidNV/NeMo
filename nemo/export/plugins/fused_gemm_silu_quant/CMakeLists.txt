cmake_minimum_required(VERSION 3.18)
project(FusedGemmSiluQuant CUDA CXX)

# Require C++17 and CUDA; multi-arch: SM90 (H100) + SM89 (RTX 6000 Ada)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "90;90a;89")

# BUILD_SM90=OFF: SM89-only (no CUTLASS). Use on RTX 6000 Ada when CUTLASS is incompatible.
# BUILD_SM90=ON:  Build SM90 CUTLASS kernel too (requires CUTLASS_DIR).
option(BUILD_SM90 "Build CUTLASS SM90 kernel for H100 (requires CUTLASS)" ON)

# ============================================================================
# Find dependencies
# ============================================================================

set(CUTLASS_DIR "" CACHE PATH "CUTLASS root (parent of include/). Required only if BUILD_SM90=ON.")
if(BUILD_SM90)
    if(NOT CUTLASS_DIR)
        find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h
            PATHS
                /opt/cutlass/include
                /usr/local/cutlass/include
                $ENV{CUTLASS_DIR}/include
                ${CMAKE_SOURCE_DIR}/../../../cutlass/include
        )
        if(CUTLASS_INCLUDE_DIR)
            get_filename_component(CUTLASS_DIR ${CUTLASS_INCLUDE_DIR} DIRECTORY)
            message(STATUS "Found CUTLASS at: ${CUTLASS_DIR}")
        else()
            message(FATAL_ERROR
                "BUILD_SM90=ON but CUTLASS not found. Set -DCUTLASS_DIR=/path/to/cutlass or use -DBUILD_SM90=OFF for SM89-only.")
        endif()
    else()
        message(STATUS "Using CUTLASS at: ${CUTLASS_DIR}")
    endif()
else()
    message(STATUS "BUILD_SM90=OFF: building SM89-only (no CUTLASS)")
endif()

# TensorRT
find_path(TRT_INCLUDE_DIR NvInfer.h
    PATHS
        /usr/include/x86_64-linux-gnu
        /usr/local/tensorrt/include
        /usr/include
        $ENV{TRT_DIR}/include
)
find_library(TRT_LIB nvinfer
    PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/local/tensorrt/lib
        /usr/lib
        $ENV{TRT_DIR}/lib
)

if(NOT TRT_INCLUDE_DIR OR NOT TRT_LIB)
    message(FATAL_ERROR "TensorRT not found. Set TRT_DIR or install TensorRT.")
endif()
message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib: ${TRT_LIB}")

# CUDA
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# Build the plugin shared library (SM89 always; SM90 optional)
# ============================================================================

set(PLUGIN_SOURCES plugin.cpp kernel_sm89.cu)
if(BUILD_SM90)
    list(APPEND PLUGIN_SOURCES kernel_sm90.cu)
else()
    list(APPEND PLUGIN_SOURCES kernel_sm90_stub.cpp)
endif()

add_library(fused_gemm_silu_quant SHARED ${PLUGIN_SOURCES})

# kernel_sm90.cu uses CUTLASS, compiled only for Hopper; kernel_sm89.cu for Ada.
if(BUILD_SM90)
    set_source_files_properties(kernel_sm90.cu PROPERTIES CUDA_ARCHITECTURES "90;90a")
endif()
set_source_files_properties(kernel_sm89.cu PROPERTIES CUDA_ARCHITECTURES "89")

set(PLUGIN_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${TRT_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
if(BUILD_SM90 AND CUTLASS_DIR)
    list(APPEND PLUGIN_INCLUDES ${CUTLASS_DIR}/include ${CUTLASS_DIR}/tools/util/include)
endif()
target_include_directories(fused_gemm_silu_quant PRIVATE ${PLUGIN_INCLUDES})

target_link_libraries(fused_gemm_silu_quant PRIVATE
    ${TRT_LIB}
    CUDA::cudart
)

# CUTLASS needs relaxed constexpr for Hopper kernel; -gencode comes from CUDA_ARCHITECTURES
target_compile_options(fused_gemm_silu_quant PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Output
set_target_properties(fused_gemm_silu_quant PROPERTIES
    OUTPUT_NAME "fused_gemm_silu_quant"
    PREFIX "lib"
    SUFFIX ".so"
)

message(STATUS "Plugin will be built as: ${CMAKE_BINARY_DIR}/libfused_gemm_silu_quant.so")
message(STATUS "Usage: trtexec --plugins=libfused_gemm_silu_quant.so --loadEngine=...")
