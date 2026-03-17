cmake_minimum_required(VERSION 3.16)
project(gpu_toy LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA configuration
option(ENABLE_CUDA "Enable CUDA builds" ON)
option(ENABLE_OPENMP "Enable OpenMP for CPU targets" ON)

if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP enabled for CPU targets.")
    else()
        message(WARNING "ENABLE_OPENMP is ON but OpenMP was not found. CPU targets will run without OpenMP.")
    endif()
endif()

if(ENABLE_CUDA AND NOT CMAKE_CUDA_COMPILER)
    message(WARNING "ENABLE_CUDA is ON but no CUDA compiler was found. CUDA-specific targets will be skipped.")
endif()

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    if(NOT DEFINED CUDA_ARCHITECTURES OR CUDA_ARCHITECTURES STREQUAL "")
        # Default architectures; override with -DCUDA_ARCHITECTURES="..."
        set(CUDA_ARCHITECTURES "86" CACHE STRING "CUDA GPU architectures to build for" FORCE)
    endif()

    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
    message(STATUS "CUDA enabled. Using architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra)
endif()

# Collect all source files in src/
file(GLOB SOURCES 
    "demo/warp_example.cu"
    "demo/stride_example.cu"
)

# Create one executable per source file
foreach(SOURCE_FILE ${SOURCES})
    # Extract filename without extension
    get_filename_component(EXEC_NAME ${SOURCE_FILE} NAME_WE)
    
    # Create executable
    add_executable(${EXEC_NAME} ${SOURCE_FILE})

    # Make headers in demo/utils available (e.g., csv_writer.h)
    target_include_directories(${EXEC_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/demo/utils
    )

    # Link the executable to the CUDA runtime
    target_link_libraries(${EXEC_NAME} PRIVATE ${CUDA_LIBRARIES})
endforeach()

# CPU comparison target
add_executable(cpu_example "demo/CPU_example.cpp")
target_include_directories(cpu_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/demo/utils
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cpu_example PRIVATE OpenMP::OpenMP_CXX)
endif()

# Print available targets
message(STATUS "Building CUDA examples:")
foreach(SOURCE_FILE ${SOURCES})
    get_filename_component(EXEC_NAME ${SOURCE_FILE} NAME_WE)
    message(STATUS "  - ${EXEC_NAME}")
endforeach()
message(STATUS "Building CPU example:")
message(STATUS "  - cpu_example")
