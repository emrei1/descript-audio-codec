cmake_minimum_required(VERSION 3.21)
project(dac_native LANGUAGES C CXX VERSION 0.1.0)

# Native CPU implementation of the Descript Audio Codec (encoder + decoder) and a desktop
# ".dac" file handler for Windows, Linux and macOS.
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release
#
# DAC_BACKEND = AUTO | ONEDNN | GEMM
#   AUTO picks oneDNN on x86-64 Windows/Linux and the tap-GEMM backend elsewhere
#   (Apple Accelerate on macOS, a CBLAS such as OpenBLAS on other CPUs).
set(DAC_BACKEND AUTO CACHE STRING "Convolution backend: AUTO, ONEDNN or GEMM")
set(DAC_SGEMM AUTO CACHE STRING "SGEMM for the GEMM backend: AUTO, ACCELERATE, CBLAS, DNNL or REFERENCE")
option(DAC_BUILD_PLAYER "Build the desktop .dac player / file handler" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _arch)
if (_arch MATCHES "x86_64|amd64|x64")
    set(DAC_X86 ON)
endif()

if (DAC_BACKEND STREQUAL "AUTO")
    if (DAC_X86 AND NOT APPLE)
        set(DAC_BACKEND ONEDNN)
    else()
        set(DAC_BACKEND GEMM)
    endif()
endif()

find_package(OpenMP)

# ---------------------------------------------------------------- oneDNN
set(_need_dnnl OFF)
if (DAC_BACKEND STREQUAL "ONEDNN" OR DAC_SGEMM STREQUAL "DNNL")
    set(_need_dnnl ON)
endif()
if (_need_dnnl)
    find_package(dnnl CONFIG QUIET)
    if (NOT dnnl_FOUND)
        message(STATUS "oneDNN not found - building v3.9.1 from source (this takes a few minutes)")
        include(FetchContent)
        set(ONEDNN_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
        set(ONEDNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(ONEDNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "" FORCE)
        set(ONEDNN_ENABLE_WORKLOAD INFERENCE CACHE STRING "" FORCE)
        set(ONEDNN_GPU_RUNTIME NONE CACHE STRING "" FORCE)
        if (OpenMP_CXX_FOUND)
            set(ONEDNN_CPU_RUNTIME OMP CACHE STRING "" FORCE)
        else()
            set(ONEDNN_CPU_RUNTIME SEQ CACHE STRING "" FORCE)
        endif()
        FetchContent_Declare(onednn
            GIT_REPOSITORY https://github.com/uxlfoundation/oneDNN.git
            GIT_TAG v3.9.1
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(onednn)
        add_library(DNNL::dnnl ALIAS dnnl)
    endif()
endif()

# ---------------------------------------------------------------- core library
add_library(dac_core STATIC
    src/common.cpp
    src/model.cpp
    src/snake.cpp
    src/engine.cpp
    src/dsp.cpp
    src/audio_io.cpp
    src/dacfile.cpp
    src/codec.cpp
    src/paths.cpp
)
target_include_directories(dac_core PUBLIC src PRIVATE third_party/dr_libs)

if (DAC_X86)
    target_sources(dac_core PRIVATE src/snake_avx2.cpp)
    if (MSVC)
        set_source_files_properties(src/snake_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(src/snake_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
endif()

if (DAC_BACKEND STREQUAL "ONEDNN")
    target_sources(dac_core PRIVATE src/backend_onednn.cpp)
    target_link_libraries(dac_core PUBLIC DNNL::dnnl)
else()
    target_sources(dac_core PRIVATE src/backend_gemm.cpp)
    if (DAC_SGEMM STREQUAL "AUTO")
        if (APPLE)
            set(DAC_SGEMM ACCELERATE)
        else()
            set(BLA_VENDOR OpenBLAS)
            find_package(BLAS)
            find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
            if (BLAS_FOUND AND CBLAS_INCLUDE_DIR)
                set(DAC_SGEMM CBLAS)
            else()
                message(WARNING "No CBLAS found - using the slow reference SGEMM")
                set(DAC_SGEMM REFERENCE)
            endif()
        endif()
    endif()
    if (DAC_SGEMM STREQUAL "ACCELERATE")
        target_compile_definitions(dac_core PRIVATE DAC_SGEMM_ACCELERATE)
        target_link_libraries(dac_core PUBLIC "-framework Accelerate")
    elseif (DAC_SGEMM STREQUAL "CBLAS")
        if (NOT BLAS_FOUND)
            find_package(BLAS REQUIRED)
            find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas REQUIRED)
        endif()
        target_compile_definitions(dac_core PRIVATE DAC_SGEMM_CBLAS)
        target_include_directories(dac_core PRIVATE ${CBLAS_INCLUDE_DIR})
        target_link_libraries(dac_core PUBLIC ${BLAS_LIBRARIES})
    elseif (DAC_SGEMM STREQUAL "DNNL")
        target_compile_definitions(dac_core PRIVATE DAC_SGEMM_DNNL)
        target_link_libraries(dac_core PUBLIC DNNL::dnnl)
    endif()
endif()

if (OpenMP_CXX_FOUND)
    target_link_libraries(dac_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if (MSVC)
    target_compile_definitions(dac_core PUBLIC UNICODE _UNICODE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    target_compile_options(dac_core PUBLIC /utf-8 /EHsc)
endif()
if (WIN32)
    target_link_libraries(dac_core PUBLIC shell32 ole32)
endif()
message(STATUS "DAC native: backend=${DAC_BACKEND} sgemm=${DAC_SGEMM} openmp=${OpenMP_CXX_FOUND}")

# ---------------------------------------------------------------- command-line tool
add_executable(dac-native src/cli_main.cpp)
target_link_libraries(dac-native PRIVATE dac_core)

# ---------------------------------------------------------------- desktop player / file handler
if (DAC_BUILD_PLAYER)
    add_library(dac_player_core STATIC src/player.cpp)
    target_link_libraries(dac_player_core PUBLIC dac_core)
    if (WIN32)
        add_executable(dac-player WIN32 src/player_win.cpp)
        target_link_libraries(dac-player PRIVATE dac_player_core comctl32 advapi32)
    elseif (APPLE)
        enable_language(OBJCXX)
        set(_plist ${CMAKE_CURRENT_BINARY_DIR}/Info.plist)
        configure_file(packaging/macos/Info.plist.in ${_plist} @ONLY)
        add_executable(dac-player MACOSX_BUNDLE src/player_mac.mm)
        set_target_properties(dac-player PROPERTIES
            OUTPUT_NAME "DAC Player"
            MACOSX_BUNDLE_INFO_PLIST ${_plist})
        target_compile_options(dac-player PRIVATE -fobjc-arc)
        target_link_libraries(dac-player PRIVATE dac_player_core "-framework Cocoa" "-framework CoreServices")
    else()
        add_executable(dac-player src/player_linux.cpp)
        target_link_libraries(dac-player PRIVATE dac_player_core)
    endif()
endif()
