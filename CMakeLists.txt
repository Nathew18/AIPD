cmake_minimum_required(VERSION 3.24)   # 3.24+: CUDA_ARCHITECTURES native
project(Akceleracja LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Domyślnie Release — pomiary wydajności w Debug nie mają sensu.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(USE_CUDA        "Buduj wersję GPU (wymaga NVIDIA CUDA Toolkit)" OFF)
option(NBODY_USE_FLOAT "Licz w float zamiast double (zalecane dla GPU)" ON)
option(NBODY_NATIVE    "Optymalizuj pod procesor tego komputera (AVX2/AVX-512)" ON)

# ---------------------------------------------------------------- OpenMP
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE LIBOMP_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(LIBOMP_PREFIX)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "Nie znaleziono OpenMP - wersja 'omp' będzie działać na 1 wątku.")
endif()

# ---------------------------------------------------------------- źródła
set(SOURCES
    src/main.cpp
    src/simulation.cpp
    src/nbody_seq.cpp
    src/nbody_openmp.cpp
)

if(USE_CUDA)
    # "native" = architektura karty w tym komputerze (RTX 5070 -> sm_120,
    # wymaga CUDA Toolkit >= 12.8). Można nadpisać: -DCMAKE_CUDA_ARCHITECTURES=120
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    list(APPEND SOURCES src/nbody_cuda.cu)
else()
    list(APPEND SOURCES src/nbody_cuda_stub.cpp)
endif()

add_executable(nbody ${SOURCES})
target_include_directories(nbody PRIVATE include)

if(NBODY_USE_FLOAT)
    target_compile_definitions(nbody PRIVATE NBODY_USE_FLOAT)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX)
endif()

if(USE_CUDA)
    target_compile_definitions(nbody PRIVATE NBODY_HAS_CUDA)
endif()

# Optymalizacje. Bez -ffast-math, żeby wyniki CPU/GPU były porównywalne.
# -fno-math-errno: std::sqrt nie musi ustawiać errno => pętla sił może być
# wektoryzowana (bez tego wersja OpenMP była ~2x wolniejsza od sekwencyjnej).
if(MSVC)
    target_compile_options(nbody PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /W3 /utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
else()
    target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -fno-math-errno -Wall -Wextra>)
endif()

# Instrukcje wektorowe procesora (ten sam kod dla seq i omp => uczciwie).
if(NBODY_NATIVE)
    if(MSVC)
        # MSVC nie ma "native". AVX2 działa na każdym współczesnym CPU;
        # Ryzen 9800X3D (Zen 5) obsługuje też /arch:AVX512.
        target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    else()
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
        check_cxx_compiler_flag(-mcpu=native  HAS_MCPU_NATIVE)   # Apple Silicon
        if(HAS_MARCH_NATIVE)
            target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
        elseif(HAS_MCPU_NATIVE)
            target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mcpu=native>)
        endif()
    endif()
endif()

# ---------------------------------------------------------------- testy
enable_testing()
add_test(NAME correctness COMMAND nbody --mode test)
