cmake_minimum_required(VERSION 3.18)

# set langauges
project(cudeft LANGUAGES CXX CUDA)

# set versions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# connect project
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# set includes for externals
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/externals/gpuintegration/cuda ${CMAKE_CURRENT_SOURCE_DIR}/externals/gpuintegration)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(OpenMP REQUIRED)

SET( CMAKE_CXX_FLAGS  "-O3 -fpic")

# create exe
add_executable(cudeft main.cu matter.cu interpolator.cu integrate.cu spt.cu)

# create python bindings
execute_process(
	COMMAND python -m pybind11 --cmakedir
	OUTPUT_VARIABLE pybind11_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(pycudeft python.cu matter.cu interpolator.cu integrate.cu spt.cu)

target_compile_options(pycudeft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
target_link_libraries(pycudeft PRIVATE OpenMP::OpenMP_CXX)
