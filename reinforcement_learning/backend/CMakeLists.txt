cmake_minimum_required(VERSION 3.1)

project(Telegraph   VERSION 0.0.1
                    DESCRIPTION "Reinforcement Learning Library Backend"
                    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

add_compile_options(-Wall -Wextra -pedantic)
add_compile_options(-O3 -shared -std=c++14 -fPIC)

add_compile_options(-fopenmp)
add_link_options(-fopenmp)

add_definitions(-DXTENSOR_USE_XSIMD -DXTENSOR_USE_OPENMP)

include_directories(include)

include_directories(extern/xtl/include)
include_directories(extern/xsimd/include)
include_directories(extern/xtensor/include)
include_directories(extern/pybind11/include)
include_directories(extern/xtensor-python/include)

set(PYBIND11_CPP_STANDARD -std=c++14)
add_subdirectory(extern/pybind11)

add_subdirectory(src)

enable_testing()
include(GoogleTest)
add_subdirectory(tests)
