cmake_minimum_required(VERSION 3.1)

project(Telegraph   VERSION 0.0.1
                    DESCRIPTION "Reinforcement Learning Library Backend"
                    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

add_compile_options(-Wall -Wextra -Werror -Wno-deprecated-copy)

add_compile_options(-fopenmp)
add_link_options(-fopenmp)

add_definitions(-DXTENSOR_USE_XSIMD -DXTENSOR_USE_OPENMP)

include_directories("${CMAKE_SOURCE_DIR}/include")

include_directories("${CMAKE_SOURCE_DIR}/extern/xtl/include")
include_directories("${CMAKE_SOURCE_DIR}/extern/xsimd/include")
include_directories("${CMAKE_SOURCE_DIR}/extern/xtensor/include")
include_directories("${CMAKE_SOURCE_DIR}/extern/pybind11/include")

enable_testing()
include(GoogleTest)
add_subdirectory(tests)
