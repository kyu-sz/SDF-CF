cmake_minimum_required(VERSION 3.10)
project(tracking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

include_directories(${PROJECT_SOURCE_DIR}/src)

file(GLOB_RECURSE TRACKER_SRC "src/*.cpp" "src/*.hpp" "src/*.c" "src/*.h")
file(GLOB_RECURSE TRACKER_SRC "src/*.cpp" "src/*.hpp" "src/*.c" "src/*.h")

add_subdirectory(src/external_libs/pytorch)
add_library(tracker ${TRACKER_SRC} caffe2)

add_executable(demo demo.cpp)