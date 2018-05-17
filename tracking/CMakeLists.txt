cmake_minimum_required(VERSION 3.10)
project(tracking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

include_directories(${PROJECT_SOURCE_DIR}/src/external_libs/pytorch/aten/src)
include_directories(${PROJECT_SOURCE_DIR}/src/external_libs/pytorch)
include_directories(${PROJECT_SOURCE_DIR}/src)

file(GLOB_RECURSE TRACKER_SRC "src/tracker/*.cpp" "src/tracker/*.hpp" "src/tracker/*.c" "src/tracker/*.h")
file(GLOB_RECURSE FEATURE_SRC
        "src/feature_extraction/*.cpp" "src/feature_extraction/*.hpp"
        "src/feature_extraction/*.c" "src/feature_extraction/*.h")

add_subdirectory(src/external_libs/pytorch)
add_library(feature ${FEATURE_SRC})
add_library(tracker ${TRACKER_SRC})

add_executable(demo demo.cpp)

find_package(OpenCV REQUIRED)

target_link_libraries(feature caffe2)
target_link_libraries(tracker feature)
target_link_libraries(demo tracker ${OpenCV_LIBS})