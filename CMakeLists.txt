cmake_minimum_required (VERSION 3.0)
project(lucas_kanada)

set(CMAKE_CXX_STANDARD 11)
# enable gdb
set(CMAKE_BUILD_TYPE Release)
add_definitions(-g -Wall -O2)

find_package( OpenCV REQUIRED )

add_library(lucas_kanada
        src/lucas_kanada.cpp)
target_link_libraries(lucas_kanada
        ${OpenCV_LIBS})

add_executable(lucas_kanada_test
        src/lucas_kanada_test.cpp)

target_link_libraries(lucas_kanada_test
        lucas_kanada
        ${OpenCV_LIBS})


