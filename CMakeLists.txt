cmake_minimum_required(VERSION 3.14)
project(minitorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_library(minitorch INTERFACE)
target_include_directories(minitorch INTERFACE include)
target_link_libraries(minitorch INTERFACE Threads::Threads)

add_executable(spiral src/main.cpp)
target_link_libraries(spiral PRIVATE minitorch)
target_compile_options(spiral PRIVATE -O3 -Wall -Wextra)
