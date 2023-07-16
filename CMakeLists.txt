
cmake_minimum_required(VERSION 3.14)
project(IntelliStream CXX)


# Custom CMake find instructions and macros
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
include(cmake/macros.cmake)
include(cmake/default.cmake)


# C++ Standard
#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#gcc 10 g++10


find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(LIBRARIES ${LIBRARIES} ${TORCH_LIBRARIES})
# Set Optimization Flags
set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Werror=return-type -Werror=unused-variable -Werror=unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DNO_RACE_CHECK -DIntelliStream_DEBUG_MODE=1")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-ignored-qualifiers -Wno-sign-compare -O3")
#set(CMAKE_CUDA_STANDARD 11)
#set(CMAKE_CUDA_FLAGS "-std=c++11")


#set(CMAKE_CUDA_ARCHITECTURES 55)
# Set LOGGING_LEVEL Flag
if (IntelliStream_LOGGING_LEVEL)
    get_log_level_value(IntelliStream_LOGGING_VALUE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIntelliStream_LOGGING_LEVEL=${IntelliStream_LOGGING_VALUE}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DIntelliStream_LOGGING_LEVEL=${IntelliStream_LOGGING_VALUE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DIntelliStream_LOGGING_LEVEL=${IntelliStream_LOGGING_VALUE}")
else (IntelliStream_LOGGING_LEVEL)
    message("---Everything will be logged")
endif (IntelliStream_LOGGING_LEVEL)

message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
#pytorch
#set(Torch_DIR "/home/tony/.local/lib/python3.10/site-packages/torch/share/cmake" )
# Log4cc
#find_package(Log4cxx REQUIRED)
#include_directories(${Log4cxx_INCLUDE_DIR})
#set(LIBRARIES ${LIBRARIES} ${Log4cxx_LIBRARY})


option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")


# Print all used include directories
message(STATUS "INCLUDE_DIRS:")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach (dir ${dirs})
    message(STATUS "  - ${dir}")
endforeach ()

message(STATUS " Libraries: ${LIBRARIES}")
#add_subdirectory(pytorchNN)
# Add Source Code
add_subdirectory(src)

# Add Library
get_sources(IntelliStream_SOURCE_FILES)
get_headers(IntelliStream_HEADER_FILES)
add_library(IntelliStream SHARED ${IntelliStream_SOURCE_FILES} ${IntelliStream_HEADER_FILES} ${CMAKE_CURRENT_BINARY_DIR} )
set_property(TARGET IntelliStream PROPERTY CXX_STANDARD 20)

target_include_directories(IntelliStream PUBLIC "include")
target_include_directories(IntelliStream PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(IntelliStream PUBLIC ${LIBRARIES})
#add_subdirectory(test)
#Add benchmarks with command
add_subdirectory(benchmark)
set_property(TARGET benchmark PROPERTY CXX_STANDARD 20)
#Add tests (based on GTest suits)
#include(GoogleTest)

if (ENABLE_UNIT_TESTS)
    add_subdirectory(test)
endif ()


install(DIRECTORY "include" DESTINATION "/IntelliStream" COMPONENT IntelliStream)
# copy scripts
file(GLOB allCopyFiles "${PROJECT_SOURCE_DIR}/scripts/*")
file(COPY ${allCopyFiles} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/benchmark/scripts)
file(COPY ${allCopyFiles} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/benchmark/scripts)
#file(COPY ${allCopyFiles} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/scripts)