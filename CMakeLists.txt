cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

project(simpleCNN VERSION 0.0.1)
# Does not build into a library. Has to build a subdirectory.

# Enables project() command manages VERSION variables.
if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

#add_definitions(
#        -D_CMAKE_C_COMPILER=gcc-5.4.0
#        -D_CMAKE_CXX_COMPILER=g++-5.4.0
#        -D_ENABLE_LOG=1
#)

####
# Setup the compiler options
#
# Compiler currently available

# set c++ standard to c++11
set(CMAKE_CXX_STANDARD 11)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has C++11 support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has C++98 support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#####
# Set the default build type to Debug for now.
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified: defaulting to debug.")
    set(CMAKE_BUILD_TYPE Debug)
else(NOT CMAKE_BUILD_TYPE)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("==========================================================================================")
    message(STATUS "Build type: Debug.")
    message("==========================================================================================")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif ()

#####
# OpenBLAS
find_package( BLAS REQUIRED )
if(BLAS_FOUND)
    message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
    list(APPEND REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
else()
    message(FATAL_ERROR "Cannot find required package OpenBLAS. Please set BLAS_INCLUDE_DIR")
endif()

#####
# Define user options (To be implemented when source is build as library)
option(BUILDS_TEST      "Set to ON to build tests"      ON)

####
# Source files
file(GLOB_RECURSE simpleCNN_headers "${CMAKE_SOURCE_DIR}/simpleCNN/*.h")

####
# Build and run tests && build and run an example
if(BUILDS_TEST)
    add_subdirectory(test)
endif()

#if(NOT BUILDS_TEST)
add_subdirectory(examples)
#endif()

