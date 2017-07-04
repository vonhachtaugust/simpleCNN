cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

project(simpleCNN VERSION 0.0.1)
# Does not build into a library. Has to build activate subdirectory.

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
# Set the default build type to Release.
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified: defaulting to release.")
    set(CMAKE_BUILD_TYPE Release)
else(NOT CMAKE_BUILD_TYPE)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_BUILD_TYPE Release)
        message("==========================================================================================")
        message(STATUS "Build type: ${CMAKE_BUILD_TYPE}.")
        message("==========================================================================================\n")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif (NOT CMAKE_BUILD_TYPE)

#####
# OpenBLAS
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message(STATUS "Found BLAS!")
    message(STATUS "Linking: ${BLAS_LIBRARIES}\n")
    add_definitions(-DUSE_OPENBLAS)
    list(APPEND REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
else()
    message(FATAL_ERROR "Cannot find required package OpenBLAS. Please set BLAS_INCLUDE_DIR")
endif()

#####
# OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV!")
    message(STATUS "Linking: ${OpenCV_LIBRARIES}\n")
    add_definitions(-DUSE_OPENCV)
    list(APPEND REQUIRED_LIBRARIES ${OpenCV_LIBRARIES})
else()
    message(STATUS "Cannot find required package OpenCV. Please set OPENCV_INCLUDE_DIR")
endif()

#####
# CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "Found CUDA!")
    message(STATUS "Linking: ${CUDA_LIBRARIES} and ${CUDA_CUBLAS_LIBRARIES}\n")

    # Cuda files
    file(GLOB_RECURSE simpleCNN_cuda "${CMAKE_SOURCE_DIR}/simpleCNN/*.cu")
    message(STATUS "Cuda files to include: ${simpleCNN_cuda}")
    add_definitions(-DUSE_CUDA)
    list(APPEND REQUIRED_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
else()
    message(STATUS "Cannot find required package CUDA. Please set CUDA_INCLUDE_DIR")
endif()


set(HAVE_CUDNN_H "/third_party/cudnn/include/cudnn.h")
set(HAVE_CUDNN_SO "/third_party/cudnn/lib64/libcudnn.so")

if (NOT HAVE_CUDNN_H)
    message(STATUS "Could not find CUDNN")
else()
    message(STATUS "Found CUDNN!")
    if (NOT HAVE_CUDNN_SO)
        message(FATAL_ERROR "Could not find libcudnn.so")
    else()
        message(STATUS "Linking: ${HAVE_CUDNN_SO}\n")

        set(CUDNN_LIBRARY_DIRS "${PROJECT_SOURCE_DIR}/third_party/cudnn/lib64/third_party/cudnn/lib64")
        set(CUDNN_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/third_party/cudnn/include/third_party/cudnn/include")
        set(CUDNN_LIBRARIES "${PROJECT_SOURCE_DIR}/third_party/cudnn/lib64")

        add_definitions(-DUSE_CUDNN)
        include_directories(${CUDNN_INCLUDE_DIRS})
        link_directories(${CUDNN_LIBRARY_DIRS})
        list(APPEND REQUIRED_LIBRARIES ${CUDNN_LIBRARIES}/libcudnn.so)
    endif()
endif()

#####
# Define user options (To be implemented when source is build as library)
option(BUILDS_TEST      "Set to ON to build tests"      OFF)
option(BUILDS_EXAMPLE    "Set to ON to build examples"  ON)

####
# Source files
file(GLOB_RECURSE simpleCNN_headers "${CMAKE_SOURCE_DIR}/simpleCNN/*.h")

####
# Build and run tests && build and run an example
if(BUILDS_TEST)
    add_subdirectory(test)
endif()

if(BUILDS_EXAMPLE)
    add_subdirectory(examples)
endif()

