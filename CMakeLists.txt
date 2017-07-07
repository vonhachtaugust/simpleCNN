cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

project(simpleCNN VERSION 0.1)
# Does not build into a library. Has to build activate subdirectory.

# Enables project() command manages VERSION variables.
if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

if (CMAKE_COMPILER_IS_GNUCC)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
            OUTPUT_VARIABLE GCC_VERSION)
    string(REGEX MATCHALL "[0-9]+" GCC_VERSION_COMPONENTS ${GCC_VERSION})
    list(GET GCC_VERSION_COMPONENTS 0 GCC_MAJOR)
    list(GET GCC_VERSION_COMPONENTS 1 GCC_MINOR)
endif()

####
# Setup the compiler options
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
    set(ERR_MESSAGE "The compiler ${CMAKE_CXX_COMPILER} has 'NO' C++11 support.")
    set(EX_MESSAGE "Try to use a newer C++ compiler. If there is a C++11 compatible compiler in use, try to specify it.\n Example:\n export CC=/usr/local/bin/gcc \n export CXX=/usr/local/bin/g++ \n")
    message(FATAL_ERROR "${ERR_MESSAGE} ${EX_MESSAGE}")
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
set(CPU False)
find_package(BLAS QUIET)
if(BLAS_FOUND)
    message(STATUS "Found BLAS!")
    message(STATUS "Linking: ${BLAS_LIBRARIES}\n")
    add_definitions(-DUSE_OPENBLAS)
    set(CPU True)
    list(APPEND REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
else()
    message(STATUS "Cannot find OpenBLAS. Please make sure they are available or try setting BLAS_INCLUDE_DIR\n")
    message(STATUS "CPU computations are unavailable.")
endif()

#####
# OpenCV
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV!")
    message(STATUS "Linking: ${OpenCV_LIBRARIES}\n")
    add_definitions(-DUSE_OPENCV)
    list(APPEND REQUIRED_LIBRARIES ${OpenCV_LIBRARIES})
else()
    message(STATUS "Cannot find OpenCV. Please make sure it is available or try setting OPENCV_INCLUDE_DIR\n")
endif()

#####
# CUDA
set(GPU False)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "Found CUDA!")
    message(STATUS "Linking: ${CUDA_LIBRARIES} and ${CUDA_CUBLAS_LIBRARIES}")

    # Cuda files
    file(GLOB_RECURSE simpleCNN_cuda "${CMAKE_SOURCE_DIR}/simpleCNN/*.cu")
    message(STATUS "Cuda files to include: ${simpleCNN_cuda}\n")
    add_definitions(-DUSE_CUDA)
    list(APPEND REQUIRED_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
else()
    message(STATUS "Cannot find CUDA. Please make sure it is are available or try setting CUDA_INCLUDE_DIR\n")
    message(STATUS "GPU computations are unavailable")
endif()

if (NOT GPU AND NOT CPU)
    message(FATAL_ERROR "Neither CPU nor GPU computations are available.")
endif()

if(EXISTS "${ROOT}/third_party/cudnn/include/cudnn.h")
    message(STATUS "Found CUDNN!")
    if(EXISTS "${ROOT}/third_party/cudnn/lib64/libcudnn.so")
        message(STATUS "Linking: ${HAVE_CUDNN_SO}\n")

        set(CUDNN_LIBRARY_DIRS "${PROJECT_SOURCE_DIR}/third_party/cudnn/lib64/third_party/cudnn/lib64")
        set(CUDNN_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/third_party/cudnn/include/third_party/cudnn/include")
        set(CUDNN_LIBRARIES "${PROJECT_SOURCE_DIR}/third_party/cudnn/lib64")

        add_definitions(-DUSE_CUDNN)
        set(GPU True)
        include_directories(${CUDNN_INCLUDE_DIRS})
        link_directories(${CUDNN_LIBRARY_DIRS})
        list(APPEND REQUIRED_LIBRARIES ${CUDNN_LIBRARIES}/libcudnn.so)
    else()
        message(STATUS "Cannot find libcudnn.so. Please make sure it is available at the right location: ${ROOT}/third_party/cudnn/lib64/libcudnn.so")
    endif()
else()
    message(STATUS "Cannot find CUDNN. Please make sure it is available at the right location: ${ROOT}/third_party/cudnn/include/cudnn.h.\n")
endif()

#####
# Define user options (To be implemented when source is build as library)
option(BUILDS_TEST      "Set to ON to build tests"     OFF)
option(BUILDS_EXAMPLE   "Set to ON to build examples"  OFF)

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

