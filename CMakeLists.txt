cmake_minimum_required(VERSION 3.6)
project(simpleCNN)

add_definitions(
        -D_CMAKE_C_COMPILER=gcc-5.4.0
        -D_CMAKE_CXX_COMPILER=g++-5.4.0
        -D_ENABLE_LOG=1
)

find_package( BLAS REQUIRED )
message( STATUS BLAS found: ${BLAS_LIBRARIES} )

set(CMAKE_CXX_STANDARD 11)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has C++11 support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has C++98 support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


set(SOURCE_FILES
        src/main.cpp
        include/core/framework/tensor.h
        include/core/framework/tensor_storage.h
        include/core/framework/tensor_utils.h
        include/util/aligned_allocator.h
        include/util/simple_error.h
        include/util/colored_print.h
        include/layers/convolutional_layer.h
        include/util/im2col2im.h)

add_executable(simpleCNN ${SOURCE_FILES})
target_link_libraries(simpleCNN -lopenblas)

