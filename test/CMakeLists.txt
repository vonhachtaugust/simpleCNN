include(CTest)
enable_testing()

include(../cmake/DownloadProject/DownloadProject.cmake)

download_project(
        PROJ googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG master
        UPDATE_DISCONNECTED 1
)

if("${googletest_SOURCE_DIR}" STREQUAL "")
    message(FATAL_ERROR "CMake unable to download googletest source dir: ${googletest_SOURCE_DIR}")
endif()

if("${googletest_BINARY_DIR}" STREQUAL "")
    message(FATA_ERROR "CMake unable to download googletest binary dir: ${googletest_BINARY_DIR}")
endif()
message(STATUS "Googletest succesfully downloaded! : ${googletest_SOURCE_DIR} ${googletest_BINARY_DIR}")

add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})

enable_testing()

#####
# Create test executable and link gtest
if (CUDA_FOUND)
    if(${GCC_MAJOR} LESS 5)
        cuda_add_executable(simpleCNN_test
                test.cpp
                test_math_functions.h
                test_maxpooling_layer.h
                test_convolutional_layer.h
                test_connected_layer.h
                test_tensor.h
                test_tensor_multiplication.h
                test_loss_layer.h
                test_activation_layer.h
                test_network.h
                test_weight_init.h
                test_optimizer.h
                test_dropout_layer.h
                test_cuda.h
                test_serialization.h
                ${simpleCNN_cuda}
                ${simpleCNN_headers})
        target_link_libraries(simpleCNN_test ${REQUIRED_LIBRARIES} gtest gmock cuda cudart)
        else()
        message(STATUS "NOTE: Currently there is no workaround for letting nvcc use GCC version less than 5 and at the same time build this project. The .cu file cannot be built for now. \n")
        add_executable(simpleCNN_test
                test.cpp
                test_math_functions.h
                test_maxpooling_layer.h
                test_convolutional_layer.h
                test_connected_layer.h
                test_tensor.h
                test_tensor_multiplication.h
                test_loss_layer.h
                test_activation_layer.h
                test_network.h
                test_weight_init.h
                test_optimizer.h
                test_dropout_layer.h
                test_cuda.h
                test_serialization.h
                ${simpleCNN_headers})
        target_link_libraries(simpleCNN_test ${REQUIRED_LIBRARIES} gtest gmock)
    endif()
endif ()


#####
# Add executable to all tests to be performed
# add_test(NAME all_tests COMMAND simpleCNN_test)
add_custom_target(run_tests COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS simpleCNN_test )

# Clang format check
include(../cmake/clang-cxx-dev-tools.cmake)