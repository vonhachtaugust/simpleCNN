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
add_executable(simpleCNN_test
        test.cpp
        test_maxpooling_layer.h
        test_convolutional_layer.h
        ${simpleCNN_headers})
target_link_libraries(simpleCNN_test ${REQUIRED_LIBRARIES} gtest gmock)

#####
# Add executable to all tests to be performed
# add_test(NAME all_tests COMMAND simpleCNN_test)
add_custom_target(run_tests COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS simpleCNN_test)

# Clang format check
include(../cmake/clang-cxx-dev-tools.cmake)