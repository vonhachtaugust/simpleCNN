
####
# Example build in use
if (CUDA_FOUND)
    if(${GCC_MAJOR} LESS 5)
        cuda_add_executable(example_mnist
                mnist/train.cpp
                ${simpleCNN_cuda}
                ${simpleCNN_headers})

        cuda_add_executable(example_cifar10
                cifar10/train.cpp
                ${simpleCNN_cuda}
                ${simpleCNN_headers})
    else()
        message(STATUS "NOTE: Currently there is no workaround for letting nvcc use GCC version less than 5 and at the same time build this project. The .cu file cannot be built for now. \n")
        add_executable(example_mnist
                mnist/train.cpp
                ${simpleCNN_headers})

        add_executable(example_cifar10
                cifar10/train.cpp
                ${simpleCNN_headers})
    endif()
endif()
target_link_libraries(example_mnist ${REQUIRED_LIBRARIES})
target_link_libraries(example_cifar10 ${REQUIRED_LIBRARIES})
