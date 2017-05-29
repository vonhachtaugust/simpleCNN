
####
# Example build in use
if (CUDA_FOUND)
    cuda_add_executable(example_mnist
            mnist/train.cpp
            ${simpleCNN_cuda}
            ${simpleCNN_headers})

    cuda_add_executable(example_cifar10
            cifar10/train.cpp
            ${simpleCNN_cuda}
            ${simpleCNN_headers})
else()
    add_executable(example_mnist
            mnist/train.cpp
            ${simpleCNN_headers})

    add_executable(example_cifar10
            cifar10/train.cpp
            ${simpleCNN_headers})
endif()
target_link_libraries(example_mnist ${REQUIRED_LIBRARIES})
target_link_libraries(example_cifar10 ${REQUIRED_LIBRARIES})
