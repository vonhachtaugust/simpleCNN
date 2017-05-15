
####
# Example build in use

add_executable(examples main.cpp ${simpleCNN_headers})
target_link_libraries(examples ${REQUIRED_LIBRARIES})

message(STATUS "${REQUIRED_LIBRARIES}")

add_executable(example_mnist mnist/train.cpp ${simpleCNN_headers})
target_link_libraries(example_mnist ${REQUIRED_LIBRARIES})
