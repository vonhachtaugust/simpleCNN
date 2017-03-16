
####
# Example build in use

add_executable(examples main.cpp ${simpleCNN_headers})
target_link_libraries(examples ${REQUIRED_LIBRARIES})