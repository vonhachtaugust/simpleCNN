file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/simpleCNN/*.h
        ${CMAKE_SOURCE_DIR}/test/*.h
        ${CMAKE_SOURCE_DIR}/test/*.cpp
        ${CMAKE_SOURCE_DIR}/examples/*.h
        ${CMAKE_SOURCE_DIR}/examples/*.cpp
        )

set(clang-format "clang-format")

# Adding clang-format target if executable is found
find_program(CLANG_FORMAT ${clang-format})
if(CLANG_FORMAT)
    message(STATUS "${clang-format} was found!")
    add_custom_target(
            clang-format
            COMMAND /apps/Hebbe/software/MPI/GCC/5.4.0-2.26/OpenMPI/1.10.3/Clang/3.8.1/bin/clang-format
            -i
            ${ALL_CXX_SOURCE_FILES}
    )
else()
    message(STATUS "${clang-format} was not found")
endif()