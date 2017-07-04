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
            ${CLANG_FORMAT}
            -i
            ${ALL_CXX_SOURCE_FILES}
    )
else()
    message(STATUS "${clang-format} was not found")
endif()