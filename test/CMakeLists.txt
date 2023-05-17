
#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Werror=return-type -Werror=unused-variable -Werror=unused-parameter")
enable_testing()
add_executable(cpp_test SystemTest/SimpleTest.cpp)
add_executable(sketch_test SystemTest/SketchTest.cpp)
message(STATUS " Libraries: ${LIBRARIES}")
target_link_libraries(cpp_test
        IntelliStream
        )
target_link_libraries(sketch_test
        IntelliStream
        )
# avoid undefined reference to "main" in test.cpp

add_test(
        NAME catch_test
        COMMAND $<TARGET_FILE:cpp_test> --success
)
add_test(
        NAME sketch_test
        COMMAND $<TARGET_FILE:sketch_test> --success
)


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/torchscripts
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})