add_subdirectory(Utils)
add_subdirectory(MatrixLoader)
add_subdirectory(Parallelization)
add_subdirectory(CPPAlgos)
add_subdirectory(Streaming)
add_sources(
        myVecAdd.cpp
)
include_directories("../include/")
add_executable(RIPCPPAlgo "CPPAlgos/RIPCPPAlgo.cpp")
target_link_libraries(RIPCPPAlgo IntelliStream)