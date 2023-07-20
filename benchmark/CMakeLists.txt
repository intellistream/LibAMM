# Copyright (C) 2021 by the INTELLI team (https://github.com/intellistream)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if (BUILD_BENCHMARKS)
    message("---Building benchmarks ${source_dir}")
    include_directories("${source_dir}/include/")
    include_directories("include/")
    add_executable(benchmark "src/Benchmark.cpp")
    target_link_libraries(benchmark IntelliStream)
    add_executable(benchmarkMM "src/BenchmarkMM.cpp")
    target_link_libraries(benchmarkMM IntelliStream)
    add_executable(benchmarkPCA "src/BenchmarkPCA.cpp")
    target_link_libraries(benchmarkPCA IntelliStream)
    add_executable(benchmarkCCA "src/BenchmarkCCA.cpp")
    target_link_libraries(benchmarkCCA IntelliStream)

    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/torchscripts
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/pythonTest.py
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config.csv
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

    set_target_properties(benchmarkMM PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/scripts/MM)
    set_target_properties(benchmarkPCA PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/scripts/PCA)
    set_target_properties(benchmarkCCA PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/scripts/CCA)


    file(COPY ${PROJECT_SOURCE_DIR}/src/CL
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
    message("---Done building benchmarks")
else (BUILD_BENCHMARKS)
    message("---Not building benchmarks!")
endif (BUILD_BENCHMARKS)