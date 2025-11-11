/**
 * Simple test to verify Eigen-based LibAMM core functionality
 * Compile: g++ -std=c++17 -I./include -I./build/_deps/eigen-src test_eigen_migration.cpp -L./build -lLibAMM -o test_eigen
 * Run: LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH ./test_eigen
 */

#include "Utils/EigenTensor.h"
#include "MatrixLoader/RandomMatrixLoader.h"
#include "CPPAlgos/CRSCPPAlgo.h"
#include <iostream>
#include <memory>

int main() {
    std::cout << "=== LibAMM Eigen Migration Test ===\n\n";
    
    // Test 1: Basic Tensor creation
    std::cout << "Test 1: Basic Tensor operations\n";
    LibAMM::Tensor a = LibAMM::rand({3, 4});
    LibAMM::Tensor b = LibAMM::ones({4, 2});
    std::cout << "  Created tensors a(3x4) and b(4x2)\n";
    std::cout << "  a = " << a << "\n";
    std::cout << "  b = " << b << "\n";
    
    // Test 2: Matrix multiplication
    std::cout << "\nTest 2: Matrix multiplication\n";
    LibAMM::Tensor c = LibAMM::matmul(a, b);
    std::cout << "  c = a @ b = " << c << "\n";
    std::cout << "  c.size(0) = " << c.size(0) << ", c.size(1) = " << c.size(1) << "\n";
    
    // Test 3: Transpose
    std::cout << "\nTest 3: Transpose\n";
    LibAMM::Tensor at = a.t();
    std::cout << "  a.t() = " << at << "\n";
    
    // Test 4: Element-wise operations
    std::cout << "\nTest 4: Element-wise operations\n";
    LibAMM::Tensor d = LibAMM::ones({3, 4});
    LibAMM::Tensor e = a + d;
    std::cout << "  a + ones(3x4) = " << e << "\n";
    
    // Test 5: Indexing
    std::cout << "\nTest 5: Indexing\n";
    float val = a(0, 0);
    std::cout << "  a(0, 0) = " << val << "\n";
    
    // Test 6: torch namespace compatibility
    std::cout << "\nTest 6: torch namespace compatibility\n";
    torch::Tensor t1 = torch::zeros({2, 2});
    torch::Tensor t2 = torch::rand({2, 2});
    torch::Tensor t3 = torch::matmul(t1 + t2, t2.t());
    std::cout << "  torch::matmul works: " << t3 << "\n";
    
    // Test 7: Threading stub
    std::cout << "\nTest 7: torch::set_num_threads (stub)\n";
    torch::set_num_threads(4);
    std::cout << "  torch::set_num_threads(4) called successfully (no-op)\n";
    
    // Test 8: RandomMatrixLoader
    std::cout << "\nTest 8: RandomMatrixLoader\n";
    try {
        auto loader = std::make_shared<LibAMM::RandomMatrixLoader>();
        auto config = std::make_shared<INTELLI::ConfigMap>();
        config->edit("m", (int64_t)5);
        config->edit("n", (int64_t)3);
        loader->setConfig(config);
        LibAMM::Tensor A = loader->getA();
        LibAMM::Tensor B = loader->getB();
        std::cout << "  Loaded A: " << A << "\n";
        std::cout << "  Loaded B: " << B << "\n";
    } catch (const std::exception& ex) {
        std::cout << "  RandomMatrixLoader test failed: " << ex.what() << "\n";
    }
    
    // Test 9: CRS Algorithm
    std::cout << "\nTest 9: CRS Algorithm\n";
    try {
        LibAMM::CRSCPPAlgo crs;
        auto config = std::make_shared<INTELLI::ConfigMap>();
        config->edit("k", (int64_t)2);
        crs.setConfig(config);
        
        LibAMM::Tensor A = LibAMM::rand({10, 8});
        LibAMM::Tensor B = LibAMM::rand({8, 6});
        
        LibAMM::Tensor result = crs.amm(A, B, 10);
        std::cout << "  CRS sketch result: " << result << "\n";
    } catch (const std::exception& ex) {
        std::cout << "  CRS algorithm test failed: " << ex.what() << "\n";
    }
    
    std::cout << "\n=== All tests completed ===\n";
    return 0;
}
