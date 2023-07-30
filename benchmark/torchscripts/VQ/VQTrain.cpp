
//
// Created by haolan on 25/6/23.
//
#include <vector>
#include <AMMBench.h>
#include <iostream>
#include <CPPAlgos/VectorQuantization.h>

using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace DIVERSE_METER;
using namespace AMMBench;


int main() {

    AMMBench::MatrixLoaderTable mLoaderTable;
    std::shared_ptr<AMMBench::AbstractMatrixLoader> basePtr = mLoaderTable.findMatrixLoader("MNIST"); // need to manually change fileName in AMMBench/src/MatrixLoader/MNISTMatrixLoader.cpp
    std::shared_ptr<AMMBench::MNISTMatrixLoader> matLoaderPtr = std::dynamic_pointer_cast<AMMBench::MNISTMatrixLoader>(basePtr);
    ConfigMapPtr cfg = newConfigMap();
    cfg->edit("aRow", (uint64_t) 0);
    cfg->edit("aCol", (uint64_t) 0);
    cfg->edit("bCol", (uint64_t) 0);
    matLoaderPtr->setConfig(cfg);
    auto A1 = matLoaderPtr->getA();
    A1 = A1.to(torch::kDouble);
    auto B1 = matLoaderPtr->getB();
    B1 = B1.to(torch::kDouble);
    B1 = B1.t();

    // Printing shape and data type for tensor A1
    std::cout << "Tensor A1:" << std::endl;
    std::cout << "Shape: " << A1.sizes() << std::endl;
    std::cout << "Data Type: " << A1.dtype() << std::endl;

    // Printing shape and data type for tensor B1
    std::cout << "Tensor B1:" << std::endl;
    std::cout << "Shape: " << B1.sizes() << std::endl;
    std::cout << "Data Type: " << B1.dtype() << std::endl;

    int m = 10;
    int l = 6000;
    PQMM pqmm(A1, B1, l, m);
    pqmm.setFilePath(
        "CCA/CCAcolumnCodeIndexX.txt",
        "CCA/CCArowCodeIndexY.txt",
        "CCA/CCAcolumnCodeBookX.txt",
        "CCA/CCArowCodeBookY.txt"
    );
    pqmm.runAMM(true);
    return 0;
}


