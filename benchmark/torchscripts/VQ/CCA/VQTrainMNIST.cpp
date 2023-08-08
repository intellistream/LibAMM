
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
    A1 = A1.to(torch::kDouble); // 392*60000
    auto B1 = matLoaderPtr->getB();
    B1 = B1.to(torch::kDouble); // 392*60000

    // Printing shape and data type for tensor A1
    std::stringstream ssA1; // Create a stringstream for tensor A1
    for (int64_t dim : A1.sizes()) {
        ssA1 << dim << ", "; // Concatenate the dimensions to the stringstream
    }
    INTELLI_INFO("A1: [" + ssA1.str() + "]");

    // Printing shape and data type for tensor B1
    std::stringstream ssB1; // Create a stringstream for tensor B1
    for (int64_t dim : B1.sizes()) {
        ssB1 << dim << ", "; // Concatenate the dimensions to the stringstream
    }
    INTELLI_INFO("B1: [" + ssB1.str() + "]");
    
    int m = 10;
    int l = 6000;
    // PQMM pqmm(A1, B1.t(), l, m);
    // pqmm.setFilePath(
    //     "CCA/ABcolumnCodeIndexX.txt",
    //     "CCA/ABrowCodeIndexY.txt",
    //     "CCA/ABcolumnCodeBookX.txt",
    //     "CCA/ABrowCodeBookY.txt"
    // );
    // PQMM pqmm(A1, A1.t(), l, m);
    // pqmm.setFilePath(
    //     "CCA/AAcolumnCodeIndexX.txt",
    //     "CCA/AArowCodeIndexY.txt",
    //     "CCA/AAcolumnCodeBookX.txt",
    //     "CCA/AArowCodeBookY.txt"
    // );
    PQMM pqmm(B1, B1.t(), l, m);
    pqmm.setFilePath(
        "CCA/BBcolumnCodeIndexX.txt",
        "CCA/BBrowCodeIndexY.txt",
        "CCA/BBcolumnCodeBookX.txt",
        "CCA/BBrowCodeBookY.txt"
    );
    pqmm.runAMM(true);
    INTELLI_INFO("Done");
    return 0;
}


