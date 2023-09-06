
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

int main(){

    // PCA
    ConfigMapPtr cfg = newConfigMap();
    cfg->edit("filePath", "../../datasets/SIFT/siftsmall_base.fvecs");
    AMMBench::MatrixLoaderTable mLoaderTable;
    auto matLoaderPtr = mLoaderTable.findMatrixLoader("SIFT");
    assert(matLoaderPtr);
    matLoaderPtr->setConfig(cfg);
    auto A = matLoaderPtr->getA();
    auto B = matLoaderPtr->getB();

    std::string directoryName = "MtxPt";

    auto pickledA = torch::pickle_save(A);
    std::ofstream foutA(directoryName+"/"+"SIFT_A.pt", std::ios::out | std::ios::binary);
    foutA.write(pickledA.data(), pickledA.size());
    foutA.close();

    auto pickledB = torch::pickle_save(B);
    std::ofstream foutB(directoryName+"/"+"SIFT_B.pt", std::ios::out | std::ios::binary);
    foutB.write(pickledB.data(), pickledB.size());
    foutB.close();

    // CCA
    cfg->edit("filePath", "../../datasets/MNIST/train-images.idx3-ubyte");
    matLoaderPtr = mLoaderTable.findMatrixLoader("MNIST");
    assert(matLoaderPtr);
    matLoaderPtr->setConfig(cfg);
    A = matLoaderPtr->getA();
    B = matLoaderPtr->getB();

    pickledA = torch::pickle_save(A);
    std::ofstream foutA2(directoryName+"/"+"MNIST_A.pt", std::ios::out | std::ios::binary);
    foutA2.write(pickledA.data(), pickledA.size());
    foutA2.close();

    pickledB = torch::pickle_save(B);
    std::ofstream foutB2(directoryName+"/"+"MNIST_B.pt", std::ios::out | std::ios::binary);
    foutB2.write(pickledB.data(), pickledB.size());
    foutB2.close();

    return 0;
}


