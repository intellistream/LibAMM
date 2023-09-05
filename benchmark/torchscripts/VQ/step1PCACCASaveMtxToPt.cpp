
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

    ConfigMapPtr cfg = newConfigMap();
    cfg->edit("filePath", "../../datasets/SIFT/siftsmall_base.fvecs");
    AMMBench::MatrixLoaderTable mLoaderTable;
    std::string matrixLoaderTag = cfg->tryString("matrixLoaderTag", "SIFT", true);
    INTELLI_INFO("matrixLoaderTag: " + matrixLoaderTag);
    auto matLoaderPtr = mLoaderTable.findMatrixLoader(matrixLoaderTag);
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

    return 0;
}


