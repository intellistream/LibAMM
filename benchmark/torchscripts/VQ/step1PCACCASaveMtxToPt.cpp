
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
    // Define the file paths
    std::vector<std::string> filePaths = {
        "../../datasets/SIFT/siftsmall_base.fvecs",
        "../../datasets/SIFT/sift_base.fvecs",
        "../../datasets/SIFT/gist_base.fvecs"
    };

    // Define the corresponding output file names
    std::vector<std::string> outputNames = {
        "SIFT10K",
        "SIFT1M",
        "GIST1M"
    };

    std::string directoryName = "MtxPt";

    for (int i = 0; i < int(filePaths.size()); ++i) {
        // Create a ConfigMap for each file (assuming you have a function to create it)
        ConfigMapPtr cfg = newConfigMap();
        cfg->edit("filePath", filePaths[i]);

        // Load the matrix loader and set the config
        AMMBench::MatrixLoaderTable mLoaderTable;
        auto matLoaderPtr = mLoaderTable.findMatrixLoader("SIFT");
        assert(matLoaderPtr);
        matLoaderPtr->setConfig(cfg);

        // Get the matrices A and B
        auto A = matLoaderPtr->getA();
        auto B = matLoaderPtr->getB();

        // Save the matrices with the corresponding names
        auto pickledA = torch::pickle_save(A);
        std::ofstream foutA(directoryName + "/" + outputNames[i] + "_A.pt", std::ios::out | std::ios::binary);
        foutA.write(pickledA.data(), pickledA.size());
        foutA.close();

        auto pickledB = torch::pickle_save(B);
        std::ofstream foutB(directoryName + "/" + outputNames[i] + "_B.pt", std::ios::out | std::ios::binary);
        foutB.write(pickledB.data(), pickledB.size());
        foutB.close();
    }


    // CCA
    // cfg->edit("filePath", "../../datasets/MNIST/train-images.idx3-ubyte");
    // matLoaderPtr = mLoaderTable.findMatrixLoader("MNIST");
    // assert(matLoaderPtr);
    // matLoaderPtr->setConfig(cfg);
    // A = matLoaderPtr->getA();
    // B = matLoaderPtr->getB();

    // pickledA = torch::pickle_save(A);
    // std::ofstream foutA2(directoryName+"/"+"MNIST_A.pt", std::ios::out | std::ios::binary);
    // foutA2.write(pickledA.data(), pickledA.size());
    // foutA2.close();

    // pickledB = torch::pickle_save(B);
    // std::ofstream foutB2(directoryName+"/"+"MNIST_B.pt", std::ios::out | std::ios::binary);
    // foutB2.write(pickledB.data(), pickledB.size());
    // foutB2.close();

    return 0;
}


