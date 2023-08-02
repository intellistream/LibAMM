
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

    torch::Tensor A = torch::rand({400, 6000});
    A = A.to(torch::kDouble);
    torch::Tensor B = torch::rand({6000, 400});
    B = B.to(torch::kDouble);
    auto realC = torch::matmul(A, B);
    std::cout << "realC Maximum Value: " << realC.max().item<float>() << std::endl;
    std::cout << "realC Mean Value: " << realC.mean().item<float>() << std::endl;
    std::cout << "realC Minimum Value: " << realC.min().item<float>() << std::endl;
    
    int m = 10;
    int l = 600;
    PQMM pqmm(A, B, l, m);
    pqmm.setFilePath(
        "columnCodeIndexX_100.txt",
        "rowCodeIndexY_100.txt",
        "columnCodeBookX_100.txt",
        "rowCodeBookY_100.txt"
    );
    torch::Tensor ammC1 = pqmm.runAMM(true);
    double froError1 = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC1);
    std::cout << "pqmm.runAMM(true): " << froError1 << std::endl;
    std::cout << "ammC1 Maximum Value: " << ammC1.max().item<float>() << std::endl;
    std::cout << "ammC1 Mean Value: " << ammC1.mean().item<float>() << std::endl;
    std::cout << "ammC1 Minimum Value: " << ammC1.min().item<float>() << std::endl;
    
    pqmm.setFilePath(
        "columnCodeIndexX_100.txt",
        "rowCodeIndexY_100.txt",
        "columnCodeBookX_100.txt",
        "rowCodeBookY_100.txt"
    );
    torch::Tensor ammC2 = pqmm.runAMM(false);
    double froError2 = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC2);
    std::cout << "pqmm.runAMM(false)" << froError2 << std::endl;
    std::cout << "ammC2 Maximum Value: " << ammC2.max().item<float>() << std::endl;
    std::cout << "ammC2 Mean Value: " << ammC2.mean().item<float>() << std::endl;
    std::cout << "ammC2 Minimum Value: " << ammC2.min().item<float>() << std::endl;
    return 0;
}


