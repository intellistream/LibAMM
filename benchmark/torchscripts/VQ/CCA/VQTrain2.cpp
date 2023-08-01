
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

    int rows = 100;
    int columns = 100;
    torch::Tensor A = torch::ones({rows, columns});
    A = A.to(torch::kDouble);
    torch::Tensor B = torch::ones({rows, columns});
    B = B.to(torch::kDouble);
    auto realC = torch::matmul(A, B);
    
    int m = 1;
    int l = 100;
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
    
    pqmm.setFilePath(
        "columnCodeIndexX_100.txt",
        "rowCodeIndexY_100.txt",
        "columnCodeBookX_100.txt",
        "rowCodeBookY_100.txt"
    );
    torch::Tensor ammC2 = pqmm.runAMM(false);
    double froError2 = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC2);
    std::cout << "pqmm.runAMM(false)" << froError2 << std::endl;
    return 0;
}


