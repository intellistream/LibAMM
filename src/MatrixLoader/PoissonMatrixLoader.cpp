//
// Created by haolan on 6/6/23.
//
#include <MatrixLoader/PoissonMatrixLoader.h>
#include <Utils/IntelliLog.h>
void AMMBench::PoissonMatrixLoader::paraseConfig(INTELLI::ConfigMapPtr cfg) {
    aRow = cfg->tryU64("aRow", 100, true);
    aCol = cfg->tryU64("aCol", 1000, true);
    bCol = cfg->tryU64("bCol", 500, true);
    seed = cfg->tryU64("seed", 114514, true);
    INTELLI_INFO(
            "Generating [" + to_string(aRow) + "x" + to_string(aCol) + "]*[" + to_string(aCol) + "x" + to_string(bCol)
            + "]");
}
void AMMBench::PoissonMatrixLoader::generateAB() {
    torch::manual_seed(seed);
    A = torch::poisson(torch::ones({(long) aRow, (long) aCol}));
    B = torch::poisson(torch::ones({(long) aCol, (long) bCol}));
}

//do nothing in abstract class
bool AMMBench::PoissonMatrixLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
    paraseConfig(cfg);
    generateAB();
    return true;
}
torch::Tensor AMMBench::PoissonMatrixLoader::getA() {
    return A;
}
torch::Tensor AMMBench::PoissonMatrixLoader::getB() {
    return B;
}