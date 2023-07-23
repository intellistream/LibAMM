/*! \file Benchmark.h*/

/**
 * @brief This is the main entry point of the entire program.
 * We use this as the entry point for benchmarking.
 */
#include <AMMBench.h>
#include <Utils/UtilityFunctions.h>
#include <Streaming/Streamer.h>

using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace DIVERSE_METER;
using namespace AMMBench;

void benchmarkPCA(std::string configName){

    INTELLI_INFO("Running Benchmark PCA");

    // 1. Set up environments
    ConfigMapPtr cfg = newConfigMap();
    cfg->fromFile(configName);

    INTELLI_INFO("cppAlgoTag: " + cfg->tryString("cppAlgoTag", "mm", true));

    AMMBench::MatrixLoaderTable mLoaderTable;
    std::string matrixLoaderTag = cfg->tryString("matrixLoaderTag", "SIFT", true);
    INTELLI_INFO("matrixLoaderTag: " + matrixLoaderTag);
    auto matLoaderPtr = mLoaderTable.findMatrixLoader(matrixLoaderTag);
    assert(matLoaderPtr);
    matLoaderPtr->setConfig(cfg);
    auto A = matLoaderPtr->getA();
    auto B = matLoaderPtr->getB();

    uint64_t sketchDimension;
    sketchDimension = cfg->tryU64("sketchDimension", 1, true);
    INTELLI_INFO("sketchDimension: " + to_string(sketchDimension));

    // 2. Get metrics of AMM in streaming and parallelism
    INTELLI_INFO("Run AMM");
    Streamer streamer;
    torch::Tensor C = streamer.run(cfg, A, B, sketchDimension, "AMM");
    ConfigMapPtr allMetrics = streamer.getMetrics();

    // 3. Get metrics of the rest of PCA except AMM, we call it elseMetrics
    INTELLI_INFO("Start SVD task..");
    ThreadPerf pef(-1);
    pef.setPerfList();
    pef.start();
    torch::Tensor U, S, Vh;
    std::tie(U, S, Vh) = torch::linalg::svd(torch::div(C, A.size(1)), false, c10::nullopt); // // covirance matrix estimator
    int k = 5; // use eigenvector 0-49
    torch::Tensor Ak = torch::matmul(Vh.narrow(0, 0, k), A); // Ak feature dimension reduced to k
    pef.end();
    ConfigMapPtr elseMetrics = pef.resultToConfigMap();
    elseMetrics->addPrefixToKeys("else");
    elseMetrics->cloneInto(*allMetrics);

    // 4. Calculate end-to-end error
    Ak =  torch::matmul(U.narrow(1, 0, k), Ak); // back to original feature dimension for error evaluation

    torch::Tensor realC = torch::matmul(A, B);
    torch::Tensor realU, realS, realVh;
    std::tie(realU, realS, realVh) = torch::linalg::svd(torch::div(realC, A.size(1)), false, c10::nullopt);
    INTELLI_INFO(to_string(realS[k].item<double>()));

    torch::Tensor UError, SError, VhError;
    std::tie(UError, SError, VhError) = torch::linalg::svd(torch::div(torch::matmul(A-Ak, (A-Ak).t()), A.size(1)), false, c10::nullopt);
    INTELLI_INFO(to_string(SError[0].item<double>()));

    // double relativeSpectralNormError = torch::div(SError[0], realS[k]).item<double>()-1; // SError[0] is the spectral norm of (A-Ak), real[k] is the spectral norm of (A-realAk)
    double relativeSpectralNormError = torch::div(SError[0], realS[0]).item<double>();
    allMetrics->edit("PCAError", relativeSpectralNormError);
    
    // 5 Save results
    std::string ruName = "PCA";
    allMetrics->toFile(ruName + ".csv");
    INTELLI_INFO("Done. here is overall result");
    INTELLI_INFO(allMetrics->toString());
}

int main(int argc, char **argv) {
    string configName, outPrefix = "";
    if (argc >= 2) {
        configName += argv[1];
    } else {
        configName = "config.csv";
    }
    benchmarkPCA(configName);
    return 0;
}

