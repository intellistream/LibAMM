/*! \file Benchmark.h*/

/**
 * @brief This is the main entry point of the entire program.
 * We use this as the entry point for benchmarking.
 */
#include <AMMBench.h>
#include <Utils/UtilityFunctions.h>

using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace DIVERSE_METER;


void benchmarkMM(std::string configName){

    // Step1. Set up environments
    ConfigMapPtr cfg = newConfigMap();
    cfg->fromFile(configName);

    // 1.1 AMM algorithm
    AMMBench::CPPAlgoTable cppAlgoTable;
    std::string cppAlgoTag = cfg->tryString("cppAlgoTag", "mm", true);
    AMMBench::AbstractCPPAlgoPtr cppAlgoPtr = cppAlgoTable.findCppAlgo(cppAlgoTag);
    cppAlgoPtr->setConfig(cfg);
    INTELLI_INFO("1.1 algo: " + cppAlgoTag);

    // 1.2 matrixLoader
    AMMBench::MatrixLoaderTable mLoaderTable;
    std::string matrixLoaderTag = cfg->tryString("matrixLoaderTag", "random", true);
    INTELLI_INFO("1.2 matrixLoaderTag: " + matrixLoaderTag);
    auto matLoaderPtr = mLoaderTable.findMatrixLoader(matrixLoaderTag);
    assert(matLoaderPtr);
    matLoaderPtr->setConfig(cfg);
    auto A = matLoaderPtr->getA();
    auto B = matLoaderPtr->getB();

    // 1.3 sketch dimension
    uint64_t sketchDimension;
    sketchDimension = cfg->tryU64("sketchDimension", 50, true);
    INTELLI_INFO("1.3 sketch dimension: " + to_string(sketchDimension));

    // 1.4 coreBind
    uint64_t coreBind = cfg->tryU64("coreBind", 0, true);
    UtilityFunctions::bind2Core((int) coreBind);
    INTELLI_INFO("1.4 corebind:" + to_string(coreBind));

    // Step2. Test elapsedTime and error on AMM single thread
    struct timeval tstart, tend;
    ThreadPerf pef(-1);
    pef.setPerfList();

    // 2.1 AMM
    pef.start();
    gettimeofday(&tstart, NULL);
    torch::Tensor C = cppAlgoPtr->amm(A, B, sketchDimension);
    gettimeofday(&tend, NULL);
    pef.end();
    auto resultCsv = pef.resultToConfigMap();
    resultCsv->edit("AMMElapsedTime", (uint64_t) INTELLI::UtilityFunctions::timeLast(tstart, tend));
    INTELLI_INFO("2.1 AMM finished in :" + to_string(INTELLI::UtilityFunctions::timeLast(tstart, tend)));

    // 2.2 MM
    torch::Tensor realC = torch::matmul(A, B);
    double relativeFroError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, C);
    resultCsv->edit("AMMError", (double) relativeFroError);
    INTELLI_INFO("2.2 AMM error :" + to_string(relativeFroError));
    
    // 2.3 Save results
    std::string ruName = "MM";
    resultCsv->toFile(ruName + ".csv");
    INTELLI_INFO("Done. here is overall result");
    std::cout << resultCsv->toString() << endl;
}

int main(int argc, char **argv) {
    string configName, outPrefix = "";
    if (argc >= 2) {
        configName += argv[1];
    } else {
        configName = "config.csv";
    }
    benchmarkMM(configName);
    return 0;
}

