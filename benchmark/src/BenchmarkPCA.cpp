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


void benchmarkPCA(std::string configName){

    // Step1. Set up environments
    ConfigMapPtr cfg = newConfigMap();
    cfg->fromFile(configName);

    // 1.1 AMM algorithm
    AMMBench::CPPAlgoTable cppAlgoTable;
    std::string cppAlgoTag = cfg->tryString("cppAlgoTag", "mm", true);
    AMMBench::AbstractCPPAlgoPtr cppAlgoPtr = cppAlgoTable.findCppAlgo(cppAlgoTag);
    cppAlgoPtr->setConfig(cfg);
    INTELLI_INFO("1.1 algo: " + cppAlgoTag);

    // 1.2 matrixLoader uses SIFT dataset
    AMMBench::MatrixLoaderTable mLoaderTable;
    std::string matrixLoaderTag = cfg->tryString("matrixLoaderTag", "SIFT", true);
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

    // 1.5 Meter
    // MeterTable meterTable;
    // AbstractMeterPtr eMeter = nullptr;
    // uint64_t usingMeter = cfg->tryU64("usingMeter", 0, true);
    // std::string meterTag = cfg->tryString("meterTag", "intelMsr", true);
    // INTELLI_INFO("1.5 meterTag:" + meterTag);
    // if (usingMeter) {
    //     eMeter = meterTable.findMeter(meterTag);
    //     if (eMeter != nullptr) {
    //         eMeter->setConfig(cfg);
    //         double staticPower = cfg->tryDouble("staticPower", 0.0, false);
    //         if (staticPower == 0.0) {
    //             eMeter->testStaticPower(2);
    //         } else {
    //             INTELLI_INFO("use pre-defined static power");
    //             eMeter->setStaticPower(staticPower);
    //         }
    //         INTELLI_INFO("static power is " + to_string(eMeter->getStaticPower()) + " W");
    //     } else {
    //         INTELLI_ERROR("No meter found: " + meterTag);
    //     }
    // }

    // 1.6 ThreadPerf
    ThreadPerf pef(-1);
    pef.setPerfList();

    // Step2. Test elapsedTime and error on AMM in streaming and parallelism
    // 2.1 Run AMM
    // TODO Fow now, use single thread + no streaming, later will change
    INTELLI_INFO("2.1 no streaming, single thread, force MP");
    AMMBench::BlockPartitionRunner br;
    br.setConfig(cfg);
    br.createABC(A, B);

    // if (eMeter != nullptr) {
    //     eMeter->startMeter();
    // }
    struct timeval tstart, tend;
    // pef.start();
    torch::Tensor C = br.parallelForward(); // TODO change here, run a function that execute AMM in streaming/non-streaming and single-thread/multi-thread according to the config file
    // pef.end();
    // if (eMeter != nullptr) {
    //     eMeter->stopMeter();
    // }
    INTELLI_INFO("AMM finished in " + to_string(br.getElapsedTime()));

    // 2.2 elapsed time and error for AMM
    ConfigMapPtr resultCsv = newConfigMap();
    // auto resultCsv = pef.resultToConfigMap();
    // if (eMeter != nullptr) {
    //     double energyConsumption = eMeter->getE();
    //     double staticEnergyConsumption = eMeter->getStaicEnergyConsumption(resultCsv->tryU64("perfElapsedTime", 0, false));
    //     double pureEnergy = energyConsumption - staticEnergyConsumption;
    //     resultCsv->edit("energyAll", (double) energyConsumption);
    //     resultCsv->edit("energyOnlyMe", (double) pureEnergy);
    // }
    resultCsv->edit("AMMElapsedTime", (uint64_t) br.getElapsedTime());
    br.appendThreadInfo(resultCsv);

    torch::Tensor realC = torch::matmul(A, B);
    double relativeFroError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, C);
    resultCsv->edit("AMMError", (double) relativeFroError);

    // Step3. Test accuracy on PCA task
    // 3.1 elapsed time for other tasks in PCA except AMM
    INTELLI_INFO("Start SVD task..");
    gettimeofday(&tstart, NULL);
    torch::Tensor UCovC;
    torch::Tensor SCovC;
    torch::Tensor VhCovC;
    torch::Tensor covC = C/A.size(1); // covirance estimator
    std::tie(UCovC, SCovC, VhCovC) = torch::linalg::svd(covC, false, c10::nullopt);
    gettimeofday(&tend, NULL);
    INTELLI_INFO("SVD finished in " + to_string(INTELLI::UtilityFunctions::timeLast(tstart, tend)));
    resultCsv->edit("ElseElapsedTime", (uint64_t) INTELLI::UtilityFunctions::timeLast(tstart, tend));

    // 3.2 PCA relative spectral error
    torch::Tensor covRealC = realC/A.size(1);
    double relativeSpectralNormError = INTELLI::UtilityFunctions::relativeSpectralNormError(covRealC, covC);
    resultCsv->edit("PCAError", (double) relativeSpectralNormError);
    
    // 3.3 Save results
    std::string ruName = "PCA";
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
    benchmarkPCA(configName);
    return 0;
}

