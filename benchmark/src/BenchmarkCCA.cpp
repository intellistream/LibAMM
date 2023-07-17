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

void printStatsOfTensor(torch::Tensor M){
    torch::Tensor mean = at::mean(M);
    torch::Tensor std = at::std(M);
    torch::Tensor max_val = at::max(M);
    torch::Tensor min_val = at::min(M);

    // Print the results
    std::cout << "Mean:" << mean.item<float>() << std::endl;
    std::cout << "Standard Deviation:" << std.item<float>() << std::endl;
    std::cout << "Max value: " << max_val.item<float>() << std::endl;
    std::cout << "Min value: " << min_val.item<float>() << std::endl;
}

void benchmarkCCA(std::string configName){

    // Step1. Set up environments
    ConfigMapPtr cfg = newConfigMap();
    cfg->fromFile(configName);

    // 1.1 AMM algorithm
    AMMBench::CPPAlgoTable cppAlgoTable;
    std::string cppAlgoTag = cfg->tryString("cppAlgoTag", "mm", true);
    AMMBench::AbstractCPPAlgoPtr cppAlgoPtr = cppAlgoTable.findCppAlgo(cppAlgoTag);
    cppAlgoPtr->setConfig(cfg);
    INTELLI_INFO("1.1 algo: " + cppAlgoTag);

    // 1.2 matrixLoader uses MNIST dataset
    AMMBench::MatrixLoaderTable mLoaderTable;
    std::shared_ptr<AMMBench::AbstractMatrixLoader> basePtr = mLoaderTable.findMatrixLoader("MNIST");
	std::shared_ptr<AMMBench::MNISTMatrixLoader> matLoaderPtr = std::dynamic_pointer_cast<AMMBench::MNISTMatrixLoader>(basePtr);
    INTELLI_INFO("1.2 matrixLoaderTag: MNIST");
    assert(matLoaderPtr);
    matLoaderPtr->setConfig(cfg);
    auto A = matLoaderPtr->getA();
    auto B = matLoaderPtr->getB();
    // torch::save(A, "A.pt");
    // torch::save(B, "B.pt");
    matLoaderPtr->calculate_correlation(); // cleaner code

    // 1.3 sketch dimension
    uint64_t sketchDimension;
    sketchDimension = cfg->tryU64("sketchDimension", 50, true);
    INTELLI_INFO("1.3 sketch dimension: " + to_string(sketchDimension));

    // 1.4 coreBind
    uint64_t coreBind = cfg->tryU64("coreBind", 0, true);
    UtilityFunctions::bind2Core((int) coreBind);
    INTELLI_INFO("1.4 corebind: " + to_string(coreBind));

    // 1.6 ThreadPerf
    ThreadPerf pef(-1);
    pef.setPerfList();

    // Step2. Test elapsedTime and error on CCA task, including AMM and other operations ELSE

    INTELLI_INFO("2. no streaming, single thread, force MP, will be changed in future");
    struct timeval tstart, tend;
    double relativeFroError;
    ConfigMapPtr resultCsv = newConfigMap();
    AMMBench::BlockPartitionRunner br;
    br.setConfig(cfg);

    // 2.1 AMM Sxx
    std::cout << "\033[1;34m 2.1 Sxx \033[0m" << std::endl;
    gettimeofday(&tstart, NULL);
    torch::manual_seed(123);
    torch::Tensor Sxx = cppAlgoPtr->amm(A, A.t(), sketchDimension);
    gettimeofday(&tend, NULL);
    Sxx = Sxx/A.size(1);
    resultCsv->edit("AMMSxxElapsedTime", (uint64_t) INTELLI::UtilityFunctions::timeLast(tstart, tend));
    relativeFroError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(matLoaderPtr->getSxx(), Sxx);
    resultCsv->edit("AMMSxxError", (double) relativeFroError);
    INTELLI_INFO("2.1 AMMSxxElapsedTime: " + to_string(INTELLI::UtilityFunctions::timeLast(tstart, tend)) + " AMMSxxError: " + to_string(relativeFroError));
    printStatsOfTensor(Sxx);
    printStatsOfTensor(matLoaderPtr->getSxx());
    printStatsOfTensor(matLoaderPtr->getSxx()-Sxx);

    // 2.2 AMM Syy
    std::cout << "\033[1;34m 2.2 Syy \033[0m" << std::endl;
    gettimeofday(&tstart, NULL);
    torch::manual_seed(123);
    torch::Tensor Syy = cppAlgoPtr->amm(B, B.t(), sketchDimension);
    gettimeofday(&tend, NULL);
    Syy = Syy/A.size(1);
    resultCsv->edit("AMMSyyElapsedTime", (uint64_t) INTELLI::UtilityFunctions::timeLast(tstart, tend));
    relativeFroError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(matLoaderPtr->getSyy(), Syy);
    resultCsv->edit("AMMSyyError", (double) relativeFroError);
    INTELLI_INFO("2.2 AMMSyyElapsedTime: " + to_string(INTELLI::UtilityFunctions::timeLast(tstart, tend)) + " AMMSyyError: " + to_string(relativeFroError));
    printStatsOfTensor(Syy);
    printStatsOfTensor(matLoaderPtr->getSyy());
    printStatsOfTensor(matLoaderPtr->getSyy()-Syy);

    // 2.3 AMM: Sxy
    std::cout << "\033[1;34m 2.3 Sxy \033[0m" << std::endl;
    gettimeofday(&tstart, NULL);
    torch::manual_seed(123);
    torch::Tensor Sxy = cppAlgoPtr->amm(A, B.t(), sketchDimension);
    gettimeofday(&tend, NULL);
    Sxy = Sxy/A.size(1);
    resultCsv->edit("AMMSxyElapsedTime", (uint64_t) INTELLI::UtilityFunctions::timeLast(tstart, tend));
    relativeFroError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(matLoaderPtr->getSxy(), Sxy);
    resultCsv->edit("AMMSxyError", (double) relativeFroError);
    INTELLI_INFO("2.3 AMMSxyElapsedTime: " + to_string(INTELLI::UtilityFunctions::timeLast(tstart, tend)) + " AMMSxyError: " + to_string(relativeFroError));
    printStatsOfTensor(Sxy);
    printStatsOfTensor(matLoaderPtr->getSxy());
    printStatsOfTensor(matLoaderPtr->getSxy()-Sxy);

    // 2.4 Sxx^(-1/2), Syy^(-1/2)
    // Sxx^(-1/2)
    std::cout << "\033[1;34m 2.4 Sxx^(-1/2) \033[0m" << std::endl;
    gettimeofday(&tstart, NULL);
    torch::Tensor eigenvaluesSxx, eigenvectorsSxx;
	std::tie(eigenvaluesSxx, eigenvectorsSxx) = torch::linalg::eig(Sxx); // diagonization
	torch::Tensor diagonalMatrixSxx = torch::diag(1.0 / torch::sqrt(eigenvaluesSxx+torch::full({}, 1e-12))); // 1/sqrt(eigenvalue+epsilon) +epsilon to avoid nan
	torch::Tensor SxxNegativeHalf = torch::matmul(torch::matmul(eigenvectorsSxx, diagonalMatrixSxx), eigenvectorsSxx.t());
    SxxNegativeHalf = at::real(SxxNegativeHalf); // ignore complex part, it comes from numerical computations
    gettimeofday(&tend, NULL);
	resultCsv->edit("SxxNegativeHalfElapsedTime", (uint64_t) INTELLI::UtilityFunctions::timeLast(tstart, tend));
    relativeFroError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(matLoaderPtr->getSxxNegativeHalf(), SxxNegativeHalf);
    resultCsv->edit("SxxNegativeHalfError", (double) relativeFroError);
    INTELLI_INFO("2.4 SxxNegativeHalfElapsedTime: " + to_string(INTELLI::UtilityFunctions::timeLast(tstart, tend)) + " SxxNegativeHalfError: " + to_string(relativeFroError));
    printStatsOfTensor(SxxNegativeHalf);
    printStatsOfTensor(matLoaderPtr->getSxxNegativeHalf());
    printStatsOfTensor(matLoaderPtr->getSxxNegativeHalf()-SxxNegativeHalf);
    
    // Syy^(-1/2)
    std::cout << "\033[1;34m 2.4 Syy^(-1/2) \033[0m" << std::endl;
    gettimeofday(&tstart, NULL);
	torch::Tensor eigenvaluesSyy, eigenvectorsSyy;
	std::tie(eigenvaluesSyy, eigenvectorsSyy) = torch::linalg::eig(Syy);
	torch::Tensor diagonalMatrixSyy = torch::diag(1.0 / torch::sqrt(eigenvaluesSyy+torch::full({}, 1e-12)));
	torch::Tensor SyyNegativeHalf = torch::matmul(torch::matmul(eigenvectorsSyy, diagonalMatrixSyy), eigenvectorsSyy.t());
	SyyNegativeHalf = at::real(SyyNegativeHalf);
    gettimeofday(&tend, NULL);
    resultCsv->edit("SyyNegativeHalfElapsedTime", (uint64_t) INTELLI::UtilityFunctions::timeLast(tstart, tend));
    relativeFroError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(matLoaderPtr->getSyyNegativeHalf(), SyyNegativeHalf);
    resultCsv->edit("SyyNegativeHalfError", (double) relativeFroError);
    INTELLI_INFO("2.4 SyyNegativeHalfElapsedTime: " + to_string(INTELLI::UtilityFunctions::timeLast(tstart, tend)) + " SyyNegativeHalfError: " + to_string(relativeFroError));
    printStatsOfTensor(SyyNegativeHalf);
    printStatsOfTensor(matLoaderPtr->getSyyNegativeHalf());
    printStatsOfTensor(matLoaderPtr->getSyyNegativeHalf()-SyyNegativeHalf);

    // 2.5 AMM M
    std::cout << "\033[1;34m 2.5 M \033[0m" << std::endl;
    gettimeofday(&tstart, NULL);
    torch::Tensor M = torch::matmul(torch::matmul(SxxNegativeHalf, Sxy), SyyNegativeHalf);
    gettimeofday(&tend, NULL);
    resultCsv->edit("AMMMElapsedTime", (uint64_t) INTELLI::UtilityFunctions::timeLast(tstart, tend));
    relativeFroError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(matLoaderPtr->getM(), M);
    resultCsv->edit("AMMMError", (double) relativeFroError);
    INTELLI_INFO("2.5 AMMMElapsedTime: " + to_string(INTELLI::UtilityFunctions::timeLast(tstart, tend)) + " AMMMError: " + to_string(relativeFroError));
    printStatsOfTensor(M);
    printStatsOfTensor(matLoaderPtr->getM());
    printStatsOfTensor(matLoaderPtr->getM()-M);

    // 2.6 correlation
    std::cout << "\033[1;34m 2.6 correlation \033[0m" << std::endl;
    gettimeofday(&tstart, NULL);
    torch::Tensor U, S, Vh;
    std::tie(U, S, Vh) = torch::linalg::svd(M, false, c10::nullopt);
    torch::Tensor correlation = torch::clamp(S, -1.0, 1.0);
    gettimeofday(&tend, NULL);
    resultCsv->edit("ElseCorrelationElapsedTime", (uint64_t) INTELLI::UtilityFunctions::timeLast(tstart, tend));
    relativeFroError = (correlation - matLoaderPtr->getCorrelation()).abs().max().item<double>();
    resultCsv->edit("CorrelationError", (double) relativeFroError);
    INTELLI_INFO("2.6 max correlation error : " + to_string(relativeFroError));
    printStatsOfTensor(correlation);
    printStatsOfTensor(matLoaderPtr->getCorrelation());
    printStatsOfTensor(matLoaderPtr->getCorrelation()-correlation);
    
    // Save results
    std::string ruName = "CCA";
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
    benchmarkCCA(configName);
    return 0;
}