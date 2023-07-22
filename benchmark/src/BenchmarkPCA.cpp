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

void benchmarkPCA(std::string configName) {

  INTELLI_INFO("Running Benchmark PCA");

  // Step1. Set up environments
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
  INTELLI_INFO("sketchDimension: " + sketchDimension);

  // Step2. Test elapsedTime and error on AMM in streaming and parallelism
  // 2.1 Run AMM
  INTELLI_INFO("Run AMM");
  Streamer streamer;
  torch::Tensor C = streamer.run(cfg, A, B, sketchDimension, "AMM");
  ConfigMapPtr resultCsv = streamer.getMetrics();

  // Step3. Test accuracy on PCA task
  // 3.1 elapsed time for other tasks in PCA except AMM
  INTELLI_INFO("Start SVD task..");
  struct timeval tstart, tend;
  gettimeofday(&tstart, NULL);
  torch::Tensor UCovC;
  torch::Tensor SCovC;
  torch::Tensor VhCovC;
  torch::Tensor covC = C / A.size(1); // covirance estimator
  std::tie(UCovC, SCovC, VhCovC) = torch::linalg::svd(covC, false, c10::nullopt);
  gettimeofday(&tend, NULL);
  INTELLI_INFO("SVD finished in " + to_string(INTELLI::UtilityFunctions::timeLast(tstart, tend)));
  resultCsv->edit("SVDElapsedTime", (uint64_t) INTELLI::UtilityFunctions::timeLast(tstart, tend));

  // 3.2 PCA relative spectral error
  torch::Tensor realC = torch::matmul(A, B);
  torch::Tensor covRealC = realC / A.size(1);
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

