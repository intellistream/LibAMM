//
// Created by tony on 25/12/22.
//

#include <AMMBench.h>
#include <string>
#include <Streaming/Streamer.h>

using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace AMMBench;
int64_t algoTag = 0;
std::string algoTagStr = "mm";
/**
 *
 * @brief The c++ bindings to crs.
 * @param a
 * @param b
 * @note please keep input and output as tensors
 * @return tensor
 */
torch::Tensor AMMBench_crs(torch::Tensor a, torch::Tensor b) {
  AMMBench::CRSCPPAlgo algo;
  auto w = a.sizes()[1] / 10;
  return algo.amm(a, b, (uint64_t) w);
}
torch::Tensor AMMBench_ammDefault(torch::Tensor a, torch::Tensor b) {
  AMMBench::CPPAlgoTable cppAlgoTable;
  AMMBench::AbstractCPPAlgoPtr cppAlgoPtr = cppAlgoTable.findCppAlgo(algoTagStr);
  auto w = a.sizes()[1] / 10;
  return cppAlgoPtr->amm(a, b, (uint64_t) w);
}

torch::Tensor AMMBench_ammSpecifySs(torch::Tensor a, torch::Tensor b, int64_t ss) {
  AMMBench::CPPAlgoTable cppAlgoTable;
  AMMBench::AbstractCPPAlgoPtr cppAlgoPtr = cppAlgoTable.findCppAlgo(algoTagStr);
  return cppAlgoPtr->amm(a, b, (uint64_t) ss);
}

void AMMBench_setTag(std::string tag) {
  algoTagStr = tag;
}

torch::Tensor AMMBench_ammForMadness(torch::Tensor A, torch::Tensor B, string configPath, string metricSavePath) {

  // For madness. we need to 
  // 1. load cfg, extract amm method, and related amm paramater (e.g. vq codebook path).
  // 2. generate metric including AMM time, AMM fro error 
  // we do not need to
  // 1. create matrixLoader as a,b is passed in already.

  // 1. Set up environments
  ConfigMapPtr cfg = newConfigMap();
  cfg->fromFile(configPath);

  INTELLI_INFO("cppAlgoTag: " + cfg->tryString("cppAlgoTag", "mm", true));

  uint64_t coreBind = cfg->tryU64("coreBind", 0, true);
  UtilityFunctions::bind2Core((int) coreBind);

  uint64_t sketchDimension;
  sketchDimension = cfg->tryU64("sketchDimension", 1, true);
  INTELLI_INFO("sketchDimension: " + to_string(sketchDimension));

  // 2. run AMM
  Streamer streamer;
  torch::Tensor C = streamer.run(cfg, A, B, sketchDimension, "AMM");
  ConfigMapPtr allMetrics = streamer.getMetrics();

  allMetrics->toFile(metricSavePath);
  INTELLI_INFO("Done. here is overall result");
  INTELLI_INFO(allMetrics->toString());

  return C;
}

// END o
/**
 * @brief Declare the function to pytorch
 * @note The of lib is myLib
 */
TORCH_LIBRARY(AMMBench, m2) {
  m2.def("crs", AMMBench_crs);
  m2.def("setTag", AMMBench_setTag);
  m2.def("ammDefault", AMMBench_ammDefault);
  m2.def("ammSpecifySs", AMMBench_ammSpecifySs);
  m2.def("ammForMadness", AMMBench_ammForMadness);
  //m2.def("myVecSub", myVecSub);
}