//
// Created by tony on 24/05/23.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <AMMBench.h>
#include <Utils/ThreadPerf.hpp>
using namespace std;
using namespace INTELLI;
using namespace torch;

void runSingleThreadTest(std::string configName) {
  ConfigMapPtr cfg = newConfigMap();
  cfg->fromFile(configName);
  AMMBench::MatrixLoaderTable mLoaderTable;
  uint64_t sketchDimension;
  sketchDimension = cfg->tryU64("sketchDimension", 50, true);
  uint64_t coreBind = cfg->tryU64("coreBind", 0, true);
  UtilityFunctions::bind2Core((int) coreBind);
  torch::set_num_threads(1);
  std::string ptFile = cfg->tryString("ptFile", "torchscripts/FDAMM.pt", true);

  //uint64_t customResultName = cfg->tryU64("customResultName", 0, true);
  INTELLI_INFO("Place me at core" + to_string(coreBind));
  INTELLI_INFO(
      "with sketch" + to_string(sketchDimension));
  torch::jit::script::Module module;
  INTELLI_INFO("Try pt file " + ptFile);
  module = torch::jit::load(ptFile);
  std::string matrixLoaderTag = cfg->tryString("matrixLoaderTag", "random", true);
  auto matLoaderPtr = mLoaderTable.findMatrixLoader(matrixLoaderTag);
  assert(matLoaderPtr);
  matLoaderPtr->setConfig(cfg);
  auto A = matLoaderPtr->getA();
  auto B = matLoaderPtr->getB();
  /*torch::manual_seed(114514);
//555
auto A = torch::rand({(long) aRow, (long) aCol});
auto B = torch::rand({(long) aCol, (long) bCol});*/
  INTELLI_INFO("Generation done, conducting...");
  ThreadPerf pef((int) coreBind);
  pef.setPerfList();
  pef.start();
  auto C =module.forward({A, B, (long) sketchDimension}).toTensor();
  pef.end();
  std::string ruName = "default";

  auto resultCsv = pef.resultToConfigMap();
  resultCsv->toFile(ruName + ".csv");
  INTELLI_INFO("Done. here is result");
  std::cout << resultCsv->toString() << endl;
}

TEST_CASE("Test the parallelization, thread=2", "[short]")
{
  int a = 0;
  ConfigMapPtr cfg = newConfigMap();
  cfg->edit("ptFile", "torchscripts/RAWMM.pt");
  cfg->edit("threads", (uint64_t) 2);
  torch::manual_seed(114514);
  auto A = torch::rand({(long) 4, (long) 4});
  auto B = torch::rand({(long) 4, (long) 4});
  AMMBench::BlockPartitionRunner br;
  br.setConfig(cfg);
  auto C1 = br.runAMM(A, B);
  auto C2 = torch::matmul(A, B);
  std::cout << "parallel MM" << endl;
  std::cout << C1 << std::endl;
  std::cout << "raw MM" << endl;
  std::cout << C2 << std::endl;
  //runSingleThreadTest("scripts/config_CRS.csv");
  // place your test here
  REQUIRE(a == 0);
}