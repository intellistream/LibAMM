// Copyright (C) 2021 by the IntelliStream team (https://github.com/intellistream)

/**
 * @brief This is the main entry point of the entire program.
 * We use this as the entry point for benchmarking.
 */
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <string>
#include <memory>
#include <Utils/ConfigMap.hpp>
#include <Utils/IntelliLog.h>
#include <Utils/ThreadPerf.hpp>
#include <Utils/UtilityFunctions.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
void runSingleThreadTest(std::string configName) {
  ConfigMapPtr cfg = newConfigMap();
  cfg->fromFile(configName);
  //555
  uint64_t aRow, aCol, bCol, sketchDimension;
  aRow = cfg->tryU64("aRow", 100, true);
  aCol = cfg->tryU64("aCol", 1000, true);
  bCol = cfg->tryU64("bCol", 500, true);
  sketchDimension = cfg->tryU64("sketchDimension", 50, true);
  uint64_t coreBind = cfg->tryU64("coreBind", 0, true);
  UtilityFunctions::bind2Core((int) coreBind);
  std::string ptFile = cfg->tryString("ptFile", "torchscripts/FDAMM.pt", true);
  uint64_t customResultName = cfg->tryU64("customResultName", 0, true);
  INTELLI_INFO("Place me at core" + to_string(coreBind));
  INTELLI_INFO(
      "Generating [" + to_string(aRow) + "x" + to_string(aCol) + "]*[" + to_string(aCol) + "x" + to_string(bCol)
          + "], with sketch" + to_string(sketchDimension));
  torch::jit::script::Module module;
  INTELLI_INFO("Try pt file " + ptFile);
  module = torch::jit::load(ptFile);
  torch::manual_seed(114514);
  //555
  auto A = torch::rand({(long) aRow, (long) aCol});
  auto B = torch::rand({(long) aCol, (long) bCol});
  INTELLI_INFO("Generation done, conducting...");
  ThreadPerf pef((int) coreBind);
  pef.setPerfList();
  pef.start();
  auto C =module.forward({A, B, (long) sketchDimension}).toTensor();
  pef.end();
  std::string ruName = "default";
  if (customResultName) {
    ruName = "ru_core_" + to_string(coreBind) + "_matrix_" + to_string(aRow) + "x" + to_string(aCol) + "amm"
        + to_string(aCol) + "x" + to_string(bCol) + "_sketch_" + to_string(sketchDimension);
  }

  auto resultCsv = pef.resultToConfigMap();
  resultCsv->toFile(ruName + ".csv");
  INTELLI_INFO("Done. here is result");
  std::cout << resultCsv->toString() << endl;
}

int main(int argc, char **argv) {
  string configName, outPrefix = "";
  if (argc >= 2) {
    configName += argv[1];
  } else {
    configName = "config.csv";
  }
  runSingleThreadTest(configName);
  return 0;
}

