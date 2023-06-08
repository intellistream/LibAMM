/*! \file AbstractMatrixLoader.h*/
//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/AbstractCPPAlgo.h>
#include <Utils/UtilityFunctions.h>
#include <time.h>
void AMMBench::AbstractCPPAlgo::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
}
torch::Tensor AMMBench::AbstractCPPAlgo::amm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize) {
  assert(sketchSize);
  struct timeval tstart;
  //INTELLI_INFO("I am mm");
  gettimeofday(&tstart,NULL);
  auto C= torch::matmul(A, B);
  fABTime=INTELLI::UtilityFunctions::timeLastUs(tstart);
  return C;
}
INTELLI::ConfigMapPtr AMMBench::AbstractCPPAlgo::getBreakDown() {
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  cfg->edit("buildATime", (uint64_t)buildATime);
  cfg->edit("buildBTime", (uint64_t)buildBTime);
  cfg->edit("fABTime", (uint64_t)fABTime);
  cfg->edit("postProcessTime",(uint64_t)postProcessTime);
  return cfg;
}