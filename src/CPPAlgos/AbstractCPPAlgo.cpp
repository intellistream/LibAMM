/*! \file AbstractMatrixLoader.h*/
//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/AbstractCPPAlgo.h>
#include <Utils/UtilityFunctions.h>
#include <time.h>
#include <chrono>
void LibAMM::AbstractCPPAlgo::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  useCuda = cfg->tryU64("useCuda", 0, false);
}

LibAMM::Tensor LibAMM::AbstractCPPAlgo::amm(LibAMM::Tensor A, LibAMM::Tensor B, uint64_t sketchSize) {
  assert(sketchSize);
  auto start = std::chrono::high_resolution_clock::now();
  // std::cout << "Tensor A size: " << A.sizes() << std::endl;
  // std::cout << "Tensor B size: " << B.sizes() << std::endl;
  LibAMM::Tensor C;
  if (useCuda) {
    INTELLI_INFO("I am mm, USING CUDA");
    auto ac = A.to(LibAMM::kCUDA);
    buildATime = chronoElapsedTime(start);
    auto bc = B.to(LibAMM::kCUDA);
    buildBTime = chronoElapsedTime(start) - buildATime;
    auto cc = LibAMM::matmul(ac, bc);
    fABTime = chronoElapsedTime(start) - buildATime - buildBTime;
    C = cc.to(LibAMM::kCPU);
    postProcessTime = chronoElapsedTime(start) - buildATime - buildBTime - fABTime;
  } else {
    C = LibAMM::matmul(A, B);
    fABTime = chronoElapsedTime(start);
  }

  return C;
}

INTELLI::ConfigMapPtr LibAMM::AbstractCPPAlgo::getBreakDown() {
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  cfg->edit("buildATime", (uint64_t) buildATime);
  cfg->edit("buildBTime", (uint64_t) buildBTime);
  cfg->edit("fABTime", (uint64_t) fABTime);
  cfg->edit("postProcessTime", (uint64_t) postProcessTime);
  return cfg;
}