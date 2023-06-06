/*! \file AbstractMatrixLoader.h*/
//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/AbstractCPPAlgo.h>
void AMMBench::AbstractCPPAlgo::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
}
torch::Tensor AMMBench::AbstractCPPAlgo::amm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize) {
  std::cout << sketchSize;
  INTELLI_INFO("I am mm");
  return torch::matmul(A, B);
}