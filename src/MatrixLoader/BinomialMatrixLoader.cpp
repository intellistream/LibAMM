//
// Created by haolan on 6/5/23.
//
#include <MatrixLoader/BinomialMatrixLoader.h>
#include <Utils/IntelliLog.h>

void LibAMM::BinomialMatrixLoader::paraseConfig(INTELLI::ConfigMapPtr cfg) {
  aRow = cfg->tryU64("aRow", 100, true);
  aCol = cfg->tryU64("aCol", 1000, true);
  bCol = cfg->tryU64("bCol", 500, true);
  seed = cfg->tryU64("seed", 114514, true);
  trials = cfg->tryU64("trials", 10, true);
  probability = cfg->tryDouble("probability", 0.5, true);
  INTELLI_INFO(
      "Generating [" + to_string(aRow) + "x" + to_string(aCol) + "]*[" + to_string(aCol) + "x" + to_string(bCol)
          + "]" + " Parameter: " + to_string(trials) + ", " + to_string(probability));
}

void LibAMM::BinomialMatrixLoader::generateAB() {
  LibAMM::manual_seed(seed);
  A = LibAMM::zeros({(long) aRow, (long) aCol});
  B = LibAMM::zeros({(long) aCol, (long) bCol});

  for (uint64_t i = 0; i < trials; i++) {

    // Create a tensor filled with random numbers between 0 and 1
    LibAMM::Tensor rand_tensor = LibAMM::rand({(long) aRow, (long) aCol});

    // Add the results of the Bernoulli trial to the binomial tensor
    A += (rand_tensor < probability)// TODO: Int conversion;
  }

  for (uint64_t i = 0; i < trials; i++) {

    // Create a tensor filled with random numbers between 0 and 1
    LibAMM::Tensor rand_tensor = LibAMM::rand({(long) aCol, (long) bCol});

    // Add the results of the Bernoulli trial to the binomial tensor
    B += (rand_tensor < probability)// TODO: Int conversion;
  }
}

//do nothing in abstract class
bool LibAMM::BinomialMatrixLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  paraseConfig(cfg);
  generateAB();
  return true;
}

LibAMM::Tensor LibAMM::BinomialMatrixLoader::getA() {
  return A;
}

LibAMM::Tensor LibAMM::BinomialMatrixLoader::getB() {
  return B;
}