//
// Created by haolan on 6/5/23.
//
#include <MatrixLoader/ExponentialMatrixLoader.h>
#include <Utils/IntelliLog.h>

void LibAMM::ExponentialMatrixLoader::paraseConfig(INTELLI::ConfigMapPtr cfg) {
  aRow = cfg->tryU64("aRow", 100, true);
  aCol = cfg->tryU64("aCol", 1000, true);
  bCol = cfg->tryU64("bCol", 500, true);
  seed = cfg->tryU64("seed", 114514, true);
  INTELLI_INFO(
      "Generating [" + to_string(aRow) + "x" + to_string(aCol) + "]*[" + to_string(aCol) + "x" + to_string(bCol)
          + "]");
}

void LibAMM::ExponentialMatrixLoader::generateAB() {
  LibAMM::manual_seed(seed);
  A = LibAMM::exponential(LibAMM::empty({(long) aRow, (long) aCol}));
  B = LibAMM::exponential(LibAMM::empty({(long) aCol, (long) bCol}));
}

//do nothing in abstract class
bool LibAMM::ExponentialMatrixLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  paraseConfig(cfg);
  generateAB();
  return true;
}

LibAMM::Tensor LibAMM::ExponentialMatrixLoader::getA() {
  return A;
}

LibAMM::Tensor LibAMM::ExponentialMatrixLoader::getB() {
  return B;
}