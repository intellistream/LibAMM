//
// Created by tony on 10/05/23.
//

#include <MatrixLoader/AbstractMatrixLoader.h>

//do nothing in abstract class
bool LibAMM::AbstractMatrixLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  return true;
}

LibAMM::Tensor LibAMM::AbstractMatrixLoader::getA() {
  return LibAMM::rand({1, 1});
}

LibAMM::Tensor LibAMM::AbstractMatrixLoader::getB() {
  return LibAMM::rand({1, 1});
}
