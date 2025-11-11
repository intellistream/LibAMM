//
// Created by haolan on 6/5/23.
//
#include <MatrixLoader/CCAMatrixLoader.h>
#include <Utils/IntelliLog.h>
#include <LibAMM.h>
#include <fstream>
#include <iostream>

void LibAMM::CCAMatrixLoader::paraseConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
}

void LibAMM::CCAMatrixLoader::generateAB() {
  INTELLI_INFO("This function is not used");
}

//do nothing in abstract class
bool LibAMM::CCAMatrixLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  paraseConfig(cfg);
  generateAB();
  return true;
}

LibAMM::Tensor LibAMM::CCAMatrixLoader::getA() {
  return A;
}

LibAMM::Tensor LibAMM::CCAMatrixLoader::getB() {
  return B;
}

LibAMM::Tensor LibAMM::CCAMatrixLoader::getAt() {
  return At;
}

LibAMM::Tensor LibAMM::CCAMatrixLoader::getBt() {
  return Bt;
}

LibAMM::Tensor LibAMM::CCAMatrixLoader::getSxx() {
  return Sxx;
}

LibAMM::Tensor LibAMM::CCAMatrixLoader::getSyy() {
  return Syy;
}

LibAMM::Tensor LibAMM::CCAMatrixLoader::getSxy() {
  return Sxy;
}

LibAMM::Tensor LibAMM::CCAMatrixLoader::getSxxNegativeHalf() {
  return SxxNegativeHalf;
}

LibAMM::Tensor LibAMM::CCAMatrixLoader::getSyyNegativeHalf() {
  return SyyNegativeHalf;
}

LibAMM::Tensor LibAMM::CCAMatrixLoader::getM() {
  return M;
}

LibAMM::Tensor LibAMM::CCAMatrixLoader::getM1() {
  return M1;
}

LibAMM::Tensor LibAMM::CCAMatrixLoader::getCorrelation() {
  return correlation;
}

void LibAMM::CCAMatrixLoader::calculate_correlation() {
  
  // Sxx, Syy, Sxy: covariance matrix
  Sxx = LibAMM::matmul(A, At) / A.size(1); // 120*120
  Syy = LibAMM::matmul(B, Bt) / A.size(1); // 101*101
  Sxy = LibAMM::matmul(A, Bt) / A.size(1); // 120*101

  // Sxx^(-1/2), Syy^(-1/2), M
  // Sxx^(-1/2) 120*120
  LibAMM::Tensor eigenvaluesSxx, eigenvectorsSxx;
  std::tie(eigenvaluesSxx, eigenvectorsSxx) = torch::linalg::eig(Sxx); // diagonization
  LibAMM::Tensor diagonalMatrixSxx = LibAMM::diag(
      1.0 / LibAMM::sqrt(eigenvaluesSxx + torch::full({}, 1e-12))); // 1/sqrt(eigenvalue+epsilon) +epsilon to avoid nan
  SxxNegativeHalf = LibAMM::matmul(LibAMM::matmul(eigenvectorsSxx, diagonalMatrixSxx), eigenvectorsSxx.t());
  SxxNegativeHalf = at::real(SxxNegativeHalf); // ignore complex part, it comes from numerical computations
  // Syy^(-1/2) 101*101
  LibAMM::Tensor eigenvaluesSyy, eigenvectorsSyy;
  std::tie(eigenvaluesSyy, eigenvectorsSyy) = torch::linalg::eig(Syy);
  LibAMM::Tensor diagonalMatrixSyy = LibAMM::diag(1.0 / LibAMM::sqrt(eigenvaluesSyy + torch::full({}, 1e-12)));
  SyyNegativeHalf = LibAMM::matmul(LibAMM::matmul(eigenvectorsSyy, diagonalMatrixSyy), eigenvectorsSyy.t());
  SyyNegativeHalf = at::real(SyyNegativeHalf);
  // M 120*101
  M1 = LibAMM::matmul(SxxNegativeHalf.t(), Sxy);
  M = LibAMM::matmul(M1, SyyNegativeHalf);

  // correlation
  LibAMM::Tensor U, S, Vh;
  std::tie(U, S, Vh) = torch::linalg::svd(M, false, c10::nullopt);
  correlation = LibAMM::clamp(S, -1.0, 1.0);
}
