//
// Created by haolan on 6/5/23.
//
#include <MatrixLoader/MediaMillMatrixLoader.h>
#include <Utils/IntelliLog.h>
#include <LibAMM.h>
#include <fstream>
#include <iostream>

void LibAMM::MediaMillMatrixLoader::paraseConfig(INTELLI::ConfigMapPtr cfg) {
  filePath = cfg->tryString("filePath", "datasets/Mediamill/MediaMill.pth", true);
}

void LibAMM::MediaMillMatrixLoader::generateAB() {

  torch::jit::script::Module tensors = torch::jit::load(filePath);
  // A, B already normalized
  A = tensors.attr("A").toTensor().contiguous().to(torch::kFloat); // 120*43907
  B = tensors.attr("B").toTensor().contiguous().to(torch::kFloat); // 101*43907

  At = A.t().contiguous().to(torch::kFloat); // 43907*120
  Bt = B.t().contiguous().to(torch::kFloat); // 43907*101

  // std::cout << "Maximum Value: " << A.max().item<float>() << std::endl;
  // std::cout << "Mean Value: " << A.mean().item<float>() << std::endl;
  // std::cout << "Minimum Value: " << A.min().item<float>() << std::endl;
  // std::cout << "Maximum Value: " << B.max().item<float>() << std::endl;
  // std::cout << "Mean Value: " << B.mean().item<float>() << std::endl;
  // std::cout << "Minimum Value: " << B.min().item<float>() << std::endl;

  INTELLI_INFO(
      "Generating [" + to_string(A.size(0)) + " x " + to_string(A.size(1)) + "]*[" + to_string(B.size(0)) + " x "
          + to_string(B.size(1)) + "]");
}

//do nothing in abstract class
bool LibAMM::MediaMillMatrixLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  paraseConfig(cfg);
  generateAB();
  return true;
}

LibAMM::Tensor LibAMM::MediaMillMatrixLoader::getA() {
  return A;
}

LibAMM::Tensor LibAMM::MediaMillMatrixLoader::getB() {
  return B;
}

LibAMM::Tensor LibAMM::MediaMillMatrixLoader::getAt() {
  return At;
}

LibAMM::Tensor LibAMM::MediaMillMatrixLoader::getBt() {
  return Bt;
}

LibAMM::Tensor LibAMM::MediaMillMatrixLoader::getSxx() {
  return Sxx;
}

LibAMM::Tensor LibAMM::MediaMillMatrixLoader::getSyy() {
  return Syy;
}

LibAMM::Tensor LibAMM::MediaMillMatrixLoader::getSxy() {
  return Sxy;
}

LibAMM::Tensor LibAMM::MediaMillMatrixLoader::getSxxNegativeHalf() {
  return SxxNegativeHalf;
}

LibAMM::Tensor LibAMM::MediaMillMatrixLoader::getSyyNegativeHalf() {
  return SyyNegativeHalf;
}

LibAMM::Tensor LibAMM::MediaMillMatrixLoader::getM() {
  return M;
}

LibAMM::Tensor LibAMM::MediaMillMatrixLoader::getM1() {
  return M1;
}

LibAMM::Tensor LibAMM::MediaMillMatrixLoader::getCorrelation() {
  return correlation;
}

void LibAMM::MediaMillMatrixLoader::calculate_correlation() {
  
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
