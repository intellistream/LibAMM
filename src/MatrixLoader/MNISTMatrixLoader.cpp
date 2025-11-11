//
// Created by haolan on 6/5/23.
//
#include <MatrixLoader/MNISTMatrixLoader.h>
#include <Utils/IntelliLog.h>
#include <LibAMM.h>
#include <fstream>
#include <iostream>

void LibAMM::MNISTMatrixLoader::paraseConfig(INTELLI::ConfigMapPtr cfg) {
  filePath = cfg->tryString("filePath", "datasets/MNIST/train-images.idx3-ubyte", true);
}

int reverseInt(int i) {

  unsigned char c1, c2, c3, c4;

  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;

  return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

void LibAMM::MNISTMatrixLoader::generateAB() {

  int magic_number = 0;
  int number_of_images = 0;
  int n_rows = 0;
  int n_cols = 0;

  // Dynamically allocate memory for left half and right half
  At = LibAMM::zeros({60000, 28 * 14});
  Bt = LibAMM::zeros({60000, 28 * 14});

  // Read in file
  ifstream file(filePath, ios::binary);
  if (file.is_open()) {

    // cout << "Reading metadata ..." << endl;

    file.read((char *) &magic_number, sizeof(magic_number));//幻数（文件格式）
    file.read((char *) &number_of_images, sizeof(number_of_images));//图像总数
    file.read((char *) &n_rows, sizeof(n_rows));//每个图像的行数
    file.read((char *) &n_cols, sizeof(n_cols));//每个图像的列数

    magic_number = reverseInt(magic_number);
    number_of_images = reverseInt(number_of_images);
    n_rows = reverseInt(n_rows);
    n_cols = reverseInt(n_cols);
    INTELLI_INFO(
        "File format:" + to_string(magic_number) + " Number of Images:" + to_string(number_of_images)
            + " Number of Rows:" + to_string(n_rows) + " Number of Cols:" + to_string(n_cols));

    // cout << "Reading images......" << endl;

    for (int i = 0; i < number_of_images; i++) {
      for (int j = 0; j < n_rows * n_cols; j++) {
        unsigned char temp = 0;
        file.read((char *) &temp, sizeof(temp));
        //可以在下面这一步将每个像素值归一化
        float pixel_value = float(temp);
        if (pixel_value > 255 || pixel_value < 0) {
          std::cout << pixel_value << " " << endl;
        }
        if ((j % 28) < 14) {
          At[i][int(floor(static_cast<double>(j) / 28) * 14 + (j % 28))] = pixel_value; // 60000*392
        } else {
          Bt[i][int(floor(static_cast<double>(j) / 28) * 14 + (j % 28) - 14)] = pixel_value; // 60000*392
        }
      }
    }

    // cout << "Finished reading images......" << endl;
  }
  file.close();

  // auto pickledA = torch::pickle_save(A);
  // std::ofstream foutA("A.pt", std::ios::out | std::ios::binary);
  // foutA.write(pickledA.data(), pickledA.size());
  // foutA.close();

  // auto pickledB = torch::pickle_save(B);
  // std::ofstream foutB("B.pt", std::ios::out | std::ios::binary);
  // foutB.write(pickledB.data(), pickledB.size());
  // foutB.close();


  // normalization and transpose
  At = (At - At.mean(/*dim=*/0))/(At.std(/*dim=*/0)+1e-7); // standardized along feature (column) 60000*392
  Bt = (Bt - Bt.mean(/*dim=*/0))/(Bt.std(/*dim=*/0)+1e-7); // standardized along feature (column) 60000*392

  A = At.t().contiguous(); // 392*60000
  B = Bt.t().contiguous(); // 392*60000

  std::cout << "Maximum Value: " << A.max().item<float>() << std::endl;
  std::cout << "Mean Value: " << A.mean().item<float>() << std::endl;
  std::cout << "Minimum Value: " << A.min().item<float>() << std::endl;
  std::cout << "Maximum Value: " << B.max().item<float>() << std::endl;
  std::cout << "Mean Value: " << B.mean().item<float>() << std::endl;
  std::cout << "Minimum Value: " << B.min().item<float>() << std::endl;

  INTELLI_INFO(
      "Generating [" + to_string(A.size(0)) + " x " + to_string(A.size(1)) + "]*[" + to_string(B.size(0)) + " x "
          + to_string(B.size(1)) + "]");
}

//do nothing in abstract class
bool LibAMM::MNISTMatrixLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  paraseConfig(cfg);
  generateAB();
  return true;
}

LibAMM::Tensor LibAMM::MNISTMatrixLoader::getA() {
  return A;
}

LibAMM::Tensor LibAMM::MNISTMatrixLoader::getB() {
  return B;
}

LibAMM::Tensor LibAMM::MNISTMatrixLoader::getAt() {
  return At;
}

LibAMM::Tensor LibAMM::MNISTMatrixLoader::getBt() {
  return Bt;
}

LibAMM::Tensor LibAMM::MNISTMatrixLoader::getSxx() {
  return Sxx;
}

LibAMM::Tensor LibAMM::MNISTMatrixLoader::getSyy() {
  return Syy;
}

LibAMM::Tensor LibAMM::MNISTMatrixLoader::getSxy() {
  return Sxy;
}

LibAMM::Tensor LibAMM::MNISTMatrixLoader::getSxxNegativeHalf() {
  return SxxNegativeHalf;
}

LibAMM::Tensor LibAMM::MNISTMatrixLoader::getSyyNegativeHalf() {
  return SyyNegativeHalf;
}

LibAMM::Tensor LibAMM::MNISTMatrixLoader::getM() {
  return M;
}

LibAMM::Tensor LibAMM::MNISTMatrixLoader::getM1() {
  return M1;
}

LibAMM::Tensor LibAMM::MNISTMatrixLoader::getCorrelation() {
  return correlation;
}

void LibAMM::MNISTMatrixLoader::calculate_correlation() {
  
  // Sxx, Syy, Sxy: covariance matrix
  Sxx = LibAMM::matmul(A, At) / A.size(1); // 392x60000 * 60000x392 max 12752.4 min -3912.51
  Syy = LibAMM::matmul(B, Bt) / A.size(1); // 392x60000 * 60000x392 max 12953.4 min -5121.09
  Sxy = LibAMM::matmul(A, Bt) / A.size(1); // 392x60000 * 60000x392 max 10653.1 min -5307.15

  // Sxx^(-1/2), Syy^(-1/2), M
  // Sxx^(-1/2)
  LibAMM::Tensor eigenvaluesSxx, eigenvectorsSxx;
  std::tie(eigenvaluesSxx, eigenvectorsSxx) = torch::linalg::eig(Sxx); // diagonization
  LibAMM::Tensor diagonalMatrixSxx = LibAMM::diag(
      1.0 / LibAMM::sqrt(eigenvaluesSxx + torch::full({}, 1e-12))); // 1/sqrt(eigenvalue+epsilon) +epsilon to avoid nan
  SxxNegativeHalf = LibAMM::matmul(LibAMM::matmul(eigenvectorsSxx, diagonalMatrixSxx), eigenvectorsSxx.t());
  SxxNegativeHalf = at::real(SxxNegativeHalf); // ignore complex part, it comes from numerical computations
  // Syy^(-1/2)
  LibAMM::Tensor eigenvaluesSyy, eigenvectorsSyy;
  std::tie(eigenvaluesSyy, eigenvectorsSyy) = torch::linalg::eig(Syy);
  LibAMM::Tensor diagonalMatrixSyy = LibAMM::diag(1.0 / LibAMM::sqrt(eigenvaluesSyy + torch::full({}, 1e-12)));
  SyyNegativeHalf = LibAMM::matmul(LibAMM::matmul(eigenvectorsSyy, diagonalMatrixSyy), eigenvectorsSyy.t());
  SyyNegativeHalf = at::real(SyyNegativeHalf);
  // M
  M1 = LibAMM::matmul(SxxNegativeHalf.t(), Sxy);
  M = LibAMM::matmul(M1, SyyNegativeHalf);

  // correlation
  LibAMM::Tensor U, S, Vh;
  std::tie(U, S, Vh) = torch::linalg::svd(M, false, c10::nullopt);
  correlation = LibAMM::clamp(S, -1.0, 1.0);
}
