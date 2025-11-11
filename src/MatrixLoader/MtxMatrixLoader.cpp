//
// Created by tony on 10/05/23.
//

#include <MatrixLoader/MtxMatrixLoader.h>
#include <Utils/IntelliLog.h>
#include <cmath>
LibAMM::Tensor LibAMM::scaleIntoPN1(LibAMM::Tensor a){
  LibAMM::Tensor min_value = a.min();
  LibAMM::Tensor max_value = a.max();
  LibAMM::Tensor normalized_tensor;
  if (std::abs(min_value.item<float>()) > std::abs(max_value.item<float>())) {
    normalized_tensor=a/min_value;
  }
  else
  {
    normalized_tensor=a/max_value;
  }
  return normalized_tensor;
}
LibAMM::Tensor LibAMM::normalizeIntoPN1(LibAMM::Tensor a){
  LibAMM::Tensor min_value = a.min();
  LibAMM::Tensor max_value = a.max();

  // Normalize the tensor to -1 to 1
  LibAMM::Tensor normalized_tensor = 2 * (a - min_value) / (max_value - min_value) - 1;

  return normalized_tensor;
}
LibAMM::Tensor LibAMM::loadMatrixFromMatrixMarket(const std::string &filename) {
  ifstream file(filename);
  if (!file.is_open()) {
    //cerr << "Error: Unable to open the file " << filename << endl;
    INTELLI_ERROR("Unable to open the file " + filename);
    return LibAMM::Tensor();
  }

  string header;
  size_t rows, cols, nonzeros;

  // Read the MatrixMarket header
  file >> header;
  if (header != "%%MatrixMarket") {
    INTELLI_ERROR("Invalid MatrixMarket format");
    return LibAMM::Tensor();
  }
  string line;
  getline(file, line);


  // Read the matrix dimensions and number of non-zero entries
  file >> rows >> cols >> nonzeros;

  LibAMM::Tensor result = LibAMM::zeros({(int64_t) rows, (int64_t) cols});
  // Read and store the matrix elements as COO format
  for (size_t i = 0; i < nonzeros; ++i) {
    size_t row, col;
    float value;
    file >> row >> col >> value;

    // MatrixMarket format uses 1-based indexing, we convert it to 0-based indexing
    --row;
    --col;

    if (row >= rows || col >= cols) {
      INTELLI_ERROR("Invalid row or column index in the MatrixMarket file");
      return LibAMM::Tensor();
    }
    result[row][col] = value;
  }

  file.close();

  //LibAMM::Tensor matrix = torch::sparse_coo_tensor(indices, values_tensor, {static_cast<int64_t>(rows), static_cast<int64_t>(cols)});

  return result.clone();
}
void LibAMM::MtxMatrixLoader::paraseConfig(INTELLI::ConfigMapPtr cfg) {
  transposeA = cfg->tryU64("transposeA", 0, true);
  transposeB = cfg->tryU64("transposeB", 1, true);
  normalizeA= cfg->tryU64("normalizeA", 0, true);
  normalizeB= cfg->tryU64("normalizeB", 0, true);
  oneSrcForAB = cfg->tryU64("oneSrcForAB", 0, true);
  srcA = cfg->tryString("srcA", "datasets/ZENIOS/zenios.mtx", true);
  if (oneSrcForAB) {
    srcB = srcA;
  } else {
    srcB = cfg->tryString("srcB", "datasets/ZENIOS/zenios.mtx", true);
  }

}

void LibAMM::MtxMatrixLoader::generateAB() {
  A = loadMatrixFromMatrixMarket(srcA);
  if (transposeA) {
    A = A.t().contiguous();
  }
  if(normalizeA)
  {
    A= normalizeIntoPN1(A);
  }else if(scaleA)
  {
    A= scaleIntoPN1(A);
  }

  B = loadMatrixFromMatrixMarket(srcB);
  if (transposeB) {
    B = B.t().contiguous();
  }
  if(normalizeB)
  {
    B= normalizeIntoPN1(B);
  }else if(scaleB)
  {
    B= scaleIntoPN1(B);
  }

  INTELLI_INFO(
      "Generating [" + to_string(A.size(0)) + " x " + to_string(A.size(1)) + "]*[" + to_string(B.size(0)) + " x "
          + to_string(B.size(1)) + "]");

}

//do nothing in abstract class
bool LibAMM::MtxMatrixLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  paraseConfig(cfg);
  generateAB();
  return true;
}

LibAMM::Tensor LibAMM::MtxMatrixLoader::getA() {
  return A.clone();
}

LibAMM::Tensor LibAMM::MtxMatrixLoader::getB() {
  return B.clone();
}
