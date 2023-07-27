//
// Created by tony on 10/05/23.
//

#include <MatrixLoader/MtxMatrixLoader.h>
#include <Utils/IntelliLog.h>

torch::Tensor AMMBench::loadMatrixFromMatrixMarket(const std::string &filename) {
  ifstream file(filename);
  if (!file.is_open()) {
    //cerr << "Error: Unable to open the file " << filename << endl;
    INTELLI_ERROR("Unable to open the file " + filename);
    return torch::Tensor();
  }

  string header;
  size_t rows, cols, nonzeros;

  // Read the MatrixMarket header
  file >> header;
  if (header != "%%MatrixMarket") {
    INTELLI_ERROR("Invalid MatrixMarket format");
    return torch::Tensor();
  }
  string line;
  getline(file, line);


  // Read the matrix dimensions and number of non-zero entries
  file >> rows >> cols >> nonzeros;

  torch::Tensor result = torch::zeros({(int64_t) rows, (int64_t) cols});
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
      return torch::Tensor();
    }
    result[row][col] = value;
  }

  file.close();

  //torch::Tensor matrix = torch::sparse_coo_tensor(indices, values_tensor, {static_cast<int64_t>(rows), static_cast<int64_t>(cols)});

  return result.clone();
}
void AMMBench::MtxMatrixLoader::paraseConfig(INTELLI::ConfigMapPtr cfg) {
  transposeA = cfg->tryU64("transposeA", 0, true);
  transposeB = cfg->tryU64("transposeB", 1, true);
  oneSrcForAB = cfg->tryU64("oneSrcForAB", 0, true);
  srcA = cfg->tryString("srcA", "datasets/ZENIOS/zenios.mtx", true);
  if (oneSrcForAB) {
    srcB = srcA;
  } else {
    srcB = cfg->tryString("srcB", "datasets/ZENIOS/zenios.mtx", true);
  }

}

void AMMBench::MtxMatrixLoader::generateAB() {
  A = loadMatrixFromMatrixMarket(srcA);
  if (transposeA) {
    A = A.t();
  }
  B = loadMatrixFromMatrixMarket(srcB);
  if (transposeB) {
    B = B.t();
  }

}

//do nothing in abstract class
bool AMMBench::MtxMatrixLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  paraseConfig(cfg);
  generateAB();
  return true;
}

torch::Tensor AMMBench::MtxMatrixLoader::getA() {
  return A.clone();
}

torch::Tensor AMMBench::MtxMatrixLoader::getB() {
  return B.clone();
}
