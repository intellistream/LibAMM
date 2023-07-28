//
// Created by haolan on 6/5/23.
//
#include <MatrixLoader/SIFTMatrixLoader.h>
#include <Utils/IntelliLog.h>
#include <AMMBench.h>

void AMMBench::SIFTMatrixLoader::paraseConfig(INTELLI::ConfigMapPtr cfg) {
  SIFTSize = cfg->tryString("SIFTSize", "10K", true);
}

void AMMBench::SIFTMatrixLoader::generateAB() {

  // Step1. locate file
  std::string filename;
  if (SIFTSize == "10K"){
     filename = "../../../../../../datasets/SIFT/siftsmall_base.fvecs"; //benchmark execute path e.g. /home/heyuhao/AMMBench/build/benchmark/scripts/PCA/results/scansketchDimension_datasetSIFT/crs/100, dataset file path e.g. /home/heyuhao/AMMBench/build/benchmark/datasets/siftsmall_base.fvecs
  }
  else if (SIFTSize == "1M"){
     filename = "../../../../../../datasets/SIFT/sift_base.fvecs"; //benchmark execute path e.g. /home/heyuhao/AMMBench/build/benchmark/scripts/PCA/results/scansketchDimension_datasetSIFT/crs/100, dataset file path e.g. /home/heyuhao/AMMBench/build/benchmark/datasets/siftsmall_base.fvecs
  }
  else{
    INTELLI_ERROR("You can typing in a wrong SIFTSize: " + SIFTSize + ", only 10K or 1M are allowed");
    exit(-1);
  }
 
  float *data = NULL;
  unsigned num, dim;

  // Step2. read in binary
  std::ifstream in(filename, std::ios::binary);    //以二进制的方式打开文件
  if (!in.is_open()) {
    INTELLI_ERROR(
        "Double check your executed path, the dataset file should be relative to your exe path like this: " + filename
            + " e.g. benchmark execute path: /home/user/AMMBench/build/benchmark/scripts/PCA/results/scansketchDimension_datasetSIFT/crs/100; dataset file path /home/user/AMMBench/build/benchmark/datasets/siftsmall_base.fvecs");
    exit(-1);
  }
  in.read((char *) &dim, 4);    //读取向量维度
  in.seekg(0, std::ios::end);    //光标定位到文件末尾
  std::ios::pos_type ss = in.tellg();    //获取文件大小（多少字节）
  size_t fsize = (size_t) ss;
  num = (unsigned) (fsize / (dim + 1) / 4);    //数据的个数
  data = new float[(size_t) num * (size_t) dim];

  in.seekg(0, std::ios::beg);    //光标定位到起始处
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);    //光标向右移动4个字节
    in.read((char *) (data + i * dim), dim * 4);    //读取数据到一维数据data中
  }
  in.close();

  // Step3. convert to torch tensor and standardize the matrix
  torch::TensorOptions options(torch::kFloat32);
  B = torch::from_blob(data, {(int) num, (int) dim}, options).clone();

  // 3.1 Compute the mean and standard deviation along each feature (column)
  torch::Tensor mean = B.mean(/*dim=*/0);
  torch::Tensor std = B.std(/*dim=*/0);

  // 3.2 Standardize the matrix
  torch::Tensor standardizedB = (B - mean) / std;

  // 3.3 Check if need to resize
  B = standardizedB;

  A = B.t();

  int ACol = A.size(0);
  int ARow = A.size(1);
  int BCol = B.size(0);
  int BRow = B.size(1);
  INTELLI_INFO(
      "Generating [" + to_string(ACol) + "x" + to_string(ARow) + "]*[" + to_string(BCol) + "x" + to_string(BRow) + "]");

  delete[] data; // deallocate
}

//do nothing in abstract class
bool AMMBench::SIFTMatrixLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  paraseConfig(cfg);
  generateAB();
  return true;
}

torch::Tensor AMMBench::SIFTMatrixLoader::getA() {
  return A;
}

torch::Tensor AMMBench::SIFTMatrixLoader::getB() {
  return B;
}

// int main() {
//     AMMBench::MatrixLoaderTable mLoaderTable;
//     auto matLoaderPtr = mLoaderTable.findMatrixLoader("SIFT");
//     assert(matLoaderPtr);

//     INTELLI::ConfigMapPtr cfg = newConfigMap();
//     matLoaderPtr->setConfig(cfg);

//     auto A = matLoaderPtr->getA();
//     auto B = matLoaderPtr->getB();
//     std::cout << A.sizes() << endl;
//     std::cout << B.sizes() << endl;
// }