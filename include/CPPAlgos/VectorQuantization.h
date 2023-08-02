//
// Created by haolan on 17/7/23.
//

#ifndef INTELLISTREAM_VECTORQUANTIZATION_H
#define INTELLISTREAM_VECTORQUANTIZATION_H
#include <CPPAlgos/AbstractCPPAlgo.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_CppAlgos The algorithms written in c++
 * @{
 */
/**
 * @class VectorQuantization CPPAlgos/VectorQuantization.h
 * @brief The Vector Quantization AMM class of c++ algos
 *
 */
class VectorQuantization : public AMMBench::AbstractCPPAlgo {
  protected:
    string columnCodeIndexXPath;
    string rowCodeIndexYPath;
    string columnCodeBookXvecPath;
    string rowCodeBookYvecPath;

 public:
  VectorQuantization() {

  }

  ~VectorQuantization() {

  }

  /**
   * @brief the virtual function provided for outside callers, rewrite in children classes
   * @param A the A matrix
   * @param B the B matrix
   * @param sketchSize the size of sketc or sampling
   * @return the output c matrix
   */
  virtual torch::Tensor amm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize);

  virtual void setConfig(INTELLI::ConfigMapPtr cfg);

};

/**
 * @ingroup AMMBENCH_CppAlgos
 * @typedef AbstractMatrixCppAlgoPtr
 * @brief The class to describe a shared pointer to @ref VectorQuantizationAlgo

 */
typedef std::shared_ptr<class AMMBench::VectorQuantization> VectorQuantizationPtr;
/**
 * @ingroup AMMBENCH_CppAlgos
 * @def newVectorQuantizationAlgo
 * @brief (Macro) To creat a new @ref  VectorQuantizationAlgounder shared pointer.
 */
#define newVectorQuantizationAlgo std::make_shared<AMMBench::VectorQuantization>

class PQMM {
 public:

  torch::Tensor X, Y; // raw input matrixs  MxN, NxK
  int l;  // size of the sub-codebook
  int m;  // number of sub-codebook

  string columnCodeIndexXPath;
  string rowCodeIndexYPath;
  string columnCodeBookXvecPath;
  string rowCodeBookYvecPath;

  std::vector<torch::Tensor> columnCodeBookX, rowCodeBookY;  // codebooks mxlxM, mxlxK
  vector<vector<int>> columnCodeIndexX, rowCodeIndexY; // code indexs mxl
  torch::Tensor res;

  PQMM(torch::Tensor x, torch::Tensor y, int l, int m) : X(std::move(x)), Y(std::move(y)), l(l), m(m) {}
  ~PQMM() = default;

  
  virtual torch::Tensor matrixOuterProduct(torch::Tensor A, torch::Tensor B);
  
  virtual torch::Tensor runAMM(bool training = false);

  virtual void setFilePath(string columnCodeIndexXPathPassedIn, string rowCodeIndexYPathPassedIn, string columnCodeBookXvecPathPassedIn, string rowCodeBookYvecPathPassedIn);
  virtual void save3DVectorDoubleToFile(string filename, vector<vector<vector<double>>> &vec);
  virtual void save2DVectorIntToFile(string filename, vector<vector<int>> &vec);
  virtual void load3DVectorDoubleFromFile(string filename, vector<vector<vector<double>>> &vec);
  virtual void load2DVectorIntFromFile(string filename, vector<vector<int>> &vec);

  virtual void constructCodeBooks();
};
}
/**
 * @}
 */
#endif //INTELLISTREAM_VECTORQUANTIZATION_H
