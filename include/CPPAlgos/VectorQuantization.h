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
    string pqvqCodewordLookUpTablePath;
    torch::Tensor codewordsA;
    torch::Tensor codewordsB;
    torch::Tensor lookUpTable;

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
}
/**
 * @}
 */
#endif //INTELLISTREAM_VECTORQUANTIZATION_H
