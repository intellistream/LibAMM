/*! \file FastJLTCPPAlgo.h*/
//
// Created by luv on 6/18/23.
//

#ifndef INTELLISTREAM_FASTFLTCPPALGO_H
#define INTELLISTREAM_FASTJLTCPPALGO_H

#include <CPPAlgos/AbstractCPPAlgo.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_CppAlgos The algorithms writtrn in c++
 * @{
 */
/**
 * @class FastJLTCPPAlgo CPPAlgos/FastJLTCPPAlgo.h
 * @brief The tug of war class of c++ algoS
 */
class FastJLTCPPAlgo : public AMMBench::AbstractCPPAlgo {
 public:
  FastJLTCPPAlgo() {

  }

  ~FastJLTCPPAlgo() {

  }

  /**
   * @brief the virtual function provided for outside callers, rewrite in children classes
   * @param A the A matrix
   * @param B the B matrix
   * @param sketchSize the size of sketc or sampling
   * @return the output c matrix
   */
  virtual torch::Tensor amm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize);

 private:
};

/**
 * @ingroup AMMBENCH_CppAlgos
 * @typedef AbstractMatrixCppAlgoPtr
 * @brief The class to describe a shared pointer to @ref FastJLTCppAlgo

 */
typedef std::shared_ptr<class AMMBench::FastJLTCPPAlgo> FastJLTCPPAlgoPtr;
/**
 * @ingroup AMMBENCH_CppAlgos
 * @def newFastJLTCppAlgo
 * @brief (Macro) To creat a new @ref  FastJLTCppAlgounder shared pointer.
 */
#define newFastJLTCPPAlgo std::make_shared<AMMBench::FastJLTCPPAlgo>
}
/**
 * @}
 */
#endif //INTELLISTREAM_FASTJLTCPPALGO_H
