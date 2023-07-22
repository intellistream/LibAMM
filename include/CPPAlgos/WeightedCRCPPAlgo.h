/*! \file WeightedCRCPPAlgo.h*/
//
// Created by yuhao on 27/05/23.
//

#ifndef INTELLISTREAM_INCLUDE_CPPALGOS_WeightedCRCPPAlgo_H_
#define INTELLISTREAM_INCLUDE_CPPALGOS_WeightedCRCPPAlgo_H_

#include <CPPAlgos/AbstractCPPAlgo.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_CPPAlgos The algorithms writtrn in c++
 * @{
 */
/**
 * @class WeightedCRCPPlgo CPPAlgos/WeightedCRCPPAlgo.h
 * @brief The weighted cloumn row sampling class of c++ algos
 *
 */
class WeightedCRCPPAlgo : public AMMBench::AbstractCPPAlgo {
 public:
  WeightedCRCPPAlgo() {

  }

  ~WeightedCRCPPAlgo() {

  }

  /**
   * @brief the virtual function provided for outside callers, rewrite in children classes
   * @param A the A matrix
   * @param B the B matrix
   * @param sketchSize the size of sketc or sampling
   * @return the output c matrix
   */
  virtual torch::Tensor amm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize);

};

/**
 * @ingroup AMMBENCH_CPPAlgos
 * @typedef AbstractMatrixCPPAlgoPtr
 * @brief The class to describe a shared pointer to @ref WeightedCRCPPAlgo

 */
typedef std::shared_ptr<class AMMBench::WeightedCRCPPAlgo> WeightedCRCPPAlgoPtr;
/**
 * @ingroup AMMBENCH_CPPAlgos
 * @def newWeightedCRCPPAlgo
 * @brief (Macro) To creat a new @ref  WeightedCRCPPAlgounder shared pointer.
 */
#define newWeightedCRCPPAlgo std::make_shared<AMMBench::WeightedCRCPPAlgo>
}
/**
 * @}
 */

#endif //INTELLISTREAM_INCLUDE_CPPALGOS_WeightedCRCPPAlgo_H_

