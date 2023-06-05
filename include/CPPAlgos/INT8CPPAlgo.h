/*! \file INT8CPPAlgo.h*/
//
// Created by tony on 25/05/23.
//

#ifndef INTELLISTREAM_INCLUDE_CPPALGOS_INT8CPPALGO_H_
#define INTELLISTREAM_INCLUDE_CPPALGOS_INT8CPPALGO_H_
#include <Utils/AbstractC20Thread.hpp>
#include <Utils/ConfigMap.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <vector>
#include <CPPAlgos/AbstractCPPAlgo.h>
namespace AMMBench {
/**
 * @ingroup AMMBENCH_CppAlgos The algorithms written in c++
 * @{
 */
/**
 * @class INT8CPPAlgo CPPAlgos/INT8CPPAlgo.h
 * @brief The INT8 MM class of c++ algos
 */
class INT8CPPAlgo : public AMMBench::AbstractCPPAlgo {
 public:
  INT8CPPAlgo() {

  }
  ~INT8CPPAlgo() {

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
 * @ingroup AMMBENCH_CppAlgos
 * @typedef INT8MatrixCppAlgoPtr
 * @brief The class to describe a shared pointer to @ref INT8CPPAlgo

 */
typedef std::shared_ptr<class AMMBench::INT8CPPAlgo> INT8CPPAlgoPtr;
/**
 * @ingroup AMMBENCH_CppAlgos
 * @def newINT8CppAlgo
 * @brief (Macro) To creat a new @ref  INT8CppAlgo shared pointer.
 */
#define newINT8CPPAlgo std::make_shared<AMMBench::INT8CPPAlgo>
}
/**
 * @}
 */

#endif //INTELLISTREAM_INCLUDE_CPPALGOS_ABSTRACTCPPALGO_H_
