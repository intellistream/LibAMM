/*! \file BlockPartitionRunner.h*/
//
// Created by tony on 24/05/23.
//

#ifndef INTELLISTREAM_INCLUDE_PARALLELIZATION_BLOCKPARTITIONRUNNER_H_
#define INTELLISTREAM_INCLUDE_PARALLELIZATION_BLOCKPARTITIONRUNNER_H_
#include <Utils/AbstractC20Thread.hpp>
#include <Utils/ConfigMap.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <vector>
namespace AMMBench {

#define  newTensor make_shared<torch::Tensor>
typedef std::shared_ptr<torch::Tensor> TensorPtr;
/**
 * @ingroup AMMBENCH_PARALLELIZATION
 * @{
 * @defgroup PARTITION_RUNNER The partition-based parallelization
 */
 /**
  * @class BlockPartitionWorker Parallelization/BlockPartitionRunner.h
  * @ingroup PARTITION_RUNNER
  * @brief The basic partition worker
  */
 class BlockPartitionWorker:public  INTELLI::AbstractC20Thread{
 protected:
   virtual void inlineMain();
   struct timeval tstart, tend;
  /**
   * @brief Input matrix A
   */
  TensorPtr matA= nullptr;  // Input matrix A
  /**
    * @brief Input matrix B
    */
  TensorPtr matB=nullptr;  // Input matrix B
  /**
  * @brief OUTput matrix C
  */
  TensorPtr matC=nullptr;  // Output matrix C

  INTELLI::ConfigMapPtr cfg;
  torch::jit::script::Module module;
  uint64_t sketchDimension=0;
  int coreBind;
 public:
   torch::Tensor irC,subA;
   uint64_t startRow=0;  // Start row index for the assigned range
   uint64_t endRow=0;  // End row index (exclusive) for the assigned range

  BlockPartitionWorker(){

  }
   /**
   * @brief set the config map
   * @param _cfg
   */
  void setConfig(INTELLI::ConfigMapPtr _cfg);
  /**
   * @brief set the pointer to A,B matrix
   */
  void setAB(TensorPtr A,TensorPtr B);
  void setWorkParameters(uint64_t aStart,uint64_t aEnd,int mycore);
  ~BlockPartitionWorker()
  {

  }
  uint64_t getElapsedTime();

};
/**
* @ingroup PARTITION_RUNNER
* @def newBlockPartitionWorker
* @brief (Macro) To creat a new @ref BlockPartitionWorker under shared pointer.
*/
#define  newBlockPartitionWorker std::make_shared<AMMBench::BlockPartitionWorker>
typedef std::shared_ptr<AMMBench::BlockPartitionWorker> BlockPartitionWorkerPtr;
/**
 * @class BlockPartitionRunner Parallelization/BlockPartitionRunner.h
 * @ingroup PARTITION_RUNNER
 * @brief The top entity to control all workers, see also @ref BlockPartitionWorker
 * @note parameters
 * - threads, U64, the number of worker threads, default 2
 *
 */
class BlockPartitionRunner {

 protected:
  INTELLI::ConfigMapPtr cfg;
  uint64_t threads=0;
  /**
   * @brief Input matrix A
   */
  TensorPtr matA=nullptr;  // Input matrix A
  /**
    * @brief Input matrix B
    */
  TensorPtr matB=nullptr;  // Input matrix B
  /**
  * @brief OUTput matrix C
  */
  TensorPtr matC=nullptr;  // Output matrix C
  std::vector<BlockPartitionWorkerPtr> workers;
 public:BlockPartitionRunner(){}
 ~BlockPartitionRunner(){}
 /**
  * @brief set the config map
  * @param _cfg
  */
 void setConfig(INTELLI::ConfigMapPtr _cfg);
 /**
  * @brief create the A,B,C matrix and pass it to all workers
  *  @param A The A matrix
   * @param B The B matrix
   * @warnning call after @ref setConfig
  */
 void createABC(torch::Tensor A,torch::Tensor B);
  /**
   * @brief conducte the multithread AMM and return
   * @param A The A matrix
   * @param B The B matrix
   * @return The AMM(A,B)
   * @warnning call after @ref setConfig
   */
   /**
    * @brief run a parallel forward of A,B, and return C
    * @return C=matA*matB
    *  @warnning call after @ref createABC
    */
   torch::Tensor parallelForward();
  torch::Tensor runAMM(torch::Tensor A,torch::Tensor B);
  uint64_t getElapsedTime();
};

} // AMMBench
/**
 * @}
 */
#endif //INTELLISTREAM_INCLUDE_PARALLELIZATION_BLOCKPARTITIONRUNNER_H_
