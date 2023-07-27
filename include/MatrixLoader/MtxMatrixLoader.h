/*! \file MtxMatrixLoader.h*/
//
// Created by tony on 10/05/23.
//

#ifndef INTELLISTREAM_INCLUDE_MATRIXLOADER_MTXMATRIXLOADER_H_
#define INTELLISTREAM_INCLUDE_MATRIXLOADER_MTXMATRIXLOADER_H_

#include <MatrixLoader/AbstractMatrixLoader.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
namespace AMMBench {

/**
 * @ingroup AMMBENCH_MatrixLOADER
 * @{
 */
/**
 * @ingroup AMMBENCH_MatrixLOADER_Mtx The loader of matrix market mtx matrixes
 * @{
 */
/**
 * @brief the stan-alone function to load a matrix from matrix market mitx file
 * @param filename the name of the mtx file
 * @return the loaded tensor
 */
torch::Tensor loadMatrixFromMatrixMarket(const string &filename);
/**
 * @class MtxMatrixLoader MatrixLoader/MtxMatrixLoader.h
 * @brief The matrix loader to load matrixes stored in matrix market mtx format
 * @ingroup AMMBENCH_MatrixLOADER
 * @note:
 * - Must have a global config by @ref setConfig
 * @note  Default behavior
* - create
* - call @ref setConfig, this function will also generate the tensor A and B correspondingly
* - call @ref getA and @ref getB (assuming we are benchmarking torch.mm(A,B))
 * @note: require config parameters and default values
 * - "srcA" The file source for A matrix, String, "datasets/ZENIOS/zenios.mtx"
 * - "oneSrcForAB", U64, whether A and B shares the same source file
 * - "srcB" The file source for B matrix, String, "datasets/ZENIOS/zenios.mtx"
 * - "transposeA" Whether or not transpose A matrix, U64, 0
 * -  "transposeB" Whether or not transpose B matrix, U64, 1
 * @note: default name tags
 * "mtx": @ref MtxMatrixLoader
 */
class MtxMatrixLoader : public AbstractMatrixLoader {
 protected:
  torch::Tensor A, B;
  std::string srcA, srcB;
  uint64_t oneSrcForAB, transposeA, transposeB;
  /**
   * @brief Inline logic of reading a config file
   * @param cfg the config
   */
  void paraseConfig(INTELLI::ConfigMapPtr cfg);

  /**
   * @brief inline logic of generating A and B
   */
  void generateAB();

 public:
  MtxMatrixLoader() = default;

  ~MtxMatrixLoader() = default;

  /**
     * @brief Set the GLOBAL config map related to this loader
     * @param cfg The config map
      * @return bool whether the config is successfully set
      * @note
     */
  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  /**
   * @brief get the A matrix
   * @return the generated A matrix
   */
  virtual torch::Tensor getA();

  /**
  * @brief get the B matrix
  * @return the generated B matrix
  */
  virtual torch::Tensor getB();
};

/**
 * @ingroup AMMBENCH_MatrixLOADER_Random
 * @typedef MtxMatrixLoaderPtr
 * @brief The class to describe a shared pointer to @ref MtxMatrixLoader

 */
typedef std::shared_ptr<class AMMBench::MtxMatrixLoader> MtxMatrixLoaderPtr;
/**
 * @ingroup AMMBENCH_MatrixLOADER_Random
 * @def newMtxMatrixLoader
 * @brief (Macro) To creat a new @ref MtxMatrixLoader under shared pointer.
 */
#define newMtxMatrixLoader std::make_shared<AMMBench::MtxMatrixLoader>
/**
 * @}
 */
/**
 * @}
 */
}
#endif //INTELLISTREAM_INCLUDE_MATRIXLOADER_RANDOMMATRIXLOADER_H_
