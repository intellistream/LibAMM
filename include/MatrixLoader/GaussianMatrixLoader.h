//
// Created by haolan on 6/5/23.
//

#ifndef INTELLISTREAM_GAUSSIANMATRIXLOADER_H
#define INTELLISTREAM_GAUSSIANMATRIXLOADER_H

#include <MatrixLoader/AbstractMatrixLoader.h>

namespace AMMBench {
/**
 * @ingroup AMMBENCH_MatrixLOADER
 * @{
 */
/**
 * @ingroup AMMBENCH_MatrixLOADER_Gaussian The Gaussian Random generator
 * @{
 */
/**
 * @class GaussianMatrixLoader MatrixLoader/GaussianMatrixLoader.h
 * @brief The Gaussian Random class of matrix loader
 * @ingroup AMMBENCH_MatrixLOADER_Gaussian
 * @note:
 * - Must have a global config by @ref setConfig
 * @note  Default behavior
* - create
* - call @ref setConfig, this function will also generate the tensor A and B correspondingly
* - call @ref getA and @ref getB (assuming we are benchmarking torch.mm(A,B))
 * @note: require config parameters and default values
 * - "aRow" The rows in matrix A, U64, 100
 * - "aCol" The cols in matrix B, U64, 1000
 * - "bCol" The rows in matrix B, U64, 500
 * - "seed" The seed of inline random generator,U64,114514
 * @note: default name tags
 * "random": @ref GaussianMatrixLoader
 */
class GaussianMatrixLoader : public AbstractMatrixLoader {
 protected:
  torch::Tensor A, B;
  uint64_t aRow, aCol, bCol, seed;

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
  GaussianMatrixLoader() = default;

  ~GaussianMatrixLoader() = default;

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
 * @ingroup AMMBENCH_MatrixLOADER_Gaussian
 * @typedef GaussianMatrixLoaderPtr
 * @brief The class to describe a shared pointer to @ref GaussianMatrixLoader

 */
typedef std::shared_ptr<class AMMBench::GaussianMatrixLoader> GaussianMatrixLoaderPtr;
/**
 * @ingroup AMMBENCH_MatrixLOADER_Gaussian
 * @def newGaussianMatrixLoader
 * @brief (Macro) To creat a new @ref GaussianMatrixLoader under shared pointer.
 */
#define newGaussianMatrixLoader std::make_shared<AMMBench::GaussianMatrixLoader>
/**
 * @}
 */
/**
 * @}
 */
}
#endif //INTELLISTREAM_GAUSSIANMATRIXLOADER_H
