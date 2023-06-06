//
// Created by tony on 10/05/23.
//

#include <MatrixLoader/MatrixLoaderTable.h>
#include <MatrixLoader/RandomMatrixLoader.h>
#include <MatrixLoader/SparseMatrixLoader.h>
#include <MatrixLoader/ExponentialMatrixLoader.h>
#include <MatrixLoader/GaussianMatrixLoader.h>
#include <MatrixLoader/BinomialMatrixLoader.h>
namespace AMMBench {
/**
 * @note revise me if you need new loader
 */
AMMBench::MatrixLoaderTable::MatrixLoaderTable() {
  loaderMap["random"] = newRandomMatrixLoader();
  loaderMap["sparse"] = newSparseMatrixLoader();
  loaderMap["gaussian"] = newGaussianMatrixLoader();
  loaderMap["exponential"] = newExponentialMatrixLoader();
  loaderMap["binomial"] = newBinomialMatrixLoader();
}

} // AMMBench