//
// Created by tony on 10/05/23.
//

#include <MatrixLoader/MatrixLoaderTable.h>
#include <MatrixLoader/RandomMatrixLoader.h>
#include <MatrixLoader/SparseMatrixLoader.h>
#include <MatrixLoader/ExponentialMatrixLoader.h>
#include <MatrixLoader/GaussianMatrixLoader.h>
#include <MatrixLoader/PoissonMatrixLoader.h>
#include <MatrixLoader/BinomialMatrixLoader.h>
#include <MatrixLoader/BetaMatrixLoader.h>
#include <MatrixLoader/SIFTMatrixLoader.h>
#include <MatrixLoader/MNISTMatrixLoader.h>

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
  loaderMap["poisson"] = newPoissonMatrixLoader();
  loaderMap["beta"] = newBetaMatrixLoader();
  loaderMap["SIFT"] = newSIFTMatrixLoader();
  loaderMap["MNIST"] = newMNISTMatrixLoader();
}

} // AMMBench