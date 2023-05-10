//
// Created by tony on 10/05/23.
//

#include <MatrixLoader/MatrixLoaderTable.h>
#include <MatrixLoader/RandomMatrixLoader.h>
namespace AMMBench {
/**
 * @note revise me if you need new loader
 */
AMMBench::MatrixLoaderTable::MatrixLoaderTable() {
  loaderMap["random"] = newRandomMatrixLoader();
}


} // AMMBench