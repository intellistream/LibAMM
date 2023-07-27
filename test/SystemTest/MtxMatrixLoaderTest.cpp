//
// Created by tony on 27/07/23.
//
#include <vector>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <AMMBench.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
TEST_CASE("Test the mtx loader", "[short]")
{
  AMMBench::MatrixLoaderTable mLoaderTable;
  ConfigMapPtr cfg = newConfigMap();
  auto matLoaderPtr = mLoaderTable.findMatrixLoader("mtx");
  assert(matLoaderPtr);
  matLoaderPtr->setConfig(cfg);
  /*auto A = matLoaderPtr->getA();
  auto B = matLoaderPtr->getB();
 auto C=torch::matmul(A,B);
  INTELLI_INFO("rows="+to_string(C.size(0))+"cols="+to_string(C.size(1)));*/
}
