//
// Created by tony on 27/07/23.
//
#include <vector>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <LibAMM.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
TEST_CASE("Test the zero masked loader", "[short]")
{
  LibAMM::MatrixLoaderTable mLoaderTable;
  ConfigMapPtr cfg = newConfigMap();
  cfg->edit("aRow", (uint64_t)10);
  cfg->edit("aCol", (uint64_t)10);
  cfg->edit("bCol", (uint64_t)10);
  cfg->edit("nnzA", (double)0.2);
  cfg->edit("nnzB", (double)0.1);
  auto matLoaderPtr = mLoaderTable.findMatrixLoader("zeroMasked");
  assert(matLoaderPtr);
  matLoaderPtr->setConfig(cfg);
  auto A = matLoaderPtr->getA();
  auto B = matLoaderPtr->getB();
  auto C = torch::matmul(A, B);
  std::cout<<"mat A"<<std::endl;
  std::cout << A << std::endl;
  std::cout<<"mat B"<<std::endl;
  std::cout << B << std::endl;
  std::cout<<"mat C"<<std::endl;
  //INTELLI_INFO("rows="+to_string(C.size(0))+"cols="+to_string(C.size(1)));
  std::cout << C << std::endl;
}
