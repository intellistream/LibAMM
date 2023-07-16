//
// Created by tony on 25/12/22.
//

#include <AMMBench.h>
#include <string>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace AMMBench;
int64_t algoTag=0;
std::string algoTagStr="mm";
/**
 *
 * @brief The c++ bindings to crs.
 * @param a
 * @param b
 * @note please keep input and output as tensors
 * @return tensor
 */
torch::Tensor AMMBench_crs(torch::Tensor a, torch::Tensor b) {
  AMMBench::CRSCPPAlgo algo;
  auto w=a.sizes()[1]/10;
  return algo.amm(a, b, (uint64_t)w);
}
torch::Tensor AMMBench_ammDefault(torch::Tensor a, torch::Tensor b) {
  AMMBench::CPPAlgoTable cppAlgoTable;
  AMMBench::AbstractCPPAlgoPtr cppAlgoPtr = cppAlgoTable.findCppAlgo(algoTagStr);
  auto w=a.sizes()[1]/10;
  return cppAlgoPtr->amm(a, b, (uint64_t)w);
}

torch::Tensor AMMBench_ammSpecifySs(torch::Tensor a, torch::Tensor b,int64_t ss) {
  AMMBench::CPPAlgoTable cppAlgoTable;
  AMMBench::AbstractCPPAlgoPtr cppAlgoPtr = cppAlgoTable.findCppAlgo(algoTagStr);
  return cppAlgoPtr->amm(a, b, (uint64_t)ss);
}
void AMMBench_setTag(std::string tag) {
 algoTagStr=tag;
}
    // END o
/**
 * @brief Declare the function to pytorch
 * @note The of lib is myLib
 */
TORCH_LIBRARY(AMMBench, m2) {
  m2.def("crs",AMMBench_crs );
  m2.def("setTag", AMMBench_setTag );
  m2.def("ammDefault",AMMBench_ammDefault);
  m2.def("ammSpecifySs",AMMBench_ammSpecifySs);
    //m2.def("myVecSub", myVecSub);
}