
// Created by tony on 25/05/23.
//

#include <CPPAlgos/INT8CPPAlgo.h>
torch::Tensor AMMBench::INT8CPPAlgo::amm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize) {
  std::cout << sketchSize;
  float scalingA,scalingB;
 // int16_t mulab;

  /*auto AINT8=(A*scalingA).to(torch::kInt8);
  auto BTINT8=(Bt*scalingB).to(torch::kInt8);*/
 // return C;
  /*
  for(int64_t i=0;i<rows;i++)
  {
    for(int64_t j=0;j<cols;j++)
    {
       ta=AINT8[i];
       tb=BTINT8[j];

      int32_t ru=0;
      float ruf=0.0;

      for(int64_t k=0;k<sumS;k++)
      {

       int8_t tak=ta[k].item<int8_t>();
        int8_t tbk=tb[k].item<int8_t>();
         mulab=tak*tbk;
        ru+=mulab;
      }
      ruf=ru;
      C[i][j]=ruf;
    }
  }*/
  scalingA=torch::abs(A).max().item<float>()/127.0;
  scalingB=torch::abs(B).max().item<float>()/127.0;
  auto ta=(A/scalingA).to(torch::kInt8);
  auto tb= (B/scalingB).to(torch::kInt8);
 // torch::matmul(ta, tb);
  return torch::zeros({A.size(0), B.size(1)});
  return torch::matmul(ta, tb).to(torch::kFloat)*scalingA*scalingB;
 //return C*scalingA*scalingB/127.0/127.0;
}