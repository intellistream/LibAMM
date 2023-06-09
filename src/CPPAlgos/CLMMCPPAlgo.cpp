
// Created by tony on 25/05/23.
//

#include <CPPAlgos/CLMMCPPAlgo.h>
#include <Utils/UtilityFunctions.h>
void clTest(TONY_CL_HOST::CLContainerPtr clc,float *h_a,float *h_b,float  *h_c,int64_t *params,uint64_t local0,uint64_t local1)
{

  clc->addHostOutPara(HostPara(h_a,params[0]*params[1]*sizeof(float)));
  clc->addHostOutPara(HostPara(h_b,params[1]*params[2]*sizeof(float)));
  clc->addHostOutPara(HostPara(params,3*sizeof(int64_t)));
  clc->addHostInPara(HostPara(h_c,params[0]*params[2]*sizeof(float)));
  std::vector<size_t> localSz(2);
  std::vector<size_t> globalSz(2);
  localSz[0]=local0;
  localSz[1]=local1;
  globalSz[0]=params[0];
  globalSz[1]=params[2];

  clc->setWorkDimension(2);
  clc->execute(globalSz,localSz);

  return ;
}
torch::Tensor AMMBench::CLMMCPPAlgo::clmm(torch::Tensor tensor1, torch::Tensor tensor2)
{
  auto A_size = tensor1.sizes();
  auto B_size =tensor2.sizes();
  struct timeval tstart;
  //INTELLI_INFO("I am mm");
  gettimeofday(&tstart,NULL);
  int64_t rows1 = A_size[0];
  int64_t cols1 = A_size[1];
  int64_t cols2 = B_size[1];
  /**
   * @brief build A into std vector
   */
  std::vector<float> matrix1(tensor1.data_ptr<float>(), tensor1.data_ptr<float>() + rows1 * cols1);
  buildATime=INTELLI::UtilityFunctions::timeLastUs(tstart);
  std::vector<float> matrix2(tensor2.data_ptr<float>(), tensor2.data_ptr<float>() + cols1 * cols2);
  buildBTime=INTELLI::UtilityFunctions::timeLastUs(tstart)-buildATime;
  // Create the output matrix
 std::vector<float> result(rows1 * cols2, 0.0);
 /* for (int64_t i = 0; i < rows1; ++i) {
    for (int64_t j = 0; j < cols2; ++j) {
      for (int64_t k = 0; k < cols1; ++k) {
        result[i * cols2 + j] += matrix1[i * cols1 + k] * matrix2[k * cols2 + j];
      }
    }
  }*/
 int64_t params[3];
 params[0]=rows1;
 params[1]=cols1;
 params[2]=cols2;
/* float *m1=(float*) malloc((rows1*cols1)*sizeof(float));
  float *m2=(float*) malloc((cols1*cols2)*sizeof(float));
  float *m3=(float *) malloc((rows1*cols2)*sizeof(float));*/



  /*
  clc->addHostOutPara(HostPara(m1,(rows1*cols1)*sizeof(float)));
  clc->addHostOutPara(HostPara(m2,(cols1*cols2)*sizeof(float)));
  clc->addHostOutPara(HostPara(params,3*sizeof(int64_t)));
  clc->addHostInPara(HostPara(m3,(rows1*cols2)*sizeof(float)));
*/
  clTest(clc,matrix1.data(),matrix2.data(),result.data(),params,localSize0,localSize1);
  // exit(-1);


  torch::Tensor resultTensor = torch::from_blob(result.data(), {rows1,cols2});
  //torch::Tensor resultTensor = torch::zeros({rows1,cols2});
  postProcessTime=INTELLI::UtilityFunctions::timeLastUs(tstart)-buildATime-buildBTime-fABTime;
  /**
   * @brief fix the time measure related to cl
   */
  fABTime = INTELLI::UtilityFunctions::timeLastUs(tstart)-buildATime-buildBTime-clc->tRun-clc->tOut;
  buildATime+=clc->tIn/2;
  buildBTime+=clc->tIn/2;
  postProcessTime+=clc->tOut;
  return resultTensor.clone();
}

torch::Tensor AMMBench::CLMMCPPAlgo::clmm(torch::Tensor tensor1, torch::Tensor tensor2)
{
  auto A_size = tensor1.sizes();
  auto B_size =tensor2.sizes();
  struct timeval tstart;
  //INTELLI_INFO("I am mm");
  gettimeofday(&tstart,NULL);
  int64_t rows1 = A_size[0];
  int64_t cols1 = A_size[1];
  int64_t cols2 = B_size[1];
  /**
   * @brief build A into std vector
   */
  std::vector<float> matrix1(tensor1.data_ptr<float>(), tensor1.data_ptr<float>() + rows1 * cols1);
  buildATime=INTELLI::UtilityFunctions::timeLastUs(tstart);
  std::vector<float> matrix2(tensor2.data_ptr<float>(), tensor2.data_ptr<float>() + cols1 * cols2);
  buildBTime=INTELLI::UtilityFunctions::timeLastUs(tstart)-buildATime;
  // Create the output matrix
  std::vector<float> result(rows1 * cols2, 0.0);
  /* for (int64_t i = 0; i < rows1; ++i) {
     for (int64_t j = 0; j < cols2; ++j) {
       for (int64_t k = 0; k < cols1; ++k) {
         result[i * cols2 + j] += matrix1[i * cols1 + k] * matrix2[k * cols2 + j];
       }
     }
   }*/
  int64_t params[3];
  params[0]=rows1;
  params[1]=cols1;
  params[2]=cols2;
/* float *m1=(float*) malloc((rows1*cols1)*sizeof(float));
  float *m2=(float*) malloc((cols1*cols2)*sizeof(float));
  float *m3=(float *) malloc((rows1*cols2)*sizeof(float));*/



  /*
  clc->addHostOutPara(HostPara(m1,(rows1*cols1)*sizeof(float)));
  clc->addHostOutPara(HostPara(m2,(cols1*cols2)*sizeof(float)));
  clc->addHostOutPara(HostPara(params,3*sizeof(int64_t)));
  clc->addHostInPara(HostPara(m3,(rows1*cols2)*sizeof(float)));
*/
  clTest(clc,matrix1.data(),matrix2.data(),result.data(),params,localSize0,localSize1);
  // exit(-1);


  torch::Tensor resultTensor = torch::from_blob(result.data(), {rows1,cols2});
  //torch::Tensor resultTensor = torch::zeros({rows1,cols2});
  postProcessTime=INTELLI::UtilityFunctions::timeLastUs(tstart)-buildATime-buildBTime-fABTime;
  /**
   * @brief fix the time measure related to cl
   */
  fABTime = INTELLI::UtilityFunctions::timeLastUs(tstart)-buildATime-buildBTime-clc->tRun-clc->tOut;
  buildATime+=clc->tIn/2;
  buildBTime+=clc->tIn/2;
  postProcessTime+=clc->tOut;
  return resultTensor.clone();
}
void AMMBench::CLMMCPPAlgo::setConfig(INTELLI::ConfigMapPtr cfg) {
  AbstractCPPAlgo::setConfig(cfg);
  clFile=cfg->tryString("clFile","CL/CLMM.cl",true);
  clc=newCLContainer(1,CL_DEVICE_TYPE_DEFAULT,"CLMM",clFile);
  localSize0=cfg->tryU64("localSize0",1,true);
  localSize1=cfg->tryU64("localSize1",1,true);
}
torch::Tensor AMMBench::CLMMCPPAlgo::amm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize) {
  assert(sketchSize);
  return clmm(A,B);
}