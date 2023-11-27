//
// Created by tony on 27/11/23.
//
#include <AMMBench.h>
#include <Utils/UtilityFunctions.h>
#include <include/papi_config.h>
#include <Utils/ThreadPerf.hpp>
#if AMMBENCH_PAPI == 1
#include <Utils/ThreadPerfPAPI.hpp>
#endif
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace DIVERSE_METER;

// Function to generate a 2D Gaussian kernel
torch::Tensor gaussian_kernel(int size, double sigma) {
    int center = size / 2;
    torch::Tensor kernel = torch::empty({size, size});

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int x = i - center;
            int y = j - center;
            kernel[i][j] = std::exp(-(x * x + y * y) / (2.0 * sigma * sigma));
        }
    }

    // Normalize the kernel
    kernel /= kernel.sum();

    return kernel;
}
float qcdMSE(const torch::Tensor& A, const torch::Tensor& B) {
    // Compute the square difference
    auto A2=torch::pow(A, 2);
   
    auto B2=torch::pow(B, 2);
   
    torch::Tensor square_difference = (A2-B2)/A2;
    // Compute the mean
    torch::Tensor mean_square_difference = torch::mean(square_difference);

    // Return as a float
    return std::abs(mean_square_difference.item<float>());
}
class QCDPairs
{
private:

    /* data */
public:
    torch::Tensor qcdA,qcdB,conv_kernel;
    int64_t padded_rows,padded_cols;
    uint64_t prepareTime;
    QCDPairs(/* args */){}
    ~QCDPairs(){}
    void genPairs(torch::Tensor qcdMat,int size,double sigma);
    torch::Tensor getResult(torch::Tensor mmRu);

};
void QCDPairs::genPairs(torch::Tensor qcd_matrix,int kernel_size,double sigma)
{  struct timeval ts;
  gettimeofday(&ts, NULL);
    conv_kernel = gaussian_kernel(kernel_size, sigma);
    // Padding to ensure the output has the same size as the input
    int padding = 1;

    // Add padding to the input matrix
    auto padded_matrix = torch::nn::functional::pad(qcd_matrix, torch::nn::functional::PadFuncOptions({padding, padding, padding, padding}));

    // Get the size of the padded matrix
     padded_rows = padded_matrix.size(0);
     padded_cols = padded_matrix.size(1);

    // Reshape the padded matrix into a column matrix
    auto reshaped_matrix = padded_matrix.unfold(0, conv_kernel.size(0), 1).unfold(1, conv_kernel.size(1), 1);

    // Flatten the unfolded matrix to prepare for matrix multiplication
    auto flattened_matrix = reshaped_matrix.reshape({-1, conv_kernel.size(0) * conv_kernel.size(1)});

    // Reshape the convolution kernel for matrix multiplication
    auto reshaped_kernel = conv_kernel.view({-1, 1});
    qcdA=flattened_matrix;
    qcdB=reshaped_kernel;
    prepareTime=UtilityFunctions::timeLastUs(ts);
}
torch::Tensor QCDPairs::getResult(torch::Tensor result)
{
    return result.view({padded_rows - conv_kernel.size(0) + 1, padded_cols - conv_kernel.size(1) + 1});
}

void runSingleThreadTest(std::string configName) {
  MeterTable meterTable;
  AbstractMeterPtr eMeter = nullptr;
  ConfigMapPtr cfg = newConfigMap();
  cfg->fromFile(configName);
  AMMBench::MatrixLoaderTable mLoaderTable;
  uint64_t sketchDimension;
  ConfigMapPtr breakDownResult = nullptr;
  INTELLI_INFO("cppAlgoTag: "+cfg->tryString("cppAlgoTag", "mm", true));
 
  sketchDimension = cfg->tryU64("sketchDimension", 50, true);
  uint64_t coreBind = cfg->tryU64("coreBind", 0, true);
  uint64_t usingMeter = cfg->tryU64("usingMeter", 0, true);
  std::string meterTag = cfg->tryString("meterTag", "intelMsr", true);
  uint64_t useCPP = cfg->tryU64("useCPP", 0, true);
  uint64_t forceMP = cfg->tryU64("forceMP", 0, true);
  QCDPairs qcdp;
  if (usingMeter) {
    eMeter = meterTable.findMeter(meterTag);
    if (eMeter != nullptr) {
      eMeter->setConfig(cfg);
      double staticPower = cfg->tryDouble("staticPower", 0.0, false);
      if (staticPower == 0.0) {
        eMeter->testStaticPower(2);
      } else {
        INTELLI_INFO("use pre-defined static power");
        eMeter->setStaticPower(staticPower);
      }
      INTELLI_INFO("static power is " + to_string(eMeter->getStaticPower()) + " W");
    } else {
      INTELLI_ERROR("No meter found: " + meterTag);
    }

  }
  UtilityFunctions::bind2Core((int) coreBind);
  //torch::set_num_threads(1);
  std::string ptFile = cfg->tryString("ptFile", "torchscripts/FDAMM.pt", true);

  //uint64_t customResultName = cfg->tryU64("customResultName", 0, true);
  INTELLI_INFO("Place me at core" + to_string(coreBind));
  INTELLI_INFO("with sketch" + to_string(sketchDimension));
 
  std::string matrixLoaderTag = cfg->tryString("matrixLoaderTag", "random", true);
  auto matLoaderPtr = mLoaderTable.findMatrixLoader(matrixLoaderTag);
  assert(matLoaderPtr);
  matLoaderPtr->setConfig(cfg);
  qcdp.genPairs(matLoaderPtr->getA(),8,1.0);
  INTELLI_INFO("Pre-amm time for QCD ="+to_string(qcdp.prepareTime));
  auto A = qcdp.qcdA;
  auto B = qcdp.qcdB;
  torch::Tensor C;

  //555
  /*torch::manual_seed(114514);
//555
auto A = torch::rand({(long) aRow, (long) aCol});
auto B = torch::rand({(long) aCol, (long) bCol});*/
  INTELLI_INFO("Generation done, conducting...");
  uint64_t threads = cfg->tryU64("threads", 0, true);
  ThreadPerfPtr pef;
#if AMMBENCH_PAPI == 1
  if (cfg->tryU64("usePAPI", 1)) {
    pef = newThreadPerfPAPI(-1);
  } else {
    pef = newThreadPerf(-1);
  }
#else
  pef=newThreadPerf(-1);
#endif
  pef->initEventsByCfg(cfg);
  AMMBench::BlockPartitionRunner br,br2;
  if (threads > 1 || forceMP) {
    INTELLI_WARNING("use multithread");
    br.setConfig(cfg);
    br.createABC(A, B);
    if (eMeter != nullptr) {
      eMeter->startMeter();
    }
    pef->start();
    C = br.parallelForward();
    pef->end();
    if (eMeter != nullptr) {
      eMeter->stopMeter();
    }
    breakDownResult = br.getBreakDown();
  } else {
    AMMBench::CPPAlgoTable cppAlgoTable;
    std::string cppAlgoTag = cfg->tryString("cppAlgoTag", "mm", true);
    AMMBench::AbstractCPPAlgoPtr cppAlgoPtr = cppAlgoTable.findCppAlgo(cppAlgoTag);
    cppAlgoPtr->setConfig(cfg);
    INTELLI_WARNING("single thread, algo " + cppAlgoTag);
    if (eMeter != nullptr) {
      eMeter->startMeter();
    }
    pef->start();
   
    INTELLI_WARNING("this is pure c++");
    C = cppAlgoPtr->amm(A, B, sketchDimension);
   
    pef->end();
    if (eMeter != nullptr) {
      eMeter->stopMeter();
    }
    if (useCPP && cppAlgoPtr) {
      breakDownResult = cppAlgoPtr->getBreakDown();
    }
  }

  std::string ruName = "default";

  auto resultCsv = pef->resultToConfigMap();
  if (eMeter != nullptr) {
    eMeter->stopMeter();
    double energyConsumption = eMeter->getE();
    double staticEnergyConsumption = eMeter->getStaicEnergyConsumption(
        resultCsv->tryU64("perfElapsedTime", 0, false));
    double pureEnergy = energyConsumption - staticEnergyConsumption;
    resultCsv->edit("energyAll", (double) energyConsumption);
    resultCsv->edit("energyOnlyMe", (double) pureEnergy);
  }
  if (threads > 1 || forceMP) {
    INTELLI_WARNING("consider multithread elapsed time");
    resultCsv->edit("perfElapsedTime", (uint64_t) br.getElapsedTime());
    br.appendThreadInfo(resultCsv);
  }
  // error
   torch::Tensor realC; 
  INTELLI_WARNING("evaluating the error, may takes some time");
   if (threads > 1 || forceMP) {
    INTELLI_WARNING("use multithread to evaluate error");
    std::string ev="mm";

    cfg->edit("cppAlgoTag", ev);
    br2.setConfig(cfg);
    br2.createABC(A, B);
    if (eMeter != nullptr) {
      eMeter->startMeter();
    }
    pef->start();
    realC = br2.parallelForward();
    pef->end();
  }
  else
  {
    realC = torch::matmul(A, B);
  }
  uint64_t otherTime=qcdp.prepareTime;
  auto endCCal=qcdp.getResult(C);
  auto endCReal=qcdp.getResult(realC);
  double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, C);
  double qcdError= qcdMSE(realC, C);
  double froBNormal = B.norm().item<double>();
  double errorBoundRatio = froError / froBNormal;
  INTELLI_INFO("B normal is " + to_string(froBNormal));
  resultCsv->edit("froError", (double) froError);
  resultCsv->edit("qcdError", (double) qcdError);
  resultCsv->edit("errorBoundRatio", (double) errorBoundRatio);
  resultCsv->edit("otherTime", (uint64_t)otherTime);
  resultCsv->toFile(ruName + ".csv");
  INTELLI_INFO("Done. here is overall result");
  std::cout << resultCsv->toString() << endl;
  if (breakDownResult) {
    INTELLI_INFO("I also have some break down result");
    std::cout << breakDownResult->toString() << endl;
    breakDownResult->toFile(ruName + "_breakdown.csv");
  }

}

int main(int argc, char **argv) {
  string configName, outPrefix = "";
  if (argc >= 2) {
    configName += argv[1];
  } else {
    configName = "config.csv";
  }
  runSingleThreadTest(configName);
  return 0;
}
