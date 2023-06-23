//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/BlockLRACPPAlgo.h>
#include <ATen/ATen.h>
#include <vector>
#include <AMMBench.h>
using namespace std;
using namespace INTELLI;
using namespace torch;

namespace AMMBench {
torch::Tensor AMMBench::BlockLRACPPAlgo::amm(torch::Tensor A, torch::Tensor B, uint64_t blockSize,  float ARankRatio, float BRankRatio) {
  
  // Input size and block size
  uint64_t m = A.size(0);
  uint64_t k = A.size(1);
  uint64_t n = B.size(1);

  assert(m % blockSize == 0);
  assert(k % blockSize == 0);
  assert(n % blockSize == 0);

  // for each block, calculate LRA
  torch::Tensor finalLRA = torch::zeros({m, n});

  for (uint64_t I=0; I<m/blockSize; ++I){
    for (uint64_t J=0; J<n/blockSize; ++J){
      torch::Tensor subFinalLRA = torch::zeros({blockSize, blockSize});
      for (uint64_t K=0; K<k/blockSize; ++K){
        // get sub matrix
        torch::Tensor AIK = A.index({torch::indexing::Slice(I*blockSize, (I+1)*blockSize), torch::indexing::Slice(K*blockSize, (K+1)*blockSize)});
        torch::Tensor BKJ = B.index({torch::indexing::Slice(K*blockSize, (K+1)*blockSize), torch::indexing::Slice(J*blockSize, (J+1)*blockSize)});
        // SVD
        torch::Tensor UA;
        torch::Tensor SA;
        torch::Tensor VhA;
        torch::Tensor UB;
        torch::Tensor SB;
        torch::Tensor VhB;
        std::tie(UA, SA, VhA) = torch::linalg::svd(AIK, false, c10::nullopt);
        std::tie(UB, SB, VhB) = torch::linalg::svd(BKJ, false, c10::nullopt);
        // LRA
        torch::Tensor UATruncated = UA.narrow(1,0,ARankRatio*blockSize);
        torch::Tensor SATruncated = torch::diag(SA.narrow(0,0,ARankRatio*blockSize));
        torch::Tensor VhATruncated = VhA.narrow(0,0,ARankRatio*blockSize);
        torch::Tensor UBTruncated = UB.narrow(1,0,BRankRatio*blockSize);
        torch::Tensor SBTruncated = torch::diag(SB.narrow(0,0,BRankRatio*blockSize));
        torch::Tensor VhBTruncated = VhB.narrow(0,0,BRankRatio*blockSize);
        // MM
        subFinalLRA += torch::matmul(torch::matmul(UATruncated, torch::matmul(torch::matmul(SATruncated, torch::matmul(VhATruncated,UBTruncated)), SBTruncated)), VhBTruncated);
      }
    finalLRA.index_put_({torch::indexing::Slice(I*blockSize, (I+1)*blockSize), torch::indexing::Slice(J*blockSize, (J+1)*blockSize)}, subFinalLRA);
    }
  }

  return finalLRA;
}
} // AMMBench


int main() {
  
  torch::manual_seed(114514);
  AMMBench::BlockLRACPPAlgo wcr;
  auto A = torch::rand({500, 400});
  auto B = torch::rand({400, 600});
  auto realC = torch::matmul(A, B);
  auto ammC = wcr.amm(A, B, 100, 0.5, 0.5);
  double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC);
  std::cout << "froError: " << froError << std::endl;

  return 0;
}

