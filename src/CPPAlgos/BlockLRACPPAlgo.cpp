//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/BlockLRACPPAlgo.h>
#include <ATen/ATen.h>
#include <vector>
#include <AMMBench.h>
#include <cassert>
using namespace std;

namespace AMMBench {
void AMMBench::BlockLRACPPAlgo::setConfig(INTELLI::ConfigMapPtr cfg) {
  ARankRatio = cfg->tryDouble("algoARankRatio", 0.1, true); // 0.1 is kinda same as sketch_size=0.1
  BRankRatio = cfg->tryDouble("algoBRankRatio", 0.1, true); // 0.1 is kinda same as sketch_size=0.1
}

torch::Tensor AMMBench::BlockLRACPPAlgo::amm(torch::Tensor A, torch::Tensor B, uint64_t l) {

  l=0; // l is meaningless, as "sketch_size" is already set in rank, and blockSize is not about sketch size also.

  // Input size and block size
  uint64_t m = A.size(0);
  uint64_t k = A.size(1);
  uint64_t n = B.size(1);
  int blockSizeLimit = 30; // blockSize can not be large, as svd is very time consuming
  uint64_t blockSize = 1;

  uint64_t gcdOfmkn = std::gcd(std::gcd(m, k), n);
  for (int divisor = gcdOfmkn; divisor >= 1; divisor--) {
        if (m % divisor == 0 && k % divisor == 0 && n % divisor == 0 && divisor<=blockSizeLimit) {
          blockSize = divisor;
          break;
        }
    }

  if (blockSize == 1) {
    INTELLI_ERROR("m=" + to_string(m) + ", k=" + to_string(k) + ", n=" + to_string(n)
                      + ", gcd(m, k, n)=1, BlockLRA is not usable anymore");
    throw std::runtime_error("m=" + to_string(m) + ", k=" + to_string(k) + ", n=" + to_string(n)
                      + ", gcd(m, k, n)=1, BlockLRA is not usable anymore");
  }
  INTELLI_INFO("BlockLRA with adjusted blockSize: " + to_string(blockSize));

  // for each block, calculate LRA
  torch::Tensor finalLRA = torch::zeros({(long) m, (long) n});

  std::vector<std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>> svdA_blocks(m / blockSize, std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>(k / blockSize));

  std::vector<std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>> svdB_blocks(k / blockSize, std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>(n / blockSize));

  for (uint64_t I = 0; I < m / blockSize; ++I) {
      for (uint64_t K = 0; K < k / blockSize; ++K) {
          torch::Tensor AIK = A.index({torch::indexing::Slice(I * blockSize, (I + 1) * blockSize),
                                        torch::indexing::Slice(K * blockSize, (K + 1) * blockSize)});
          torch::Tensor UA, SA, VhA;
          std::tie(UA, SA, VhA) = torch::linalg::svd(AIK, false, c10::nullopt);
          torch::Tensor UATruncated = UA.narrow(1, 0, ARankRatio * blockSize);
          torch::Tensor SATruncated = torch::diag(SA.narrow(0, 0, ARankRatio * blockSize));
          torch::Tensor VhATruncated = VhA.narrow(0, 0, ARankRatio * blockSize);
          svdA_blocks[I][K] = std::make_tuple(UATruncated, SATruncated, VhATruncated);
      }
  }

  for (uint64_t K = 0; K < k / blockSize; ++K) {
      for (uint64_t J = 0; J < n / blockSize; ++J) {
          torch::Tensor BKJ = B.index({torch::indexing::Slice(K * blockSize, (K + 1) * blockSize),
                                        torch::indexing::Slice(J * blockSize, (J + 1) * blockSize)});
          torch::Tensor UB, SB, VhB;
          std::tie(UB, SB, VhB) = torch::linalg::svd(BKJ, false, c10::nullopt);
          torch::Tensor UBTruncated = UB.narrow(1, 0, BRankRatio * blockSize);
          torch::Tensor SBTruncated = torch::diag(SB.narrow(0, 0, BRankRatio * blockSize));
          torch::Tensor VhBTruncated = VhB.narrow(0, 0, BRankRatio * blockSize);
          svdB_blocks[K][J] = std::make_tuple(UBTruncated, SBTruncated, VhBTruncated);
      }
  }

  for (uint64_t I = 0; I < m / blockSize; ++I) {
      for (uint64_t J = 0; J < n / blockSize; ++J) {
          torch::Tensor subFinalLRA = torch::zeros({(long) blockSize, (long) blockSize});
          for (uint64_t K = 0; K < k / blockSize; ++K) {
              const auto& svdA = svdA_blocks[I][K];
              const auto& svdB = svdB_blocks[K][J];
              torch::Tensor UATruncated = std::get<0>(svdA);
              torch::Tensor SATruncated = std::get<1>(svdA);
              torch::Tensor VhATruncated = std::get<2>(svdA);
              torch::Tensor UBTruncated = std::get<0>(svdB);
              torch::Tensor SBTruncated = std::get<1>(svdB);
              torch::Tensor VhBTruncated = std::get<2>(svdB);
              subFinalLRA += torch::matmul(torch::matmul(UATruncated, 
                              torch::matmul(torch::matmul(SATruncated,
                                torch::matmul(VhATruncated,UBTruncated)), SBTruncated)), VhBTruncated);
          }

          finalLRA.index_put_({torch::indexing::Slice(I * blockSize, (I + 1) * blockSize),
                              torch::indexing::Slice(J * blockSize, (J + 1) * blockSize)}, subFinalLRA);
      }
  }

  return finalLRA;
}}