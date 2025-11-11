//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/BlockLRACPPAlgo.h>
// #include <ATen/ATen.h> // Removed: PyTorch dependency
#include <vector>
#include <LibAMM.h>
#include <cassert>
using namespace std;

namespace LibAMM {
void LibAMM::BlockLRACPPAlgo::setConfig(INTELLI::ConfigMapPtr cfg) {
  ARankRatio = cfg->tryDouble("algoARankRatio", 0.1, true); // 0.1 is kinda same as sketch_size=0.1
  BRankRatio = cfg->tryDouble("algoBRankRatio", 0.1, true); // 0.1 is kinda same as sketch_size=0.1
}

LibAMM::Tensor LibAMM::BlockLRACPPAlgo::amm(LibAMM::Tensor A, LibAMM::Tensor B, uint64_t l) {

  l=0; // l is meaningless, as "sketch_size" is already set in rank, and blockSize is not about sketch size also.
  uint64_t blockSize = 30;

  // Input size and block size, add 0 means before padding
  uint64_t m0 = A.size(0);
  uint64_t k0 = A.size(1);
  uint64_t n0 = B.size(1);

  // after padding, m0->m, k0->k, n0->n
  uint64_t m = (A.size(0)+(blockSize-1))/blockSize*blockSize;
  uint64_t k = (A.size(1)+(blockSize-1))/blockSize*blockSize;
  uint64_t n = (B.size(1)+(blockSize-1))/blockSize*blockSize;

  // Pad the matrix with zeros
  LibAMM::Tensor padded_A = LibAMM::zeros({(int)m, (int)k});
  padded_A.slice(0, 0, m0).slice(1, 0, k0) = A;
  A = padded_A;
  LibAMM::Tensor padded_B = LibAMM::zeros({(int)k, (int)n});
  padded_B.slice(0, 0, k0).slice(1, 0, n0) = B;
  B = padded_B;
  
  // for each block, calculate LRA
  LibAMM::Tensor finalLRA = LibAMM::zeros({(long) m, (long) n});

  std::vector<std::vector<std::tuple<LibAMM::Tensor, LibAMM::Tensor, LibAMM::Tensor>>> svdA_blocks(m / blockSize, std::vector<std::tuple<LibAMM::Tensor, LibAMM::Tensor, LibAMM::Tensor>>(k / blockSize));

  std::vector<std::vector<std::tuple<LibAMM::Tensor, LibAMM::Tensor, LibAMM::Tensor>>> svdB_blocks(k / blockSize, std::vector<std::tuple<LibAMM::Tensor, LibAMM::Tensor, LibAMM::Tensor>>(n / blockSize));

  for (uint64_t I = 0; I < m / blockSize; ++I) {
      for (uint64_t K = 0; K < k / blockSize; ++K) {
          LibAMM::Tensor AIK = A.index({torch::indexing::Slice(I * blockSize, (I + 1) * blockSize),
                                        torch::indexing::Slice(K * blockSize, (K + 1) * blockSize)});
          LibAMM::Tensor UA, SA, VhA;
          std::tie(UA, SA, VhA) = torch::linalg::svd(AIK, false, c10::nullopt);
          LibAMM::Tensor UATruncated = UA.narrow(1, 0, ceil(ARankRatio * blockSize)); // ceil in case ARankRatio * blockSize < 1, then we set it to 1 at least
          LibAMM::Tensor SATruncated = LibAMM::diag(SA.narrow(0, 0, ceil(ARankRatio * blockSize)));
          LibAMM::Tensor VhATruncated = VhA.narrow(0, 0, ceil(ARankRatio * blockSize));
          svdA_blocks[I][K] = std::make_tuple(UATruncated, SATruncated, VhATruncated);
      }
  }

  for (uint64_t K = 0; K < k / blockSize; ++K) {
      for (uint64_t J = 0; J < n / blockSize; ++J) {
          LibAMM::Tensor BKJ = B.index({torch::indexing::Slice(K * blockSize, (K + 1) * blockSize),
                                        torch::indexing::Slice(J * blockSize, (J + 1) * blockSize)});
          LibAMM::Tensor UB, SB, VhB;
          std::tie(UB, SB, VhB) = torch::linalg::svd(BKJ, false, c10::nullopt);
          LibAMM::Tensor UBTruncated = UB.narrow(1, 0, ceil(BRankRatio * blockSize));
          LibAMM::Tensor SBTruncated = LibAMM::diag(SB.narrow(0, 0, ceil(BRankRatio * blockSize)));
          LibAMM::Tensor VhBTruncated = VhB.narrow(0, 0, ceil(BRankRatio * blockSize));
          svdB_blocks[K][J] = std::make_tuple(UBTruncated, SBTruncated, VhBTruncated);
      }
  }

  for (uint64_t I = 0; I < m / blockSize; ++I) {
      for (uint64_t J = 0; J < n / blockSize; ++J) {
          LibAMM::Tensor subFinalLRA = LibAMM::zeros({(long) blockSize, (long) blockSize});
          for (uint64_t K = 0; K < k / blockSize; ++K) {
              const auto& svdA = svdA_blocks[I][K];
              const auto& svdB = svdB_blocks[K][J];
              LibAMM::Tensor UATruncated = std::get<0>(svdA);
              LibAMM::Tensor SATruncated = std::get<1>(svdA);
              LibAMM::Tensor VhATruncated = std::get<2>(svdA);
              LibAMM::Tensor UBTruncated = std::get<0>(svdB);
              LibAMM::Tensor SBTruncated = std::get<1>(svdB);
              LibAMM::Tensor VhBTruncated = std::get<2>(svdB);
              subFinalLRA += LibAMM::matmul(LibAMM::matmul(UATruncated, 
                              LibAMM::matmul(LibAMM::matmul(SATruncated,
                                LibAMM::matmul(VhATruncated,UBTruncated)), SBTruncated)), VhBTruncated);
          }

          finalLRA.index_put_({torch::indexing::Slice(I * blockSize, (I + 1) * blockSize),
                              torch::indexing::Slice(J * blockSize, (J + 1) * blockSize)}, subFinalLRA);
      }
  }

  return finalLRA.slice(0,0,m0).slice(1,0,n0);
}}