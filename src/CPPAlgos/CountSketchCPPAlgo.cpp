//
// Created by haolan on 5/26/23.
//

#include <CPPAlgos/CountSketchCPPAlgo.h>

namespace LibAMM {
LibAMM::Tensor LibAMM::CountSketchCPPAlgo::amm(LibAMM::Tensor A, LibAMM::Tensor B, uint64_t k2) {
  int64_t m1 = A.size(0);
  int64_t n = A.size(1);
  int64_t m2 = B.size(1);
  int64_t k = (int64_t) k2;
  // Initialize sketch matrices
  LibAMM::Tensor Ca = LibAMM::zeros({m1, k});
  LibAMM::Tensor Cb = LibAMM::zeros({k, m2});

  LibAMM::Tensor L = LibAMM::randint(k, {n});
  LibAMM::Tensor G = LibAMM::randint(2, {n});

  // Modify the random column with random sign
  for (int64_t i = 0; i < n; ++i) {
    if (G[i].item<int>() == 1) {
      Ca.index({torch::indexing::Slice(), L[i]}).add_(A.index({torch::indexing::Slice(), i}));
      Cb.index({L[i], torch::indexing::Slice()}).add_(B.index({i, torch::indexing::Slice()}));
    } else {
      Ca.index({torch::indexing::Slice(), L[i]}).sub_(A.index({torch::indexing::Slice(), i}));
      Cb.index({L[i], torch::indexing::Slice()}).sub_(B.index({i, torch::indexing::Slice()}));
    }
  }

  return LibAMM::matmul(Ca, Cb);
}
} // LibAMM
