//
// Created by haolan on 5/29/23.
//

#include <CPPAlgos/EWSCPPAlgo.h>

namespace LibAMM {
LibAMM::Tensor LibAMM::EWSCPPAlgo::amm(LibAMM::Tensor A, LibAMM::Tensor B, uint64_t k2) {
  auto A_size = A.sizes();
  int64_t m = A_size[0];
  int64_t n = A_size[1];
  int64_t k = (int64_t) k2;
  TORCH_CHECK(n == B.size(0));
  TORCH_CHECK(k < n);

  int64_t p = B.size(1);

  // probability distribution
  LibAMM::Tensor probs = LibAMM::rand({m, n});

  // S matrix that samples A with scaling
  LibAMM::Tensor mask = LibAMM::rand_like(probs) < probs;
  LibAMM::Tensor S = LibAMM::zeros_like(A);
  S.masked_scatter_(mask, A.masked_select(mask) / probs.masked_select(mask));

  // R matrix that samples B with scaling
  LibAMM::Tensor probs_r = LibAMM::rand({n, p});  // a different probabilistic distribution
  LibAMM::Tensor mask_r = LibAMM::rand_like(probs_r) < probs_r;
  LibAMM::Tensor R = LibAMM::zeros_like(B);
  R.masked_scatter_(mask_r, B.masked_select(mask_r) / probs_r.masked_select(mask_r));

  return LibAMM::matmul(S, R);
}
}