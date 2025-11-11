//
// Created by haolan on 5/26/23.
//

#include <CPPAlgos/CRSV2CPPAlgo.h>

namespace LibAMM {
LibAMM::Tensor LibAMM::CRSV2CPPAlgo::amm(LibAMM::Tensor A, LibAMM::Tensor B, uint64_t k2) {
  A = A.t();
  auto A_size = A.sizes();
  int64_t n = A_size[0];
  // int64_t m = A_size[1];
  int64_t k = (int64_t) k2;
  assert(n == B.size(0));
  //TORCH_CHECK(n == B.size(0));
  //TORCH_CHECK(k < n);

  //INTELLI_INFO("Running CRS V2 CPP");

  // probability distribution
  LibAMM::Tensor sample = LibAMM::rand({n}); // default: uniform

  // diagonal scaling matrix D (nxn)
  sample = sample.div(sample.sum());
  LibAMM::Tensor D = LibAMM::diag(1.0 / LibAMM::sqrt(k * sample));

  // sampling matrix S (kxn)
  LibAMM::Tensor column_indices = LibAMM::multinomial(sample, k, true);
  LibAMM::Tensor S = LibAMM::zeros({k, n});
  for (int64_t row = 0; row < k; row++) {
    int64_t col = column_indices[row].item<int64_t>();
    S[row][col] = 1;
  }

  LibAMM::Tensor a = LibAMM::matmul(LibAMM::matmul(A.t(), D), S.t());
  LibAMM::Tensor b = LibAMM::matmul(LibAMM::matmul(a, S), D);

  return LibAMM::matmul(b, B);
}
} // LibAMM
