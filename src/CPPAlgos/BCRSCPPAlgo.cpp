//
// Created by haolan on 5/29/23.
//
#include <CPPAlgos/BCRSCPPAlgo.h>

namespace LibAMM {
LibAMM::Tensor BCRSCPPAlgo::amm(LibAMM::Tensor A, LibAMM::Tensor B, uint64_t k2) {
  A = A.t();
  auto A_size = A.sizes();
  int64_t n = A_size[0];
  //int64_t m = A_size[1];
  int64_t k = (int64_t) k2;
  assert(n == B.size(0));
  //TORCH_CHECK(n == B.size(0));
  //TORCH_CHECK(k < n);

  //INTELLI_INFO("Running Bernoulli CRS CPP");

  // probability distribution
  LibAMM::Tensor sample = LibAMM::rand({n}); // default: uniform
  sample = sample.div(sample.sum() / k);  // sum = k as per the paper

  // diagonal scaling matrix P (nxn)
  LibAMM::Tensor P = LibAMM::diag(1.0 / LibAMM::sqrt(sample));

  // random diagonal sampling matrix K (nxn)
  sample = (LibAMM::rand({n}) < sample).to("float32");
  LibAMM::Tensor K = LibAMM::diag(sample);

  LibAMM::Tensor a = LibAMM::matmul(LibAMM::matmul(A.t(), P), K);
  LibAMM::Tensor b = LibAMM::matmul(LibAMM::matmul(a, K), P);

  return LibAMM::matmul(b, B);
}
}