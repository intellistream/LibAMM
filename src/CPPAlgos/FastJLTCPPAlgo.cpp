//
// Created by luv on 6/18/23.
//

#include <CPPAlgos/FastJLTCPPAlgo.h>

namespace AMMBench {
torch::Tensor hadamard_transform_matrix(int64_t n) {
  torch::Tensor H = torch::ones({1, 1});
  int64_t i = 1;
  while (i < n) {
    auto H_top = torch::cat({H, H}, 1);
    auto H_bottom = torch::cat({H, -H}, 1);
    H = torch::cat({H_top, H_bottom}, 0);
    i *= 2;
  }
  return H;
}

torch::Tensor FastJLTCPPAlgo::amm(torch::Tensor A, torch::Tensor B, uint64_t d_) {
  int64_t d = (int64_t) d_;
  int64_t N = A.size(0);
  int64_t D = A.size(1);
  int64_t M = B.size(1);

  // Pad A and B for FHT
  int64_t log2_D = std::ceil(std::log2(static_cast<float>(D)));
  int64_t D_pad = static_cast<int64_t>(std::pow(2, log2_D));
  torch::Tensor A_pad = torch::zeros({N, D_pad});
  A_pad.narrow(1, 0, D) = A;
  torch::Tensor B_pad = torch::zeros({D_pad, M});
  B_pad.narrow(0, 0, D) = B;

  // Construct and apply random signs for each dimension
  torch::Tensor randsigns = (torch::randint(0, 2, {D_pad}) * 2 - 1).to(torch::kFloat32);
  randsigns *= 1.0 / std::sqrt(static_cast<float>(D_pad));
  A_pad *= randsigns;
  B_pad *= randsigns.view({-1, 1});

  // Apply Fast Hadamard Transform
  torch::Tensor H = hadamard_transform_matrix(D_pad);
  A_pad = torch::matmul(A_pad, H.t());
  B_pad = torch::matmul(H, B_pad);

  // Dimensionality reduction
  float q = 0.5;
  auto mask = torch::rand({d, D_pad});
  mask = (mask < q).to(torch::kFloat32); // 1 with probability q, 0 with probability 1-q
  auto normal_dist_tensor = torch::randn({d, D_pad}) * std::sqrt(1.0 / q);
  auto P = mask * normal_dist_tensor / std::sqrt(static_cast<float>(d));

  return torch::matmul(torch::matmul(A_pad, P.t()), torch::matmul(P, B_pad));
}
} // AMMBench
