//
// Created by luv on 6/18/23.
//

#include <CPPAlgos/FastJLTCPPAlgo.h>

namespace LibAMM {
LibAMM::Tensor hadamard_transform_matrix(int64_t n) {
  LibAMM::Tensor H = LibAMM::ones({1, 1}, torch::kInt8);
  int64_t i = 1;
  while (i < n) {
    auto H_top = torch::cat({H, H}, 1);
    auto H_bottom = torch::cat({H, -H}, 1);
    H = torch::cat({H_top, H_bottom}, 0); // view or shallow copy of H_top and H_bottom
    // std::cout << "i: " << i << " n: " << n << std::endl;
    // size_t H_memory_bytes = H.numel() * H.element_size();
    // double H_memory_gigabytes = static_cast<double>(H_memory_bytes) / (1024 * 1024 * 1024);
    // std::cout << "Memory used by the H: " << H_memory_gigabytes << " GB" << std::endl;
    i *= 2;
  }
  return H;
}

LibAMM::Tensor FastJLTCPPAlgo::amm(LibAMM::Tensor A, LibAMM::Tensor B, uint64_t d_) {
  int64_t d = (int64_t) d_;
  int64_t N = A.size(0);
  int64_t D = A.size(1);
  int64_t M = B.size(1);

  // Pad A and B for FHT
  int64_t log2_D = std::ceil(std::log2(static_cast<float>(D)));
  int64_t D_pad = static_cast<int64_t>(std::pow(2, log2_D));
  LibAMM::Tensor A_pad = LibAMM::zeros({N, D_pad});
  A_pad.narrow(1, 0, D) = A;
  LibAMM::Tensor B_pad = LibAMM::zeros({D_pad, M});
  B_pad.narrow(0, 0, D) = B;

  // Construct and apply random signs for each dimension
  LibAMM::Tensor randsigns = (LibAMM::randint(0, 2, {D_pad}) * 2 - 1).to("float32");
  randsigns *= 1.0 / std::sqrt(static_cast<float>(D_pad));
  A_pad *= randsigns;
  B_pad *= randsigns.view({-1, 1});

  // Apply Fast Hadamard Transform
  // for MNIST, A columns=60000, that means H dimension=65536*65536, which is too big for silver(32G) to perform the following computation. Thus we comment out below and do it in batch
  LibAMM::Tensor H = hadamard_transform_matrix(D_pad);
  // A_pad = LibAMM::matmul(A_pad, H.t());
  // B_pad = LibAMM::matmul(H, B_pad);
  // batch process below achieves the same as above
  int64_t chunk_size = 10000;
  int64_t num_chunks = (D_pad + chunk_size - 1) / chunk_size;
  LibAMM::Tensor A_pad_result = LibAMM::zeros({A_pad.size(0), A_pad.size(1)});
  for (int64_t i = 0; i < num_chunks; ++i) {
    int64_t start_idx = i * chunk_size;
    int64_t end_idx = std::min((i + 1) * chunk_size, D_pad);
    // std::cout << i << " s: " << start_idx << " e: " << end_idx << std::endl;
    A_pad_result.narrow(1, start_idx, end_idx - start_idx) = LibAMM::matmul(A_pad, H.slice(0, start_idx, end_idx).t().to("float32"));
  }
  A_pad = A_pad_result;
  LibAMM::Tensor B_pad_result = LibAMM::zeros({B_pad.size(0), B_pad.size(1)});
  for (int64_t i = 0; i < num_chunks; ++i) {
    int64_t start_idx = i * chunk_size;
    int64_t end_idx = std::min((i + 1) * chunk_size, D_pad);
    B_pad_result.narrow(0, start_idx, end_idx - start_idx) = LibAMM::matmul(H.slice(0, start_idx, end_idx).to("float32"), B_pad);
  }
  B_pad = B_pad_result;

  // Dimensionality reduction
  float q = static_cast<float>(log2_D * log2_D) / static_cast<float>(D_pad);
  auto mask = LibAMM::rand({d, D_pad});
  mask = (mask < q).to("float32"); // 1 with probability q, 0 with probability 1-q
  auto normal_dist_tensor = LibAMM::randn({d, D_pad}) * std::sqrt(1.0 / q);
  auto P = mask * normal_dist_tensor / std::sqrt(static_cast<float>(d));
  return LibAMM::matmul(LibAMM::matmul(A_pad, P.t()), LibAMM::matmul(P, B_pad));
  // LibAMM::Tensor probs = LibAMM::ones(D_pad) / D_pad;  // default: uniform
  // LibAMM::Tensor indices = LibAMM::multinomial(probs, d, true);
  // LibAMM::Tensor A_sampled = A_pad.t().index_select(0, indices);
  // A_sampled = (A_sampled / (int) d).t().div(LibAMM::ones(1) / D_pad);
  // LibAMM::Tensor B_sampled = B_pad.index_select(0, indices);
  // return LibAMM::matmul(A_sampled, B_sampled);
}
} // LibAMM
