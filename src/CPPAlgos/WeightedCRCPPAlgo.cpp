//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/WeightedCRCPPAlgo.h>
// #include <ATen/ATen.h> // Removed: PyTorch dependency

namespace LibAMM {
LibAMM::Tensor LibAMM::WeightedCRCPPAlgo::amm(LibAMM::Tensor A, LibAMM::Tensor B, uint64_t c) {

  int64_t n = A.size(1); // A: m*n, B: n*d
  // std::cout << "A shape: " << A.sizes() << std::endl;
  // std::cout << "B shape: " << B.sizes() << std::endl;

  // Probability distribution
  LibAMM::Tensor col_norm_A = torch::norm(A, /*p=*/2, /*dim=*/0); // norm on columns of A
  LibAMM::Tensor row_norm_B = torch::norm(B, /*p=*/2, /*dim=*/1); // norm on rows of B
  LibAMM::Tensor probability_distribution = torch::mul(col_norm_A, row_norm_B);
  probability_distribution /= probability_distribution.sum();
  // LibAMM::Tensor probability_distribution = LibAMM::ones(n) / n;

  // unique indices and occurences
  LibAMM::manual_seed(999);
  LibAMM::Tensor sample_indices = LibAMM::multinomial(probability_distribution, /*num_samples*/c, /*replacement*/true);

  LibAMM::Tensor unique_indices, _, occurences;
  std::tie(unique_indices, _, occurences) =
      at::_unique2(sample_indices, /*sorted*/false, /*return_inverse*/false, /*return_counts*/true);

  // sample with occurences as weight 
  LibAMM::Tensor At_sampled = A.t().index_select(0, unique_indices);
  LibAMM::Tensor A_sampled = At_sampled.t() * occurences / (int) c * n;
  LibAMM::Tensor B_sampled = B.index_select(0, unique_indices);
  LibAMM::Tensor weighted_CR = LibAMM::matmul(A_sampled, B_sampled);

  return weighted_CR;
}
} // LibAMM
