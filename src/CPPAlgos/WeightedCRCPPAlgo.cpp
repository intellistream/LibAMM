//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/WeightedCRCPPAlgo.h>
#include <ATen/ATen.h>

namespace AMMBench {
torch::Tensor AMMBench::WeightedCRCPPAlgo::amm(torch::Tensor A, torch::Tensor B, uint64_t c) {
  
  int64_t n = A.size(1); // A: m*n, B: n*d

  // Probability distribution
  torch::Tensor probability_distribution = torch::zeros({n});
    for (int i = 0; i < n; ++i) {
      probability_distribution[i] = torch::norm(A.t()[i]) * torch::norm(B[i]);
    }
    probability_distribution /= probability_distribution.sum();

  // S
  torch::Tensor sample_indices = torch::multinomial(probability_distribution, /*num_samples*/c, /*replacement*/true);
  
  torch::Tensor unique_indices, _, occurences;
  std::tie(unique_indices, _, occurences) = at::_unique2(sample_indices, /*sorted*/false, /*return_inverse*/false, /*return_counts*/true);

  torch::Tensor S = torch::zeros({n, unique_indices.size(0)});

  for (int trial = 0; trial < unique_indices.size(0); ++trial) {
    int index = unique_indices[trial].item<int>();
    S[index][trial] = 1;
  }

  // D
  torch::Tensor D = torch::diag(torch::sqrt(occurences) / torch::sqrt(torch::tensor({static_cast<long>(c)}, torch::kLong) * probability_distribution.index_select(0, unique_indices)));

  // ASD(SD)^TB
  torch::Tensor SS = torch::matmul(S, D);
  torch::Tensor C = torch::matmul(A, SS);
  torch::Tensor R = torch::matmul(SS.t(), B);
  torch::Tensor weighted_CR = torch::matmul(C, R);

  return weighted_CR;
}
} // AMMBench
