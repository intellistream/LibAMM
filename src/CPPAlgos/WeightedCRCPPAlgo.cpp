//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/WeightedCRCPPAlgo.h>
#include <ATen/ATen.h>

namespace AMMBench {
    torch::Tensor AMMBench::WeightedCRCPPAlgo::amm(torch::Tensor A, torch::Tensor B, uint64_t c) {

        int64_t n = A.size(1); // A: m*n, B: n*d
        std::cout << "A shape: " << A.sizes() << std::endl;
        std::cout << "B shape: " << B.sizes() << std::endl;

        // Probability distribution
        torch::Tensor col_norm_A = torch::norm(A, /*p=*/2, /*dim=*/0); // norm on columns of A
        torch::Tensor row_norm_B = torch::norm(B, /*p=*/2, /*dim=*/1); // norm on rows of B
        torch::Tensor probability_distribution = torch::mul(col_norm_A, row_norm_B);
        probability_distribution /= probability_distribution.sum();

        // S
        torch::Tensor sample_indices = torch::multinomial(probability_distribution, /*num_samples*/c, /*replacement*/
                                                          true);

        torch::Tensor unique_indices, _, occurences;
        std::tie(unique_indices, _, occurences) = at::_unique2(sample_indices, /*sorted*/false, /*return_inverse*/
                                                               false, /*return_counts*/true);

        torch::Tensor S = torch::zeros({n, unique_indices.size(0)});
        torch::Tensor trial = torch::arange(unique_indices.size(0));
        S.index_put_({unique_indices, trial}, 1);

        // D
        torch::Tensor D = torch::diag(torch::sqrt(occurences) / torch::sqrt(
                torch::tensor({static_cast<long>(c)}, torch::kLong) *
                probability_distribution.index_select(0, unique_indices)));

        // ASD(SD)^TB
        torch::Tensor SS = torch::matmul(S, D);
        torch::Tensor C = torch::matmul(A, SS);
        torch::Tensor R = torch::matmul(SS.t(), B);
        torch::Tensor weighted_CR = torch::matmul(C, R);

        return weighted_CR;
    }
} // AMMBench
