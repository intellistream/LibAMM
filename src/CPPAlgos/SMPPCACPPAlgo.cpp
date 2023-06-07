//
// Created by yuhao on 6/6/23.
//

#include <CPPAlgos/SMPPCACPPAlgo.h>

namespace AMMBench {
torch::Tensor AMMBench::SMPPCACPPAlgo::amm(torch::Tensor A, torch::Tensor B, uint64_t k) {
   // Step 1: Input
    A = A.t(); // d*n1
    int64_t d = A.size(0);
    int64_t n1 = A.size(1);
    int64_t n2 = B.size(1);
    
    // Step 2: Get sketched matrix
    torch::Tensor pi = 1/std::sqrt(k) * torch::randn({k, d}); // Gaussian sketching matrix
    torch::Tensor A_tilde = torch::matmul(pi, A); // k*n1
    torch::Tensor B_tilde = torch::matmul(pi, B); // k*n2

    torch::Tensor A_tilde_B_tilde = torch::matmul(A_tilde.t(), B_tilde);
    
    // Compute column norms of A and B
    torch::Tensor col_norm_A = torch::linalg::vector_norm(A, 2, {0}, false, c10::nullopt); // ||Ai|| for i in [n1]
    torch::Tensor col_norm_B = torch::linalg::vector_norm(B, 2, {0}, false, c10::nullopt); // ||Bj|| for j in [n2]

    torch::Tensor col_norm_A_tilde = torch::linalg::vector_norm(A_tilde, 2, {0}, false, c10::nullopt); // ||Ai|| for i in [n1]
    torch::Tensor col_norm_B_tilde = torch::linalg::vector_norm(B_tilde, 2, {0}, false, c10::nullopt); // ||Bj|| for j in [n2]
    
    // Compute M_tilde
    torch::Tensor col_norm_A_col_norm_B = torch::matmul(col_norm_A.reshape({n1,1}), col_norm_B.reshape({1,n2}));
    torch::Tensor col_norm_A_tilde_col_norm_B_tilde = torch::matmul(col_norm_A_tilde.reshape({n1,1}), col_norm_B_tilde.reshape({1,n2}));
    torch::Tensor M_tilde = torch::div(torch::mul(A_tilde_B_tilde, col_norm_A_col_norm_B), col_norm_A_tilde_col_norm_B_tilde);

    return M_tilde;
}
} // AMMBench