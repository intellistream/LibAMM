//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/RIPCPPAlgo.h>
#include <AMMBench.h>

using namespace std;
using namespace INTELLI;
using namespace torch;

torch::Tensor createRandomTensor(int N) {
    // return a tensor 
    torch::Tensor tensor = torch::empty(N);
    tensor.uniform_(-1, 1); // Fill the tensor with values uniformly sampled from [-1, 1]
    tensor.sign_(); // Set the sign of each element to either 1 or -1
    
    return tensor;
}

namespace AMMBench {
    torch::Tensor AMMBench::RIPCPPAlgo::amm(torch::Tensor A, torch::Tensor B, uint64_t k2) {

        // Step 1: Input A:n1*d B:d*n2
        A = A.t(); // d*n1
        int64_t d = A.size(0);
        int64_t k = (int64_t) k2;

        // Step 2: RIP matrix k*d
        torch::Tensor RIP = torch::zeros({k, d});

        // first row as Rademacher random vector, X=+1 w.p. 1/2, X=-1 w.p. 1/2 1*d
        torch::Tensor firstRow = 1 / std::sqrt(k) * createRandomTensor(d);
        // std::cout << RIP.sizes() << std::endl;
        // std::cout << RIP << std::endl;

        // from paper: each subsequent row is created by rotating one element to the right relative to the preceding row vector d*d, then sample k rows from d*d, get RIP matrix k*d
        // in implementation, we create k*d directly to avoid create such a big matrix d*d
        torch::Tensor indices = torch::multinomial(torch::ones(d)/d, k, /*replacement*/false);
        // std::cout << indices << std::endl;

        for (int64_t i = 0; i < k; ++i) {
            RIP[i] = torch::roll(firstRow, indices[i].item<int>());
        }
        // std::cout << RIP << std::endl;

        // Step 3: Randomized column signs
        torch::Tensor D = torch::diag(createRandomTensor(d));
        // std::cout << D << std::endl;

        // Step 4: Compute AVVTB
        torch::Tensor A_prime = torch::matmul(torch::matmul(RIP, D), A); // A' = k*d * d*d * d*n1 = k*n1
        torch::Tensor B_prime = torch::matmul(torch::matmul(RIP, D), B); // B' = k*d * d*d * d*n2 = k*n2
        // std::cout << A_prime.sizes() << std::endl;
        // std::cout << B_prime.sizes() << std::endl;

        return torch::matmul(A_prime.t(), B_prime);
    }
} // AMMBench

// GDB debug purpose
int main() {
    // torch::manual_seed(114513);
    AMMBench::RIPCPPAlgo rip;
    auto A = torch::rand({600, 400});
    auto B = torch::rand({400, 1000});
    auto realC = torch::matmul(A, B);
    auto ammC = rip.amm(A, B, 20);
    double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC);
    std::cout << froError << std::endl;
    // REQUIRE(froError < 0.5);
}
