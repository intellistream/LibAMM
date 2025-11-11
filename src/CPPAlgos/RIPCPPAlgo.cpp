//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/RIPCPPAlgo.h>
#include <LibAMM.h>

using namespace std;
using namespace INTELLI;
using namespace torch;

LibAMM::Tensor createRandomTensor(int N) {
  // return a tensor
  LibAMM::Tensor tensor = LibAMM::empty(N);
  tensor.uniform_(-1, 1); // Fill the tensor with values uniformly sampled from [-1, 1]
  tensor.sign_(); // Set the sign of each element to either 1 or -1

  return tensor;
}

namespace LibAMM {
LibAMM::Tensor LibAMM::RIPCPPAlgo::amm(LibAMM::Tensor A, LibAMM::Tensor B, uint64_t k2) {

  // Step 1: Input A:n1*d B:d*n2
  A = A.t(); // d*n1
  int64_t d = A.size(0);
  int64_t k = (int64_t) k2;

  // Step 2: RIP matrix k*d
  LibAMM::Tensor RIP = LibAMM::zeros({k, d});

  // first row as Rademacher random vector, X=+1 w.p. 1/2, X=-1 w.p. 1/2 1*d
  // LibAMM::Tensor firstRow = 1 / std::sqrt(k) * createRandomTensor(d);

  // from paper: each subsequent row is created by rotating one element to the right relative to the preceding row vector d*d, then sample k rows from d*d, get RIP matrix k*d
  // in implementation, we create k*d directly to avoid create such a big matrix d*d
  // LibAMM::Tensor indices = LibAMM::multinomial(LibAMM::ones(d) / d, k, /*replacement*/false);
  // for (int64_t i = 0; i < k; ++i) {
  //   INTELLI_INFO(to_string(i) + "/" + to_string(k));
  //   RIP[i] = torch::roll(firstRow, indices[i].item<int>());
  // }
  // we comment above for flat matrix A, cuz roll operation too expensive
  RIP = LibAMM::empty({k, d});
  RIP.uniform_(-1, 1);
  RIP.sign_();
  RIP = RIP * 1 / std::sqrt(k);

  // Step 3: Randomized column signs
  LibAMM::Tensor D = createRandomTensor(d);
  
  // Step 4: Compute AVVTB
  LibAMM::Tensor A_prime = LibAMM::matmul(RIP, A*D.view({-1, 1})); // A' = k*d * d*n1 = k*n1
  LibAMM::Tensor B_prime = LibAMM::matmul(RIP, B*D.view({-1, 1})); // B' = k*d * d*n2 = k*n2
  return LibAMM::matmul(A_prime.t(), B_prime);
}
} // LibAMM

// GDB debug purpose
// int main() {
//     // LibAMM::manual_seed(114513);
//     LibAMM::RIPCPPAlgo rip;
//     auto A = LibAMM::rand({600, 400});
//     auto B = LibAMM::rand({400, 1000});
//     auto realC = LibAMM::matmul(A, B);
//     auto ammC = rip.amm(A, B, 20);
//     double froError = INTELLI::UtilityFunctions::relativeFrobeniusNorm(realC, ammC);
//     std::cout << froError << std::endl;
//     // REQUIRE(froError < 0.5);
// }
