//
// Created by tony on 25/05/23.
//

#include <CPPAlgos/CRSCPPAlgo.h>
#include <Utils/UtilityFunctions.h>
#include <chrono>
namespace LibAMM {
LibAMM::Tensor LibAMM::CRSCPPAlgo::amm(LibAMM::Tensor A, LibAMM::Tensor B, uint64_t k) {
  LibAMM::Tensor C;
  auto start = std::chrono::high_resolution_clock::now();



  // Sample k rows from B

  if (useCuda) {
    INTELLI_INFO("CRS under cuda!");
    // Probability distribution
    int64_t n = A.size(0);
    LibAMM::Tensor probs = LibAMM::ones(n) / n;  // default: uniform
    probs=probs.to(LibAMM::kCUDA);
    A = A.to(LibAMM::kCUDA);
    B = B.to(LibAMM::kCUDA);
    buildATime = chronoElapsedTime(start);
    LibAMM::Tensor B_sampled;
    A = A.t();
    // Sample k indices from range 0 to n for given probability distribution
    LibAMM::Tensor indices = LibAMM::multinomial(probs, k, true);
    indices=indices.to(LibAMM::kCUDA);
    // Sample k columns from A
    LibAMM::Tensor A_sampled = A.index_select(0, indices);
    // int64_t ratio = std::ceil(static_cast<double>(n) / k);
    // A_sampled = (A_sampled / (int) k).t().div(probs.index_select(0, LibAMM::arange(0, n, ratio)));
    A_sampled = (A_sampled / (int) k).t().div(LibAMM::ones(1,LibAMM::kCUDA) / n);

    auto ac = A_sampled.to(LibAMM::kCUDA);

    B_sampled = B.index_select(0, indices);
    auto bc = B_sampled.to(LibAMM::kCUDA);
    buildBTime = chronoElapsedTime(start) - buildATime;
    auto cc = LibAMM::matmul(ac, bc);
    fABTime = chronoElapsedTime(start) - buildATime - buildBTime;
    C = cc.to(LibAMM::kCPU);
    postProcessTime = chronoElapsedTime(start) - buildATime - buildBTime - fABTime;
  } else {
    LibAMM::Tensor B_sampled;
    A = A.t();
    //INTELLI_INFO("I am CPP-CRS");
    int64_t n = A.size(0);
    //int64_t m = A.size(1);

    assert(n == B.size(0));
    // Probability distribution
    LibAMM::Tensor probs = LibAMM::ones(n) / n;  // default: uniform

    // Sample k indices from range 0 to n for given probability distribution
    LibAMM::Tensor indices = LibAMM::multinomial(probs, k, true);

    // Sample k columns from A
    LibAMM::Tensor A_sampled = A.index_select(0, indices);
    // int64_t ratio = std::ceil(static_cast<double>(n) / k);
    // A_sampled = (A_sampled / (int) k).t().div(probs.index_select(0, LibAMM::arange(0, n, ratio)));
    A_sampled = (A_sampled / (int) k).t().div(LibAMM::ones(1) / n);
    buildATime = chronoElapsedTime(start);
    B_sampled = B.index_select(0, indices);
    buildBTime = chronoElapsedTime(start) - buildATime;
    C = LibAMM::matmul(A_sampled, B_sampled);
    fABTime = chronoElapsedTime(start) - buildATime - buildBTime;
  }
  return C;
}
} // LibAMM