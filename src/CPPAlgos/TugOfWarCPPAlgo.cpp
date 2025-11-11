//
// Created by luv on 5/30/23.
//

#include <CPPAlgos/TugOfWarCPPAlgo.h>

namespace LibAMM {
void TugOfWarCPPAlgo::setConfig(INTELLI::ConfigMapPtr cfg) {
  algoDelta = cfg->tryDouble("algoDelta", algoDelta, true);
}

LibAMM::Tensor TugOfWarCPPAlgo::generateTugOfWarMatrix(int64_t m, int64_t n) {
  LibAMM::Tensor matrix = LibAMM::empty({m, n}, "float32");
  LibAMM::randn_out(matrix, {m, n});
  matrix.sign_();
  matrix = matrix/float(std::sqrt(m));
  return matrix;
}

LibAMM::Tensor TugOfWarCPPAlgo::amm(LibAMM::Tensor A, LibAMM::Tensor B, uint64_t l2) {
  int64_t n = A.size(1);
  int64_t p = B.size(1);
  int64_t l = (int64_t) l2;

  double delta = algoDelta;

  int64_t i_iters = static_cast<int64_t>(-std::log(delta));
  int64_t j_iters = static_cast<int64_t>(2 * (-std::log(delta) + std::log(-std::log(delta))));

  LibAMM::Tensor z = LibAMM::empty({i_iters});
  std::vector<LibAMM::Tensor> AS;
  std::vector<LibAMM::Tensor> SB;

  for (int64_t i = 0; i < i_iters; ++i) {
    LibAMM::Tensor S = generateTugOfWarMatrix(l, n);
    SB.push_back(S.matmul(B));
    AS.push_back(A.matmul(S.t()));

    LibAMM::Tensor y = LibAMM::empty({j_iters});

    for (int64_t j = 0; j < j_iters; ++j) {
      LibAMM::Tensor Q = generateTugOfWarMatrix(16, p);
      LibAMM::Tensor X = A.matmul(B.matmul(Q.t()));
      LibAMM::Tensor X_hat = AS[i].matmul(SB[i].matmul(Q.t()));
      y[j] = torch::norm(X - X_hat).pow(2);
    }

    z[i] = at::median(y);
  }

  LibAMM::Tensor z_argmin = z.argmin();
  int64_t i_star = z_argmin.item<int64_t>();

  return AS[i_star].matmul(SB[i_star]);
}
} // LibAMM
