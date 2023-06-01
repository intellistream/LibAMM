//
// Created by luv on 5/30/23.
//

#include <CPPAlgos/TugOfWarCPPAlgo.h>

namespace AMMBench {
    torch::Tensor TugOfWarCPPAlgo::generateTugOfWarMatrix(int64_t m, int64_t n) {
      double e = 1.0 / std::sqrt(m);
      torch::Tensor M = torch::randint(2, {m, n});
      return e * (2 * M - 1);
    }

    torch::Tensor TugOfWarCPPAlgo::amm(torch::Tensor A, torch::Tensor B, int l) {
        int n = A.size(1);
        int p = B.size(1);

        double delta = 0.2;

        int i_iters = static_cast<int>(-std::log(delta));
        int j_iters = static_cast<int>(2 * (-std::log(delta) + std::log(-std::log(delta))));

        torch::Tensor z = torch::empty({i_iters});
        std::vector<torch::Tensor> AS;
        std::vector<torch::Tensor> SB;

        for (int i = 0; i < i_iters; ++i) {
          torch::Tensor S = generateTugOfWarMatrix(l, n);
          SB.push_back(S.matmul(B));
          AS.push_back(A.matmul(S.t()));

          torch::Tensor y = torch::empty({j_iters});

          for (int j = 0; j < j_iters; ++j) {
            torch::Tensor Q = generateTugOfWarMatrix(16, p);
            torch::Tensor X = A.matmul(B.matmul(Q.t()));
            torch::Tensor X_hat = AS[i].matmul(SB[i].matmul(Q.t()));
            y[j] = torch::norm(X - X_hat).pow(2);
          }


          z[i] = at::median(y);
        }

        torch::Tensor z_argmin = z.argmin();
        int i_star = z_argmin.item<int>();

        return AS[i_star].matmul(SB[i_star]);
    }
} // AMMBench
