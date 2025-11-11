//
// Created by haolan on 5/29/23.
//
#include <CPPAlgos/CoOccurringFDCPPAlgo.h>

namespace LibAMM {
LibAMM::Scalar get_first_element(const LibAMM::Tensor &tensor) {
  if (tensor.numel() == 1) {
    return tensor.item();
  } else {
    return tensor[0].item();
  }
}

bool is_empty_tensor(const LibAMM::Tensor &tensor) {
  return tensor.numel() == 0;
}

LibAMM::Tensor medianReduceRank(const LibAMM::Tensor &SV, float delta) {
  return LibAMM::clamp(SV - delta, 0);
}

LibAMM::Tensor CoOccurringFDCPPAlgo::amm(const LibAMM::Tensor A, const LibAMM::Tensor B, uint64_t l2) {
  LibAMM::Tensor B_t = B.t();

  TORCH_CHECK(A.size(1) == B_t.size(1), "Shapes of A and B are incompatible");
  int64_t mx = A.size(0);
  int64_t my = B_t.size(0);
  int64_t n = A.size(1);
  int64_t l = (int64_t) l2;
  // Initialize sketch matrices
  LibAMM::Tensor BX = LibAMM::zeros({mx, l});
  LibAMM::Tensor BY = LibAMM::zeros({my, l});

  // The first l iterations
  for (int i = 0; i < l; ++i) {
    BX.slice(1, i, i + 1) = A.slice(1, i, i + 1);
    BY.slice(1, i, i + 1) = B_t.slice(1, i, i + 1);
  }

  LibAMM::Tensor zero_columns = LibAMM::tensor({0});
  zero_columns = zero_columns.slice(0, 1);

  // Iteration l to n: insert if available, else shrink sketch matrices
  for (int i = l; i < n; ++i) {
    // Acquire the index of a zero-valued column
    if (!is_empty_tensor(zero_columns)) {
      int idx = get_first_element(zero_columns).toInt();
      BX.slice(1, idx, idx + 1) = A.slice(1, i, i + 1);
      BY.slice(1, idx, idx + 1) = B_t.slice(1, i, i + 1);
      zero_columns = zero_columns.slice(0, 1);
    }
      // If no zero-valued column, shrink accordingly
    else {
      LibAMM::Tensor QX, RX;
      std::tie(QX, RX) = torch::linalg_qr(BX);
      LibAMM::Tensor QY, RY;
      std::tie(QY, RY) = torch::linalg_qr(BY);
      LibAMM::Tensor U, SV, V;
      std::tie(U, SV, V) = torch::svd(LibAMM::matmul(RX, RY.t()));

      // Find the median of singular values
      LibAMM::Tensor S_sorted, S_indices;
      std::tie(S_sorted, S_indices) = SV.sort();

      float delta;
      if (S_sorted.size(0) % 2 == 1) {
        delta = S_sorted[S_sorted.size(0) / 2].item().toFloat();
      } else {
        delta = torch::median(S_sorted).item().toFloat();
      }
      // Shrink the singular values with delta
      LibAMM::Tensor SV_shrunk = medianReduceRank(SV, delta);

      // Restore SV diagonal matrix
      SV = LibAMM::diag_embed(SV_shrunk);
      LibAMM::Tensor SV_sqrt = LibAMM::sqrt(SV);

      // Update indices of zero-valued columns
      LibAMM::Tensor zero_indices = torch::nonzero(SV_shrunk == 0).squeeze();
      try {
        zero_columns = torch::cat({zero_columns, zero_indices});
      } catch (const c10::Error& ){
      }
      // Convert tensor to a std::vector
      std::vector<int64_t>
          vec(zero_columns.data_ptr<int64_t>(), zero_columns.data_ptr<int64_t>() + zero_columns.numel());

      // Sort the vector
      std::sort(vec.begin(), vec.end());

      // Remove duplicates
      vec.erase(std::unique(vec.begin(), vec.end()), vec.end());

      // Convert std::vector back to a tensor
      zero_columns = LibAMM::from_blob(vec.data(), {static_cast<int64_t>(vec.size())}, torch::kInt64).clone();

      // Update sketch matrices
      BX = LibAMM::matmul(LibAMM::matmul(QX, U), SV_sqrt);
      BY = LibAMM::matmul(LibAMM::matmul(QY, V), SV_sqrt);
    }
  }

  return LibAMM::matmul(BX, BY.t());
}
}