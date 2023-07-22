//
// Created by haolan on 25/6/23.
//
#include <CPPAlgos/ProductQuantizationHash.h>

int compute_hash_bucket(const std::vector<int> &split_indices,
                        const std::vector<torch::Tensor> &split_thresholds,
                        const torch::Tensor &x);
torch::Tensor AMMBench::ProductQuantizationHash::amm(torch::Tensor A, torch::Tensor B, uint64_t sketchSize) {
  const int D = A.size(1);
  int C;
  if (sketchSize < 50) C = (int) sketchSize;
  C = 10;
  const int D_c = D / C;

  torch::Tensor prototypes;

  torch::serialize::InputArchive archive;
  archive.load_from("torchscripts/prototypes.pt");
  archive.read("prototypes", prototypes);

  std::vector<torch::Tensor> A_encoded;

  std::vector<int> split_indices = {59, 54, 24, 79};
  torch::Tensor v1;
  torch::Tensor v2;
  torch::Tensor v3;
  torch::Tensor v4;

  v1 = torch::tensor({{2.4377}});
  v2 = torch::tensor({{2.0435, 0.0047}});
  v3 = torch::tensor({{2.5292, -0.2861, 0.5683, 0.5683}});
  v4 = torch::tensor({{2.3362, -0.2747, -1.0972, -1.0972, 0.4769, 0.4769, 0.4769, 0.4769}});
  std::vector<torch::Tensor> split_thresholds = {v1, v2, v3, v4};

  for (int i = 0; i < A.size(0); ++i) {
    torch::Tensor a = A[i];
    std::vector<torch::Tensor> a_encoded;
    for (int c = 0; c < C; ++c) {
      //auto prototypes_c = prototypes[c];
      auto a_subvector = a.slice(0, c * D_c, (c + 1) * D_c);

      auto closest_prototype_index = compute_hash_bucket(split_indices, split_thresholds, a_subvector);

      a_encoded.push_back(torch::tensor(closest_prototype_index).unsqueeze(0));
    }
    A_encoded.push_back(torch::cat(a_encoded));
  }
  torch::Tensor A_encoded_tensor = torch::stack(A_encoded);

  std::vector<torch::Tensor> tables;

  for (int c = 0; c < C; ++c) {
    auto prototypes_c = prototypes[c];
    auto B_subspace = B.slice(0, c * D_c, (c + 1) * D_c);

    std::vector<torch::Tensor> table_c;
    for (int i = 0; i < prototypes_c.size(0); ++i) {
      auto prototype = prototypes_c[i];
      auto dot_products = prototype.matmul(B_subspace);
      table_c.push_back(dot_products);
    }
    tables.push_back(torch::stack(table_c));
  }

  std::vector<torch::Tensor> result;

  for (int i = 0; i < A_encoded_tensor.size(0); ++i) {
    auto a_encoded = A_encoded_tensor[i];
    auto row_sum = torch::zeros({B.size(1)});
    for (int c = 0; c < C; ++c) {
      int prototype_index = a_encoded[c].item<int>();
      auto table_c = tables[c];
      auto dot_products = table_c[prototype_index];
      row_sum += dot_products;
    }
    result.push_back(row_sum);
  }

  return torch::stack(result);
}

int compute_hash_bucket(const std::vector<int> &split_indices,
                        const std::vector<torch::Tensor> &split_thresholds,
                        const torch::Tensor &x) {
  int i = 0;
  for (int t = 0; t < 4; t++) {
    int j_t = split_indices[t];
    torch::Tensor v_t = split_thresholds[t];
    float v = v_t[0][i].item<float>();
    int b = (x[j_t].item<float>() >= v) ? 1 : 0;
    i = 2 * i - 1 + b;
  }
  return i;
}