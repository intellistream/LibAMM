#include <torch/torch.h>
#include <fstream>

int main() {
  const int N = 3000;
  const int D = 1000;
  const int M = 2000;
  const int C = 10;
  const int K = 16;
  const int D_c = D / C;

  torch::Tensor A = torch::rand({N, D});
  torch::Tensor B = torch::rand({D, M});

  std::vector<torch::Tensor> prototypes;
  std::ifstream in_file("/home/haolan/PQ/prototypes.pt", std::ios::binary);
  if (!in_file.is_open()) {
    std::cerr << "Error opening file prototypes.pt\n";
    return 1;
  }
  torch::load(prototypes, in_file);

  std::vector<torch::Tensor> A_encoded;

  for (int i = 0; i < A.size(0); ++i) {
    auto a = A[i];
    std::vector<torch::Tensor> a_encoded;

    for (int c = 0; c < C; ++c) {
      torch::Tensor prototypes_c = prototypes[c];
      torch::Tensor a_subvector = a.slice(0, c * D_c, (c + 1) * D_c);

      torch::Tensor distances = torch::norm(prototypes_c - a_subvector.expand_as(prototypes_c), 1);
      torch::Tensor closest_prototype_index = torch::argmin(distances);
      a_encoded.push_back(closest_prototype_index);
    }
    A_encoded.push_back(torch::stack(a_encoded));
  }
  torch::Tensor A_encoded_tensor = torch::stack(A_encoded);

  std::vector<torch::Tensor> tables;

  for (int c = 0; c < C; ++c) {
    torch::Tensor prototypes_c = prototypes[c];
    torch::Tensor B_subspace = B.slice(0, c * D_c, (c + 1) * D_c);

    std::vector<torch::Tensor> table_c;
    for (int i = 0; i < prototypes_c.size(0); ++i) {
      auto prototype = prototypes_c[i];
      torch::Tensor dot_products = prototype.matmul(B_subspace);
      table_c.push_back(dot_products);
    }
    tables.push_back(torch::stack(table_c));
  }

  std::vector<torch::Tensor> result;

  for (int i = 0; i < A_encoded_tensor.size(0); ++i) {
    auto a_encoded = A_encoded_tensor[i];
    torch::Tensor row_sum = torch::zeros({B.size(1)});
    for (int c = 0; c < C; ++c) {
      int prototype_index = a_encoded[c].item<int>();
      torch::Tensor table_c = tables[c];
      torch::Tensor dot_products = table_c[prototype_index];
      row_sum += dot_products;
    }
    result.push_back(row_sum);
  }
  torch::Tensor result_tensor = torch::stack(result);

  return 0;
}
