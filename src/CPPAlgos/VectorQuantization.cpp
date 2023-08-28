//
// Created by yuhao on 16/8/23.
//
#include <CPPAlgos/VectorQuantization.h>
#include <regex>

using namespace std;

void AMMBench::VectorQuantization::setConfig(INTELLI::ConfigMapPtr cfg) {
    pqvqCodewordLookUpTablePath = cfg->tryString("pqvqCodewordLookUpTablePath", "", true);
    INTELLI_INFO("pqvqCodewordLookUpTablePath: " + pqvqCodewordLookUpTablePath);
    assert(pqvqCodewordLookUpTablePath!="");

    // Use regex to find the number after '_m', meaning the number of subspace
    std::regex pattern("_m(\\d+)_");
    std::smatch match;

    if (std::regex_search(pqvqCodewordLookUpTablePath, match, pattern)) {
        std::string numberStr = match[1].str();
        m = std::stoi(numberStr); // Convert string to integer
        std::cout << "VQ/PQ num of subspaces: " << m << std::endl;
    } else {
        std::cout << "Number not found." << std::endl;
    }

    // make sure vq _m1 and pq _m10 (not restricted to 10, but any number>1, 10 is an example)
    string algo = cfg->tryString("cppAlgoTag", "vq", true);
    assert ((algo=="vq" && m==1) || (algo=="pq" && m>1));
    }

torch::Tensor AMMBench::VectorQuantization::amm(const torch::Tensor A, const torch::Tensor B, uint64_t l) {
    
    l=0;
    int A_rows = A.sizes()[0];
    int B_cols = B.sizes()[1];

    // INTELLI_INFO("Load start");
    torch::jit::script::Module tensors = torch::jit::load(pqvqCodewordLookUpTablePath);
    codewordsA = tensors.attr("codewordsA").toTensor();
    codewordsB = tensors.attr("codewordsB").toTensor();
    lookUpTable = tensors.attr("lookUpTable").toTensor();
    // INTELLI_INFO("Load end");
    
    int CA = codewordsA.sizes()[2];
    int CB = codewordsB.sizes()[2];

    // quantization
    // Initialize vectors to store quantized indices for matrices A and B
    std::vector<torch::Tensor> A_quantized;
    std::vector<torch::Tensor> B_quantized;

    // INTELLI_INFO("Quantize A start");
    // Find the nearest codeword indices for matrix A
    for (int i = 0; i < m; ++i) {
        torch::Tensor codewords_a = codewordsA[i]; // codewordsA already computed
        torch::Tensor A_subspace = A.slice(1, i * CA, (i + 1) * CA);

        // Compute distances
        torch::Tensor distances = torch::norm(codewords_a.unsqueeze(0) - A_subspace.unsqueeze(1), 2, 2);
        torch::Tensor closest_codeword_indices = torch::argmin(distances, 1);
        A_quantized.push_back(closest_codeword_indices);
    }

    // Convert vectors to tensors for matrices A and B
    torch::Tensor A_quantized_tensor = torch::stack(A_quantized, 1);
    // std::cout << "A_quantized_tensor shape: " << A_quantized_tensor.sizes() << std::endl;
    // INTELLI_INFO("Quantize A end");

    // INTELLI_INFO("Quantize B start");
    // Find the nearest codeword indices for matrix B
    for (int k = 0; k < m; ++k) {
        torch::Tensor codewords_b = codewordsB[k]; // codewordsB already computed
        torch::Tensor B_subspace = B.slice(0, k * CB, (k + 1) * CB).t(); // Transpose for proper dimensions

        // Compute distances
        torch::Tensor distances = torch::norm(codewords_b.unsqueeze(0) - B_subspace.unsqueeze(1), 2, 2);
        torch::Tensor closest_codeword_indices = torch::argmin(distances, 1);
        B_quantized.push_back(closest_codeword_indices);
    }

    // Convert vectors to tensors for matrices A and B
    torch::Tensor B_quantized_tensor = torch::stack(B_quantized, 0);
    // std::cout << "B_quantized_tensor shape: " << B_quantized_tensor.sizes() << std::endl;
    // INTELLI_INFO("Quantize B end");

    // Define the batch size for batch processing
    int batch_size_A = A_rows;
    int batch_size_B = B_cols;

    // Initialize the matrix products
    torch::Tensor matrix_products = torch::zeros({A_rows, B_cols});

    // Perform matrix multiplication using batch processing
    // INTELLI_INFO("Lookup table start");
    for (int i = 0; i < A_rows; i += batch_size_A) {
        for (int j = 0; j < B_cols; j += batch_size_B) {
            torch::Tensor batch_result = torch::zeros({batch_size_A, batch_size_B});
            for (int k = 0; k < m; ++k) {
                // Gather quantized indices for the current batch
                torch::Tensor A_indices = A_quantized_tensor.slice(0, i, i + batch_size_A).select(1, k);
                torch::Tensor B_indices = B_quantized_tensor.slice(1, j, j + batch_size_B)[k];
                // Broadcast batch indices to match the shape of B_indices
                auto batch_indices = A_indices.unsqueeze(1).expand({-1, B_indices.sizes()[0]});
                // Gather elements from A using batch and B_indices
                torch::Tensor batch_lookup = lookUpTable[k].index({batch_indices, B_indices});
                // Accumulate the batch result
                batch_result += batch_lookup;
            }
            
            // Assign the batch result to the corresponding position in the matrix products
            matrix_products.slice(0, i, i + batch_size_A)
                        .slice(1, j, j + batch_size_B)
                        .copy_(batch_result);
        }
    }
    // INTELLI_INFO("Lookup table end");
    return matrix_products;
}
