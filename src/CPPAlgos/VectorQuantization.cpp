//
// Created by yuhao on 16/8/23.
//
#include <CPPAlgos/VectorQuantization.h>
#include <regex>

using namespace std;

void LibAMM::VectorQuantization::setConfig(INTELLI::ConfigMapPtr cfg) {
    pqvqCodewordLookUpTablePath = cfg->tryString("pqvqCodewordLookUpTablePath", "", true);
    INTELLI_INFO("pqvqCodewordLookUpTablePath: " + pqvqCodewordLookUpTablePath);
    assert(pqvqCodewordLookUpTablePath!="");

    // Use regex to find the number after '_m', meaning the number of subspace
    std::regex pattern("_m(\\d+)");
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
    //assert ((algo=="vq" && m==1) || (algo=="pq" && m>1));
    }

LibAMM::Tensor LibAMM::VectorQuantization::amm(const LibAMM::Tensor A, const LibAMM::Tensor B, uint64_t l) {
    
    l=0;
    int A_rows = A.sizes()[0];
    int B_cols = B.sizes()[1];

    torch::jit::script::Module tensors = torch::jit::load(pqvqCodewordLookUpTablePath);
    codewordsA = tensors.attr("codewordsA").toTensor();
    codewordsB = tensors.attr("codewordsB").toTensor();
    lookUpTable = tensors.attr("lookUpTable").toTensor();
    
    int CA = codewordsA.sizes()[2];
    int CB = codewordsB.sizes()[2];

    // quantization
    // Initialize vectors to store quantized indices for matrices A and B
    std::vector<LibAMM::Tensor> A_quantized;
    std::vector<LibAMM::Tensor> B_quantized;
    int chunk_size = 100; // Initial chunk size
    
    // INTELLI_INFO("Quantize A start");
    // Find the nearest codeword indices for matrix A
    for (int i = 0; i < m; ++i) {
        LibAMM::Tensor codewords_a = codewordsA[i]; // codewordsA already computed
        LibAMM::Tensor A_subspace = A.slice(1, i * CA, (i + 1) * CA);
        // LibAMM::Tensor distances = torch::norm(codewords_a.unsqueeze(0) - A_subspace.unsqueeze(1), 2, 2);
        // LibAMM::Tensor closest_codeword_indices = torch::argmin(distances, 1);
        // A_quantized.push_back(closest_codeword_indices);
        std::vector<LibAMM::Tensor> closest_codeword_indices_list; // Store indices for each chunk

        for (int j = 0; j < A_subspace.size(0); j += chunk_size) {
            int chunk_end = std::min(j + chunk_size, static_cast<int>(A_subspace.size(0)));
            LibAMM::Tensor A_subspace_sub = A_subspace.slice(0, j, chunk_end);
            
            LibAMM::Tensor distances_sub = torch::norm(codewords_a.unsqueeze(0) - A_subspace_sub.unsqueeze(1), 2, 2);
            LibAMM::Tensor closest_codeword_indices_sub = torch::argmin(distances_sub, 1);
            closest_codeword_indices_list.push_back(closest_codeword_indices_sub);
        }

        LibAMM::Tensor closest_codeword_indices = torch::cat(closest_codeword_indices_list, 0);
        A_quantized.push_back(closest_codeword_indices);
    }

    // Convert vectors to tensors for matrices A and B
    LibAMM::Tensor A_quantized_tensor = torch::stack(A_quantized, 1);
    
    for (int k = 0; k < m; ++k) {
        LibAMM::Tensor codewords_b = codewordsB[k]; // codewordsB already computed
        LibAMM::Tensor B_subspace = B.slice(0, k * CB, (k + 1) * CB).t(); // Transpose for proper dimensions
        // LibAMM::Tensor distances = torch::norm(codewords_b.unsqueeze(0) - B_subspace.unsqueeze(1), 2, 2);
        // LibAMM::Tensor closest_codeword_indices = torch::argmin(distances, 1);
        // B_quantized.push_back(closest_codeword_indices);
        std::vector<LibAMM::Tensor> closest_codeword_indices_list;  // Store closest codeword indices for each chunk

        for (int j = 0; j < B_subspace.size(0); j += chunk_size) {
            int chunk_end = std::min(j + chunk_size, static_cast<int>(B_subspace.size(0)));
            LibAMM::Tensor B_subspace_sub = B_subspace.slice(0, j, chunk_end);

            LibAMM::Tensor distances_sub = torch::norm(codewords_b.unsqueeze(0) - B_subspace_sub.unsqueeze(1), 2, 2);
            LibAMM::Tensor closest_codeword_indices_sub = torch::argmin(distances_sub, 1);
            closest_codeword_indices_list.push_back(closest_codeword_indices_sub);
        }

        LibAMM::Tensor closest_codeword_indices = torch::cat(closest_codeword_indices_list, 0);
        B_quantized.push_back(closest_codeword_indices);
    }

    // Convert vectors to tensors for matrices A and B
    LibAMM::Tensor B_quantized_tensor = torch::stack(B_quantized, 0);

    // Define the batch size for batch processing
    int batch_size_A = A_rows;
    int batch_size_B = B_cols;

    // Initialize the matrix products
    LibAMM::Tensor matrix_products = LibAMM::zeros({A_rows, B_cols});

    // Perform matrix multiplication using batch processing
    for (int i = 0; i < A_rows; i += batch_size_A) {
        for (int j = 0; j < B_cols; j += batch_size_B) {
            LibAMM::Tensor batch_result = LibAMM::zeros({batch_size_A, batch_size_B});
            for (int k = 0; k < m; ++k) {
                // Gather quantized indices for the current batch
                LibAMM::Tensor A_indices = A_quantized_tensor.slice(0, i, i + batch_size_A).select(1, k);
                LibAMM::Tensor B_indices = B_quantized_tensor.slice(1, j, j + batch_size_B)[k];
                // Broadcast batch indices to match the shape of B_indices
                auto batch_indices = A_indices.unsqueeze(1).expand({-1, B_indices.sizes()[0]});
                // Gather elements from A using batch and B_indices
                LibAMM::Tensor batch_lookup = lookUpTable[k].index({batch_indices, B_indices});
                // Accumulate the batch result
                batch_result += batch_lookup;
            }
            
            // Assign the batch result to the corresponding position in the matrix products
            matrix_products.slice(0, i, i + batch_size_A)
                        .slice(1, j, j + batch_size_B)
                        .copy_(batch_result);
        }
    }
    return matrix_products;
}
