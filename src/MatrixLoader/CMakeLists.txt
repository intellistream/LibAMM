add_sources(
        AbstractMatrixLoader.cpp
        RandomMatrixLoader.cpp
        # TODO: These matrix loaders require advanced PyTorch features
        # SparseMatrixLoader.cpp      # Requires: 2D indexing mat[i][j]
        # GaussianMatrixLoader.cpp    # Requires: randn()
        # ExponentialMatrixLoader.cpp # Works but minor issues
        # BinomialMatrixLoader.cpp    # Requires: operator<(Tensor, Scalar)
        # PoissonMatrixLoader.cpp     # Requires: torch::poisson()
        # BetaMatrixLoader.cpp        # Requires: randn()
        # SIFTMatrixLoader.cpp        # Requires: TensorOptions, mean(), std(), contiguous()
        # MNISTMatrixLoader.cpp       # Requires: from_file reading
        # MediaMillMatrixLoader.cpp   # Requires: file I/O operations
        # CCAMatrixLoader.cpp         # Requires: linalg::eig, linalg::svd, full(), at::real()
        # MtxMatrixLoader.cpp         # Requires: file I/O
        MatrixLoaderTable.cpp
        # ZeroMaskedMatrixLoader.cpp  # Requires: bernoulli
        # ZipfMatrixLoader.cpp        # Requires: arange with 3 params, view(), max()
)