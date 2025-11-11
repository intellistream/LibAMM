/*! \file EigenTensor.h
 * Eigen-based Tensor wrapper to replace PyTorch tensors
 * Provides a PyTorch-like API using Eigen for linear algebra operations
 */

#ifndef LIBAMM_EIGENTENSOR_H
#define LIBAMM_EIGENTENSOR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <vector>
#include <random>
#include <stdexcept>

namespace LibAMM {

/**
 * @brief Tensor class wrapping Eigen::MatrixXf
 * Provides PyTorch-like API for easier migration
 */
class Tensor {
public:
    using MatrixType = Eigen::MatrixXf;
    using Scalar = float;

private:
    std::shared_ptr<MatrixType> data_;
    bool is_cuda_ = false;  // Track if tensor is on CUDA (for compatibility, always false in Eigen version)

public:
    // Constructors
    Tensor() : data_(std::make_shared<MatrixType>()) {}
    
    explicit Tensor(const MatrixType& mat) 
        : data_(std::make_shared<MatrixType>(mat)) {}
    
    Tensor(int rows, int cols) 
        : data_(std::make_shared<MatrixType>(rows, cols)) {}

    // Shape queries (PyTorch-like API)
    int size(int dim) const {
        if (dim == 0) return data_->rows();
        if (dim == 1) return data_->cols();
        throw std::out_of_range("Dimension out of range");
    }

    int rows() const { return data_->rows(); }
    int cols() const { return data_->cols(); }
    
    std::vector<int> sizes() const {
        return {static_cast<int>(data_->rows()), static_cast<int>(data_->cols())};
    }

    // Element access
    Scalar& operator()(int i, int j) { return (*data_)(i, j); }
    const Scalar& operator()(int i, int j) const { return (*data_)(i, j); }

    // Get underlying Eigen matrix
    MatrixType& matrix() { return *data_; }
    const MatrixType& matrix() const { return *data_; }

    // Device transfer (no-op in CPU-only version, kept for API compatibility)
    Tensor to(const std::string& device) const {
        // In Eigen version, we ignore device specification
        // Could add GPU support later with CUDA-Eigen
        return *this;
    }

    // Matrix operations
    Tensor t() const {
        Tensor result;
        result.data_ = std::make_shared<MatrixType>(data_->transpose());
        return result;
    }

    Tensor transpose() const { return t(); }

    // Arithmetic operations
    Tensor operator+(const Tensor& other) const {
        Tensor result;
        result.data_ = std::make_shared<MatrixType>(*data_ + *other.data_);
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        Tensor result;
        result.data_ = std::make_shared<MatrixType>(*data_ - *other.data_);
        return result;
    }

    Tensor operator*(Scalar scalar) const {
        Tensor result;
        result.data_ = std::make_shared<MatrixType>(*data_ * scalar);
        return result;
    }

    Tensor operator/(Scalar scalar) const {
        Tensor result;
        result.data_ = std::make_shared<MatrixType>(*data_ / scalar);
        return result;
    }

    // In-place operations
    Tensor& operator+=(const Tensor& other) {
        *data_ += *other.data_;
        return *this;
    }

    Tensor& operator-=(const Tensor& other) {
        *data_ -= *other.data_;
        return *this;
    }

    // Conversion to different types
    template<typename T>
    Tensor to(T /*type*/) const {
        // For now, we only support float
        // Could add int/double support later
        return *this;
    }

    // Check if tensor is on CUDA
    bool is_cuda() const { return is_cuda_; }

    // Clone tensor
    Tensor clone() const {
        Tensor result;
        result.data_ = std::make_shared<MatrixType>(*data_);
        return result;
    }
};

// Type aliases for compatibility
enum DeviceType {
    kCPU,
    kCUDA
};

using Scalar = Tensor::Scalar;

// Factory functions (PyTorch-like API)

/**
 * @brief Create tensor filled with zeros
 */
inline Tensor zeros(const std::vector<int>& sizes) {
    if (sizes.size() != 2) {
        throw std::invalid_argument("Only 2D tensors supported");
    }
    Tensor result(sizes[0], sizes[1]);
    result.matrix().setZero();
    return result;
}

/**
 * @brief Create tensor filled with ones
 */
inline Tensor ones(const std::vector<int>& sizes) {
    if (sizes.size() != 2) {
        throw std::invalid_argument("Only 2D tensors supported");
    }
    Tensor result(sizes[0], sizes[1]);
    result.matrix().setOnes();
    return result;
}

/**
 * @brief Create tensor filled with random values from uniform distribution [0, 1)
 */
inline Tensor rand(const std::vector<int>& sizes) {
    if (sizes.size() != 2) {
        throw std::invalid_argument("Only 2D tensors supported");
    }
    Tensor result(sizes[0], sizes[1]);
    result.matrix().setRandom();
    // Eigen's setRandom() gives [-1, 1], convert to [0, 1)
    result.matrix() = (result.matrix().array() + 1.0f) / 2.0f;
    return result;
}

/**
 * @brief Create tensor filled with random integers
 */
inline Tensor randint(int high, const std::vector<int>& sizes) {
    if (sizes.size() != 1) {
        throw std::invalid_argument("randint only supports 1D tensors in current implementation");
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, high - 1);
    
    Tensor result(sizes[0], 1);
    for (int i = 0; i < sizes[0]; ++i) {
        result.matrix()(i, 0) = static_cast<float>(dis(gen));
    }
    return result;
}

/**
 * @brief Create 1D tensor with evenly spaced values
 */
inline Tensor arange(int end, const std::string& dtype = "float32") {
    Tensor result(end, 1);
    for (int i = 0; i < end; ++i) {
        result.matrix()(i, 0) = static_cast<float>(i);
    }
    return result;
}

/**
 * @brief Create diagonal matrix
 */
inline Tensor diag(const Tensor& input) {
    int n = input.rows();
    Tensor result(n, n);
    result.matrix().setZero();
    
    if (input.cols() == 1) {
        // Create diagonal matrix from vector
        for (int i = 0; i < n; ++i) {
            result.matrix()(i, i) = input.matrix()(i, 0);
        }
    } else {
        throw std::invalid_argument("diag expects 1D tensor (nx1 matrix)");
    }
    return result;
}

/**
 * @brief Matrix multiplication
 */
inline Tensor matmul(const Tensor& a, const Tensor& b) {
    Tensor result;
    result.matrix() = a.matrix() * b.matrix();
    return result;
}

/**
 * @brief Element-wise square root
 */
inline Tensor sqrt(const Tensor& input) {
    Tensor result;
    result.matrix() = input.matrix().array().sqrt();
    return result;
}

/**
 * @brief Element-wise exponential
 */
inline Tensor exp(const Tensor& input) {
    Tensor result;
    result.matrix() = input.matrix().array().exp();
    return result;
}

/**
 * @brief Clamp tensor values to range [min, max]
 */
inline Tensor clamp(const Tensor& input, Scalar min_val, Scalar max_val = std::numeric_limits<Scalar>::max()) {
    Tensor result;
    result.matrix() = input.matrix().cwiseMax(min_val).cwiseMin(max_val);
    return result;
}

/**
 * @brief Create tensor from single value
 */
inline Tensor tensor(Scalar value) {
    Tensor result(1, 1);
    result.matrix()(0, 0) = value;
    return result;
}

/**
 * @brief Create tensor from vector
 */
inline Tensor tensor(const std::vector<Scalar>& values) {
    Tensor result(values.size(), 1);
    for (size_t i = 0; i < values.size(); ++i) {
        result.matrix()(i, 0) = values[i];
    }
    return result;
}

/**
 * @brief Element-wise power
 */
inline Tensor pow(const Tensor& base, Scalar exponent) {
    Tensor result;
    result.matrix() = base.matrix().array().pow(exponent);
    return result;
}

/**
 * @brief Sum all elements in tensor
 */
inline Scalar sum(const Tensor& input) {
    return input.matrix().sum();
}

/**
 * @brief Set random seed (no-op in current implementation, uses thread-local random)
 */
inline void manual_seed(unsigned int seed) {
    std::srand(seed);
}

/**
 * @brief Create empty tensor (uninitialized)
 */
inline Tensor empty(const std::vector<int>& sizes) {
    if (sizes.size() != 2) {
        throw std::invalid_argument("Only 2D tensors supported");
    }
    Tensor result(sizes[0], sizes[1]);
    // Leave uninitialized for performance
    return result;
}

/**
 * @brief Create tensor from raw data pointer
 * @note This creates a copy of the data
 */
inline Tensor from_blob(const void* data, const std::vector<int64_t>& sizes, const std::string& /*options*/ = "") {
    if (sizes.size() != 2) {
        throw std::invalid_argument("Only 2D tensors supported");
    }
    Tensor result(sizes[0], sizes[1]);
    const float* src = static_cast<const float*>(data);
    std::copy(src, src + sizes[0] * sizes[1], result.matrix().data());
    return result;
}

/**
 * @brief Exponential distribution sampling
 */
inline Tensor exponential(const Tensor& shape_tensor, Scalar lambda = 1.0) {
    int rows = shape_tensor.rows();
    int cols = shape_tensor.cols();
    
    Tensor result(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::exponential_distribution<> dist(lambda);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.matrix()(i, j) = dist(gen);
        }
    }
    return result;
}

/**
 * @brief Multinomial sampling (simplified version)
 */
inline Tensor multinomial(const Tensor& probs, int num_samples, bool replacement = true) {
    // Simplified implementation: sample according to probabilities
    int n = probs.rows();
    Tensor result(num_samples, 1);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.matrix().data(), probs.matrix().data() + n);
    
    for (int i = 0; i < num_samples; ++i) {
        result.matrix()(i, 0) = static_cast<float>(dist(gen));
    }
    return result;
}

/**
 * @brief Type conversion to Float64 (double)
 */
inline Tensor toFloat64(const Tensor& input) {
    // For now, keep as float32 since our Tensor uses float
    // Could be extended to support double precision
    return input;
}

/**
 * @brief Type conversion to Int
 */
inline Tensor toInt(const Tensor& input) {
    Tensor result;
    result.matrix() = input.matrix().cast<int>().cast<float>();
    return result;
}

/**
 * @brief Element-wise exponential (alias for exp)
 */
inline Tensor exponential(Tensor&& input) {
    return exp(input);
}

// Type constants for compatibility
namespace kFloat64 {
    inline std::string value() { return "float64"; }
}

namespace kFloat32 {
    inline std::string value() { return "float32"; }
}

namespace kInt {
    inline std::string value() { return "int"; }
}

} // namespace LibAMM

// Namespace alias for easier migration from torch::
namespace torch = LibAMM;

#endif // LIBAMM_EIGENTENSOR_H
