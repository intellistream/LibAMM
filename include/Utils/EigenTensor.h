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
#include <map>
#include <tuple>

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
    
    // PyTorch-compatible dimension methods
    int ndimension() const { return 2; } // Eigen matrices are always 2D
    int dim() const { return 2; } // Alias for ndimension()

    // Element access
    Scalar& operator()(int i, int j) { return (*data_)(i, j); }
    const Scalar& operator()(int i, int j) const { return (*data_)(i, j); }

    // Get underlying Eigen matrix
    MatrixType& matrix() { return *data_; }
    const MatrixType& matrix() const { return *data_; }

    // Device transfer (no-op in CPU-only version, kept for API compatibility)
    Tensor to(const std::string& /*device*/) const {
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

    // Element-wise multiplication operator
    Tensor operator*(const Tensor& other) const {
        return mul(other);
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

    Tensor& operator/=(const Tensor& other) {
        data_->array() /= other.data_->array();
        return *this;
    }

    Tensor& operator/=(Scalar scalar) {
        *data_ /= scalar;
        return *this;
    }

    Tensor& operator*=(Scalar scalar) {
        *data_ *= scalar;
        return *this;
    }

    // Element-wise multiplication (Hadamard product)
    Tensor mul(const Tensor& other) const {
        Tensor result;
        result.data_ = std::make_shared<MatrixType>(data_->array() * other.data_->array());
        return result;
    }

    // Indexing operator for 1D tensors
    Scalar& operator[](int64_t idx) {
        if (data_->cols() == 1) {
            return (*data_)(idx, 0);
        } else if (data_->rows() == 1) {
            return (*data_)(0, idx);
        } else {
            throw std::runtime_error("operator[] only works for 1D tensors");
        }
    }

    const Scalar& operator[](int64_t idx) const {
        if (data_->cols() == 1) {
            return (*data_)(idx, 0);
        } else if (data_->rows() == 1) {
            return (*data_)(0, idx);
        } else {
            throw std::runtime_error("operator[] only works for 1D tensors");
        }
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

    // Element-wise division
    Tensor div(const Tensor& other) const {
        Tensor result;
        result.data_ = std::make_shared<MatrixType>(data_->array() / other.data_->array());
        return result;
    }

    Tensor div(Scalar scalar) const {
        return *this / scalar;
    }

    // Sum methods
    Tensor sum() const {
        // Return a 1x1 tensor containing the sum
        Tensor result(1, 1);
        result.matrix()(0, 0) = data_->sum();
        return result;
    }

    Tensor sum(int dim) const {
        Tensor result;
        if (dim == 0) {
            // Sum along rows (result is 1 x cols)
            result.data_ = std::make_shared<MatrixType>(data_->colwise().sum());
        } else if (dim == 1) {
            // Sum along columns (result is rows x 1)
            result.data_ = std::make_shared<MatrixType>(data_->rowwise().sum());
        } else {
            throw std::invalid_argument("Invalid dimension for sum");
        }
        return result;
    }

    // Norm (Frobenius norm by default, or p-norm)
    Scalar norm() const {
        return data_->norm();
    }

    Scalar norm(double p) const {
        if (p == 2.0) {
            return data_->norm();
        } else if (p == 1.0) {
            return data_->lpNorm<1>();
        } else if (std::isinf(p)) {
            return data_->lpNorm<Eigen::Infinity>();
        } else {
            // General p-norm (approximate using element-wise power)
            return std::pow(data_->array().abs().pow(p).sum(), 1.0 / p);
        }
    }

    // Slice operation: extract sub-tensor
    Tensor slice(int dim, int start, int end, int step = 1) const {
        Tensor result;
        if (dim == 0) {
            // Slice rows
            int count = (end - start + step - 1) / step;
            result.data_ = std::make_shared<MatrixType>(count, data_->cols());
            for (int i = 0; i < count; ++i) {
                result.data_->row(i) = data_->row(start + i * step);
            }
        } else if (dim == 1) {
            // Slice columns
            int count = (end - start + step - 1) / step;
            result.data_ = std::make_shared<MatrixType>(data_->rows(), count);
            for (int i = 0; i < count; ++i) {
                result.data_->col(i) = data_->col(start + i * step);
            }
        } else {
            throw std::invalid_argument("Invalid dimension for slice");
        }
        return result;
    }

    // Index select: select rows or columns by indices
    Tensor index_select(int dim, const Tensor& indices) const {
        Tensor result;
        if (dim == 0) {
            // Select rows
            result.data_ = std::make_shared<MatrixType>(indices.rows(), data_->cols());
            for (int i = 0; i < indices.rows(); ++i) {
                int idx = static_cast<int>(indices.matrix()(i, 0));
                result.data_->row(i) = data_->row(idx);
            }
        } else if (dim == 1) {
            // Select columns
            result.data_ = std::make_shared<MatrixType>(data_->rows(), indices.rows());
            for (int i = 0; i < indices.rows(); ++i) {
                int idx = static_cast<int>(indices.matrix()(i, 0));
                result.data_->col(i) = data_->col(idx);
            }
        } else {
            throw std::invalid_argument("Invalid dimension for index_select");
        }
        return result;
    }

    // Comparison operators
    Tensor operator<(const Tensor& other) const {
        Tensor result;
        result.data_ = std::make_shared<MatrixType>(
            (data_->array() < other.data_->array()).cast<Scalar>()
        );
        return result;
    }

    Tensor operator>(const Tensor& other) const {
        Tensor result;
        result.data_ = std::make_shared<MatrixType>(
            (data_->array() > other.data_->array()).cast<Scalar>()
        );
        return result;
    }

    Tensor operator<=(const Tensor& other) const {
        Tensor result;
        result.data_ = std::make_shared<MatrixType>(
            (data_->array() <= other.data_->array()).cast<Scalar>()
        );
        return result;
    }

    Tensor operator>=(const Tensor& other) const {
        Tensor result;
        result.data_ = std::make_shared<MatrixType>(
            (data_->array() >= other.data_->array()).cast<Scalar>()
        );
        return result;
    }

    // Number of elements
    int64_t numel() const {
        return data_->rows() * data_->cols();
    }

    // Raw data pointer access
    template<typename T = Scalar>
    T* data_ptr() {
        return reinterpret_cast<T*>(data_->data());
    }

    template<typename T = Scalar>
    const T* data_ptr() const {
        return reinterpret_cast<const T*>(data_->data());
    }

    // Reshape tensor
    Tensor reshape(const std::vector<int>& new_shape) const {
        if (new_shape.size() != 2) {
            throw std::invalid_argument("Only 2D reshaping supported");
        }
        int total = data_->rows() * data_->cols();
        int new_total = new_shape[0] * new_shape[1];
        if (total != new_total) {
            throw std::invalid_argument("Reshape: total number of elements must match");
        }
        
        Tensor result(new_shape[0], new_shape[1]);
        // Copy data in row-major order
        Eigen::Map<Eigen::VectorXf>(result.data_->data(), new_total) = 
            Eigen::Map<const Eigen::VectorXf>(data_->data(), total);
        return result;
    }

    // Item: get single scalar value (for 1x1 tensors)
    Scalar item() const {
        if (data_->rows() != 1 || data_->cols() != 1) {
            throw std::runtime_error("item() only works for 1x1 tensors");
        }
        return (*data_)(0, 0);
    }

    // Template version for type conversion
    template<typename T>
    T item() const {
        return static_cast<T>(item());
    }

    // copy_: in-place copy (PyTorch-like)
    Tensor& copy_(const Tensor& other) {
        *data_ = *other.data_;
        return *this;
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

inline Tensor zeros(int size) {
    Tensor result(size, 1);
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

inline Tensor ones(int size) {
    Tensor result(size, 1);
    result.matrix().setOnes();
    return result;
}

inline Tensor ones(int rows, DeviceType /*device*/) {
    // Ignore device parameter (always CPU in Eigen version)
    return ones(rows);
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
inline Tensor arange(int end, const std::string& /*dtype*/ = "float32") {
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
 * @brief Concatenate tensors along a dimension
 * @param tensors Vector of tensors to concatenate
 * @param dim Dimension to concatenate along (0=rows, 1=cols)
 */
inline Tensor cat(const std::vector<Tensor>& tensors, int dim = 0) {
    if (tensors.empty()) {
        throw std::invalid_argument("Cannot concatenate empty tensor list");
    }
    
    if (dim == 0) {
        // Concatenate along rows
        int total_rows = 0;
        int cols = tensors[0].cols();
        for (const auto& t : tensors) {
            if (t.cols() != cols) {
                throw std::invalid_argument("All tensors must have same number of columns");
            }
            total_rows += t.rows();
        }
        
        Tensor result(total_rows, cols);
        int current_row = 0;
        for (const auto& t : tensors) {
            result.matrix().block(current_row, 0, t.rows(), cols) = t.matrix();
            current_row += t.rows();
        }
        return result;
    } else if (dim == 1) {
        // Concatenate along columns
        int rows = tensors[0].rows();
        int total_cols = 0;
        for (const auto& t : tensors) {
            if (t.rows() != rows) {
                throw std::invalid_argument("All tensors must have same number of rows");
            }
            total_cols += t.cols();
        }
        
        Tensor result(rows, total_cols);
        int current_col = 0;
        for (const auto& t : tensors) {
            result.matrix().block(0, current_col, rows, t.cols()) = t.matrix();
            current_col += t.cols();
        }
        return result;
    } else {
        throw std::invalid_argument("Invalid dimension for cat");
    }
}

/**
 * @brief Scalar division (scalar / Tensor)
 */
inline Tensor operator/(Scalar scalar, const Tensor& tensor) {
    Tensor result;
    result.matrix() = scalar / tensor.matrix().array();
    return result;
}

/**
 * @brief Scalar multiplication (scalar * Tensor)
 */
inline Tensor operator*(Scalar scalar, const Tensor& tensor) {
    return tensor * scalar;
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
 * @brief Calculate norm (as a function, returns Tensor for compatibility)
 */
inline Tensor norm(const Tensor& input) {
    Tensor result(1, 1);
    result.matrix()(0, 0) = input.norm();
    return result;
}

inline Tensor norm(const Tensor& input, double p) {
    Tensor result(1, 1);
    result.matrix()(0, 0) = input.norm(p);
    return result;
}

/**
 * @brief Calculate norm along a dimension
 * @param input Input tensor
 * @param p Norm type (1, 2, inf)
 * @param dim Dimension to reduce (0=rows, 1=cols)
 */
inline Tensor norm(const Tensor& input, double p, int dim) {
    Tensor result;
    if (dim == 0) {
        // Norm along rows (result is 1 x cols)
        result = Tensor(1, input.cols());
        for (int j = 0; j < input.cols(); ++j) {
            if (p == 2.0) {
                result.matrix()(0, j) = input.matrix().col(j).norm();
            } else if (p == 1.0) {
                result.matrix()(0, j) = input.matrix().col(j).lpNorm<1>();
            } else {
                result.matrix()(0, j) = std::pow(input.matrix().col(j).array().abs().pow(p).sum(), 1.0/p);
            }
        }
    } else if (dim == 1) {
        // Norm along columns (result is rows x 1)
        result = Tensor(input.rows(), 1);
        for (int i = 0; i < input.rows(); ++i) {
            if (p == 2.0) {
                result.matrix()(i, 0) = input.matrix().row(i).norm();
            } else if (p == 1.0) {
                result.matrix()(i, 0) = input.matrix().row(i).lpNorm<1>();
            } else {
                result.matrix()(i, 0) = std::pow(input.matrix().row(i).array().abs().pow(p).sum(), 1.0/p);
            }
        }
    } else {
        throw std::invalid_argument("Invalid dimension for norm");
    }
    return result;
}

/**
 * @brief Element-wise multiplication (Hadamard product)
 */
inline Tensor mul(const Tensor& a, const Tensor& b) {
    return a.mul(b);
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
inline Tensor multinomial(const Tensor& probs, int num_samples, bool /*replacement*/ = true) {
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
 * @brief Find unique elements and optionally return counts
 * Simplified implementation of PyTorch's _unique2
 * @return tuple of (unique_values, inverse_indices, counts)
 */
inline std::tuple<Tensor, Tensor, Tensor> _unique2(
    const Tensor& input, 
    bool /*sorted*/ = true, 
    bool /*return_inverse*/ = false, 
    bool return_counts = false) {
    
    // Simplified implementation: assumes 1D input
    std::map<int, int> value_count;
    std::vector<int> unique_values;
    
    for (int i = 0; i < input.rows(); ++i) {
        int val = static_cast<int>(input.matrix()(i, 0));
        if (value_count.find(val) == value_count.end()) {
            unique_values.push_back(val);
            value_count[val] = 0;
        }
        value_count[val]++;
    }
    
    // Create unique tensor
    Tensor unique(unique_values.size(), 1);
    for (size_t i = 0; i < unique_values.size(); ++i) {
        unique.matrix()(i, 0) = static_cast<float>(unique_values[i]);
    }
    
    // Create inverse indices tensor (placeholder, not implemented)
    Tensor inverse;
    
    // Create counts tensor if requested
    Tensor counts;
    if (return_counts) {
        counts = Tensor(unique_values.size(), 1);
        for (size_t i = 0; i < unique_values.size(); ++i) {
            counts.matrix()(i, 0) = static_cast<float>(value_count[unique_values[i]]);
        }
    }
    
    return std::make_tuple(unique, inverse, counts);
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

// Stream output operator for Tensor (must be outside namespace for ADL)
inline std::ostream& operator<<(std::ostream& os, const LibAMM::Tensor& tensor) {
    os << "Tensor(shape=[";
    for (int i = 0; i < tensor.ndimension(); ++i) {
        if (i > 0) os << ", ";
        os << tensor.size(i);
    }
    os << "], data=<" << tensor.sizes()[0];
    if (tensor.ndimension() > 1) {
        os << "x" << tensor.sizes()[1];
    }
    os << " matrix>)";
    return os;
}

// Namespace aliases for easier migration from torch::
namespace at = LibAMM;

// Additional torch namespace with compatibility functions
namespace torch {
    // Import all LibAMM symbols into torch namespace
    using namespace LibAMM;
    
    // Random seed control stub (no-op in Eigen-only mode)
    inline void manual_seed(uint64_t seed) {
        // Note: Eigen uses std::rand() which can be seeded with std::srand()
        // This is a compatibility stub - actual seeding happens in rand() function
        std::srand(static_cast<unsigned int>(seed));
    }
    
    // Threading control stub (no-op in Eigen-only mode)
    inline void set_num_threads(int num_threads) {
        // Note: Eigen uses OpenMP/TBB threading automatically
        // This is a compatibility stub for PyTorch code
        (void)num_threads; // Suppress unused parameter warning
    }
    
    // TorchScript compatibility stubs
    namespace jit {
        class Module {
        public:
            Module() = default;
            
            // Stub for loading TorchScript modules
            static Module load(const std::string& filename) {
                throw std::runtime_error(
                    "torch::jit::Module is not supported in Eigen-only build. "
                    "Original file: " + filename
                );
            }
            
            // Stub for forward pass
            template<typename... Args>
            LibAMM::Tensor forward(Args&&... args) {
                throw std::runtime_error(
                    "torch::jit::Module::forward is not supported in Eigen-only build."
                );
            }
        };
        
        // torch::jit::script namespace compatibility
        namespace script {
            using Module = jit::Module;
        }
        
        // Global load function (in jit namespace)
        inline Module load(const std::string& filename) {
            return Module::load(filename);
        }
    }
}

#endif // LIBAMM_EIGENTENSOR_H
