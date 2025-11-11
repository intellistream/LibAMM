# Quick Reference: Eigen-based LibAMM

## Build & Test

### Compile Core Library
```bash
cd build
cmake ..
make -j8
# Output: libLibAMM.so (651KB)
```

### Run Verification Tests
```bash
# Compile header-only test
g++ -std=c++20 -I./include -I./build/_deps/eigen-src test_tensor_only.cpp -o test_tensor

# Run test
./test_tensor
# Expected: All tests passed! ✅
```

---

## API Quick Reference

### Tensor Creation
```cpp
#include "Utils/EigenTensor.h"

// Basic creation
LibAMM::Tensor a = LibAMM::zeros({3, 4});    // 3x4 zero matrix
LibAMM::Tensor b = LibAMM::ones({3, 4});     // 3x4 ones matrix
LibAMM::Tensor c = LibAMM::rand({3, 4});     // 3x4 random matrix
LibAMM::Tensor d = LibAMM::empty({3, 4});    // 3x4 uninitialized

// Using torch namespace (identical API)
torch::Tensor t = torch::rand({3, 4});
```

### Matrix Operations
```cpp
// Matrix multiplication
LibAMM::Tensor c = LibAMM::matmul(a, b);

// Transpose
LibAMM::Tensor at = a.t();
LibAMM::Tensor bt = b.transpose();  // Same as t()

// Element-wise operations
LibAMM::Tensor sum = a + b;
LibAMM::Tensor diff = a - b;
LibAMM::Tensor prod = a * b;       // Element-wise (Hadamard)
LibAMM::Tensor quot = a / b;
```

### Indexing & Access
```cpp
// Get dimensions
int rows = a.size(0);              // Number of rows
int cols = a.size(1);              // Number of columns
int ndim = a.ndimension();         // Always 2 for Eigen matrices

// Element access
float val = a(0, 0);               // Get element
a(0, 0) = 3.14f;                   // Set element

// Slicing (advanced)
LibAMM::Tensor sub = a.slice(0, 0, 2);  // Rows 0-1
```

### Reduction Operations
```cpp
// Sum along dimension
LibAMM::Tensor col_sum = a.sum(0);  // Sum each column (result: 1xN)
LibAMM::Tensor row_sum = a.sum(1);  // Sum each row (result: Mx1)

// Norm
LibAMM::Tensor norm = a.norm(2, 0); // L2 norm along dim 0
```

### Shape Operations
```cpp
// Reshape
LibAMM::Tensor reshaped = a.reshape({6, 2});

// Concatenate
std::vector<LibAMM::Tensor> tensors = {a, b, c};
LibAMM::Tensor concatenated = LibAMM::cat(tensors, 0);  // Along rows

// Clone
LibAMM::Tensor copy = a.clone();
```

---

## Working Algorithms

### 1. Count-based Row Sampling (CRS)
```cpp
#include "CPPAlgos/CRSCPPAlgo.h"

LibAMM::CRSCPPAlgo crs;
auto config = std::make_shared<INTELLI::ConfigMap>();
config->edit("k", (int64_t)10);  // Number of samples
crs.setConfig(config);

LibAMM::Tensor A = LibAMM::rand({100, 50});
LibAMM::Tensor B = LibAMM::rand({50, 30});
LibAMM::Tensor sketch = crs.amm(A, B, 100);  // Approximate A*B
```

### 2. Weighted Count Row Sampling
```cpp
#include "CPPAlgos/WeightedCRCPPAlgo.h"

LibAMM::WeightedCRCPPAlgo wcr;
auto config = std::make_shared<INTELLI::ConfigMap>();
config->edit("k", (int64_t)10);
wcr.setConfig(config);

LibAMM::Tensor sketch = wcr.amm(A, B, 100);
```

### 3. Count Row Sampling V2
```cpp
#include "CPPAlgos/CRSV2CPPAlgo.h"

LibAMM::CRSV2CPPAlgo crsv2;
auto config = std::make_shared<INTELLI::ConfigMap>();
config->edit("k", (int64_t)10);
crsv2.setConfig(config);

LibAMM::Tensor sketch = crsv2.amm(A, B, 100);
```

### 4. Block Count Row Sampling
```cpp
#include "CPPAlgos/BCRSCPPAlgo.h"

LibAMM::BCRSCPPAlgo bcrs;
auto config = std::make_shared<INTELLI::ConfigMap>();
config->edit("k", (int64_t)10);
bcrs.setConfig(config);

LibAMM::Tensor sketch = bcrs.amm(A, B, 100);
```

---

## Matrix Loaders

### Random Matrix Loader
```cpp
#include "MatrixLoader/RandomMatrixLoader.h"

auto loader = std::make_shared<LibAMM::RandomMatrixLoader>();
auto config = std::make_shared<INTELLI::ConfigMap>();
config->edit("m", (int64_t)100);  // Rows
config->edit("n", (int64_t)50);   // Columns
loader->setConfig(config);

LibAMM::Tensor A = loader->getA();
LibAMM::Tensor B = loader->getB();
```

---

## Compatibility Stubs

### Threading Control
```cpp
// Set number of threads (no-op, Eigen uses OpenMP/TBB automatically)
torch::set_num_threads(4);
```

### TorchScript (Not Supported)
```cpp
// These will throw std::runtime_error
torch::jit::Module module = torch::jit::load("model.pt");  // ❌ Throws
torch::jit::script::Module m2;                             // ❌ Throws when used
```

---

## Debugging & Output

### Print Tensors
```cpp
LibAMM::Tensor a = LibAMM::rand({3, 4});
std::cout << "a = " << a << std::endl;
// Output: Tensor(shape=[3, 4], data=<3x4 matrix>)
```

### Get Shape Info
```cpp
std::vector<int> shape = a.sizes();
std::cout << "Shape: " << shape[0] << "x" << shape[1] << std::endl;
```

---

## Common Pitfalls

### ❌ Don't Use
```cpp
// These are NOT implemented:
torch::linalg::svd(a);           // SVD not available
torch::multinomial(a, 10);       // Sampling not available
torch::randn({3, 4});            // Normal distribution not available
torch::mean(a);                  // Statistical functions not available
torch::jit::load("model.pt");    // TorchScript not supported
```

### ✅ Use Instead
```cpp
// Use available operations:
LibAMM::matmul(a, b);            // ✅ Matrix multiplication
a.t();                           // ✅ Transpose
a + b;                           // ✅ Element-wise addition
a.sum(0);                        // ✅ Sum along dimension
LibAMM::rand({3, 4});            // ✅ Uniform random
```

---

## File Structure

```
libamm/
├── include/
│   └── Utils/
│       └── EigenTensor.h          # Main Tensor API (897 lines)
├── src/
│   ├── CPPAlgos/
│   │   ├── CRSCPPAlgo.cpp         # ✅ Working
│   │   ├── WeightedCRCPPAlgo.cpp  # ✅ Working
│   │   ├── CRSV2CPPAlgo.cpp       # ✅ Working
│   │   └── BCRSCPPAlgo.cpp        # ✅ Working
│   ├── MatrixLoader/
│   │   └── RandomMatrixLoader.cpp # ✅ Working
│   └── Utils/
│       └── UtilityFunctions.cpp   # Modified (AT_ERROR → runtime_error)
├── build/
│   └── libLibAMM.so               # Output library (651KB)
├── test_tensor_only.cpp           # Verification test ✅
└── MIGRATION_STATUS.md            # Detailed report
```

---

## Performance Notes

- **Eigen** automatically uses **OpenMP** or **TBB** for parallelization
- No explicit threading needed (unlike PyTorch's `set_num_threads`)
- Performance comparable to PyTorch for basic operations
- **SIMD** optimizations enabled by default

---

## Troubleshooting

### Compilation Error: "undefined reference to vtable"
**Problem**: Disabled algorithms/loaders referenced in tables  
**Solution**: Don't link against disabled components, or implement them

### Runtime Error: "torch::jit::Module is not supported"
**Problem**: Code tries to load TorchScript models  
**Solution**: Use Eigen-based models only, or implement jit::load

### Missing Function: "torch::randn not found"
**Problem**: Normal distribution not implemented  
**Solution**: Use `LibAMM::rand()` or implement randn using `std::normal_distribution`

---

## Quick Start Example

```cpp
#include "Utils/EigenTensor.h"
#include "CPPAlgos/CRSCPPAlgo.h"
#include <iostream>

int main() {
    // Create random matrices
    torch::Tensor A = torch::rand({100, 50});
    torch::Tensor B = torch::rand({50, 30});
    
    // Initialize CRS algorithm
    LibAMM::CRSCPPAlgo crs;
    auto config = std::make_shared<INTELLI::ConfigMap>();
    config->edit("k", (int64_t)10);
    crs.setConfig(config);
    
    // Compute approximate matrix multiplication
    torch::Tensor result = crs.amm(A, B, 100);
    
    std::cout << "Result shape: " 
              << result.size(0) << "x" << result.size(1) 
              << std::endl;
    
    return 0;
}
```

**Compile**:
```bash
g++ -std=c++20 -I./include -I./build/_deps/eigen-src \
    example.cpp -L./build -lLibAMM -o example
```

**Run**:
```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH ./example
```

---

**Version**: Eigen 3.4.0  
**C++ Standard**: C++20  
**Library Size**: 651KB  
**Status**: ✅ MVP Complete
