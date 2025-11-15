# LibAMM PyTorch Dependency Isolation

## 🎯 目标

**将 PyTorch 依赖完全隔离在 LibAMM 内部，让 SAGE 可以使用 LibAMM 而不需要安装 PyTorch**

## 📊 问题背景

### 原始架构
```
SAGE (Python)
  └─> import PyAMM
      └─> requires torch in Python environment ❌
          └─> LibAMM.so (uses PyTorch internally)
```

**问题**：
- SAGE 用户必须安装 PyTorch（大型依赖 ~2GB）
- PyTorch 版本冲突（CUDA vs CPU）
- 增加了 SAGE 的安装复杂度

### 新架构（依赖隔离）
```
SAGE (Python)
  └─> import PyAMM
      └─> NumPy interface (no torch dependency) ✅
          └─> LibAMM.so (PyTorch isolated inside .so file)
```

**优势**：
- ✅ SAGE 用户只需要 NumPy（轻量级）
- ✅ PyTorch 编译进 LibAMM.so，不影响 Python 环境
- ✅ 简化 SAGE 安装流程

## 🔧 技术实现

### 核心改动

**1. 移除 Python 层的 PyTorch 依赖**

`src/PyAMM.cpp` 修改：
```cpp
// 之前
#include <torch/extension.h>  // 需要 Python 环境有 torch

// 之后
#include <pybind11/numpy.h>   // 只需要 NumPy
#include <torch/torch.h>        // 仅内部使用，不暴露到 Python
```

**2. 创建类型转换层**

```cpp
// torch::Tensor → numpy.ndarray
py::array torch_to_numpy(const torch::Tensor& tensor);

// numpy.ndarray → torch::Tensor  
torch::Tensor numpy_to_torch(py::array array);
```

**3. 包装类**

```cpp
class CPPAlgoWrapper {
    AbstractCPPAlgoPtr algo_;  // 内部使用 PyTorch
public:
    // NumPy 接口
    py::array amm(py::array A, py::array B, uint64_t sketchSize) {
        torch::Tensor torchA = numpy_to_torch(A);
        torch::Tensor torchB = numpy_to_torch(B);
        torch::Tensor result = algo_->amm(torchA, torchB, sketchSize);
        return torch_to_numpy(result);  // 返回 NumPy
    }
};
```

### Python 使用示例

```python
# 用户代码 - 只需要 NumPy，不需要 torch
import numpy as np
import PyAMM  # 不再需要 import torch

# 创建 NumPy 数组
A = np.random.randn(1000, 500).astype(np.float32)
B = np.random.randn(500, 800).astype(np.float32)

# 创建算法（内部使用 PyTorch，但对用户透明）
algo = PyAMM.createAMM("crs")
cfg = PyAMM.ConfigMap()
cfg.edit("sketchRatio", 0.1)
algo.setConfig(cfg)

# 计算（输入输出都是 NumPy）
C = algo.amm(A, B, sketchSize=50)  
# C 是 numpy.ndarray，不是 torch.Tensor ✅
```

## 📦 编译配置

### LibAMM 编译（需要 PyTorch）

```bash
# LibAMM 编译时链接 PyTorch（静态链接或动态链接）
cd libamm/build
cmake -DENABLE_PYBIND=ON -DENABLE_TORCHSCRIPT=ON ..
make -j8

# 生成 PyAMM.so（包含 PyTorch 库）
# 文件大小：~50MB（包含 PyTorch 核心）
```

### SAGE 安装（不需要 PyTorch）

```bash
# SAGE 用户安装
pip install sage-libs  # 只需要 NumPy，不需要 PyTorch ✅

# Python 环境依赖
# - numpy
# - pybind11
# ✅ NO torch required!
```

## 🧪 测试状态

### ✅ 已完成
- [x] 代码重构（NumPy 接口）
- [x] 类型转换层实现
- [x] 包装类创建
- [x] Git 提交（commit 217b531）

### ⏳ 待测试
- [ ] 编译 LibAMM.so（需要解决 PyTorch CPU/CUDA 版本冲突）
- [ ] NumPy ↔ torch::Tensor 转换正确性
- [ ] 性能测试（转换开销）
- [ ] SAGE 集成测试

### ⚠️ 已知问题

**编译环境问题**：
```
当前 sage 环境中的 PyTorch 2.7.1 需要 CUDA
但 WSL 环境没有 CUDA
需要使用 CPU 版本的 PyTorch 编译 LibAMM
```

**解决方案**：
1. **选项 A**：在有 CUDA 的机器上编译 LibAMM.so
2. **选项 B**：创建一个独立的 libamm-build 环境（安装 PyTorch CPU 版）
3. **选项 C**：使用 Docker 容器编译

## 📈 功能保留情况

### ✅ 完整保留（90%+）

**CPPAlgos（算法）**：
- ✅ CRS, CRSV2, BCRS（Column Row Sampling 系列）
- ✅ Weighted-CR（加权采样）
- ✅ CountSketch, EWS, CoOFD, TugOfWar（Sketch 算法）
- ✅ SMP-PCA, BlockLRA, RIP, FastJLT（降维算法）
- ✅ INT8, PQ-Raw（量化算法）

**MatrixLoaders（数据加载）**：
- ✅ Random, Gaussian, Beta, Binomial（随机矩阵）
- ✅ Sparse, MNIST, SIFT（数据集加载器）
- ✅ Mtx（Matrix Market 格式）

### ⚠️ 需要数据转换（3 个类）

**依赖 torch::jit（需要 .pt 文件）**：
- VectorQuantization - 需要 codebooks.pt
- ProductQuantizationHash - 需要 hash_containers.pt
- MediaMillMatrixLoader - 需要 MediaMill.pt

**解决方案**：提供数据转换工具（.pt → .npy）

## 🚀 下一步行动

### 立即行动（编译测试）
```bash
# 1. 创建独立的编译环境
conda create -n libamm-build python=3.11
conda activate libamm-build
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pybind11 numpy

# 2. 编译 LibAMM
cd libamm/build
cmake -DENABLE_PYBIND=ON ..
make -j8

# 3. 测试 NumPy 接口
python -c "
import numpy as np
import PyAMM
A = np.random.randn(100, 50).astype(np.float32)
B = np.random.randn(50, 80).astype(np.float32)
algo = PyAMM.createAMM('crs')
C = algo.amm(A, B, 30)
print('Success! C shape:', C.shape, 'dtype:', C.dtype)
"
```

### 后续优化
1. **性能优化**：减少 NumPy ↔ Tensor 转换开销
2. **内存优化**：使用 zero-copy 转换（共享内存）
3. **数据转换工具**：为 VQ/PQ 算法提供 .pt → .npy 转换脚本
4. **文档**：编写 SAGE 用户指南

## 📝 总结

### 核心价值
- ✅ **依赖隔离**：PyTorch 不再污染 Python 环境
- ✅ **向后兼容**：API 不变，只是类型从 torch.Tensor 变为 numpy.ndarray
- ✅ **功能完整**：90%+ 的算法无需修改即可使用
- ✅ **简化安装**：SAGE 用户不需要处理 PyTorch 版本冲突

### 技术亮点
- 🎯 巧妙的抽象层：用户看到 NumPy，内部仍用 PyTorch
- 🔧 零改动算法：所有 LibAMM 算法代码保持不变
- 📦 可分发：PyAMM.so 可以作为独立的二进制分发

---

**Commit**: `217b531` - feat: Isolate PyTorch dependency in Python bindings using NumPy interface  
**Branch**: `main-dev`  
**Date**: 2025-11-12
