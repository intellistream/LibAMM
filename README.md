# LibAMM

> A high-performance library for Approximate Matrix Multiplication (AMM) algorithms with PyTorch integration and Python bindings.

[![CMake CI](https://github.com/intellistream/LibAMM/actions/workflows/cmake.yml/badge.svg)](https://github.com/intellistream/LibAMM/actions/workflows/cmake.yml)
[![Build & Publish](https://github.com/intellistream/LibAMM/actions/workflows/build-and-publish.yml/badge.svg)](https://github.com/intellistream/LibAMM/actions/workflows/build-and-publish.yml)

## 🚀 Quick Start

### PyPI Installation (Recommended)

```bash
pip install isage-libamm
```

```python
import PyAMM
# Use LibAMM algorithms
```

### From Source

**One-line build (CPU-only)**:
```bash
./buildCPUOnly.sh
python3 setup.py install --user
```

**With CUDA support**:
```bash
./buildWithCuda.sh
python3 setup.py install --user
```

## 📋 Requirements

- **Compiler**: GCC/G++ 11+ (Ubuntu 22.04+ default)
- **Python**: 3.8-3.12 (3.11 recommended)
- **PyTorch**: 2.0.0+
- **Memory**: 64GB+ recommended for full build (swap acceptable)

### Install Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install gcc g++ cmake python3-dev

# Install PyTorch (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**With CUDA**: Install CUDA toolkit before PyTorch, then:
```bash
pip install torch torchvision torchaudio
```

**📖 Detailed installation**: See [Installation Guide](#installation-details) below

## 🏗️ Build Options

CMake options (use `-D<option>=ON/OFF`):

- `ENABLE_PYBIND`: Build Python bindings (default: ON)
- `ENABLE_UNIT_TESTS`: Build C++ tests (default: OFF)
- `ENABLE_PAPI`: Enable PAPI performance counters (default: OFF)
- `ENABLE_HDF5`: Enable HDF5 support (default: OFF)
- `LIBAMM_SKIP_DATASET_COPY`: Skip copying benchmark datasets (default: OFF)

### Manual Build

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc  # If using CUDA
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make -j$(nproc)
```

## 🧪 Testing

```bash
# Python test
cd build/benchmark
python3 pythonTest.py

# C++ tests (if ENABLE_UNIT_TESTS=ON)
cd build/test
./cpp_test
```

## 📚 Documentation

- **API Reference**: Run `doxygen Doxyfile` to generate docs in `doc/html/`
- **Examples**: See `benchmark/scripts/PyAMM/*.ipynb` for Jupyter notebooks
- **Configuration**: See `benchmark/config.csv` template

### Generate Documentation

```bash
sudo apt-get install doxygen graphviz
mkdir doc && doxygen Doxyfile
```

Open `doc/html/index.html` in browser.

## 🔧 Configuration Parameters

Key parameters in `config.csv`:

- `aRow`, `aCol`, `bCol`: Matrix dimensions
- `sketchDimension`: Sketch matrix dimension (default: 50)
- `ptFile`: Path to TorchScript model (*.pt)
- `usePAPI`: Enable PAPI counters (requires PAPI setup)

## 🐛 Known Issues

1. **CUDA + C++20**: NVCC doesn't fully support C++20, so we use `-std=c++20` for G++ only
2. **Heavy algorithms**: May need `forceMP=1, threads=1` in config to avoid OS killing process
3. **Python 3.12**: Fully supported in latest version

## 📦 PyPI Package

Published as `isage-libamm` on PyPI. Includes pre-built wheels for:
- Python 3.9, 3.10, 3.11, 3.12
- Linux x86_64
- CPU-optimized builds

## 📄 License

MIT License - see LICENSE file for details.

---

## Installation Details

<details>
<summary><b>📖 Click to expand detailed installation instructions</b></summary>

### System Requirements

**Ubuntu 22.04+ (Recommended)**:
```bash
sudo apt-get update
sudo apt-get install gcc g++ cmake python3 python3-pip python3-dev
```

**Older Ubuntu (< 22.04)**:
```bash
sudo add-apt-repository 'deb http://mirrors.kernel.org/ubuntu jammy main universe'
sudo apt-get update
sudo apt-get install gcc g++ cmake
```

⚠️ **Warning**: Don't install `python3` from jammy on older systems!

### PyTorch Installation

**CPU-only** (faster, recommended for CI/CD):
```bash
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
```

**With CUDA** (install CUDA first!):
```bash
pip install torch==2.2.0 torchvision torchaudio
```

**Jetson (JetPack 6.1)**:
```bash
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip install --upgrade pip
pip install --no-cache $TORCH_INSTALL
```

### Optional: PAPI Performance Counters

```bash
cd thirdparty
./installPAPI.sh

# Verify PAPI
cd thirdparty/papi_build/bin
sudo ./papi_avail        # Should show available events
./papi_native_avail      # Show native event tags
```

Use in LibAMM:
- Set `ENABLE_PAPI=ON` in CMake
- Add to config: `usePAPI=1`, `perfUseExternalList=1`
- Edit `perfLists/perfList.csv` with event tags

### Optional: GraphViz

```bash
sudo apt-get install graphviz
pip install torchviz
```

</details>

---

**Questions?** Open an issue at [GitHub Issues](https://github.com/intellistream/LibAMM/issues)
