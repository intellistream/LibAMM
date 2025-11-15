# LibAMM Data Management

## Overview

LibAMM uses a **Git LFS + Symbolic Link** approach for data management, following industry best practices.

## Architecture

```
SAGE/
├── packages/sage-benchmark/src/sage/data/  # Git LFS管理的数据仓库
│   └── libamm-benchmark/
│       ├── models/          # PyTorch模型 (33MB, LFS tracked)
│       ├── test-data/       # VQ测试数据 (18MB, LFS tracked)
│       └── datasets/        # Benchmark数据集 (LFS tracked)
│
└── packages/sage-libs/src/sage/libs/libamm/
    ├── benchmark/
    │   ├── models -> (symlink to data/libamm-benchmark/models)
    │   └── datasets -> (symlink to data/libamm-benchmark/datasets)
    └── test/torchscripts/VQ/
        ├── data -> (symlink to data/libamm-benchmark/test-data)
        └── *.txt -> (symlinks to individual test files)
```

## Setup

### Automatic Setup (Recommended)

Run the setup script after cloning:

```bash
cd packages/sage-libs/src/sage/libs/libamm
bash tools/setup_data.sh
```

Or use Python:

```bash
python tools/data_manager.py
```

### Manual Setup

```bash
export SAGE_DATA_ROOT=/path/to/SAGE/packages/sage-benchmark/src/sage/data
cd packages/sage-libs/src/sage/libs/libamm
bash tools/setup_data.sh
```

## Usage

### For Developers

After setup, data files are accessible via symlinks:

```cpp
// C++ code can use relative paths as before
std::string path = "benchmark/datasets/QCD/qcda_small.mtx";
torch::Tensor data = loadMatrixFromMatrixMarket(path);
```

```python
# Python code
from tools.data_manager import LibAMMDataManager

manager = LibAMMDataManager()
dataset_path = manager.get_dataset_path("QCD/qcda_small.mtx")
model_path = manager.get_model_path("qcdS1_m1.pth")
```

### For New Users

When you first clone SAGE:

```bash
# Clone SAGE
git clone https://github.com/intellistream/SAGE.git
cd SAGE

# Update submodules (including libamm and sageData)
git submodule update --init --recursive

# Pull LFS files from sageData
cd packages/sage-benchmark/src/sage/data
git lfs pull

# Setup LibAMM data links
cd ../../../../sage-libs/src/sage/libs/libamm
bash tools/setup_data.sh
```

## Benefits

✅ **Clean Git History**: No large files in code repository (LibAMM: 644MB → 10MB)  
✅ **Version Control**: Data versions tracked via Git LFS  
✅ **Lazy Download**: Only download data when needed (`git lfs pull`)  
✅ **Flexible**: Use environment variables to customize data location  
✅ **Industry Standard**: Follows PyTorch, HuggingFace, TensorFlow patterns  

## Data Location

Data can be located in three places (checked in order):

1. **Environment variable**: `$SAGE_DATA_ROOT/libamm-benchmark/`
2. **Relative path**: `../../../../../sage-benchmark/src/sage/data/libamm-benchmark/`
3. **Custom path**: Pass to `LibAMMDataManager(data_root="/custom/path")`

## Troubleshooting

### Symlinks not created

```bash
# Re-run setup script
cd packages/sage-libs/src/sage/libs/libamm
bash tools/setup_data.sh
```

### LFS files are pointers (133 bytes)

```bash
# Pull actual LFS files
cd packages/sage-benchmark/src/sage/data
git lfs pull
```

### Data not found

```bash
# Check sageData is cloned
ls packages/sage-benchmark/src/sage/data/libamm-benchmark/

# Set custom path
export SAGE_DATA_ROOT=/path/to/your/data
```

## For CI/CD

In CI environments, you may want to skip data download:

```bash
# Skip LFS in CI
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init

# Or download only needed files
git lfs pull --include="libamm-benchmark/datasets/MNIST/*"
```

## Migration from Old Structure

Old (before 2025-11-12):
- Data was in Git history (644MB repository)
- Data files committed directly

New (current):
- Data in separate LFS-managed repository
- Symlinks in code repository
- 98.5% size reduction

## See Also

- [Git LFS Documentation](https://git-lfs.github.com/)
- [SAGE Data Repository](https://github.com/intellistream/sageData)
- [LibAMM Documentation](../README.md)
