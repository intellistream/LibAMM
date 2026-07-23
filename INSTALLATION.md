# LibAMM Benchmark Suite - Installation Guide

## System Requirements

### Ubuntu/Debian

```bash
# 基础编译工具
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config

# PAPI 性能计数器库（必需，用于硬件性能分析）
sudo apt-get install -y libpapi-dev

# HDF5（可选，用于数据存储）
sudo apt-get install -y libhdf5-dev

# 其他工具
sudo apt-get install -y graphviz
```

### CentOS/RHEL

```bash
sudo yum install -y gcc-c++ cmake pkg-config
sudo yum install -y papi-devel
sudo yum install -y hdf5-devel
sudo yum install -y graphviz
```

## Quick Start

### 1. Install AMM Algorithms

```bash
# Option 1: Install from PyPI (after publishing)
pip install isage-amms

# Option 2: Install from SAGE repository (for development)
cd /path/to/SAGE
pip install -e packages/sage-libs
```

### 2. Install Benchmark Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Datasets

```bash
cd tools
bash setup_data.sh
```

### 4. Run Benchmarks

```bash
cd benchmark
python scripts/run_benchmark.py --config config.csv
```

## Dependency Details

### Core Dependency: isage-amms

The benchmark suite depends on `isage-amms` package which contains:
- AMM algorithm implementations (Simple, FD, PCA, ROWR, etc.)
- Unified AmmIndex interface
- Algorithm registry and factory

### Installation Methods

**Production (PyPI):**
```bash
pip install isage-amms
```

**Development (SAGE Repository):**
```bash
# From SAGE root directory
pip install -e packages/sage-libs[amms]
```

**Verify Installation:**
```python
from sage.libs.amms import create_amm_index
index = create_amm_index("simple", input_dim=1000, sketch_dim=100)
print("AMM algorithms installed successfully!")
```

## Troubleshooting

### Issue: Cannot find isage-amms

**Solution 1**: Install from PyPI
```bash
pip install isage-amms
```

**Solution 2**: Install from SAGE source
```bash
git clone https://github.com/SAGE-Research/SAGE.git
cd SAGE
pip install -e packages/sage-libs
```

### Issue: Import errors

Make sure you have installed all dependencies:
```bash
pip install -r requirements.txt
```

## For Contributors

If you're developing AMM algorithms:

1. Clone SAGE repository
2. Install in editable mode: `pip install -e packages/sage-libs`
3. Run benchmarks to test your changes
4. Submit PR to SAGE repository

Algorithm implementations go to:
`SAGE/packages/sage-libs/src/sage/libs/amms/implementations/`
