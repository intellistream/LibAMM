# LibAMM Benchmark Suite

LibAMM is the DataSys benchmark suite for Approximate Matrix Multiplication (AMM).
The framework-neutral algorithm implementations live in
[AMM-Algorithms](https://github.com/DataSysResearch/AMM-Algorithms). The existing
`isage-amms` package name and `sage.libs.amms` import path are retained for compatibility.

## Overview

This repository contains the benchmark suite for evaluating Approximate Matrix Multiplication (AMM) algorithms. It provides comprehensive performance evaluation tools, scripts, and datasets.

## 🎯 Purpose

This is a **benchmark-only** repository. For AMM algorithm implementations, please refer to:
- **Algorithm library**: [DataSysResearch/AMM-Algorithms](https://github.com/DataSysResearch/AMM-Algorithms)
- **PyPI Package**: `pip install isage-amms`

## 📂 Structure

```
benchmark/
├── scripts/          # Evaluation scripts
├── config.csv        # Benchmark configurations
├── perfLists/        # Performance metric lists
└── src/              # Benchmark runner code

tools/
├── data_manager.py   # Dataset management
└── setup_data.sh     # Dataset setup script
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- AMM algorithms package: `pip install isage-amms`

### Run Benchmarks
```bash
cd benchmark
python scripts/run_benchmark.py --config config.csv
```

## 📊 Benchmark Types

1. **End-to-End (E2E)**: Full pipeline evaluation
2. **Downstream Tasks**: PCA, CCA, DNN inference, QCD
3. **Performance Profiling**: CPU cycles, instructions, energy
4. **Scalability**: Thread scaling, batch size, event rate

## 🔗 Ecosystem

- **DataSys** owns this benchmark and the framework-neutral AMM implementation library.
- **SAGE** consumes AMM capabilities through its application and orchestration interfaces.
- **IntelliStream** incubated the project before its graduation to DataSys.

## 📖 Documentation

See [benchmark/README.md](benchmark/README.md) for detailed benchmark documentation.

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) file.
