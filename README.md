# LibAMM Benchmark Suite

> **Note**: The algorithm implementations have been moved to [SAGE Framework](https://github.com/intellistream/SAGE) under `packages/sage-libs/src/sage/libs/amms/`.

## Overview

This repository contains the benchmark suite for evaluating Approximate Matrix Multiplication (AMM) algorithms. It provides comprehensive performance evaluation tools, scripts, and datasets.

## 🎯 Purpose

This is a **benchmark-only** repository. For AMM algorithm implementations, please refer to:
- **SAGE Framework**: [sage-libs/amms](https://github.com/intellistream/SAGE/tree/main/packages/sage-libs/src/sage/libs/amms)
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

## 🔗 Related Projects

- **SAGE Framework**: https://github.com/intellistream/SAGE
- **Algorithm Implementations**: https://github.com/intellistream/SAGE/tree/main/packages/sage-libs/src/sage/libs/amms

## 📖 Documentation

See [benchmark/README.md](benchmark/README.md) for detailed benchmark documentation.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.
