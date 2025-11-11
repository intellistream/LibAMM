# LibAMM Benchmark Datasets - MOVED

⚠️ **The benchmark datasets have been moved to a centralized location.**

## New Location

The LibAMM benchmark datasets are now hosted in the **sageData** repository for centralized management:

📦 **Repository**: https://github.com/intellistream/sageData

📁 **Path**: `libamm-benchmark/datasets/`

## Why the Move?

- **Centralized Management**: Share datasets across multiple SAGE ecosystem projects
- **Reduced Repository Size**: LibAMM repo is now ~325MB lighter
- **Git LFS Support**: Large binary files are efficiently managed
- **Better Organization**: Datasets grouped by type/domain

## How to Use Datasets

### Option 1: Skip Dataset Copy (Default for embedded usage)

When using LibAMM as a library (e.g., in sageDB), datasets are not needed:

```cmake
set(LIBAMM_SKIP_DATASET_COPY ON CACHE BOOL "Skip dataset copy" FORCE)
```

### Option 2: Use Centralized Datasets from sageData

If you need to run LibAMM benchmarks:

1. Clone sageData repository:
   ```bash
   git clone https://github.com/intellistream/sageData.git
   cd sageData
   git lfs install
   git lfs pull  # Download LFS files
   ```

2. Point LibAMM to the datasets:
   ```cmake
   set(LIBAMM_SKIP_DATASET_COPY OFF)
   set(LIBAMM_DATASET_SOURCE_DIR "/path/to/sageData/libamm-benchmark/datasets")
   ```

### Option 3: Download Manually

You can also download specific datasets you need from the sageData repository.

## Available Datasets

See [sageData/libamm-benchmark/README.md](https://github.com/intellistream/sageData/tree/main/libamm-benchmark) for:
- Complete dataset list
- Size and domain information  
- Usage instructions

## Migration Date

Datasets migrated: November 2025

## Questions?

For dataset-related issues, please open an issue in the sageData repository.
