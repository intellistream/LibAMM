This section is to run MADNESS evaluation on CIFAR datasets with our C++ AMM algorithms

## Setup bolt dependency

The **[bolt](https://github.com/dblalock/bolt)** library is required for downstream inference benchmarks.
Run the setup script to download it on-demand:

```bash
# From this directory (benchmark/scripts/Downstream_Inference/)
./setup_bolt.sh
```

This will:
1. Clone the bolt repository
2. Download the kmc2 dependency
3. Convert them to regular directories (no submodule management needed)

After running the script, complete the setup:

```bash
# Set up kmc2 submodule
cd bolt/third_party/kmc2
pip install numpy==1.23.1 cython numba zstandard seaborn
python3 setup.py build_ext --build-lib=.  # Compile cython to .so
```

Evaluate AMM

```
# Perform evaluation on AMM
cd ../../experiments
python3 -m python.amm_main # Reproduce Madness paper accuracy results
python3 -m python.amm_main -t cifar10 -c ../../config_dnn_inference.csv -m ../../metrics.csv # Interface to use intellistream AMM c++ API. After this you should see a metrics.csv, where you can check the AMM latency, AMM fro error, ending accuracy in it.
```
