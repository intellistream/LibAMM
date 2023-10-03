This section is to run MADNESS evaluation on CIFAR datasets with our C++ AMM algorithms


If you just cloned AMMBench repo, remember to initialize the **[bolt](https://github.com/dblalock/bolt)** submodule

```bash
# Assume you are at AMMBench/
git submodule init
git submodule update
```


Then follow **bolt** repo set up procedure

```bash
# Init submodule in bolt
cd benchmark/scripts/Downstream_Inference/bolt
git submodule init # Init submodule 'third_party/kmc2'
git submodule update

# Set up kmc2 submodule
cd third_party/kmc2
pip install numpy
pip install kmc2
pip install . # Compile cython

```


Evaluate AMM

```
# Perform evaluation on AMM
cd ../../experiments
python3 -m python.amm_main # Reproduce Madness paper accuracy results
python3 -m python.amm_main -c ../../mm_temp_test_madness.csv -m ../../metrics.csv # Interface to use intellistream AMM c++ API
```
