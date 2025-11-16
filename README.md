# LibAMM

LibAMM aggregates prevalent AMM algorithms, enabling standardized evaluations and efficient experiment management. This project is compatable with libtorch (C++).

## Extra Cmake options (set by cmake -Dxxx=ON/OFF)

- ENABLE_OPENCL, this will enable opencl support in compiling (OFF by default)
    - you have to make sure your compiler knows where opencl is
- ENABLE_PAPI, this will enable PAPI-based perf tools (OFF by default)
    - you need first cd to thirdparty and run installPAPI.sh to enable PAPI support, or also set REBUILD_PAPI to ON
- ENABLE_PYBIND, this will enable you to make python binds, i.e., PyAMM (OFF by default)
    - we have included the source code of pybind 11 in third party folder, please make sure it is complete
    - please compile at a machine with 64GB + memory (swap is also acceptable) when PYBIND is to be compiled

## One-Key build examples with auto solving of dependencies
Please ensure your machine has 64GB + memory (swap is also acceptable) to do either the following. Otherwise, you may need to disable 
the compilation of PyBind or try to copy the binary from another one for compiling it.
- buildWithCuda.sh To build LibAMM and PyAMM with cuda support, make sure you have cuda installed before it
- buildCPUOnly.sh This is a CPU-only version
- After either one, you can run the following to add PyAMM To your default python environment
```shell
 python3 setup.py install --user
```
###  Native python interface (NEW EXPERIMENTAL feature)
If you have compiled PyAMM and make it available, please refer to <buildPath>/benchmark/scripts/PyAMM/*.ipynb for details.
We have opened the python call to AbstractCPPAlgo and AbstractMatrixLoader. They are totally the same as C++ classes.
#### Known issues of pybind
Some U64 settings is not well supported in the python wrapper, so we temporarily convert I64 python list into U64 inside it.
## Requires G++11

The default version of gcc/g++ on ubuntu 22.04 (jammy) is good enough.

### For x64 ubuntu older than 21.10

run following first

```shell
sudo add-apt-repository 'deb http://mirrors.kernel.org/ubuntu jammy main universe'
sudo apt-get update
```

Then, install the default gcc/g++ of ubuntu22.04

```shell
sudo apt-get install gcc g++ cmake python3 python3-pip
```

### For other architectures

Please manually edit your /etc/sources.list, and add a jammy source, then

```shell
sudo apt-get update
sudo apt-get install gcc g++ cmake 
```

Please invalidate the jammy source after installing the gcc/g++ from jammy, as some packs from
jammy may crash down your older version

#### WARNNING

Please do not install the python3 from jammy!!! Keep it raw is safer!!!

## Requires Torch

You may refer to https://pytorch.org/get-started/locally/ for mor details, following are the minimal requirements

### (Optional) Cuda-based torch

Note:

- useCuda config is only valid when cuda is installed
- this branch only allows blackbox call on torch-cuda functions!
  You may wish to install cuda for faster pre-training on models, following is a reference procedure. Please refer
  to https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu

```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda
sudo apt-get install nvidia-gds
sudo apt-get install libcudnn8 libcudnn8-dev libcublas-11-7
```

The libcublas depends on your specific version.
Then you may have to reboot for enabling cuda.

DO INSTALL CUDA BEFORE INSTALL CUDA-BASED TORCH!!!

#### Cuda on Jetson

There is no need to install cuda if you use a pre-build jetpack on jetson. It will neither work,:(
Instead, please only check your libcudnn8 and libcublas

```shell
sudo apt-get install libcudnn8 libcudnn8-dev libcublas-*
```

### (Required) Install pytorch (should install separately)

```shell
sudo apt-get install python3 python3-pip
```

(w/ CUDA):
(Please make all cuda dependencies installed before pytorch!!!)

```shell
pip3 install torch==2.2.0 torchvision torchaudio
```

(w/o CUDA)

```shell
pip3 install --ignore-installed torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

*Note: torch 2.2.0 and above support Python 3.8-3.11*

#### (Optional) Pytorch with Cuda backend on jetson at Jetpack 6.1)

Refer to https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html.
The following steps may be outdated

```shell
sudo apt-get -y update
sudo apt-get install -y  python3-pip libopenblas-dev
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
python3 -m pip install --upgrade pip; python3 -m pip install --no-cache $TORCH_INSTALL
```

### (Optional) Install graphviz

```shell
sudo apt-get install graphviz
pip install torchviz
```

## (optional) Requires PAPI (contained source in this repo as third party)

PAPI is a consistent interface and methodology for collecting performance counter information from various hardware and
software components: https://icl.utk.edu/papi/.
, LibAMM includes it in thirdparty/papi_7_0_1.

### How to build PAPI

- cd to thirdparty and run installPAPI.sh, PAPI will be compiled and installed in thirdparty/papi_build

### How to verify if PAPI works on my machine

- cd to thirdparty/papi_build/bin , and run papi_avail by sudo, there should be at least one event avaliable
- the run papi_native_avail, the printed tags are valid native events.
- please report to PAPI authors if you find your machine not supported

### How to use PAPI in LibAMM

- set -DENABLE_PAPI=ON in cmake LibAMM
- in your top config file, add two config options:
    - usePAPI,1,U64
    - perfUseExternalList,1,U64
    - if you want to change the file to event lists, please also set the following:
        - perfListSrc,<the path to your list>,String
- edit the perfLists/perfList.csv in your BINARY build path of benchmark (or your own list path), use the following
  format
    - <the event name tag you want LibAMM to display>, <The inline PAPI tags from papi_native_avail/papi_avail>,
      String
- please note that papi has a limitation of events due to hardware constraints, so only put 2~3 in each run

## How to build

(CUDA-related is only necessary if your pytorch has cuda, but it's harmless if you don't have cuda.)

### Build in shell

```shell
export CUDACXX=/usr/local/cuda/bin/nvcc
mkdir build && cd build
```

Build for release by default:

```shell
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make 
```

Or you can build with debug mode to debug cpp dynamic lib:

```shell
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make 
```

## An open box test

```shell
cd build/benchmark
python3 pythonTest.py
```

This test will load our lib into libtorch, and call our function in python.

### Tips for build in Clion

There are bugs in the built-in cmake of Clion, so you can not run
-DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`.
Following may help:

- Please run 'import torch;print(torch.utils.cmake_prefix_path)' manually first, and copy the path
- Paste the path to -DCMAKE_PREFIX_PATH=
- Manually set the environment variable CUDACXX as "/usr/local/cuda/bin/nvcc" in Clion's cmake settings

### Tips for using balck-box torchscripts

Please place your *.pt file under benchmark/torchscripts and test/torchscripts, so they can be found easily in building.

## Local generation of the documents

You can also re-generate them locally, if you have the doxygen and graphviz. Following are how to install them in ubuntu
21.10/22.04

```shell
sudo apt-get install doxygen
sudo apt-get install graphviz
```

Then, you can do

```shell
mkdir doc
doxygen Doxyfile
```

to get the documents in doc/html folder, and start at index.html

## Benchmark parameters

Please specify a configfile in running benchmark program, assuming we are using AMM(A,B)
The following are config parameters:

- aRow (U64) the rows of tensor A, default 100
- aCol (U64) the columns of tensor A, default 1000
- bCol (U64) the columns of tensor B, default 500
- sketchDimension (U64) the dimension of sketch matrix, default 50
- coreBind (U64) the specific core tor run this benchmark, default 0
- ptFile (String) the path for the *.pt to be loaded, default torchscripts/FDAMM.pt
  See also the template config.csv

## Evaluation scripts

They are place under benchmark.scripts, for instance, the following allows to scan the number of elements in
the matrix A's row
(Assuming you are at the root dir of this project, and everything is built under build folder)

```shell
cd build/benchmark/scripts
cd scanARow #enter what you want to evaluate
sudo ls # run an ls to make it usable, as perf events requires sudo
python3 drawTogether.py # no sudo here, sudo is insider *.py
cd ../figures
```

You will find the figures then.

## Known issues

1. If you use Torch with cuda, the nvcc will refuse to work as it doesn't support c++20 yet. Therefore, we disabled the
   global requirement check of C++ 20, and only leave an "-std=c++20" option for g++. This will be fixed after nvidia
   can support c++20 in cuda.
2. Some pytorch version can not work well with liblog4cxx and googletest, so we diabled it.
3. Clion may fail to render and highlight the torch apis. In this case, kindly type a random line of "555"
   to validate the highlight when you need it, and comment it during a compile. :)
4. When setting up torch for cpu under python version > 3.10, torch == 1.13.0 would conflict with torchaudio according
   to https://pytorch.org/audio/stable/installation.html. Use Python version <= 3.10 for smooth installation.
5. Some heavy-weight algos like co-occurring FD may be treated as a zombie thread by some OS like ubuntu if running in
   default benchmark program, in this case
   , please force them to be executed as a running child thread by setting forceMP(U64)=1 and threads(U64)=1 in config
