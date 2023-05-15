# AMMBench

An universal parallel approximate matrix multiplication (AMM) on multiple devices. This project is compatable with
libtorch

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

Note: this branch only allows blackbox call on torch-cuda functions!
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

### (Required) Install pytorch

```shell
sudo apt-get install python3 python3-pip
```

(w/ CUDA):
(Please make all cuda dependencies installed before pytorck!!!)

```shell
pip3 install torch==1.13.0 torchvision torchaudio
```

(w/o CUDA)

```shell
pip3 install torch==1.13.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

#### (Optional) Pytorch with Cuda backend on jetson

Refer to https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html.
The following steps may be outdated

```shell
sudo apt-get -y update; 
sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev;
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v51/pytorch/torch-2.0.0a0+fe05266f.nv23.04-cp38-cp38-linux_aarch64.whl
python3 -m pip install --upgrade pip; python3 -m pip install aiohttp numpy=='1.19.4' scipy=='1.5.3' export "LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"; python3 -m pip install --upgrade protobuf; python3 -m pip install --no-cache $TORCH_INSTALL
```

### (Optional) Install graphviz

```shell
sudo apt-get install graphviz
pip install torchviz
```

## How to build

(CUDA-related is only necessary if your pytorch has cuda, but it's harmless if you don't have cuda.)

### Build in shell

```shell
export CUDACXX=/usr/local/cuda/bin/nvcc
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
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
