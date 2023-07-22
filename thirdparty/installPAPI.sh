#!/bin/bash
cp -r papi-7.0.1 papi_temp
cd papi_temp/src
./configure --prefix=$PWD/../../papi_build 
make -j24
make install
cd ../..
rm -rf papi_temp
