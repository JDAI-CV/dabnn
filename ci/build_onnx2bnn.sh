#! /usr/bin/env bash
set -e

nproc=$(ci/get_cores.sh)

mkdir build_onnx2bnn && cd build_onnx2bnn
cmake ..
cmake --build . -- -j$nproc
cd -
