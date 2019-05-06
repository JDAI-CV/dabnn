#! /usr/bin/env bash
set -e

mkdir build_onnx2bnn && cd build_onnx2bnn
# azure pipeline image aliases cmake to their cmake 3.12
/usr/bin/cmake ..
/usr/bin/cmake --build . -- -j$(nproc)
cd -
