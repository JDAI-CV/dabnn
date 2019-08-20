#! /usr/bin/env bash
set -e

mkdir build_onnx2bnn && cd build_onnx2bnn
cmake ..
cmake --build .
cd -
