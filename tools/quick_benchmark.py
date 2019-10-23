#!/usr/bin/env python3

import argparse
import inspect
import subprocess
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--onnx', type=str, required=True)
args, others = parser.parse_known_args()

filename = inspect.getframeinfo(inspect.currentframe()).filename
base_dir = Path(filename).resolve().parent.parent
android_ndk = Path(os.getenv('ANDROID_NDK', default='')).resolve()
onnx_model = Path(args.onnx).resolve()
dabnn_build_dir = base_dir/'.build_dabnn_release'
onnx2bnn_build_dir = base_dir/'.build_onnx2bnn_release'
temp_dab_model = onnx2bnn_build_dir/'dabnn_quick_benchmark.dab'
quick_benchmark_bin = dabnn_build_dir/'benchmark'/'benchmark_single_model'
os.makedirs(dabnn_build_dir, exist_ok=True)
os.makedirs(onnx2bnn_build_dir, exist_ok=True)
print("Build dabnn..")
subprocess.check_call('cmake -DCMAKE_TOOLCHAIN_FILE={}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a ..'.format(android_ndk/'build'/'cmake'/'android.toolchain.cmake'), cwd=dabnn_build_dir, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

subprocess.check_call('cmake --build .', cwd=dabnn_build_dir, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Build onnx2bnn..")
subprocess.check_call('cmake ..', cwd=onnx2bnn_build_dir, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

subprocess.check_call('cmake --build .', cwd=onnx2bnn_build_dir, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Generating daBNN model..")
subprocess.check_call('./tools/onnx2bnn/onnx2bnn {} {} {}'.format(' '.join(others), onnx_model, temp_dab_model), cwd=onnx2bnn_build_dir, shell=True)
print("Pushing daBNN model..")
subprocess.check_call('adb push {} /data/local/tmp/'.format(temp_dab_model), cwd=onnx2bnn_build_dir, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.check_call('adb push {} /data/local/tmp/'.format(quick_benchmark_bin), cwd=dabnn_build_dir, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("Benchmarking..")
subprocess.check_call('adb shell /data/local/tmp/{} /data/local/tmp/{}'.format(quick_benchmark_bin.name, temp_dab_model.name), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.check_call('adb shell rm /data/local/tmp/{} /data/local/tmp/{}'.format(quick_benchmark_bin.name, temp_dab_model.name), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
