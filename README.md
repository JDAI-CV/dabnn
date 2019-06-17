# dabnn

[![Build Status](https://dev.azure.com/daquexian/dabnn/_apis/build/status/Android%20Build%20%26%20Test?branchName=master)](https://dev.azure.com/daquexian/dabnn/_build/latest?definitionId=2&branchName=master)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE) 
[![jcenter](https://img.shields.io/badge/dynamic/json.svg?label=jcenter&query=name&url=https%3A%2F%2Fapi.bintray.com%2Fpackages%2Fdaquexian566%2Fmaven%2Fdabnn%2Fversions%2F_latest)](https://bintray.com/daquexian566/maven/dabnn/_latestVersion)
[![Gitter Chat](https://img.shields.io/gitter/room/dabnn/dabnn.svg)](https://gitter.im/dabnn/dabnn)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/JDAI-CV/dabnn/pulls)

*Enjoy binary neural networks on mobile!*

[English](README.md) [中文](README_CN.md)

Join chat at [Gitter (English)](https://gitter.im/dabnn/dabnn) or QQ Group (Chinese, 1021964010, answer: nndab)

## Introduction

Binary neural networks (BNNs) have great potential on edge devices since they replace float operations by efficient bit-wise operations. However, to leverage the efficiency of bit-wise operations, the reimplmentation of convolution layer and also other layers is needed. 

To our best knowledge, dabnn is the first highly-optimized binary neural networks inference framework for mobile platform. We implemented binary convolutions with ARM assembly. On Google Pixel 1, our dabnn is as **800%~2400% faster** as [BMXNet](https://github.com/hpi-xnor/BMXNet) (the only one open-sourced BNN inference framework except dabnn to our best knowledge) on a single binary convolution, and as about **700% faster** as it on binarized ResNet-18.

## Benchmark and Comparison

Benchmark result on Google Pixel 1 (single thread):

```
2019-05-06 10:36:48
Running data/local/tmp/dabnn_benchmark
Run on (4 X 1593.6 MHz CPU s)
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
--------------------------------------------------------------------
Benchmark                             Time           CPU Iterations
--------------------------------------------------------------------
dabnn_5x5_256                   3661928 ns    3638192 ns        191     <--- input: 14*14*256, kernel: 256*5*5*256, output: 14*14*256, padding: 2
dabnn_3x3_64                    1306391 ns    1281553 ns        546     <--- input: 56*56*64,  kernel: 64*3*3*64, output: 56*56*64, padding: 1
dabnn_3x3_128                    958388 ns     954754 ns        735     <--- input: 28*28*128, kernel: 128*3*3*128, output: 28*28*128, padding: 1
dabnn_3x3_256                    975123 ns     969810 ns        691     <--- input: 14*14*256, kernel: 256*3*3*256, output: 14*14*256, padding: 1
dabnn_3x3_256_s2                 268310 ns     267712 ns       2618     <--- input: 14*14*256, kernel: 256*3*3*256, output: 7*7*256, padding: 1, stride: 2
dabnn_3x3_512                   1281832 ns    1253921 ns        588     <--- input:  7* 7*512, kernel: 512*3*3*512, output:  7* 7*512, padding: 1
dabnn_bireal18_imagenet        61920154 ns   61339185 ns         10     <--- Bi-Real Net 18, 56.4% top-1 on ImageNet
dabnn_bireal18_imagenet_stem   43294019 ns   41401923 ns         14     <--- Bi-Real Net 18 with stem module (The network structure will be described in detail in the coming paper), 56.4% top-1 on ImageNet
```

The following is the comparison between our dabnn and [Caffe](http://caffe.berkeleyvision.org) (full precision), [TensorFlow Lite](https://www.tensorflow.org/lite) (full precision) and [BMXNet](https://github.com/hpi-xnor/BMXNet) (binary). We surprisingly observe that BMXNet is even slower than the full precision TensorFlow Lite. It suggests that the potential of binary neural networks is far from exploited until our dabnn is published.

![Comparison](images/comparison_en.png)

## Build

We provide pre-built onnx2bnn and also dabnn Android package. However, you need to build it if you want to deploy BNNs on non-Android ARM devices.

We use CMake build system like most C++ projects. Check out [docs/build.md](docs/build.md) for the detail instructions.

## Convert ONNX Model

We provide a conversion tool, named onnx2bnn, to convert an ONNX model to a dabnn model. We provide onnx2bnn pre-built binaries for all platforms in [GitHub Releases](https://github.com/JDAI-CV/dabnn/releases). For Linux users, the onnx2bnn pre-built binary is [AppImage](https://appimage.org) format, see https://appimage.org for details.

Note: Binary convolution is a custom operator, so whether the ONNX model is dabnn-comptabile heavily depends on the implementation of the binary convolution in the training code. Please check out [our wiki](https://github.com/JDAI-CV/dabnn/wiki/Train,-export-and-convert-a-dabnn-model) for the further information.

After conversion, the generated dabnn model can be deployed on ARM devices (e.g., mobile phones and embedded devices). For Android developer, we have provided Android AAR package and published it on [jcenter](https://bintray.com/daquexian566/maven/dabnn/_latestVersion), for the usage please check out [example project](https://github.com/JDAI-CV/dabnn-example).

## Pretrained Models

We publish two pretrained binary neural network models based on [Bi-Real Net](https://arxiv.org/abs/1808.00278) on ImageNet. More pretrained models will be published in the future.

* Bi-Real Net 18, 56.4% top-1 on ImageNet, 61.3ms/image on Google Pixel 1 (single thread). [[dabnn](https://drive.google.com/uc?export=download&id=1Oau5CtFR9nWXmlBBU47Jg5ypMiIEMtvo)] [[ONNX](https://drive.google.com/uc?export=download&id=1Xp3HB51H6Nhl6e555ieJubVutQake5sR)]

* Bi-Real Net 18 with Stem Module, 56.4% top-1 on ImageNet, 43.2ms/image on Google Pixel 1 (single thread). The detailed network structure will be described in the coming paper. [[dabnn](https://drive.google.com/uc?export=download&id=1ArsirMdbtJ9lvHSjc1hkQ7dIXDKh-D1t)] [[ONNX](https://drive.google.com/uc?export=download&id=1zu48CFptAGZ91IDCBPJSPM0bxDuPm9HS)]

## Implementation Details

* The Implementation of Binary Convolutions: [docs/bconv.md](docs/bconv.md)

* Model Conversion: [docs/onnx2bnn.md](docs/onnx2bnn.md)

## Example project

Android app demo: https://github.com/JDAI-CV/dabnn-example

## License

[BSD 3 Clause](LICENSE)
