# dabnn

[![Build Status](https://dev.azure.com/daquexian/dabnn/_apis/build/status/Android%20Build%20%26%20Test?branchName=master)](https://dev.azure.com/daquexian/dabnn/_build/latest?definitionId=2&branchName=master)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE) 
[![jcenter](https://img.shields.io/badge/dynamic/json.svg?label=jcenter&query=name&url=https%3A%2F%2Fapi.bintray.com%2Fpackages%2Fdaquexian566%2Fmaven%2Fdabnn%2Fversions%2F_latest)](https://bintray.com/daquexian566/maven/dabnn/_latestVersion)
[![Gitter Chat](https://img.shields.io/gitter/room/dabnn/dabnn.svg)](https://gitter.im/dabnn/dabnn)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/JDAI-CV/dabnn/pulls)

*Enjoy binary neural networks on mobile!*

Gitter: [dabnn/dabnn](https://gitter.im/dabnn/dabnn), QQ group (Chinese)：1021964010, answer: nndab

[[English](README.md)] [[Chinese/中文](README_CN.md)]

Our ACM MM paper: https://arxiv.org/abs/1908.05858

## Introduction

Binary neural networks (BNNs) have great potential on edge devices since they replace float operations by efficient bit-wise operations. However, to leverage the efficiency of bit-wise operations, the reimplmentation of convolution layer and also other layers is needed. 

To our best knowledge, dabnn is the first highly-optimized binary neural networks inference framework for mobile platform. We implemented binary convolutions with ARM assembly. On Google Pixel 1, our dabnn is as **800%~2400% faster** as [BMXNet](https://github.com/hpi-xnor/BMXNet) (the only one open-sourced BNN inference framework except dabnn to our best knowledge) on a single binary convolution, and as about **700% faster** as it on binarized ResNet-18.

![Comparison](/images/comparison_en.png)

## Build

We provide pre-built onnx2bnn and also dabnn Android package. However, you need to build it if you want to deploy BNNs on non-Android ARM devices.

We use CMake build system like most C++ projects. Check out [docs/build.md](docs/build.md) for the detailed instructions.

## Convert ONNX Model

We provide a conversion tool, named onnx2bnn, to convert an ONNX model to a dabnn model. We provide onnx2bnn pre-built binaries for all platforms in [GitHub Releases](https://github.com/JDAI-CV/dabnn/releases). For Linux users, the onnx2bnn pre-built binary is [AppImage](https://appimage.org) format, see https://appimage.org for details.

Note: Binary convolution is a custom operator, so whether the ONNX model is dabnn-comptabile heavily depends on the implementation of the binary convolution in the training code. **Please read the [documentation about model conversion](/docs/model_conversion.md) carefully.**

After conversion, the generated dabnn model can be deployed on ARM devices (e.g., mobile phones and embedded devices). For Android developer, we have provided Android AAR package and published it on [jcenter](https://bintray.com/daquexian566/maven/dabnn/_latestVersion), for the usage please check out [example project](https://github.com/JDAI-CV/dabnn-example).

## Pretrained Models

We publish two pretrained binary neural network models based on [Bi-Real Net](https://arxiv.org/abs/1808.00278) on ImageNet. More pretrained models will be published in the future.

* Bi-Real Net 18, 56.4% top-1 on ImageNet, 61.3ms/image on Google Pixel 1 (single thread). [[dabnn](https://drive.google.com/uc?export=download&id=1Oau5CtFR9nWXmlBBU47Jg5ypMiIEMtvo)] [[ONNX](https://drive.google.com/uc?export=download&id=1Xp3HB51H6Nhl6e555ieJubVutQake5sR)]

* Bi-Real Net 18 with Stem Module, 56.4% top-1 on ImageNet, 43.2ms/image on Google Pixel 1 (single thread). The detailed network structure is described in [our paper](https://arxiv.org/abs/1908.05858). [[dabnn](https://drive.google.com/uc?export=download&id=1ArsirMdbtJ9lvHSjc1hkQ7dIXDKh-D1t)] [[ONNX](https://drive.google.com/uc?export=download&id=1zu48CFptAGZ91IDCBPJSPM0bxDuPm9HS)]

## Implementation Details

* The Implementation of Binary Convolutions: [docs/bconv.md](docs/bconv.md)

* Model Conversion: [docs/onnx2bnn.md](docs/onnx2bnn.md)

For more details please read [our ACM MM paper](https://arxiv.org/abs/1908.05858).

## Example project

Android app demo: https://github.com/JDAI-CV/dabnn-example

## Related works using dabnn

The following two papers use dabnn to measure the latency of their binary networks on real devices:

* [IR-Net: Forward and Backward Information Retention for Highly Accurate Binary Neural Networks](https://arxiv.org/abs/1909.10788)

* [Balanced Binary Neural Networks with Gated Residual](https://arxiv.org/abs/1909.12117)

## License and Citation

[BSD 3 Clause](LICENSE)

Please cite daBNN in your publications if it helps your research:

```
@misc{zhang2019dabnn,
  Author = {Jianhao Zhang and Yingwei Pan and Ting Yao and He Zhao and Tao Mei},
  Title = {daBNN: A Super Fast Inference Framework for Binary Neural Networks on ARM devices},
  Year = {2019},
  Eprint = {arXiv:1908.05858},
}
```
