## About ONNX

[ONNX](http://onnx.ai) (Open Neural Network Exchange) is an open format which is greatly supported or officially integrated by [many frameworks and tools](http://onnx.ai/supported-tools).

## How onnx2bnn converts the models

1. Recognizing binary convolutions, whose weights will be bit-packed. The developers of dabnn added several [optimizer](https://github.com/onnx/onnx/blob/master/docs/Optimizer.md) to ONNX in order to recognize binary convolutions. The details is in dabnn_*.h of https://github.com/daquexian/onnx/tree/optimizer_for_bnn/onnx/optimizer/passes. For bit-packing, please check out [this documentation](bconv.md)

2. Update the weight and bias of BN layers following binary conv layers. Since -1 in binary convs is represented by a unset bit (i.e., 0), and bitcount returns the number of set bits (i.e., 1) in a N-bit operand, a correction is needed to get the correct result of binary convs. Specifically, denote a as an N-bit operand, b as the number of set bits in a, c as the number of unset bits in a, the result we want is

> b - c = b - (N - b) = 2 * b - N = 2 * bitcount(a) - N

It is an affine transform of bitcount(a), so we accordingly update the weight and bias of the corresponding BN layers.

The details is in https://github.com/JDAI-CV/dabnn/blob/master/tools/onnx2bnn/OnnxConverter.cpp#L522.

3. Other layers are converted as usual.

## Notes (Need Attention)

There are some notes for model conversion.

1. **The number of input channels of binary convs must be 64 or a multiple of 128 for now.**

2. Binary convolutions are custom operations in training frameworks (e.g., TensorFlow, PyTorch), so the implementations are various. Unfortunately, the most existing implementations of binary convs are not correct. For example, they always pad 0 to their input, while the input should only be +1 or -1. The developers of dabnn provide [a standard implementation of binary convs in PyTorch](https://gist.github.com/daquexian/7db1e7f1e0a92ab13ac1ad028233a9eb). We advise trainers of BNNs to use this implementation, or implement binary convs in their own training frameworks according to this implementation.

3. onnx2bnn has multiple recognizing levels. It can even recognize the incorrect binary convs described above (the result will be incorrect though). Please check out [this documentation](https://github.com/JDAI-CV/dabnn/wiki/Train,-export-and-convert-a-dabnn-model) for details.

4. `group` is not supported for now.
