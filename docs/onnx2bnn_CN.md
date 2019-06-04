ONNX (Open Neural Network Exchange) 是一个独立于训练框架的模型格式，[众多框架和工具](http://onnx.ai/supported-tools) 支持 ONNX 格式。

#### 模型转换流程

1. 识别二值卷积，对二值卷积的 weight 进行 bit-packing。dabnn 开发者给 onnx 增加了多个 optimizer，用来识别二值卷积，具体实现可参考 https://github.com/daquexian/onnx/tree/optimizer_for_bnn/onnx/optimizer/passes 中的 dabnn_*.h。关于 bit-packing 可以参考 [这篇文档](docs/bconv_CN.md);

1. 修改紧跟着二值卷积的 BN 层的权重。因为 bit 只有 1 和 0 两个值，所以二值卷积中的 -1 被用 0 表示，bitcount 可以得到一个 N-bit 操作数中，值为 1 的 bit 的数量，这忽略了 -1 的存在。具体来说，设 a 为一个 N-bit 操作数，b 为一个自然数，且

> b = bitcount(a)

实际上我们应该得到的值是

> c = bitcount(a) - (N - bitcount(a)) = 2 * bitcount(a) - N = 2 * b - N

这个值可以经过一个对 b 的线性变换得到，因此我们将这个变换融合进二值卷积之后的 BN 层之中。

具体实现在 https://github.com/JDAI-CV/dabnn/blob/master/tools/onnx2bnn/OnnxConverter.cpp#L530。

1. 其他 Layer 正常处理。

#### 注意事项（必看）

模型转换过程中有些规则或限制需要额外说明。

1. **二值卷积的输入 channel 暂时需要是 128 的倍数或 64**；

1. 二值卷积是自定义操作，因此可能存在多种实现，网上存在的二值卷积的自定义实现几乎全部是错的，例如它们用 0 进行 pad，而忽略了二值卷积的输入只能有 +1 和 -1。dabnn 开发者提供了一个[标准的二值卷积 PyTorch 实现](https://gist.github.com/daquexian/7db1e7f1e0a92ab13ac1ad028233a9eb)，我们建议所有二值网络的训练者使用这个实现，或是按照这个实现来在他们用的训练框架中自行实现二值卷积。

1. onnx2bnn 有多种针对二值卷积的识别模式，例如会根据卷积的权重（是否为 +1/-1）识别、根据 Sign operator 识别，在用户选择 aggressive 模式时，甚至可以识别上一条所述的非正确的二值卷积（但在运算时仍会以 -1 而不是 0 来 pad，因此会导致结果不完全一致）。具体请看 [这篇文档](https://github.com/JDAI-CV/dabnn/wiki/Train,-export-and-convert-a-dabnn-model)。

1. 目前暂时不支持 `group` 参数。
