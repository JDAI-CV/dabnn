ONNX (Open Neural Network Exchange) 是一个独立于训练框架的模型格式，[众多框架和工具](http://onnx.ai/supported-tools) 支持 ONNX 格式。

#### 模型转换流程

1. 读取所有参数和 Layer。对二值卷积的 weight 进行 bit-packing。关于 bit-packing 可以参考 [这篇文档](docs/bconv_CN.md);
1. 修改二值卷积之后的 BN 层的权重。因为 bit 只有 1 和 0 两个值，所以二值卷积中的 -1 被用 0 表示，bitcount 可以得到一个 N-bit 操作数中，值为 1 的 bit 的数量，这忽略了 -1 的存在。具体来说，设 a 为一个 N-bit 操作数，b 为一个自然数，且

> b = bitcount(a)

实际上我们应该得到的值是

> c = bitcount(a) - (N - bitcount(a)) = 2 * bitcount(a) - N = 2 * b - N

这个值可以经过一个对 b 的线性变换得到，因此我们将这个变换融合进二值卷积之后的 BN 层之中。

具体实现在 https://github.com/JDAI-CV/dabnn/blob/master/tools/onnx2bnn/OnnxConverter.cpp#L530。

1. 其他 Layer 正常处理。

#### 注意事项（必看）

由于 BNN 商业化程度不高，且 onnx 当前版本对 BNN 支持不够完善，

`onnx2bnn` 依赖的是经过 dabnn 开发者自定义的 onnx，其中增加了多个 optimizer，用来识别二值卷积，具体实现可参考 https://github.com/daquexian/onnx/tree/optimizer_for_bnn/onnx/optimizer/passes 中的 dabnn_*.h

模型转换过程中有些规则或限制需要额外说明。

1. **二值卷积的输入 channel 暂时需要是 128 的倍数或 64**；

1. 目前暂时不支持 `group` 参数。

1. 由于 `onnx` 没有二值卷积的 operator，onnx2bnn 会把 `domain` 字段设为 `dabnn` 以标记此层为二值卷积层；

1. `onnx` 的元素布局为 NCHW，而 `dabnn` 的元素布局为 NHWC 或 NC1HWC2。具体可以关注我们即将 publish 的 paper；
