#### 模型转换流程

1. 请先阅读开源版本的 [onnx 格式说明](https://github.com/onnx/onnx/blob/master/docs/IR.md);
2. 读取所有参数和 Layer。对于二值卷积，把参数转为二进制格式。
```
    bin_val = 1 if float_val > 0.0f else 0
```
此处实现和 `bmxnet` 相比，在`0.0f`的处理上有差异。`bmxnet`的转换函数为：
```
    bin_val = 1 if float_val >= 0.0f else 0
```
此处和一些同学讨论过，有的认为权重刚好是 0.0f 的情况几乎不存在；也有的说如果权重是 0.0f，训练几乎失败。目前的结论是不影响。

3. 其他 Layer 正常处理。

#### 注意事项（必看）

由于`BNN`商业化程度不高，且`onnx`当前版本对`BNN`支持不够完善，模型转换过程中有些硬编码规则需要额外说明。

1. bias 参数在转换模型的过程中会被重命名为 `{name}_conv_b`，而`kernel`权重会被重命名为`{name}_conv_w`；

2. 由于`onnx`当前版本还没有对应的名称，训练平台（如`PyTorch`）导出为`onnx`格式时，需要把`domain`字段设为`dabnn`以标记此层为二值卷积层；

3. `onnx`格式为 NCHW，而`dabnn`内置格式为 NHWC，与 `TensorFlow`/`SNPE` 一致。这么设计是为了方便后续卷积计算；

4. 二值卷积参数之和暂时需要是 64 的倍数；

5. `group` 必须要为 1，因此目前不支持分组卷积和`MobileNet`。
