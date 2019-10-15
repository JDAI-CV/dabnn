### Train and export a dabnn-compatible ONNX model

binary convolutions are not supported natively by training frameworks (e.g., TensorFlow, PyTorch, MXNet). To implement correct and dabnn-compatible binary convolutions by self, there is something needed attention:

1. The input of binary convolutions should only be +1/-1, but the padding value of convolution is 0.

2. PyTorch doesn't support export ONNX sign operator (until [my PR](https://github.com/pytorch/pytorch/pull/20470) gets merged and published in a new release)

Therefore, we provide a ["standard" PyTorch implementation](https://gist.github.com/daquexian/7db1e7f1e0a92ab13ac1ad028233a9eb) which is compatible with dabnn and produces a correct result. The implementations TensorFlow, MXNet and other training frameworks should be similar. 

### Convert ONNX to dabnn

#### Optimization levels

The converter `onnx2bnn` has three levels in terms of how it recognizes binary convolutions:

* Aggressive (default). In this level, onnx2bnn will mark all convolutions with binary (+1/-1) weights as binary convolutions. It is for the existing BNN models, which may not use the correct padding value. Note: The output of the generated dabnn model is different from that of the ONNX model since the padding value is 0 instead of -1.
* Moderate. This level is for our "standard" implementation -- A Conv operator with binary weight and following a -1 Pad operator.
* Strict. In this level, onnx2bnn only recognizes the following natural and correct "pattern" of binary convolutions: A Conv operator, whose input is got from a Sign op and a Pad op (the order doesn't matter), and weight is got from a Sign op.

For now "Aggressive" is the default level. To enable another level, pass "--moderate" or "--strict" command-line argument to onnx2bnn.

#### Set binary convolutions manually

For benchmarking, onnx2bnn supports users to determine which convolutions are binary ones manually by passing "--binary-list filename" command-line argument. Each line of the file is the **output name** of a convolution, which will be treated as a binary convolution.
