### Train and export a dabnn-compatible ONNX model

Binary convolutions are not supported natively by training frameworks (e.g., TensorFlow, PyTorch, MXNet). To implement correct and dabnn-compatible binary convolutions by self, there is something needed attention:

1. The input of binary convolutions should only be +1/-1, but the padding value of convolution is 0.

2. PyTorch doesn't support export ONNX sign operator until PyTorch 1.2.

Therefore, we provide a ["standard" PyTorch implementation](https://gist.github.com/daquexian/7db1e7f1e0a92ab13ac1ad028233a9eb) which is compatible with dabnn and produces a correct result. The implementations TensorFlow, MXNet and other training frameworks should be similar. 

### Convert ONNX to dabnn

#### How does dabnn recognize binary convolutions in ONNX model

The converter `onnx2bnn` has three mode in terms of how it recognizes binary convolutions:

* Aggressive (default). In this mode, onnx2bnn will mark all convolutions whose weights consist of only +1 or -1 as binary convolutions. The aggressive mode is for the existing BNN models which do not have the correct padding value (-1 rather than 0). Note: The output of the generated dabnn model is different from that of the ONNX model since the padding value is 0 instead of -1.
* Moderate. This mode is for our "standard" implementation -- A Conv operator with binary weight and following a -1 Pad operator.
* Strict. In this mode, onnx2bnn only recognizes the following natural and correct "pattern" of binary convolutions: A Conv operator, whose input is got from a Sign op and a Pad op (the order doesn't matter), and weight is got from a Sign op.

For now "Aggressive" is the default mode. To enable moderate or strict mode, pass "--moderate" or "--strict" command-line argument to onnx2bnn.

#### Set binary convolutions manually

For quick benchmarking, you can determine which convolutions are binary ones manually by passing "--binary-list filename" command-line argument. Each line of the file is the **output name** of a convolution, which will be treated as a binary convolution.

For example, you have a model named `model.onnx`, and you want three convolutions, whose outputs are "34", "36", "55" respectively, to be binary convolutions. To converting this model, you should create a text file with

> 34
> 36
> 55

Let's assume the text file is named "my_binary_convs", you can convert the model by

```bash
./onnx2bnn model.onnx model.dab --binary-list my_binary_convs
```
