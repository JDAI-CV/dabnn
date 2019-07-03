## Background

Binary Neural Networks is proposed in [Binary Neural Networks](https://arxiv.org/abs/1602.02830) and [XNOR-Net](https://arxiv.org/abs/1603.05279). In the following papers, [Bi-Real Net](https://arxiv.org/abs/1808.00278) presented some new training method in order to improve the performance, [BENN](https://arxiv.org/abs/1806.07550) leverages emsemble on BNNs.

BNNs can save 10X+ memory, and several times as fast as float NNs. What's more, it theoretically [saves 10X energy](https://camo.githubusercontent.com/e725038be60ce4bb698b22480603b636a92beeaf/687474703a2f2f66696c652e656c656366616e732e636f6d2f776562312f4d30302f35352f37392f7049594241467373565f5341504f63534141435742546f6d6531633033392e706e67), so the battery life of devices will be expanded a lot.

## Some notes

1. The BNN models can be trained by any frameworks which support ONNX. Note that binary convs are custom operations, please check out [onnx2bnn.md](docs/onnx2bnn.md) for how to make the model comptabile with dabnn.

2. For the implementation of binary convolutions, please check out [bconv.md](bconv.md).
