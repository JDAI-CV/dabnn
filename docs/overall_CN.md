## 背景和发展方向

二值网络比较年轻，最初的两篇文章是 2016 年的 [Binary Neural Networks](https://arxiv.org/abs/1602.02830) 和 [XNOR-Net](https://arxiv.org/abs/1603.05279)。后续的工作中，[Bi-Real Net](https://arxiv.org/abs/1808.00278) 提出了一些精度提升方法，[BENN](https://arxiv.org/abs/1806.07550v2) 用 ensemble 方法进一步提升了 BNN 在分类任务上的表现，结果甚至超过单精度浮点模型。

但是从移动端工程应用的角度来看，定点网络可以节省数十倍的内存、提升数倍推理速度，同时降低十倍以上能耗。这意味着原本设备充电一次只能用一个小时，现在理论上可以用十小时以上。能耗相关可参见[相关测试](https://camo.githubusercontent.com/e725038be60ce4bb698b22480603b636a92beeaf/687474703a2f2f66696c652e656c656366616e732e636f6d2f776562312f4d30302f35352f37392f7049594241467373565f5341504f63534141435742546f6d6531633033392e706e67)。

综合算法和工程来看，部分二值网络实用意义和竞争优势可能在以下两点：

1. 与已量产设备融合。嵌入式设备在设计过程中，为了节约成本往往会做成“刚好够用”的状态。二值卷积在计算过程中既不需要大量的 SIMD 乘法操作，内存需求也远低于 8bit 模型，对原有系统干扰极小；
2. 在分类任务中以混合精度的方式替换已有方法。

卷积曾出现过很多变种，但是其中大部分已被历史淘汰。BNN 要想避免此命运，最简单的方法莫过于尽快落在某个产品或项目上，证明自己的价值。


## 软件架构
在使用流程和软件结构方面，dabnn 和已开源的推理库（如 [ncnn](https://github.com/Tencent/ncnn)、[Tengine](https://github.com/OAID/Tengine)、[FeatherCNN](https://github.com/Tencent/FeatherCNN) 等）差距不大：

1. 模型训练可使用任意一种可以导出 ONNX 模型的框架，但需要注意的是，二值卷积是自定义操作，为了让模型中二值卷积可以被 dabnn 正确识别，请看 [onnx2bnn_CN.md](onnx2bnn_CN.md)。
2. 部署模型前需要把 onnx 格式转换成 dabnn 内部格式。在转换过程中，会把二值卷积的权重转换为 1-bit （而不是默认的 32-bit），大大减小模型文件的体积。流程和**注意事项**可参照 [onnx2bnn_CN.md](onnx2bnn_CN.md)；
3. 二值卷积实现请查阅 [bconv_CN.md](bconv_CN.md)
