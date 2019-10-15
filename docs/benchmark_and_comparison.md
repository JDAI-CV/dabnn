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
dabnn_bireal18_imagenet_stem   43294019 ns   41401923 ns         14     <--- Bi-Real Net 18 with stem module (The network structure is described in detail in [our paper](https://arxiv.org/abs/1908.05858)), 56.4% top-1 on ImageNet
```

The following is the comparison between our dabnn and [Caffe](http://caffe.berkeleyvision.org) (full precision), [TensorFlow Lite](https://www.tensorflow.org/lite) (full precision) and [BMXNet](https://github.com/hpi-xnor/BMXNet) (binary). We surprisingly observe that BMXNet is even slower than the full precision TensorFlow Lite. It suggests that the potential of binary neural networks is far from exploited until our dabnn is published.

![Comparison](/images/comparison_en.png)


