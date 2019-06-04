## Bit-packing
在执行二值卷积之前，网络需要手动插入一层`Binarize`。是指将 N 个 32 位的 float/integer，根据和 0 的大小关系，二值化为 N 个 bit （即 0 或 1），并打包成一个 N-bit 的整体，例如对 128 个浮点数进行 bit-packing 之后，就会产生一个 128-bit 的操作数。这一步叫做 bit-packing，做了这一步，后续才可以进行位运算 xnor/xor。

Bit-packing 的具体实现在 

* https://github.com/JDAI-CV/dabnn/blob/master/dabnn/bitpack.h#L20 （高优化版，针对 128 和以上 channel 数的 tensor） 
* https://github.com/JDAI-CV/dabnn/blob/master/dabnn/bitpack.h#L204 （低优化版，针对 128 channel 以下的 tensor） 

高优化版和低优化版的性能差距在 4 倍左右。在高优化版中，bit-packing 算法直接利用 IEEE 754 float 和 int32 的符号位，而不需要把每一个数都和 0 比较，并使用了 SIMD 指令加速这一算法。值得一提的是，使用 SIMD 指令进行 bit-packing 后，输出的 N-bit 操作数的 N 个 bit 和 N 个输入不是按顺序对应的，但只要 xnor/xor 的两个操作数的每个 bit 一一对应，就不会对运算产生任何影响，因此，在适用高优化 bit-packing 的场景下，我们会对 weight 进行重排，使它的每个 bit 和 input 的每个 bit 一一对应，这一步的具体代码在 https://github.com/JDAI-CV/dabnn/blob/master/dabnn/net.cpp#L82。

卷积实现有很多种办法，dabnn 提供了如下两种优化实现。

## BGEMM

GEMM 是实现浮点卷积的通用方法。它要求先用 [im2col](https://github.com/JDAI-CV/dabnn/blob/master/dabnn/im2col.h) 重排输入，经过 im2col 之后，卷积即可被表示为矩阵和矩阵的乘法，即 GEMM。GEMM 的加速方法在 CNN 火热起来之前，就已经得到了深入的研究。不过在二值卷积中，不能利用已有的 GEMM 库，因为它们是为 double、float 或 integer 准备的，因此 dabnn 实现了 BGEMM （Binary GEMM）。它的优点是性能不低，实现方便，一套 GEMM 代码即可处理所有的情况。

BGEMM 的具体实现在 https://github.com/JDAI-CV/dabnn/blob/master/dabnn/bgemm.h。

## Binary Direct Convolution

然而 BGEMM 在 ARM 设备上并不高效，因为二值乘-加操作中，加法需要两步 - bitcount 和普通的加法。Bitcount 用来得到一个 N-bit 操作数中有多少 bit 是 1。在 ARMv8 设备上，bitcount 需要两条指令，ARMv7 设备上需要更多条指令。这大大限制了 BGEMM 的速度。因此 dabnn 提出了直接卷积的方法，称为 Binary Direct Convolution （BDC），它是指直接按照卷积的定义来计算卷积。在 BDC 中，通过一个简单的变换，大部分 bitcount 指令会被消除。它的优点是性能比 BGEMM 更高，但不能像 BGEMM 一样用一套代码覆盖所有的情况。

关于 BDC 如何消除大部分 bitcount 指令，请留意我们即将 publish 的 paper。

BDC 的具体实现在 https://github.com/JDAI-CV/dabnn/blob/master/dabnn/bconv.h。
