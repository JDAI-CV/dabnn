## Bit-packing

Bit-packing is performed in `Binarize` layers. It pack N 32-bit float/integer to an N-bit operand according their signs. For example, performing bit-packing on 128 float numbers produces a 128-bit operand. xnor/xor is only enabled on these packed operands.

The details of bit-packing are in 

* https://github.com/JDAI-CV/dabnn/blob/master/dabnn/bitpack.h#L20 (optimized, for tensors of 128 and more channels)
* https://github.com/JDAI-CV/dabnn/blob/master/dabnn/bitpack.h#L204 (normal, for tensors of less than 128 channels)

The optmized version is 4X faster than the normal version. Bit-packing algorithm directly leverage the sign bits of int32 and IEEE 754 float numbers, and then eliminate the comparison with zeros. SIMD instructions are also used to speed up this process. Note that after SIMD instructions is performed, the N bit in the result will be re-arranged so that they are not in the same order with the N 32-bit inputs. Fortunately, the output of xnor/xor is not affected as long as the input and weight is re-arranged in the same way. Given this observation, we re-arranged the weights of binary convs whose inputs is bit-packed in the optmized way. The details are in https://github.com/JDAI-CV/dabnn/blob/master/dabnn/net.cpp#L82.

dabnn present the following two optmized implementation of binary convs.

## BGEMM

SGEMM (Single float GEneral Matrix Multiplication) is a widely adopted approach to implement float convolutions in various high-performance scientific programs. In the context of BNNs, an alternative operation to SGEMM is BGEMM, which performs binary matrix multiplication for binary convolution after [im2col](https://github.com/JDAI-CV/dabnn/blob/master/dabnn/im2col.h). dabnn present optmized BGEMM. The advantage of GEMM is that it covers all cases of convolutions (various kernel size, stride, padding, ..) and it is easy to implement.

The detailed implementation of BGEMM is in https://github.com/JDAI-CV/dabnn/blob/master/dabnn/bgemm.h.

## Binary Direct Convolution

However, we argue that BGEMM is sub-optimal for BGEMM especially on ARM devices.

In addition to the common multiplication and add operations, BGEMM includes extra operations that count how many 1s are in a vector. Specifically, we denote $U^{M \times N}$ as the space of matrices with dimension $M \times N$ and each element of it is a bit-packed vector. Given two matrices (i.e., $ \boldsymbol{A} \in U^{M \times K}$ and $\boldsymbol{B} \in U^{K \times N}$), $\boldsymbol{C} \in \mathbb{N}^{M \times N}$ ($\mathbb{N}$ represents the set of non-negative integers), $\boldsymbol{C} = BGEMM(\boldsymbol{A}, \boldsymbol{B})$ is measured as:
$$
    C_{i,j} = \sum\nolimits_{k} bitcount(xnor(\Vec{A_{i,k}}, \Vec{B_{k,j}})),
$$

where $\Vec{A_{i,k}}$ and $\Vec{B_{k,j}}$ denotes each element in $ \boldsymbol{A}$ and $\boldsymbol{B}$. In SGEMM, to amortize the cost of loading memory, $\boldsymbol{C}$ is often calculated as
$$
    \boldsymbol{C^{k}} = \boldsymbol{m^{k}}\boldsymbol{n^{k}},
$$
$$
    \boldsymbol{C} \mathrel{+}= \boldsymbol{C^{k}},
$$

where $\boldsymbol{m^{k}}$ is the $k_{th}$ column of $\boldsymbol{A}$ and $\boldsymbol{n^{k}}$ is the $k_{th}$ row of $\boldsymbol{B}$.

In particular, on ARMv8 (the 64-bit ARM architecture) devices, the operation of bitcount contains two instructions: "cnt" and "addv". "cnt" takes an $N$-byte vector $\alpha$ as input and outputs an $N$-byte vector $\beta$, which $\beta_{i} = the\_number\_of\_1s(\alpha_{i})$ where $\alpha_{i}$ and $\beta_{i}$ are the $i_{th}$ byte of $\alpha$ and $\beta$ respectively. "addv" sums up all bytes in a vector and outputs the aggregated scalar. The equation is then expanded as:
$$
    C_{i,j} \mathrel{+}= addv(cnt(xnor(\Vec{m^{k}_{i}}, \Vec{n^{k}_{j}}))).
$$

Thus, the above equation shows that the operation of binary multiply-addition on ARMv8 devices consists of four instructions: xnor, cnt, addv, and addition. Moreover, on ARMv7 (the 32-bit ARM architecture) devices, there is even no "addv" instruction and $\lceil \log_{2}N \rceil$ instructions are needed to sum up all bytes in an $N$-byte vector, so the operation of binary multiply-addition consists of $\lceil \log_{2}N \rceil+3$ instructions on these devices. To improve the efficiency of this operation, we re-arrange the calculation order and calculate $\boldsymbol{C}=BGEMM(\boldsymbol{A},\boldsymbol{B})$ as the multiplication of a row vector $\boldsymbol{p} \in U^{1 \times N}$ and $\boldsymbol{q} \in U^{M \times 1}$:
$$
    C_{i,j} = \boldsymbol{p^{i}}\boldsymbol{q^{j}},
$$

where $\boldsymbol{p^{i}}$ is the $i_{th}$ row of $\boldsymbol{A}$ and $\boldsymbol{q^{j}}$ is the $j_{th}$ column of $\boldsymbol{B}$.

In this way, the cost of "addv" instructions can be mostly squeezed by summing up the results of "cnt" in advance:
$$
    \Vec{C_{i,j}} = \sum\nolimits_{k} cnt(xnor(\Vec{A_{i,k}}, \Vec{B_{k,j}})),
$$
$$
    C_{i,j} = addv(\Vec{C_{i,j}}).
$$

Please note that the same transformation can not be employed in BGEMM because $\boldsymbol{C}$ is stored as 32-bit integers to save the valuable registers. Therefore in the equation of BGEMM, we have to utilize "addv" to reduce the vector into an integer before every instruction of "addition". Taking a close look on the above two equations, we can observe some interesting connections between them and the operation of convolution. Specifically, if we treat $\boldsymbol{A} \in U^{M \times K}$ and $\boldsymbol{B} \in U^{K \times N}$ as the weight and the im2col-ed input ($M$: the number of output channels, $N$: output height $\times$ output width, and $K$: the number of bit-packed vectors in a weight filter), the above two equations can be directly interpreted as the definition of convolution. As such, the refined operation of binary convolution is dubbed as "Binary Direct Convolution".

The implementation of Binary Direct Convolution is in https://github.com/JDAI-CV/dabnn/blob/master/dabnn/bconv.h.
