---
title: "CUDA_set_gridDim"
date: 2021-08-17T14:12:38+08:00
draft: false
tags: [""]
categories: ["CUDA"]
---

# 0. 前言

在一个 CUDA 课程的考试中由于这个地方的理解问题导致没有成功 pass，应该如何设置 BlockNum 呢？

# 1. 参数

+ compute capability, CC

这个也就是计算架构，对应于具体的 NVIDIA 显卡型号，可以在编译时作为 option 输入

+ ThreadsPerBlock

这个参数是最常见的，也就是 `blockDim`, 自己设置的是1024, 计算架构会决定 block 内线程数的上限

+ RegistersPerThread

一直没有设置过这个参数，ptxas info 会给给出具体的使用数据

## 1.1 问题

接下来问题出现了，到底应该怎么设置 gridDim 呢？也就是 BlockNums. 在测试代码中数据量 `N=10000000`, 自己的理解是我一个 block 用1024个 threads，使用 256 个block，这样的话总共有 256*1024 个 threads 可以并行工作，那么我用 for 循环加上步长 `stride = blockDim.x * gridDim.x` 来实现 `N` 个数据的计算，也就是

```c++
#define N 10000000
#define THREADS_PER_BLOCK 1024
#define BLOCK_NUMS 256
//#define BLOCK_NUMS ((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)

__global__ void gpu_histogram(int *input, int count, int *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index;i<count;i+=stride){           
            int target = input[index] / 50;
            assert(target >= 0 && target < BUCKETS_COUNT);
            atomicAdd(&output[target],1);
    }
    __syncthreads();  
}
int main(){
...
    gpu_histogram<<<BLOCK_NUMS, THREADS_PER_BLOCK>>>(...);
...
}
```

**但是这样是没法通过的，把 `#define BLOCK_NUMS 256` 修改为 `#define BLOCK_NUMS ((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) ` 才能得到正确结果**

## 1.2 思考

我不要去关注 `gridDim` 这个参数的设置，设置好 ThreadsPerBlock 之后通过计算直接得到 BlockNums 这个数，不需要去考虑硬件最多支持多少 block 并行。

我们在程序中设定了需要跑多少个 block，并不意味着计算机会同时运行完这些 block，有多少 block 在并行是由计算机硬件决定的。设置好数量和处理逻辑之后交给计算机去执行就OK。所以推荐都通过 `((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)` 这种方式去计算

至于为什么设置为 256 结果会出错，目前还没有理解
