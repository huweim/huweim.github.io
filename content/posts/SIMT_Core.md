---
title: "SIMT_Core"
date: 2021-09-04T19:04:57+08:00
draft: false
tags: ["官方文档", "GPGPU-Sim"]
categories: ["GPGPU"]
---

# 0. 前言

搞懂 SIMT Core 对于理解 GPGPU 的指令 fetch、指令发射、内存访问、数据传输等步骤非常重要，按照 GPGPU-Sim 的官方文档进行一个简单的梳理

SIMT Core 的微架构模型分为

<img src="D:\ShanghaiTech\2021-Fall\Note_Archive\GPGPU-Sim\Img\Simt-core.png" align=left style="zoom:50%;" />

<img src="D:\ShanghaiTech\2021-Fall\Note_Archive\GPGPU-Sim\Img\Fig3.1.png" align=left style="zoom:50%;" />

# 000 放一个硬件概念对应表

# 1. Front End

+ Instruction cache access
+ Instruction buffer logic
+ Scoreboard
+ Scheduling logic
+ SIMT stack

## 1.1 Fetch and Decode

这里介绍整个指令 Fetch and Decode 阶段，涉及到的硬件单元主要是 Fetch, I-Cache, Decode, I-Buffer, ScoreBoard

##### I. Fetch

Fetch 单元是一个调度器，作用

+ 根据 PC 的值，从 I-Cache 中取指令，即发送内存请求。
+ Check 是否有 warp 已经完成执行，以更新 I-Buffer 信息

对于一个 warp，如果在 I-Buffer 中没有任何 valid 指令 (valid bit 作用在 III. I-Buffer 中有介绍)，那么这个 warp 就可以进行 instruction fetch。

默认情况下，一个 warp 的**两条**连续的指令被取出。

当 Fetch 取到一个 warp 的指令，I-Buffer 中对应的 entry 有效位置为1， 直到该 Warp 的所有指令均执行完毕。

##### II. Decode, I-Buffer

一条指令从 instruction cache fetch 出来后会进行解码，然后存入 instruction buffer (I-Buffer)。每个 warp 有两个 I-Buffer entry, I-Buffer entry 的信息如下

+ Valid bit (1 bit): Valid bit 为1表示这个 warp 在 I-Buffer 中还存在未发射的指令
  + Valid bit 主要是和 Fetch 单元进行交互，如果一个 warp 在 I-Buffer 中还有未发射的指令，那么就不会进行指令 fetch 的操作
+ Ready bit (1 bit): Ready bit 为1表示这个 warp 的这条指令已经可以发射到 execution pipeline, 具体何时发射由 warp scheduler 以及调度策略决定
  + 是否 Ready 主要是由 Scoreboard 决定，
+ I-Buffer entry 中还存有解码后的指令 (decoded instruction) 信息

Decode 单元一次解码2条指令，解码后的指令 fill 到 I-Buffer 中

##### III. I-Cache

指令 cache 是 read-only, non-blocking set-associative cache, 可以使用 FIFO 或是 LRU 替换策略，以及 on-miss 或是 on-fill 分配策略。对 I-Cache 的请求会导致3种状态, hit, miss or reservation fail. 

+ 如果 未命中状态保持寄存器 (MSHR) 已满或者 cache set 中没有可替换的块，则会导致 reservation fail，因为所有 block 都由先前的 pending request 保留 (细节在2.3.4的 cache 部分)。

在 hit 和 miss 情况下，轮询 fetch 调度器都会移动到下一个 warp。

+ 对于 hit, fetched 指令送到 decode stage. 开始解码
+ 对于 miss, 指令 cache 会产生一个请求。
  + 当接收到 miss response, cache block 会填入指令 cache, warp 会再次访问指令 cache. 
  + 当 miss 仍在等待 (pending), warp 不会访问指令 cache

> 这一点对应源码 hader_core_ctx::fetch 函数中，需要 access_ready 才访问指令 cache，否则就去找下一个 I-Buffer 有空间、指令 cache 没有等待 Miss 等待的 active warp

如果 warp 的所有线程都已完成执行而没有任何 outstanding stores or pending writes to local registers, 则 warp 执行完成并且不再被 fetch 调度器考虑。

在 decode stage, 最近 fetched 指令会被解码并存入相应的 I-Buffer entry 等待被发射。

## 1.2 Instruction Issue

第二个轮询调度器 (issue 单元) 选择 I-Buffer 中的一个 warp 将其发射到流水线。这个轮询调度器 (issue) 和用于调度指令 cache 访问的轮询调度器 (fetch) 分离。发射调度器可以配置，每周期发射同一 warp 的多条指令。当前检查的 warp 中的每条 valid 指令 (解码后未发射的指令) 满足以下条件时有资格被发射 (eligible warp)

+ warp 没有在等待 barrier
+ I-Buffer 中有有效指令 (valit bit is set)
+ 通过了 scoreboard 检查
+ 指令流水线的操作数访问阶段 (operand stage) 没有 stall

内存指令 (Load, store, memory barriers) 发射到内存流水线。其他指令可以使用 SP 和 SFU 流水线，不过一般常去 SP 流水线。然而，如果有 control hazard, I-Buffer 中的指令会被刷掉 (flush). 发射指令到流水线后更新 warp 的 PC，将其指向下一条指令 (假设所有分支 not-taken). 下一节1.3会介绍更多 control hazard 的细节

> GPU 是有内存流水线和 ALU 流水线的，两种类型的指令由不同的硬件单元执行

在 issue stage 会执行 barrier 操作。同样，会更新 SIMT stack 以及追踪寄存器依赖 (scoreboard). warp 在 issue stage 会等待 barrier (`__syncthreads()`)

> CUDA 编程中常用这个 barrier, 等待所有游客 (thread) 到齐了再开大巴

## 1.3 SIMT Stack

SIMT Stack 是每个 warp 都有的资源。SIMT Stack 用于处理 SIMT 架构的分支问题 (branch divergence). 因为 GPU 中分支会降低 SIMT 架构的效率，所以有很多降低分支危害的技术。其中最简单的技术是 post-dominator stack-based reconvergence mechanism. 这个技术在最早的保证聚合点 (guaranteed reconvergence point) 同步分支以提高 SIMT 架构的效率。GPGPU-Sim 3.x 使用了这个机制。

SIMT Stack entry 代表不同的分支等级。在每遇到一个分支，一个新的 entry 入栈。到达聚合点时栈顶 entry 出栈。每个 entry 存储新分支的目标 PC、the immediate post dominator reconvergence PC (也就是聚合点 PC) 以及发散到该分支的线程的活动掩码 (active mask). 在这个模型中，每个 warp 的 SIMT Stack 在每条指令发射后更新。

+ 没有分支的 target PC 会正常更新为 next PC.
+ 有分支的情况下，会入栈新的 target PC、相应的线程 active mask、the immediate post dominator reconvergence PC

**因此，如果 SIMT 堆栈顶部入口处的下一个 PC 不等于当前正在检查的指令的 PC，则检测到控制危险。**

NVIDIA and AMD 实际上使用特殊指令修改了他们的 divergence stack 的内容。这些 divergence stack 指令未在 PTX 中公开，但在实际硬件 SASS 指令集中可见（使用 decuda 或 NVIDIA 的 cuobjdump 可见）。 当当前版本的 GPGPU-Sim 3.x 被配置为通过 PTXPlus 执行 SASS（参见 PTX 与 PTXPlus）时，它会忽略这些低级指令，而是创建一个类似的控制流图来识别 immediate post-dominators。 我们计划在 GPGPU-Sim 3.x 的未来版本中支持低级分支指令的执行。

# 2. Register Access and the Operand Collector

# 3. ALU Pipeline

# 4. Interconnection Network

SIMT Core Cluster 之间不会和对方直接通信，因此在 interconnection network 中没有 coherence 通道，只有4种 packet types

-Read-request

-Write-request from SIMT Core to Memory Partition

-Read-replys

-Write-acknowledges sent from Memory Partition to SIMT Core Clusters

# 5. Memory Pipeline (LDST unit)

## 5.1 L1 Data Cache

private, per-SIMT core, non-blocking

L1 Cache 没有划分为 bank

> In Figure 8, 32 consecutive threads access 32 consecutive words. The memory access is sequential and aligned, and is therefore coalesced.

<img src="D:\ShanghaiTech\2021-Fall\Note_Archive\GPGPU-Sim\Img\coalesced.png" align=left style="zoom:50%;" />

内存访问连续且对齐，只需一次访存操作，取到32个线程需要的数据，就是 coalesced

### 5.1.1 MSHR

> On a cache hit, a request will be served by sending data to the register ﬁle immediately. On a cache miss, the miss handling logic will ﬁrst check the miss status holding register (MSHR) to see if the same request is currently pending from prior ones. If so, this request will be merged into the same entry and no new data request needs to be issued. Otherwise, a new MSHR entry and cache line will be reserved for this data request. A cache status handler may fail on resource unavailability events such as when there are no free MSHR entries, all cache blocks in that set have been reserved but still haven’t been ﬁlled, the miss queue is full, etc.

MSHR (Miss Status Holding Registers): 请求发生 miss 时，会首先查看 MSHR 中相同缓存请求是否已经存在。如果存在则请求合并（也就是忽略了这一次的请求）。如果不在 MSHR 中，则将这条 cache line 请求加入 MSHR entry，且这条 cache line 被置为 reserved 状态。将该请求数据放入 miss 队列，排队向下一层缓存发送 data request。如果 MSHR entry 满了，或是请求数据的所有 cache blocks 已经被置为 reserved 状态，或是 miss queue is full，cache status handler may fail on resource unavailability.

MSHR entry 记录 cache block address, block offset, associated register

> MSHR相当于一个大小固定的数组，用于存放所请求数据还没返回到L1缓存中的miss请求。当数据返回到L1缓存中后，即从MSHR中删除所对应的miss请求。

发生内存访问 miss 后，如果 cache line 没有 pending request (等待请求?)，那么 cache line 就会发送 fill request，将其插入到 cache，对应的 MSHR entry 会标记为 filled 状态。filled MSHR entries 的响应是在每个周期的一个请求中生成的。当 filled MSHR entry 中的所有 request 都被响应，MSHR entry is freed.

## 5.2 Texture Cache

The texture cache model is a prefetching texture cache.

texture memory 大多数访问都有 空间局部性，该最好的是 16KB

特点是内存访问延迟比较大而 small cache size，因此何时在 cache 中分配 line 这个问题至关重要

prefetching texture cache 通过将 cache tag 的状态和 cache blocks 的状态 分开来解决这个问题



