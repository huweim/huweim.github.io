---
title: "GPGPU_Architecture"
date: 2021-07-24T16:35:27+08:00
draft: false
tags: ["GPGPU"]
categories: ["知识"]
---

# GPGPU Architecture

从有缩进的那一段开始成为第一段

# 1. Introduction

## 1.1 The Landspace Of Computation Accelerators

1 提升性能不能光依赖于摩尔定律了，需要从 Computer Arch 中去寻找提升

2 GPU 的性能优势, vector HW

3 专用的硬件对应用的性能提升帮助很大，如谷歌 TPU

4 modern GPUs support a Turing Complete programming model, 这是人们对 GPU 感兴趣的一大原因

By Turing Complete, we mean that any computation can be run given enough time and memory.  

## 1.2 GPU Hardware Basic 

1 API for GPUs, 

> These APIs function by providing convenient interfaces that hide the complexity of managing communication between the CPU and GPU rather than eliminating the need for a CPU entirely.  

2 CPU (DDR) and GPU (GDDR), different DRAM

> The CPU DRAM is typically optimized for low latency access whereas the GPU DRAM is optimized for high throughput.  

集成GPU和独立GPU

<img src="D:\ShanghaiTech\2021-Fall\ZhaoXin\Textbook\Note\Img\Fig1.1.png" align=left style="zoom:50%;" />

3 Nv Pascal 架构有软硬件支持，可以自动将 CPU memory 的 data 转移到 GPU memory

unified memory 的概念，在 CPU 和 GPU 中都有的一个虚拟内存支持

> This can be achieved by leveraging virtual memory support [Gelado et al., 2010], both on the CPU and GPU. NVIDIA calls this “unified memory.”  

在集成的 CPU GPU 架构中，二者使用同样的 内存结构，无需考虑上述问题，但是要注意二者共用 cache 带来的 cache coherence 问题

4 CPU, GPU, kernel, how to run

5 描述 SIMT 架构，不过很好奇这里的 SMIT Core 在不同的物理架构中对应的是什么单元

<img src="D:\ShanghaiTech\2021-Fall\ZhaoXin\Textbook\Note\Img\Fig1.2.png" align=left style="zoom:50%;" />

> Each core also typically contains first-level instruction and data caches.  
>
> The large number of threads running on a core are used to hide the latency to access memory when data is not found in the first-level caches.  

6 Often, each memory channel 链接到 LLC 的一部分

这个 interconnection network 值得讨论，有一些 paper 中有不同的实现方法

> The GPU cores and memory partitions are connected via an on-chip interconnection network such as a crossbar  

# 2.Programming Model

## 2.2 GPU Instruction Set Architecture

+ Backward compatibility -> means that a program compiled for a prior generation architecture will run on the next generation architecture without any changes
  + 因此，是向后兼容

### 2.2.1 NVIDIA

 Parallel Thread Execution ISA, or **PTX**

**SASS**: short for “Streaming ASSembler”

# 3. The SIMT Core: Instruction and Register Data Flow  

<img src="D:\ShanghaiTech\2021-Fall\ZhaoXin\Textbook\Note\Img\Fig3.1.png" align=left style="zoom:50%;" />

+ Fetch: 负责读取内存中的指令。
+ I-Cache: 即instruction cache，为L1 缓存，从内存中读取的指令放入到L1缓存中以便下次能够快速获取到指令。
+ Decode:负责解析读取到的指令，包含解析出指令需要使用到的源寄存器和目的寄存器 以及操作指令。
+ I-Buffer: instruction buffer ，为L1缓存，用于缓存一些指令，主要用于识别缓存中这些指令之间是否有依赖，如果指令之间没有依赖则可以乱序执行，以最大可能提高效率。
+ Score Board: 指令打分系统，通过该模块用于识别指令之间的依赖。
+ SIMT-Stack: 用于解决同一个warp内分支问题，通过mask 在分支执行过程中，将不需要执行的线程通过mask 去掉。
+ Issue:指令发送模块。
+ Operand Collector:主要用于解决在cycle 切换时通过进行warp切换，寄存器过多而造成 port过多问题，通过统一的 bank 来解决此问题。
+ ALU：计算单元，该数目与 warp 大小相等，按照其功能和数据划分INT32, FP32，FP64等
+ MEM：GPU内存管理模块，包括与外部所接off-line模块。

1

> it is necessary to employ an architecture that can sustain large
> off-chip bandwidths.  
>
> 维持高带宽的需求

2

> GPU pipeline  
>
> The pipeline can be divided into a SIMT front-end and a SIMD back-end.  
>
> The pipeline consists of three scheduling “loops” acting together in a single pipeline: 
>
> + an instruction fetch loop, 
>   + includes the blocks labeled Fetch, I-Cache, Decode, and I-Buffer.  
> + an instruction issue loop, and 
>   + includes the blocks labeled I-Buffer, Scoreboard, Issue, and SIMT Stack  
> + a register access scheduling loop.   
>   + includes the blocks labeled Operand Collector, ALU, and Memory  

搞清楚三个 loop 是什么

3

> from high-level view of the overall GPU pipeline and then fill in details.  
>
> 从 overview 到 细节

## 3.1 One-Loop Approximation

considering that GPU has a single scheduler  

one-loop approximation 可以认为是 instruction fetch loop

> 上面提到GPU内部含有很多core，为了方便进行对这些core进行调度，将一定数量的core组成warps进行调度（AMD 称为wavefronts)，在同一个warp内的core使用相同的PC(program counter)指，针即在同一个warp内的所有core同一时刻执行相同的指令。

同一个 warp 内的线程使用相同的 PC

> Fetch模块根据PC指向的指令从内存中获取到相应的指令（PC指针指向的指令是下一个将要执行的指令），之后SIMT将获取到的指令进行译码，并根据译码之后的结果从register file从获取到寄存器，与此同时 SIMT 进行mask处理，在接下来的处理流程中，根据SIMT mask结果，进行预测，哪些线程(core)将会执行。

也就是说 SIMT execution mask value 决定 core 是否执行

> 该指令得到调度之后，SIMT中的core(图中的ALU)根据mask 结果按照SIMD(single instruction multiple data)模型进行执行，即warp内的每个 core 或者也可以称为 ALU，执行相同指令，但是数据不一样。
>
> 同时GPU为了尽可能提高执行效率，将执行的指令按照功能或者执行的数据划分为专门的硬件单元进行执行，例如 NV 中将 ALU 划分为 SFU(specical function unit)、load/store unit、floating-point function unit、integer function unit，V100中还专门划分了Tensor 处理单元用于深度学习，即每个单元仅仅执行指令中的一部分功能。

special function unit. 它可以做sin/cos/sqrt/reciprocal这类运算，且每周期每线程吞吐一条指令（不过精度有限）

> SIMT 内 core 的数量即 warp 大小相等一般与 lane 相等， 各个 GPU在时钟周期 (clock cycles) 内使用不同的 warp 执行策略，可以通过提高时钟频率以提高 warp内切换线程数目来取得较高的性能。而提高 SIMT性能 有两种方法分为为增加流水线深度或者提高时钟频率（以增加单位时间内能够执行更多的指令）。
>
> 该循环为一个SIMT 流水线执行过程的大概逻辑过程，为整个SIMT整个执行流程。

##### lane? 

lane 应该就是硬件中的通道

> PCIe Gen3每个通道（每个Lane）的双向带宽是2B/s，GPU一般是16个Lane的PCIe连接，所以PCIe连接的GPU通信双向带宽可以达到32GB/s，要知道PCIe总线堪称PC系统中第二快的设备间总线（排名第一的是内存总线）。但是在NVLink 300GB/s的带宽面前，只有被碾压的份儿。

每个 PCIe 2.0 的通道 (lane) 理论上可以提供 500MB/s 的带宽

1 介绍了 GPU Pipeline 的一些步骤

> Thus, the unit of scheduling is a warp.  
>
> warp 是基本的调度单元
>
> In each cycle, the hardware selects a warp for scheduling.  
>
> 调度方法，每个周期选一个 warp
>
> In the one loop approximation the warp’s program counter is used to access an instruction memory to find the next instruction to execute for the warp.   
>
> Warp PC 访问指令内存找到 warp 执行的下一条指令
>
> In parallel with fetching source operands from the register file, the SIMT execution mask values are determined.   
>
> 并行地从寄存器堆中 fetch 源操作数，确定 SMIT 执行 :question: 掩码值 (execution mask value)

2

> After the execution masks and source registers are available, execution proceeds in a single-instruction, multiple-data manner.  
>
> SIMD manner

:question: function unit 是做什么的？

special function unit. 它可以做sin/cos/sqrt/reciprocal这类运算，且每周期每线程吞吐一条指令（不过精度有限）

3

> 增加时钟频率的方法，流水线化执行或增加流水线深度

### 3.1.1 SIMT Execution Masking  

#### 1. 博客

##### I. What's the SIMT stack?

在实际的应用中，不可避免有一些 branch, 即 some threads execute onee branch, some execute another branch -> cause 同一 warp 内线程间的指令不相同

We know 一个 warp 内 only one PC pointer. 在同一个 warp 内执行不同 ins -> warp divergence

GPU 中没有 branch predictor 机制，因此要尽量避免 warp divergence

Thus, 为了解决上述问题，引入了 SIMT stack 模块

##### II. SIMT

> GPU 真正的调度执行模式是按照SIMT(single-instruction multiple-thread)模式，即每个SIMT包含一定数量的core用于执行（GPU中的core一定属于某个SIMRT）
>
> 在NV中 SIMT 被称之为wrap, AMD中称之为wavefront，它是硬件调度的并行的最小单位，即一个SMIT 可以同时并行运行的线程数目。而SM一般由多个或者一个warp组成，一般程序开发人员感知不到warp的存在，只能感知到SM,一个通用别的GPU架构一般由下图组成
> ————————————————
> 版权声明：本文为CSDN博主「Huo的藏经阁」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
> 原文链接：https://blog.csdn.net/weixin_42730667/article/details/109838089

原来 SIMT 就是 warp。注意到程序开发人员感知不到 warp 的存在，只能感知到 SM。说明 SM 是最基本的调度单元，Warp 是最小的执行单位

##### III. SIMT Stack 以及 SIMT mask 的作用 :star:

> SIMT执行模型是GPU中的一个重要特性，甚至可以说是GPU整个调度核心所在，SIMT可以使开发人员从功能角度来来各个线程之间是独立的，各个线程之间相互独立执行相同的指令。但是在实际开发过程中，不可避免的要使用到一些分支处理，即有些线程执行一个分支，而另外一些线程执行另外一个分支，这样就造成了线程之间执行指令并不相同，而SIMT模型中可以知道，在同一个warp内只有一个PC指针，如果在同一个warp内的线程内分别执行不同的分支，就会造成执行指令分化（称之为线程束分化，可以参考下面文章https://zhikunhuo.blog.csdn.net/article/details/105167996），线程束的分化要尽量避免，要解决线程束分化问题不仅仅要从上层应用开发解决，整个GPU硬件内部调度角度也需要解决该问题。
>
> GPU SIMT执行模型为了解决上述问题，引入了SIMT STACK模块。
>
> SIMT stack主要用来解决两个关键问题：
>
> 1：控制流嵌套问题（nested control flow): 当kernel存在多个分支，且其中一个分支依赖与另外一个分支，严重影响了线程独立性
>
> 2：如何跳过计算过程（skip computation)：由于分支的存在，造成了在同一个warp内的有些线程并不必要执行一些计算指令。

在原文中有关于这一段的描述，以下的例子也来自原文

> 为了解决上述分支情况，SIMT stack模块使用 mask 来进行标记，其中置为 1 表示该线程需要执行，置0表示该线程不需要执行，每一个bit位代表了一个线程。
>
> 假设一个GPU，warp大小为4 即同时支持4个线程并行运行，当上述kernel代码运行到A处时，所有的线程都运行，则标记位”A/1111“，其中mask bit位从低位到高位分别表示四个线程从第一个到第四个线程。当代码运行到if(t3 != t4) 时出现分支造成线程分化，其中第四个线程运行F分支，第一到三个线程运行B分支，则可分别表示"F/0001"和”B/1110"。处于B线程继续执行又遇到if( t5 != t4 )分支，其中第一个线程执行C，第二和三个线程执行D，mask可分别表示位"C/1000"和"D/0110"，当C和D执行完毕之后到E线程有重新聚合，即"E/1110"。而在G出 又出现了线程聚合，即"G/1111"。整个上述kernel执行控制流图（control flow graph)CFG，如下图所示

<img src="D:\ShanghaiTech\2021-Fall\ZhaoXin\Textbook\Note\Img\Fig3.4_a.png" align=left style="zoom:50%;" />

> GPU上述执行遇到上述分支时，只能采取顺序执行，即先执行其中一个分支然后再执行另外一个分支，当执行其中一个分支时，根据提前标记好的mask决定线程是否运行，如果在该分支有些线程mask 被置0，则只能进行等待状态。
>
> 这时因为SIMT执行模式在同一个warp内只有一个PC执行即在warp内所有线程都只能执行这一条命令，因此只能选择执行或者不执行，而不能在不执行时选择其他指令进行执行。

不能在等待时选择其他指令执行，因为所有线程共用一个 PC

> 即遇到分支时，warp只能按照顺序串行化进行执行，直到所有线程都执行完毕。上述kernel 执行时，在SIMT stack其他预测的执行顺序如下

实心代表线程执行，空心代表线程不执行，被屏蔽

<img src="D:\ShanghaiTech\2021-Fall\ZhaoXin\Textbook\Note\Img\Fig3.4_b.png" align=left style="zoom:50%;" />

这个时候可以体会 SIMT Stack 的作用了，在SIMT stack中为了指明整个序列号执行路径，按照一条条entry记录，使用类似如下图所示内容

<img src="D:\ShanghaiTech\2021-Fall\ZhaoXin\Textbook\Note\Img\Fig3.4_c.png" align=left style="zoom:50%;" />

> 上述SIMT stack中在初始化c图部分，TOS指向下一个Next PC为B，即下一个要执行的指令为B出现分支，执行B指令的mask为1110。
>
> 而B 指令的entry指向 d图的TOS，即指向完B之后 开始执行C 执行而B执行聚合点指令为E，即执行完C之后之后还要执行聚合点同样为E的指令 即D指令，只有聚合点都是E的指令执行完毕之后，才能进行往下执行，而 执行的指令则为E指令。此时同样E指令的聚合点指令为G，需要把所有聚合点位G的指令执行完毕之后 才能执行G 指令。
>
> 其中每个entry有三项分别为 
>
> + reconvergence program counter(RPC或者Ret/Reconv PC，聚合点PC)、
> + the address of the next instruction to execute(Next PC 要执行的下一条PC)、
> + Active Mask(记录当前指令哪些线程执行，哪些线程被屏蔽），
>
> 其中 TOS 即为 top of stack 处于栈顶位置，指向的当前指令对应的entry，上述 i 和 ii 以及 iii 分别代表三个分支的entry。每个 entry 中如果 Next PC 和 Ret/Reconv PC 不相等则说明存在分支情况（注意这是个人看法，暂时还没有从相关论文中找到这个明确说法）。
>
> 那么现在出现一个问题：出现分支之后，要先执行哪一个分支？具体采用什么样的准则？
>
> AMD的官方文档中给出了建议，NV采用类似方法：
>
> To reduce the maximum depth of the reconvergence stack to be logarithmicn in the number of threads in a warp it is best to put the entry with the most active threads on the stack first and then the entry with fewer active threads.
>
> 为了减少 reconvergence stack的深度，先执行活跃线程最多的分支，然后再执行活跃少的分支。

#### 2. 原文

1

SIMT, handle 2 key issue,  **nested control flow**

##### I. 控制流

>  控制流就是控制程序的运行方向，符合条件的代码执行，不符合的不执行；程序代码是从上到下执行，从左到右执行，和人的行为习惯是类似；	
>
>  控制流分为**`顺序控制流`、`分支控制流`、`嵌套控制流`、`循环控制流`**；
>
>  `**顺序控制流**`的基本语法为：`if...else`，条件满足执行if的条件代码块，否则执行else的代码块；
>
>  `**分支控制流**`就是在顺序控制流`if...else`的基础上，在if和else的中间加上不定量的 `elif`，可以加入多个elif，执行顺序是自上而下，满足条件则执行；
>
>  `**嵌套控制流**`，顾名思义就是在控制流的语句里面里面嵌套一个或者多个控制流，比如：if顺序里面再嵌套新的if顺序控制流，建议不要嵌套太深，一般不大于3层以上为佳，执行顺序依旧是自上而下，满足条件则执行；
>
>  `**循环控制流**`分为四种情况：
>
>  ```python3
>  (1) while循环 就是在某条件下，循环执行某段程序，多用于处理需要重复处理的相同任务
>   while 判断条件：
>     执行语句
>  
>  (2) while...else循环 和while循环的不同在于，它会在循环正常执行完毕后执行 else 语句块中的代码
>  
>  (3) for循环可以遍历任何序列的项目，如一个列表或者一个字符串
>  
>  (4) for...else循环和while...else循环一样，会在循环正常执行完毕后执行 else 语句块中的代码
>  
>  (5) while...else和for...else如果循环代码中有break、return、或者错误产生的时候，不会执行其else语句哦。
>   循环控制流需要注意，循环需要有结束循环的条件，否则无限执行就会耗尽资源而导致代码崩溃中止。
>  ```

**skipping computation** entirely while all threads in a warp avoid a control flow path

由于分支的存在，在同一个 warp 内有些 thread 并不必要执行一些计算指令

CPU 有分支预测，来处理嵌套控制流

2

SMIT 被一些特殊的指令所管理

3，4

图3.2，3.3，3.4 给出了 SIMT stack operation 的CUDA C，PTX code 实例

### 3.1.2 SIMT Deadlock And Stackless SIMT Architectures  

#### 1. 博客

##### I. 介绍

> SIMT mask 可以解决 warp 内分支执行问题，通过串行执行完毕分支之后，线程在reconverge point 时有重新聚合在一起以便最大提高其并行能力。但是对于一个程序来说，如果出现分支就意味这每个分支的指令和处理不一致，这就容易造成对一些共享的数据的一致性，

出现分支意味着执行了不同的汇编指令

> 失去一致性就意味着在同一个 warp 内的如果存在分支则线程之间不能够交互或者交互数据，在一些实际算法中为了进行数据交互 则需要使用 lock 机制，而mask 恰恰会因为调度问题造成一个死锁 deadlock 问题.

给出了一个 cuda 代码的实例，以及 cuda 中源自操作函数 atomicCAS 的用法

> 造成上述死锁deadlock问题原因恰是SIMT mask调度引起的。上述代码逻辑在CPU执行过程中没有问题，但是在GPU执行过程中由于出现分支，SIMT stack的调度原则是首先执行活跃数线程最多的分支，上述例子是首先执行分支为31个线程的情况，而一直循环，而另外一个prev返回0的分支一直由于上述31个线程一直处于循环之中而造成得不到调度，这样会一直处于一个dealock状态，而无法继续向前执行。
>
> 造成上述问题的根本原因就是在warp内的所有线程只有一个PC指针，无法同时执行其他命令，而这种SIMD模型正式GPU的核心所在，看似这种问题是属于GPU SIMT执行模型的天生问题无法解决

##### II. NV V100 提出的解决方法 :question:

`还有一些细节需要后续来了解`

为warp内每个线程都分配了一个PC指针和Stack, 但还是以 warp 为单位进行调度

> V100 中NV 为warp内每个线程都分配了一个PC指针和Stack，这样将PC指针的颗粒度细化到了每一个线程中去，但是这样就于GPU 的根基SIMT执行模型又有明显的冲突，因为这样每个线程都有自己的PC，岂不是和CPU没什么本质上的差别。
>
> 为了解决上述问题，在V100内部调用中，硬件还是使用的 warp 这一单位进行调度线程，V100内部中使用了a schedule optimizer硬件模块决定哪些线程可以在一个warp内进行调度(这样就涉及到另外一种技术rearrange thread 稍后再讲)，讲相同的指令重新进行组织排布到一个warp内，执行SIMD模型，以保证最大利用效率。

V100可以将分支组织成一个sub-warp来保证处于同一分支的在同一个sub-warp内

那么 sub-warp 就没有分支问题了

> NV已经将上述方法申请了专利 《Execution of divergent threads using a convergence barrier》最后可能和真正商用稍有不同，但是其基本思想基本一致，该专利中使用a convergence barrier方法来解决上述死锁问题，就是如果出现上述分支依赖情况，就需要创建一个a convergence barrier，来保证第一分支先执行，后一个分支后执行，而为了保存维护这些信息，每个线程都需要维护各自的entry信息。

#### 2. 原文

1

通过 mask 状态位来解决 branch 问题，通过串行执行完分支后，thread 在 reconverge point 重新聚合时，可继续并行执行

2 知识点，关于 automic CAS, compare-and-swap, A-B-A -> A1-B2-A3 来区分是否 modified

3

Warp 内 all threads 只有 1 PC pointer，无法同时执行其他命令

Solution: NVIDIA, V100. 为 every warp allocate 1 PC pointer and stack.

但为了保留 GPU 特性，与 CPU 区分。仍使用 warp 为单位调度 thread，V100 使用了 a schedule optimize hardware module 决定哪些线程可以在一个 warp 内调度 (rearrange thread)

### 3.1.3 Warp Scheduling

> warp调度为GPU执行过程中非常关键的一部分，直接决定了每个时钟周期哪些线程在warp得到调度运行，不同的GPU其调度算法不同，各式各样。但是都有一个前提是在每个warp执行时，每个warp都只能同时执行相同指令，只有前一个指令执行完毕之后，warp才会通过调度执行下一个指令。
>
> 在一种理想 GPU 状态下，GPU内的每个 warp 内的线程访问内存延迟都相等，那么可以通过在 warp 不断切换线程可以隐藏内存访问的延迟，比如在同一个 warp 内，此时执行的内存读取指令，那么可以采用异步方式，在读取内存等待过程中，下一刻切换线程其他指令进行并行执行，这样 GPU 就可以一边进行读取内存指令，一边执行计算指令动作。

注意这里的描述是 切换线程的其他指令，理解为将 warp 正在执行的指令切换为另一条，warp 内的线程继续去执行另一条指令，而内存访问的操作交给 LD/ST 单元

> 该方法主要时因为 GPU 将不同类型的指令分配给不同的单元进行执行，读取内存使用 LD/ST 硬件单元，而执行计算指令可能使用 INT32 或者 FP32 硬件单元。这样就可以通过循环调用（round robin）隐藏内存延迟问题。在理想状态下，可以完全通过这种循环调用方式完全隐藏掉内存延迟问题。
>
> 使用循环调用的调度方式时，就需要在每个时钟周期内不断切换线程，每个线程都需要有自己的专用寄存器保存私有相关信息，随着 warp 切换数量不断增加，warp其所需要的寄存器会不断增多，同样会造成芯片面积不断增大。同样在固定面积大小的芯片上，随着 warp 数量增加，core的数量将不断减少，同样core数量减少将造成性能下降，反回来会不足够完成隐藏掉内存延迟，这本身时称为一个矛盾问题。
>
> 在实际上内存延迟问题还取决于 应用程序访问的内存位置以及每个线程对off-chip 内存的访问数量。
>
> 内存延迟问题 影响着 warp 调度，可以通过合理的warp 调度隐藏掉内存延迟问题。

## 3.2 Two-Loop Approximation

#### 1. 博客 

two-loop 即instruction issue loop，主要解决即将执行的指令，将需要执行的指令发送到相应的ALU中进行执行。

##### I. 除了线程切换外的优化

> 在上篇文章中最后 warp 调度中提到为了解决内存操作延迟，通过在时钟周期内不断切换线程隐藏掉内存延迟。但是当时间执行的线程不够多或者内存延迟不够时，仅仅依靠warp调度的层次不能完全解决掉该问题，还要在其他方面同时进行解决。
>
> 本节主要是说明在同一个 warp 内，在单个线程内通过调整正发送到 ALU 将要执行的指令顺序，同样可以隐藏掉一部分内存延迟问题。
>
> 通过调整发送的指令思路比较简单：假如有两个指令 一个指令为内存读取指令，另外一个指令为加法指令。而前一个指令执行读取内存指令（读取指令可能会消耗几个或者几十个甚至上百个时钟周期），在数据读取完成之前，其实core什么事情都干不了，只能等待读取指令周期执行完毕之后，再往下执行。既然读取指令和加法指令使用的是不同的硬件单元，那么再第一个时钟周期执行内存读取指令之后，下一个时钟周期不必等待读取内存指令而是执行加法指令，从而实现一个一边计算一边读取并行的执行，从而提高整个运行效率。

当然，前提是加法指令不依赖于内存读取指令的结果

> 然而在实际情况中，后一个指令是依赖于前一个指令的读取结果。为了解决该办法，就需要就 GPU 提前进行对指令之间的依赖关系进行预测，解析出指令之间的独立性，是否对其他指令有依赖关系。
>
> 为了解析要该指令集是否对上一个指令有依赖关系，GPU需要提前从内存中读取一些指令，而不是只读取一个指令。为了存储这些指令，SIMT中增加了instruction buffer(I-buffer)，将这些指令存储起来。I-buffer在实际GPU设计的过程中一般采用的是L1 cache，以便提高读取效率。

注意 I-buffer在实际GPU设计的过程中一般采用的是L1 cache

> 这些指令被从内存读取出来放入到I-buffer中，但是这样做还不够，还无法识别出这些指令的依赖。
>
> 那么如何解决这些指令的依赖性？可以借鉴CPU中的设计，在CPU中 为了识别指令集的依赖关系一般采用 scoreboard 和reservation stations两种方法，其中 reservation stations 为了识别出指令集之间的依赖关系，需要创建出一种associative logic关联逻辑关系，而创建该关系不仅仅会增加芯片面积还会增加芯片消耗，显然对于GPU来说该方法不合适。

在 CPU 设计中，用一个 bit 位来标识寄存器是否会被写

> scoreboard 方法与reservation stations相比相对要简单许多，在CPU中为了解析指令之间的独立性，为每个寄存器都增加一个bit位，用于表示该寄存器是会被写，如果置1则表示该寄存器被写，此时如果另外一个指令中操作的源或者目的寄存器发现该寄存器bit位被置1，则会处于一直等待状态（说明该指令依赖与前一个指令），一直到该寄存器的bit位被清零（表明之前写寄存器操作完成）。这样就可以防止前面一个命令对该寄存器写之后，另外一个指令同时对该寄存器写造成出现数据不一致。同样也可以防止对该寄存器入write-after-read操作顺序（先write后读）变成read-after-write。

##### GPU 中运用 soreboard 存在的问题

> 但是将scoreboard应用到GPU还存在两个问题需要解决
>
> + 由于GPU寄存器要比CPU多的多，如果为每个一个寄存器增加一位那么将需要更多的额外内存，比如假设一个GPU 每个core有64 个warps，每个warp有128个寄存器，那么为每个core增加8192bits大小内存。
> + 在GPU一旦一个指令由于依赖性被堵塞，那么将会一直进行轮询查看scoreboard中的状态，直到前一个指令集执行完成将该bit位被清零。在GPU中，由于同时会有很多个线程在相同时刻执行相同指令，一旦其执行的指令被堵塞，那么将会有很多线程同时访问scoreboard，将会造成很大压力，同时i-buffer中有没有依赖的指令，也无法发出去。
>   + 也就是需要所有线程都没有依赖后才能发送下一条指令，可能会造成 stall

##### 解决方案

> 为了解决上述问题，Coon等人在《Tracking Register Usage During Multithreaded Processing Using a Scorebard having Separate Memory Regions and Storing Sequential Register Size Indicators》提出了动态解决方案，在本文中提出为每个warp 创建一个表，表中的每个entry记录被指令做写操作的寄存器。这样当一个指令集从内存中读取出来放入到I-buffer时将该指令集中的源寄存器和目的寄存器与entry做比较，是否有其他指令集已经对该寄存器再做写操作，如果有则返回一个bit vector与该寄存器一起写入到I-buffer中。如果该指令集的写操作完成，将会刷新I-buffer中的该指令集寄存器去的bit vector，将bit vector清除掉。另外如果一个指令集做写操作，需要将该寄存器放入的entry中，但是此entry已经满了，那么该指令集将会一直等待 或者被丢弃过一定时钟周期被重新获取再次查看entry是否满。

## 3.3 Three-Loop Approximation  

#### 1. 博客

> 通过再warp内不在每个时钟周期不断切换线程来隐藏内存访问延迟问题，但是切换线程就意味着需要大量 register file 来保存上下文信息，比如在NV(Kepler,MAXwell等）架构中，其 register file 达到了256KB。

之前在 LearnSys 中有人提到过切换时需要保存上下文信息，这个时候就需要用到大量寄存器

> 理想情况下每个时钟周期每个指令操作的寄存器都需要一个port，这样就需要大量port来解决访问问题，显然在实际中不太可能。为了减少port数量，Coon等人通过 mutiple banks of single-ported memories来解决问题，设计出了著名的operand collector结构。

### 3.3.1 Operand Collector

#### Collector Unit

##### I. 工作的阶段

> After an instruction is decoded, a hardware unit called a collector unit is allocated to buffer the source operands of the instruction.

指令译码后，一个叫 `Collector Unit` 的硬件单元开始工作，它负责将指令的源操作数缓冲起来。

##### II. 功能

针对的是数据依赖，记住是处理 `Bank Conflict` 的解决方法

`Collector Unit`的功能不是通过 寄存器重命名 的方式来消除寄存器的数据依赖，而是将 寄存器文件 分成多个Bank，并且尽量保证让一条指令中的寄存器访问的是不同的 Bank

##### III. 原理

<img src="D:\ShanghaiTech\2021-Fall\ZhaoXin\Textbook\Note\Img\Fig3.12.png" align=left style="zoom:50%;" />

> 上图为operand collector原型图，该图中拥有4个bank 通过crossbar链接链接到 Pipleline Register，pipleline Register 为stage register，用于临时存储从regiester file读取过来的指令/数据，stage register最后将得到的数据/指令发送到SIMD执行单元中。

Bank 理解为复用单个寄存器资源，让 warp 中的线程“看起来”独享了一个寄存器，而实际是通过调度实现这一过程

<img src="D:\ShanghaiTech\2021-Fall\ZhaoXin\Textbook\Note\Img\Fig3.13.png" align=left style="zoom:60%;" />

> 上图一个寄存器一个为 4 个 bank 的布局，其中 r0 表示的是 r0 寄存器， w0 代表warp 0，那么 w0:r0 代表来自于 w0 的 r0 寄存器分布在 bank0 中，来自于 w0 的 r1 寄存器分布在 bank1 中，依次进行排布，当寄存器的数量大于 bank 时，那边将轮转重新排布，即 w0:r4 排布到bank 0寄存器中。依次 w1:r0 重新从 bank0 开始排布。
>
> 如果在一个warp0和wapr1中如果warp 0读取的是r0寄存器 warp1读取的是r1寄存器，分布到两个不同的bank内，即可以同时进去读取。
>
> 如果是warp0和warp读取 r0寄存器那么将会产生一个bank冲突，造成无法并行，**bank冲突会严重影响性能**。下面有个例子将会说明bank冲突将会怎样影响性能调度

<img src="D:\ShanghaiTech\2021-Fall\ZhaoXin\Textbook\Note\Img\Fig3.14.png" align=left style="zoom:60%;" />

注意图中有 4 个 bank

> 假设有两条指令，其中i1为mad 指令（乘加指令）以及i2 为add(加指令）。i1指令使用到源寄存器位r5、r4、r6分别使用bank1、bank0、bank2，目的寄存器位r2，使用的是bank2。i2 指令使用到的源寄存器为r5和r1，都是用到的bank1，目的寄存器为r5。
>
> 假设有这样一个时钟执行周期顺序，第1个时钟周期w3执行i1指令，第2个时钟周期w0执行i2指令，第4个时钟周期w1执行i2指令。在开始执行第一个w3:i1指令时，首先需要从源数据中读取数据，由于r5,r4,r6分别位于不同的bank则可以同时进行读取这三个寄存器数据分别占用bank0，bank1,bank2。在下一个时钟周期切换到w0:i2中 由于r5和r1使用相同bank 1,故只能一次读取一个寄存器 ，首先读取r1寄存器。接着下个时钟周期，w3计算完毕需要将结果保存到r2中，需要占用bank1。同时执行w0：i2指令继续执行读取r5值。到第4个时钟周期 切换到指令w1:i2指令，同样首先读取i2的值。第5个时钟周期将w0:i2指令结果保存到r5，同时w1要读取r5寄存器的值指令由于bank被占用 无法读取，只能在下一个周期执行。
>
> 由上面执行可以看到由于r5和r1寄存器的bank冲突，只能使程序串行，无法完全并行。

#### 2. 扩展, Bank and Bank Conflict

> bank 是CUDA中一个重要概念，是内存的访问时一种划分方式，在CPU中，访问某个地址的内存时，为了减少读写内次次数，访问地址并不是随机的，而是一次性访问bank内的内存地址，类似于内存对齐一样，一次性获取到该bank内的所有地址内存，以提高内存带宽利用率，一般CPU认为如果一个程序要访问某个内存地址时，其附近的数据也有很大概率会在接下来会被访问到。
>
> 在CUDA中 在理解bank之前，需要了解共享内存。

即空间局部性

##### I. Shared Memory

这个是 GPU 中非常常见的

> shared memory为CUDA中内存模型中的一中内存模式，为一个片上内存，比全局内存（global memory)要快很多，在同一个block内的所有线程都可以访问到该内存中的数据，与local 或者global内存相比具有高带宽、低延迟的作用。
>
> Because it is <u>on-chip</u>, shared memory has much <u>higher bandwidth and much lower latency</u> than local or global memory.

为了提高share memory的访问速度 除了在硬件上采用片上内存的方式之外，还采用了很多其他技术。其中为了提高内存带宽，shared memory 被划分为相同大小的内存模型，称之为 bank,，这样就可以将n个地址读写合并成n个独立的bank，这样就有效提高了带宽。

> To achieve high bandwidth, shared memory is divided into equally-sized memory modules, called banks, which can be accessed simultaneously. Any memory read or write request made of n addresses that fall in n distinct memory banks can therefore be serviced simultaneously, yielding an overall bandwidth that is n times as high as the bandwidth of a single module.

所以什么是Bank？

+ 被划分后的 shared memory 的单元
+ 一个 Bank 被一个 thread 独享

映射关系如下所图

<img src="D:\ShanghaiTech\2021-Fall\ZhaoXin\Textbook\Note\Img\Bank.png" align=left style="zoom:60%;" />

如上图共享内存映射为bank采用列映射方式，例如warp size = 32, banks = 16,（计算能力1.x的设备）数据映射关系如下

<img src="D:\ShanghaiTech\2021-Fall\ZhaoXin\Textbook\Note\Img\Bank_1.png" align=left style="zoom:60%;" />

例如对于一个 32*32大小的float数组

```cuda
__shared__ float sData[32][32];
```

在一个warp size = 32,bank=32的GPU中 中bank的映射关系为

<img src="D:\ShanghaiTech\2021-Fall\ZhaoXin\Textbook\Note\Img\Bank_2.png" align=left style="zoom:60%;" />

##### II. Bank Conflicts

如果在block内多个线程访问的地址落入到同一个bank内，那么就会访问同一个bank就会产生bank conflict，这些访问将是变成串行，在实际开发调式中重视bank conflict.

> However, if two addresses of a memory request fall in the same memory bank, there is a bank conflict and the access has to be serialized. The hardware splits a memory request with bank conflicts into as many separate conflict-free requests as necessary, decreasing throughput by a factor equal to the number of separate memory requests. If the number of separate memory requests is n, the initial memory request is said to cause n-way bank conflicts.

<img src="D:\ShanghaiTech\2021-Fall\ZhaoXin\Textbook\Note\Img\Bank_3.png" align=left style="zoom:60%;" />

part1, 2, 3 访问同一个 bank 会造成 bank conflicts，造成程序串行而影响性能

> 在上述数组例子中，如果有多个线程同时访问一个列中的不同数组将会产生bank conflict

这个时候可能就需要程序员自己来避免访问 一个列 中的不同数组

> 如果多个线程同时访问同一列中相同的数组元素 不会产生bank conflict，将会出发广播，这是CUDA中唯一的解决方案，在一个warp内访问到相同内存地址，将会将内存广播到其他线程中，同一个warp内访问同一个bank内的不同地址貌似还没看到解决方案。
>
> 不同的线程访问不同的bank，不会产生bank conflict

##### III. Modify Bank Size

以下内容设计到 CUDA 编程的知识

注意 shared memory 中每个SM的bank数量目前是无法修改的，但是可以修改bank中单个数组元素容纳的字节数，例如上个例子中

```cuda
__shared__ float sData[32][32];
```

每个数组元素大小为4个字节，一般cuda中默认是按照4个字节进行组织被划分到bank中，CUDA提供可修改按照8个字节进行组织API:

```cuda
__host__cudaError_t cudaDeviceSetSharedMemConfig ( cudaSharedMemConfig config )
```

其中 cudaSharedMemConfi为一个枚举型：

```cuda
cudaSharedMemBankSizeDefault = 0

cudaSharedMemBankSizeFourByte = 1

cudaSharedMemBankSizeEightByte = 2
```

只支持在host端进行调用，不支持在device端调用。

CUDA API中还支持获取bank size大小：

```cuda
__host__ __device__ cudaError_t cudaDeviceGetSharedMemConfig ( cudaSharedMemConfig ** pConfig )
```



### 3.3.2 Instruction Replay: Handling Structural Hazards  

1 介绍背景，为什么需要 instruction replay  

CPU 中通过 stall 处理结构危害

但是 多线程架构不建议这么做，why？

+ 对于 a full graphics pipeline, a stall signal may impact the critical path. Stall 所需的额外 buffer 也会增加面积
+ 一个 warp 的指令停顿可能会造成其他 warp 的指令停顿

2 引出 instruction replay  

instruction replay 源于 CPU 设计的一个推测机制。根据上一条指令的情况预测下一条指令是否命中

> Instead, instruction replay is used in GPUs to avoid clogging the pipeline and the circuit area and/or timing overheads resulting from stalling.  

3 2015 年的工作有介绍这个方法

## 3.4 Research Directions on Branch Divergence  

开始介绍一些研究方向了

### 3.4.1 Warp Compaction

warps are usually running the same compute kernel, thus, 很可能 follow the same execution path, and encounter branch divergence at the same set of data-dependent branches.   

主要就是处理 branch divergence, 几个主题如下

##### I. Dynamic Warp Formation  07

compact 有分支的 -> non-divergent warp

11 年有人指出了这个方法的 performance pathologies，即性能问题。1.starve some threads to reduce SIMD efficiency. 2.Thread regrouping in DWF increases non-coalesced memory accesses and shared memory bank conflicts.  

##### II. Thread Block Compaction  11

压缩的调度单元更大，but 减少了 available TLP

遇到 divergence, TB 中的其他 warp will keep HW busy

相比 DWF, more robust and simple mechanism  

##### III. Large Warp Microarchitecture  11

> LWM requires warps within the group to execute in complete lockstep, so that it can compact the group at every instruction.   

比 TBC 减少了更多 available accuracy

##### IV. Compaction-Adequacy Predictor  12

> only synchronizes the threads at branches where the compaction is predicted to yield a benefit.  
>
> a simple history-based predictor similar to a single-level branch predictor is sufficient to achieve high accuracy  

同步会获得 benefit 的线程

##### V. Intra-Warp Compaction   13

> divides a single execution group into multiple subgroups that match the hardware width.  
>
> SIMD execution group that suffers from divergence can run faster on the narrow hardware by skipping subgroups that are completely idle.  

##### VI. Simultaneous Warp Interweaving   12

> They extend the GPU SIMT front-end to support issuing two different instructions per cycle.  
>
> 每个 cycle 发射两条不同的 instructions
>
> They compensate this increased complexity by widening the warp to twice its original size  
>
> 把 warp size 扩为原来的两倍

##### VII. Impact on Register File Microarchitecture   07

> Registers for threads in the same warp are stored in consecutive regions in the same SRAM bank, so that they can be accessed together via a single wide port. 
>
> 同一个 warp 的线程的寄存器，存储在相同的 SRAM bank 的连续区域  

这一点在学习了 第四章 memory system 后应该会理解得更深入

##### VIII. Dynamic Micro-Kernels  10

> The programmer is given primitives to break iterations in a data-dependent loop into successive micro-kernel launches.  

##### IX. Warp Compaction in Software  00-15

> The regrouping involves moving the thread and its private data in memory, potentially introducing a significant memory bandwidth overhead.  
>
> 重新组合涉及将线程及其私有数据移动到内存中，可能引入显着的内存带宽开销。

10年，Zhang 提出的一个 runtime system, remaps thread into different warps on the fly to improve **SIMD efficiency** as well as memory access spatial locality  

> 15年，Collective Context Collection (CCC)
>
> a compiler technique that transforms a given GPU compute kernel with potential branch divergence penalty to improve its SIMD efficiency on existing GPUs. 
>
> 对潜在的分支一定的惩罚值  

##### X. Impacts of Thread Assignment within a Warp

> threads with consecutive thread IDs are statically fused together to form warps  
>
> 带有连续线程ID的线程静态融合在一起以形成 warp
>
> 不过一些研究在寻找 替代方案

##### XI. SIMD Lane Permutation  13

> A key limitation of most warp compaction and formation work is that when threads are assigned to a new warp, they cannot be assigned to a different lane, or else their register file state would have to be moved to a different lane in the vector register.  

不能被分配给不同的 lane，否则 register file state 也必须移动到不同的 lane

##### XII. Intra-warp Cycle Compaction 13

> the width of the SIMD datapath does not always equal the warp width  
>
> For example, in NVI [2009], the SIMD width is 16, but the warp size is 32. 32-thread warp 执行需要 2 core cycle

##### XIII. Warp Scalarization 14

### 3.4.2 Intra-Warp Divergent Path Management 

##### I. Multi-Path Parallelism  

> Each warp-split consists of threads following the same branch target  
>
> 走了同一分支的的线程称为 warp-split

# 4. Memory System

GPU computing kernels  通过 ld/st 指令和 memory system, such as texture, constant, and render surfaces 进行交互

现代 GPU 中存在由程序员进行管理的 scratchpad memory  

## 4.1 Fitst-Level Memory Structures

focus on the unified **L1 data cache** and scratch pad “**shared memory**” and how these interact with the core pipeline  

### 4.1.1 Scratchpad Memory and L1 Data Cache

bank conflict 的处理是关键

> A bank conflict arises when more than one thread accesses the same bank on a given cycle  

1

>In some architectures the L1 cache contains only locations that are not modified by kernels  

一些架构中 L1 cache 是没法被 kernel 修改的。因为根据了解，有的架构中，程序员可以手动调整 L1 cache and shared memory size

讲述 coalesced，意为所有线程都访问一个 L1 data cache block，这样只会发生一次 cache miss, 后续都会命中。而 uncoalesced  means threads within a warp access different cache blocks, then multiple memory accesses need to be generated  

2 GPU cache 组织

>The design supports a non-stalling interface with the instruction pipeline by using a replay mechanism when handling bank conflicts and L1 data cache misses.  

##### I. Shared Memory Access Operations  

1

>If the requested addresses would cause one or more bank conflicts, the arbiter splits the request into two parts.   

first part 是 warp 中没有 bank conflict 的 thread 子集, second part 反之。second part 的请求返回到 instruction pipeline，可能是下个 cycle 重新执行。This subsequent execution is known as a “replay.”  

trade-off, power 换 area

2

> shared memory is direct mapped  

shared memory 是直接映射

>the latency of the direct mapped memory lookup is constant in the absence of bank conflicts.  

直接映射的存储器查找的延迟在没有 bank 冲突的情况下是恒定的。

tag unit 决定 thread's request 映射到哪个 bank，以便 control the address crossbar which distributes addresses to the individual banks within the data array  

bank 的配置

> Each bank inside the data array is 32-bits wide and has its own decoder allowing for independent access to different rows in each bank.  

##### II. Cache Read Operations  

1

>The L1 cache block size is 128 bytes in Fermi and Kepler and is further divided into four 32-byte sectors [Liptay, 1968] in Maxwell and Pascal [NVIDIA Corp.].   

128B cache block size 被分为4个32-byte sector

>Each 128-byte cache block is composed of 32-bit entries at the same row in each of the 32 banks.  

2 arbiter 如何工作

> The arbiter may reject a request if enough resources are not available.  

arbiter 是数字电路中的一个具体模块

3

访问 tag unit, 当 cache miss, arbiter 通知 LD/ST unit replay the request 并且并行地将请求信息发送到待处理的请求表 pending request table  （PRT）

4

>Page-based virtual memory is still advantageous within a GPUs even when it is limited to running a single OS application at a time, because it helps simplify memory allocation and reduces memory fragmentation.   

即使当它一次仅限于运行单个OS应用程序时，基于页表的虚拟内存在 GPU 中 也是有利的，因为它有助于简化内存分配并减少内存碎片。

##### III. Cache Write Operations  

1

> The L1 data cache in Figure 4.1 can support both write through and write back policies.  

Accesses to global memory in many GPGPU applications  开销是很大的

>For such accesses a write through with no write allocate [Hennessy and Patterson, 2011] policy might make sense.  

3

> Note that the cache organization described in Figure 4.1 does <u>not support cache coherence.</u>  

怎么理解呢？SM 1 reads memory location A and the value is stored in SM 1's L1 data cache. Then SM 2 writes memory location A. 这个时候 memory location A 发生改变，但 SM 1's L1 data cache 还未更新，这时 SM 1 的下一个线程  reads memory location A, it would read the old value.

### 4.1.2 L1 Texture Cache

>Recent GPU architectures from NVIDIA combine the L1 data cache and texture cache to save area.   

将 L1 data cache and texture cache 合并了

1

>To achieve this realism with the high frame rates required for real time rendering, graphics APIs employ a technique called texture mapping [Catmull, 1974].  

在3D图形中，希望使场景尽可能逼真。为了实现这种现实主义，通过实时渲染所需的高帧速率，图形API采用称为纹理映射的技术[Catmull，1974]。

texture 使得图形表面看起来更加真实。

2 Fig4.2 给出了 texture cache 微架构

> 与第4.1.1节中描述的 L1 data cache 相比，tag array and data array 由FIFO 缓冲器分隔。

3, 4 介绍 unit 的具体作用 and how texture cache works

### 4.1.3 Unified Texture and Data Cache

> In recent GPU architectures from NVIDIA and AMD caching of data and texture values is performed using a unified L1 cache structure.  To accomplish this in this most straightforward way, only data values that can be guaranteed to **read-only** are cached in the L1.  

如上所述，倾向于合并 L1 data cache and texture cache

## 4.2 On-Chip Interconnection Network

>The SIMT cores connect to the memory partition units via an on-chip interconnection network. The on-chip interconnection networks described in recent patents for NVIDIA are crossbars [Glasco et al., 2013, Treichler et al., 2015]. GPUs from AMD have sometimes been described as using ring networks [Shrout, 2007].  

crossbars 来实现 Interconnection Network，2015年提出

## 4.3 Memory Partition Unit

>The L2 cache contains both graphics and compute data.  

### 4.3.1 L2 Cache

>To optimize throughput in the common case of coalesced writes that completely overwrite each sector on a write miss no data is first read from memory.  

在写入未命中时，完全覆盖每个扇区的 coalesced writes 的常见情况下，为了优化吞吐量，不会首先从内存中读取数据。

>To reduce area of the memory access scheduler, data that is being written to memory is buffered in cache lines in the L2 while writes awaiting scheduling.  

为了减少内存访问调度器的面积，正在写入内存的数据在等待调度的写入时缓存在 L2 的缓存行中。

### 4.3.2 Atomic Operation

>A sequence of atomic operations accessing the same memory location can be pipelined as the ROP unit includes a local ROP cache.  

### 4.3.3 Memory Access Scheduler

>To, for example, read values from these capacitors a row of bits, called a page, is first read into a small memory structure called a row buffer.   

例如，为了从这些电容器读取值，首先将一行称为页的位读取到称为行缓冲区的小型存储器结构中。

>To accomplish this operation the bitlines connecting the individual storage capacitors to the row buffer, and which have capacitance themselves, must first be precharged to a voltage half way between 0 and the supply voltage.   

为了实现该操作，将单个存储电容器连接到行缓冲器的位线 (bitlines)，并且它们本身具有电容，必须首先将其预充电到0和 supply voltage 之间的一半。

> To mitigate these overheads DRAMs contain multiple banks, each with their own row buffer.   

注意理解 bank 的概念

1

>To enable access to DRAM, each memory partition in the GPU may contain multiple memory access schedulers [Keil and Edmondson, 2012] connecting the portion of L2 cache it contains to off-chip DRAM.   

为了能够访问 DRAM，GPU 中的每个内存分区可能包含多个内存访问调度程序 [Keil 和 Edmondson，2012]，将其包含的 L2 缓存部分连接到片外 DRAM。

## 4.4 Research Directions For GPU Memory Systems

### 4.4.1 Memory Access Scheduling and Interconnection Network Design 09 13

>They observe that requests generated by a single streaming multiprocessor (SM) have row-buffer locality.   

由单个流式多处理器 (SM) 生成的请求具有行缓冲区局部性 row-buffer locality

如果出现在序列附近的请求访问同一 DRAM 组中的同一 DRAM 行，则对给定内存分区的内存请求序列称为具有行缓冲区位置。

然而，当来自一个 SM 的内存请求被发送到内存分区时，它们会与来自其他 SM 的请求混合到同一内存分区。

结果是进入内存分区的请求序列的行缓冲区局部性较低。

>Yuan et al. [2009] propose reducing the complexity of memory access scheduling by modifying the interconnection network to maintain row buffer locality.  

1

>Bakhoda et al. [2010, 2013] explore the design of on-chip interconnection networks for GPUs.   

### 4.4.2 Caching Effectiveness 09 12

> Bakhoda et al. [2009] studied the impact of adding L1 and/or L2 caches for global memory

通过 GPGPU-Sim 来进行研究实验

1

12年 Jia 等人延续了这一实验。

> cache hit rates alone are insufficient to predict whether caching will improve performance.  

仅缓存命中率不足以预测缓存是否会提高性能。相反，他们发现有必要考虑缓存对L2缓存(例如，内存分区)的请求流量的影响。

Within warp locality: 单个 warp 中不同线程访问同一个 cache block  

Within block locality: 单个 block 中不同 warp 访问同一个 cache block

Cross-instruction locality: memory read access from different load instructions execute by threads in the same thread block access the same cache block.  

当来自由同一线程块中的线程执行的不同加载指令的存储器读访问访问同一高速缓存块时，发生跨指令局部性。

### 4.4.3 Memory Request Prioritization and Cache Bypassing 16

>work by Rogers et al. [2012] which demonstrated warp scheduling can improve cache effectiveness (described in Section 5.1.2), Jia et al. [2014] proposed memory request prioritization and cache bypassing techniques for GPUs.  

Jia等人[2014]提出了GPU的内存请求优先化和缓存旁路技术，这些研究证明了 warp scheduling 可以提高 cache effectiveness (在第5.1.2节中描述)。

相对于线程数量而言，关联度较低的缓存可能会遭受严重的冲突遗漏[Chen和Aamodt，2009]。

1 cross-warp contention  

>This form of cache contention results when one warp evicts data brought in by another warp.  

一个 warp evict 另一个 warp 的 data

>To address this form of contention, Jia et al. [2014] suggest employing a structure they call a “memory request prioritization buffer” (MRPB).   

贾等人[2014]建议采用一种他们称为“内存请求优先化缓冲区” (MRPB) 的结构

像CCWS [Rogers等人，2012]一样，MRPB通过修改对高速缓存的访问顺序来增加局部性，从而减少容量未命中。

然而，与通过线程调度间接实现这一点的CCWS不同，MRPB试图通过在线程被调度后改变单个内存访问的顺序来增加局部性。

means 只改变访存的顺序而不改变线程的顺序？

2 介绍 MRPB 如何工作

MRPB就在 first-level data cache 之前实现 memory request 重新排序。

3

> 详细的评估显示，使用MRPB旁路和重新排序的组合机制在64路16 KB上实现了4%的几何平均加速。

4

> 与Rogers等人[2013]类似，Jia等人[2014]表明，他们的程序员透明的提高性能的方法可以缩小使用缓存的简单代码和使用暂存共享内存的更高度优化的代码之间的差距。

5

> Arunkumar等人[2016]基于静态指令中存在的内存差异级别，探索了旁路和改变高速缓存行大小的效果。他们使用观察到的重用距离模式和内存差异程度来预测旁路和最佳缓存行大小。

6

> Lee和Wu  [2016]提出了一种基于控制环的缓存旁路方法，试图在运行时逐个指令地预测重用行为。监控高速缓存行的重用行为。如果由特定程序计数器加载的高速缓存行没有经历足够的重用，则绕过对该指令的访问

### 4.4.4 Expleiting Inter-Warp Heterogeneity 15

> Ausavarungnirun等人[2015]在GPU的共享L2和内存控制器上提出了一系列改进，以减轻不规则GPU应用中的内存延迟差异。

这些技术被统称为 Memory Divergence Correction (MeDiC)，

memory divergence 也是一个很值得研究的点

>The authors demonstrate that there is little benefit in having warps that are not all hit, since warps the mostly hit must wait for the slowest access to return before they are able to proceed.  

这段话意思是 命中最多的 warp 必须等待最慢的访问返回，然后才能继续。因此平衡整体的 hit rate 才能够提升性能

> 他们还证明了L2缓存的排队延迟会对性能产生不小的影响，并且这种影响可以通过为所有请求(甚至是那些可能命中的请求)绕过L2缓存来减轻，因为所有 warp 都没有被全部命中。

通过减少 queueing latency 来减少 access latency

> 他们还证明，即使对于全命中的 warp，L2 cache bank 之间的 queueing latency 差异也可能导致额外的潜在可避免的 queueing latency，因为L2 bank 之间的 queueing latency 不平衡

1

The microarchitectural mechanism proposed by the authors consists of four components  

+ a warp-type detection block  
  + 它将GPU中的扭曲分类为五种潜在类型之一:All-miss, mostly-miss, balanced, mostly-hit, or all-hit  
+ a warp-type-aware bypass logic block which decides if requests should bypass the L2 cache  
  + warp 类型感知旁路逻辑块，其决定请求是否应该绕过 L2 cache
  + 因为 miss 后绕过会减少不必要的延迟
+ a warp-type-aware insertion policy, which determines where insertions in the L2 will be placed in the LRU stack  
  + 确定L2中的插入将被放置在LRU堆栈中的什么位置
+ a warp-type-aware memory scheduler that orders how L2  misses/bypasses are sent to DRAM  
  + 其命令如何将 L2 未命中/旁路发送到 DRAM

2 detection mechanism

> 检测机制通过以时间间隔为基础对每个扭曲的命中率 (总命中/访问 total hits/accesses) 进行采样来运行。

3 bypass mechanism

> 旁路机制位于 L2 cache 的前面，接收标记有产生请求的 warp 类型的内存请求。这种机制试图消除所有未命中 warp 的访问，并将大部分未命中 warp 转换为所有 All-miss warp。该块简单地将所有标记为来自全部未命中和大部分未命中 warp 的请求直接发送到内存调度器。

索性直接将 mostly-miss 转换为 all-miss，重新发送访存请求

4, 5 MeDiC  

> MeDiC 的缓存管理策略通过改变从 DRAM 返回的请求在 L2 LRU 堆栈中的位置来运行。mostly-miss warp 请求的缓存行被插入到 LRU 位置，而所有其他请求被插入到传统的 MRU 位置。

### 4.4.5 Coordinated Cache Bypassing :star: Interesting 15

> 谢等人[2015]探索了选择性启用高速缓存旁路以提高高速缓存命中率的潜力。

选择性 enable cache，感觉挺有趣的

> profiling GPGPU 应用程序的每个 static load instruction 是具有好的局部性，差的局部性还是中等的局部性。
>
> 好的局部性的 load operation 可以使用 L1 cache，而差的被绕过。中等局部性采用一种自适应机制，给定一个阈值。

### 4.4.6 Adaptive Cache Management 14

> Chen et al. [2014b] propose coordinated cache bypassing and warp throttling that takes advantage of both warp throttling and cache bypassing to improve performance on highly cachesensitive applications.  

Chen等人[2014b]提出了协调 cache bypassing and warp throttling，其利用 warp throttling and cache bypassing 来提高高速缓存敏感应用的性能。

> 该机制通过现有的保护距离的中央处理器高速缓存旁路技术来实现高速缓存旁路，该技术防止高速缓存线由于多次访问而被逐出。
>
> 在插入到高速缓存中时，线路被分配一个保护距离，计数器跟踪线路的剩余保护距离。一旦剩余保护距离达到0，线路将不再受到保护，并且可以被逐出。
>
> 当新的内存请求试图将新的行插入到没有未受保护的行的集合中时，内存请求会绕过缓存。

1

> 保护距离是全局设置的，最佳值因工作负载而异。在这项工作中，Chen等人[2014b]扫描了静态保护距离，并证明了GPU工作负载对保护距离值相对不敏感

### 4.4.7 Cache Prioritization 15

> 他们提出了一种为 warp 分配标记的机制，以确定哪些 warp 可以将 lines 分配到 L1 cache中。
>
> Additional “non-polluting warps” are not given a token so that while they can execute they are not permitted to evict data from the L1.  
>
> 额外的“无污染扭曲”没有被赋予一个标记，因此虽然它们可以执行，但不允许从 L1 驱逐数据。

意为不能将别的 data 从 L1 cache evict

> 这导致了一个优化空间，其中可以调度的 warp 数量(W)和具有标记的数量 (T) 都可以设置为小于可以执行的最大 warp 数量。他们表明，静态选择最佳的 W 和 T 值能够比静态 warp 限制的 CCWS 提高17%。

1 基于这一观察，李等人[2015]探索了两种机制来学习W和T的最佳值。

### 4.4.8 Virtual Memory Page Placement 15

> Agarwal等人[2015]观察了当前的操作系统页面放置策略，例如在 Linux 中部署的策略，这些策略没有考虑内存带宽的不均匀性。
>
> Agarwal et al. [2015] 研究了一种未来的系统，在这种系统中，图形处理器可以以低延迟访问低带宽/高容量的中央处理器内存，100 core cycle。他们的实验使用了配置了额外 MSHR 资源的 GPU-Sim  3.2.2 的修改版本来模拟最近的 GPU。

1

> 通过这种设置，他们首先发现，对于内存带宽有限的应用程序，通过使用 CPU 和 GPU 内存来增加总内存带宽，有很大的机会获得性能。

2

> 为了改进页面放置，他们提出了一个系统，该系统涉及使用NVIDIA开发工具nvcc和ptxas的修改版本以及现有CUDA  API的扩展来实现的分析过程，以包括页面放置提示。使用配置文件引导的页面放置提示可以获得oracle页面放置算法90%的好处。他们把页面迁移策略留给未来的工作。

### 4.4.9 Data Placement 14

> 由于GPU上有各种类型的可用内存，选择哪些数据应该放在程序员难以确定的地方，并且通常不能从一个GPU架构移植到下一个。
>
> PORPLE的目标是可扩展的、输入自适应的，并且通常适用于常规和不规则的数据访问。他们的方法依赖于三种解决方案。

1

> 第一个解决方案是内存规范语言，以帮助实现可扩展性和可移植性。内存规范语言根据对这些空间的访问被序列化的条件描述了图形处理器上所有不同形式的内存。例如，对相邻全局数据的访问是合并的，因此是并发访问，但对同一组共享内存的访问必须序列化。

2

> 第二种解决方案是名为 PORPLE-C 的源到源编译器，它将原始 GPU 程序转换为与布局无关的版本。 编译器在对内存的访问周围插入保护，选择与预测的数据最佳位置相对应的访问。

3

> 最后，为了预测哪种数据放置是最理想的，他们通过代码分析使用propele-C来寻找静态访问模式。

### 4.4.10 Multi-Chip-Modules GPUs 17

> Arunkumar等人[2017]指出，摩尔定律的放缓将导致 GPU 性能的增长放缓。他们建议通过在多芯片模块上用较小的 GPU 模块构建大型 GPU 来扩展性能扩展(见图4.4)。

理解为多任务 GPU

> 他们证明，通过结合远程数据的本地缓存、考虑局部性和第一次接触页面分配的模块的 CTA 调度，可以获得单个大型(且不可实现)单片 GPU  10%的性能。根据他们的分析，这比在同一工艺技术中使用最大的可实现单片图形处理器的性能好45%

# 5. Crosscutting Research on GPU Computing Architectures

## 5.1 Thread Scheduling 

并行性是 GPU 的一大特征，GPU采用几种机制来聚合和调度所有这些线程。线程有三种主要的组织和调度方式



# Reference

https://blog.csdn.net/weixin_42730667/article/details/109838089

