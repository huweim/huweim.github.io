---
title: "GPGPU-Sim中的CTA & warp scheduling"
date: 2021-11-14T20:42:52+08:00
lastmod: 
draft: false
author: "Cory"
tags: ["GPGPU-Sim", "Warp Scheduling"]
categories: ["编程"]
---

# CTA Scheduling

CTA/Thread Block/Work Group

调度发生在 `shader_core_ctx::issue_block2core(...)`，`shader_core_config::max_cta(...)` 计算 core 中能容纳的 max CTA。这个取决于各硬件资源的短板，在报告中能看到是被什么限制了。

```c++
printf("GPGPU-Sim uArch: CTA/core = %u, limited by:", result);
    if (result == result_thread) printf(" threads");
    if (result == result_shmem) printf(" shmem");
    if (result == result_regs) printf(" regs");
    if (result == result_cta) printf(" cta_limit");
```

在矩阵乘法代码中 create, verify are both limited by thread, multiple is limited by regs. 

When each thread finishes, the SIMT core calls `register_cta_thread_exit(...)` to update the active thread block's state. CTA 中的所有线程执行完毕后，active CTA 数量减一，允许新的 CTA 在下个周期被调度。

# Warp Scheduling

## 1. From Code

+ A new front-end that models instruction caches and separates the **warp scheduling (issue) stage** from the fetch and decode stage

##### Flow

`shader_core_ctx::issue()` call `scheduler_unit::cycle()`。

In function `scheduler_unit::cycle()` , call `order_warps()` to sort warps according to their priority. 

排序后的 warp 放在 vector `m_next_cycle_prioritized_warps` 中，对其进行遍历来处理这个 vector 中的 warp。

进入 for 循环，拿到 warp id，判断

+ I-Buffer 是否为空；是否处于 waiting 状态。如果都通过，进入一个 while 循环
  + 如果指令是有效的 `if(pI)`
    + 如果出现分支 `if(pc != pI->pc)`，刷掉 I-Buffer
    + 如果没有分支，此时 `valid=true`，指令是有效的。如果通过 scoreboard 检测，终于可以执行了。先读取 active mask 确定要执行哪些线程，然后判断 `pI->op` 是 内存操作 还是 运算操作。如果相应的寄存器可以使用 `has_free()`，则 call `issue_warp()` 将寄存器、指令、active mask、warp id、scheduler id 发送并执行。
    + `warp_inst_issued = true;`
  + else if 下一条指令是有效的
    + ...
  + 如果指令成功发射 `if (warp_inst_issued)`
    + call `do_on_warp_issued(warp_id, issued, iter);`
  + checked++

##### scheduler_size()

scheduler.size 就是2，代表一个 core 中 warp scheduler 的数量

##### lrr 特征

+ 单双数 round-robin 1-3-5-7-9, 2-4-6-8-10

Why? there are two schedulers per SM, an even and odd scheduler that concurrently execute even and odd warps.

##### gto 特征

贪心，没有 stall 的话会看到连续的 warp id 相同

OP: 8 (LOAD) 后面可能会接一个相同 warp id 的 OP: 1 (ALU_OP)

```shell
warp id:24 OP:6	#INTP_OP
warp id:24 OP:8
warp id:24 OP:1
```

### 1.2 构造 scheduler

根据 warp scheduling policy 构造 scheduler。config 文件传入调度策略，`shader_core_ctx::create_schedulers()` 根据传入的字符构造对应的 scheduler 类。

#### 1.3.1 class scheduler_unit

##### I. std::vector< shd_warp_t* > m_next_cycle_prioritized_warps

This is the prioritized warp list that is looped over each cycle to determine which warp gets to issue.

作为 order 函数的 result_list。每个 cycle 遍历这个 list，决定发射哪个 warp。已经按调度策略/优先级排序。size 为 24

##### II. std::vector< shd_warp_t* > m_supervised_warps;

The m_supervised_warps list is **all the warps** this scheduler is supposed to arbitrate between.  This is useful in systems where there is more than one warp scheduler. In a single scheduler system, this is simply all the warps assigned to this core.

这个作为 order 函数的 input_list，里面装了所有待仲裁的 warp。size 为 24

在构造 scheduler 时，通过函数 `add_supervised_war_id(i)` 向 m_supervised_warps 添加 warp。warp 0 -> scheduler 0, warp 1 -> scheduler 1, warp 2 -> scheduler 0...

```c++
    for (unsigned i = 0; i < m_warp.size(); i++) {
        //distribute i's evenly though schedulers;
        schedulers[i%m_config->gpgpu_num_sched_per_core]->add_supervised_warp_id(i);
    }
```



##### III. std::vector< shd_warp_t* >::const_iterator m_last_supervised_issued;

记录上一个被发射的 warp

## 2. From Paper

### 2.1 Warp Scheduler

The warp scheduler connects to the back end of the SIMT  processor and is responsible for issuing instructions.

> warp scheduler 是 front-end and back-end 之间的连接

Because the registers for each warp are independent,  saving and restoring warp states are not needed.

> warp 有自己独享的寄存器，所以无需专门保存 warp state

The warp  scheduler only needs to keep track of those warps whose  instructions are ready for issue. 

> 在 ready intructions 选一个来发射

If none of the warps are ready  for issue or the back-end of the SIMT processor does not have  enough free space, the warps are **stalled** and no instruction is  sent to the back-end of the SIMT processor. 

> 可以在 Output 中观察到，LRR and GTO 的 stall cycle 是不同的

### 2.2 Functional Units

The memory instructions and arithmetic  instructions operate in different pipelines.

Each back-end  pipeline owns a set of dedicated collector units shared in a  pool. The data from the collector units are sent to the functional  units including arithmetic logic units (ALU) and load-store  units (LSU).

When an instruction from a warp occupies the  arithmetic logic units, the other warps may use the load-store  units to enhance instruction level parallelism and hide the  latency.

> 一个 warp 使用 ALU，其他的 warp 指令仍然可以使用 LSU。以此来隐藏延迟，增加并行度

As for the load-store units, the coalescer combines the  memory requests within a warp, which allows fewer requests to  be sent to the L1 cache memory. There are also miss status  holding registers (MSHR) supporting multiple outstanding  requests to external memory, which enhances the memory level  parallelism to hide the latency. 

> coalescing access, MSHR
