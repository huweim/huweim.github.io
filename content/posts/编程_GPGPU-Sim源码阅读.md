---
title: "GPGPU-Sim源码阅读"
date: 2021-09-28T15:51:46+08:00
lastmod: 2021-10-23T10:02:46+08:00
draft: false
author: "Cory"
keywords: [""]
tags: ["GPGPU-Sim"]
categories: ["GPGPU"]
---

# Shader.cc

shader_core_stats 类型含有非常多的数据统计，包括 cycle 数，m_num_decoded_insn, m_num_FPdecoded_insn, m_num_loadqueued_insn, m_num_INTdecoded_insn 等等

m_stats 也就是 shader_core_stats 类型的变量

num_shaer 就是 n_simt_clusters*n_simt_cores_per_cluster，也就是 SIMT Core 的数量

tw_get_oracle_CPL_counter 计算 warp 的 CPL counter 值

### shader_core_ctx::decode 函数

检查 fetch buffer 中的指令是否有效，如有效则进入循环。获得当前指令的 pc，并取指令。

指令用变量 pI1 存储，调用函数 ibuffer_fill, 将 pI 装进对应 warp id 的 I-Buffer, 并将 valid bit 置为1

随后会取下一条指令，用变量 pI2 存储，注意下一条指令的 `pc = pc + pI1 -> isize`。也就是我们常说的 pc = pc + 1, 这里的1实际上是一条指令的长度

每个 warp 有两个 ibuffer slot, 也就是 ibuffer_fill 中的0和1

##### ifetch_buffer_t(address_type pc, unsigned nbytes, unsigned warp_id)

是一个结构体，包含 m_valid, m_pc, m_nbytes, m_warp_id

构造时 valid 直接置为1

其变量作为 fetch 和 decode stage 之间的 pipeline register

理解为用于使得 fetch and decode 可以流水线执行的一个结构体

### shader_core_ctx::fetch 函数

访问内存 (L1 Cache or memory)，获取指令的 pc, size, warp_id

如果 L1 Cache 的 access ready, 也就是已经可以内存访问（之前 Miss 的时候需要的 data 已经从内存中取到了）

如果没有 access ready, 就去找一个 active, 并且在 I-Buffer 中还有空间,  没有在等待 cache miss 的 warp，取其 next instruction from i-cache

> 第3层中的第1个 if 语句检查 warp 是否已经完成执行，第3层中的第2个 if 语句检查当前 warp 对应的 entry 是否已经存储了有效的指令

### issue_warp 函数

free 掉相应的 I-Buffer

### scheduler_unit::cycle()

In function `scheduler_unit::cycle()` , call `order_warps()` to sort warps according to their priority. 

排序后的 warp 放在 vector `m_next_cycle_prioritized_warps` 中，对其进行遍历来处理这个 vector 中的 warp。

:exclamation: 值得注意的是在 order_warp() 后，for 循环会遍历  vector `m_next_cycle_prioritized_warps` 中的所有 warp。而不是发射一个 warp 就重新排序一次。

> 这一点和自己的理解与猜想不太一样

---

进入 for 循环，拿到 warp id，判断

+ I-Buffer 是否为空；是否处于 waiting 状态。如果都通过，进入一个 while 循环
  + 如果指令是有效的 `if(pI)`
    + 如果出现分支 `if(pc != pI->pc)`，刷掉 I-Buffer
    + 如果没有分支，此时 `valid=true`，指令是有效的。如果通过 scoreboard 检测，终于可以执行了。先读取 active mask 确定要执行哪些线程，然后判断 `pI->op` 是 内存操作 还是 运算操作。如果相应的寄存器可以使用 `has_free()`，则 call `issue_warp()` 将寄存器、指令、active mask、warp id、scheduler id 发送并执行。
    + `warp_inst_issued = true; issued++; issued_inst = true`
  + else if 下一条指令是有效的
    + ...
  + 如果指令成功发射 `if (warp_inst_issued)`，执行了 issue_warp() 后会进入这个 if 语句，做一些 warp 发射后的统计信息等等
    + call `do_on_warp_issued(warp_id, issued, iter);`
  + checked++
+ 从 while 循环出来，如果至少有一个 warp 被发射 `if(issued)`，遍历 `m_supervised_warps`，找到那个被发射的 warp，然后将其赋值给 `m_last_supervised_issued`

---

##### scheduler_size()

scheduler.size 就是2，代表一个 core 中 warp scheduler 的数量

## 关于类

阅读一个类，应该先观察他还包含哪些子类，继承自哪个类，从全局上把握他的作用

class opndcoll_rfu_t

+ class op_t
+ class allocation_t
+ class arbiter_t
+ class input_port_t
+ class collector_unit_t
+ dispatch_unit_t

## 地址信息

src/abstract_hardware_model.h

```c++
  struct per_thread_info {
    per_thread_info() {
      for (unsigned i = 0; i < MAX_ACCESSES_PER_INSN_PER_THREAD; i++)
        memreqaddr[i] = 0;
    }
    dram_callback_t callback;
    new_addr_type
        memreqaddr[MAX_ACCESSES_PER_INSN_PER_THREAD];  // effective address,
                                                       // upto 8 different
                                                       // requests (to support
                                                       // 32B access in 8 chunks
                                                       // of 4B each)
  };
...
std::vector<per_thread_info> m_per_scalar_thread;
```

现在我们关注每个 kernel_launch_uid 中的访存信息和打印出来的访存次数是否匹配

+ :heavy_check_mark: 匹配。前两个 kernel (都是 create matrix)  `gpgpu_n_param_mem_insn + gpgpu_n_store_insn = number of memaddr`
+ 不过要注意 Memory Access Statistics 的信息应该是总和而非单一 kernel

# Tracing

## 4个 Cycle() 函数调用关系

4个 cycle() 函数

shader_core_ctx::cycle() 在 issue 中调用 scheduler_unit::cycle(), 这两个应该是负责 SIMT Front 部分，从指令 fetch, decode, 到准备好后的调度发射 (sheduling and issue).

ldst_unit::cycle() 负责各个 memory 的时钟建模，包括 shared memory, L1 latebcy queue, constant menory, texture memory

##### shader_core_ctx::cycle()

- SIMT Core Cluster clock domain = frequency of the pipeline stages in a core clock (i.e. the rate at which `simt_core_cluster::core_cycle()` is called)
  - `simt_core_cluster::core_cycle()` will call `shader_core_ctx::cycle()`

+ :star: `m_thread[tid]->ptx_exec_inst(inst, t);` 用于执行 ptx 指令的执行

```c++
shader_core_ctx::cycle()
|--	writeback();
|--	execute();
	|-- m_fu[n]->cycle(); //m_fu[] contains ldst_unit, sfu_unit, sp_unit
		|-- ldst_unit::cycle();
			|-- writeback();
			|-- m_operand_collector->step();
	|-- issue(register_set &source_reg)
|--	read_operands();
|--	issue();
	|-- scheduler_unit::cycle();
        |-- order_warps();
		|-- m_shader->get_pdom_stack_top_info(warp_id, pI, &pc, &rpc);
		|-- m_shader->issue_warp();
			|-- (*pipe_reg)->warp_inst_t::issue();
			|-- func_exec_inst(**pipe_reg);
				|-- execute_warp_inst_t(inst);
					|-- m_thread[tid]->ptx_exec_inst(inst, t);
						|-- insn_memaddr = last_eaddr();
						|-- inst.set_addr(lane_id, insn_memaddr); //util this, we have 											//the address in the class warp_inst_t
			|-- updateSIMTStack(warp_id, *pipe_reg);
			|-- reserveRegisters(*pipe_reg);
  			|-- set_next_pc(next_inst->pc + next_inst->isize);
|-- decode();
    |-- ibuffer_fill(0, pI1);
    |-- inc_inst_in_pipeline();
|-- fetch();
```

##### scheduler_unit::cycle()

```c++
|-- order_warps();
//if warp is valid and not waiting
|-- const warp_inst_t *pI = warp(warp_id).ibuffer_next_inst();
|-- m_shader->get_pdom_stack_top_info(warp_id, pI, &pc, &rpc);
//deal with the control hazard
//pc is the PC in the top of SIMT stack, pI->pc is the PC in the I-Buffer
//not equal means jump or ohter control hazard
while{
if(pI){
    if (pc != pI->pc) {
		|-- warp(warp_id).set_next_pc(pc);
		|-- warp(warp_id).ibuffer_flush(); //need to flush
	}
	else{
    	if(!m_scoreboard->checkCollision(warp_id, pI)){
       		|-- m_shader->get_active_mask(warp_id, pI);
        	// need to check which pipieline to send, MEM, SP, SFU... 
        	// the only different is the first parameter, register_set *m_XX_out
        	|-- m_shader->shader_core_ctx::issue_warp(*m_mem_out, pI, active_mask, warp_id, m_id);
    	}
	}
}
else if(valid){
    // this case can happen after a return instruction in diverged warp
    |-- warp(warp_id).set_next_pc(pc);
	|-- warp(warp_id).ibuffer_flush(); //need to flush
}
}//while
...
```

执行这个 issue_warp 的时候需要的源操作数的寄存器已经拿到了 (判断 has_free()才会进入这个条件语句)

##### ldst_unit::cycle()

```c++
ldst_unit::cycle()
|-- writeback();
	|-- Scoreboard::releaseRegister();
	|-- warp_inst_complete(m_next_wb);
	|-- clear();
|-- m_operand_collector->step();
	|-- dispatch_ready_cu(); //把 ready_cu 发射到执行单元
		|-- cu->dispatch(); 
	|-- allocate_reads(); // process read requests that do not have conflicts. Map bank 
						  // and collector unit, 连接了 bank 和 cu, 准确到 cu 的哪个操作数
	for()|-- allocate_cu(p);
			|-- allocated = cu->allocate(inp.m_in[i], inp.m_out[i]);//终于找到了
			|-- m_arbiter.add_read_requests(cu); //把 cu 对特定 bank 的读请求入队 
												//m_queue[bank].push_back(op);
	|-- process_banks(); //reset allocation, free cu
|-- move_warp(warp_inst_t *&dst, warp_inst_t *&src); // move src to dst
|-- m_L1T/m_L1C/m_L1D->fill(); //deal with fill request
|-- m_L1T/m_L1C/m_L1D->cycle(); //send next request to lower level of memory
```

在上述的 `allocated = cu->allocate(inp.m_in[i], inp.m_out[i])` 函数中~~为寄存器赋值~~ 确定读哪个bank的哪个寄存器

m_src_op 中装的就是32个 源操作数寄存器，去哪个位置找 寄存器 的详细信息

dispatch() 后会 reset m_src_op, 

##### pipelined_simd_unit::cycle()

用于模拟流水线，移动寄存器的 value..

```c++
if (!m_pipeline_reg[0]->empty()) {
	|-- m_result_port->move_in(m_pipeline_reg[0]); //move src to m_pipeline_reg[0]
    	|-- warp_inst_t **free = get_free(); //return a free register to variable **free
    	|-- move_warp(*free, src); //move src to *free
    active_insts_in_pipeline--;
}
if (active_insts_in_pipeline) {
    for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++)
        |-- move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage + 1]);
}
```

## 文档中 Cycle() 的介绍

##### simt_core_cluster::core_cycle()

`simt_core_cluster::core_cycle()` 方法只是按顺序 循环调用 (cycles) 每个 SIMT core. 

`simt_core_cluster::icnt_cycle()` 方法将内存请求从 interconnection network push 到 SIMT Core Cluster's response FIFO. 它也将 FIFO 中的请求出队，送到合适的 core's instruction cache or LDST unit. 这些与前面描述的硬件块密切对应。

##### shader_core_ctx::cycle()

+ 每个 core cycle, 调用 `shader_core_ctx::cycle()` 来模拟 SIMT Core 的一个 cycle。
+ operand collector 被建模为主流水线中的一个 stage, 通过函数 `shader_core_ctx::cycle()` 执行

##### scheduler_unit::cycle()

+ 在 `scheduler_unit::cycle()` 中，函数 `shader_core_ctx::issue_warp()` 将指令发送到执行单元
+ 调用 `func_exec_inst()` 执行指令
+ 调用 `simt_stack::update()` 更新 SIMT Stack

##### ldst_unit::cycle()

+ ·ldst_unit::cycle()· 处理来自 interconnect 的内存响应（存储在 m_response_fifo 中），填充 cache (`m_L1D->fill()`) 并将存储标记为完成。
+ 该函数还 cycle caches，以便它们可以将 missed data 的请求发送到 interconnect
+ 对每种类型的 L1 内存的 cache accesses 分别在 `shared cycle()`、`constant cycle()`、`texture cycle() `和 `memory cycle()` 中完成 (在 `ldst_unit::cycle()` 函数中调用)

##### gpgpu_sim::cycle()

+ `gpgpu sim::cycle()` 方法为 gpgpu - sim 中的所有架构组件的时钟，包括 Memory Partition 的队列，DRAM channel 和 L2 cache bank.
+ 对 `memory_partition_unit::dram_cycle()` 的调用将内存请求从 L2->dram queue 移动到 dram channele，从 dram channel 移动到 dram->L2 queue，并 cycles 片外 GDDR3 dram 内存。
+ 在这个函数中，调用
  + icnt_cycle()
  + dram_cycle()
  + cache_cycle()
  + core_cycle()

所以可以看到，这个函数应该是调用了每个组件的 cycle(), 以此来建模整个 GPGPU-Sim cycle

##### memory_partition_unit::cache_cycle()

+ 在 `memory_partition_unit::cache_cycle()` 中，调用 `mem_fetch *mf = m_L2cache->next_access();` 为在 filled MSHR entry 中等待的内存请求产生 replies. 
+ L2 产生的由于 read miss 的 fill 请求将从 L2's miss queu 中弹出，并通过调用 `m_L2cache->cycle();` 将其push into L2->dram queue

##### dram_t::cycle()

+ The function `dram_t::cycle()` represents a DRAM cycle
+ 每个周期，DRAM从请求队列中弹出一个请求，然后调用调度器函数，让调度器根据调度策略选择一个需要服务的请求。
