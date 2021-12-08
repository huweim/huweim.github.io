---
title: "Ca2_lab0"
date: 2021-07-28T15:23:39+08:00
draft: false
tags: ["CA2", "lab"]
categories: ["course"]
---

# 0. 前言

很久之前就想总结一下 Computer Architecture II (CA2) 这门课上学得一些东西了，尤其是关于这几个 lab。当时无论是在 Linux, C++, 还是体系结构方面，都帮助我加深了理解。现在试着整理也是复习一下，把他放在博客的文章中。

# 1. Goal

>Through this lab, you would compile Sniper, a next-generation parallel, high-speed and accurate x86 simulator **[0.5 points]**. Then you need to modify the source code of Sniper to show expected output and upload a summary to BlackBoard **[1.5 points]**. Total is 2 points.

提供了一个 C 代码文件 `toy-lab0.c`， 编译并运行模拟器 Sniper。然后修改 Sniper 源代码，找到 `CLFLUSH` 这条指令，在这条指令每次执行后打印 `[STUDENT-EMAIL-ACCOUNT, function, line number] CLFLUSH instruction executed`

修改前

```bash
...
User program begins
<toy-lab0.c, clflush, 21> clflush to be run
<toy-lab0.c, clflush, 21> clflush to be run
<toy-lab0.c, clflush, 21> clflush to be run
<toy-lab0.c, clflush, 21> clflush to be run
User program ends
...
```

修改后

```bash
...
User program begins
<toy-lab0.c, clflush, 21> clflush to be run
[STUDENT-EMAIL-ACCOUNT, function, line number] CLFLUSH instruction executed
<toy-lab0.c, clflush, 21> clflush to be run
[STUDENT-EMAIL-ACCOUNT, function, line number] CLFLUSH instruction executed
<toy-lab0.c, clflush, 21> clflush to be run
[STUDENT-EMAIL-ACCOUNT, function, line number] CLFLUSH instruction executed
<toy-lab0.c, clflush, 21> clflush to be run
[STUDENT-EMAIL-ACCOUNT, function, line number] CLFLUSH instruction executed
User program ends
...
```

# 2. Report

经历许多天的代码研究，也经历了很多曲折，终于用一种比较笨的方法得到了要求的结果。先说一下结果，最后通过修改trace_thread.cc文件中 656 行的 handleInstructionDetailed 函数，对指令结构体中 sinst 的 data[] 这个数组进行逻辑等运算来达到输出效果。

​		刚开始做的时候，在很多文件中进行 printf 操作，观察各个打印在各个地方的位置。最初是在standalone.cc中观察，在最后发现他做的是指令执行前后的事情，没有进入到那条指令中。而 “asm volatile("clflush %0" : "+m" (*(volatile char *)ptr));” 这个语句明显告诉我们和 CLFLUSH 有关。联想到 ISA 的过程，以及指令执行的几个阶段，真正想进入指令的中间需要找到别的文件。因为 PDF 中写了 xed 这个文件夹是进行 instruction decoder，所以过了一遍 xed 文件夹中的内容。通过 printf 操作发现很多文件并没有执行，重新思考了一下，终于定位到 sift_writer 和 sift_reader 这两个文件。

​		在指令执行过程中多次调用了 sift_writer.cc 中的 Instruction 函数，而我认为函数的传入参数是比较重要的，能够定位到 CLFLUSH这条指令的关键。在和 CLFLUSH 有关的文件中去找了很久，但是因为知识有限，没有能够找到 CLFLUSH 在 Instruction 函数对应的参数。随后，就想出了一个比较笨的方法：把传入的参数，如 size、addr 这些都打印出来，观察是否能找到唯一的参数。事实证明，这个方法可以筛选出一部分数据，但我仍然没法确定哪一个是 CLFLUSH，而且并不能够完全过滤，还是会进行很多次操作。这个方法可以说给了我一些启发，但是并不能算成功，于是又去 sift_reader.cc 文件中观察，过了一遍整个文件，Instruction 这个结构体引起了我的注意，我认为这里面会存放指令的信息，跟随过去发现果然有所收获。这个结构体存放了 addr、size、num_addresses，还有 is_branch、taken、is_predicate 等描述指令状态的布尔变量。

​		最关键的是，在 const StaticInstruction *sinst 中有一个数组 data[]，看到这里收获很大，因为此前的字符串匹配无法完全过滤，但是如果是数组的话，数组里面的地址指向的数据，加上顺序存放的一个特性，只要取前面几个数据进行逻辑相等运算，就可以在形式上实现在 print“<toy-lab0.c, clflush, 21> clflush to be run” 之后，再次输出一个 printf 语句的方法。

​		这就是在很多天对源代码研究的过程中想出的一个不太巧妙的方法，成功实现了要求，但是对于指令的处理并不是非常完美。因为最终没有能够定位到 CLFLUSH 这条指令，我认为应该是可以找到 CLFLUSH 指令的一些特征，并且能够通过源代码实现 “CLFLUSH instruction executed” 这条语句，因为指令中是含有 executed 这一变量，来对指令是否执行进行一个标记。

2021/07/28: 确实是可以根据指令 CLFLUSH 的 operand code 来唯一确定这条指令的。不过当时我是先将所有执行的指令打印出来，然后根据二进制数去进行筛选最终过滤的。后来发现可以查指令操作码来直接确定，这个也算是弯路之一。

# 3. Result

## Code(Update)

```
    int a[4]={0x0F,0xAE,0x3b,0};
    if((a[0]==inst.sinst->data[0])&&(a[1]==inst.sinst->data[1])&&(a[2]==inst.sinst->data[2])&&(a[3]==inst.sinst->data[3])){
       printf("[huwm1@shanghaitech.edu.cn, %s, %d] CLFLUSH instruction executed \n",__func__,__LINE__);
    }
```

## Output(Update)

![image-20200922210848781](Img\Output.png).
