---
title: "Ca2_lab1"
date: 2021-07-28T16:09:45+08:00
draft: false
tags: ["CA2", "lab"]
categories: ["Course"]
---

# 0. 前言

很久之前就想总结一下 Computer Architecture II (CA2) 这门课上学得一些东西了，尤其是关于这几个 lab。当时无论是在 Linux, C++, 还是体系结构方面，都帮助我加深了理解。现在试着整理也是复习一下，把他放在博客的文章中。

# 1. Goal

主要是由两个目标

+ 实现 Cache Replacement Policy 中的 OPT 算法，也就是假设已经得知了对 cache line 的访问序列，每次都 evict 最久之后访问的那个 cache line
+ 将 Sniper 中的 inclusive policy 改为 non-inclusive policy

通过这个 lab 更深刻地理解了一些 cache replacement, 模拟器中访问 cache 和内存的 flow, cache 的地址映射方式等等

# 2. 思路和 Report

## Optimal算法

+ 首先，每一条指令的access操作都会经过函数 AccessSingleLine，在这个地方执行文件 IO 操作。
  + 第一遍执行程序的时候进行文件写入（假设两次执行程序的指令序列是完全一样的），将指令的 set_index 和 tag 写入future_list.txt文件（预先将指令序列写入文件）
  + 第二遍指令的时候，相当于我们是**已知未来序列**的，在第一次调用 AccessSingleLine 的时候，将文件读入一个二维数组future_list，存放所有指令的 set_index 和 tag。后续调用 AccessSingleLine 的时候不再进行文件 IO 操作（写一个条件判断，只执行一次文件 IO 操作）。
  + 二维数组 future_list 中存放了指令 access 序列，将其在 class Cache 中定义，定义为 long long int 型的静态变量，并且有足够大的空间。
+ 根据对 sniper 代码的阅读，此模拟器执行的是 LRU 替换算法，为了不大量修改一些函数接口和逻辑（比如当 cache 为空时的替换和替换算法的选择），我选择直接在 cache_set_lru.cc 中进行 optimal 算法的 coding，把 lru 思想换成 optimal 算法。
  + 这样可以直接使用 isValid() 或者 isValidReplacement() 等函数
+ 关于 Optimal 算法的实现 getReplacementIndex
  + 在 sniper 模拟器的 cache_set_lru.cc 中直接进行修改，设置一个全局变量 counter_getrep 用于记录当前处理的 access 指令是第几条。
  + getReplacementIndex 中第一个 for 循环可以直接用，首先是去找一个空的块，有空的块就可以直接用，但是刚插进来的 block需要调用 FindtheNextAccess（寻找下一次使用的位置，返回一个步长存入m_lru_bits[]）
  + 之后的逻辑，如果这个 for 循环没有找到一个可以 return 的 index，那么就要用一个 for 循环，找到 set 中 next（m_lru_bits[i]）最大的 block 进行替换。用 index 记录这个最大的 next 的行号
  + 找到了最大的这个即将被替换的 block，仍然需要先找到当前 access（即将被插入的 block）的 next，所以需要执行FindtheNextAccess（类似于 LRU 中的 movetoMRU），找到当前 access 指令的 next 后，存入 m_lru_bits[i]。

## Inclusive to Non-inclusive

​		There must be an interface related to inclusive policy in the replacement algorithm. So focus on the parameters **qbs_reject** and **attempt**. The processing of inclusive policy may be included in **isInLowerLevelCache**. Therefore, the solution I chose is to set **qbs_reject** to false, that is, not to follow the conditional branch of if execution. After comparing the cache miss data before and after performing this operation, I found that this operation hardly affects the miss rate of L1 and L2 Cache. Therefore, the judgment about the interface of the inclusive policy must be in the replacement policy is not necessarily correct.

​	Inclusive to Non-inclusive 的修改没有成功

# 3. Result

实际上这个 lab 做的不算成功，我自己的 OPT 算法和原始的 LRU 相比，减少了 L2 TLB 的 miss rate，而 L1-I Cache 和 L1-D Cache 的表现反而更差了。

.<img src="./Img/miss rate of LRU.png" alt="miss rate of OPT" style="zoom: 33%;" /><img src="./Img/miss rate of OPT.png" alt="miss rate of OPT" style="zoom: 33%;" />



​       miss rate of LRU(left)                miss rate of OPT(right)     
