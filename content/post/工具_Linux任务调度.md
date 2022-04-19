---
title: "Linux任务调度"
date: 2021-11-29T23:00:09+08:00
lastmod: 2022-03-16 08:44:28
draft: false
author: "Cory"
tags: ["Linux"]
categories: ["工具"]
---

# 0. 前言

记录一下 `Linux` 中 `fg`、`bg`、`jobs`、`&`、`ctrl + z` 等相关指令对任务进程的操作。

也正好借此机会学习一下进程（process）的概念

# 1. Process

## 1.0 进程类型

+ 前台进程（交互式进程）
  + 这些进程由终端会话初始化和控制。换句话说，需要有一个连接到系统中的用户来启动这样的进程；它们不是作为系统功能/服务的一部分自动启动。
+ 后台进程

## 1.1 并发执行

To run commands concurrently you can use the `&` command separator

```shell
~$ command1 & command2 & command3
```

This will start `command1`, then runs it in the background. The same with `command2`. Then it starts `command3` normally.

> 这样的话 command3 是在前台运行

The output of all commands will be garbled together, but if that is not a problem for you, that would be the solution.

If you want to have a separate look at the output later, you can pipe the output of each command into `tee`, which lets you specify a file to mirror the output to.

```bsh
~$ command1 | tee 1.log & command2 | tee 2.log & command3 | tee 3.log
```

> 这样可以分别在对应的 log 中查看不同 command output，我也是这么做的
>
> 加&的作用其实就是将命令放到后台执行

### 1.1.1 终止并发执行的后台进程

- 方法一： 通过 `jobs` 命令查看任务号（假设为 `num`），然后执行：`kill %num`
- 方法二： 通过 `ps` 命令查看任务的进程号（`PID`，假设为 `pid`），然后执行：`kill pid`
## 1.2 进程及性能分析

### 1.2.1 进程状态

+ R： **RUNNING & RUNNABLE**，正在运行或在运行队列中等待
+ S：**INTERRRUPTABLE_SLEEP**，休眠中, 受阻, 在等待某个条件的形成或接受到 signal
+ D：**UNINTERRUPTABLE_SLEEP**，不可中断（usually IO）
+ T：**STOPPED**，ctrl + z 进入这个状态
+ Z：**ZOMBIE**，进程已终止, 但进程描述符存在, 直到父进程调用wait4()系统调用后释放

### 1.2.2 查看进程状态 ps

```shell
$ ps -aux #显示所有包含其他使用者的行程 
```

**%CPU**：该 process 使用掉的 CPU 资源百分比

**%MEM**：该 process 所占用的物理内存百分比

**TTY** ：该 process 是在那个终端机上面运作，若与终端机无关，则显示 ?，另外， tty1-tty6 是本机上面的登入者程序，若为 pts/0 等等的，则表示为由网络连接进主机的程序。

**STAT**：该程序目前的状态，主要的状态参考 **1.2进程状态**

**START**：该 process 被触发启动的时间

**TIME** ：该 process 实际使用 CPU 运作的时间

### 1.2.3 性能分析 top

##### 信息查看实例

```shell
top - 19:39:49 up 77 days,  8:54,  0 users,  load average: 5.00, 5.00, 5.00
# 系统当前时间；开机到现在77days 8h 54 mins；0 users 在线；系统1分钟、5分钟、15分钟的CPU负载信息
Tasks:  13 total,   1 running,  12 sleeping,   0 stopped,   0 zombie
%Cpu(s): 20.9 us,  0.0 sy,  0.0 ni, 79.1 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
# %us：用户态进程占用CPU时间百分比，不包含renice值为负的任务占用的CPU的时间。
# %sy：内核占用CPU时间百分比
# %ni：改变过优先级的进程占用CPU的百分比
# %id：空闲CPU时间百分比
# %wa：等待I/O的CPU时间百分比
# %hi：CPU硬中断时间百分比
# %si：CPU软中断时间百分比
# 这里显示数据是所有cpu的平均值，如果想看每一个cpu的处理情况，按1即可；折叠，再次按1
MiB Mem :  48192.1 total,  25777.1 free,   2605.3 used,  19809.7 buff/cache
MiB Swap:  19530.0 total,  19491.7 free,     38.2 used.  45002.0 avail Mem

```

##### 查看单个进程使用CPU情况

```shell
$ top -p pid
$ ps -aux #这个好像更加方便一点
```

## 1.3 为什么top查看只有1 running，其他sleeping?

##### Stackoverflow回答

There is no correlation between CPU usage as reported by `top` and process state. The [man page](http://linux.die.net/man/1/top) says (*emphasis* mine):

> **%CPU** -- CPU usage
>
> The task's share of the elapsed CPU time *since the last screen update*, expressed as a percentage of total CPU time.

So, your process indeed used a huge amount of processor time since the last screen update. It is sleeping, yes, but that's because the currently running process is `top` itself (which makes sense, since it's currently updating the screen).

##### 理解

这个1 running 就是 top process 本身，其他的 process 其实也是在运行的。每次 top 面板更新时活跃 process 正好是 top，所以 only 1 running。但是查看详细信息可以看到其实是在运行的。

# 2. 进程调度

## 2.1. 基本用法

### 2.1.1 `&` 和 `jobs` 指令

`&` 用在一个命令的最后，可以把这个命令转换为**后台运行**的任务进程。

`jobs` 查看当前终端有多少在后台运行的进程。

- `jobs` 命令执行的结果，`＋` 表示是一个当前的作业，`-` 减号表示是一个当前作业之后的一个作业。
- `jobs -l` 选项可显示所有任务的进程号 `pid`
- `jobs` 的状态可以是 `running`，`stopped`，`terminated`。但是如果任务进程被终止了（`kill`），当前的终端环境中也就删除了任务的进程标识；也就是说 **jobs 命令显示的是当前 shell 环境中后台正在运行或者被挂起的任务进程信息**

### 2.1.3 `fg` 和 `bg` 指令

`fg` 将后台任务进程调至**前台**继续运行，如果后台中有多个任务进程，可以用 `fg %num` 将选中的任务进程调至前台。

`bg` 将挂起的任务进程重新启动运行，如果有多个暂停的任务进程，可以用 `bg %num` 将选中的任务进程启动运行。

> `%num` 是通过 `jobs` 命令查到的后台正在执行的任务的序号（不是 `pid`）

:exclamation: 使用 ispass_run.sh 的时候最好使用 `fg`，如果使用 `bg` 则无法再次挂起，~~猜测是因为~~，原因是 `ctrl + z` 是用于挂起前台进程，使用 `stop` 将其挂起即可

> 使用 stop 2，好像也无法将其挂起？

## 2.2. 进程的挂起

### 2.2.1 后台进程的挂起

- 在 `solaris` 中通过 `stop` 命令执行，通过 `jobs` 命令查看任务号（假设为 `num`），然后执行：`stop %num`
- 在 `redhat` 中，不存在 `stop` 命令，可通过执行命令 `kill -stop PID`，将进程挂起

### 2.2.2 前台进程的挂起 ctrl+z

`ctrl + z`：可以将一个正在s前台执行的任务放到后台运行，并且挂起

## 2.3. 挂起进程重新运行 bg, fg 

- 通过 `bg %num` 即可将挂起的任务进程的状态由 `stopped` 改为 `running`，仍在后台运行
- 通过 `fg %num` 即可将挂起的任务进程转为前台执行

## 2.4. 进程的终止

### 2.4.1 后台进程的终止 kill, killall

- 方法一： 通过 `jobs` 命令查看任务号（假设为 `num`），然后执行：`kill %num`
- 方法二： 通过 `ps` 命令查看任务的进程号（`PID`，假设为 `pid`），然后执行：`kill pid`
- `killall CMD`：通过进程名 kill 感觉比根据 pid 方便很多

### 2.4.2 前台进程的终止 ctrl+c

执行 `ctrl+c` 即可终止前台执行任务进程

> 假设要后台运行 `xmms`，可通过命令：`xmms &`。但万一你运行程序时忘记使用 `&` 了，又不想重新执行，你可以先使用 `ctrl+z` 挂起任务进程，然后敲入`bg` 命令，这样任务进程就在后台继续运行了。

# 3. 查看CPU信息

+ 查看物理CPU个数

```shell
cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l
```

+ 查看每个物理CPU中core的个数(即核数)

```shell
cat /proc/cpuinfo| grep "cpu cores"| uniq
```

+ 查看逻辑CPU的个数

```shell
cat /proc/cpuinfo| grep "processor"| wc -l
```

# 4. Why `jobs` 看不到后台 process

## 4.1 关闭终端造成孤儿进程

> jobs是查看当前后台作业（进程），是获取本次bash下的后台作业。
> 当本次终端退出后，后台作业变成**孤儿进程**，孤儿进程有系统父进程接管。
> 当再次连接终端时，**原作业**与**当前终端**，不存在关系父子关系，故看不到进程。
> 但是原作业，会在系统中一致运行，直到完成或被停止。
> 这就是为什么终端退出后，jobs看不到的原因了

## 4.2 nohup

### 4.2.1 &

守护进程貌似跟nohup + &方式启动的进程差不多。都可以实现与终端的无关联。

& 可以让进程在后台运行，ctrl + C 无法终端，对 SIGINT 信号免疫，但是直接关闭 shell 后进程会消失。& 后台没有那么硬 :)，对 SIGHUP 信号不免疫

### 4.2.2 nohup  作用

nohup 的作用是忽略 SIGHUP 信号，也就是不挂断地运行

 &和nohup没有半毛钱的关系， 要让进程真正不受shell中Ctrl C和shell关闭的影响， 那该怎么办呢？ 那就用nohup ./start.sh &吧， 两全其美。

# Reference

https://ehlxr.me/2017/01/18/Linux-%E4%B8%AD-fg%E3%80%81bg%E3%80%81jobs%E3%80%81-%E6%8C%87%E4%BB%A4/

https://www.cnblogs.com/machangwei-8/p/10391440.html