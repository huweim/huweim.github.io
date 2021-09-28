---
title: "搭建GPGPU-Sim实验环境"
date: 2021-09-27T22:54:06+08:00
draft: false
tags: ["GPGPU-Sim", "环境搭建"]
categories: ["GPGPU"]
---

# 0. 前言

第一个思路是 

+ 服务器OS->Docker Container->Ubuntu中运行GPGPU-Sim。
+ Docker Container update Docker Image->Docker Image->XXX.tar->复制到你的电脑Windows->复制到你的虚拟机Ubuntu->XXX.tar->Docker Image->Docker Container->Ubuntu中运行GPGPU-Sim->修改GPGPU-Sim
+ 然后同样使用上述过程移植到服务器，运行

这样是有问题的。首先这个过程没有意义，如果这样在你自己的虚拟机里面运行Docker, 那么仍然是命令行界面，和在服务器上运行的区别在哪？

这样实现了Docker的其中一个作用

+ 我在服务器上能跑，在我自己的虚拟机上也能跑。实现了在不同的环境下运行，而且无需安装多余的依赖。因为本质上我用的是 Docker 中的 Ubuntu 14
+ 但我没有实现自己的目的

我的目的是什么？

+ 在自己的Ubuntu上使用VScode修改模拟器，简单地编译测试性能。修改后需要跑大量benchmark, 这个时候我不能用自己的电脑跑了，我需要移植了。

+ 把跑benchmark需要用到的东西放在服务器上，用服务器的计算资源运行。需要用到的东西是什么？

  + benchmark: 一般是一些 .cu/.cl 代码编译后生成的可执行文件

  + > 编译成功gpgpusim以后，实际上主要是生成了一个`libcudart.so`。

    那么就需要这个 libcudart.so

+ 所以理论上来说如果我使用一台固定的服务器，好像不需要一直更新Docker?无需安装 gcc4.5.1, cuda4.2。每次把这几个文件拷贝过去即可。

# 0.1 测试

##### 在 gpgpu-sim_distribution 目录下只放置 lib 文件夹

也是可以 Run 的，说明程序运行时只会 call libcudart.so 这个文件 

修改 gpgpu-sim_distribution 文件夹的名字之后，ISPASS 中的 benchmark application 就还走不到 libcudart.so 文件了，这个应该和脚本写的文件路径有关

# 1. 虚拟机--Windows

都在脚本 `d/gpgpusim/script/host2vbox.sh` 中

## 1.1 从Windows到虚拟机

如果有这个必要的话，当然最好还是就在虚拟机里面直接编程。就是 VB 的虚拟机环境实在太慢了，后期思考一下解决方法

```bash
mv gpgpu-sim_distribution backup_gpgpu-sim_distribution
cp shared/gpgpu-sim_distribution ~/gpgpu-sim_distribution
cd ~/gpgpu-sim_distribution
source setup_environment release
make
make docs
```

## 1.1 从虚拟机到Windows

使用 Windows 和 VirtualBox 之间的共享文件夹，Win的共享文件夹放在 `d/gpgpusim/pchsare`. 

将vbox虚拟机里编译好的`/home/gpgpu-sim/gpgpu-sim_distribution/lib/gcc-4.6.4/cuda-4020/release/libcudart.so`及其软链接 (这里是 libcudart.so2/3/4) 拷贝到 `/home/shared/libsim`, 将 `/home/gpgpu-sim/cuda/toolkit/4.2/cuda/bin`文件夹拷贝到 `/home/shared/cuda`, i.e. `/home/shared/cuda/`里只有一个bin文件夹

```bash
cp gpgpu-sim_distribution/lib/gcc-4.6.4/cuda-4020/release/* ~/shared/libsim/
```

# 2. 从 Windows 到服务器

此时 Win `d/gpgpusim/pchsare` 中已经有了 `libsim`, `cuda` 两个文件夹，直接通过 mobaxterm 拖拽上传到服务器 `~/huweim/gpgpusim` 目录下即可

# 3. 从服务器到 Docker

## 3.1 挂载文件夹

已经在服务器上配置了可以运行GPGPU-Sim的Docker容器，且环境和编译GPGPU-Sim的虚拟机环境一致。把服务器上的外部文件夹挂载到Docker里面，

```bash
$ docker run --name gpgpusim -it -v /home/vsp/huweim/gpgpusim:/root/sim/ huweim/gpgpu-sim:v2 /bin/bash
#--name 给容器命名
#/home/huweim/gpgpusim:/home/sim/ 本地的绝对路径:容器的绝对路径
#huweim/gpgpu-sim:v2  镜像名:Tag 不指定Tag会尝试从Reposity pull latest, 因为是本地镜像没有远程reposity, 会报错，因此加上tag
```

## 3.2 在服务器上跑起来

```bash
$ cp ~/sim/libsim/* ~/gpgpu-sim_distribution/lib/gcc-4.4.7/cuda-4020/release/
$ cd ~/test
$ ~/ispass2009-benchmarks/bin/release/LPS | tee ~/output/LPS.log
$ ~/ispass2009-benchmarks/bin/release/RAY 4 4 | tee ~/output/RAY.log
```

# 4. 压榨性能

## 4.1 Shell 并发

wait 相当于 syncthread()

```bash
# !/bin/bash
for i in `seq 1 10` 
do
    echo $i &
done

wait
echo "----end----"
```

## 4.2 减少输出的屏幕

减少在屏幕上的输出是否能有优化呢？做一个时间测试

> 非引用，自己猜想。有一种把输出重定向到垃圾池 > /dev/null 的操作，是否可以看做为了减少输出到终端的开销而采取的方式。那么是不是可以说明输出到终端会造成开销。

改用重定向的方式而非 `tee`, 观察能够加速多少。测试总共6个比较快的benchmark

+ tee: 180s
+ `>` : 不到1s

看来终端打印的确开销很大啊

