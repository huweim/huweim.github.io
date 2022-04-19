---
title: "搭建GPGPU-Sim实验环境"
date: 2021-09-27T22:54:06+08:00
lastmod: 2022-04-19T09:45:02+08:00
draft: false
tags: ["GPGPU-Sim", "环境搭建"]
categories: ["实验"]
---

# 服务器

## 0. 前言

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

### 0.1 测试

##### 在 gpgpu-sim_distribution 目录下只放置 lib 文件夹

也是可以 Run 的，说明程序运行时只会 call libcudart.so 这个文件 

修改 gpgpu-sim_distribution 文件夹的名字之后，ISPASS 中的 benchmark application 就还走不到 libcudart.so 文件了，这个应该和脚本写的文件路径有关

## 1. 虚拟机--Windows

都在脚本 `d/gpgpusim/script/host2vbox.sh` 中

### 1.1 从Windows到虚拟机

如果有这个必要的话，当然最好还是就在虚拟机里面直接编程。就是 VB 的虚拟机环境实在太慢了，后期思考一下解决方法

```bash
mv gpgpu-sim_distribution backup_gpgpu-sim_distribution
cp shared/gpgpu-sim_distribution ~/gpgpu-sim_distribution
cd ~/gpgpu-sim_distribution
source setup_environment release
make
make docs
```

### 1.2 从虚拟机到Windows

使用 Windows 和 VirtualBox 之间的共享文件夹，Win的共享文件夹放在 `d/gpgpusim/pchsare`. 

将vbox虚拟机里编译好的`/home/gpgpu-sim/gpgpu-sim_distribution/lib/gcc-4.6.4/cuda-4020/release/libcudart.so`及其软链接 (这里是 libcudart.so2/3/4) 拷贝到 `/home/shared/libsim`, 将 `/home/gpgpu-sim/cuda/toolkit/4.2/cuda/bin`文件夹拷贝到 `/home/shared/cuda`, i.e. `/home/shared/cuda/`里只有一个bin文件夹

```bash
cp gpgpu-sim_distribution/lib/gcc-4.6.4/cuda-4020/release/* ~/shared/libsim/
```

## 2. 从 Windows 到服务器

此时 Win `d/gpgpusim/pchsare` 中已经有了 `libsim`, `cuda` 两个文件夹，直接通过 mobaxterm 拖拽上传到服务器 `~/huweim/gpgpusim` 目录下即可

## 3. 从服务器到 Docker

### 3.1 挂载文件夹

已经在服务器上配置了可以运行GPGPU-Sim的Docker容器，且环境和编译GPGPU-Sim的虚拟机环境一致。把服务器上的外部文件夹挂载到Docker里面，

```bash
$ docker run --name gpgpusim -it -v /home/vsp/huweim/gpgpusim:/root/sim/ huweim/gpgpu-sim:v2 /bin/bash
#--name 给容器命名
#/home/huweim/gpgpusim:/home/sim/ 本地的绝对路径:容器的绝对路径
#huweim/gpgpu-sim:v2  镜像名:Tag 不指定Tag会尝试从Reposity pull latest, 因为是本地镜像没有远程reposity, 会报错，因此加上tag
```

### 3.2 在服务器上跑起来

```bash
$ cp ~/sim/libsim/* ~/gpgpu-sim_distribution/lib/gcc-4.4.7/cuda-4020/release/
$ cd ~/test
$ ~/ispass2009-benchmarks/bin/release/LPS | tee ~/output/LPS.log
$ ~/ispass2009-benchmarks/bin/release/RAY 4 4 | tee ~/output/RAY.log
```

## 4. 压榨性能

### 4.1 Shell 并发

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

### 4.2 减少输出的屏幕

减少在屏幕上的输出是否能有优化呢？做一个时间测试

> 非引用，自己猜想。有一种把输出重定向到垃圾池 > /dev/null 的操作，是否可以看做为了减少输出到终端的开销而采取的方式。那么是不是可以说明输出到终端会造成开销。

改用重定向的方式而非 `tee`, 观察能够加速多少。测试总共6个比较快的benchmark

+ tee: 180s
+ `>` : 不到1s

看来终端打印的确开销很大啊

## 5. PTXPlus 支持

直接修改 gpgpusim.config 会报错

```shell
GPGPU-Sim PTX: ERROR ** could not execute $GPGPUSIM_ROOT/build/$GPGPUSIM_CONFIG/cuobjdump_to_ptxplus/cuobjdump_to_ptxplus _cuobjdump_2.ptx _cuobjdump_2.sass _cuobjdump_2.elf _ptxplus_pRcNmc

```

### 5.1 解决方法

+ 在 `src/cuda-sim/ptx_loader.cc` 中，将 `$GPGPUSIM_ROOT/build/$GPGPUSIM_CONFIG/cuobjdump_to_ptxplus/cuobjdump_to_ptxplus` 修改为 `cuobjdump_to_ptxplus`。
+ `make` 并生成相应的 libcudatr.so 文件，copy to Docker。将文件 `gpgpusim-dev/build/gcc-4.4.7/cuda-4020/release/cuobjdump_to_ptxplus/cuobjdump_to_ptxplus` 复制到 Docker `/root/cuda/bin/` 文件夹下
+ 即每次 make，除了 copy lobcudart.so，还要 copy cuobjdump_to_ptxplus
  + 也有可能 copy 一次就行，猜测对 gpgpu-sim 的改动不会影响 cuobjdump_to_ptxplus
+ 给权限，`chmod 777 ~/cuda/bin/cuobjdump_to_ptxplus`
+ `cd ~/test`; `./ispass_run.sh MUM lrr`；成功运行



# Docker

## 0. 前言

编译在自己的虚拟机上完成，在服务器的 Docker 环境中运行。

## 1. Ubuntu20.04 in Docker

### 1.1 Pull

为了保证运行环境和编译环境一致，拉取 Ubuntu20.04 的 docker image。启动镜像的同时挂载上共享文件夹

```bash
$ docker pull ubuntu:20.04
$ docker run -it -v /home/vsp/huweim/gpgpusim:/root/share ubuntu:20.04 /bin/bash
```

### 1.2 工作文件

进入 /root 目录，需要一个 library 文件夹用来放 **动态链接库** (libcudart.so)，然后把自己的工作目录 test 拷贝到 /root 目录

```bash
$ mkdir library
$ mkdir test
$ cp ~/share/libsim/lib* ~/library
$ cp ~/share/test.tar ~/
$ tar -xvf ~/share/test.tar -C ~/test
```

### 1.3 环境

#### 1.3.1 安装 vi, make

```bash
$ apt-get update
$ apt-get install vim
$ apt-get install make
```

#### 1.3.2 LD_LIBRARY_PATH

LD_LIBRARY_PATH 这个系统变量需要指向动态链接库所在的文件夹

```bash
$ vi ~/.bashrc
#添加一行 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/library
$ source ~/.bashrc
```

尝试运行 shared_matrix，发现还缺少 `libGL.so.1` 

```shell
linux-vdso.so.1 (0x00007fff8ecfb000)
libcudart.so.4 => /root/library/libcudart.so.4 (0x00007f68b3596000)
libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f68b33b2000)
libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f68b3263000)
libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f68b3248000)
libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f68b3056000)
libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f68b303a000)
libGL.so.1 => not found	#缺少这个
libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f68b3015000)
/lib64/ld-linux-x86-64.so.2 (0x00007f68b395b000)

```

#### 1.3.3 libGL.so.1

```bash
$ apt install libgl1-mesa-glx
$ ldd hello
        linux-vdso.so.1 (0x00007ffdf7743000)
        libcudart.so.4 => /root/library/libcudart.so.4 (0x00007fa0358cd000)
        libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fa0356e8000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fa035599000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fa03557e000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fa03538c000)
        libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007fa035370000)
        libGL.so.1 => /lib/x86_64-linux-gnu/libGL.so.1 (0x00007fa0352e6000)
        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fa0352c3000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fa035c92000)
        libGLdispatch.so.0 => /lib/x86_64-linux-gnu/libGLdispatch.so.0 (0x00007fa03520b000)
        libGLX.so.0 => /lib/x86_64-linux-gnu/libGLX.so.0 (0x00007fa0351d7000)
        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007fa0351d1000)
        libX11.so.6 => /lib/x86_64-linux-gnu/libX11.so.6 (0x00007fa035092000)
        libxcb.so.1 => /lib/x86_64-linux-gnu/libxcb.so.1 (0x00007fa035068000)
        libXau.so.6 => /lib/x86_64-linux-gnu/libXau.so.6 (0x00007fa035062000)
        libXdmcp.so.6 => /lib/x86_64-linux-gnu/libXdmcp.so.6 (0x00007fa03505a000)
        libbsd.so.0 => /lib/x86_64-linux-gnu/libbsd.so.0 (0x00007fa035040000)
```

不过运行时还是有问题，看来还是需要把 CUDA 文件夹拷贝过来

#### 1.3.4 拷贝 cuda 文件夹

```shell
$ cd ~
$ mkdir cuda
$ cp -r ~/share/cuda/bin/ ~/cuda/
$ vi ~/.bashrc
#添加
# export CUDA_INSTALL_PATH=/root/cuda/
# export PATH=$PATH:/root/cuda/bin/
$ source ~/.bashrc
#给权限
$ chmod +x /root/cuda/bin/*
$ nvcc -V
#成功
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2012 NVIDIA Corporation
Built on Thu_Apr__5_00:24:31_PDT_2012
Cuda compilation tools, release 4.2, V0.2.1221
$ ./hello 
#这个时候运行 hello 程序已经可以了
```

#### 1.3.5 ISPASS'09 :exclamation: 其实无需编译只需运行

> 2021-11-02 14:03:43，好像又不需要这一步了，我不用再编译一次 ISPASS benchmark，直接运行即可。那就不要这一步，越简单越好
>
> 不过 SDK 也可以放在那里

这个 benchmark 拷贝过来，猜测还需要配置 `NVIDIA_COMPUTE_SDK_LOCATION` 的路径，还是拷贝过来即可。SDK 文件比较大

```bash
$ cp ~/share/sdk.tar ~/
$ cp ~/share/ispass2009-benchmarks.tar ~/
$ mkdir ~/cuda/sdk
$ mkdir ~/ispass2009-benchmarks
#tar 没有解压同时新建文件夹的选项
$ tar -xvf ~/sdk.tar -C ~/cuda/sdk
$ tar -xvf ~/ispass2009-benchmarks.tar -C ~/ispass2009-benchmarks
#设置 NVIDIA_COMPUTE_SDK_LOCATION 的路径
$ vi ~/.bashrc
#添加
# export NVIDIA_COMPUTE_SDK_LOCATION=/root/cuda/sdk/4.2
$ source ~/.bashrc
```

### 1.4 备份可以成功运行的容器

#### 1.4.1 导出容器

根据制作好的容器 ID 5c4152ae7b05 到处。大小也就是1G，非常轻便

```bash
$ docker export 5c4152ae7b05 > ubuntu_gpgpusim.tar
```

这里备份压缩包即可，下次更换环境时只需将压缩包再次导入为镜像
