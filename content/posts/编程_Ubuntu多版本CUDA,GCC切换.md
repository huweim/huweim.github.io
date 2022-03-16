---
title: "Ubuntu多版本CUDA,GCC切换"
date: 2021-11-14T22:07:35+08:00
lastmod: 
draft: false
author: "Cory"
tags: ["cuda", "gcc"]
categories: ["工具"]
---

# Ubuntu多版本CUDA,GCC切换

# 切换CUDA9.0和CUDA10.0

保证多个CUDA版本共存的前提是NVIDIA的驱动都能够支持你所安装的CUDA版本，所以驱动的版本尽可能高，越新的驱动支持的CUDA版本越多，博主的430能够支持9.0和10.0。

在先前安装的CUDA的过程中，大家一般都会选择生成cuda-x.0文件夹的软链接/usr/local/cuda，这个文件夹是实际安装的cuda-x.0文件夹的链接，不包含实际文件，是方便系统设置环境变量直接调用cuda的，安装多个版本的CUDA，然后利用软链接就可以实现版本切换。

+ 理解这个软链接，用到了很多次

## Step

### 1 更换软链接

不过之前环境变量用的 cuda11.1 的地址而非软链接，现在替换成软链接

```bash
sudo rm -rf /usr/local/cuda  #删除之前生成的软链接
sudo ln -s /home/huweim/cuda/toolkit/4.2/cuda /usr/local/cuda #生成新的软链接
sudo ln -s /usr/local/cuda-11.1 /usr/local/cuda	#使用11.1版本
```

### 2 Check 环境变量的地址

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda/
export PATH=$PATH:/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin:$CUDA_INSTALL_PATH/bin:$MPI_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_INSTALL_PATH/lib64	#这个不用改
export NVIDIA_COMPUTE_SDK_LOCATION=~/cuda/sdk/4.2
```

### 3 查看版本信息

上述步骤全部没问题就可以弹出版本信息了，`source ~/.bashrc` 或者重启终端

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2012 NVIDIA Corporation
Built on Thu_Apr__5_00:24:31_PDT_2012
Cuda compilation tools, release 4.2, V0.2.1221
```

### 4. Bug

#### 4.1 sh: 1: nvopencc: Permission denied

解决方法

```shell
sudo chmod -R 777 /usr/local/cuda
```



# 切换gcc, g++ 9 -> 5

Reference

+ https://blog.csdn.net/EternallyAccompany/article/details/108865331

linux上可以gcc多版本共存，可以通过**修改软链接**的方式选择自己要用的gcc版本，该方法简单方便，可以随时依据自己的需求将gcc降级或升级，解决不同的软件要求不同的环境的问题。

apt-get安装gcc、g++，默认下载最新版本的，此时ubuntu里的gcc和g++版本均为9。

```bash
1. sudo apt-get install gcc
2. sudo apt-get install g++
3. gcc -v   //查看的版本为9.0.0
4. g++ -v   //查看的版本为9.0.0
```

## 1. 软件源

打开sources.list

```bash
sudo gedit /etc/apt/sources.list
```

Add

```bash
deb http://dk.archive.ubuntu.com/ubuntu/ xenial main
deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe
```

Update

```bash
sudo apt-get update
```

## 2. 安装gcc5, g++5

apt-get 安装gcc、g++ 5版本。

```bash
sudo apt-get install g++-5 gcc-5sudo apt-get -f install   #if need 
```

2021-07-09 10:40:42 gcc5版本又出现了一些问题，找[教程](https://blog.csdn.net/qq_42566274/article/details/106399531)安装了 gcc4.8 版本，这个文章里面说最低 4.7 版本，那就先用 4.8 版本试试 gpgpu-sim 能不能 work

```bash
sudo apt-get install gcc-4.8sudo apt-get install g++-4.8
```



## 3. 查看

```bash
ls /usr/bin/gcc*ls /usr/bin/g++*
```

## 4. 设置优先级

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 90sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 80sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 90sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 80
```

增加

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 95sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 95
```

成功配置上了 gcc 4.8

> 由于gpgpusim必须使用gcc4.7及以前的版本，而修改自己的电脑系统可能带来不方便，因此使用docker来运行程序，docker在运行程序时，性能损失大概在10%以内，但也比vbox快多了。

好吧，必须使用 gcc4.7 以前的版本

## 5. 选择gcc/g++版本

```bash
sudo update-alternatives --config gccsudo update-alternatives --config g++
```

输入编号选择gcc/g++版本

:warning: *gcc/g++版本须保持一致*

## 6. Check 版本

```bash
gcc -vg++ -v
```

