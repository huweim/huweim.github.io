---
title: "MPI Intro and Practice"
date: 2021-03-09T22:36:19+08:00
draft: false
---

# MPI Intro and Practice

### Intro

##### Definition

wiki:

+ **Message Passing Interface** (**MPI**) is a standardized and portable message-passing standard
+ designed by a group of researchers from academia and industry to function on a wide variety of parallel computing architectures.

##### Feature

+ an interface, not a programming language
+ Main model of HPC
+ a cross-language communication protocol

##### Functions

+ Communication
  + Point-to-point communication
    + Send
    + Recv
  + Collective communication
    + Broadcast, scatter/ gather, all to all, reduce, scan, barrier
+ Almost all parallel programs can be described using the message passing model.

##### Concept

+ Communicator: 
  + Def: Communicator objects connect groups of processes in the MPI session.
  + Each communicator gives each contained process an independent identifier(id, called `rank`) and arranges its contained processes in an ordered topology.

### Set up in Ubuntu 20.04

在这个阶段，简单地通过网络的教程进行 library or software 的安装是不现实的，还是得从根本上学会去解决问题，去看源文档(doc)的说明。

##### I. Download

+ Download `mpich-3.4.1.tar.gz` at `https://www.mpich.org/downloads/`
+ 版本会更新，地址应该不会变

##### II. 解压后进入目录

```bash
tar -xzvf mpich-3.4.1.tar.gz
cd mpich-3.4.1
```

##### III. 配置

```bash
./configure --prefix=/home/Desktop/HPC/mpich-3.4.1/mpich-install 2>&1 | tee c.txt
```

+ 这个地方根据自己的安装路径，我就配置在当前的`/home/Desktop/HPC/mpich-3.4.1`文件夹下

+ 然后出现两个坑
  + error: no ch4 netmod selected
  + ![error_1](../Image/error_1.png)
  + 根据提示加上 `--with-device=ch4:ofi` 即可
  + 加上后再次报错`No Fortran compiler found. If you don't need to build any Fortran programs, you can disable Fortran support using --disable-fortran. If you do want to build Fortran programs, you need to install a Fortran compiler such as gfortran or ifort before you can proceed.`
  + 这是因为没有安装Fortran compiler
  + 根据提示加上 `--disable-fortran` 即可

```bash
./configure --disable-fortran  --with-device=ch4:ofi  --prefix=/home/Desktop/HPC/mpich-3.4.1/mpich-install 2>&1 | tee c.txt
```

+ 成功配置

  ![Config completed](../Image/Config.png)

接下来 ->

```bash
make    		#等待一段漫长的时间
make install    #权限不够加 sudo
```

##### IV. 添加环境变量

```bash
sudo gedit ~/.bashrc
```

+ 打开`.bashrc` 文件后在末尾添加

```bash
export MPI_ROOT=/home/Desktop/HPC/mpich-3.4.1/mpich-install #这一步对应你自己的安装地址
export PATH=$MPI_ROOT/bin:$PATH
export MANPATH=$MPI_ROOT/man:$MANPATH
```

+ 然后激活

```bash
source ~/.bashrc
```

+ which mpicc 查看位置信息
+ mpichversion 查看版本信息，出现版本号说明安装成功

![Version Info](../Image/Version.png).

### 运行程序

##### I. 创建文件hello.c

```c++
#include "mpi.h"
#include <stdio.h>

int main( int argc, char *argv[] )
{
    int rank, size;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    printf( "I am %d of %d\n", rank, size );
    MPI_Finalize();
    return 0;
}
```

##### II. 编译

```bash
mpicc hellow.c -o hellow
```

##### III. 运行

```bash
mpirun -n 2 ./hellow
```

![Compile and Execute](../Image/Compile.png).