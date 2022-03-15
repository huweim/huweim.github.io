---
title: "编译运行ISPASS2009、Rodinia、Parboil"
date: 2021-12-08T09:45:02+08:00
lastmod: 
draft: false
author: "Cory"
tags: ["gpgpu-sim", "benchmark"]
categories: ["gpgpu"]
---

# 1. ISPASS'09

Ubuntu20.04 下使用 GPGPU-Sim 运行 ISPASS2009 benchmark

## 1.0. 前言

之前介绍了安装，现在就尝试跑一下 ISPASS'09 的那篇经典 paper，Analyzing CUDA workloads using a detailed GPU simulator 上的几个 benchamrk. 这篇文章现在已经870次引用了，很多工作都使用了其中的 benchmark

## 1.1. 安装 CUDA Toolkit and CUDA SDK

CUDA 5之后，SDK 和 Toolkit 都在一个包里面，ISPASS'09 benchmark 会用到 build CUDA SDK 时创建的库，所以需要 CUDA SDK。

之前安装的是 cuda 11.1 版本，现在切换到 cuda 4.2 版本，在 GPGPU-Sim [官方](http://www.gpgpu-sim.org/)提供的虚拟机中可以找到 cuda 目录，里面有 cuda 4.2 版本的 Toolkit and SDK，复制过来，给读写权限，设置好环境变量就可以直接使用了

### 1.1.1 设置软链接

```bash
sudo rm -rf /usr/local/cuda  #删除之前生成的软链接
sudo ln -s /home/huweim/cuda/toolkit/4.2/cuda /usr/local/cuda #生成新的软链接
```

### 1.1.2 Check 环境变量的地址

`vim ~/.bashrc`，加入以下环境变量

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda/toolkit/4.2/cuda
export PATH=$PATH:/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin:$CUDA_INSTALL_PATH/bin:$MPI_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_INSTALL_PATH/lib64	#这个不用改
export NVIDIA_COMPUTE_SDK_LOCATION=~/cuda/sdk/4.2
```

### 1.1.3 查看版本信息

上述步骤全部没问题就可以弹出版本信息了，`source ~/.bashrc` 或者重启终端

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2012 NVIDIA Corporation
Built on Thu_Apr__5_00:24:31_PDT_2012
Cuda compilation tools, release 4.2, V0.2.1221
```

## 1.2. gcc, g++ 版本

CUDA 更换为 4.2 版本后，重新 build 模拟器得知 `unsupported GNU version! gcc 4.7 and up are not supported!`。之前在别人的博客也看到需要 4.7 以下版本 gcc，现在安装 gcc g++ 4.4

```bash
sudo add-apt-repository 'deb http://archive.ubuntu.com/ubuntu/ trusty main'
sudo add-apt-repository 'deb http://archive.ubuntu.com/ubuntu/ trusty universe'
sudo apt update
sudo apt install gcc-4.4

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.4 50	#设置优先级
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.4 50	#设置优先级
sudo update-alternatives --config gcc	#输入编号选择gcc/g++版本
sudo update-alternatives --config g++	#输入编号选择gcc/g++版本
gcc -v	#Check 版本
g++ -v	#Check 版本
```

## 1.3. Compile Nvidia SDK 4.2

3、4 参考 http://www.danielwong.org/classes/ee260-w17/assignment1

+ `sudo apt-get install libboost-dev libboost-system-dev libboost-filesystem-dev libboost-all-dev mpich2 binutils-gold libcuda1-304`
+ `sudo gedit ~/cuda/sdk/4.2/C/common/common.mk`, line like “`LIB += … ${OPENGLLIB} …. $(RENDERCHECKGLLIB) …`” should have `$(RENDERCHECKGLLIB)` moved before `${OPENGLLIB}`. There should be 3 lines like this, it may be line 271, 275 and 282.
  + 将 line26 `CUDA_INSTALL_PATH ?= /home/gpgpu-sim/cuda/toolkit/4.2/cuda` 修改为 `CUDA_INSTALL_PATH ?= /home/huweim/cuda/toolkit/4.2/cuda`
    + 这一步应该不用了，用软链接代替路径即可
  + 因为这个文件直接拷贝过来的，所以要设置为自己的安装路径
+ `sudo gedit ~/cuda/sdk/4.2/CUDALibraries/common/common_cudalib.mk` and do the same thing.

<img src="D:\ShanghaiTech\2021-Fall\GPGPU\Image\Fig_1.png" align="left" style="zoom:50%;" />

+ `cd ~/cuda/sdk/4.2`
+ Edit Makefile through `sudo gedit ./Makefile`. Comment all lines with `CUDALibraries` and `OpenCL` as we only want the application binaries. You comment by placing `#` in the front of the line.

<img src="D:\ShanghaiTech\2021-Fall\GPGPU\Image\Fig_2.png" align="left" style="zoom:50%;" />

+ `make`

### 1.3.1 Bug

遇到 bug

```shell
/usr/bin/ld: cannot find -lcudart
/usr/bin/ld: cannot find -lcufft

/usr/bin/ld: cannot find -lXi
/usr/bin/ld: cannot find -lXmu
/usr/bin/ld: cannot find -lglut
```

### 1.3.2 解决

建立 cudart 的软链接，安装需要的 lib

```shell
sudo ln -s /usr/local/cuda/lib64/libcudart.so.4.2.9 /usr/lib/libcudart.so
sudo ln -s /usr/local/cuda/lib64/libcufft.so.4.2.9 /usr/lib/libcufft.so
sudo apt-get install libxi-dev
sudo apt-get install libxmu-dev
sudo apt-get install freeglut3-dev
```

## 1.4. ISPASS2009 Benchmarks

按照以下步骤编译

+ `git clone https://github.com/gpgpu-sim/ispass2009-benchmarks.git`, we also upload it to the pan.shanghaitech.
+ `sudo gedit ~/.bashrc`, add `export NVIDIA_COMPUTE_SDK_LOCATION=~/cuda/sdk/4.2`
+ `cd ispass2009-benchmarks`
+ `sudo gedit ./Makefile.ispass-2009`
  + comment out line 16 and 28. The AES and WP benchmark does not compiler readily.
    + :exclamation: Note not line 16 to 28, only this two lines 
  + Change `BOOST_LIB` to the following path: `/usr/lib/x86_64-linux-gnu/`
  + Change `BOOST_ROOT` to the following path: `/usr/include/boost/`
  + Change `OPENMPI_BINDIR` to the following path: `/usr/bin/`
+ `sudo gedit ./DG/Makefile`
  + Line 54, change `-I/opt/mpich/include` to `-I/usr/include/mpich`
  + Line 56, append `-I/usr/include/mpich` to end of the line
  + Line 59-61, add `.mpich2` to the end of each line. For example, `NEWCC = $(OPENMPI_BINDIR)mpicc.mpich2`
+ `make -f Makefile.ispass-2009`. Note: If you do a `make -f Makefile.ispass-2009 clean`, you may have to recompile the SDK.

# 2. Rodinia

Ubuntu20.04下使用GPGPU-Sim运行 Rodinia

## 2.0. 前言

Rodinia 的类型更多，尝试将其编译

## 2.1. 修改路径

修改 `gpu-rodinia-master/common/make.config` 中 CUDA_INSTALL_PATH，CUDA_LIB_DIR，SDK_DIR 为 your own path

对我来说只用修改 SDK_DIR

```shell
SDK_DIR = /home/huweim/cuda/sdk/4.2/C
```

## 2.2. Data Set

惊喜地发现 data set 是空的，download from http://lava.cs.virginia.edu/Rodinia/download_links.htm

好家伙一个 G Data Set

## 2.3. 进入 cuda 文件夹逐一编译即可

### 2.3.0 找不到 -lcuda 大多数文件

原因就是没找到动态链接库，guess 因为我们没有安装驱动 :heavy_check_mark:

~~make.config 中 `CUDA_LIB_DIR` 改为 libcudart.so 所在的路径~~，需要的是 libcuda.so 而不是 libcudart.so

:x: ~~去[官网](https://developer.nvidia.com/cuda-toolkit-42-archive)下载驱动~~，电脑没有 NVIDIA GPU 无法安装 Driver

:heavy_check_mark: 去掉 -lcuda 编译选项，出现 `undefined reference to symbol '__gxx_personality_v0@@CXXABI_1.3'` 错误时，把 `CC = gcc` 更改为 `CC = g++`，或者添加编译选项 `-lstdc++`

暂不清楚去掉 -lcuda 编译选项后的影响

### 2.3.1 kmeans, leukocyte

**BUG**：提示缺少 .o 

**Solution**：Makefile 这个 [ch] 会读到 .h，从而编译不出 kmeans.o，

```makefile
%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c
#改成后正确编译
%.o: %.[c]
	$(CC) $(CC_FLAGS) $< -c
```

kmeans： `libgomp.so.1 => not found`

```shell
$ apt-get install libgomp1
```

### 2.3.2 huffman, srad

架构版本改成 `-arch=sm_20`

### 2.3.3 hybridsort

bucketsort.cu:10: fatal error: GL/glew.h: No such file or directory

看起来是没有 OpenGL

```shell
$ sudo apt-get install libglew-dev
```

## 2.4. Run

### 2.4.1 Why these benchmark so slow?

streamcluster, leu, fdtd2d 等等

Rodinia 大多数 benchmark 都跑得很慢，有的在服务器上两三天都没跑完，不知道是数据量太大还是编译过程除了问题？尽管 GPGPU-Sim 可以自行在达到一定的 cycle 数时停止，还是想知道为什么 Rodinia 如此之慢。

# 3. Parboil

## 3.0. 前言

Ubuntu20.04 下使用 GPGPU-Sim 编译运行 Parboil

## 3.1. 下载和修改文件
### 3.1.1 下载

最好在[官网](http://impact.crhc.illinois.edu/parboil/parboil_download_page.aspx)下载后部署，Github 上的文件多了几个 benchmark，但是没有他们的数据集。


在官网下载好需要的目录，按照 README 中的要求部署，`Parboil` 中需要有 4 个目录，`benchmarks`，`common`，`driver`，`datasets`

<img src="./Img/Parboil_compile_1.png" align=left alt="image-20211205103147660" style="zoom:50%;" />

### 3.1.2 Makefile.conf

进入 common 目录，里面有对应设备的一些 Makefile examples。主要使用的是cuda，check 文件 Makefile.conf.example-nvidia 中的 CUDA 路径，**复制**并改名为 Makefile.conf。

### 3.1.3 CUDA_LIB_PATH

修改为 gpgpu-sim 的中的 libcudart.so 路径即可

`CUDA_LIB_PATH=/home/huweim/gpgpu-sim_distribution/lib/gcc-4.4.7/cuda-4020/release/`

## 3.2. Data Set

download from http://impact.crhc.illinois.edu/parboil/parboil_download_page.aspx，在 1.1 的步骤中应该已经下载了，但是在 Github 中不带 data set，如果 data set 不全，使用他的脚本编译可能会报错，所以建议在官网下载。

包含 11 个 benchmark data set

## 3.3. Compile

### 3.3.1 Bug

这里就是之前说的 BUG，所以东西都去官网下载就不会有这些 BUG 了。

#### 3.3.1.1 Dataset 不全

<img src="./Img/Parboil_bug_1.png" align=left alt="image-20211205103147660" style="zoom:50%;" />

16 benchmark src，但是 dataset 只有11个

:heavy_check_mark: 解决方法，直接去官方网站下载，不在 github 上下载，11 benchmark 对应 11 dataset 

#### 3.3.1.2 gcc

`error: identifier "__builtin_ia32_vec_set_v8hi" is undefined`，网上也没有搜索到。
尝试切换一下 gcc 版本，发现只能用 gcc 4.7 及以前的版本。

:heavy_check_mark: 解决方法，直接去官方网站下载，不在 github 上下载，11 benchmark 对应 11 dataset。这个时候 compile 也没有错误了。

### 3.3.2 Compile 命令

```shell
$ ./parboil compile benchmark_name platform_name
$ ./parboil compile bfs cuda
#bfs 可执行文件在目录 parboil_Dir/benchmarks/bfs/build/cuda_default/bfs
```

接下来逐一编译11个 benchmark 即可

## 3.4. Run

### 3.4.1 Run 命令

2021-12-05 13:20:02 :heavy_check_mark: 搞定

:star: 使用 GPGPU-Sim 需要在运行目录下放置 3 个 config 文件

GPGPU-Sim 配置文件放在 `~/Parboil/benchmarks/bfs` 目录下即可，成功运行

```shell
$ cp ~/gpgpu-sim_distribution/configs/GTX480/* ~/Parboil/benchmarks/bfs/
$ ./parboil run bfs cuda 1M
```

### 3.5. Script

编写脚本实行后台并行运行

# Reference

http://www.gpgpu-sim.org/

http://www.danielwong.org/classes/ee260-w17/assignment1
