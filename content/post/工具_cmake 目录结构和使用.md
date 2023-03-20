---
title: "cmake 目录结构和使用"
date: 2022-05-09T22:17:43+08:00
lastmod: 2023-03-20T22:36:43+08:00
draft: false
author: "Cory"
tags: ["cmake"]
categories: ["编程"]
---

# 0. 前言

cuTLASS 使用到了 cmake，之前没有接触过，先学习一下他的目录结构和编译过程。

2023-03-16 11:25:11，在学习一个工具时，Getting Started 或许是最快的入门方式。

## 0.1 Cmake 简介

对于 C++ 程序，手动编写 Makefile 非常麻烦。cmake 用于自动编写 Makefile。通过读取 CMakeLists.txt 文件，可以自动生成 make 文件。cmake 中 macro 和 function 的使用，使得 cmake 更像是一个脚本语言。

## 0.2 安装

官方给出了建议的环境，cmake 没有安装，apt-get install 安装的是 3.12 版本，不符合要求。手动安装一下 3.20 cmake，[教程](https://gist.github.com/bmegli/4049b7394f9cfa016c24ed67e5041930)

```shell
# get and build CMake
wget https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0.tar.gz
tar -zvxf cmake-3.24.0.tar.gz
cd cmake-3.24.0
./bootstrap
make -j8

# add path
export PATH=$PATH:$CUDA_INSTALL_PATH/bin:~/cmake-3.24.0/bin
```

注意有个 BUG，` Could NOT find OpenSSL,`，`apt-get install libssl-dev` 即可

## 0.3 基本特性

cmake 命令不区分大小写，但是变量区分大小写。

# 1. cmake 目录

## 1.1 目录结构

```shell
|-- bin  #存放可执行文件
|-- build  #目录用于存放编译生成的文件
|  |-- dependencies  #存放 external libraries or dependencies that are required by the project being built
|  |-- CMakeFiles
|-- CMakeLists.txt
|-- include  #统一存放头文件
|  |-- hello.h
|  |-- gpgpu.h
|-- lib
|-- README.md
|-- src
|  |-- CMakeLists.txt
|  |-- main.cpp
|  |-- model1
|  |  |-- CMakeLists.txt
|  |  |-- gpgpu.cpp
|  |  |-- model.cpp
|  |-- model2
|  |  |-- CMakeLists.txt
|  |  |-- hello.cpp
|  |  |-- model.cpp
```
对于大一点的项目可能还会需要 util 目录，library 目录夹或者 tool 目录

**src**: 这个 example 包含了 2 models，main.cpp 依赖于 2 基础 models，另外，注意到他们都包含 `CMakeLists.txt` 文件

## 1.2 理解 CMakeLists.txt

注意 cmake 文件中不区分大小写

> In a CMake-based project, each directory containing C++ source code should contain a CMakeLists.txt file. This file describes the rules for building the code in that directory.

2023-03-16 14:19:11，每个 dir 下都要有一个 CMakeLists.txt

### 1.2.1 Baseline

最基础的包含以下一些信息，
```shell
# 规定该CMakeLists.txt适用的cmake最小版本，这里是 3.12，自己手动安装了 3.20 版本
cmake_minimum_required(VERSION 3.12.4 FATAL_ERROR)

# 项目名称，也就是 cutlass
project(CUTLASS VERSION 2.9.0 LANGUAGES CXX)

# 定义生成的可执行文件(程序)的名称，假设为 gemm
# 这里没找到 cutlass 对应的，cutlass 中有一个 function(cutlass_add_executable_tests NAME TARGET)
add_executable (gemm gemm.cxx)

# 指定头文件搜索路径，根目录下 include
include_directories (include)
```

# 2. cmake 编译过程
## 2.1 build, 编译并运行

build 目录用于存放编译生成的文件，一般的编译过程：
```shell
$ mkdir build && cd build
$ cmake .. -DCUTLASS_NVCC_ARCHS=75  # compile for NVIDIA Turing GPU architecture
```
## 2.2 问题，可执行程序在哪个目录生成？ 

看起来似乎是和 Makefile 文件同一目录，而 cutlass 中，执行 make 操作之后，会对程序进行编译，然后直接运行。由于在 sim 上运行需要把 gpgpusim.config 放到程序运行的目录下，所以我们需要知道是在哪个目录运行的。

cmake 设置 library and executable 文件的存放路径：
```shell
set(LIBRARY_OUTPUT_PATH path)
set(EXECUTABLE_OUTPUT_PATH path)
```
> 如果子目录中的某个CMakeLists.txt中设置了 set(...)，就以当前文件中设置的路径为主，否则以父目录中设置的路径为主

如果都没设置呢？从简单的程序来看，在 `build` 目录下 `cmake .. XXXX`，在 `build` 目录下生成 Makefile，执行 `make & make install`，可执行程序就在当前（`build`） 目录下。

## 2.3 cmake 文件变量赋值

cutlass 编译 example 报错，不确定问题是不是出在变量的传递。

2023-03-14 15:49:24，似乎是编译 example 的方式不对。

## 2.4 制定 cuda 编译器

使用 CMAKE_CUDA_COMPILER 这个内建变量可以做到。

可以通过命令 `cmake -DCMAKE_CUDA_COMPILER="xxx"` 来修改

# 3. 命令行参数

## 3.1 -D

原来 -D 是一个传参选项，怪不得在 CMakeList 里面直接搜索 -DCUTLASS_NVCC_ARCHS，没有找到这个命令。

-D 的作用就是定义变量的默认值，比如 `-DCUTLASS_NVCC_ARCHS=75`，也就是定义了 `CUTLASS_NVCC_ARCHS` 的值

此外，`CUTLASS_NVCC_ARCHS` 的属性应该得是一个 CACHE STRING，否则可能无法被改变

```shell
set(CUTLASS_NVCC_ARCHS ${CUTLASS_NVCC_ARCHS_SUPPORTED} CACHE STRING "The SM architectures requested.")
set(CUTLASS_LIBRARY_KERNELS "" CACHE STRING "Comma delimited list of kernel name filters. If unspecified, only the largest tile size is enabled. If 'all' is specified, all kernels are enabled.")
```

那么，尝试修改 `CMAKE_CUDA_ARCHITECTURES`，找到给其赋值，并且属性为 CACHE STRING 的 `TCNN_CUDA_ARCHITECTURES`，使用命令行实现：

```shell
cmake . -B build -DTCNN_CUDA_ARCHITECTURES="75"
```

## 3.2 添加编译选项
对于 C/C++ 代码，一般来说，编译选项命名为 FLAG。可以通过 `add_compilie_options` 命令设置编译选项，也可以通过 `set` 命令修改 `CMAKE_CXX_FLAGS` 或者 `CMAKE_C_FLAGS`。

+ `add_compile_options` 添加的选项是针对所有编译器，包括 C 和 C++
+ set 命令设置的  `CMAKE_CXX_FLAGS` 或者 `CMAKE_C_FLAGS` 变量分别针对 C 和 C++

e.g.

```cmake
#判断编译器类型,如果是gcc编译器,则在编译选项中加入c++11支持
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
    message(STATUS "optional:-std=c++11")
endif(CMAKE_COMPILER_IS_GNUCXX)

add_compile_options(<option> ...)
# 例子
add_compile_options(-Wall -Wextra -pedantic -Werror -g)
```

## 3.3 cmake_progress_start



# 4. Command 命令

## 4.1 set

### 4.1.1 Set Normal Value

`set (<variable> <value>... [PARENT_SCOPE])`
`set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")`


> If the PARENT_SCOPE option is given the variable will be set in the scope above the current scope. Each new directory or function() command creates a new scope. A scope can also be created with the block() command.

目前还没有太明白 scope 的概念。
### 4.1.2 Set Cache Entry

`set(<variable> <value>... CACHE <type> <docstring> [FORCE])`

`FORCE` option to overwrite existing entries

## 4.2 message

这个函数的功能是打印

`message(STATUS "Obtained CUDA architectures from CMake variable TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES}")`

## 4.3 function

```cmake
function(<name> [<arg1> ...])
  <commands>
endfunction()
```

NOTE: A function opens a new scope: see set(var PARENT_SCOPE) for details.

function 伴随 scope 的概念，或许可以用局部变量和全局变量去理解。

## 4.4 list

list 有很多子命令，包括 `APPEND`, `INSERT` 等，用于修改变量。

```cmake
if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcudafe=--diag_suppress=unrecognized_gcc_pragma")
```

# 5. Object file and link

在目录 `mlp_learning_an_image.dir` 生成 `.o` 文件之后，同一个目录下有 `link.txt` 文件，这个文件就是用来进行链接的。

解析一下

```shell
/usr/bin/c++ -O3 -DNDEBUG CMakeFiles/mlp_learning_an_image.dir/mlp_learning_an_image.cu.o CMakeFiles/mlp_learning_an_image.dir/__/dependencies/stbi/stbi_wrapper.cpp.o -o ../mlp_learning_an_image   -L/usr/local/cuda/targets/x86_64-linux/lib  ../libtiny-cuda-nn.a -lcuda ../dependencies/fmt/libfmt.a -lcudadevrt -lcudart_static -lrt -lpthread -ldl 
# ../libtiny-cuda-nn.a 这个是 tiny-cuda-nn 那一堆编译生成的，也会被链接进来

nvcc -o mlp_learning_an_image CMakeFiles/mlp_learning_an_image.dir/mlp_learning_an_image.cu.o CMakeFiles/mlp_learning_an_image.dir/__/dependencies/stbi/stbi_wrapper.cpp.o -L/usr/local/cuda/targets/x86_64-linux/lib  ../libtiny-cuda-nn.a -lcuda ../dependencies/fmt/libfmt.a -lcudadevrt -lcudart_static -lrt -lpthread -ldl

```

`/usr/bin/ar qc libtiny-cuda-nn.a "CMakeFiles/tiny-cuda-nn.dir/src/common.cu.o"...`，`ar` 命令用于创建和管理 static lib, qc 是其 opti 

# Reference

https://zhuanlan.zhihu.com/p/93895403