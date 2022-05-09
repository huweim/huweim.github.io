---
title: "cmake 目录结构和使用"
date: 2022-05-09T22:17:43+08:00
lastmod: 
draft: false
author: "Cory"
tags: ["cmake"]
categories: ["工具"]
---

# 0. 前言

cuTLASS 使用到了 cmake，之前没有接触过，先学习一下他的目录结构和编译过程。

## 0.1 Cmake 简介

对于 C++ 程序，手动编写 Makefile 非常麻烦。cmake 用于自动编写 Makefile。通过读取 CMakeLists.txt 文件，可以自动生成 make 文件。cmake 中 macro 和 function 的使用，使得 cmake 更像是一个脚本语言。

# 1. cmake 目录

## 1.1 目录结构

```shell
|-- bin  #存放可执行文件
|-- build  #目录用于存放编译生成的文件
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
