---
title: "GPU_benchmark说明（转）"
date: 2021-11-14T20:34:48+08:00
lastmod: 
draft: false
author: "Cory"
tags: ["gpgpu-sim"]
categories: ["gpgpu"]
---

# Introduction

本文内容主要系摘录翻译自[Ang Li](http://parse.ele.tue.nl/ali)的博士毕业论文。

# 1.Perfect

Power Efficiency Revolution for Embedded Computing

http://hpc.pnl.gov/PERFECT/

| **Application Domains**        | **Kernels**                       |
| :----------------------------- | :-------------------------------- |
| PERFECT Application 1          | Discrete Wavelet Transform        |
|                                | 2D Convolution                    |
|                                | Histogram Equalization            |
| Space Time Adaptive Processing | System Solver                     |
|                                | Inner Product                     |
|                                | Outer Product                     |
| Synthetic Aperture Radar       | Interpolation 1                   |
|                                | Interpolation 2                   |
|                                | Back Projection (Non-Fourier SAR) |
| Wide Area Motion Imaging       | Debayer                           |
|                                | Image Registration                |
|                                | Change Detection                  |
| Required Kernels               | Sort                              |
|                                | FFT 1D                            |
|                                | FFT 2D                            |

# 2. AxBench

A Multiplatform Benchmark Suite for Approximate Computing

One of the goals of AxBench is to provide a diverse set of applications to further facilitate research and development in approximate computing.

http://ieeexplore.ieee.org/abstract/document/7755728/

下载地址

http://axbench.org/

| benchmark      | platform       | domain                 | Quality Metric      |
| :------------- | :------------- | :--------------------- | :------------------ |
| binarization   | GPU            | Image Processing       | Image Diff          |
| blackscholes   | CPU, GPU       | Finance                | Avg. Relative Error |
| brent-kung     | ASIC           | Arithmetic Computation | Avg. Relative Error |
| canneal        | CPU            | Optimization           | Avg. Relative Error |
| convolution    | GPU            | Machine Learning       | Avg. Relative Error |
| fastwalsh      | GPU            | Signal Processing      | Image Diff          |
| fft            | CPU            | Signal Processing      | Avg. Relative Error |
| fir            | ASIC           | Signal Processing      | Avg. Relative Error |
| forwardk2j     | CPU, ASIC      | Robotics               | Avg. Relative Error |
| inversek2j     | CPU, GPU, ASIC | Robotics               | Avg. Relative Error |
| jmeint         | CPU, GPU       | 3D Gaming              | Miss Rate           |
| jpeg           | CPU            | Image Processing       | Image Diff          |
| kmeans         | CPU, ASIC      | Machine Learning       | Image Diff          |
| kogge-stone    | ASIC           | Arithmetic Computation | Avg. Relative Error |
| laplacian      | GPU            | Image Processing       | Image Diff          |
| meanfilter     | GPU            | Machine Vision         | Image Diff          |
| neural network | ASIC           | Machine Learning       | Avg. Relative Error |
| newton-raph    | GPU            | Numerical Analysis     | Avg. Relative Error |
| sobel          | CPU, GPU, ASIC | Image Processing       | Image Diff          |
| srad           | GPU            | Medical Imaging        | Image Diff          |
| wallace-tree   | ASIC           | Arithmetic Computation | Avg. Relative Error |

# 3. Rodinia

http://rodinia.cs.virginia.edu/

下载页面：

http://lava.cs.virginia.edu/Rodinia/download_links.htm

|                                                              |                      |                           |                |                   |
| :----------------------------------------------------------- | :------------------- | :------------------------ | :------------- | :---------------- |
| Applications                                                 | Dwarves              | Domains                   | Parallel Model | Incre. Ver.       |
| [Leukocyte](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Leukocyte) | Structured Grid      | Medical Imaging           | CUDA, OMP, OCL | √                 |
| [Heart Wall](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Heart_Wall) | Structured Grid      | Medical Imaging           | CUDA, OMP, OCL |                   |
| [MUMmerGPU](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/MUMmerGPU) | Graph Traversal      | Bioinformatics            | CUDA, OMP      |                   |
| [CFD Solver1](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/CFD_Solver) | Unstructured Grid    | Fluid Dynamics            | CUDA, OMP, OCL |                   |
| [LU Decomposition](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/LU_Decomposition) | Dense Linear Algebra | Linear Algebra            | CUDA, OMP, OCL | √                 |
| [HotSpot](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/HotSpot) | Structured Grid      | Physics Simulation        | CUDA, OMP, OCL |                   |
| [Back Propagation](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Back_Propagation) | Unstructured Grid    | Pattern Recognition       | CUDA, OMP, OCL |                   |
| [Needleman-Wunsch](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Needleman-Wunsch) | Dynamic Programming  | Bioinformatics            | CUDA, OMP, OCL | √                 |
| [Kmeans](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/K-Means) | Dense Linear Algebra | Data Mining               | CUDA, OMP, OCL |                   |
| [Breadth-First Search1](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Graph_traversal) | Graph Traversal      | Graph Algorithms          | CUDA, OMP, OCL |                   |
| [SRAD](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/SRAD) | Structured Grid      | Image Processing          | CUDA, OMP, OCL | √                 |
| [Streamcluster1](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Streamcluster) | Dense Linear Algebra | Data Mining               | CUDA, OMP, OCL |                   |
| [Particle Filter](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Particle_Filter) | Structured Grid      | Medical Imaging           | CUDA, OMP, OCL |                   |
| [PathFinder](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Shortest_Path) | Dynamic Programming  | Grid Traversal            | CUDA, OMP, OCL |                   |
| [Gaussian Elimination](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Gaussian_Elimination) | Dense Linear Algebra | Linear Algebra            | CUDA, OCL      |                   |
| [k-Nearest Neighbors](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Nearest_Neighbor) | Dense Linear Algebra | Data Mining               | CUDA, OMP, OCL |                   |
| [LavaMD2](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/LavaMD) | N-Body               | Molecular Dynamics        | CUDA, OMP, OCL |                   |
| [Myocyte](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Myocyte) | Structured Grid      | Biological Simulation     | CUDA, OMP, OCL |                   |
| [B+ Tree](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/B%2B_Tree) | Graph Traversal      | Search                    | CUDA, OMP, OCL |                   |
| [GPUDWT](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/GPUDWT) | Spectral Method      | Image/Video Compression   | CUDA, OCL      |                   |
| [Hybrid Sort](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Hybrid_Sort) | Sorting              | Sorting Algorithms        | CUDA, OCL      |                   |
| [Hotspot3D](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php?title=Hotspot3D&action=edit&redlink=1) | Structured Grid      | Physics Simulation        | CUDA, OCL, OMP | Hotspot for 3D IC |
| [Huffman](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php?title=Huffman&action=edit&redlink=1) | Finite State Machine | Lossless data compression | CUDA, OCL      |                   |

Ang Li的分类：

| Application    | Description                                            | Domain                | CUDA | OpenCL | OpenMP |
| :------------- | :----------------------------------------------------- | :-------------------- | :--- | :----- | :----- |
| backprop       | Perceptron back propagation                            | Neural Network        | Yes  | Yes    | Yes    |
| bfs            | Breadth first search                                   | Graph Algorithm       | Yes  | Yes    | Yes    |
| b+tree         | B+tree Operation                                       | Searching Yes         | Yes  | Yes    |        |
| leukocyte      | Detect leukocytes in blood vessel video                | Medical Imaging       | Yes  | Yes    | Yes    |
| heartwall      | Tracks the mouse heart movement by stimulus            | Medical Imaging       | Yes  | No     | Yes    |
| cfd            | Finite volume solver for 3D Euler equations for flow   | Fluid Dynamics        | Yes  | Yes    | Yes    |
| lud            | Calculate the solutions of a set of linear equations   | Linear Algebra        | Yes  | Yes    | Yes    |
| hotspot        | Estimate processor temperature                         | Physical Simulation   | Yes  | Yes    | Yes    |
| nw             | Optimization method for DNA sequence alignments        | Bioinformatics        | Yes  | Yes    | Yes    |
| kmeans         | Clustering algorithm                                   | Data Mining           | Yes  | Yes    | Yes    |
| srad           | Speckle reducing anisotropic diffusion                 | Image Processing      | Yes  | Yes    | Yes    |
| streamcluster  | Finds medians to assign points to nearest centers      | Data Mining           | Yes  | Yes    | Yes    |
| particlefilter | Locate object location based on Noise and path         | Medical Imaging       | Yes  | Yes    | Yes    |
| pathfinder     | Dynamic programming to find a path on a 2D grid Grid   | Traversal             | Yes  | Yes    | Yes    |
| gaussian       | Solving variables in a linear system                   | Linear Algebra        | Yes  | Yes    | No     |
| nn             | Find k-nearest neighbors from an unstructured data set | Data Mining           | Yes  | Yes    | Yes    |
| lavaMD         | Calculate particle potential and relocation in 3D      | Molecular Dynamics    | Yes  | Yes    | Yes    |
| myocyte        | Simulate the behavior of cardiac hear muscle cell      | Biological Simulation | Yes  | Yes    | Yes    |

# 4. Parboil

Parboil强调面向吞吐量的流媒体应用。其中的每个应用都有原生的CUDA应用和优化过的应用。

| Application  | Description                                                | Domain              | CUDA | OpenCL | C    |
| :----------- | :--------------------------------------------------------- | :------------------ | :--- | :----- | :--- |
| bfs          | Breadth-first-search                                       | Graph Algorithm     | Yes  | Yes    | Yes  |
| cutcp        | Compute Coulombic potential for a 3D grid                  | Molecular Dynamics  | Yes  | Yes    | Yes  |
| histogram    | Compute 2D saturating histogram with maximum 256 bins      | Data Mining         | Yes  | Yes    | Yes  |
| lbm          | Fluid dynamics simulation using Lattice-Bolzmann Method    | Fluid Dynamics      | Yes  | Yes    | Yes  |
| mm           | Dense matrix-matrix multiply                               | Linear Algebra      | Yes  | Yes    | Yes  |
| mri-gridding | Compute regular data grid via weighted interpolation       | Medical Imaging     | Yes  | Yes    | Yes  |
| mir-q        | Compute scanner configuration for calibration in 3D MRI    | Medical Imaging     | Yes  | Yes    | Yes  |
| sad          | Sum of absolute differences kernel in MPEG video encoders  | Image Processing    | Yes  | Yes    | Yes  |
| spmv         | Compute the product of a sparse matrix with a dense vector | Linear Algebra      | Yes  | Yes    | Yes  |
| stencil      | An iterative Jacobi stencil operation on a regular 3D grid | Cellular Automation | Yes  | Yes    | Yes  |
| tpacf        | Analyze the spatial distribution of astronomical bodies    | Data Mining         | Yes  | Yes    | Yes  |

# 5. Shoc

测量协处理的稳定性和性能，such as GPUs, Xeon-Phi, etc。

| Application  | Description                                       | Domain              | CUDA | OpenCL | C    |
| :----------- | :------------------------------------------------ | :------------------ | :--- | :----- | :--- |
| qtclustering | Group genes into high quality clusters            | Bioinformatics      | Yes  | No     | No   |
| s3d          | Compute chemical reaction rate across a 3D grid   | Simulation          | Yes  | Yes    | No   |
| scan         | Parallel prefix sum of floating point numbers     | Data Mining         | Yes  | Yes    | No   |
| reduction    | Sum reduction operation of floating point numbers | Data Mining         | Yes  | Yes    | No   |
| md           | Lennard-Jones potential computations              | Molecular Dynamics  | Yes  | Yes    | No   |
| fft          | Fast Fourier transform                            | Signal Processing   | Yes  | Yes    | No   |
| sgemm        | Single precision general matrix multiply          | Linear Algebra      | Yes  | Yes    | No   |
| sort         | Fast radix sort program                           | Data Mining         | Yes  | Yes    | No   |
| stencil2d    | Standard 2d 9 points stencil calculation          | Cellular Automation | Yes  | Yes    | No   |
| bfs          | Breadth-first-search                              | Graph Algorithm     | Yes  | Yes    | No   |
| spmv         | Sparse matrix vector multiplication               | Linear Algebra      | Yes  | Yes    | Yes  |

# 6. Polybench

包含从[非]结构循环嵌套转换的Kernel。这些循环以前用于评估基于多面体模型的优化工具。

| Application | Description                                | Domain         | CUDA | OpenCL | C    |
| :---------- | :----------------------------------------- | :------------- | :--- | :----- | :--- |
| 2dconv      | 2D convolution                             | Linear Algebra | Yes  | Yes    | Yes  |
| 2mm         | 2 matrix multiply                          | Linear Algebra | Yes  | Yes    | Yes  |
| 3dconv      | 3D convolution                             | Linear Algebra | Yes  | Yes    | Yes  |
| 3mm         | 3 matrix multiply                          | Linear Algebra | Yes  | Yes    | Yes  |
| atax        | Matrix transpose and vector multiplication | Linear Algebra | Yes  | Yes    | Yes  |
| bicg        | Bicg kernel for BiCGStab linear solver     | Linear Algebra | Yes  | Yes    | Yes  |
| corr        | Correlation computation                    | Linear Algebra | Yes  | Yes    | Yes  |
| covar       | Covariance computation                     | Linear Algebra | Yes  | Yes    | Yes  |
| fdtd2d      | 2D finite difference time domain kernel    | Simulation     | Yes  | Yes    | Yes  |
| gemm        | matrix multiply                            | Linear Algebra | Yes  | Yes    | Yes  |
| gesummv     | Scalar vector and matrix multiplication    | Linear Algebra | Yes  | Yes    | Yes  |
| gramschm    | Gram-schmidt process                       | Linear Algebra | Yes  | Yes    | Yes  |
| mvt         | Matrix vector product and transpose        | Linear Algebra | Yes  | Yes    | Yes  |
| syr2k       | Symmetric rank-2k operations               | Linear Algebra | Yes  | Yes    | Yes  |
| syrk        | Symmetric rank-k operations                | Linear Algebra | Yes  | Yes    | Yes  |

# 7. Mars

用map reduce实现的data-mining的benchmark。

| Application | Description                                   | Domain         | CUDA | OpenCL | C    |
| :---------- | :-------------------------------------------- | :------------- | :--- | :----- | :--- |
| sm          | Find the position of a string in a file       | Data Mining    | Yes  | No     | No   |
| ii          | Build inverted index for links in HTML files  | Data Mining    | Yes  | No     | No   |
| ss          | Compute pair-wise similarity score for docs   | Data Mining    | Yes  | No     | No   |
| mm          | Multiply two matrices                         | Linear Algebra | Yes  | No     | No   |
| pvc         | Count distinct page views from web logs       | Data Mining    | Yes  | No     | No   |
| pvr         | Find the top ten hottest pages in the web log | Data Mining    | Yes  | No     | No   |

# 8. Longstar

关注于不规则的应用，主要是数据依赖和拓扑依赖。

| Application | Description                                            | Domain               | CUDA | OpenCL | C    |
| :---------- | :----------------------------------------------------- | :------------------- | :--- | :----- | :--- |
| bfs         | Breadth first search                                   | Graph Algorithm      | Yes  | No     | No   |
| bh          | Simulate the gravitational forces in Barnes-Hut        | algorithm Simulation | Yes  | No     | No   |
| dc          | Lossless compression upon double-precision FP data     | Signal Processing    | Yes  | No     | No   |
| dmr         | Meshrefinement algorithm from computational geometry   | Image Processing     | Yes  | No     | No   |
| pta         | Andersen’s flow/context-insensitive points-to analysis | Graph Algorithm      | Yes  | No     | No   |
| sp          | Heuristic SAT-solver based on BaYesian inference       | Graph Algorithm      | Yes  | No     | No   |
| sssp        | Shortest path in a directed graph with weighted edges  | Graph Algorithm      | Yes  | No     | No   |
| tsp         | Traveling salesman problem                             | Graph Algorithm      | Yes  | No     | No   |

# 9. CUDA SDK

| Application        | Description                                             | Domain                          | CUDA                | OpenCL | C    |
| :----------------- | :------------------------------------------------------ | :------------------------------ | :------------------ | :----- | :--- |
| bilateralFilter    | Edge-preserving non-linear smoothing filter             | Image Processing                | Yes                 | Yes    | Yes  |
| binomialOption     | Evaluate option call price using binomial model         | Computational Finance           | Yes                 | Yes    | Yes  |
| BlackScholes       | Evaluate option call price using Black-Scholes model    | Computational Finance           | Yes                 | Yes    | Yes  |
| convolutionFFT2D   | 2D convolutions using FFT                               | Image Processing                | Yes                 | Yes    | Yes  |
| dct8x8             | Discrete cosine transform for blocks of 8 by 8 pixels   | Image Processing                | Yes                 | Yes    | Yes  |
| dxtc               | High quality DXT compression                            | Image Processing                | Yes                 | Yes    | Yes  |
| dwtHaar1D          | 1D discrete Haar wavelet decomposition                  | Image Processing                | Yes                 | Yes    | Yes  |
| eigenvalues        | Eigenvalues of a tridiagonal symmetric matrix           | Linear Algebra                  | Yes                 | Yes    | Yes  |
| fastWalshTransform | Hadamard-ordered Fast Walsh transform                   | Linear Algebra                  | Yes                 | Yes    | Yes  |
| FDTD3d             | Finite differences                                      | time domain progression stencil | Cellular Automation | Yes    | Yes  |
| grabcutNPP         | GrabCut approach using the 8 neighborhood               | Graph Algorithm                 | Yes                 | Yes    | Yes  |
| histogram          | 64/256 bin histogram                                    | Data Mining                     | Yes                 | Yes    | Yes  |
| imageDenoising     | Using KNN and NLM for image denoising                   | Image Processing                | Yes                 | Yes    | Yes  |
| lineOfSight        | A simple line-of-sight algorithm                        | Graphic Application             | Yes                 | Yes    | Yes  |
| Mandelbrot         | Mandelbrot or Julia sets interactively                  | Graphic Application             | Yes                 | Yes    | Yes  |
| matrixMul          | Matrix multiplication                                   | Linear Algebra                  | Yes                 | Yes    | Yes  |
| mergeSortv         | Merge Sort algorithm                                    | Data Mining                     | Yes                 | Yes    | No   |
| MersenneTwister    | The Mersenne Twister random number generator            | Signal Processing               | Yes                 | Yes    | Yes  |
| MonteCarlo         | Evaluate option call price using Monte Carlo approach   | Computational Finance           | Yes                 | Yes    | Yes  |
| nbody              | All-pairs gravitational n-body simulation               | Simulation                      | Yes                 | Yes    | Yes  |
| oceanFFT           | Simulate an Ocean height field                          | Simulation                      | Yes                 | Yes    | Yes  |
| reduction          | Compute the sum of a large arrays of values             | Data Mining                     | Yes                 | Yes    | No   |
| scalarProd         | Calculate scalar products of input vector pairs         | Linear Algebra                  | Yes                 | Yes    | Yes  |
| scan               | Parallel prefix sum                                     | Data Mining                     | Yes                 | Yes    | Yes  |
| SobelFilter        | Sobel edge detection filter for 8-bit monochrome images | Image Processing                | Yes                 | Yes    | Yes  |
| SobolQRNG          | Sobol Quasirandom Sequence Generator                    | Computational Finance           | Yes                 | Yes    | Yes  |
| transpose          | Matrix transpose                                        | Linear Algebra                  | Yes                 | Yes    | Yes  |

# 10. GPGPU-Sim

| Application | Description                                                  | Domain                | CUDA | OpenCL | C    |
| :---------- | :----------------------------------------------------------- | :-------------------- | :--- | :----- | :--- |
| aes         | AES algorithm in CUDA to encrypt and decrypt files           | Cryptography          | Yes  | No     | No   |
| dc          | A discontinuous Galerkin time-domain solver                  | Simulation            | Yes  | No     | No   |
| lps         | 3D Laplace Solver                                            | Computational Finance | Yes  | No     | No   |
| lib         | Monte Carlo simulation in London-interbank-offered-rate Model | Computational Finance | Yes  | No     | No   |
| mum         | Pairwise local sequence alignment for DNA string             | Bioinformatics        | Yes  | No     | No   |
| nn          | Convolutional neural network to recognize handwritten digits | Machine Learning      | Yes  | No     | No   |
| nqu         | The N-Queen solver                                           | Simulation            | Yes  | No     | No   |
| ray         | Ray-tracing (rendering graphics with near photo-realism)     | Graphic Application   | Yes  | No     | No   |
| sto         | Sliding-window implementation of the MD5 algorithm           | Data Mining           | Yes  | No     | No   |
| wp          | Accelerate part of the Weather Research and Forecast Model (WRF) | Simulation            | Yes  | No     | No   |

# Reference

https://www.findhao.net/easycoding/2304.html

