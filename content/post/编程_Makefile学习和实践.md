---
title: "Makefile学习和实践"
date: 2023-03-23T11:00:42+08:00
lastmod: 2023-03-23T11:00:42+08:00
draft: false
author: "Cory"
tags: ["makefile"]
categories: ["编程"]
---

# 1. 简介和基础知识

C语言中文网    http://c.biancheng.net/view/7096.html

## 1.1 Makefile文件是什么？

### 1.1.1 Definition

Makefile是什么？

+ 用于描述编译规则的工程文件
  + 即哪些文件先编译，哪些文件无需编译
  + 使得项目的编译自动化，不需要每次都手动输入一堆源文件和参数。
  + 可以理解为一个脚本语言，类似 Shell, Perl, Python

Windows 下的集成开发环境（IDE）已经内置了 Makefile，或者说会自动生成 Makefile，无需手写

对于 Linux，不懂 Makefile，就操作不了多文件编程，就完成不了相对于大的工程项目的操作。如果你想在 Linux(Unix) 环境下做开发的话，Makefile 是必须掌握的一项技能。

### 1.1.2 Case

比如多文件编译生成一个文件

```bash
gcc -o outfile name1.c name2.c ...
```

文件数量多了，就会有问题

### 1.1.3 链接库

+ C语言，编译的时候 gcc 只会默认链接一些基本的C语言标准库，很多源文件依赖的标准库都需要我们手动链接。
+ name1.c 用到了数学计算库 math 中的函数，我们得手动添加参数 -Im；
+ name5.c 使用到了线程，我们需要去手动添加参数 -lpthread。
  + 这个情况在写 CUDA 的时候就遇到了
+ 可以把要链接的库文件放在 Makefile 中，制定相应的规则和对应的链接顺序。这样只需要执行 make 命令，工程就会自动编译。

编译大的工程会花费很长的时间

+ Makefile 支持多线程并发操作，会极大的缩短我们的编译时间，

> 具体操作应该是 `make -j8`，使用八个线程

+ 并且当我们修改了源文件之后，编译整个工程的时候，make 命令只会编译我们修改过的文件，没有修改的文件不用重新编译，也极大的解决了我们耗费时间的问题。

## 1.2 Makefile 文件结构

### 1.2.1 目标，依赖 target，prerequisite

它的规则主要是两个部分组成，分别是依赖的关系和执行的命令，其结构如下所示：

```makefile
targets : prerequisites
    command
```
or
```makefile
targets : prerequisites; command
    command
```

- targets：规则的目标，可以是 Object File（一般称它为中间文件），也可以是可执行文件，还可以是一个标签；
  - 也就是 command 的输出
- prerequisites：是我们的依赖文件，要生成 targets 需要的文件或者是目标。可以是**多个**，也可以是没有
  - a "prerequisite" is a file or target that must exist before a particular target can be built；
  - 构造 target 的前置条件
- command：make 需要执行的命令（任意的 shell 命令）。可以有多条命令，每一条命令占一行。
- recipe：a series of commands that are executed to create or update a target

:exclamation:  Notion：我们的目标和依赖文件之间要使用冒号分隔开，命令的开始一定要使用`Tab`键。

+ 使用了空格就会报错
+ `Makefile:2: *** 缺失分隔符。 停止。`

> 2022-11-25 19:55:02。这个特点和 shell 文件类似，空格的标准是一开始很容易出错而意识不到的地方
#### 1.2.1.1 Case

```makefile
main : main.c fun1.c fun2.c
	gcc -o main main.c fun1.c fun2.c
```
实际情况中一行规则是肯定不够用的

## 1.3 变量

### 1.3.1 变量定义

基本语法:exclamation:

```makefile
变量的名称 = 值列表
```

变量的名称可以由大小写字母、阿拉伯数字和下划线构成。

:exclamation:等号左右的空白符没有明确的要求，因为在执行 make 的时候多余的空白符会被自动的删除。

> 2022-11-25 19:54:29。shell 赋值的话，在 `=` 两边不能留有空白。
+ 这一点很重要，因为 gcc 之前是必须用 `Tab`，而变量赋值无需

至于值列表，既可以是零项，又可以是一项或者是多项。
调用变量的时候可以用 "\$(VALUE_LIST)" 或者是 "${VALUE_LIST}" 来替换，这就是变量的引用

```makefile
OBJ=main.o test.o test1.o test2.o
test:$(OBJ)
      gcc -o test $(OBJ)
```

### 1.3.2 变量的基本赋值

- 简单赋值 ( := ) 编程语言中常规理解的赋值方式，只对当前语句的变量有效。
- 递归赋值 ( = ) 赋值语句可能影响多个变量，所有目标变量相关的其他变量都受影响。
- 条件赋值 ( ?= ) 如果变量未定义，则使用符号中的值定义变量。如果该变量已经赋值，则该赋值语句无效。
- 追加赋值 ( += ) 原变量用空格隔开的方式追加一个新值。

详细说明如下：

#### 1.3.2.1 简单赋值 :=

```makefile
x:=foo
y:=$(x)b
x:=new
test：
      @echo "y=>$(y)"
      @echo "x=>$(x)"
```

在 shell 命令行执行`make test`我们会看到:

```bash
y=>foob
x=>new
```

#### 1.3.2.2 递归赋值 =

```makefile
x=foo
y=$(x)b
x=new
test：      
	@echo "y=>$(y)"      
	@echo "x=>$(x)"
```

在 shell 命令行执行`make test`我们会看到:

```bash
y=>newb	#理解为当某变量 x 更新后，所以和 x 相关的变量都会更新
x=>new
```

#### 1.3.2.3 条件赋值 ?=

```makefile
x:=foo
y:=$(x)b
x?=new
test：      
	@echo "y=>$(y)"      
	@echo "x=>$(x)"
```

在 shell 命令行执行`make test`我们会看到:

```bash
y=>foob
x=>foo # x 已经定位为 foo 所以忽略掉 new 的赋值
```

#### 1.3.2.4 追加赋值 +=

```makefile
x:=foo
y:=$(x)b
x+=$(y)
test：      
	@echo "y=>$(y)"      
	@echo "x=>$(x)"
```

在 shell 命令行执行`make test`我们会看到:

```bash
y=>foob
x=>foo foob #为什么有空格？
```

### 1.3.3 通配符

回想一下数据库的知识

| 通配符 | 使用说明                           | 自动变量 | 说明                             |
| ------ | ---------------------------------- | -------- | -------------------------------- |
| *      | 匹配0个或者是任意个字符            | $<       | 第一个依赖文件                   |
| ？     | 匹配任意一个字符    # 注意是 1 个  | $@       | 目标                             |
| []     | 我们可以指定匹配的字符放在 "[]" 中 | $^       | 所有不重复的依赖文件，以空格分开 |

#### 1.3.3.1 Case 1

测试可用 :arrow_down:

```makefile
test:*.c	gcc -o $@ $^
```

这个实例可以说明我们的通配符不仅可以使用在规则的命令中，还可以使用在规则中。用来表示生所有的以 .c 结尾的文件。

+ 表示所有以 .c 结尾的文件同时编译，生成 test 文件

#### 1.3.3.2 Case 2

:warning: 讲述变量和通配符不要混用

```makefile
OBJ=*.c
test:$(OBJ)    
	gcc -o $@ $^
```

:star:我们去执行这个命令的时候会出现错误，提示我们没有 "\*.c" 文件，实例中我们相要表示的是当前目录下所有的 ".c" 文件，但是我们在使用的时候并没有展开，而是直接识别成了一个文件。文件名是 "\*.c"。

:parking: 不过自己测试的时候可以成功编译，这可能是 makefile 自己做了优化

### 1.3.4 wildcard 函数

如果我们就是相要通过引用变量的话，我们要使用一个函数 "wildcard"，这个函数在我们引用变量的时候，会帮我们展开。我们把上面的代码修改一下就可以使用了。

```makefile
OBJ=$(wildcard *.c)test:$(OBJ)    gcc -o $@ $^
```

这样我们再去使用的时候就可以了。调用函数的时候，会帮我们自动展开函数。

还有一个和通配符 "*" 相类似的字符，这个字符是 "%"，也是匹配任意个字符，使用在我们的的规则当中。

```makefile
test:test.o test1.o    gcc -o $@ $^%.o:%.c    gcc -o $@ $^
```

"%.o" 把我们需要的所有的 ".o" 文件组合成为一个列表，从列表中挨个取出的每一个文件，"%" 表示取出来文件的文件名（不包含后缀），然后找到文件中和 "%"名称相同的 ".c" 文件，然后执行下面的命令，直到列表中的文件全部被取出来为止。

## 1.4 简单实例

来自知乎    https://www.zhihu.com/question/23792247/answer/600773044


### 1.4.1 第一版

测试过，可以 work

```makefile
main : main.c fun1.c fun2.c	gcc -o main main.c fun1.c fun2.c
```

:exclamation: 缺点

+ 对于简单代码还好，而对于大型项目，具有成千上万代码来说，仅用一行规则是完全不够的，即使够的话也需要写很长的一条规则
+ 任何文件只要稍微做了修改就需要整个项目完整的重要编译
  + Which means 有办法在修改一部分时只编译那一小部分

2023-03-23 10:28:52，自己的理解：这样写的话，所有的 .c 文件都是 `main` 的依赖，那么任意改其中一个 .c 文件，`main` 都需要重新编译，src code 之间的独立性不强。

### 1.4.2 第二版

为了避免改动任何代码就需要重新编译整个项目的问题，我们将主规则的各个依赖替换成各自的中间文件
即main.c --> main.o，fun1.c --> fun1.o，fun2.c --> fun2.o，再对每个中间文件的生成分别写一条规则
比如对于main.o，规则为：

```makefile
main.o: main.c      
  gcc -c main.c -o main.o  
```

这样做的好处是，当有一个文件发生改动时，只需重新编译此文件即可，而无需重新编译整个项目。完整Makefile如下：

```makefile
app : main.o fun1.o fun2.o  	
  gcc main.o fun1.o fun2.o -o app  
main.o : main.c  	
  gcc -c main.c -o main.o  
fun1.o : fun1.c  	
  gcc -c fun1.c -o fun1.o  
fun2.o : fun2.c  	
  gcc -c fun2.c -o fun2.o
```

:warning: 注意不要加多余的空格，(gcc) 命令的开始一定要使用`Tab`键，不能是空格

:exclamation: 缺点

+ 里面存在一些重复的内容，可以考虑用变量代替；
+ 后面三条规则非常类似，可以考虑用一条模式规则代替。
  + 说白了也就是有重复

2023-03-23 10:31:39，但是这里的重复只是写法上的重复，而 `app` 对 src code 的依赖关系已经独立开来，能够提升重新编译的效率。
### 1.4.3 第三版

引入了变量的概念

在第三版Makefile中，我们使用变量及模式规则使Makefile更加简洁。使用的三个变量如下：

```makefile
obj = main.o fun1.o fun2.o  
target = app  
CC = gcc  
```

使用的模式规则为：

```makefile
%.o: %.c  	
  $(CC) -c $< -o $@ 
```

Means -> 所有的.o文件都由对应的.c文件生成。在规则里，我们又看到了两个自动变量：`$<` 和 `$@`。
其实自动变量有很多，常用的有三个：

+ `$<`：第一个依赖文件；
+ `$@`：目标；
+ `$^`：所有不重复的依赖文件，以空格分开

直接复制过来的源码是有问题的，需要手动调整一下 Tab。:arrow_down: 是可以使用的。

```makefile
obj = main.o fun1.o fun2.o  
target = app  
CC = gcc  
$(target): $(obj)  	
  $(CC) $(obj) -o $(target)  
%.o: %.c  	
  $(CC) -c $< -o $@     #这个 百分号% 就类似于数据库中的匹配 所有以 .o 结尾的是 target 以 .c 结尾的是 prerequisites
```

:exclamation: 缺点

+ obj对应的文件需要一个个输入，工作量大；
+ 文件数目比较少时还好，文件数目一旦很多的话，obj将很长；
+ 而且每增加/删除一个文件，都需要修改Makefile。

2023-03-23 10:35:37，这里的意思应该是要事先准备好所有需要的 .o 文件，而 .o 文件需要从 .c 文件编译得到。其中仍然是存在将所有 .c 编译成对应的 .o 文件的过程，不够自动化。

### 1.4.4 第四版

在第四版Makefile中，推出了两个函数：**wildcard** 和 **patsubst**。

#### 1.4.4.1 wildcard

作用：扩展通配符，搜索指定文件。在此我们使用

```makefile
src = $(wildcard ./*.c)，#代表在当前目录下搜索所有的.c文件，并赋值给src。函数执行结束后，src的值为：main.c fun1.c fun2.c。
```

#### 1.4.4.2 patsubst

作用：替换通配符，按指定规则做替换。在此我们使用

```makefile
obj = $(patsubst %.c, %.o, $(src))
```

代表将src里的每个文件都由.c替换成.o。函数执行结束后，`obj` 的值为 `main.o fun1.o fun2.o`，其实跟第三版 Makefile 的 obj 值一模一样，只不过在这里它更智能一些，也更灵活。

除了使用 `patsubst` 函数外，我们也可以使用模式规则达到同样的效果，比如：

```makefile
obj = $(src:%.c=%.o)
```

也是代表将src里的每个文件都由.c替换成.o。

几乎每个 Makefile 里都会有一个 **伪目标** clean，这样我们通过执行make clean命令就是将中间文件如 .o 文件及目标文件全部删除，留下干净的空间。一般是如下写法：

```makefile
.PHONY: clean  clean:  	rm -rf $(obj) $(target)  
```

.PHONY代表声明clean是一个 **伪目标**，这样每次执行 `make clean` 时，下面的规则都会被执行。

```makefile
src = $(wildcard ./*.c)  
obj = $(patsubst %.c, %.o, $(src))  
obj = $(src:%.c=%.o)  
target = app  
CC = gcc  
$(target): $(obj)
  	$(CC) $(obj) -o $(target)  
%.o: %.c
  	$(CC) -c $< -o $@  
.PHONY: clean  
clean:
    rm -rf $(obj) $(target)
```

# 2. 常用的编译选项 CLI option

给出几个自己用到过的选项

## 2.1 -n 打印所有编译命令，但并不执行

`make -n mlp_learning_an_image`

## 2.2 -s -f

`-s`：禁止输出编译命令信息 -s。
`-f`：用于指定 makefile 文件的名称或路径。它告诉 make 命令使用指定的 makefile 文件而不是默认的 Makefile 文件来执行编译。

下面的代码中，Makefile 文件是 `build.make`，因为 `make` 命令默认是找目录下的名为 `Makefile` 文件。这里的 Makefile 并非默认的名称，因此需要用 `-f` 选项来指定文件。而 `make` target 是 `dependencies/fmt/CMakeFiles/fmt.dir/build`

```makefile
make -s -f dependencies/fmt/CMakeFiles/fmt.dir/build.make dependencies/fmt/CMakeFiles/fmt.dir/build
```

# 3. 变量 variable

# 4. 条件语句和函数 conditional syntax and functions

# 5. 关键字 built-in targets

## 5.1 .PHONY

> In a Makefile, .PHONY is a special target that is used to declare a target that doesn't represent an actual file. Instead, it is used to specify a target that is always considered out of date, and therefore, its recipe will always be executed whenever it is requested.

也就是说在对于 .PHONY 关键字后面的内容，只要有请求，就会执行。还是很绕，直接看例子吧。使用 .PHONY 声明了 CMakeFiles/tiny-cuda-nn.dir/depend，也就是说只要输入命令 `make CMakeFiles/tiny-cuda-nn.dir/depend`，就会执行其命令行，不管有哪种依赖

```Makefile
CMakeFiles/tiny-cuda-nn.dir/depend:
	cd /home/wmhu/work/tiny-cuda-nn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wmhu/work/tiny-cuda-nn /home/wmhu/work/tiny-cuda-nn /home/wmhu/work/tiny-cuda-nn/build /home/wmhu/work/tiny-cuda-nn/build /home/wmhu/work/tiny-cuda-nn/build/CMakeFiles/tiny-cuda-nn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tiny-cuda-nn.dir/depend
```

再来一个例子，关于 clean。声明之后，不管 .o 文件和 bin 文件是不是存在，只要 `make clean`，就会执行 `rm -f *.o myprogram`

```Makefile
.PHONY: clean
clean:
    rm -f *.o myprogram
```

## 5.2 .DEFAULT

# 6. 应用技巧

这一章描述一些实践过程中的技巧，可以参考 chatgpt 给的大纲，查缺补漏，细细完善。


```shell
Makefile 基础知识

什么是 Makefile
Makefile 的工作原理
Makefile 的基本语法
Makefile 中的变量
Makefile 中的目标和依赖关系
Makefile 的常用命令

make 命令
clean 命令
distclean 命令
install 命令
uninstall 命令
Makefile 的高级应用

Makefile 中的条件语句
Makefile 中的函数
Makefile 中的自动化变量
Makefile 中的模式规则
Makefile 中的多目标规则
实际问题和解决方案

如何在 Makefile 中设置编译器和编译选项
如何在 Makefile 中编译和链接多个源文件
如何在 Makefile 中编译和链接静态库和动态库
如何在 Makefile 中定义和使用宏定义
如何在 Makefile 中使用条件编译
Makefile 的调试技巧

如何使用 Makefile 的调试选项
如何输出 Makefile 中的变量和宏定义
如何查看 Makefile 中的依赖关系
如何查看 Makefile 中的编译和链接命令
Makefile 的最佳实践

如何编写可移植的 Makefile
如何编写可维护的 Makefile
如何编写高效的 Makefile
如何组织 Makefile 的目录结构
以上是一些可以作为 Makefile 笔记的主题和子主题，你可以根据自己的需要进行扩充和调整。在整理笔记的过程中，你可以结合自己的实际学习和应用情况，将理论知识和实践问题结合起来，这样能够使笔记更加丰富和实用。
```


# 7. 通过实例学习

## 7.1 cutlass 实例

### 7.1.2 make 中的选项

`make -s -f CMakeFiles/Makefile2 samples/CMakeFiles/mlp_learning_an_image.dir/all`

这一句命令表示使用 CMakeFiles/Makefile2 文件中定义的规则来编译 samples/CMakeFiles/mlp_learning_an_image.dir/all 目标。-s 选项指示 make 命令在执行编译过程时不显示编译命令。

all 是一个伪目标，指示 make 命令编译所有的目标。在这个上下文中，samples/CMakeFiles/mlp_learning_an_image.dir/all 是一个特殊的目标，它实际上是由 CMake 自动生成的一个中间目标。它的作用是将 samples/CMakeFiles/mlp_learning_an_image.dir 目录中所有的源文件编译成目标文件，并将它们链接为一个可执行文件。

在 Makefile 中，通常会定义一个名为 all 的伪目标，用于编译所有的目标。all 目标通常是默认目标，也就是在不指定目标的情况下，make 命令将会编译它。当然，开发者也可以定义其他的伪目标来执行指定的编译任务。

什么是伪目标？

实际目标指的是编译生成的文件或其他需要执行的任务。
伪目标则是指用来控制 Makefile 中规则执行顺序的目标，不生成对应的文件。

伪目标的作用是在执行 make 命令时，告诉 make 命令要执行哪些规则。它通常用来控制 Makefile 中规则的执行顺序，或者执行一些特定的任务，如清除生成的文件等。

在 Makefile 中，伪目标的特征是没有对应的文件名，并且在规则中使用了 .PHONY 声明，以告诉 make 命令它是一个伪目标。例如：
```makefile
.PHONY: clean
clean:
    rm -f *.o
```

在上面的例子中，clean 是一个伪目标，它的作用是删除当前目录下的所有 .o 文件。.PHONY 声明告诉 make 命令 clean 是一个伪目标，不对应任何文件，只是一个控制规则执行顺序的目标。

需要注意的是，如果不使用 .PHONY 声明声明一个伪目标，那么当在当前目录下存在一个与目标同名的文件时，make 命令会将其误认为是实际目标，导致错误的行为。因此，定义伪目标时一定要加上 .PHONY 声明。

---

# 0. Cmake 和 Makefile 异同

这一章参考 [知乎文章](https://www.zhihu.com/question/27455963/answer/36722992)

## 0.1 功能描述

+ `make` command 是用来执行 Makefile 的
+ Makefile 是类 UNIX 环境下(比如Linux)的类似于批处理的"脚本"文件。
其基本语法是: **目标+依赖+命令**，只有在**目标**文件不存在，或**目标**比**依赖**的文件更旧，**命令**才会被执行。由此可见，Makefile和make可适用于任意工作，不限于编程。比如，可以用来管理latex。

> 2022-11-25 19:47:20，从这个描述来看，猜测判断 **目标** 是否 `Already Update..` 的依据是系统时间。判断 **目标** 的修改时间是否晚于 **依赖** 文件的修改时间。

+ Makefile+make可理解为类unix环境下的项目管理工具，但它太基础了，抽象程度不高，而且在windows下不太友好(针对visual studio用户)，于是就有了跨平台项目管理工具cmake
  
+ cmake是跨平台项目管理工具，它用更抽象的语法来组织项目。虽然，仍然是目标，依赖之类的东西，但更为抽象和友好，比如你可用math表示数学库，而不需要再具体指定到底是math.dll还是libmath.so，在windows下它会支持生成visual studio的工程，在linux下它会生成Makefile，甚至它还能生成eclipse工程文件。也就是说，从同一个抽象规则出发，它为各个编译器定制工程文件。
  
+ cmake是抽象层次更高的项目管理工具，cmake命令执行的CMakeLists.txt文件

+ cmake 抽象程度更高

+ qmake是Qt专用的项目管理工具，对应的工程文件是*.pro，在Linux下面它也会生成Makefile，当然，在命令行下才会需要手动执行qmake，完全可以在qtcreator这个专用的IDE下面打开*.pro文件，使用qmake命令的繁琐细节不用你管了。

**总结**

总结一下，make用来执行Makefile，cmake用来执行CMakeLists.txt，qmake用来处理*.pro工程文件。Makefile的抽象层次最低，cmake和qmake在Linux等环境下最后还是会生成一个Makefile。cmake和qmake支持跨平台，cmake的做法是生成指定编译器的工程文件，而qmake完全自成体系。

## 0.2 个人理解

+ :star: ​具体使用时，Linux下，小工程可手动写Makefile，
  + 所以学会自己手写 Makefile 对于一些测试工作也是比较重要的
+ 大工程用automake来帮你生成Makefile，
+ 要想跨平台，就用cmake。
+ 如果GUI用了Qt，也可以用qmake+*.pro来管理工程，这也是跨平台的。当然，cmake中也有针对Qt的一些规则，并代替qmake帮你将qt相关的命令整理好了。

另外，需要指出的是，make和cmake主要命令只有一条，make用于处理Makefile，cmake用来转译CMakeLists.txt，而qmake是一个体系，用于支撑一个编程环境，它还包含除qmake之外的其它多条命令(比如uic，rcc,moc)。

上个简图，其中cl表示visual studio的编译器，gcc表示linux下的编译器