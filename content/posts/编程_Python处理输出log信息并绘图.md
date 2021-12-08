---
title: "Python处理输出log信息并绘图"
date: 2021-12-08T09:05:41+08:00
lastmod: 2021-12-08 09:06:13
draft: false
author: "Cory"
tags: ["Python", "可视化"]
categories: ["编程"]
---

# 0. 前言

修改 GPGPU-Sim 并跑 benchmark，如果一次用12个benchmark，3种调度算法，那么一次会生成36个 output log。需要使用 python 脚本可视化这些数据来进行 high level 的分析，因此自己写了一个脚本进行输出数据的可视化工作。

需要两个绘图工具

+ 对于单个 benchmark，分析其 ipc，cache hit/miss，mem_stall 等等
+ 对于多个 benchmark，分析总体的 ipc，cache hit/miss，mem_stall 等等

# 1. 正则表达式

## 1.1 实例

```python
def read_string(file,metrics):
    
    output={}
    for it_metrics in metrics:
        if(it_metrics=="gpu_ipc"):
            pattern=re.compile(it_metrics+" =(\s+)(\d+\.\d+)")
        elif(it_metrics=="Stall"):
            pattern=re.compile(it_metrics+":(\d+)")
        else:
            pattern=re.compile(it_metrics+" = (\d+)")
        output_sum=0
        for i,line in enumerate(open(get_file_path()+file)):
            for match in re.finditer(pattern, line):
                if(it_metrics=="gpu_ipc"):
                    output_part=list(match.group(2))
                else:
                    output_part=list(match.group(1))
                output_part=float(''.join(output_part))
                output_sum+=output_part
        output[it_metrics]=output_sum
    return output
```

其实没有找到最舒服的正则表达式，理想情况是读取到表示数据的一串字符串，然后直接转化为浮点数。但是各种匹配方法似乎都是一次匹配一个数字/字符串，所以先使用现成的。

`for i,line in enumerate(open(file))` 遍历 log 的每一行，一定要加上 `i` 否则会报错。

`match.group(1)` 每次会返回一个整数，这个时候整数是元组的形式，所以需要先转化为 list，然后转化为 float 类型，这个时候便可以用于算术运算。

### 1.1.1 group 

+ 似乎有点理解这个匹配了，对于 `gpu_ipc`，\s+ 匹配的是至少一个空格，实际匹配的是多个空格，如果使用 group(1) 就会返回这个元组
+ 我们想要的是 \d+，所以返回第二个元组，再将其转化为 float 即可 
+ 2021-10-28 20:11:01：搞定，小数部分也能保留

## 1.2 字符串开头/结尾匹配

检查字符串开头或结尾的一个简单方法是使用 `str.startswith()` 或者是 `str.endswith()` 方法。

```python
>>> filename = 'spam.txt'
>>> filename.endswith('.txt')
True
```

# 2. 文件操作

## 2.1 Python OS 文件/目录方法

**os** 模块提供了非常丰富的方法用来处理文件和目录。常用的方法如下表所示

`os.listdir(path)`，返回path指定的文件夹包含的文件或文件夹的名字的列表。

# 3. 绘图

## 3.1 并列柱状图

绘制并列柱状图与堆叠柱状图类似，都是绘制多组柱体，只需要控制好每组柱体的位置和大小即可。例如：

:exclamation: 注意设置柱状图宽度不可忽略，否则无法正常显示

```python
import numpy as np
import matplotlib.pyplot as plt

size = 5
x = np.arange(size)
a = np.random.random(size)
b = np.random.random(size)
c = np.random.random(size)

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, a,  width=width, label='a')
plt.bar(x + width, b, width=width, label='b')
plt.bar(x + 2 * width, c, width=width, label='c')
plt.legend()
plt.show()
```

<img src="./Img/Python_柱状图.png" alt="img" style="zoom:67%;" />

# 4. 调试

## 4.1 调试时需要输入

# 5. (), [], {}

## 5.1 () 元组

代表tuple元祖数据类型，元祖是一种不可变序列。创建方法很简单，大多数时候都是小括号括起来的。

## 5.2 [] 列表

代表list列表数据类型，列表是一种可变序列。创建方法既简单又特别。

## 5.3 {} 字典

代表dict字典数据类型，字典是Python中唯一内建的映射类型。字典中的值没有特殊的顺序，但都是存储在一个特定的键（key）下。键可以是数字、字符串甚至是元祖。

### 5.3.1 字典

字典的每个键值 **key=>value** 对用冒号 **:** 分割，每个键值对之间用逗号 **,** 分割，整个字典包括在花括号 **{}** 中 ,格式如下所示：

```python
d = {key1 : value1, key2 : value2 }
```

相当于是键值对索引，key 只能是 `字符串，数字或元组`

+ python 中单引号和双引号字符串没有区别

### 5.3.2 用到了字典排序、字典转化为 list

```python
    #按键(key)排序:
    sorted(y_lrr_dict)
    sorted(y_gto_dict)
    sorted(y_cory_dict)
    y_lrr=list(y_lrr_dict)
    y_gto=list(y_gto_dict)
    y_cory=list(y_cory_dict)
```
