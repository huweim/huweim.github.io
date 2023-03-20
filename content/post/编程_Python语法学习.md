---
title: "Python 语法学习"
date: 2022-09-14T11:15:41+08:00
lastmod: 2022-09-14T11:15:41+08:00
draft: false
author: "Cory"
tags: ["Python"]
categories: ["编程"]
---

##### 函数

`def forward(self, x: torch.Tensor) -> torch.Tensor:`，箭头 `->` 后面表示返回值

尝试在 class 内部调用成员函数，报了错。调用 class 内部成员函数要加上前缀 `self.`，否则会被当成外部函数

**Python 不支持函数重载**

# 1. Attribute

## 1.1 string

用 for 循环实现了 list 中的元素转为 string，目的是用来索引。for 感觉比较麻烦，不易读也不优雅，有没有更好的方法？

### 1.1.1 Operation

#### 1.1.1.1 slice 
```python
b = "Hello, World!"
print(b[2:5])
# llo
print(b[2:])
# llo, World!                     
```

#### 1.1.1.2 replace
```python
string.replace(oldvalue, newvalue, count)
# replaces a specified phrase with another specified phrase.
```

#### 1.1.1.3 Split
The split() method splits a string into a list
```python
string.split(separator, maxsplit)
for file in files:
    # file -> '64_768_192.log'
    str_tmp = file.split('.')[0]
    # file.split('.') -> ['64_768_192', 'log']
    # file.split('.')[0] -> '64_768_192'
    gemm_data[str_tmp] = process_result("ant", file)
```

#### 1.1.1.4 Format

```python
txt1 = "My name is {fname}, I'm {age}".format(fname = "John", age = 36)
txt2 = "My name is {0}, I'm {1}".format("John",36)
txt3 = "My name is {}, I'm {}".format("John",36)
print(txt1)
# My name is John, I'm 36
print(txt2)
# My name is John, I'm 36
print(txt3)
# My name is John, I'm 36
```
### 1.1.2 实际使用时的一些需求

#### 1.1.2.1 list 转 string
**list item 是 string 类型，join**

join, return values are strings

注意，使用这个方法，list 中的元素也需要是字符串，所以对于 list item 不是字符串的情况，也许还是得用 for loop。
```python
# list = [1, 2, 3, 4, 5]
list = ['1', '2', '3', '4', '5']
''.join(list) # get "12345"
','.join(list) # get "1,2,3,4,5" 
```

**list item 不是 string 类型**      

```python
for it in list:
    conv_list += str(it) + ' '
```

#### 1.1.2.2 string 转 list
**使用 list 函数**
```python
import string
str_ = 'abcde'
list1 = list(str_)
print(list1)
# ['a', 'b', 'c', 'd', 'e']
```

**使用 split() 函数**，根据 string 中的某个分隔符来划分 element

```python
for file in files:
    # file -> '64_768_192.log'
    str_tmp = file.split('.')[0]
    # file.split('.') -> ['64_768_192', 'log']
    # file.split('.')[0] -> '64_768_192'
    gemm_data[str_tmp] = process_result("ant", file)
```
# 1.2 list

### 1.2.1 一些性质

+ `m_list[-1]` 直接索引到最后一个元素；`m_list[-2]` 索引倒数第二个

### 1.2.2 实际使用时的需要

#### 1.2.2.1 multi-dimension to one-dimension

也就是多维列表转一维列表，有些时候只是想拿到所有数据并绘图或者做其他处理，不需要其 shape

对于二维列表，方法比较多，对于多维，目前找到一种方法，并且必须知道具体维度，写对应数量的 for 循环处理

```python
# from_iterable 方法
from itertools import chain
tmp_list = list(chain.from_iterable(grad_output[0].tolist()))
# grad_output[0].tolist() 是一个二维 list

# 使用 for 循环遍历，对于 四维 list
tmp_list = [element for batch in grad_output[0].tolist() for channel in batch for height in channel for element in height]
```

#### 1.2.2.2 切片

```python
tmp_list = ['CS', 'EE', 'EECS']
# 取前两个元素
tmp_list[:2]
# 从 index = 1 开始，取两个元素
tmp_list[1:3]

# 前 10 个数，每 2 个数取一个
tmp_list[:10:2]

# 所有数，每 5 个取一个
tmp_list[::5]
```
#### 1.2.2.3 迭代

```python
# 判断是否可以迭代
>>> from collections.abc import Iterable
>>> isinstance('abc', Iterable) # str是否可迭代
True
>>> isinstance(123, Iterable) # 整数是否可迭代
False

# 下标循环，用 enumerate
>>> for i, value in enumerate(['A', 'B', 'C']):
...     print(i, value)
...
0 A
1 B
2 C
# 二维 list 遍历
>>> for x, y in [(1, 1), (2, 4), (3, 9)]:
...     print(x, y)
...
1 1
2 4
3 9
```

#### 1.2.2.4 列表生成

```python
# 最基本的用 range
>>> list(range(1, 11))
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 用 for 循环
>>> [x * x for x in range(1, 11)]
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# 比如列出当前目录下的所有文件和目录名
>>> import os 
>>> [d for d in os.listdir('.')] # os.listdir可以列出文件和目录
['.emacs.d', '.ssh', '.Trash', 'Adlm', 'Applications', 'Desktop', 'Documents', 'Downloads', 'Library', 'Movies', 'Music', 'Pictures', 'Public', 'VirtualBox VMs', 'Workspace', 'XCode']
```


## 1.3 tuple

### 1.3.1 基本性质

`m_tuple = ('CS', 'EE', 'EECS')`

和 list 比较类似，不过 tuple 不能修改，只读；不过看了一些拓展，这个不可变似乎只是指针的指向不变，而指针对象本身是否改变就不知道了。

**只有一个元素的 tuple 为了消除歧义**，会表示为 `t = (1, )`；这一点和我打印出来的结果是一致的，一开始还困惑为什么有个 `,`

### 1.3.2 操作

**切片**，和 list 类似

## 1.4 dict

### 1.4.1 判断 key/value 是否存在

**key**

```python
dict_tmp.has_key()
```

**value**

```python
# 判断 key 是否存在
'CS' in Major
# 不存在则返回 None
Major.get('CS')
# 不存在则返回 -1
Major.get('CS', -1)
```

### 1.4.2 存放

dict 内部存放的顺序和 key 放入的顺序是没有关系的。

key 必须是不可变对象，比如 string, 整数可以作为 key，但是 list 不能作为 key，因为 list 可变

### 1.4.3 迭代

```python

# 默认迭代 key
d = {'a': 1, 'b': 2, 'c': 3}
for key in d:
    pass
# 迭代 value
for value in d.values():
    pass
# 同时迭代
for k, v in d.items():
    pass
```

## 1.5 set

看起来就是去重的数组

## 1.6 array

### 1.6.1 什么是 array like

https://stackoverflow.com/questions/40378427/numpy-formal-definition-of-array-like-objects  回答 in stackoverflow

> It turns out almost anything is technically an array-like. "Array-like" is more of a statement of how the input will be interpreted than a restriction on what the input can be; if a parameter is documented as array-like, NumPy will try to interpret it as an array.

一个陈述，如何去理解 input，而非一个限制。

> The term "array-like" is used in NumPy, referring to anything that can be passed as first parameter to `numpy.array()` to create an array ().

直接的定义，可以作为 `numpy.array()` 的第一个参数

https://thecleverprogrammer.com/2021/03/19/difference-between-tensors-and-arrays/

The difference between a NumPy array and a tensor is that the tensors are backed by the accelerator memory like GPU and they are immutable, unlike NumPy arrays.

总的来说，**Tensors are like arrays**, both are data structures that are used to store data that can be indexed individually.

**综上，tensor 是否是 array-like?**

**是的**

## 1.X 其他特性

### 1.X.1 generator 

暂时没有用过，先挖个坑

# 2. 数据类型，变量，语法
## 2.1 data type
python 支持的 data type 应该足够满足日常使用了。

<div align = left>
<img src = Img/data_type.png width = 90%>
</div>

complex 在 cutlass 里面也有支持，不过平时没有使用过。tuple 使用也比较少

### 2.1.1 float

可以写成 `1.23e9`

### 2.1.2 string

python 中，用 `''` 和 `""` 有什么区别？

从 stackoverflow 回答来看是完全一样的


## 2.2 类型转换

```python 
x = 25
str_ = str(x)
int_ = int(str_)
```

## 2.3 编码

> UTF-8 编码，把一个Unicode字符根据不同的数字大小编码成1-6个字节，常用的英文字母被编码成1个字节，汉字通常是3个字节，只有很生僻的字符才会被编码成4-6个字节。

看来这里也有哈夫曼的思想

## 2.4 Global Variables

在函数内部创建变量时，默认是局部的

在函数外部定义变量时，默认是全局的，此时无需使用 global 关键字；在函数外使用 global 没有作用。

在函数内读写全局变量时，需要使用 global 关键字

# 3. File 操作

## 3.1 文件读写

read, open, write
```python
f = open("demofile.txt", "r")
print(f.read())

f = open("demofile2.txt", "a")
# a, append; w, overwrite
f.write("Now the file has more content!")
f.close()
```

实际使用
```python
for i, line in enumerate(open(get_file_path() + file)):
    for match in re.finditer(pattern_cycle, line):
        tmp = list(match.group(1))
        tmp = float(''.join(tmp))
        res_data["cycle"] = tmp
```

## 3.2 文件目录操作

```shell
(torch) zdli@GPU74:/nvme/wmhu$ # 工作目录
/nvme/wmhu/shader_docker/cutlass-gpgpu-sim # python 文件所在的目录
```

一些常用的路径

```shell
path_ = os.getcwd() # 获取当前工作目录的绝对路径, /nvme/wmhu
path_ = os.path.abspath(__ file __) # 获取当前文件的绝对路径, /nvme/wmhu/shader_docker/cutlass-gpgpu-sim/data_analyze.py
path_ = os.path.dirname(__file__) # 获得当前文件所在目录的绝对路径, /nvme/wmhu/shader_docker/cutlass-gpgpu-sim
```

# 4. 函数

## 4.1 一些特性

### 4.1.1 别名和引用

Python 中可以把函数名赋值给一个对象

```python
>>> a = abs
>>> a(-1)
1
```

### 4.1.2 pass 占位符

还没想好函数或者条件语句的内容，可以用一个 pass，让代码可以先运行

```python
def nop():
    pass
if agr >= 18:
    pass
```

### 4.1.3 参数检查

参数数量不对时，Python 解释器可以检查；如果是参数 type 错误，Python 解释器可能看不出来。可以用函数 `isinstance()` 来检查，这个看起来有点像 assert()，一个典型的错误和异常处理

```python
def my_abs(x):
    if not isinstance(x, (int, float)):
        raise TypeError('bad operand type')
    if x >= 0:
        return x
    else:
        return -x
```


## 4.2 __init__ and forward()
了解函数的定义，以及调用，比如 __init__

### 4.2.1 __init__

> 个人将其理解为 C 中的构造函数
> 
创建对象时，python 解释器会自动调用它。
### 4.2.2 forward()

使用 pytorch 的训练模型的时候，不需要调用 forward 函数，只需要在实例化一个对象中（`model = ViT(model_name, pretrained=True)`）传入对应的参数，就可以自动调用 forward 函数，下面展示一个例子。来自 https://zhuanlan.zhihu.com/p/357021687

```python
 class Module(nn.Module):
    def __init__(self):
        super().__init__()
        # ......

    def forward(self, x):
        # ......
        return x

data = ......  # 输入数据
# 实例化一个对象
model = Module()
# 前向传播
model(data)
# 而不是使用下面的
# model.forward(data)  
```
### 4.2.2 Why?
为什么有这种等价关系？`model(data)` 等价于 `model.forward(data)`，因为在 class 中使用了 __call__ 函数。更深的就不继续展开了。

## 4.3 unittest

2022-08-01 14:38:19，今天接触到这玩意儿。因为组里面这边在做和 ngp 相关的东西。有同学用 jax 写了一版代码，test.py 文件看起来没有入口，所有函数都在 class 里面。调试的方式就是 `python -m unittest test.py`。所以算是接触到了一个新的东西。

编写单元测试时，我们需要编写一个测试类，从unittest.TestCase继承。

```python
# 如下，继承自 unittest.TestCase
class TestFoward(unittest.TestCase):
    pass
```

## 4.4 pytest

```shell
pip install pytest
```

### 4.4.1 命名规则

+ 测试文件名必须以“test_”开头
+ 测试类以Test开头，并且不能带有 init 方法
+ 测试函数必须以“test_”开头
+ 除了有setup/teardown，还能更自由的定义fixture装载测试用例
+ ...

### 4.4.2 case

来自 https://www.jianshu.com/p/75c27fe23b4e

```python
# test_class.py
class TestClass:
    def test_one(self):
        x = "this"
        assert 'h' in x

    def test_two(self):
        x = "hello"
        assert hasattr(x, 'check')
```

运行

```shell
$ pytest -v test_
```

## 4.5 参数

函数作为另一个函数的参数传入：
### 4.5.1 返回多个参数

其实就是返回一个 tuple，但是在语法上，接收时不需要用括号

```python
import math

def move(x, y, step, angle=0):
    nx = x + step * math.cos(angle)
    ny = y - step * math.sin(angle)
    return nx, ny

x, y = move(100, 100, 60, math.pi / 6)
```
### 4.5.2 默认参数

Python 应该不像 C 语言那样有函数的重构，可以用默认参数；必选参数必须全部放在前面，而全部可选参数/默认参数放在后面

```python
def power(x, n=2):
    s = 1
    while n > 0:
        n = n - 1
        s = s * x
    return s
>>> power(5)
25
>>> power(5, 2)
25
```
### 4.5.3 可变参数

和返回的情况类似，传入的参数是一个 tuple，下面给出语法

```python
# 加上 * 代表传入可变参数
def calc(*numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum

nums = [1, 2, 3]
# 传可变参数，过于繁琐
calc(nums[0], nums[1], nums[2])

# 类似于返回多参数的写法
calc(*nums)
```

### 4.5.4 关键字参数

2022-08-11 15:28:10，这一块就是在理论上比较薄弱的地方，也是为什么之前阅读 Python 代码存在一些阻碍。

```shell
# 除了必选参数 name, age，还可以传入任意个关键字参数 kw
def person(name, age, **kw):
    print('name:', name, 'age:', age, 'other:', kw)
```

### 4.5.5 命名关键字参数

这个感觉在很多 Python 内置库中常用，限制关键字参数的名字。
```python
# * 作为分隔符，* 之后的参数被视为命名关键字参数
def person(name, age, *, city, job):
    print(name, age, city, job)

# 也可以由一个可变参数来分隔
def person(name, age, *args, city, job):
    print(name, age, args, city, job)

>>> person('Jack', 24, city='Beijing', job='Engineer')
Jack 24 Beijing Engineer


```

对于任意函数，都可以通过类似func(*args, **kw)的形式调用它，无论它的参数是如何定义的。

## 4.5 高阶函数

2022-08-11 16:02:01，之前疑惑的一个地方，函数作为另一个函数的参数传入。这个就是高阶函数。
```python
# 高阶函数的例子
def add(x, y, f):
    return f(x) + f(y)
```

### 4.5.1 map, reduce

`map()` 的参数，一个是函数，另一个是可迭代序列，`map()` 会把传入的函数依次作用到序列的每个元素，然后把结果返回到序列中

```python
>>> def f(x):
...     return x * x
...
>>> r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> list(r)
[1, 4, 9, 16, 25, 36, 49, 64, 81]
```
`reduce()` 也就接受函数作为参数，他会把函数作用在一个序列上，并把结果和序列中的下一个元素做累计计算

`reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)`

### 4.5.2 lambda

匿名函数，简化代码的一种方式

# 5. Class

## 5.1 构造函数

```python
class Student(object):

    def __init__(self, name, score):
        self.name = name
        self.score = score

        # 如果要定义为 私有成员，加两个下划线
        self.__name = name
        self.__score = score
>>> bart = Student('Bart Simpson', 59)
>>> bart.name
'Bart Simpson'
>>> bart.score
59
```

## 5.2 继承

继承的语法

```python
class Animal(object):
    def run(self):
        print('Animal is running...')
# 把这个 object 换成需要继承的 class 即可
class Dog(Animal):
    pass

class Cat(Animal):
    pass
```

# 6. module / 模块

## 6.1 获取模块中的元素 / 遍历模块

定义了一个模块 `coda_int.py` 用来装需要绘图的所有 list，一个接一个输入名字过于繁琐，尝试获取模块的中对象来自动遍历。 

### 6.1.1 遍历模块

2022-09-14 11:13:48 搞定。用 `if not key.startswith('__')` 来过滤掉一些自带的方法，剩下的就是 `cola_int` 这个 module 中的 list。 
```python
for key, value in cola_int.__dict__.items():
    if not key.startswith('__'):
        print(value['max'])
```
# Debug

> 用Python开发程序，完全可以一边在文本编辑器里写代码，一边开一个交互式命令窗口，在写代码的过程中，把部分代码粘到命令行去验证，事半功倍！前提是得有个27'的超大显示器！

这倒是这类语言的一个debug 方法


# BUG
```python
'dict_values' object is not subscriptable

tot_stat[model_name + "_" + mode] = sum_stat.values()[len_ - 5:]
```

注意 sum_stat.values() 返回的是 dict_keys 对象而不是 list，不支持 index 索引，可以使用 

```python 
tot_stat[model_name + "_" + mode] = list(sum_stat.values())[len_ - 5:]
```

# Reference 

https://www.liaoxuefeng.com/wiki/1016959663602400/1017024645952992