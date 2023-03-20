---
title: "PyTorch 学习"
date: 2022-07-28T16:05:35+08:00
lastmod: 2022-08-11T16:05:35+08:00
draft: false
author: "Cory"
tags: ["PyTorch"]
categories: ["编程"]
---

## 1.2 Tensor

张量，多维数组。

数据类型需要注意一下

> 关于 dtype，PyTorch 提供了 9 种数据类型，共分为 3 大类：float (16-bit, 32-bit, 64-bit)、integer (unsigned-8-bit ,8-bit, 16-bit, 32-bit, 64-bit)、Boolean。模型参数和数据用的最多的类型是 float-32-bit。label 常用的类型是 integer-64-bit。

# 1. Python 模块

## 1.1 parser 模块

### 1.1.1 parser.add_argument()

在命令行给代码赋值，不需要反复在 python 中修改代码。

```python

parser.add_argument('--file-dir',type=str, required=True,help="Input file directory")

## 实例
parser.add_argument('--dataset', default='cifar10', type=str, 
                    help='dataset name')
parser.add_argument('--dataset_path', default='/state/partition/imagenet-raw-data', type=str, 
                    help='dataset path')
parser.add_argument('--model', default='resnet18', type=str, 
                    help='model name')
parser.add_argument('--train', default=False, action='store_true', 
                    help='train')
```

`action`: `-train` 设置成一个开关，
+ 如果使用了 `python -u -m --train ...`，就会把参数 `--train` 设置为 True
+ `python -u -m ...`，没有这个开关，则参数存储为 False
### 1.1.2 parser.parse_args()


# 2. torch

有很多方便的数学操作，同理，先了解有这个东西，需要用到时看具体的用法。包括 torch.rand(), torch.range(), torch.chunk(), torch.normal(), torch.add()

pytorch 主要分为五大模块
+ dataset
+ model
+ loss funtion
+ optimizer 
+ 迭代训练

## 2.1 nn

`torch.nn` 主要包含 4 个模块

+ nn.Parameter, Tensor 子类，表示可学习的参数，如 weights, bias
+ nn.Modules, 所有模型的基类，用于管理网络的属性
+ nn.functional, 函数具体实现，如 conv, pool, 激活函数
+ nn,init, 网络参数初始化方法
### 2.1.1 nn.Module

class torch.nn.Module 是所有网络的基类（Base class for all neural network modules），每个模型都应该继承这个类，参考lab1的网络模型

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, bias=False)
        self.pool = nn.MaxPool2d(2, 2) # run after each conv (hence the 5x5 FC layer)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  
        x = F.relu(self.fc1(x)) #输入是列向量
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to(device)
```

一般在 model.py 文件中定义 NN model，再举一个 ViT 的例子

```python
class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3, 
        image_size: Optional[int] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

    def forward(self, x):
        ...

```
#### 2.1.1.1 一些子函数
2022-08-06 22:31:40

named_ 系列


**model.named_parameters()**，返回两个变量，比如赋值给 name(e.g. `name` -> `stage_1.0.conv_b.weight`) 和 param (e.g. `param.requires_grad` -> `False`)
### 2.1.2 nn.Layer

**model.named_modules()**，返回所有模块的迭代器。打印的话会输出模型的结构，如同 `print(model)`

**model.named_children**，named_modules 的子集，返回子模块的迭代器

**model.children()**，返回下一级模块的迭代器

> 所以这个只是访问到一级，如果下一级是一个 Sequential，那么就还得继续迭代，这个时候用 .modules() 可能会更好

**model.modules()**，Returns an iterator over all modules in the network

model.named_modules() 会有冗余的返回，这种情况下需要结合一些函数来过滤。


nn 还包含了很多 layer，比如 `nn.Conv2d`, `nn.MaxPool1d`, `nn.ReLU`

### 2.1.3 model 的创建
主要是 2 个要素，构建子模块和拼接子模块，把子模块理解为 layer，构建子模块就是 `__init__`，拼接子模块就是 `forward()`

+ 调用 `model = ViT(model_name, pretrained=True)` 创建模型时，会调用 `__init__()` 方法创建模型的子模块
+ 训练时调用 `outputs = net(inputs)` 时，会进入 `module.py` 的 `call()` 函数中
+ 在 `__call__` 中调用 `result = self.forward(*input, **kwargs)` 函数，进入到模型的 `forward()` 函数中，进行前向传播

#### 2.1.3.1 model() 实例

```python
model = quantize_model(model)
...
outputs = model(inputs)
loss = criterion(outputs, targets)
...
```

在 `outputs = model(inputs)` 语句中会进入到 `class Conv2dQuantizer(nn.Module)` 的 forward 函数，`super(Conv2dQuantizer, self).__init__()` 是继承父类的构造函数 `__init__()`，从而使得 Conv2dQuantizer 中包含了父类

#### 2.1.3.2 model.eval()

作用：不启动 BatchNormalization 和 Dropout，保证 BN 和 Dropout 不发生变化，pytorch 框架会自动把 BN 和 Dropout 固定住，不会取平均值，而是用训练好的值，不然的话，一旦 test 的 batch_size 过小，很容易就会被 BN 层导致生成图片颜色失真极大。

Reference: https://zhuanlan.zhihu.com/p/357075502

#### 2.1.3.3 torch.no_grad()

tensor 有一个参数是 `requires_grad`，如果设置为 True，则反向传播时该 tensor 会自动求导，默认为 False，反向传播时不求导，可以极大地节约显存或者内存。

`with torch.no_grad` 的作用：所有计算得出的 tensor 的 requires_grad 都自动设置为 False

### 2.1.4 CrossEntropyLoss

This criterion combines LogSoftmax and NLLLoss in one single class.

## 2.2 Tensor

`Tensor` is a multi-dimensional matrix containing elements of a single data type

可以用 list 作为参数来构造 tensor 

```python
>>> torch.tensor([[1., -1.], [1., -1.]])
tensor([[ 1.0000, -1.0000],
        [ 1.0000, -1.0000]])
>>> torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
tensor([[ 1,  2,  3],
        [ 4,  5,  6]])
```

## 2.3 autograd

weight 更新依赖于梯度的计算，在 pytorch 中搭建好 forward 计算图，利用 `torch.autograd` 自动求导得到所有 gradient of tensor

## 2.4 data 模块

数据模块可以细分为 4 个部分

+ 数据收集：样本，label
+ 数据划分：train set, valid set, test set
+ 数据读取：pytorch dataloader 模块，dataloader 包括 sampler, dataset
  + sampler: 生成索引(index)
  + dataset: 根据生成的索引(index)读取样本以及标签(label)
+ 数据预处理：对应于 pytorch transforms

### 2.4.1 DataLoader

`torch.utils.data.DataLoader()`, 构建可迭代的数据装载器

```python
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None)
```

+ dataset: torchvision.datasets 类，决定数据从哪里读取，如何读取，以及是否下载，是否训练，给出一个例子
```python
cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
```
+ num_works: 是否多进程读取，指定读取的进程数量
+ shuffle: 每个 epoch 是否乱序
+ sampler: 指定一个 `torch.utils.data.distributed.DistributedSampler` 类型

**其他名词**

+ Epoch: 所有训练样本都已经输入到模型中，称为一个 epoch
+ Iteration: a batch of 样本已经输入到模型中
+ Batchsize: 批大小，决定一个 iteration 有多少样本，也决定了一个 Epoch 有多少个 Iteration

#### 2.4.1.1 NVIDIA.DALI 

2022-07-28 21:24:17 了解到这玩意儿

### 2.4.2 DataSet

`torch.utils.data.Dataset`, 抽象类，所有自定义大的 DataSet 都需要继承该类。

在Dataset 的初始化函数中会调用 `get_img_info()` 方法。

### 2.4.2 torchvision

计算机视觉工具包，有 3 个主要的模块

+ `torchvision.transforms`, 包括常用的图像预处理方法
+ `torchvision.datasets`, 包括常用的 dataset, e.g. MNIST, CIFAR-10, ImageNet
+ `torchvision.models`, 常用的 pre-trained models, e.g. AlexNet, VGG, ResNet, GoogleNet  

data 的数量和分布对模型训练的结果起决定性的作用，需要对 data 进行 pre-process 和数据增强。目的是增加数据的多样性，提高模型的泛化能力。

### 2.4.3 Batch Size

2022-08-09 18:24:56，学习一下梯度，训练，和 batch size 的关系。

通过举例来学习，比如目前使用的网络，batch_size = 128，训练数据行数为 $|x| = 1024$，代表每次网络模型的迭代使用了 128 个样本，128 个样本来自 $x$，可能是无序抽样，也可能是有序抽样。

每个 epoch 包含 1024/128 iterations

> 同一个 epoch，我用第一个 batch 完成了一次前向反向，接下来的第二次 iteration，换了一个 batch，但是 weight 已经更新过了
> 这个就是之前卡住我的点，要理解这一点。训练是为了让 weight 收敛。

#### 2.4.3.1 epoch, bach, iteration

epoch: 把所有训练数据丢进网络的周期

batch_size: 一次迭代的数据量；这个从一些说法中，看起来是图片的张数

iteration: 完成所有训练数据的迭代，所需要的次数。

> batch_size = 128，训练数据行数为 $|x| = 1024$，代表每次网络模型的迭代使用了 128 个样本，128 个样本来自 $x$，可能是无序抽样，也可能是有序抽样。
>每个 epoch 包含 1024/128 iterations

:star: 注意，第一个 epoch 结束之后，weight 是没有 reset 的，也就是说第二个 epoch 仍然接着更新 weight。

> epoch，背诵词典次数多了，就记牢了。当然，也有可能背傻了（过拟合）

## 2.5 模型训练

### 2.5.1 损失函数

Loss Function, 衡量模型输出与真实标签之间的差异，也就是 **一个** 样本的 output 和真实标签（label）的差异

Cost Function, 计算整个样本集的 output 和真实标签（label）的差异  

pytorch 中的损失函数也是继承于 `nn.Module`

### 2.5.2 optimizer

PyTorch 中的优化器是用于管理并更新模型中 **可学习参数的值**，使得模型输出更加接近真实标签。

#### 2.5.2.1 属性

+ defaults: 优化器的超参数，如 weight_decay, momentum
+ state: 参数的缓存，如 momentum 中需要用到前几次的梯度，缓存在这个变量中
+ param_groups: 管理的参数组，是一个 list，其中每个元素是 dict，包括 momentum, lr, weight_decay, params
+ _step_count: 记录更新次数，在学习率调整中使用

#### 2.5.2.2 optimizer 方法

+ zero_grad(): 清空所管理参数的梯度。因为 pytorch 张量的梯度不会自动清零，因此每次反向传播之后都需要清空梯度
+ step(): 执行一步梯度更新
+ add_param_group(): 添加参数组
+ state_dict(): 获取优化器当前状态
+ load_state_dict(): 加载状态信息的 dict

#### 2.5.2.3 learning rate
learning rate, 影响 loss function 收敛的重要因素，控制了梯度下降更新的步伐

## 2.6 Regularization 正则化

正则化是一种减少方差的策略

### 2.6.1 weight decay

weight decay 是优化器中的一个参数，在执行 `optim_wdecay_step()` 时，会计算 weight decay 后的梯度

### 2.6.2 Dropout

一种抑制过拟合的方法。理解为放缩数据

### 2.6.3 Normalization

Batch Normalization, 经过 normalization 后的数据服从 $N(0, 1)$ 分布，有如下优点

+ 可以使用更大的 lr，加速模型收敛
+ 可以不用精心设计 weight 初始化
+ 可以不用 dropout 或者较小的 dropout
+ 可以不用 L2 或者较小的 weight decay
+ 可以不用 LRN (Local Response Normalization)

## 2.7 Model 相关的操作

### 2.7.1 torch.save
```python
torch.save(obj, f, pickle_module, pickle_protocol=2, _use_new_zipfile_serialization=False)
```

obj 是保存的对象，f 是输出路径。还有 2 种方式

+ 保存整个 Module, `torch.savev(net, path)` 这种方法比较耗时，保存的文件比较大
+ 只保存模型的参数，`torch.savev(state_sict, path)`，推荐，保存的文件比较小

### 2.7.2 torch.load

对应于 save

### 2.7.3 Fine-tuning
一种迁移学习的方法，比如在人脸识别应用中，ImageNet 作为 source domain，人脸数据作为 target domain。通常 source domain 比 target domain 大很多，可以利用 ImageNet 训练好的网络应用到人脸识别中。

**理解**
对于一个模型，可以分为流程在前面的 feature extractor （conv 层） 和后面的 classifier。fine-tune 通常不改变 feature extractor 的 weight，也就是冻结 conv 层；改变最后一个 fc layer 的输出来适应目标任务，训练后面 classifier 的 weight。

通常 target domain 的数据比较小，不足以训练全部参数，容易导致过拟合，因此不改变 feature extractor 的 weight。

**Step**

+ 获取 pre-trained model 参数
+ `load_state_dict()` 把参数加载到模型中
+ 修改输出层
+ 固定 feature extractor 的参数，通常有 2 种做法
  + 固定 conv 层的预训练参数。可以设置 `requires_grad = False` 或者 `lr = 0`
  + 通过 `params_group` 给 feature extractor 设置一个较小的 lr
  
# 3. pytorch 分布式训练

```python
dist.init_process_group(backend='nccl')
# backend是后台利用nccl进行通信
```
调试时报错，如何在调试分布式训练的模型

## 3.1 debug 分布式训练的模型

修改 python `launch.json` 文件，把 program 换成 torch.distribution 的 launch.py

```python
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "/nvme/wmhu/anaconda3/envs/ant/lib/python3.8/site-packages/torch/distributed/launch.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "args": [
                "--nproc_per_node=1",
                "/nvme/wmhu/work/ANT/ImageNet/main.py",
                "--dataset=imagenet",
                "--model=vit_b_16",
                "--dataset_path=/nvme/imagenet",
                "--epoch=4",
                "--mode=int",
                "--wbit=4",
                "--abit=4"
            ],
            "env":{"CUDA_VISIBLE_DIVICES":"0"},
        }
    ]
}
```

注意 `nproc_per_node` 是 Python 自带的参数，因此可以写到里面，对于 `--dataset=imagenet` 会有报错。

`--dataset=imagenet` 要放到 `main.py` 的后面，这个方法相当于用调整参数的形式来达到目的。不太有通用性，相当于专门改了一个 `launch.json` 文件。

# 4. Hook

这个功能被广泛用于可视化神经网络中间层的 feature、gradient，从而诊断神经网络中可能出现的问题，分析网络有效性。

视频：https://www.youtube.com/watch?v=syLFCVYua6Q

pytorch 计算图似乎只会保留叶子节点的梯度，舍弃中间的梯度

> 简而言之，register_hook的作用是，反向传播时，除了完成原有的反传，额外多完成一些任务。你可以定义一个中间变量的hook，将它的grad值打印出来，当然你也可以定义一个全局列表，将每次的grad值添加到里面去。

什么是中间变量：有的博客里有提到，似乎是没有直接指定数值，而是通过计算得到的变量。比如下面的 z 就是中间变量。z 的梯度是不会保存的
```python 
x = torch.Tensor([0, 1, 2, 3]).requires_grad_()
y = torch.Tensor([4, 5, 6, 7]).requires_grad_()
z = x + y       # 中间变量
```

```python
output = model(input)
# 此时会做几件事，一个是调用 forward 方法计算结果，一个是判断有没有注册 forward_hook，有的话就将 forward 的输入及结果作为 hook 的实参

```

## 4.1 register_hook

`z.register_hook(hook_fn)`，这个 hook_fn 是一个用户自定义函数，返回 Tensor （如果需要对 grad 进行修改）或者 None（用于直接打印，不修改），所以直接用 lambda 函数即可，`z.register_hook(lambda grad: print(grad))`

> 个人理解下来，register_hook 可以实现保留中间变量梯度的功能，而且不像 `retain_grad` 那样会带来很大的开销

### 4.1.1 register_forward_hook

**作用**：获取中间层的 feature map

通常，pytorch 只提供了网络整体的输入和输出，对于夹在网络中间的模块，很难获得他的输入输出。除非设计网络时，在 forward 函数的返回值中包含中间 module 的输出。总而言之别的方法都比较麻烦，pytorch 设计好了 register_forward_hook 和 register_backward_hook。

相比针对 tensor 的 register_hook，这个 forward hook 没有返回值，也就是不能改变输入，只能打印。

代码实例，讲的非常清晰，来自[博客](https://cloud.tencent.com/developer/article/1475430)
```python
Class Model(nn.Module):
    def __init__(self):
        ...
    def forward(self, x):
        ...
# 全局变量，用于存储中间层的 feature
total_feature_out = []
total_feature_in = []

model = Model()

# 定义 forward hook function
def hook_fn_forward(module, input, output):
    print(module) # 用于区分模块
    print('input', input) # 首先打印出来
    print('output', output)
    total_feature_out.append(output) # 然后分别存入全局 list 中
    total_feature_in.append(input)
# 给每个 module 都装上 hook
for name, module in model.named_children():
    module.register_forward_hook(hook_fn_forward)

# 前向传播和回传
x = torch.Tensor([[1.0, 1.0, 1.0]]).requires_grad_() 
o = model(x)
o.backward()

print('==========Saved inputs and outputs==========')
for idx in range(len(total_feature_in)):
    print('input: ', total_feature_out[idx])
    print('output: ', total_feature_out[idx])
```

### 4.1.2 register_backward_hook()

#### 4.1.2.1 使用方法和示例

**作用**：用于获取梯度

**使用**：`module.register_backward_hook(hook_fn)`, `hook_fn(module, grad_input, grad_output) -> Tensor or None`

如果有多个输入或者输出，grad_input, grad_output 可以是 tuple 类型。比如对于线性模块，grad_input 是一个三元组，分别是 $g_{bias}$, $g_x$, $g_W$，对 bias 的导数，对 x 的导数以及对 weight 的导数。

直接看代码，这个代码是可以直接跑的，不得不说写的确实很好。

```python
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.initialize()

    def initialize(self):
        with torch.no_grad():
            self.fc1.weight = torch.nn.Parameter(
                torch.Tensor([[1., 2., 3.],
                              [-4., -5., -6.],
                              [7., 8., 9.],
                              [-10., -11., -12.]]))

            self.fc1.bias = torch.nn.Parameter(torch.Tensor([1.0, 2.0, 3.0, 4.0]))
            self.fc2.weight = torch.nn.Parameter(torch.Tensor([[1.0, 2.0, 3.0, 4.0]]))
            self.fc2.bias = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        o = self.fc1(x)
        o = self.relu1(o)
        o = self.fc2(o)
        return o
# 全局变量，用于存储中间层的 feature
total_grad_out = []
total_grad_in = []

model = Model()

# 定义 forward hook function
def hook_fn_backward(module, grad_input, grad_output):
    print(module) # 用于区分模块
    # 为了符合反向传播顺序，先打印 grad_output
    print('grad_output', grad_output)
    print('grad_input', grad_input) 
    total_grad_out.append(grad_output) 
    total_grad_in.append(grad_input)
# 给每个 module 都装上 hook
for name, module in model.named_children():
    module.register_backward_hook(hook_fn_backward)

# 前向传播和回传
# 这里的 requires_grad 很重要，如果不加，backward hook
# 执行到第一层，对 x 的导数将为 None，某英文博客作者这里疏忽了
# 此外再强调一遍 x 的维度，一定不能写成 torch.Tensor([1.0, 1.0, 1.0]).requires_grad_()
# 否则 backward hook 会出问题。
x = torch.Tensor([[1.0, 1.0, 1.0]]).requires_grad_()
o = model(x)
o.backward()

print('==========Saved inputs and outputs==========')
for idx in range(len(total_grad_in)):
    print('input: ', total_grad_in[idx])
    print('output: ', total_grad_out[idx])
```

**注意**，作者提到“register_backward_hook只能操作简单模块，而不能操作包含多个子模块的复杂模块。如果对复杂模块用了 backward hook，那么我们只能得到该模块最后一次简单操作的梯度信息。”不太确定什么是简单模块，不太确定诸如 resnet18 这样的网络是不是简单模块。

2022-08-06 23:14:22，这个地方应该是想说，用 for loop 遍历 model.named_children() 是有必要的，否则直接 `model = Model()model.register_backward_hook(hook_fn_backward)` 会有问题。

#### 4.1.2.2 注意事项

**形状**

+ 在卷积层中，weight 的梯度和 weight 的形状相同

+ 在全连接层中，weight 的梯度的形状是 weight 形状的转秩（观察上文中代码的输出可以验证）

**grad_input tuple 中各梯度的顺序**

+ 在卷积层中，bias 的梯度位于tuple 的末尾：grad_input = (对feature的导数，对权重 W 的导数，对 bias 的导数)

+ 在全连接层中，bias 的梯度位于 tuple 的开头：grad_input=(对 bias 的导数，对 feature 的导数，对 W 的导数)

**当 batchsize > 1 时，对 bias 的梯度处理不同**

+ 在卷积层，对 bias 的梯度为整个 batch 的数据在 bias 上的梯度之和：grad_input = (对feature的导数，对权重 W 的导数，对 bias 的导数)

+ 在全连接层，对 bias 的梯度是分开的，bach 中每条数据，对应一个 bias 的梯度：grad_input = ((data1 对 bias 的导数，data2 对 bias 的导数 ...)，对 feature 的导数，对 W 的导数)

# Misc

@classmethod 用法

想给初始类再新添功能，不需要改初始类，只要在下一个类内部新写一个方法，方法用@classmethod装饰一下即可。、

```python
@classmethod
def convert_sync_batchnorm(cls, module, process_group=None):
```

# 问题

+ Debug 的时候怎么看 tensor 变量的参数？

# Reference 

https://zhuanlan.zhihu.com/p/265394674

https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/  pytorch 中文文档

https://cloud.tencent.com/developer/article/1475430#