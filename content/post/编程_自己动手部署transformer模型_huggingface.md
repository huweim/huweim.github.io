---
title: "自己动手部署transformer模型 by huggingface"
date: 2022-10-23T11:00:17+08:00
lastmod: 2022-11-26T16:17:41+08:00
draft: false
tags: ["huggingface"]
categories: ["编程"]
---

# 0. 前言

这部分内容还是很重要的，预计会设计常见的 pytorch 模型部署方法，理解框架中，每个模块在做什么。另外，这也是工程上必备的技能。

## 0.1 模型及下载地址

| Model                                                   | Repo                                                         | Paper                                              |
| ------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------- |
| ResNet (18: 12M; 50: 26M; 152: 60M)                     | https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py |                                                    |
| BERT (110 / 330M)                                       | https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT | https://arxiv.org/abs/1810.04805                   |
| GPT-2 (1.5B)                                            | https://github.com/openai/gpt-2                              |                                                    |
| OPT (125 / 350M; 1.3 / 2.7 / 6.7 / 13 / 30 / 66 / 175B) | https://github.com/facebookresearch/metaseq                  | https://arxiv.org/pdf/2205.01068.pdf               |
| BLOOM (560M; 1.1 / 1.7 / 3 / 7.1 / 176B)                | https://huggingface.co/docs/transformers/model_doc/bloom     |                                                    |
| T5 (60 / 220 / 770M; 3 / 11B)                           | https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints | https://jmlr.org/papers/volume21/20-074/20-074.pdf |

## 0.2 模型下载方法

2023-07-08 20:10:12，重新回顾 22 年关于大模型研究的工作。从 chatgpt 爆火之后，大模型应用的框架变得火热，语言模型的社区也变得火热起来。准备在 ant_ext 工作中加入一些新的模型的评估，但是期智连接 huggingface 的网络老是抽风，导致试图从 huggingface 下载模型时出现错误。现在总结一些其他的下载方法。

Ref: https://zhuanlan.zhihu.com/p/475260268

### 0.2.1 git lfs

这个方法会下载所有框架的模型文件，flax_model.msgpack、tf_model.h5和pytorch_model.bin，可以看到有 3 种框架的模型文件，比较冗余。

```shell
git lfs install
git clone https://huggingface.co/bert-base-chinese
```

### 0.2.2 hungging face hub
这种方法其实仍然是通过访问 huggingface 网站来下载

```shell
python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="facebook/opt-13b")

# 通过参数可以指定下载的文件，屏蔽不需要的文件即可
>>> snapshot_download(repo_id="facebook/opt-13b", ignore_patterns=["*.h5", "*.ot", "*.msgpack"], local_dir="/localdata_ssd/model", local_dir_use_symlinks=False)
```

# 1. Hugging Face

简介：Hugging face 是一家总部位于纽约的聊天机器人初创服务商，开发的应用在青少年中颇受欢迎，相比于其他公司，Hugging Face更加注重产品带来的情感以及环境因素。但更令它广为人知的是Hugging Face专注于NLP技术，拥有大型的开源社区。尤其是在github上开源的自然语言处理，预训练模型库 Transformers，已被下载超过一百万次，github上超过24000个star。Transformers 提供了NLP领域大量state-of-art的 预训练语言模型结构的模型和调用框架。

BERT 模型中，一个 token-wise 就是 768 维的向量。

> 总的来说，提供了各种易用 transformer 模型，并且有很好的社区和文档。

## 1.1 简介

### 1.1.1 用途

理解了一下，感觉是提供预训练好的模型，自己可以用来解决一些任务。

### 1.1.2 重要概念

#### 1.1.2.1 Pipeline 

pipeline 函数：包含 pro-processing, model, post-processing

如果只是想使用训练好的模型的功能，这是最简单的方法。pipeline 函数中应该完成了一套 workflow。下列代码下载了大概 300MB 的预训练模型，给出了对我输入的句子的情感分析。

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
print(classifier("You like a pig"))

output: [{'label': 'NEGATIVE', 'score': 0.9961085915565491}]
```

#### 1.1.2.2 AutoClass

`AutoModelForSequenceClassification` 以及 `AutoTokenizer` 用于支持 `pipeline()`。AutoClass 是一种快捷方式，可以自动从预训练模型的名称或路径中检索模型架构。 您只需要为您的任务选择适当的 AutoClass 及其关联的预处理类。

> 在自己使用时，选择 GPT2 模型，需要 `from transformers import GPT2Model`，如果要引入其他模型也是类似的做法。`from transformers import AutoModel`，应该可以根据模型名称自己去检索下载。

2022-10-25 21:13:01，当我有可能通过参数载入多种模型时，我意识到了 AutoClass 的作用。

#### 1.1.2.3 Custom Model

主要途径是获取模型的 config，然后手动修改即可。

```python
from transformers import AutoConfig
my_config = AutoConfig.from_pretrained("distilbert-base-uncased", n_heads=12)
my_model = AutoModel.from_config(my_config)
```

#### 1.1.2.4 Trainer - a PyTorch optimized training loop

目前的很多框架中都有一套训练的 loop，hugging face 也提供了这套流程。设置好几个必备的参数和输入即可。

1. A PreTrainedModel or a `torch.nn.Module`；自己定制一个模型应该也是可以的。

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

2. `TrainingArguments`：超参数，比如 learning rate, batch size, epoch

```python
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="path/to/save/folder/",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
)
```

3. 设置预处理过程：比如 tokenizer, feature extractor, processor

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

4. 定义预处理的训练数据集和测试数据集；不过这段代码之前得先把 dataset load 进来吧。

```python
train_dataset = dataset["train"]  # doctest: +SKIP
eval_dataset = dataset["eval"]  # doctest: +SKIP
```

5. `DataCollator()` to create a batch of examples from your dataset

```python
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()
```

> 2022-10-24 09:43:20，这个 example 比较陌生

6. Put it all together

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)  # doctest: +SKIP

# start training
trainer.train()
```
#### 1.1.2.5 预处理

对于 NLP 模型，一般就是用 tokenizer 来做。经过 tokenizer 的字典有三个重要的 items

+ input_ids, the indices corresponding to each token in the sentence.
+ attention_mask, indicates whether a token should be attended to or not.
+ token_type_ids, identifies which sequence a token belongs to when there is more than one sequence.

**padding**

每个 sentence 长度不一致，用 padding 来 tensor 对齐。

```python
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True)
print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0], 
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

**Truncation**

也是一种对齐的方式

### 1.1.3 模型 Model 

支持的模型很多，比较热门的预训练模型基本都有。学术界的很多模型也有，比如 I-BERT，RoBERTa

## 1.2 API

### 1.2.1 from_pretrained

功能：`from_pretrained` 提供了模型类别判断，模型文件列表映射，模型文件下载及缓存，网络下载稳定性容错等功能。

```python
# Download from huggingface.co and cache
model = BertModel.from_pretrained("bert-base-uncased")
model.save_pretrained('./local_model_directory/')

# Load from local
model = BertModel.from_pretrained('./local_model_directory/')
```

实例：

```python
if(os.path.exists('./model/'+args.gpt2_model+'pytorch_model.bin')):
    model.GPT2ForSequenceClassification.from_pretrained('./model/'+args.gpt2_model)
else:
    model = GPT2ForSequenceClassification.from_pretrained(args.gpt2_model)
    num_labels = len(model.config.id2label)
    model = GPT2ForSequenceClassification.from_pretrained(args.gpt2_model, num_labels=num_labels)
    model.save_pretrained('./model/'+args.gpt2_model)
```

## 1.3 Fine-tune a pretrained model

可以用 1.1.2.4 中的 `Trainer()` 来实现微调和训练，也可以在 pytorch 框架中实现。下面是步骤。

### 1.3.1 Prepare a dataset

比如，从 `Yelp Reviews` 中 load 数据，然后进行预处理。下面的代码用到了 `map()` 函数，写法比较陌生，不过这个过程是通用的。

```python
# load
from datasets import load_dataset
dataset = load_dataset("yelp_review_full")
dataset["train"][100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high....'}

# tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

### 1.3.2 Train

用 `Trainer()` 训练在 1.1.2.4 中讲过了，讲一下用 native pytorch 训练，**Train in native PyTorch**

#### 1.3.2.1 DataLoader

为训练数据集和测试数据集创建一个 `DataLoader`，以便可以遍历数据的 batch。给出 BERT 框架中的示例代码。变量 `batch` 中就是一个 batch，经过 tokenizer 之后的数据。

```python
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
train_features = get_train_features(
    args.data_dir,
    args.gpt2_model,
    args.max_seq_length,
    args.do_lower_case,
    args.local_rank,
    args.train_batch_size,
    args.gradient_accumulation_steps,
    args.num_train_epochs,
    tokenizer,
    processor,
)
train_data = gen_tensor_dataset(train_features)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
        )
for e in range(args.num_train_epochs):
    ...
    for step, batch in enumerate(train_dataloader):
        ...
```
#### 1.3.2.2 Optimizer and learning rate scheduler

优化器根据模型来定义，要传入 learning rate 吗，代码示例：

```python
model, optimizer, scheduler = init_optimizer_and_amp(
    model,
    args.learning_rate,
    args.loss_scale,
    args.warmup_proportion,
    num_train_optimization_steps,
    args.fp16,
)
def init_optimizer_and_amp(model, learning_rate, loss_scale, warmup_proportion,
                           num_train_optimization_steps, use_fp16):
    ...
    optimizer, scheduler = None, None
    logger.info("using fp32")
    if num_train_optimization_steps is not None:
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            warmup=warmup_proportion,
            t_total=num_train_optimization_steps,
        )
    return model, optimizer, scheduler
```

#### 1.3.2.3 Training Loop

这个无需多说。设置好 epoch，每个 epoch 会过完训练数据集中的所有数据。（当然，也有随机抽样的形式）

```python
model.train()
for e in range(args.num_train_epochs):
    ...
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        # outputs = model(**batch)
        outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

#### 1.3.2.4 Evaluate

```python
model.eval()
for i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(eval_dataloader):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)
    with torch.no_grad():
        # label_ids = label_ids.to(torch.float)
        outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
        logits = outputs.logits
    ...
```
## 1.4 Accelerate

Easily rain and use PyTorch models with multi-GPU, TPU, mixed-precision

### 1.4.0 术语

关于分布式的一些概念和术语。

参考：

https://zhuanlan.zhihu.com/p/544273093

https://www.zhihu.com/question/453920336/answer/1828326535

+ Scatter: 把数据分发到其他线程 or 进程
+ Gather: 把数据从其他进程 or 线程中收集到一个主进程中
+ Reduce: 计算在各个进程 or 线程中进行，然后归约到一个主进程中
+ All-Reduce: 同 Reduce 一样，计算再各个进程 or 线程中进行，但是每个节点都一起归约，保持每个节点的结果一致
+ Broadcast: 把数据复制到各个进程 or 线程
+ All-Gather: 与 Gather 不同，每个进程 or 线程中完成数据的手机，保证每个节点的结果一致
+ CLI (Command Line Interface)：一种通过命令行来交互的工具或者应用，比如 mkdir, cd, scp, npm 等
+ Rank: 表示进程的编号/序号。（自己观察到一个 GPU 似乎对应一个 rank，不过 rank 和 GPU 没有严格的对应关系。如果是多进程共享 GPU，那么一个 GPU 可以为多个 rank 服务）
+ Node: 物理节点，可以是一台机器或者一个容器，节点内可以有多个 GPU
+ Rank 和 Local Rank: rank是指在整个分布式任务中进程的序号；local_rank是指在一个node上进程的相对序号，local_rank在node之间相互独立。
+ nnodes、node_rank 与 nproc_per_node： nnodes是指物理节点数量，node_rank是物理节点的序号；nproc_per_node是指每个物理节点上面进程的数量。
+ word size ： 全局（一个分布式任务）中，rank的数量。
+ csrc: 应该是 C source code
+ NVIDIA Megatron-LM: 针对Transformer类的模型提供半自动的分布式部署。
+ DeepSpeed: 微软提供的一个开源深度学习训练优化库。基于英伟达 Megatron-LM 进行了张量切分式的模型并行。

同步 SGD：每个 Worker 都是同步计算一个批量。


### 1.4.1 分布式概念 Distributed

2022-10-24 10:43:08，日后填坑。

2022-11-01 09:04:30，开始学习。

`nn.parallel.scatter(data, devices)`，可以将一组数据 split 到多个 devices 上


#### 1.4.1.1 DataParallel


`nn.DataParallel(net, devices_ids=devices)`，DP 模型。单进程，多线程，适于一台机器的情况。

做法是将 model 复制到不同的 GPU 上，各个 GPU 计算不同 batch 的数据。

#### 1.4.1.2 DistributedDataParallel

`torch.nn.parallel.DistributedDataParallel`，DDP 模型（PyTorch 官网更推荐）。

需要启动 `init_process_group`

如果想在一个有 N 个 GPU 的设备上面使用 DistributedDataParallel，则需要 spawn up N 个进程，每个进程对应0-N-1 的一个 GPU。这可以通过下面的语句实现

```python
torch.cuda.set_device(i)

# i from 0-N-1，每个进程中都需要：
torch.distributed.init_process_group(
    backend='nccl', world_size=N, init_method='...'
)
model = DistributedDataParallel(model, device_ids=[i], output_device=i)

# 实践
if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of
    # sychronizing nodes/GPUs.
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl')
```

多进程，适于多台机器的情况，可以结合模型并行的方法。

使用时建议

+ 使用一个 big dataset
+ 好的 CPU-GPU 和机器-机器带宽
+ 高效的 data load 以及 preprocess
+ 模型需要有好的计算（FLOP）通讯（model size）比
  + Inception > ResNet > AlexNet
+ 使用足够大的 batch size 来获取好的系统性能
+ 使用高效的优化算法对应大批量大小

### 1.4.2 Hugging face 文档

#### 1.4.2.1 步骤

+ import Accelerator 并且实例化

```python
from accelerator import Accelerator

accelerator = Accelerator()
```

+ 消去 model 以及 input data 的 `.to(device)` 或者 `.cuda()` 操作。`accelerator` 会帮你完成这件事
+ 把相关的对象（optimizer, model, train_dataloader, lr_scheduler）传递给 `prepare()` 方法

```python
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)
```

> 每个 GPU 会加载 training dataset 中的不同部分（batch），这看起来像是数据并行。

+ Replace the line `loss.backward()` by `accelerator.backward(loss)`.

#### 1.4.2.2 Distributed evaluation

如果需要分布式评估，同理，把 validation dataloader 发送给 `prepare()` 方法。

```python
validation_dataloader = accelerator.prepare(validation_dataloader)
```

这样的话每个 devices 只能看见 evaluation data 中的一部分，所以最后需要把结果 group 到一起，通过 `gather_for_metrics()` 方法。

```python
for inputs, targets in validation_dataloader:
    predictions = model(inputs)
    # Gather all predictions and targets
    all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
    # Example of use with a *Datasets.Metric*
    metric.add_batch(all_predictions, all_targets)
```

#### 1.4.2.3 Launch

`accelerate` 命令整合了在不同平台上发射脚本的命令，你无需自己记住所有命令。如果你自己熟悉 PyTorch 提供的发射脚本，也可以不使用 `accelerate`。

```shell
# 单 GPU 执行
CUDA_VISIBLE_DEVICES="0" accelerate launch {script_name.py} --arg1 --arg2 ...

# 默认参数，使用所有的 GPU，不启动混合精度
accelerate launch --multi_gpu {script_name.py} {--arg1} {--arg2} ...

# 当然，也可以指定参数
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 {script_name.py} {--arg1} {--arg2} ...

# 查看所有参数
accelerate launch -h
```

**accelerate config**

当然，更好的方式是配置一个 config 文件。使用了 `accelerate config` 之后，会在 `~/.cache/huggingface/accelerate` 目录下生成 `default_config.yaml` 文件。也可以自己定制 config 文件然后读取。

```shell
accelerate launch --config_file ~/.cache/huggingface/accelerate {script_name.py} {--arg1} {--arg2} ...
```

## 1.5 Dataset

### 1.5.1 load from hugging face

hugging face 下载文件存储的位置：`~/.cache/huggingface`

找到 dataset 之后如何下载：

```python
# 搜索到数据集 oscar，里面有很多 subset，比如：unshuffled_deduplicated_en, unshuffled_deduplicated_br

# 第一个参数是数据集的名称，第二个参数用于索引子集，一般还有 split 参数：train, test, validation 

from datasets import load_dataset
dataset = load_dataset(path="oscar", name="unshuffled_deduplicated_en")
# 420G，赶紧中断
dataset.save_to_disk("./glue/")

# 尝试下载 wikitext
dataset = load_dataset(path="wikitext", name="wikitext-103-v1", split="train")
# 2022-10-24 14:07:31，是可以的
print(dataset[:3])
{'text': ['', ' = Valkyria Chronicles III = \n', '']}

# 保存
dataset.save_to_disk("./")
```

### 1.5.2 数据集格式
1.5.1 的例子中，默认保存为 `dataset.arrow` 加上 `.json` 文件。

现在我们想将其保存为 `.csv`

### 1.5.3 split 作用

下面是带有 feature 的 dataset 一个示例

```python
from datasets import load_dataset
# 如果用了 split 
dataset = load_dataset("rotten_tomatoes", split="train")
print(dataset)
Dataset({
    features: ['text', 'label'],
    num_rows: 8530
})
# 如果不用 split，
dataset = load_dataset("rotten_tomatoes")
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 8530
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 1066
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 1066
    })
})
```

## 1.6 Github 文档

所有任务的 example 存放在 https://github.com/huggingface/transformers/tree/main/examples/pytorch 里面。

# 2. 知识补充

补充一些相关的，之前不熟悉的概念。

## 2.2 Data Set

### 2.2.1 GLUE

数据集是之前屏蔽的一个部分，但是如果想了解模型是在做什么，并且要动手做实验，必须了解数据集。

GLUE 是学术界非常常用的数据集。

| 数据集 | 类型           | 描述                                                         |
| ------ | -------------- | ------------------------------------------------------------ |
| MNLI   | 句子对分类任务 | 给定一个前提 (Premise) ，根据这个前提去推断假设 (Hypothesis) 与前提的关系。该任务的关系分为三种，蕴含关系 (Entailment)、矛盾关系 (Contradiction) 以及中立关系 (Neutral)。 |
| QQP    | 句子对分类任务 | 判断 Quora 上的两个问题句是否表示的是一样的意思              |
| SST-2  | 分类任务       | 情感分析                                                     |
| CoLA   | 单句分类任务   | 句子语义判断，是否是可接受的（Acceptable）                   |
| QNLI   |                | 用于判断文本是否包含问题的答案                               |
| STS-B  |                | 预测两个句子的相似性，包括5个级别。                          |
| MRPC   | 句子对分类任务 | 也是判断两个句子是否是等价的。                               |
| RTE    | 分类           | 类似于MNLI，但是只是对蕴含关系的二分类判断，而且数据集更小。 |
| SWAG   |                | 从四个句子中选择为可能为前句下文的那个                       |


# 3. 编程实例

## 3.1 通用知识

### 3.1.1 主要的组件

+ Configuration: 编程模型中的参数或是变量。用于配置词表大小，隐藏层维数，Dropout rate 等等。给出一个 BERT_base 配置的示例。

```shell
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

+ Tokenizer: 每个模型都有对应的分词器。存储 token 到 index 的映射。
+ Model: 实现模型的计算图和编码过程，实现前向传播。对于 output 层，不同的模型有不一样的封装。比如今天（2022-10-23）debug 观察的 `BaseModelOutputWithPoolingAndCrossAttentions`，`类型：SequenceClassifierOutput` 等等。

### 3.1.2 任务分类

语言生成任务，对应模型：GPT，GPT-2，XLNet

语言理解任务，对应模型：BERT，RoBERTa，XLM

## 3.2 在 GPT-2 上运行 wikitext

README：

```shell
pip install transformers
ln -s ../BERT/glue/ ./glue

```
### 3.0.1 坑1
**坑，和 BERT 区分**

BERT forward 和 GPT-2 forward 不一样。之前实现的 `model(inputs_id, token_type_ids, attention_mask)` 按位置传参，但是 GPT-2 forward 第二个参数是 `past_key_values`，所以报错，按关键字传参即可解决这个问题。

他们经过前向之后，output 也不一样。

### 3.0.2 大坑2

2022-10-26 22:31:46，复现 GPT 的结果出了大问题，好在今晚解决了。问题来源于框架中会自动启动分布式 train/eval，但是没有很好地支持分布式（单 GPU 运行 30s，4 GPU 运行 6min），而且 loss 的处理也有问题，导致最终的 metric - PPL 出了问题。用单 GPU 可以解决这个问题。

### 3.0.3 坑3

2022-10-28 16:03:02，框架K 是不支持分布式的，这导致了 local_rank 索引不到，因此执行 evaluate()，进入 quant 模块时报错。

### 3.2.1 Prepare Dataset

读取 datasets 并保存到本地

```python
from datasets import load_dataset
test_dataset = load_dataset("json", data_files="test.json", split="train")
test_dataset.save_to_disk("test.hf")
```

### 3.2.2 LM 预处理

针对生成类模型的预处理。如何从原始的数据集生成 tokenizer 之后的数据。

```python
row_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
# tokenizer
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
    remove_columns=column_names,
    load_from_cache_file=not data_args.overwrite_cache,
    desc="Running tokenizer on dataset",
)
with training_args.main_process_first(desc="grouping texts together"):
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )
# 计算 token 长度
for split, data in lm_datasets.items():
total_eval_tokens = 0        
for chunk in data['labels']:
    total_eval_tokens += len([x for x in chunk[1:] if x != padding_index])
logger.info(f'[{split}] Total eval tokens: {total_eval_tokens}')

# 得到 eval dataset 以及 train dataset
if training_args.do_train:
    if "train" not in tokenized_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = lm_datasets["train"]
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

if training_args.do_eval:
    if "validation" not in tokenized_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = lm_datasets[data_args.eval_subset]
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

# 这个时候 eval_dataset size (480, 3)，3 个维度分别是 input_ids, attention_mask, labels
```

## 3.3 在 T5 上运行 WMT

```shell
MODEL=t5-small

CUDA_VISIBLE_DEVICES=0 python -u run_translation.py  \
  --model_name_or_path ${MODEL} \
  --dataset_name wmt16 --dataset_config_name ro-en \
  --per_device_eval_batch_size=4 \
  --output_dir checkpoints-translation/${MODEL} \
  --source_lang en --target_lang ro \
  --do_eval \
  --predict_with_generate \
  --source_prefix "translate English to Romanian: "
```


# 4. 编程 flow

利用 debug 以及 hugging face 文档，了解整个流程。在 Section 1 介绍 hugging face 指导文档，这部分结合实际编程来看。

## 4.1 数据读取及预处理

### 4.2.1 数据读取

通过参数 `--data_dir`，把本地的数据加载进来，应该包含 `.tsv` 文件。hugging face 的预训练模型可以下载到本地并保存，那么 Dataset 应该也可以。

### 4.2.2 DataLoader

经过处理后的 `train_features`，长度是 8551 examples，装的是经过 tokenizer 后的数据。

原始的 `train_example`，从 `./glue/CoLA/train.tsv` 读入，长度是 8551 examples，每个 example 装的是：

+ `guid`:`train-1` or `train=2`...
+ `label`:`1` or `0`
+ `text_a`:`Our friends won't buy this analysis...`
+ `text_b`:None

`train_data` 就是通过 `gen_tensor_dataset()` 把 `train_features` 转换为 tensor；经过 `DataLoader(train_data, sampler, batch_size)` 方法之后，得到 `train_dataloader`，`sampler` 就是 8551。

在一个 epoch 中，需要遍历一遍整个 train_dataloader，一个 iteration 处理其中一个 batch，train_loader 遍历完就是一个 epoch。

> 记得数据需要转移到 device


### 4.2.X Tokenizer

`input_ids` 存放的就是 tokenizer 后的数据本身，是一个 (64, 128) 的 tensor。`mask`, `segment_id` 也是 (64, 128)，label 是 (64)，这应该就是最后的分类，因为 batch size 是 64，所以有 64 个输出。

对于 hugging face BERT，经过 tokenizer 打印出来就是一整个 mapping，也就是对应 `run_glue.py` 中遍历 `train_dataloader` 的变量 `batch`。不过 `batch` 的类型是 list，而下面的 `inputs` 是 dict。而 hugging face BERT 的 `output` 也是一个 `BaseModelOutputWithPoolingAndCrossAttentions` 类型。

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
print(inputs)
{'input_ids': tensor([[  101,  7592,  1010,  2026,  3899,  2003, 10140,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])}

outputs = model(**inputs)
# output 输出包含 last_hidden_state, pooler_output，其他属性 hidden_states, past_key_values, attentions, cross_attentions 为 None
```

对于 hugging face BertForSequenceClassification，

+ `output` 类型：SequenceClassifierOutput(loss=None, logits=tensor([[-2.2095,  2.5760]]), hidden_states=None, attentions=None)

```python

```

## 4.3 计算 loss

hugging face 有一个普通 trainer 的实例，关于如何计算 loss。看一下能不能套用到原先 BERT 的框架中。

```python
from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

2022-10-23 18:32:34，直接用 forward 函数中计算的 loss

```python
label_ids = label_ids.to(torch.float)
outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
logits = outputs.logits
loss = outputs.loss
```
注意，对于 `single_label_classification` 分类任务，不要将 `label_ids` 转化成 int。

# 5. 理解

## 5.1 深度学习研究者在做什么

转载自 https://www.zhihu.com/question/433274875/answer/2240764095

以下阶段层层递进

+ 把已有的开源模型下载下来，换成自己的数据集。这个时候应该是偏向于应用型研究，把动物图像数据集换成自己领域的，医学图像数据集，并且跑出 SOTA 的准确率。

> 这个过程要求看懂开源的代码架构，loss, optimizer, 基本的前向反向传播应该无需修改，但是需要了解数据的预处理，输入的数据是什么样的，如何进行数据转换，数据集从哪里下载，用哪个函数 load data。

+ 从理论上理解模型的一些算法和思想，知道一些重要的超参数背后的思想。有信心调整它，并符合自己预期的效果

> 最显然的，learning rate, batch size，又比如量化时的 scale factor 搜索范围

+ 深入了解模型的代码实现和细节，能够为模型添加一些主流的，提升性能的 module 或者 trick；能够分析修改前后，模型的差异；对模型的推理和训练过程了如指掌

> 比如分析推理时某个 epoch 的激活分布情况，分布出现了什么问题，如何解决，loss 变化有什么问题等等

+ 能够根据自己的业务需求，拓展模型的功能，让模型能做更多的事情

> 这个可能更多体现在业务需求层面，如何去部署公司需要的模型。不过我认为做了量化框架的拓展，也达到这个水平了。


# Reference 
https://zhuanlan.zhihu.com/p/120315111