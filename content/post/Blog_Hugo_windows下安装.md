---
title: "Hugo Windows下安装"
date: 2021-07-24T16:33:34+08:00
draft: false
tags: ["Hugo"]
categories: ["工具"]
---

# Installing on Windows

## 假设

1. 你知道如何打开一个命令提示窗口。
2. 你运行的是一个现代64位的 Windows。
3. 你的网站地址是 `example.com`。
4. 你将使用 `D:\Hugo\Sites` 作为网站的起点。
5. 你将使用 `D:\Hugo\bin` 存储可执行文件。

## 设置你的文件夹

你将需要一个存储 Hugo 可执行文件、博客内容（你创建的的那些文件），以及生成文件（Hugo 为你创建的 HTML）的地方。

1. 打开 Windows Explorer。
2. 创建一个新的文件夹，`D:\Hugo`。
3. 创建一个新的文件夹，`D:\Hugo\bin`。
4. 创建一个新的文件夹，`D:\Hugo\Sites`。

## 下载预先编译好的 Windows 版本的 Hugo 可执行文件

> 2021-07-07 11:25:22 为什么找不到 hugo 的命令了，可能是因为把文件夹 STU 改名改成了 ShanghaiTech，而 windows 需要配置环境变量。重新配置环境变量中的路径应该就可以了
>
> :heavy_check_mark: 就是这个原因，不过注意是系统变量中
>
> 环境变量，简单来说就是在系统层面给这个程序的安装路径进行登记，使得我们通过CMD或Git直接输入程序名就能全局调用。

> 使用 go 编译 Hugo 的一个优势就是仅有一个二进制文件。你不需要运行安装程序来使用它。相反，你需要把这个二进制文件复制到你的硬盘上。我假设你将把它放在 `D:\Hugo\bin` 文件夹内。如果你选择放在其它的位置，你需要在下面的命令中替换为那些路径。
>
> 1. 在浏览器中打开 https://github.com/spf13/hugo/releases。
> 2. 当前的版本是 hugo_0.13_windows_amd64.zip。
> 3. 下载那个 ZIP 文件，并保存到 `D:\Hugo\bin` 文件夹中。
> 4. 在 Windows Explorer 中找到那个 ZIP 文件，并从中提取所有的文件。
> 5. 你应该可以看到一个 `hugo_0.13_windows_amd64.exe` 文件。
> 6. 把那个文件重命名为 `hugo.exe`。
>    + 之后不要改动其位置
> 7. 确保 `hugo.exe` 文件在 `D:\ShanghaiTech\Hugo\bin` 文件夹。（有可能提取过程会把它放在一个子文件夹中。如果确实这样的话，使用 Windows Explorer 把它移动到 `D:\ShanghaiTech\Hugo\bin`。）
> 8. 使用 `D:\ShanghaiTech\Hugo\bin>set PATH=%PATH%;D:\ShanghaiTech\Hugo\bin` 把 hugo.exe 可执行文件添加到你的 PATH路径中。
>    + 在 windows 中手动添加环境变量就可以了

## 验证版本 

右键点击 git bash here 打开 git 命令行，输入

```
hugo version
```

返回了相应的版本号则说明 Hugo 已成功配置
