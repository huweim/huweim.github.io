---
title: "Blog_Hugo_基本部署"
date: 2021-07-24T16:28:22+08:00
draft: false
tags: ["Hugo"]
categories: ["博客搭建"]
---

# 前言

记录一下从0开始的部署，之前3月弄的没有记笔记，7月就忘记了，还是 要好好整理好好记录。

假设现在已经安装好了 Hugo 环境，我使用的是 windows 下安装。

## 添加主题

有很多 Hugo Theme 可以选择

这里一开始用的是 archie，现在改成 Ink

```
cd blog;\
git init;\
git submodule add https://github.com/knadh/hugo-ink.git themes/hugo-ink;\

# Edit your config.toml configuration file
# and add the Ananke theme.
echo 'theme = "ananke"' >> config.toml
```

> 切换主题后 push github 报错的本质原因是没有执行 `git submodule add`, 即没有在文件 `.gitmodules` 中加入新的主题

## 发布文章

```hugo
hugo new posts/XXX.md
```

会在 `contene/posts` 文件夹下生成 `XXX.md` 文件

```
title: "Blog_Hugo_基本部署"
date: 2021-07-24T16:28:22+08:00
draft: false
tags: ["博客", "技巧"]
categories: ["Hugo"]
```

为文章添加 tags, categories 以便分类

:star: tags and categories 最好不要使用一模一样的名字，否则会出现 ambiguous 错误

## 预览

在根目录下打开 Git，输入命令

```
hugo -D server
```

在自己的浏览器上访问网址：http://localhost:1313/ 即可预览

