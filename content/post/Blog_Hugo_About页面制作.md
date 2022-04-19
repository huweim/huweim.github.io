---
title: "Hugo About页面制作"
date: 2021-07-24T19:19:09+08:00
draft: false
tags: ["Hugo"]
categories: ["博客搭建"]
---

# 前言

最近在学习使用 hugo 制作自己的博客，把制作过程的记录下来。我想博客应该会是之后的学习工作中会频繁使用和交互的东西。

本文记录添加 about 页面的过程。

注：这个是使用 markdown 进行添加，并非制作 html 页面。Hugo 主题基于 Ink

## 添加 About 页面

右键打开 Git 命令行，输入

```
hugo new about.md
```

在文件夹 `posts` 的同级目录下新建了一个 `about.md` 文件。

修改 markdown 文件顶部的选项使其能够出现在首页菜单栏

```
title: "About"
date: 2021-07-08T17:46:31+08:00
menu: "main"
weight: 60
comment: false
```

## 预览

在根目录下打开 Git，输入命令

```
hugo -D server
```

在自己的浏览器上访问网址：http://localhost:1313/ 即可预览

# Reference

https://cpurely.github.io/post/hugo%E5%A6%82%E4%BD%95%E6%B7%BB%E5%8A%A0about%E5%92%8C%E8%87%AA%E5%AE%9A%E4%B9%89%E9%A1%B5%E9%9D%A2/

