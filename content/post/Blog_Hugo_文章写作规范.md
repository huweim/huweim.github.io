---
title: "Hugo 文章写作格式"
date: 2021-10-08T12:13:07+08:00
lastmod: 2022-06-11 11:13:15
draft: false
author: "Cory"
keywords: ["文章写作规范"]
tags: ["Hugo"]
categories: ["博客搭建"]
---

# 0. 前言

记录一下文章写作和 扉页标签的内容，文章完全转载自 [博客](https://shuzang.github.io/2019/hugo-blog-article-write/#3-%E6%8E%92%E7%89%88%E9%A3%8E%E6%A0%BC%E8%A7%84%E8%8C%83)

---

本文是 Hugo 使用记录的第二篇，介绍关于文章写作的一些问题，包括分类管理、排版技巧、特殊语法等，所有语法基于 LoveIt 主题。

Hugo支持的文章格式为`.md`，即用markdown语言编辑的文章。所有的文章都放在`content/posts`目录下，支持级联目录，即在`posts`目录下按分类建立多个子文件夹放置文章，比如本博客的文章按分类放在四个子文件夹下。

```sh
$ ls posts
爱编程爱技术的孩子/  我所热爱的生活啊/
平日里的白日梦/      研究生的区块链学习之路/
```

下面是三条方便清晰管理和生成文章的目录结构建议:

- 保持博客文章存放在 `content/posts` 目录, 例如: `content/posts/我的第一篇文章.md`
- 保持简单的静态页面存放在 `content` 目录, 例如: `content/about.md`
- 保持图片之类的媒体资源存放在 `static` 目录, 例如: `static/images/screenshot.png`

## 1. 前置参数

Hugo 允许在文章内容前面添加 `yaml`, `toml` 或者 `json` 格式的前置参数，LoveIt 默认文章模板提供的前置参数有

> 不过自己现在用的模板似乎不支持这么多参数

```yaml
---
title: "我的第一篇文章"
subtitle: ""
date: 2020-03-04T15:58:26+08:00
lastmod: 2020-03-04T15:58:26+08:00
draft: true
author: ""
authorLink: ""
description: ""
license: ""

tags: []
categories: []
hiddenFromHomePage: false

featured_image: ""
featured_image_preview: ""

toc: false
autoCollapseToc: true
math: true
mapbox:
    accessToken: ""
    lightStyle: ""
    darkStyle: ""
    navigation: true
    geolocate: true
    scale: true
    fullscreen: true
lightgallery: true
linkToMarkdown: true
share:
  enable: true
comment: true
---
```

- **title**: 文章标题.
- **subtitle**: 文章副标题.
- **date**: 这篇文章创建的日期时间. 它通常是从文章的前置参数中的 `date` 字段获取的, 但是也可以在 [网站配置](../theme-documentation-basics/#site-configuration) 中设置.
- **lastmod**: 上次修改内容的日期时间.
- **draft**: 如果设为 `true`, 除非 `hugo` 命令使用了 `--buildDrafts`/`-D` 参数, 这篇文章不会被渲染.
- **author**: 文章作者.
- **authorLink**: 文章作者的链接.
- **description**: 文章内容的描述.
- **license**: 这篇文章特殊的许可.
- **tags**: 文章的标签.
- **categories**: 文章所属的类别.
- **hiddenFromHomePage**: 如果设为 `true`, 这篇文章将不会显示在主页上, 但是此行为可以在 [网站配置](../theme-documentation-basics/#site-configuration) 中设置的.
- **featuredImage**: 文章的特色图片.
- **featuredImagePreview**: 用在主页预览的文章特色图片.
- **toc**: 如果设为 `true`, 这篇文章会显示右侧目录.
- **autoCollapseToc**: 如果设为 `true`, 文章目录会自动折叠.
- **math**: 如果设为 `true`, 将自动渲染文章中的数学公式.
- **mapbox**: 和 [网站配置](../theme-documentation-basics/#site-configuration) 中的 `params.mapbox` 对象相同.
- **lightgallery**: 如果设为 `true`, 文章中的图片将可以按照画廊形式呈现.
- **linkToMarkdown**: 如果设为 `true`, 内容的页脚将显示指向原始 Markdown 文件的链接.
- **share**: 和 [网站配置](../theme-documentation-basics/#site-configuration) 中的 `params.share` 对象相同.
- **comment**: 如果设为 `true`, 将启用评论系统.

## 2. 标签与分类

标签和分类字段是归档文章最常用的两种形式。为了更好的使用，我们需要弄清楚它们的区别。以生活与健身的韦恩图为例[1](#fn:1)，饮食与睡眠同时属于这两个分类，当出现这类型文章时，就可以添加标签来管理。



![标签与分类](https://cdn.sspai.com/2019/11/17/07062af1ba37669fa8b7e91484e1d3da.jpg?imageView2/2/w/1120/q/90/interlace/1/ignore-error/1)标签与分类



博客文章的存储方式决定了它无法拥有太多的分类，因此，树形逐级检索的方法不适用这种情况，我们应当在将博客分为几个大类的同时，使用标签来管理和检索文章，当同样也要注意不应滥用标签，否则会带来视觉上的混乱和管理的不便。

在每篇文章的元数据(Front-Matter字段)中添加`categories`字段作为文章分类，添加`tags`字段作为文章标签，比如本文：

```yaml
tags: [博客写作]
categories: [爱编程爱技术的孩子]
```

标签可以有多个，以逗号分隔，不过即使只有一个标签，外面的中括号也不能省略。

## 3. 排版风格规范

本节内容参考少数派写作排版指南[2](#fn:2)

### 3.1 空格

行文时在中文与英文、中文与数字、英文与数字之间增加空格。例如：

- **推荐**：苹果公司在 2015 年 9 月 9 日发布了 iPhone 6s。
- **不推荐**：苹果公司在2015年9月9日发布了iPhone 6s。

一段文字中有超链接的部分，在超链接的前后使用空格。例如：

- **推荐**：你可以前往 [苹果官网](https://www.apple.com/cn) 了解详情。
- **不推荐**：你可以前往[苹果官网](https://www.apple.com/cn)了解详情。

英文前后接全角标点符号或者表示单位的角标符号时，不需要加空格。例如：

- **正确**：新款屏幕的可视角度为 125°，相比旧款提升了 25%。
- **错误**：新款屏幕的可视角度为 125 °，相比旧款提升了 25 %。

对于有特殊用法的专有名词，如 4K、1080p、iOS 10 等，英文和数字之间是否空格以官方标准为准。

每段文字的开头不需要空两格。

### 3.2 标点符号

引号使用直角引号「」，而不是弯引号 “”。对于微软拼音输入法可以键入`uubd`四个字母开启标点符号的选择。

省略号使用 …… 的标准用法，正确输入方法是 shift + 6。不要使用三个句号。。。，也不要使用三个英文句点 …。

不要重复使用标点符号，尤其是在表达强烈情感的时候。例如：

- **推荐**：这个提议真棒！我喜欢。
- **不推荐**：这个提议真棒！！！我喜欢~~~~

同时，抒发情感的方法有很多，不建议在文中大量使用感叹号等表达强烈情感的符号，也不建议使用波浪号。

### 3.3 中文和西文符号

一般情况下，一个中文句子中出现了英文部分，仍然使用中文标点，即全角符号。例如：

- **正确**：我常用的电子设备是 Kindle、iPad Pro、iPhone。
- **错误**：我常用的电子设备是 Kindle, iPad Pro, iPhone.

如果引用一段完整的英文句子，或是出现在专有名词中的标点，则不需要更改标点符号。例如：

- **正确**：乔布斯说「Stay hungry, Stay foolish.」
- **错误**：乔布斯说「Stay hungry，Stay foolish。」
- **正确**：我最喜欢的手机游戏是 Lifeline: Silent Night。
- **错误**：我最喜欢的手机游戏是 Lifeline：Silent Night。

### 3.4 专有名词

所有品牌名称以官方网站写法为准，应用名称遵循 Google Play 或 App Store 的命名。若无官方中文名称，可使用网络上被大家熟知的通用译名，否则请直接使用外文名称，不要自行翻译。

请正确书写常用英文名词的大小写和空格。例如：

- **正确**：iOS 10、macOS、App Store、Android
- **错误**：iOS10、MacOS、AppStore、android

若应用名称过长可在文中自定缩写，但需用括号注明缩写名称，并声明「下同」。

一般情况下，品牌和应用名称不需要使用直角引号「」括起来。

在称呼 app 时，请使用「应用」「应用程序」或「app」，而非「软件」「程序」。

### 3.5 样式工具

虽然可以使用加粗、斜体、删除线、引用等样式工具，这些工具的使用也可以增加文章的可读性，但是过度使用则会造成排版混乱，因此建议正确、克制地使用样式工具。

例如，需要着重显示的部分请使用「加粗」功能，不要使用「斜体」，更不要使用「加粗 + 斜体」的组合。

### 3.6 注明引用来源

文中有使用外站或外部内容的，务必在引用最后部分注明来源。

文中有使用外站图片，必须在文末标明。来源如果来自外站必须添加链接。例如，

- 本文部分图片来自 [The Verge](http://www.theverge.com/)
- 题图来自电影《春娇与志明》截图

若文章为全文翻译，必须在文中注明原作者及原文地址，并添加原文链接。

### 3.7 写作建议

理清文章结构，动笔之前可以先列一下写作大纲。

用主动语态，不要用被动语态。一般情况下，主动语态比被动语态更有力。

使用具体、明确、展示细节的词汇，能激发想象，使读者自己代入情境。「把硬币放进口袋里，他咧开嘴笑了」，远远强过「他满意地拿走了辛苦挣来的奖赏」。

减少形容词的使用，少用 「的」。

文中涉及到参数规格、数据的部分，要保证严谨性。

文章完成之后通读一遍，记住，不要让读者猜测你在讲什么。

# Reference

https://shuzang.github.io/2019/hugo-blog-article-write/#3-%E6%8E%92%E7%89%88%E9%A3%8E%E6%A0%BC%E8%A7%84%E8%8C%83
