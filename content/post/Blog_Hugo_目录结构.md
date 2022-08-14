---
title: "Hugo 目录结构"
date: 2022-04-19T18:10:17+08:00
lastmod: 2022-06-11 11:12:52
draft: false
author: "Cory"
tags: ["Hugo"]
categories: ["博客搭建"]
---

# 0. 前言

配置博客的代码高亮

# 1. 文件夹结构

其中，`config.toml` 是网站的配置文件，Hugo还可使用 `config.yaml` 或者 `config.json` 进行配置。

`content` 文件夹中存放所有的网站内容，可在此文件夹中建立其他子文件夹，即为子模块。

`layouts` 文件夹存放 `.html` 格式的模板。模板确定了静态网站渲染的样式。

`themes` 文件夹存放网站使用的theme主题模板。

`static` 文件夹存放未来网站使用的静态内容，比如图片、css、JavaScript等。当Hugo生成静态网站时，该文件夹中的所有内容会原封不动的被复制。

`archetypes` 文件夹存放网站预设置的文件模板头部，当使用 `hugo new` 时即可生成一个带有该头部的实例。

`data` 文件夹用来存储Hugo生成网站时应用的配置文件。配置文件可以是YAML，JSON或者TOML格式。

## 1.1 Post

```
---
date: "2021-07-08T19:38:26+08:00"
draft: false
title: "Blog"
---
```

> 其中，`date` 说明该博客建立时间，`draft` 说明这篇是否是草稿，若是草稿，在无特别指明情况下并不会生成静态网页，`title` 表明该文件显示的标题。
>
> 在同样文件夹下，建立其他 `.md` 文件，同样也是有相似的文件头部。该博客的文件名应和 `title` 一致，但要注意 `title` 中的空格或者 `+` 作为文件名时应该替换成`-`， 不然会报找不到404网页。文件内容在这块区域下面，使用markdown语法。

# 2. theme 目录结构

## 2.1 修改 post 页面

以主题 `jane` 为例，在文件 `theme/jane/layouts/post/single.html` 中存放着如何显示 post。根据文件中的代码，关于 table of content 的设定存放在 `theme/jane/layouts/partials/post/toc.html`

之后把原来的 tableofcontent 那部分代码放到 nav 中即可。

```html
  <nav class="hide-on-mobile section-nav">
    <h3 class="ml-1">Table of contents</h3>
    {{ .TableOfContents }}
  </nav>
```
