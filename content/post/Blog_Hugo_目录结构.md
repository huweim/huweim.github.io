---
title: "Hugo 目录结构"
date: 2022-04-19T18:10:17+08:00
lastmod: 2023-03-21 11:12:52
draft: false
author: "Cory"
tags: ["Hugo"]
categories: ["工具"]
---

# 0. 前言

配置博客的代码高亮

# 1. 目录结构

其中，`config.toml` 是网站的配置文件，Hugo还可使用 `config.yaml` 或者 `config.json` 进行配置。

`content` 目录中存放所有的网站内容，可在此目录中建立其他子目录，即为子模块。

`layouts` 目录存放 `.html` 格式的模板。模板确定了静态网站渲染的样式。

此目录存储.html文件作为布局模板，这些模板声明了内容视图在静态站点的呈现。包含的模板有[列表页lists], 您的 [主页homepage], [标签模板taxonomy templates], [部分页面模板 partials], [单独页模板singles]

`themes` 目录存放网站使用的theme主题模板。

`static` 目录存放未来网站使用的静态内容，比如图片、css、JavaScript等。当Hugo生成静态网站时，该目录中的所有内容会原封不动的被复制。

`archetypes` 目录存放网站预设置的文件模板头部，当使用 `hugo new` 时即可生成一个带有该头部的实例。

`data` 目录用来存储Hugo生成网站时应用的配置文件。配置文件可以是YAML，JSON或者TOML格式。

`public` 目录存放静态网页的代码，也就是生成的 html 文件；使用命令 `hugo` 会得到这个目录

2022-05-09 11:18:16 补充

`D` 目录，应该是自动生成的静态页面，比如在 `content` 目录下添加了 `about.md`，网页上可以看到 about 页面。但是实际上是存在 `D/about/index.html` 文件的，猜测是 Hugo 自动生成的。

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
> 在同样目录下，建立其他 `.md` 文件，同样也是有相似的文件头部。该博客的文件名应和 `title` 一致，但要注意 `title` 中的空格或者 `+` 作为文件名时应该替换成`-`， 不然会报找不到404网页。文件内容在这块区域下面，使用markdown语法。

## 1.2 static

如果在 static 目录下也添加 `about` 目录，会覆盖掉 `content` 中用 markdown 编写的 about 页面

也就是说 Hugo 程序中， static 目录优先级高于 content 中使用 markdown 生成的页面文件。

## 1.3 Layout

2022-06-10 19:55:20，在 static 目录下放入 `index.html` 文件，上传到 github 后首页没有变化。猜测 Hugo 是根据 Layout 的设置来生成静态页面，因此应该从 layout 着手修改。

根目录下的 layout 应该是有最高的优先级，然后会加载 theme 目录中的 layout，来达到根据不同的主题布局的效果。

2022-06-10 20:27:38，修改 theme/jane/layouts 目录下的 index.html，it works :heavy_check_mark:

注意，修改 theme 属于更新 submodule，不能直接 `git add`，要使用命令 `git submodule update`。

2022-06-10 20:54:13 搞定。不用搞这么麻烦，直接把 index.html 复制到根目录下的 layouts 目录即可。

## 1.4 显示的标题等级设置

2023-03-21 15:23:20，发现目录不显示一级标题，自己的习惯在笔记中会从一级标题（一个#）开始使用，不显示的话看起来很不方便。

解决方法：https://github.com/olOwOlo/hugo-theme-even/issues/240

在 `config.toml` 中加入 

```yaml
[markup]
  [markup.tableOfContents]
    endLevel = 5
    ordered = false
    startLevel = 1
```

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