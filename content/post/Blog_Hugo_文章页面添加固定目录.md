---
title: "Hugo 文章页面添加固定目录栏"
date: 2022-08-14T19:33:17+08:00
lastmod: 2022-08-14T19:33:17+08:00
draft: false
author: "Cory"
tags: ["Hugo"]
categories: ["博客搭建"]
---


# 0. 前言

阅读博客文章时有一个固定的目录会舒服很多，现在就来探索一下怎么添加这个功能。

# 1. 固定目录

以主题 `jane` 为例，在文件 `theme/jane/layouts/post/single.html` 中存放着如何显示 post。根据文件中的代码，关于 table of content 的设定存放在 `theme/jane/layouts/partials/post/toc.html`

之后把原来的 tableofcontent 那部分代码放到 nav 中即可，如下，注释掉原来的代码（15-17 行）

```html
{{ if or .Params.toc (and .Site.Params.toc (ne .Params.toc false)) }}
<div class="post-toc" id="post-toc">
  <h2 class="post-toc-title">{{ i18n "toc" }}</h2>
  <!-- <div class="post-toc-content">
    {{.TableOfContents}}
  </div> -->
  <nav class="hide-on-mobile section-nav">
    <h3 class="ml-1">Table of contents</h3>
    {{ .TableOfContents }}
  </nav>
</div>
{{- end }}
```