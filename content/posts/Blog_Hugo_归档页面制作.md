---
title: "Blog_Hugo_归档页面制作"
date: 2021-10-08T12:09:11+08:00
lastmod: 2021-10-08 12:10:44
draft: false
author: "Cory"
tags: ["Hugo"]
categories: ["博客搭建"]
---

# 0. 前言

最近在学习使用 hugo 制作自己的博客，把制作过程的记录下来。我想博客应该会是之后的学习工作中会频繁使用和交互的东西。

本文记录添加 archives 页面的过程。目前使用的 Hugo 主题 Ink 需要自己添加归档页面

# 1. 新建归档页面模板

+ 进入自己的 Hugo 主题文件夹，我自己的是 `themes/hugo-ink`

+ 在主题文件夹的 `layouts/_default` 文件夹下新建文件 `archives.html`，内容直接复制 `single.html`
+ 将 `archives.html` 文件中的 `{{ .Content }}` 替换为以下内容

```
{{ range (.Site.RegularPages.GroupByDate "2006") }}
    <h3>{{ .Key }}</h3>
        <ul class="archive-list">
        {{ range (where .Pages "Type" "posts") }}
            <li>
            {{ .PublishDate.Format "2006-01-02" }}
            ->
            <a href="{{ .RelPermalink }}">{{ .Title }}</a>
            </li>
        {{ end }}
    </ul>
{{ end }}
```

##### 解释

+ `{{ range (where .Pages "Type" "posts") }}`
  + 归档目录设置为 `content/posts`
  + 注意我的文章文件夹是 `posts`，如果你的是 `post`，请对应修改，否则无法正确解析
+ ` {{ .PublishDate.Format "2006-01-02" }}`
  + 可以选择归档的方式

# 2. 新建 archives 文档

假设文章都存在目录 `content/posts` 下，打开 Git 命令行，输入

```
hugo new posts/archives.md
```

将其顶部配置内容进行如下修改

```
---
title: "Archives"
layout: archives
hidden: true
type: posts
summary: 历史文章按照年月归档.
url: /archives/
---
```

# Reference

https://xbc.me/how-to-create-an-archives-page-with-hugo/

http://maitianblog.com/hugo-archives.html
