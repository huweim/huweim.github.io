---
title: "Hugo 文章扉页设置"
date: 2022-04-19T18:10:17+08:00
lastmod: 2022-04-19T18:10:17+08:00
draft: false
author: "Cory"
tags: ["Hugo"]
categories: ["工具"]
---

# 0. 前言

描述如何设置和自定义一篇 post 的页面内容和标签

这个在英文中的描述应该是 扉页 front matter

# 1、原型

原型是创建新页面内容（运行`hugo new`命令）时使用的模板，预先配置格式：例如md文件的**扉页**（ front matter）、其它格式等。原型文件应该存放在**archetypes**目录内。

原型（**archetypes/default.md**）内的扉页貌似不能进行日期格式转换：

- date属性只能是`date: {{ .Date }}`，因为之后的日期格式转换[基于此date属性](https://gohugo.io/functions/format/#hugo-date-and-time-templating-reference)。若`date: {{.Date.Format "2006-01-02"}}`，将会触发错误：Error: Failed to process archetype file “default.md”:: template: default:3:19: executing “default” at <.Date.format>: can't evaluate field format in type string

## 1.1 archetypes/default.md

default.md：将md文件构建为HTML的页面文件（type：缺省）。

```yaml
---
title: "{{ replace .Name "-" " " | title }}"
date: {{ .Date }}
author: "komantao"
LastModifierDisplayName: "komantao"
LastModifierEmail: komantao@hotmail.com
weight: 20
url: {{ replace .Dir "\\" "/" }}{{ replace .Name "-" " " | title }}.html
draft: false
description: "文章描述"
keywords: [keyword1, keyword2, keyword3]
tags: [标签1, 标签2]
categories: [分类1, 分类2]
---
首页描述。
```

# 3、扉页

扉页（ front matter）用来配置文章的标题、时间、链接、分类等元信息，提供给模板调用。可使用的格式有：yaml格式（默认格式，使用3个减号-）、toml格式（使用3个加号+）、json格式（使用大括号{}）。除了网站主页外，其它内容文件都需要扉页来识别文件类型和编译文件。

```yaml
---
title: "xxx"                  # 文章标题
menuTitle: "xxx"              # 文章标题在菜单栏中显示的名称
description: "xxx"            # 文章描述
keywords: ["Hugo","keyword"]  # 关键字描述
date: "2018-08-20"            # 文章创建日期
tags: [ "tag1", "tag2"]       # 自定义标签
categories: ["cat1","cat2"]   # 自定义分类
weight: 20                    # 自定义此页面在章节中的排序优先级（按照数字的正序排序）
disableToc: "false"           # 若值为false（缺省值）时，此页面启用TOC
pre: ""                       # 自定义menu标题的前缀
post: ""                      # 自定义menu标题的后缀
chapter: false                # 若值为true（缺省值）时，将此页面设置为章节（chapter）
hidden: false                 # 若值为true（缺省值）时，此页面在menu中隐藏
LastModifierDisplayName: ""   # 自定义修改者的名称，显示在页脚中
LastModifierEmail: ""         # 自定义修改者的Email，显示在页脚中
draft: false                  # true，表示草稿，Hugo将不渲染草稿
url:                          # 重置permalink，默认使用文件名
type:                         # type与layout参数将改变Hugo寻找该文章模板的顺序
layout: 
---
```

- weight属性

  - 缺省时按照date属性的倒序排序（新日期排在前面）
  - 设置时，自定义此页面在章节中的排序（按照数字值的正序排序，数字小的排在前面，若数字值相同，则按照date属性的倒序排序）

- pre属性

  在菜单栏中的标题前添加前缀：可为数字、文字、图标（[**Font Awesome**](https://fontawesome.com/v4.7.0/icons/)库）等。

  ```yaml
  +++
  title = "Github repo"
  pre = "<i class='fab fa-github'></i> "
  +++
  ```

- menuTitle属性

  - 缺省时调用title属性作为此页面在menu中显示的名称
  - 设置时，自定义此页面在menu中显示的名称

# Reference

https://kuang.netlify.app/blog/hugo.html
