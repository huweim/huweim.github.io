---
title: "Git中常用的操作"
date: 2021-03-08T19:12:40+08:00
draft: false
---

# Git中常用的操作

### 1. Remote Repo

#### 1.1 Add Remote Repo

```bash
$ git remote add origin git@github.com:huweim/repo_name.git
```

#### 1.2 Delete Remote Repo

+ Add a wrong remote repo, we could delete it.

```bash
$ git remote -v
origin  git@github.com:huweim/huweim.github.io.git (fetch)
origin  git@github.com:huweim/huweim.github.io.git (push)
```

+ Delete it 

```bash
$ git remote rm origin
```

### 2. Clone from Remote Repo

```bash
$ git clone git@github.com:huweim/repo_name.git
$ git clone https://github.com/huweim/repo_name.git
```

+ 很多时候http速度很慢，会clone失败报错，所以建议使用ssh。

### 3. Delete

#### 3.1 删除工作区文件

+ 与win中右键删除没有区别

+ If you want to remove the file from the Git repository **and the filesystem**, use:

```sh
git rm file1.txt   #删除file1.txt
git commit -m "remove file1.txt"
```

#### 3.2 仅删除暂存区中的文件

But if you want to remove the file only from the Git repository and not remove it from the filesystem, use:

```sh
git rm --cached file1.txt
git commit -m "remove file1.txt"
```

And to push changes to remote repo

```sh
git push origin branch_name
```

#### 3.3 删除文件夹 (踩坑)

+ 主要是由于之前commit过某些文件和文件夹，导致无法忽略这些文件
+ 采用命令git rm --cached "文件路径"
+ 删除文件夹的时候，出现 not removing 'game/logs' recursively without -r，说明我们需要添加参数 -r 来递归删除文件夹里面的文件
+ git rm -r --cached "文件路径"
+ 然后commit

### 4. 回滚

+ `HEAD`指向的版本就是当前版本，因此，Git允许我们在版本的历史之间穿梭，使用命令`git reset --hard commit_id`。
+ 穿梭前，用`git log`可以查看提交历史，以便确定要回退到哪个版本。
+ 要重返未来，用`git reflog`查看命令历史，以便确定要回到未来的哪个版本。

#### 4.1 查看历史版本

##### 4.1.1 查看过去

+ `git log`命令显示从最近到最远的提交日志

+ commit_id（版本号） 964ab89b23c2.......

+ ```bash
  $ git reset --hard commit_id
  $ git reset --hard 964ab89
  ```

+ 知道commit_id可以回到过去的版本

```bash
$ git log
commit 964ab89b23c2aa367c89ee2e9578e1398819d523 (HEAD -> master, origin/master)
Author: huweim <506834968@qq.com>
Date:   Mon Mar 8 15:43:24 2021 +0800

    add first post

commit bd6582181e070822896bbb8cfb2b3a3e80cf3163
Author: huweim <506834968@qq.com>
Date:   Mon Mar 8 15:30:25 2021 +0800

    Full themes of archie
```

##### 4.1.2 寻找未来

+ `git reflog`查看命令历史
+ 根据commit_id寻找当前版本的“未来”

```bash
$ git reflog
964ab89 (HEAD -> master, origin/master) HEAD@{0}: commit: add first post
bd65821 HEAD@{1}: commit: Full themes of archie
23ca6d2 HEAD@{2}: commit: modify the hugo themes
a6ad376 HEAD@{3}: reset: moving to HEAD^
4eaa987 HEAD@{4}: reset: moving to HEAD^
d7770ba HEAD@{5}: commit: Publish fitst 1
4eaa987 HEAD@{6}: commit: Publish fitst
a6ad376 HEAD@{7}: pull: Fast-forward
6db58f9 HEAD@{8}: commit (initial): Initial Commit
```

#### 4.2 撤销修改操作

##### Case 1 -> 修改了file，还未add

Case以文件 README.md 为例

```bash
git checkout -- file
```

e.g 

```bash
git checkout -- README.md
```

+ 丢弃工作区的修改，回到上一次commit时的状态。

##### Case 2 -> 修改了file，且已经add

```bash
git reset HEAD <file>    #撤销对暂存区的修改  回到case 1的状态
git checkout -- file     #情况来到case1, 撤销工作区修改
```

e.g

```bash
git reset HEAD README.md
git checkout -- README.md
```

##### case 3 -> 修改了file，且已经add，并commit

```bash
git reset --hard HEAD^    #回到上一个版本，回到case 2的状态
git reset HEAD <file>     #撤销对暂存区的修改  回到case 1的状态
git checkout -- file      #情况来到case1, 撤销工作区修改
```

e.g

```bash
git reset --hard HEAD^
git reset HEAD README.md
git checkout -- README.md
```

#### 4.3. 版本回退

```bash
git reset --hard HEAD^    #回到上一个版本
git reset --hard HEAD^^   #回到上上一个版本
git reset --hard 1084a    #回到commit_id以1084a开头的版本
git reset --hard HEAD~100 #回到上100个版本
```

#### 4.4 回滚后在提交（坑）

##### I. 基本操作

```bash
git add -A   
git commit -m '说明'
push -u origin master   
```

##### II. 报错

在执行`push -u origin master` 命令后报错

```bash
error: failed to push some refs to 'github.com:huweim/huweim.github.io.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
```

> 报错原因-> `当前提交版本低于git主版本`

##### III. 解决方法

```bash
push -u origin master -f
```

> - **`-f`** 覆盖当前主分支 如果您的仓库只有自己在玩,这个命令可以随意使用
> - 如果你是协同开发共用仓库,请务必不要带 **`-f`** 参数提交代码,后果不亚于 **删库跑路**

### 5. Reference

<https://www.jianshu.com/p/bc067471781f>

<https://www.runoob.com/git/git-basic-operations.html>

<https://www.liaoxuefeng.com/wiki/896043488029600/897013573512192>

### 6. Git常用命令速查表

![git常用命令大全](../Image/Git_Common.jpg)