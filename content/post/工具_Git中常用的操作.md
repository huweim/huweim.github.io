---
title: "Git中常用的操作"
date: 2021-03-08T19:12:40+08:00
lastmod: 2022-03-16 08:44:28
draft: false
tags: ["Git"]
categories: ["工具"]
---

# Git中常用的操作

### 0. 设置账户和 ssh

#### 0.1 Set who you are
```shell
$ git config --global user.email "506834968@qq.com"
$ git config --global user.name "huweim"
```
#### 0.2 ssh



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

#### 1.3 Pull Origin Master to Local

:x:似乎 push 之前如果有东西需要先 pull

+ 是因为远端已经创建了 README.md，local 也有README.md。有冲突，所以需要先 pull 过来同步。

**git pull** 命令用于从远程获取代码并合并本地的版本。

**git pull** 其实就是 **git fetch** 和 **git merge FETCH_HEAD** 的简写。 命令格式如下：

```bash
git pull <远程主机名> <远程分支名>:<本地分支名>
```

将远程主机 origin 的 master 分支拉取过来，与本地的 brantest 分支合并。

```bash
git pull origin master:brantest
```

如果远程分支是与当前分支合并，则冒号后面的部分可以省略。

```bash
git pull origin master
```

上面命令表示，取回 origin/master 分支，再与本地的 brantest 分支合并。

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
git rm --cached file1.txtgit commit -m "remove file1.txt"
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
  $ git reset --hard commit_id$ git reset --hard 964ab89
  ```

+ 知道commit_id可以回到过去的版本

```bash
$ git logcommit 964ab89b23c2aa367c89ee2e9578e1398819d523 (HEAD -> master, origin/master)Author: huweim <506834968@qq.com>Date:   Mon Mar 8 15:43:24 2021 +0800    add first postcommit bd6582181e070822896bbb8cfb2b3a3e80cf3163Author: huweim <506834968@qq.com>Date:   Mon Mar 8 15:30:25 2021 +0800    Full themes of archie
```

##### 4.1.2 寻找未来

+ `git reflog`查看命令历史
+ 根据commit_id寻找当前版本的“未来”

```bash
$ git reflog964ab89 (HEAD -> master, origin/master) HEAD@{0}: commit: add first postbd65821 HEAD@{1}: commit: Full themes of archie23ca6d2 HEAD@{2}: commit: modify the hugo themesa6ad376 HEAD@{3}: reset: moving to HEAD^4eaa987 HEAD@{4}: reset: moving to HEAD^d7770ba HEAD@{5}: commit: Publish fitst 14eaa987 HEAD@{6}: commit: Publish fitsta6ad376 HEAD@{7}: pull: Fast-forward6db58f9 HEAD@{8}: commit (initial): Initial Commit
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
git reset HEAD <file>    #撤销对暂存区的修改  回到case 1的状态git checkout -- file     #情况来到case1, 撤销工作区修改
```

e.g

```bash
git reset HEAD README.mdgit checkout -- README.md
```

##### case 3 -> 修改了file，且已经add，并commit

```bash
git reset --hard HEAD^    #回到上一个版本，回到case 2的状态git reset HEAD <file>     #撤销对暂存区的修改  回到case 1的状态git checkout -- file      #情况来到case1, 撤销工作区修改
```

e.g

```bash
git reset --hard HEAD^git reset HEAD README.mdgit checkout -- README.md
```

#### 4.3. 版本回退

```bash
git reset --hard HEAD^    #回到上一个版本git reset --hard HEAD^^   #回到上上一个版本git reset --hard 1084a    #回到commit_id以1084a开头的版本git reset --hard HEAD~100 #回到上100个版本
```

#### 4.4 回滚后在提交（坑）

##### I. 基本操作

```bash
git add -A   git commit -m '说明'push -u origin master   
```

##### II. 报错

在执行`push -u origin master` 命令后报错

```bash
error: failed to push some refs to 'github.com:huweim/huweim.github.io.git'hint: Updates were rejected because the tip of your current branch is behindhint: its remote counterpart. Integrate the remote changes (e.g.hint: 'git pull ...') before pushing again.hint: See the 'Note about fast-forwards' in 'git push --help' for details.
```

> 报错原因-> `当前提交版本低于git主版本`

##### III. 解决方法

```bash
push -u origin master -f
```

> - **`-f`** 覆盖当前主分支 如果您的仓库只有自己在玩,这个命令可以随意使用
> - 如果你是协同开发共用仓库,请务必不要带 **`-f`** 参数提交代码,后果不亚于 **删库跑路**

# Reference

<https://www.jianshu.com/p/bc067471781f>

<https://www.runoob.com/git/git-basic-operations.html>

<https://www.liaoxuefeng.com/wiki/896043488029600/897013573512192>

### 6. Git常用命令速查表

<img src="D:\STU\2021-Spring\Core Course\Git\Git常用命令大全.jpg" alt="Git常用命令大全" style="zoom:50%;" />