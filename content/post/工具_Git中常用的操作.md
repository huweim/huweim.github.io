---
title: "Git中常用的操作"
date: 2021-03-08T19:12:40+08:00
lastmod: 2022-11-26 16:21:28
draft: false
tags: ["Git"]
categories: ["工具"]
---

Git中常用的操作

# 几个概念

工作区 (Workspace): 通俗地理解就是本地的环境，一般来说代码就是在工作区打开和编辑的。

暂存区 (stage, index): 这个信息存放在 `.git` 目录下的 `index` 文件里面

远端

# 0. 配置 ssh

## 0.1 Generate

```shell
$ ssh-keygen -t rsa -C "506834968@qq.com"
#一直 Enter 即可

$ eval "$(ssh-agent -s)"
$ ssh-add ~/.ssh/id_rsa
```

## 0.2 在 Github 中添加 SSH

把 ~/.ssh/id_rsa.pub 中的内容复制到 Github 新生成的密钥

```shell
$ clip < ~/.ssh/id_rsa.pub
$ ssh -T git@github.com
Hi huweim! You've successfully authenticated, but GitHub does not provide shell access.
```

现在即可正常 push

## 0.3 用户名和邮件设置

```shell
git config --global user.name huweim
git config --global user.email "506834968@qq.com"
```

# 1. Remote Repo

## 1.1 Add Remote Repo

```bash
$ git remote add origin git@github.com:huweim/repo_name.git
```

## 1.2 Delete Remote Repo

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

## 1.3 Pull Origin Master to Local

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

本地有修改，但是想用 remote 文件直接覆盖本地文件
```shell
# 暂时将工作区和暂存区代码保存
git stash
git pull

# 如果想查看 stash list 以及清除之前的修改
git stash list
git stash clear
```

上面命令表示，取回 origin/master 分支，再与本地的 brantest 分支合并。

## 1.4 Clone from Remote Repo

```bash
$ git clone git@github.com:huweim/repo_name.git
$ git clone https://github.com/huweim/repo_name.git
```

+ 很多时候http速度很慢，会clone失败报错，所以建议使用ssh。

# 2. 代码提交

包括 `git add`, `git commit`, `git push`, `git merge`


## 2.1 git add


# 3. Delete

## 3.1 删除工作区文件

+ 与win中右键删除没有区别

+ If you want to remove the file from the Git repository **and the filesystem**, use:

```sh
git rm file1.txt   #删除file1.txt
git commit -m "remove file1.txt"
```

## 3.2 仅删除暂存区中的文件

But if you want to remove the file only from the Git repository and not remove it from the filesystem, use:

```sh
git ls-files #查看暂存区文件
git rm --cached file1.txt
git commit -m "remove file1.txt"
```

And to push changes to remote repo

```sh
git push origin branch_name
```

## 3.3 删除文件夹 (踩坑)

+ 主要是由于之前commit过某些文件和文件夹，导致无法忽略这些文件
+ 采用命令git rm --cached "文件路径"
+ 删除文件夹的时候，出现 not removing 'game/logs' recursively without -r，说明我们需要添加参数 -r 来递归删除文件夹里面的文件
+ git rm -r --cached "文件路径"
+ 然后commit

## 3.4 删除分支

```shell
git branch -a		#查看
git push origin --delete <branchName>  #删除
```



# 4. 回滚

+ `HEAD`指向的版本就是当前版本，因此，Git允许我们在版本的历史之间穿梭，使用命令`git reset --hard commit_id`。
+ 穿梭前，用`git log`可以查看提交历史，以便确定要回退到哪个版本。
+ 要重返未来，用`git reflog`查看命令历史，以便确定要回到未来的哪个版本。

## 4.1 查看历史版本

### 4.1.1 查看过去

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

### 4.1.2 寻找未来

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

## 4.2 撤销修改操作

**Case 1 -> 修改了file，还未add**

Case以文件 README.md 为例

```bash
git checkout -- file
```

e.g 

```bash
git checkout -- README.md
```

+ 丢弃工作区的修改，回到上一次commit时的状态。

2022-11-23 21:18:33，这个可能会比较常用，实际上回到上一次 commit 的状态也比较安全。但是当前在工作区的改动会丢失。

**Case 2 -> 修改了file，且已经add**

```bash
git reset HEAD <file>    #撤销对暂存区的修改  回到case 1的状态
git checkout -- file     #情况来到case1, 撤销工作区修改
```

e.g

```bash
git reset HEAD README.md
git checkout -- README.md
```

**case 3 -> 修改了file，且已经add，并commit**

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

## 4.3. 版本回退

```bash
git reset --hard HEAD^    #回到上一个版本
git reset --hard HEAD^^   #回到上上一个版本
git reset --hard 1084a    #回到commit_id以1084a开头的版本
git reset --hard HEAD~100 #回到上100个版本
```

## 4.4 回滚后在提交（坑）

### I. 基本操作

```bash
git add -A   
git commit -m '说明'
git push -u origin master   
```

### II. 报错

在执行`push -u origin master` 命令后报错

```bash
error: failed to push some refs to 'github.com:huweim/huweim.github.io.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
```

> 报错原因-> `当前提交版本低于git主版本`

### III. 解决方法

```bash
push -u origin master -f
```

> - **`-f`** 覆盖当前主分支 如果您的仓库只有自己在玩,这个命令可以随意使用
> - 如果你是协同开发共用仓库,请务必不要带 **`-f`** 参数提交代码,后果不亚于 **删库跑路**

# 5. 分支

## 5.1 概念
conflict，两个文件合并时可能会出现冲突，如何定义冲突？

> A conflict arises when two separate branches have made edits to the same line in a file, or when a file has been deleted in one branch but edited in the other.

看起来是按行进行索引。

出现冲突的场景

+ 不同分支，modify same file, same line
+ 不同分支，modify same file, adjacent line
## 5.2 分支合并
2022-11-23 21:27:36。分支可能是 git 的一个特色了，但自己从来没有使用过这个功能。

```shell
# 查看本地分支
$ git branch
* main

# 查看本地分支和 remote 分支
$ git branch -a
* main
  remotes/origin/HEAD -> origin/main
  remotes/origin/main

```

分支合并 (`git merge`)。独立开发了某个功能，最终还是要将他们合并到主分支的。目前没有用过这个东西，但是可以了解一下他的逻辑。

```shell
$ git branch  
* master
  branch_test
$ git merge branch_tst
# 应该是默认合并到当前 branch

$ git branch -d branch_test
```

## 5.3 解决 remote 和本地冲突 :star:

这个在自己最近（2022-11-23 22:04:27）的开发场景中经常出现，情况就是和别人合作进行开发，当别人已经 push 了新的改动，而自己在本地也做了一些修改，但是还没有 push。这时会先把别人的改动 pull 下来，但是可能会产生冲突。

粗暴的方法 1 是保存下自己的修改，直接重新 clone 一份代码，然后再加入自己的修改。但是当工程量增大时，手动操作也容易不错，并且不太方便。

我自己的方法 2 是保存下自己的修改，通过 git stash 暂存本地代码，注意 git stash 会回退到上一次的 commit。虽然自己的本地代码会保存下来，用 `git stash list` 还可以查看，但是自己从来没有用这个恢复过。总的来说也有可能出错，不太方便。

目前学到的方法 3 是可能会比较方便。

+ 首先，通过 `git add; git commit` 提交本地的修改
+ `git pull` 直接拉取并且 merge，没有冲突时会自动 merge；出现冲突时，上面描述了方法 1，2，下面是方法 3
  
```shell
# #查看一下本地分支和 remote 分支的名字
# $ git branch -a 
# * main
#   remotes/origin/HEAD -> origin/main
#   remotes/origin/main
# # 对比冲突
# $ git diff main remotes/origin/HEAD

# 使用 diff 还是不行，还是使用插件 git lens

$ git pull
Auto-merging tex/evaluation.tex
CONFLICT (content): Merge conflict in tex/evaluation.tex
Automatic merge failed; fix conflicts and then commit the result.
# 告知有冲突，此时通过可以对比冲突出现的位置，手动解决之后再次 git add, commit 然后 pull 即可。
```

> 2022-11-23 22:29:06，不能直接 git diff，这样会对比所有的文件，而我们只希望对比冲突的文件。
> 实际上，git lens 这个插件就是在实现 diff 的功能，只是其标注出了冲突的位置

### 5.3.1 实例

比如用 ISCA paper 来测试一下
```shell
# 直接 pull 出现冲突，因为在 overleaf 上已经改了很多次了
$ git pull
error: Your local changes to the following files would be overwritten by merge:
        tex/evaluation.tex
Please commit your changes or stash them before you merge.
Aborting
Updating e83209c..44817d2
先 commit 自己的提交
$ 
```

### 5.4.1 冲突测试



# 6. Git常用命令速查表

<img src="D:\STU\2021-Spring\Core Course\Git\Git常用命令大全.jpg" alt="Git常用命令大全" style="zoom:50%;" />

# 7. Bug

## 7.1 failed to push some refs to git

出现错误的主要原因是github中的README.md文件不在本地代码目录中

```
git pull --rebase origin master
```

重新 push 即可解决，理解为合并分支

## 7.2 master, main 分支名不同

error: src refspec main does not match any.
error: failed to push some refs to 'git@github.com:huweim/ant_extension.git'

解决：本地分支名字改为 main

```shell
git branch -m master main
```

# 8. submodule

Hugo 中修改 theme 属于更新 submodule，不能直接 `git add`，要使用命令 `git submodule update`
```shell
git submodule update --remote 
# 这个是从子模块的 remote 端克隆最新版本到本地，会覆盖本地的文件
```

# 9. Branch

```shell
git branch -a     # 查看所有分支
git branch        # 查看当前使用的分支
git branch XXX    # 创建分支 XXX
git checkout XXX  # 去往另一个分支
```


#### VSCode BUG

Missing or invalid credentials.
Error: connect ECONNREFUSED /run/user/2039/vscode-git-5fd9577edd.sock

解决：https://juejin.cn/post/7062237873570840589，取消 `git.terminalAuthentication` 中的勾，然后 reload。

2022-08-05 19:59:13，这个方法不太行，搞了之后需要重新输密码。再遇到这个错误可以先重启几次。

# Reference

<https://www.jianshu.com/p/bc067471781f>

<https://www.runoob.com/git/git-basic-operations.html>

<https://www.liaoxuefeng.com/wiki/896043488029600/897013573512192>