---
title: "Docker Image and Container"
date: 2021-09-17T16:25:25+08:00
draft: false
tags: ["Docker", "技巧"]
categories: ["编程"]
---

# 0. 前言

最近需要使用到 Docker, 记一下笔记和常用的操作。主要是参考菜鸟教程和阮一峰老师的教程。

# 1. 启动Docker服务

```bash
# service 命令的用法
$ sudo service docker start

# systemctl 命令的用法
$ sudo systemctl start docker
```

# 2. Image文件

**Docker 把应用程序及其依赖，打包在 image 文件里面。**只有通过这个文件，才能生成 Docker 容器。
**image 文件可以看作是容器的模板**。Docker 根据 image 文件生成容器的实例。同一个 image 文件，可以生成多个同时运行的容器实例。

```bash
# 列出本机的所有 image 文件。
$ docker image ls
$ docler images
REPOSITORY    TAG       IMAGE ID       CREATED        SIZE
ubuntu        latest    fb52e22af1b0   2 weeks ago    72.8MB
hello-world   latest    d1165f221234   6 months ago   13.3kB
ubuntu        15.10     9b9cb95443b5   5 years ago    137MB
#仓库源		标签		ID				创建时间		大小

# 删除 image 文件
$ docker image rm [imageName]
```

> image 文件是通用的，一台机器的 image 文件拷贝到另一台机器，照样可以使用。一般来说，为了节省时间，我们应该尽量使用别人制作好的 image 文件，而不是自己制作。即使要定制，也应该基于别人的 image 文件进行加工，而不是从零开始制作。
>
> 为了方便共享，image 文件制作完成后，可以上传到网上的仓库。Docker 的官方仓库 [Docker Hub](https://hub.docker.com/) 是最重要、最常用的 image 仓库。此外，出售自己制作的 image 文件也是可以的。

## 2.2 查找Image

我们可以从 Docker Hub 网站来搜索镜像，Docker Hub 网址为： **https://hub.docker.com/**

我们也可以使用 docker search 命令来搜索镜像。比如

```bash
$ docker search gpgpusim
NAME                                DESCRIPTION                                 STARS     OFFICIAL AUTOMATED
findhao/gpgpusim_runtime            gpgpusim docker                                 0                [OK]
pli11/gpgpusim                                                                      0                    
syuan3/gpgpusim                     a docker container for gpgpusim simulaator      0                    
socalucr/gpgpusim-homework          Container used to evaluate GPU homework @UCR    0                    
pitipat1998/gpgpusim                                                                0                    
yhgong/gpgpusimdev_200318                                                           0                    
lemonsien/gpgpusim                                                                  0                    
jefferlee/gpgpusim_runtime                                                          0                    
msharmavikram/gpgpusimdnn           The repository has all required elements for…   0                    
minttmdgh/gpgpusim-setting          default                                         0                    
marziehlenjani/gpgpusimwithcuda11                                                   0                    
sis013/injesim4                     gpgpusim-4.0 with jonghyun                      0     
#仓库源的名称							镜像描述				         类似github star 是否docker官方发布 自动构建
```

## 2.3 pull镜像

使用命令 docker pull 来下载镜像

```bash
$ docker pull findhao/gpgpusim_runtime
```

没想到 UCR 那个还挺大的，下了一会儿失败了，换成了 findhao/gpgpusim_runtime

## 2.4 运行镜像

```bash
$ docker run -it findhao/gpgpusim_runtime
```

## 2.4 创建镜像

两种方法

+ 从已经创建的容器中更新镜像，并且提交这个镜像。即把容器快照导入镜像
+ 使用 Dockerfile 指令来创建一个新的镜像

### 2.4.1 更新镜像

更新镜像之前，我们需要使用镜像来创建一个容器

```bash
$ docker run -t -i ubuntu:15.10 /bin/bash
root@b251c90ca048:/# 
```

在运行的容器内使用 **apt-get update** 命令进行更新。

在完成操作之后，输入 exit 命令来退出这个容器。

此时 ID 为 b251c90ca048 的容器，是按我们的需求更改的容器。我们可以通过命令 docker commit 来提交容器副本。

```bash
$ docker commit -m="has update" -a="cory" b251c90ca048  huweim/ubuntu:v2
#-m 描述信息 -a 指定作者 b251c90ca048 容器ID huweim/ubuntu:v2 创建的目标镜像名
sha256:93069e854b178767dfcd334c8ce99d29141fdc87719c2bb1251d9e16e255de73
```

> 也就是我们在镜像中做了修改，随时更新保存为新的镜像即可

### 2.4.2 构建镜像

#### 2.4.2.1 Dockerfile

首先，在项目的根目录下，新建一个文本文件`.dockerignore`，写入下面的[内容](https://github.com/ruanyf/koa-demos/blob/master/.dockerignore)。

```
.git
node_modules
npm-debug.log
```

上面代码表示，这三个路径要排除，不要打包进入 image 文件。如果你没有路径要排除，这个文件可以不新建。

我们使用命令 **docker build** ， 从零开始来创建一个新的镜像。为此，我们需要创建一个 Dockerfile 文件，其中包含一组指令来告诉 Docker 如何构建我们的镜像。

每一个指令都会在镜像上创建一个新的层，每一个指令的前缀都必须是大写的。

```bash
$ touch Dockerfile 
$ gedit Dockerfile

FROM node:8.4 #该 image 文件继承官方的 node image，冒号表示标签，这里标签是8.4，即8.4版本的 node。
COPY . /app #将当前目录下的所有文件（除了.dockerignore排除的路径），都拷贝进入 image 文件的/app目录。
WORKDIR /app #指定接下来的工作路径为/app。
RUN npm install --registry=https://registry.npm.taobao.org #在/app目录下，运行npm install命令安装依赖。注意，安装后所有的依赖，都将打包进入 image 文件。
EXPOSE 3000 #将容器 3000 端口暴露出来， 允许外部连接这个端口。
CMD node demos/01.js
```

> Ubuntu 我自己使用 touch Dockerfile, 然后 gedit Dockfile 去编辑

#### 2.4.2.2 Docker build

有了 Dockerfile 文件以后，就可以使用`docker image build`命令创建 image 文件了。

```bash
$ docker image build -t koa-demo .
# 或者
$ docker image build -t koa-demo:0.0.1 .
#-t: IMAGE名字 .: Dockerfile 文件所在目录，可以指定 Dockerfile 的绝对路径
```

使用 Dockerfile 文件，通过 docker build 命令来构建一个镜像。

在 Desktop 下可以跑

```bash
$ docker build -t runoob/centos:6.7 .
#-t: 指定要创建的目标镜像名      .: Dockerfile 文件所在目录，可以指定Dockerfile 的绝对路径
```

> 可以跑起来，不过太大了中途终止掉

### 2.4.3 设置镜像标签

```bash
$ docker tag IMAGEID runoob/centos:dev
```

### 2.4.4 生成容器

```bash
$ docker container run -p 8000:3000 -it koa-demo /bin/bash
# 或者
$ docker container run -p 8000:3000 -it koa-demo:0.0.1 /bin/bash
#-p: 容器的 3000 端口映射到本机的 8000 端口
#-it: 容器的 Shell 映射到当前的 Shell，然后你在本机窗口输入的命令，就会传入容器。也就是交互式
#koa-demo: IMAGE名字，如果有标签，还需要提供标签，默认是 latest 标签
#/bin/bash: 容器启动以后，内部第一个执行的命令。这里是启动 Bash，保证用户可以使用 Shell。
```

可以使用`docker container run`命令的`--rm`参数，在容器终止运行后自动删除容器文件。

### 2.4.5 CMD命令

> 上一节的例子里面，容器启动以后，需要手动输入命令`node demos/01.js`。我们可以把这个命令写在 Dockerfile 里面，这样容器启动以后，这个命令就已经执行了，不用再手动输入了。

`CMD node demos/01.js`，它表示容器启动后自动执行`node demos/01.js`。

> 你可能会问，`RUN`命令与`CMD`命令的区别在哪里？简单说，`RUN`命令在 image 文件的构建阶段执行，执行结果都会打包进入 image 文件；`CMD`命令则是在容器启动后执行。另外，一个 Dockerfile 可以包含多个`RUN`命令，但是只能有一个`CMD`命令。
>
> 注意，指定了`CMD`命令以后，`docker container run`命令就不能附加命令了（比如前面的`/bin/bash`），否则它会覆盖`CMD`命令。现在，启动容器可以使用下面的命令。

```bash
$ docker container run --rm -p 8000:3000 -it koa-demo:0.0.1
```

# 3. 实例：hello world

首先，运行下面的命令，将 image 文件从仓库抓取到本地。

```bash
$ docker image pull library/hello-world
```

上面代码中，`docker image pull`是抓取 image 文件的命令。`library/hello-world`是 image 文件在仓库里面的位置，其中`library`是 image 文件所在的组，`hello-world`是 image 文件的名字。

由于 Docker 官方提供的 image 文件，都放在[library](https://hub.docker.com/r/library/)组里面，所以它的是默认组，可以省略。因此，上面的命令可以写成下面这样。

```bash
$ docker image pull hello-world
```

在本机看到这个 image 文件

```bash
$ docker image ls
```

运行这个 image 文件

```bash
$ docker container run hello-world
```

`docker container run`命令会从 image 文件，生成一个正在运行的容器实例。

注意，`docker container run`命令具有自动抓取 image 文件的功能。如果发现本地没有指定的 image 文件，就会从仓库自动抓取。因此，前面的`docker image pull`命令并不是必需的步骤。

> 如果运行成功，你会在屏幕上读到下面的输出。
>
> ```bash
> $ docker container run hello-world
> 
> Hello from Docker!
> This message shows that your installation appears to be working correctly.
> 
> ... ...
> ```

输出这段提示以后，`hello world`就会停止运行，容器自动终止。

有些容器不会自动终止，因为提供的是服务。比如，安装运行 Ubuntu 的 image，就可以在命令行体验 Ubuntu 系统。

> ```bash
> $ docker container run -it ubuntu bash	#-i交互式操作 -t 终端
> ```

对于那些不会自动终止的容器，必须使用[docker container kill](https://docs.docker.com/engine/reference/commandline/container_kill/) 命令手动终止。

> ```bash
> $ docker container kill [containID]
> ```

# 4. Container文件

**image 文件生成的容器实例，本身也是一个文件，称为容器文件。**也就是说，一旦容器生成，就会同时存在两个文件： image 文件和容器文件。而且关闭容器并不会删除容器文件，只是容器停止运行而已。

```bash
# 列出本机正在运行的容器
$ docker container ls
$ docker ps #应该是一样的效果

# 列出本机所有容器，包括终止运行的容器
$ docker container ls --all
$ docker ps -a
```

上面命令的输出结果之中，包括容器的 ID。很多地方都需要提供这个 ID，比如上一节终止容器运行的`docker container kill`命令。

## 4.1 删除容器

终止运行的容器文件，依然会占据硬盘空间，可以使用`docker container rm`命令删除。

```bash
$ docker container rm [containerID]
$ docker container prune #清理所有处于终止状态的容器。
```

运行上面的命令之后，再使用`docker container ls --all`命令，就会发现被删除的容器文件已经消失了。

## 4.2 停止容器

```bash
$ docker stop <容器 ID>
```

## 4.3 启动已停止的容器

使用 docker start 启动一个已停止的容器

```bash
$ docker start b750bbbcfd88	#ID
```

## 4.4 后台运行

在大部分的场景下，我们希望 docker 的服务是在后台运行的，我们可以过 **-d** 指定容器的运行模式。

```bash
$ docker container run -itd --name ubuntu-test ubuntu /bin/bash
```

**注：**加了 **-d** 参数默认不会进入容器，想要进入容器需要使用指令 **docker exec**（下面会介绍到）

### 4.4.1 进入容器

在使用 **-d** 参数时，容器启动后会进入后台。此时想要进入容器，可以通过以下指令进入：

- **docker attach**
- **docker exec**：推荐大家使用 docker exec 命令，因为此退出容器终端，不会导致容器的停止。

#### 4.4.1.1 exec 命令

```bash
$ docker exec -it containerID /bin/bash
```

## 4.5 导入导出容器

### 4.5.1 导出容器

如果要导出本地某个容器，可以使用 **docker export** 命令。

```bash
$ docker export 1e560fca3906 > ubuntu.tar
```

导出容器 1e560fca3906 快照到本地文件 ubuntu.tar

### 4.5.2 导入容器快照

可以使用 docker import 从容器快照文件中再导入为镜像 (Image)，以下实例将快照文件 ubuntu.tar 导入到镜像 test/ubuntu:v1

```bash
$ cat docker/ubuntu.tar | docker import - test/ubuntu:v1
```

此外，也可以通过指定 URL 或者某个目录来导入，例如

```bash
$ docker import http://example.com/exampleimag	e.tgz example/imagerepo
```

## 4.6 拷贝容器文件到本机

```bash
$ docker container cp [containID]:[/path/to/file] .
```

# Reference

https://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html	Docker 入门教程

https://www.runoob.com/docker/docker-container-connection.html	**Docker 教程**

