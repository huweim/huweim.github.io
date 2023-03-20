---
title: "Tips for a New Computer Architecture PhD Student (By Swapnil Haria)"
date: 2022-08-27T23:57:17+08:00
lastmod: 2022-08-27T23:57:17+08:00
draft: false
author: "Cory"
tags: ["Tips"]
categories: ["科研"]
---

原文网址：https://www.sigarch.org/tips-for-a-new-computer-architecture-phd-student/

作者在 2019 年是 UWM 5 年级 phd，导师是 Mark Hill，毕业后去了 Google 做一名软件开发工程师。

原文

I have been fortunate enough to have many helpful senior students and two wonderful advisors to learn from throughout my PhD. Now as a senior PhD student myself, I have identified some lessons that proved most valuable as well as common mistakes made by younger graduate students. While these may be common knowledge, I am documenting these tips here to pass on some of our collectively learned lessons to future graduate students.

**1. Learn to read a paper efficiently.**
Fifty years of architecture research has produced a venerable but sizeable collection of research literature, growing by almost 200+ papers a year just from the top four conferences. To make good progress as we wade through the ocean of related work, it is important to read papers efficiently. [Here](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf) is a useful three-pass technique for reading papers efficiently. For me, reading a paper involves finding an answer to these questions (inspired by [grant writing tips](http://pages.cs.wisc.edu/~markhill/grant-tips.html)):

- What is the problem that the paper is trying to solve?
- Why is the problem relevant?
- What is the insight that drives the solution proposed by the paper?
- What are the trade-offs of the proposed solution?
- How have the authors evaluated the solution?
- What is one good trait in the paper (problem, solution, evaluation, presentation)?

> 甚至可以把这些问题贴在墙边，读论文的时候才能反复地去思考问题的答案，提升效率。

**2. Know when to stop reading papers.**
Whenever one begins a new project, it is easy to fall into the trap of endlessly reading paper after paper of related work. While surveying related literature is essential, it tends to yield diminishing returns after a certain point.  it is equally important to get one’s feet wet and actually start working on a project. There will be gaps in one’s knowledge but any critical gaps can be filled when that information is actually needed. After all, a graduate student is expected to be an information producer and not just a consumer.

> 这个其实很难。对我自己来说，读了一段时间我就会想去做一些实验，一方面也是一些调剂，一种做同一件事情总会觉得疲劳。

**3. Join or start a reading group.**
One way of being social is to participate in a reading group, which is a group of students that meet regularly to discuss an interesting paper (old or new) related to a particular area (e.g., architecture). In each meeting, one of the members is responsible for leading the discussion while the others are expected to have read the paper so as to contribute to the discussion. Through reading groups, one is forced to read papers outside of one’s particular niche. Ph.D. graduates often end up working in an area outside of the focus of their thesis. Thus, being exposed to a wide variety of topics is extremely useful. Group discussions also help bring out multiple perspectives into the same paper.

> 这个是一个很有意思的建议，不过推动这件事情通常需要一些有影响力的 host，或者是导师组织的，不知道美国是否会有比较多的自发组织的 reading group

**4. Use appropriate evaluation methodologies.**
There is no one size fits all when it comes to [evaluation methodologies](https://www.morganclaypool.com/doi/abs/10.2200/S00273ED1V01Y201006CAC010). Always select an appropriate methodology by taking into account development time, simulation speed and accuracy of results. There are three broad types of evaluation methodologies. Analytical modelling involves building mathematical models of the area of interest. At early stages of a project, it may be more useful to build simple mathematical models to get a broad sense of the efficacy of a research idea. Popular analytical models include the Roofline Model, Little’s law, Bottleneck analysis and others. Trace-based simulation involves using simulators that read in instruction or memory accesses sequences (traces) to provide reasonable estimates of runtime. Finally, execution-based simulators model hardware behavior at a cycle granularity. While such simulators offer the most accurate results, they also require the most development time as well as simulation time. As such, they may be overkill for simpler experiments. Hence, it is good to be familiar with different evaluation methodologies and use the right one at the right time.

> paper 看多了之后，基本能知道一些常用的分析方法

**5. Work on real systems.**
Although simulators are becoming more sophisticated, there is no substitute for working on real systems. This involves prototyping ideas in the linux kernel, modifying device drivers, building [FPGA-based systems](https://rise.cs.berkeley.edu/projects/firesim/) or [taping out chips](https://mshahrad.github.io/openpiton-asplos16.html). Looking at real systems gives us an idea of the complexity and robustness of industrial-strength solutions. This also forces us to confront the practical limitations of our ideas. The rise of open-source hardware along with existing open-source software has made it easy to work on real-world [operating systems](https://www.linux.org/), [compilers](https://llvm.org/), [CPUs](https://riscv.org/) and [GPUs](http://miaowgpu.org/).

> 这一点是我非常赞同的，simulation work 是一件非常危险的事情，有条件的话一定要转向真实的系统

**6. Spend time improving your programming skills.**
New computer architecture graduates may not already be good software engineers as computer architecture attracts students with backgrounds in hardware as well. However, PhD-level architecture research typically involves writing a lot of code as part of simulators, operating systems or benchmarks. Hence, it is worth spending some time early in the graduate lifecycle to improve one’s programming skills. Here are some handy resources:
[Data Structures](http://cs-www.cs.yale.edu/homes/aspnes/classes/223/notes.pdf), [Complexity Theory](http://bigocheatsheet.com/), [Design Patterns](http://pages.cs.wisc.edu/~swapnilh/resources/design-pattern-scard.pdf), [Code Style](http://google.github.io/styleguide/), [Git/Mercurial](http://rogerdudler.github.io/git-guide/), [Pointers](http://cslibrary.stanford.edu/102/PointersAndMemory.pdf).

> 体系结构的学生需要同时具备硬件和软件背景，至少他们必须会使用模拟器，熟悉操作系统，各种 benchmark

**7. Use microbenchmarks wisely.**
Microbenchmarks are simple and self-contained programs written to achieve a specific outcome. Microbenchmarks are immensely useful for preliminary exploration as well as sanity checking. An example is a program that can deterministically generate  ~100K TLB misses on every execution. Our example microbenchmark can validate a TLB simulator by comparing simulator-reported misses with microbenchmark-generated misses. Being simpler and more easily understood than full-fledged benchmarks, they can be used for initial evaluation to get results that demonstrate general trends. However, microbenchmarks should not be relied upon for any meaningful evaluation, particularly they should not be used as a proxy for real applications.

> 目前自己用的比较少，但这个绝对是体系结构领域非常重要的东西

**8. Create talks/posters/paper drafts to get early feedback.**
Once we start plugging away at implementation work, we sometimes lose track of the big research picture and instead simply generate a lot of experimental results. It is important to analyze these results in the context of the overall project. Then, crafting a talk or poster even with little to no results forces us to present our ideas in actual words and images. Different presentation media make us think about our research differently and thus refine our story. By presenting our posters and talks to other researchers, we get timely feedback that can help set the direction of our project.

> 2022-08-27 09:53:50，今天就是 poster talk 的 DDL，只能说自己确实比较拖延，正常来说可能应该找别人讨论一下，模拟一下。确实，沉浸在实验中时，容易丢失 high level 的视野，只能不断提醒自己不要忘记最初的 motivation 和故事的框架

**9. Care for your mental and physical well-being and maintain a support system.**
The typical [PhD lifetime](http://phdcomics.com/comics/archive.php?comicid=125) is long years of hard work with a few triumphant occasions such as paper acceptances mixed in. Remember that graduate school is a marathon and not a sprint. Hence, it is important to care for one’s overall health by getting enough sleep, having a healthy diet and spending time on exercise and hobbies. More importantly, graduate school is the first time many of us struggle academically or feel unproductive which may lead to [mental health issues](https://www.theatlantic.com/education/archive/2018/11/anxiety-depression-mental-health-graduate-school/576769/). Just remember that you are not alone in facing these issues and it is okay to ask for help. Most schools offer free or subsidized [mental health resources](https://www.uhs.wisc.edu/mental-health/). Furthermore, try to build and subsequently maintain a strong support structure for yourself by nurturing personal relationships with family and friends. Be social with your peers and almost never say no to going out for lunch, attending practice talks and other such activities.

> 毫无疑问，保持身体健康和心理健康非常重要

**Acknowledgements**
I thank Prof. Mark Hill, Lena Olson and Prof. Jason Lowe-Power for help improving this document. For teaching me these lessons in the first place, I am grateful to Prof. Mark Hill, Prof. Michael Swift, Prof. David Wood, Lena Olson, Jason Lowe-Power, Nilay Vaish, Jayneel Gandhi, Hongil Yoon, Rathijit Sen, Gokul Ravi, Marc Orr, Muhammad Shoaib, Joel Hestness, Prof. Tony Nowatzki, Newsha Ardalani, Vijay Thiruvengadam, Vinay Gangadhar and many others whom I encountered in graduate school.

**About the author:** [Swapnil Haria](http://pages.cs.wisc.edu/~swapnilh/) is a 5th-year PhD Candidate at the University of Wisconsin-Madison, advised by Mark Hill and Michael Swift. His research is focused on hardware and software support for persistent memory.

**Disclaimer:** *These posts are written by individual contributors to share their thoughts on the Computer Architecture Today blog for the benefit of the community. Any views or opinions represented in this blog are per*