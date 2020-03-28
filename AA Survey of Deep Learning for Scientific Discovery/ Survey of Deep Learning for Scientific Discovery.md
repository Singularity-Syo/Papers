# A Survey of Deep Learning for Scientific Discovery
## 中英双语版
## Authors: Maithra Raghu, Eric Schmidt
## 原文地址：https://arxiv.org/abs/2003.11755v1

## Abstract
Over the past few years, we have seen fundamental breakthroughs in core problems in machine learning, largely driven by advances in deep neural networks. 
At the same time, the amount of data collected in a wide array of scientific domains is dramatically increasing in both size and complexity. 
Taken together, this suggests many exciting opportunities for deep learning applications in scientific settings. 
But a significant challenge to this is simply knowing where to start. 
The sheer breadth and diversity of different deep learning techniques makes it difficult to determine what scientific problems might be most amenable to these methods, or which specific combination of methods might offer the most promising first approach.
In this survey, we focus on addressing this central issue, providing an overview of many widely used deep learning models, spanning visual, sequential and graph structured data, associated tasks and different training methods, along with techniques to use deep learning with less data and better interpret these complex models — two central considerations for many scientific use cases. 
We also include overviews of the full design process, implementation tips, and links to a plethora of tutorials, research summaries and open-sourced deep learning pipelines and pretrained models, developed by the community. 
We hope that this survey will help accelerate the use of deep learning across different scientific domains.

### 摘要
在过去的几年里，我们已经看到在机器学习的核心问题上取得了根本性的突破，这在很大程度上是由深层神经网络的进步推动的。
与此同时，在广泛的科学领域收集的数据量在规模和复杂性方面都在急剧增加。
综上所述，这为深度学习在科学环境中的应用提供了许多令人兴奋的机会。
但是一个重大的挑战是知道从哪里开始。
不同深度学习技术的广度和多样性使得很难确定哪些科学问题可能最适合这些方法，或者哪些具体的方法组合可能提供最有希望的第一种方法。
在这次调查中，我们着重于解决这个核心问题，提供了许多广泛使用的深度学习模型的概述，包括视觉的、序列的和图结构化数据、相关任务和不同训练方法，以及使用较少数据的深度学习和更好地解释这些复杂模型的技术————这是许多科学用例的两个主要考虑因素。
我们还包括整个设计过程的概述、实现技巧，以及由社区开发的大量教程、研究总结、开源的深度学习管道和预训练模型的链接。 
我们希望这项调查将有助于加速深度学习在不同科学领域的应用。

## 1 Introduction
The past few years have witnessed extraordinary advances in machine learning using deep neural networks.
Driven by the rapid increase in available data and computational resources, these neural network models and algorithms have seen remarkable developments, and are a staple technique in tackling fundamental tasks ranging from speech recognition [70, 167], to complex tasks in computer vision such as image classification, (instance) segmentation, action recognition [117, 78, 240], and central problems in natural language, including question answering, machine translation and summarization [186, 172, 233, 197]. 
Many of these fundamental tasks (with appropriate reformulation) are relevant to a much broader array of domains, and in particular have tremendous potential in aiding the investigation of central scientific questions.
However, a significant obstacle in beginning to use deep learning is simply knowing where to start. 
The vast research literature, coupled with the enormous number of underlying models, tasks and training methods makes it very difficult to identify which techniques might be most appropriate to try, or the best way to start implementing them. 
The goal of this survey is to help address this central challenge. 
In particular, it has the following attributes:
* The survey overviews a highly diverse set of deep learning concepts, from deep neural network models for varied data modalities (CNNs for visual data, graph neural networks, RNNs and Transformers for sequential data) to the many different key tasks (image segmentation, super-resolution, sequence to sequence mappings and many others) to the multiple ways of training deep learning systems.
* But the explanation of these techniques is relatively high level and concise, to ensure the core ideas are accessible to a broad audience, and so that the entire survey can be read end to end easily.
* From the perspective of aiding scientific applications, the survey describes in detail (i) methods to use deep learning with less data (self-supervision, semi-supervised learning, and others) and (ii) techniques for interpretability and representation analysis (for going beyond predictive tasks). 
  These are two exciting and rapidly developing research areas, and are also of particular significance to possible scientific use cases.
* The survey also focuses on helping quickly ramp up implementation, and in addition to overviews of the entire deep learning design process and a section on implementation tips (Section 9), the survey has a plethora of open-sourced code, research summaries and tutorial references developed by the community throughout the text, including a full section (Section 3) dedicated to this.

Who is this survey for? 
We hope this survey will be especially helpful for those with a basic understanding of machine learning, interested in (i) getting a comprehensive but accessible overview of many fundamental deep learning concepts and (ii) references and guidance in helping ramp up implementation.
Beyond the core areas of deep learning, the survey focuses on methods to develop deep learning systems with less data, and techniques for interpreting these models, which we hope will be of particular use for those interested in applying these techniques in scientific problems. However, these topics and many others presented, along with the many code/tutorial/paper references may be helpful to anyone looking to learn about and implement deep learning.

### 译文
在过去的几年里，深层神经网络在机器学习方面取得了非凡的进展。
在可用数据和计算资源快速增长的驱动下，这些神经网络模型和算法取得了显著的发展，是处理从语音识别 [70,167] 到计算机视觉中的复杂任务，如图像分类、(实例) 分割、动作识别 [117,78,240] 以及自然语言中的核心问题，包括问答、机器翻译和摘要 [186,172,233,197] 等基本任务的主要技术。
许多这些基本任务 (经过适当的重新制定) 都与更广泛的领域相关，特别是在帮助研究核心科学问题方面具有巨大潜力。
然而，开始使用深度学习的一个重大障碍是知道从哪里开始。
大量的研究文献，加上大量的基础模型、任务和培训方法，使得很难确定哪些技术可能是最适合尝试的，或者开始实施它们的最佳方式。
这项调查的目的是帮助解决这个核心挑战。
特别地，它具有以下特点:
* 调查概述了一系列高度多样化的深度学习概念，从用于不同数据模式的深度神经网络模型 (用于视觉数据的 CNNs、图形神经网络、用于序列数据的 RNNs 和 Transformers) 到许多不同的关键任务 (图像分割、超分辨率、序列到序列映射和许多其他) ，再到训练深度学习系统的多种方法。
* 但这些技巧的解释相对较高层次和简洁，以确保核心概念能够被广大受众理解，从而使整个调查能够轻松地从头到尾阅读。
* 从协助科学应用的角度出发，调查详细描述了
  （一） 使用较少数据的深度学习方法 (自我监督、半监督学习和其他)
  （二） 解释和表征分析技术 (超越预测任务)。
  这是两个令人兴奋和快速发展的研究领域，对于可能的科学用例也具有特别重要的意义。
* 调查亦着重协助快速加强实施，除了概述整个深度学习的设计过程和一个关于实施技巧的章节 (第9章) 外，调查还包含了大量由社区开发的开源代码、研究摘要和教程参考资料，包括一个完整的章节 (第3章) 专门介绍这方面的内容。

我们希望这个调查有助于那些对机器学习有基本了解，对（一）获得一个全面的，但易于理解的许多基本深度学习概念的概述和（二）帮助加快实施的参考和指南有兴趣的人。
除了深度学习的核心领域之外，这次调查的重点是用较少的数据开发深度学习系统的方法，以及解释这些模型的技术，我们希望这些技术对那些有兴趣将这些技术应用于科学问题的人有所帮助。 当然这些主题和其他的主题，以及许多代码/教程/论文参考可能有助于每个人学习和实现深度学习。
