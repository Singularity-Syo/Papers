## Improving the Robustness of Graphs through Reinforcement Learning and Graph Neural Networks

作者: Victor-Alexandru Darvariu, Stephen Hailes, Mirco Musolesi

链接：https://arxiv.org/pdf/2001.11279.pdf

## Abstract

图可以用于表示和推理现实世界的各种系统，人们设计了许多度量来量化这些图的全局特征。先前的工作都着重于度量现有图的性质，而不是动态修改图来提升目标函数值。
文章提出了一种用于提升图鲁棒性的方法，基于图神经网络架构和深度强化学习，称之为 RNet-DQN。并且研究了该方法在提升图鲁棒性上的应用，这些应用和基础设施以及通信网络有关。文章使用两个目标函数来获得鲁棒性，并将函数值的变化作为奖励信号。实验展示了该方法能够学习到使得鲁棒性提升的加边策略，比起随机选择的策略要好很多，并且在一些例子上性能甚至超过了贪婪基线。更重要的是，该方法学习到的策略能够在不同的图上进行泛化，甚至比训练所用图的规模更大。这一点很重要，因为贪婪方法对于大型图来说计算代价过高，而 RNet-DQN 能够提供相当于它 $O(|V|^3)$ 的速度。

## Introduction
图 (Graph) 是一种数学抽象，它能用于建模各种系统，如基础设施、生物网络、社会结构等等。
分析网络的各种方法和工具都已经被开发出来，经常用于理解系统，这些方法有：
* 生成图族的数学模型 [Watts and Strogatz, [1998](https://www.nature.com/articles/30918); Barabasi and Albert, [1999](https://science.sciencemag.org/content/286/5439/509)]
* 节点中心性度量 [Bianchini et al., [2005](https://dl.acm.org/doi/10.1145/1052934.1052938)]
* 全局性质 [Newman, [2018](lack)]

研究网络结构的重点在于提出具有各种特性的网络模型，再抽象地研究这些网络模型的各种拓扑结构的统计特征。传统图论起源于七桥问题，在研究中采用的网络模型多是规则图，主要关注静态的，具有特殊结构的拓扑特征。后来由 Erdos 和 Renyi 将概率论引入图论，提出了随机网络或者叫随机图理论。随即图的意思是指任意两个节点之间是否建立一条边是按照一定的随机模型确定的，他们提出的一种随机网络被称为 ER 随机网络，因为它的度分布，也就是一个节点所连边数的概率分布为泊松分布，所以也叫泊松随即网络。后来随着计算机技术的发展，对现实超大规模网络研究发现了许多重要的统计规律。Watts 和 Strogatz 在 Nature 上发表了关于现实世界网络的小世界特性研究，提出了 WS小世界网络模型，而 Barabasi 和 Albert 在 Science 上发表了无标度网络的开创性研究并提出了 BA 无标度网络模型，也称为 BA 随机图。这个模型简单说来就是演化增长和优先连接，也就是说网络不断增长，而新加入的节点倾向于与那些重要的节点也就是拥有更多边的节点相连。这两篇工作揭示了许多形态各异的网络实际上具有相同的拓扑结构特征，并给出了理论模型来阐释生成机理。

其中有一个吸引了网络科学界及从业者的度量是 Robustness，来自于 ，在原文中是称为 Resilience，意思是 n. 它被定义为图经受失败，对关键节点的目标攻击和组合的承受能力。
网络结构决定网络功能，因此结构的改变会影响网络功能的表达，因此网络结构在遭受破坏后，网络结构和功能能否保持完整性的研究就具有相当重要的意义。这里引入一个概念——网络的鲁棒性，在 [Newman [2003](http://epubs.siam.org/doi/10.1137/S003614450342480)] 里是称为 Resilience，意思是快速恢复的能力; 适应力; 还原能力; 弹力，也就是说在遭受随机故障 (random failure) 或蓄意攻击 (intentional attack) 后，网络仍能保持一定的结构完整性和功能的能力。这里的随机故障是说任意节点或者边以相同的概率发生故障，或者说按照一定的比例删除边或者节点。而蓄意攻击则是指删除具有特定特性的节点和边，比如度数比较高的节点。可以看得出来前者主要是描述现实网络遭受的无法预测的故障，后者则是描述人为的针对性的定向破坏。鲁棒性的体现就在遭受破坏后，剩余节点之间能够以较高概率保持连通并且平均最短路径长度变化不大。比如说 BA 随机图遭受随机故障时能表现出一定的抗性但面对蓄意攻击则十分脆弱。

> Related to degree distributions is the property of resilience of networks to the removal of their vertices, which has been the subject of a good deal of attention in the literature. Most of the networks we have been considering rely for their function on their connectivity, i.e., the existence of paths leading between pairs of vertices. If vertices are removed from a network, the typical length of these paths will increase, and ultimately vertex pairs will become disconnected and communication between them through the network will become impossible. Networks vary in their level of resilience to such vertex removal. There are also a variety of different ways in which vertices can be removed and different networks show varying degrees of resilience to these also. For example, one could remove vertices at random from a network, or one could target some specific class of vertices, such as those with the highest degrees. Network resilience is of particular importance in epidemiology, where “removal” of vertices in a contact network might correspond for example to vaccination of individuals against a disease. Because vaccination not only prevents the vaccinated individuals from catching the disease but may also destroy paths between other individuals by which the disease might have spread, it can have a wider reaching effect than one might at first think, and careful consideration of the efficacy of different vaccination strategies could lead to substantial advantages for public health.

>与度分布相关的是网络对去除顶点的弹性特性，这一直是文献中广泛关注的主题。 我们一直在考虑的大多数网络的功能依赖于它们的连通性，也就是说，在c成对顶点之间存在路径。 如果从网络中移除顶点，这些路径的长度将增加，最终这对顶点将变得不连接，它们之间通过网络进行通信将变得不可能。 网络对这种顶点移除的弹性水平各不相同。 还有许多不同的方法可以去除顶点，不同的网络对这些方法也表现出不同程度的弹性。 例如，可以从网络中随机移除顶点，或者针对特定类型的顶点，例如那些具有最高度的顶点。 网络弹性在流行病学中特别重要，在这种情况下，接触网络中顶点的”移除”可能相当于个人对某种疾病的免疫接种。 由于接种疫苗不仅可以防止接种疫苗的个体感染疾病，而且还可能破坏其他个体之间可能传播疾病的路径，因此它可以产生比人们最初想象的更广泛的影响，仔细考虑不同疫苗接种策略的效力可以为公共卫生带来巨大的好处。

鲁棒性定义：
* [Cohen et al.,[2000](http://europepmc.org/article/MED/11082612)] 在变成多个连通分量前需要移除一定数量的节点 
* [Albert et al., [2000](https://www.nature.com/articles/35019019)] 最短路径变长
* [Beygelzimer et al., [2005](https://www.sciencedirect.com/science/article/pii/S0378437105003523)] 最大连通分量的大小减小

已经研究过：
* [Cohen et al., [2001](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.3682)]通信网络如英特网的鲁棒性
* [Cetinay et al., [2018](https://link.springer.com/article/10.1007/s41109-018-0089-9)] 基础设施网络如交通和能量分配的鲁棒性  

在攻击策略下的最优配置也被发现——比如，在联合的弹性目标函数对于两种攻击策略，最优网络有 bi-model 或者 tri-model 度分布：
* [Valente et al., [2004](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.92.118702)]

但是从零开始建立一个鲁棒的网络是不实际的，因为网络往往是根据特定的目的设计的。因此，先前的工作已经解决了修改现有网络来提升他们的鲁棒性的问题。
* [Beygelzimer et al. [2005](https://www.sciencedirect.com/science/article/pii/S0378437105003523)]  等人通过考虑基于随机或优先的（根据节点的度）加边或重连来解决这一问题。
* [Schneider et al., [2011](https://www.pnas.org/content/108/10/3838)]，作者们提出了一种贪婪的修改方案，基于随机选边和当弹性度量提升时交换。

这些策略虽然简单易懂但可能不会得到最佳的解决方案，也无法在具有不同特征和规模的网络中得到推广，当然更好的方案可能可以用过穷举搜索得到，但探索所有可能的拓扑结构的组合的复杂性以及计算度量的成本使得这种策略并不可行。

为了解决这个问题，文章抛出了一个问题：能否学习到可泛化的鲁棒性提升的策略？
文章将修改图的边的过程形式化为 MDP，以最大化全局目标函数值。在这个框架下，智能体被给定一个固定的修改预算，比如加边，而执行修改获得的奖励，则正比于目标函数的变化。特别地，文章将两种鲁棒性度量即随机故障和蓄意攻击作为目标函数。

强化学习适用于学习如何应用这些修改行为的策略，像 DQN 通过使用深层神经网络作为函数逼近器，在处理高维决策问题上获得了巨大的成功 [Mnih et al., [2015]()]，而且最近，图神经网络架构被提出能够处理图结构数据，而强化学习算法已经被用于解决 NP 难组合优化问题并获得巨大的成功.

更具体地说，文章研究了一个新的框架用于增强图的鲁棒性，受到 [Dai et al., 2018] 中的方法的启发，即对图神经网络分类器的对抗性攻击。
在这一工作中，作者们考虑改变一个网络以欺骗外部分类器，正如过去在图像分类一样，在这项工作中我们感兴趣的地方是改变网络结构以优化图本身的特点。

文章生成这是第一个通过深度强化学习算法来学习如何建立鲁棒图的问题的工作，因为它用的是 DQN 来解决建立鲁棒网络，所以命名为 RNet-DQN。
本文认为这项工作的贡献还有方法论，这个方法能够应用于基于不同的弹性定义或采用不同的图特性的目标函数的模型的各种问题，鲁棒性可以看作这方面的一个研究。

论文的其他部分如下：
* **Section 2** 将问题形式化为马尔可夫决策过程，并定义了鲁棒性度量
* **Section 3** 提供了基于函数逼近的 RL 方法的一个具体描述
* **Section 4** 描述了实验建立过程
* **Section 5** 讨论实验的结果
* **Section 6** 对现有方法进行了分析并给出了未来研究的可能途径
* **Section 7** 回顾并比较了关键工作，
* **Section 8** 总结

## 2 Modeling Graph Robustness for Reinforcement Learning

**马尔可夫决策过程 (Markov Decision Process)**

马尔可夫决策过程是决策过程的形式化，在这一过程中，决策者称为 Agent 即智能体，它和环境进行交互，发现自己处于一个状态 State $s \in \mathcal{S}$ 中，需要从可用的动作集中选择并采取一个动作 Action $a \in \mathcal{A}(s)$。对于每一次采取的动作，智能体会获得一个奖励 Reward $\mathcal{R}(s, a)$。在一段时间内，智能体不断地进行决策，其目的是让这一段决策过程能够获得尽量多的奖励。然后智能体发现状态发生了变化，出现了一个新的状态，这个变化的过程由转移模型 $\mathcal{P}$ 定义，每一个转移对应一个概率 $P(s' , a, s)$。

智能体与环境不断进行交互，交互的过程即轨迹 Trajectory：$S_0, A_0, R_1, S_1, A_1, R_2, \cdots, S_{T−1}, A_{T−1}, R_T$。
元组 $(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R}, \gamma)$ 定义了整个马尔可夫决策过程，其中 $\gamma$ 是折扣因子。
策略 $\pi(a|s)$，即状态上的动作分布，则定义了智能体的行为。当给定一个策略，状态动作值函数 $Q_{\pi}(s, a)$ 被定义为从状态 $s$ 开始采取动作 $a$ 之后根据策略 $\pi$ 执行所能获得的期望回报 Return。有几种方法能够迭代地得到最优状态动作值函数和策略，如广义策略迭代。

**Modeling Graph Construction**

$\mathbf{G}^{(N)}$ 是一个带有标签的、无向的、无权的、具有 $N$ 个节点的连接图，$\mathbf{G}^{(N,m)}$ 是它的一个子集，即具有 $m$ 条边的图。
定义目标函数 $\mathcal{F} : \mathbf{G}^{(N)} \rightarrow [0, 1]$，将图映射到 $[0, 1]$ 之间，$L \in \mathbb{N}$ 是修改预算。
给定一个初始图 $G_0 = (V, E_0)\in \mathbf{G}^{(N,m_0)}$，具有 $m_0$ 条边，目标是对这个图应用一系列的 $L$ 次加边操作使得结果图 $G_∗ = (V, E_∗)$ 满足下式：
$$
G_∗ = \argmax_{G'\in\mathbb{G}'}\mathcal{F}(G')
$$
其中 $\mathbb{G}'= \{G = (V, E) \in \mathbb{G}^{(N,m_0+L)}|E_0\subset E\}$。

这一过程可以看作一个序贯决策问题，智能体以提升每一个即时图为目标采取动作得到一个图序列 $G_0, G_1, \cdots, G_{L−1}$，这个任务被定义为情节性的，即每一次的轨迹都是有限长的。每一个 episode 最多处理 $L$ 步，智能体耗尽修改的预算，或者没有能够应用的动作，比如图已经全连接。

**图 1** 可视化了一个 episode：
![]()

下面将问题定义为 MDP：
1. 状态，$s_t$ 即 $G_t$，在 $t$ 时刻的图
2. 动作，$a_t$ 对应加边，特别的将加边操作写为 $add(v,u)$，即对带有标签 $v$，$u$ 的节点间加连边。
3. 转移，转移函数是确定性的，即一个状态执行动作之后只会转移到一个状态，即概率为 1。
4. 奖励，定义如下
   $$
   R_t=
   \begin{cases}
   \mathcal{F}(G_t)-\mathcal{F}(G_0)&\text{if}\quad t=L\\
   0&\text{otherwise}
   \end{cases}
   $$

**Definition of Objective Functions for Robustness**

本文主要关注的是图的鲁棒性，将两种攻击作为目标函数。
给定一个图 $G$，文章定义了一个 critical fraction $p(\xi)\in [0,1]$，意思是将图按照顺序 $\xi$ 移除节点使得图变成无连接图的最小节点集，这一比例越大，这个图可以说越鲁棒。移除的顺序对 $p(\xi)$ 来说有影响，对应两种攻击策略,前者是随即顺序，而后者则是按照节点的度数排列的，如下式。
$$
\forall v, u \in V,\xi_{targeted}(v) \leq \xi_{targeted}(u) \iff d^v \geq d^u
$$
分别定义目标函数 $\mathcal{F}$ 如下:
1. _Expected Critical Fraction to Random Removal_:
$$
\mathcal{F}_{random}(G) = \mathbb{E}_{\xi_{random}}[p(\xi_{random})]
$$
2. _Expected Critical Fraction to Targeted Removal_:
$$
\mathcal{F}_{targeted}(G) = \mathbb{E}_{\xi_{targeted}}[p(\xi_{targeted})]
$$

为了获得这些量的估计，我们生成 $R$ 个排列分别得到 $p(\xi^i)$，然后加权平均：
$$
\widehat{\mathcal{F}(G)} =\frac{\sum_{i=1}^Rp(\xi^i)}{R}
$$

## 3 Learning How to Build Robust Graphs with Function Approximation

以 **Section 2** 描述的问题模型为基础，讨论建立鲁棒图的方法的相关设计和应用，当然还需要有泛化能力。尽管这个形式化问题可以用表格化的强化学习方法来求解，但是状态的数量会迅速变得难以处理，比如二十个节点的带标签的无权连接图有接近 10^57 个。
因此，需要一种方法来考虑图的属性，这些属性具有标签不可知性、排列不变性，并且在相似的状态和操作之间进行泛化。图神经网络满足这些要求。

本文方法里采用基于 structure2vec(S2V) 的图表示，一种根据平均场推断在图模型的图神经网络架构, 给定一个输入图 $G=(V,E)$，每个节点 $v\in V$ 有特征向量 $x_v$，边 $(v,u)$ 也有特征向量 $w(v,u)$。该方法的目标是对于每一个节点 $v$ 产生一个特征向量 $\mu_v$ 能够捕获图的结构以及邻居之间的交互。几轮聚合邻居的特征并应用一个非线性激活函数 $F$，比如神经网络或者核函数。对于每一轮，网络同时应用如下更新，其中 $\mathcal{N}(v)$ 是节点的邻居：
$$
\mu_v^k=F(x_v,\{\mu^k_u\}_{u\in\mathcal{N}(v)},\{w(v,u)\}_{u\in \mathcal{N}(v)};\Theta)
$$

一旦获得了节点级别的嵌入，通过将这些节点级别的嵌入求和就得到了子图的排列不变性嵌入：
$$
\mu_{\mathcal{S}}=\sum_{v_i\in\mathcal{S}}\mu_{v_i}
$$

文章采用了 Q-learning [Watkins and Dayan, 1992]，即通过估计之前介绍过的状态动作值函数来得到一个确定性的策略，即根据估计的状态动值导出动作。

智能体和环境进行交互并根据下式更新估计：
$$
Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \max_{a'\in\mathcal{A}(s')}Q(s', a') − Q(s, a)]
$$

在高维的状态和动作空间的情况下，用神经网络来估计状态动作值函数的方法已经在各种领域，从游戏到连续控制都获得了成功 [Mnih et al., 2015; Lillicrap et al., 2016]。

本文使用了 DQN 算法，一种样本有效 sample-efficient 方法，通过使用经验池和一个每隔几步更新的目标网络对状态动作值函数估计 
文章选择的是 DQN 的一种变体即 Double DQN [Van Hasselt et al., 2016] 即使用两个不同的模型分别进行贪婪动作选择和状态动作值估计来解决标准 DQN 动作值的过高估计。
除此之外，文章用两步回报而不是单步的回报进行学习。

一个关键的问题是状态的表示，一种做法是通过计算子图的 S2V 嵌入来表示状态 $S_t$，这个子图由所有到时间 $t$ 增加的边连接的节点组成，并通过在时间 $t$ 连接节点 $v$ 和 $u$ 的嵌入来表示 $A_t=add (v，u)$ 的动作。但是这种动作的定义方式无法推广到大型图，因为每一步都要考虑 $O(|V|^2)$ 种动作。所以文章采用的方法是 [Dai et al., 2018] 所用的方法，将每一个动作分解为两步 $A_t^{(1)}$ 和 $A_t^(2)$，前者对应的是选择第一个节点，后者则是第二个，这样的话智能体每个时间步只需要考虑 $O(|V|)$ 数量的动作。

## 4 Experimental Setup

**学习环境**

文章构建了一个允许定义任意图目标函数 $\mathcal{F}$ 的学习环境，并为智能体提供一个标准化的接口。
我们对环境、智能体和实验 suite 的实现基于 PyTorch [ Paszke et al. ，2019]。

**Baselines**

我们比较了以下基线:
* **Random**：随机选择一个可用的动作。
* **Greedy**：在添加一条边的情况下，使用向前查找，并选择对 $\mathcal{F}$ 的估计值有最大提升的动作。

**图模型**

我们研究下列模型生成的图的性能:
* **Erdos-Renyi(ER)**：从{G}^{(N,m)}[Erdos Renyi，1960]中统一采样的图。文章使用 $m =\frac{20}{100}∗\frac{N*(N-1)}{2}$，它代表所有可能边的 20% 。

* **Barabasi-Albert(BA)** : 一种增长模型，其中 $n$ 节点优先连接到 $M$ 现有节点[ Barabasi 和 Albert, [1999](https://science.sciencemag.org/content/286/5439/509)]。
我们使用 $M=2$。

**实验过程**

我们使用上面的两种图模型生成一组训练图 $\mathbf{G}^{train}$，文章将顶点数设置为二十，加边操作即修改预算设置为所有可能边的 1%，据此来训练智能体。

算法在每一步对 $\mathbf{G}^{train}$ 中的所有图进行处理，
在训练期间，每经过 100 步就在不相交的验证集 $\mathbf{G}^ {validate}$ 上评估智能体的性能。然后在另一个不相交的测试集上评估 RNet-DQN 和两个 Baselines 的性能。
这三个图集的大小都是 100。

使用一个 Python 开发的图论与复杂网络建模工具 networkx[Hagberg et al., 2008] 来生成图。如果生成的候选图不连通，则拒绝该候选图，并生成另一个候选图，直到集合达到指定的基数。

为了估计目标函数值，文章还使用了一定数量的排列，取值等于顶点数。而对较大图的进行目标函数估计时，则考虑以下数量 $\{30,40,50,60,70,80,90,100\}$，并且根据 $|V|$ 的不同，对 $m, L, R$ 三个数字进行修改。

**RNet-DQN 参数**
Gtrain is divided into batches of size 50. 
Training proceeds for 100000 steps. 
We use a learning rate $\alpha = 0.001$ and a discount factor $\gamma=1$ since we are in the finite horizon case. 
We use a value of the exploration rate $\epsilon$ that we decay linearly from $\epsilon = 1$ to $\epsilon = 0.1$ for the first 50000 steps, then fix $\epsilon = 0.1$ for the rest of the training.
We use 8 latent variables and a hidden layer of size 32. 
The only hyperparameter we tune is the number of message passing rounds $K$, for $K\in {2, 3, 4, 5}$, selecting the agent with the best performance over $\mathbf{G}^{validate}$.

Training, hyperparameter optimization and evaluation are performed separately for each graph family and objective function $\mathcal{F}$. 
Since deep RL algorithms are notoriously dependent on parameter initializations and stochastic aspects of the environment [Henderson et al., 2018], we aggregate the results over 5 runs of the training-evaluation loop.
$\mathbf{G}^{train}$ 训练集划分为大小为 50 的批次，训练进行了十万步，学习率 $\alpha=0.001$ 和折扣因子 $\gamma=1$，这是因为 episode 是有限的。
探索率 $\epsilon$，在前 50000 步线性地从 $\epsilon=1$ 衰减到 $\epsilon=0.1$，然后在剩下的训练中固定 $\epsilon=0.1$。
使用 8 个隐变量和一个大小为 32 的隐藏层。
调优的唯一超参数是消息传递的轮数 $K$，对于 $K\in {2,3,4,5}$，选择在验证集上性能最好的智能体。


对每种图和目标函数分别进行训练、超参数优化和评估。由于深度强化学习算法是众所周知的依赖于参数初始化和环境的随机方面[ Henderson et al., 2018] ，文章聚合了训练-评估的 5 次运行的结果。

## 5 Results

在**表 1** 中，文章列出了实验评估的主要结果。
![]()

可以看到对于 $\mathcal{F}_{random}$, 由 RNet-DQN 学习到的策略在 ER 和 BA 情况下都优于贪婪方法，而对于 $\mathcal{F}_{targeted}$ 的提升效果则不是够好，但比随机的仍然要好很多。
这一点很重要，因为 RNet-DQN 学习策略能够在一个不相交的测试图集进行泛化，而贪婪方法则必须为这个集合中的每个图单独计算。
在所有类型的图和性能度量中，RNet-DQN 在统计学上明显优于随机方法。

**有效缩放到大图**

本文方法的最理想的特性是能够在小尺寸的图进行训练，在大尺寸的图上进行预测。在较小的图上进行学习，速度会快很多，因为构建学习表示的工作量减少了，可能的动作数也减少了，目标函数的评估也更快了。
因此，本文使用在 **Section 4** 中描述的大小为 $|V|=20$ 的图上训练的模型，并评估它们和基线在顶点数量高达 $|V|=100$ 的图上的性能 (由于计算开销，贪婪方法只能计算 $|V|=50$ 的图，见下一段)。

**图 2** 中展示了所得到的结果。
![]()

文章发现在考虑 BA 图时，RNet-DQN 的性能即两个目标函数下降相对较少。对于 ER 图，性能会迅速下降，对于大小为 $|V|=50$ 及以上的图，所学到的策略比随机策略执行得更差。这表明 ER 图的鲁棒性性质随着尺寸的增大而发生根本性的变化，而且 RNet-DQN 所学到的特征无法改进目标函数。

**计算成本**
文章将 RNet-DQN 的计算成本与基线进行比较，以便理解权衡。
根据经验，文章观察到贪婪基线计算顶点数超过 $|V|=50$ 的图时计算成本过高。文章还测量了不同智能体在执行评估时的平均决策时间。

**图 3** 展示了这些结果。
虽然不同智能体的速度不能直接比较，因为实现中涉及到不同的组成部分，我们注意到贪婪基线拓展能力更糟糕，因为它的决策时间急剧上升。

文章接着分析了该方法的计算复杂度。
假设为了计算单个图实例的目标函数，我们需要执行 $B$ 操作。 
在每一步都需要考虑 $O(|V|^2)$ 的操作。
对于贪婪智能体，在选择动作的一个步骤中涉及到 $O(B\times|V|^2)$ 的计算。相比之下，RNet-DQN 代理不需要在训练后明确地评估目标函数。
正如 **Section 3** 所解释的，MDP 分解为两步意味着每次加边的复杂度是 $O(2\times|V|)=O(|V|)$; 而在网络中执行前向传播以获得状态动作值估计值是一个 $O(|V| + |E|)$ 操作。
因此，RNet-DQN 代理在每个步骤执行 $O(|V|\times(|V|+|E|)$ 操作。
在最坏的情况下，当 $|E|=O(|V|^2)$ 时，这意味着 $O(|V|^3)$ 的复杂度。
在考虑两个目标函数的情况下，对于一个给定排列的每个 critical fraction 的计算，我们需要计算排列中每个要删除的 $O(|V|)$ 节点的连通分量数 ($O(|V|+|E|)$ 操作)。
由于我们使用了许多等于 $|V|$ 的排列，因此我们得到了 $O(|V|^2\times(|V|+|E|)$。
在最坏的情况下，这意味着计算图的目标函数具有复杂度 $B=O(|V|^4)$。
因此，贪婪的代理每步最多可以获取 $O(|V|^6)$，这对于规模大的图来说太昂贵了。
这种巨大的成本差异也被前面所展示的经验结果所捕捉。

值得注意的是，上面的分析没有考虑模型的训练成本，因为模型的运行时间取决于几个超参数以及问题，所以很难确定模型的复杂性。
尽管如此，训练还是包括了对每个训练图每一步评估一次目标函数。
因此，在需要对许多图进行预测或者模型可以缩放为计算目标函数开销很大的大图的情况下，这种方法是有优势的。

# 6 Limitations and Future Work

在这一部分中，文章讨论了所提方法的一些局限性以及这项工作可能的未来方向。
因为我们使用的是深度强化学习方法，所以这类算法有一些典型的注意事项。
文章观察到在相同的超参数下，网络权重的不同随机初始化的性能差异非常显著; 虽然图表报告了方法的平均性能，但有时会发现明显更好的解决方案。
文章相信，根据目标函数和图形结构量身定制的更智能的探索策略，能够带来在不同的初始化下更加一致的解决方案
我们也试验过每个 episode 增加更多数量的边缘，但是所涉及的噪音使得算法在这些场景中非常不稳定。

虽然 RNet-DQN 能够学习可泛化的策略，并且在计算复杂性方面被认为是有效的，但是与贪婪的解决方案相比，它必然会牺牲单个图实例的性能。
然而，这两种解决方案并不一定相互排斥: 基于模型的方法可以用来在贪婪搜索之前提供一个关于有希望的边的先验信息，减少必须要考虑的动作空间。
事实上，在一些真实世界的网络(例如，交通)中，有一个基本的几何图形，边的增加可能会受到一些实际问题的限制，例如对成对距离的约束。

所提框架的适用性并不局限于健壮性。
实际上，它支持定义在图上的任意目标函数。
在网络科学社区中经常使用的性质是可沟通性和效率 [Newman, 2018]。

此外，当目标函数的评估成本很高时，这种方法可能是有利的——这是涉及流量或流行病等网络模拟的动态过程的情况 [Barrat 等人，2008]。
在这项工作中只考虑了拓扑结构，而 GNN 框架还可以包含可用的允许节点和边缘特征。如果与目标函数相关，这些方法有可能改进性能。
在目前的工作中，我们只研究了加边作为可能的行动。
人们也可以解决消除图中边的任务——布雷斯悖论 [Braess, 1968]
就是增加边反而降低效率。
允许去除边和添加可以给重新布线，这将大大增加可能的图的空间，可以用这种方式构建。

# 7 Related Work

**网络健壮性**

网络对随机错误和目标攻击的适应能力首先由 Albert 等人讨论[2000](https://www.nature.com/articles/35019019) ，他们检验了平均最短路径距离作为被移除节点数量的函数。
通过对两个无标度通信网络的分析，他们发现这种网络对随机故障具有很好的鲁棒性
但很容易受到有针对性的攻击。
Holme 等人的一个更广泛的调查[2002]分析了几个现实世界网络以及一些通过合成模型生成的网络的健壮性。
研究了不同的基于度和中间集中度的攻击策略，发现通常重新计算节点删除后的集中度可以产生更有效的攻击策略。
已经获得了描述两种攻击策略下网络模型的分解阈值的各种分析结果[ Cohen 等，[2000](http://europepmc.org/article/MED/11082612); Cohen 等,
[2001](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.3682)].
正如前面所讨论的，一些工作已经考虑到了提高现有网络弹性的问题[ Beygelzimer et.al, [2005](https://www.sciencedirect.com/science/article/pii/S0378437105003523); Schneider 等人，[2011](https://www.pnas.org/content/108/10/3838);
等人，2013]。

**图神经网络**

神经网络架构能够不仅处理欧几里德，而且与流形和图形数据近年来已经发展[ Bronstein 等人，2017] ，并应用于各种问题，其中他们的能力表示结构化，关系信息可以利用[ Battaglia 等人，2018]。
研究团体已经使用现代的递归神经网络结构来处理 np 难图问题，如最小顶点覆盖和旅行推销员问题; 已经发现了近似的解决方案来解决组合优化问题，方法是将它们定义为监督式学习问题[ Vinyals 等人，2015]或者 RL [ Bello 等人，2016]。
结合 GNNs 和 RL 算法已经产生了模型能够解决几个图优化问题与相同的架构，同时推广到图大于那些在培训中使用的数量级[ Khalil 等人，2017]。
然而，具体的问题并不涉及图本身的动态改进。
最近，通过添加边来欺骗图或节点级分类器的方式来修改图的问题被 Dai 等人研究[2018]。
事实上，在这项工作中，作者们对研究图的性质并不感兴趣，而是寻找方法来伪装网络的变化，以欺骗分类器，类似于基于深度神经网络的图像分类器的对抗性样本。
我们使用[ Dai 等人，2018]中讨论的方法，称为 RL-S2V，作为我们工作的起点。

**改进图的目标函数**

构建一个具有某些理想属性的图的问题可能是在设计神经网络体系结构时首先认识到的，这种体系结构的性能是最大化的[ Harp et al. ，1990]。
最近，出现了使用 RL [ Zoph and Le，2017]或进化算法[ Liu et al. ，2018]来发现可以在几个计算机视觉基准上提供最先进性能的体系结构的方法。

# 8 Conclusions

在这项工作中，我们通过学习如何以一种有效的方式添加边，解决了在存在随机和有针对性的节点删除的情况下提高图的鲁棒性的问题。
我们已经把提高任意全局目标函数的值的问题建模为一个马可夫决策过程，并且我们已经使用强化学习和图形神经网络结构来处理它。
据我们所知，这是第一个解决如何使用深度强化学习构建健壮图表的问题的工作。
我们评估了我们的解决方案 RNet-DQN，考虑了通过 Erdos-Renyi 和 Barabasi-Albert 模型生成的图。
我们的实验结果表明，这种方法可以执行显着优于随机添加，在某些情况下，超过了贪婪的基线。
这种新的方法有几个优点: 学习策略可以转移到样本外图以及比训练期间(无标度网络)使用的图大的图，而不需要在训练后估计目标函数。
这一点很重要，因为天真的贪婪解决方案对于计算大型图来说可能过于昂贵。
相反，我们的方法是高度可泛化的，在评估时提供了 $O(|V|^3)$ 的速度。
最后，值得注意的是，我们的贡献也是方法论的。
基于弹性的不同定义，或者考虑表示图的其他特征的完全不同的目标函数，本文提出的方法可以应用于其他问题。
学习如何提高图的鲁棒性可以作为本文提出的一般方法的一个案例研究。

# Reference

[Albert et al., 2000] Reka Albert, Hawoong Jeong, and  Albert-Laszlo Barabasi. Error and attack tolerance of complex networks. Nature, 406(6794):378–382, 2000.

[Barabasi and Albert, 1999] Albert-Laszlo Barabasi and  Reka Albert. Emergence of Scaling in Random Networks.  Science, 286(5439):509–512, 1999.

[Barrat et al., 2008] Alain Barrat, Marc Barthelemy, and Alessandro Vespignani. Dynamical Processes on Complex Networks. Cambridge University Press, 2008.

[Battaglia et al., 2018] Peter W. Battaglia, Jessica B. Hamrick, Victor Bapst, Alvaro S  anchez-Gonzalez, et al. Rela  tional inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261, 2018.

[Bello et al., 2016] Irwan Bello, Hieu Pham, Quoc V. Le, Mohammad Norouzi, and Samy Bengio. Neural Combinatorial Optimization with Reinforcement Learning. arXiv:1611.09940 [cs, stat], 2016.

[Beygelzimer et al., 2005] Alina Beygelzimer, Geoffrey Grinstein, Ralph Linsker, and Irina Rish. Improving Network Robustness by Edge Modification. Physica A, 357:593–612, 2005.

[Bianchini et al., 2005] Monica Bianchini, Marco Gori, and Franco Scarselli. Inside PageRank. ACM Trans. Internet Technol., 5(1):92–128, February 2005.

[Braess, 1968] Dietrich Braess. Uber ein paradoxon aus der ¨ verkehrsplanung. Unternehmensforschung, 12(1):258– 268, 1968.

[Bronstein et al., 2017] Michael M. Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Vandergheynst. Geometric Deep Learning: Going beyond Euclidean data. IEEE Signal Processing Magazine, 34(4):18–42, July 2017.

[Cetinay et al., 2018] Hale Cetinay, Karel Devriendt, and Piet Van Mieghem. Nodal vulnerability to targeted attacks in power grids. Applied Network Science, 3(1):34, 2018.

[Cohen et al., 2000] Reuven Cohen, Keren Erez, Daniel ben Avraham, and Shlomo Havlin. Resilience of the Internet to Random Breakdowns. Physical Review Letters, 85(21):4626–4628, 2000.

[Cohen et al., 2001] Reuven Cohen, Keren Erez, Daniel ben Avraham, and Shlomo Havlin. Breakdown of the Internet under Intentional Attack. Physical Review Letters, 86(16):3682–3685, 2001.

[Dai et al., 2016] Hanjun Dai, Bo Dai, and Le Song. Discriminative embeddings of latent variable models for structured data. In ICML, 2016.

[Dai et al., 2018] Hanjun Dai, Hui Li, Tian Tian, Xin Huang, Lin Wang, Jun Zhu, and Le Song. Adversarial attack on graph structured data. In ICML, 2018.

[Erdos and Renyi, 1960] Paul Erdos and Alfred Renyi. On  the evolution of random graphs. Publ. Math. Inst. Hung. Acad. Sci, 5(1):17–60, 1960.

[Hagberg et al., 2008] Aric Hagberg, Pieter Swart, and Daniel S. Chult. Exploring network structure, dynamics, and function using networkx. In SciPy, 2008.

[Harp et al., 1990] Steven A. Harp, Tariq Samad, and Aloke Guha. Designing application-specific neural networks using the genetic algorithm. In NeurIPS, 1990.

[Henderson et al., 2018] Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, et al. Deep reinforcement learning that matters. In AAAI, 2018.

[Holme et al., 2002] Petter Holme, Beom Jun Kim, Chang No Yoon, and Seung Kee Han. Attack vulnerability of complex networks. Physical Review E, 65(5), 2002.

[Khalil et al., 2017] Elias Khalil, Hanjun Dai, Yuyu Zhang, Bistra Dilkina, and Le Song. Learning combinatorial optimization algorithms over graphs. In NeurIPS, 2017.

[Lillicrap et al., 2016] Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, et al. Continuous control with deep reinforcement learning. In ICLR, 2016.

[Liu et al., 2018] Hanxiao Liu, Karen Simonyan, Oriol Vinyals, Chrisantha Fernando, and Koray Kavukcuoglu. Hierarchical representations for efficient architecture search. In ICLR, 2018.

[Mnih et al., 2015] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

[Newman, 2003] M. E. J. Newman. The Structure and Function of Complex Networks. SIAM Review, 45(2):189–190, 2003.

[Newman, 2018] M. E. J. Newman. Networks. Oxford University Press, 2018.

[Paszke et al., 2019] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, et al. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019.

[Schneider et al., 2011] Christian M. Schneider, Andre A.  Moreira, Jose S. Andrade, Shlomo Havlin, and Hans J.  Herrmann. Mitigation of malicious attacks on networks. PNAS, 108(10):3838–3841, 2011

[Schneider et al., 2013] Christian M. Schneider, Nuri Yazdani, Nuno A. M. Araujo, Shlomo Havlin, and Hans J.  Herrmann. Towards designing robust coupled networks. Nature Scientific Reports, 3(1), December 2013.

[Valente et al., 2004] Andre X. C. N. Valente, Abhijit Sarkar,  and Howard A. Stone. Two-Peak and Three-Peak Optimal Complex Networks. Physical Review Letters, 92(11), 2004.

[Van Hasselt et al., 2016] Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-learning. In AAAI, 2016.

[Vinyals et al., 2015] Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer Networks. In NeurIPS, 2015.

[Watkins and Dayan, 1992] Christopher J. C. H. Watkins and Peter Dayan. Q-learning. Machine Learning, 8(3-4):279– 292, 1992.

[Watts and Strogatz, 1998] Duncan J. Watts and Steven H. Strogatz. Collective dynamics of ‘small-world’ networks. Nature, 393(6684):440, June 1998.

[Zoph and Le, 2017] Barret Zoph and Quoc V. Le. Neural Architecture Search with Reinforcement Learning. In ICLR, 2017.
