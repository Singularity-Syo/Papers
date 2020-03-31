## 基本信息

文章：Improving the Robustness of Graphs through Reinforcement Learning and Graph Neural Networks

作者: Victor-Alexandru Darvariu, Stephen Hailes, Mirco Musolesi

链接：https://arxiv.org/pdf/2001.11279.pdf

## 摘要

图可以用于表示和推理现实世界的各种系统，人们设计了许多度量来量化这些图的全局特征。
先前的工作都着重于度量现有图的性质，而不是动态修改图来提升目标函数值。
本文提出了一种用于提升图鲁棒性的方法，基于图神经网络架构和深度强化学习，称之为 RNet-DQN
文章研究了该方法在提升图鲁棒性上的应用，这些应用和基础设施以及通信网络有关。
我们用两个目标函数来获得鲁棒性，并将他们值的变化作为奖励信号。
我们的实验展示了我们的方法能够学习到使得鲁棒性提升的加边策略，比起随机策略要好很多，并且在一些例子上性能甚至超过了贪婪基线。
更重要的是，学习到的策略能够在不同的图上进行泛化，甚至比训练所用图的规模更大。这一点很重要，因为自然的贪婪方法对于大型图来说计算代价过高，而我们的方法能够提供相当于它 $O(|V|^3)$ 的速度

(待修改)
## 介绍
首先介绍图，图是一种数学抽象，它能用于建模各种系统，从基础设施、生物网络到社会结构。
分析网络的各种方法和工具都已经被开发出来，经常用于理解系统，这些方法有：
* 生成图家族的数学模型 [Watts and Strogatz, [1998](); Barabasi and Albert, [1999]()]
* 节点中心性度量 [Bianchini et al., [2005]()]
* 全局性质 [Newman, [2018]()]

**问题：什么是鲁棒性？**

其中有一个吸引了网络科学界及从业者的度量是鲁棒性，来自于 [Newman [2003]()]，在原文中是称为 resilience，被定义为图经受随机失败，对关键节点的目标攻击和组合的能力
在这些攻击策略下，一个网络被认为是鲁棒的即节点集的一个重要部分在被变成多个连接部分时必须被移除， [Cohen et al.,[2000]()] 他的直径会增加[Albert et al., [2000]()]
或者它最大连接组件的大小diminishes [Beygelzimer et al., [2005]()]
先前的工作已经研究过通信网络如英特网的鲁棒性[Cohen et al., [2001]()]，和基础设施网络如交通和能量分配  [Cetinay et al., [2018]()]
在攻击策略下的最优配置也被发现——比如，在联合的resilience目标函数对于两种攻击策略，最优网络有bimodel或者trimodel 度分布 [Valente et al., [2004]()]

但是从零开始建立一个鲁棒的网络是不实际的，因为网络往往是根据特定的目的设计的

**问题：先前的方法做了什么**

因此，先前的工作已经解决了修改现有网络来提升他们的鲁棒性的问题
Beygelzimer et al. [2005]()  等人通过考虑基于随机或preferential（根据节点的度）加边或重连来解决这一问题
在[Schneider et al., [2011]()]，作者们提出了一种贪婪的修改方案，基于随机选边和当弹性度量提升时交换
这些策略虽然简单易懂但可能不会得到最佳的解决方案，也无法在具有不同特征和规模的网络中得到推广
当然更好的方案可能可以用过穷举搜索得到，但探索所有可能的拓扑结构的组合的复杂性以及计算度量的成本使得这种策略并不可行

为了解决这个问题，我们抛出了一个问题，能否学习到可泛化的鲁棒性提升的策略
我们将修改图的边的过程形式化为 MDP，以最大化全局目标函数值。
在这个框架下，智能体被给一个固定的关于用于修改图的budget，比如加边，获得奖励，正比从目标函数得到的提升。
特别地，我们将两种鲁棒性度量即随机和针对性攻击作为目标函数。
强化学习适用于学习应用这些修改的策略
方法如DQN通过使用深层神经网络作为函数逼近器已经在处理高维决策问题上获得了巨大的成功[Mnih et al., 2015]
最近，图神经网络架构被提出能够处理图结构数据，强化学习算法已经被用于解决NP难组合优化问题并获得巨大的成功
更具体地说，我们研究了一个新的框架用于增强图的鲁棒性，从首次出现在 2018中的方法来时，对图神将网络分类器的对抗性攻击
在这一工作中，作者们考虑改变一个网络以欺骗外部分类器，正如过去在图像分类一样
在这项工作中我们感兴趣的地方是改变网络结构以优化图本身的特点。

据我们所知，这是第一个通过DRL学习如何建立鲁棒图的问题的工作
因为它用的是DQN来解决建立鲁棒网络，所以命名为 RNet-DQN
本文华认为这项工作的贡献还有方法论
这个方法能够应用于基于不同的弹性定义或采用不同的图特性的目标函数的模型的各种问题
鲁棒性可以看作这方面的一个研究

论文的其他部分如下：
第二节形式化问题为MDP，并定义了鲁棒性度量
第三节提供了关于函数逼近的RL方法的一个状态表示和动作描述
第四节描述了实验建立并在第五节讨论结果
第六节对现有方法进行了分析并给出了未来研究的可能途径
最后在第七节回顾并比较了关键工作，第八节总结

## 强化学习建模图鲁棒性

**马尔可夫决策过程 (Markov Decision Process)**

马尔可夫决策过程是决策过程的形式化，在这一过程中，决策者称为 Agent 即智能体，它和环境进行交互，发现自己处于一个状态 State $s \in \mathcal{S}$ 中，需要从可用的动作集中选择并采取一个动作 Action $a \in \mathcal{A}(s)$。对于每一次采取的动作，智能体会获得一个奖励 Reward $\mathcal{R}(s, a)$。在一段时间内，智能体不断地进行决策，其目的是让这一段决策过程能够获得尽量多的奖励。然后智能体发现状态发生了变化，出现了一个新的状态，这个变化的过程由转移模型 $\mathcal{P}$ 定义，每一个转移对应一个概率 $P(s' , a, s)$。

智能体与环境不断进行交互，交互的过程即轨迹 Trajectory：$S_0, A_0, R_1, S_1, A_1, R_2, \cdots, S_{T−1}, A_{T−1}, R_T$。元组 $(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R}, \gamma)$ 定义了整个马尔可夫决策过程，其中 $\gamma$ 是折扣因子。
策略 $\pi(a|s)$，即状态上的动作分布，则定义了智能体的行为。当给定一个策略，状态动作值函数 $Q_{\pi}(s, a)$ 被定义为从状态 $s$ 开始采取动作 $a$ 之后根据策略 $\pi$ 执行所能获得的期望回报 Return。有几种方法能够迭代地得到最优状态动作值函数和策略，如广义策略迭代。

** 
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

鲁棒性目标函数的定义
本文主要关注的是图的鲁棒性，作为目标函数
给定一个图G，我们让critical fraction作为最小的节点集，即需要根据顺序被移除，使得变成无连接
这一部分越大，这个图可以说越鲁棒
移除的顺序对p来说有影响，对应两种攻击策略
must appear in the order of their degree in this permutation, i.e., 
$$
\forall v, u \in V. \xi_{targeted}(v) \leq \xi_{targeted}(u) \iff d^v \geq d^u
$$
We define the objective functions $\mathcal{F}$ in the following way:
1. _Expected Critical Fraction to Random Removal_:
$$
\mathcal{F}_{random}(G) = \mathbb{E}_{\xi_{random}}[p(\xi_{random})]
$$
2. _Expected Critical Fraction to Targeted Removal_:
$$
\mathcal{F}_{targeted}(G) = \mathbb{E}_{\xi_{targeted}}[p(\xi_{targeted})]
$$
我们考虑随机排列，对应不同
我们定义目标函数如下
为了获得这些量的估计，我们生成permutation并在p函数上平均
To obtain an estimate of these quantities, we generate $R$ permutations $\xi^i, i\in \overline{1,R}$ and average over the critical fraction $p(\xi^i)$ computed for each permutation:
$$
\widehat{\mathcal{F}(G)} =\frac{\sum_{i=1}^Rp(\xi^i)}{R}
$$
We use $\mathcal{F}_{random}(G)$ and $\mathcal{F}_{targeted}(G)$ to mean their estimates obtained in this way in the remainder of this paper.

## 3 用函数逼近学习如何创建鲁棒图

从第二节描述的问题模型开始，讨论一个可缩放的学习如何建立鲁棒图的方法的设计和应用
尽管这个形式化允许我们使用表格化强化学习方法，但状态的数量会迅速变得难以处理，比如二十个节点的带标签的无权连接图有接近10^57个
因此我们要求考虑图属性的一种方法是labelagnostic，permutation-invariant并且在相似的状态和动作上泛化
图神经网络满足这些要求
特别的，我们的方法里采用基于S2V的图表示，一种根据平均场推断在图模型的图神经网络架构
给定一个输入图，每个节点有特征向量，边也有特征向量。他的目标是对于每一个节点产生一个特征向量能够捕获图的结构并且作为邻居之间的交互
几轮聚合邻居的特征并应用一个非线性激活函数，比如神经网络或者和函数
对于每一轮，网络同时应用如下更新，其中是节点的邻居
一旦获得了节点级别的嵌入，子图的不变性嵌入能够将节点级别的嵌入进行求和导出
本文采用了Q-learning[Watkins and Dayan, 1992]，即通过估计之前介绍过的状态动作值函数来得到一个确定性的策略，即关于它的贪婪动作
智能体和环境进行交互并根据下式更新估计
For each round $k \in {1, 2, \cdots, K}$, the network simultaneously applies an update of the form:
$$
\mu_v^k=F(x_v,\{\mu^k_u\}_{u\in\mathcal{N}(v)},\{w(v,u)\}_{u\in \mathcal{N}(v)};\Theta)
$$
$$
\mu_{\mathcal{S}}=\sum_{v_i\in\mathcal{S}}\mu_{v_i}
$$
$$
Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \max_{a'\in\mathcal{A}(s')}Q(s', a') − Q(s, a)]
$$

在高维的状态和动作空间的情况下，用神经网络来估计Q的方法已经在各种领域从游戏到连续控制都获得了成功[Mnih et al., 2015; Lillicrap et al., 2016]. 
特别的我们使用DQN算法，一种样本有效方法，提升神经拟合Q迭代通过使用经验池和一个迭代更新目标网络对状态动作值函数估计[Van Hasselt et al., 2016]
我们opt for Double DQN变体使用两个不同的模型分别进行贪婪动作选择和状态动作值估计来解决标准DQN动作值的过高估计
我们用两步回报来加速并提升学习过程

一个关键的问题是状态的表示
 [Dai et al., 2018] 
一种可能性是通过计算包含所有节点的子图的 S2V 嵌入来表示状态 $s $$，这些节点由一条直到 time $t $的边连接起来，并通过连接节点 $v $和 $u $的嵌入来表示动作 $a t add (v，u) $。 但是，我们注意到以这种方式定义操作并不能扩展到大型图形，因为必须考虑每个步骤中的 $o (| v | ^ 2) $操作。
相反，我们遵循[ Dai 等人，2018]中介绍的方法，将每个操作 $a t $分解为两个步骤 $a ^ {(1)} t $和 $a ^ {(2)} t $。
$a ^ {(1)} t $对应于选择由边链接的第一个节点，$a ^ {(2)} t $对应于第二个节点。
通过这种方式，代理只需要考虑每个时间步骤中更易于管理的 $o (| v |) $操作数。

## 4 实验设置
**学习环境**

文章构建了一个允许定义任意图目标函数 $\mathcal{F}$ 的学习环境，并为智能体提供一个标准化的接口。
我们对环境、智能体和实验 suite 的实现基于 PyTorch [ Paszke et al. ，2019]。

**Baselines**

我们比较了以下基线:
* **Random**：随机选择一个可用的动作。
* **Greedy**：在添加一条边的情况下，使用lookahead向前查找，并选择对 $\mathcal{F}$ 的估计值有最大提升的动作。

**图模型**

我们研究下列模型生成的图的性能:
* **Erdos-Renyi(ER)**：从{G}^{(N,m)}[Erdos Renyi，1960]中统一采样的图。我们使用 $m =\frac{20}{100}∗\frac{N*(N-1)}{2}$，它代表所有可能边的20% 。

* **Barabasi-Albert(BA)** : 一种增长模型，其中 $n$节点优先连接到 $m $现有节点[ Barabasi 和 Albert，1999]。
我们使用 m 2 $。

**实验过程**
e generate a set of graphs $\mathbf{G}^{train}$ using the 2 graph models above. 
We train the agent using graphs with $|V| = 20$ vertices and a number of edge additions $L$ equal to 1% of total possible edges (2 for $|V| =20$). 
At each step, the algorithm proceeds over all graphs in $\mathbf{G}^{train}$. 
During training, we assess the performance of the agent on a disjoint set $\mathbf{G}^{validate}$ every 100 steps. 
We evaluate the performance of the algorithm and baselines over a disjoint set of graphs $\mathbf{G}^{test}$. 
We use $|\mathbf{G}^{train}| = |\mathbf{G}^{validate}| = |\mathbf{G}^{test}| = 100$. 
Graphs are generated using a wrapper around the _networkx_ Python package [Hagberg et al., 2008]. 
In case the candidate graph returned by the generation procedure is not connected, it is rejected, and another one generated until the set reaches the specified cardinality. 
To estimate the value of the objective functions, we use a number of permutations $R = |V|$. 
When repeating the evaluation for larger graphs, we consider $|V| \in {30, 40, 50, 60, 70, 80, 90, 100}$ and scale $m$ (for ER), $L$ and $R$ depending on $|V|$.
我们使用上面的2个图模型生成一组图 $\mathbf{G}^{train}$$ mathbf { g } ^ { train } $。
我们使用 $| v | 20 $顶点和一些加边数 $l $等于所有可能边的1% (2为 $| v | 20 $)来训练智能体。

在每个步骤中，算法对 $ mathbf { g } ^ { train } $中的所有图进行处理。

在训练期间，我们每100个步骤评估智能体在不相交集 $ mathbf { g } ^ { validate } $上的性能。

我们在一个不相交的图集 $ mathbf { g } ^ { test } $上评估算法和基线的性能。

我们使用 $| mathbf { g } ^ { train } | | mathbf { g } ^ { validate } | | | | mathbf { g } ^ { test } | 100 $。

使用 networkxpython 包的包装器生成图形[ Hagberg 等人，2008]。

如果生成过程返回的候选图没有连接，则拒绝该候选图，并生成另一个候选图，直到集合达到指定的基数。
为了估计目标函数的价值，我们使用了一些排列。
当重复对较大图的评估时，我们考虑{30,40,50,60,70,80,90,100} $的 $| v | ，并且根据 $| v | $的不同，标度为 $m $(ER) ，$l $和 $r $。


**RNet-DQN 参数**
Gtrain is divided into batches of size 50. 
Training proceeds for 100000 steps. 
We use a learning rate $\alpha = 0.001$ and a discount factor $\gamma=1$ since we are in the finite horizon case. 
We use a value of the exploration rate $\epsilon$ that we decay linearly from $\epsilon = 1$ to $\epsilon = 0.1$ for the first 50000 steps, then fix $\epsilon = 0.1$ for the rest of the training.
We use 8 latent variables and a hidden layer of size 32. 
The only hyperparameter we tune is the number of message passing rounds $K$, for $K\in {2, 3, 4, 5}$, selecting the agent with the best performance over $\mathbf{G}^{validate}$.

Training, hyperparameter optimization and evaluation are performed separately for each graph family and objective function $\mathcal{F}$. 
Since deep RL algorithms are notoriously dependent on parameter initializations and stochastic aspects of the environment [Henderson et al., 2018], we aggregate the results over 5 runs of the training-evaluation loop.
Gtrain 分为50号的批次。
训练进行了100000个步骤。
我们使用学习率 alpha 0.001 $和贴现因子 gamma 1 $，因为我们处于有限水平情形。
我们使用一个探索率的值 $ epsilon $，在前50000步骤中，我们线性地从 $ epsilon 1 $衰减到 $ epsilon 0.1 $，然后在剩下的训练中修复 $ epsilon 0.1 $。
我们使用8个潜在变量和一个32大小的隐藏层。
我们调优的唯一超参数是传递轮 $k $的消息数量，对于 $k in {2,3,4,5} $，选择性能优于 $ mathbf { g } ^ { validate } $的代理。
对每个图族和目标函数分别进行训练、超参数优化和评价。
由于深度 RL 算法是众所周知的依赖于参数初始化和环境的随机方面[ Henderson 等人，2018] ，我们聚合了训练评估循环的5次运行的结果。

## 五: 结果

在**表 1** 中，我们列出了我们实验评估的主要结果。
![]()

对于 $\mathcal{F}_{random}$, 由 RNet-DQN 学习到的策略在 ER 和 BA 情况下都优于贪婪方法，而对于 $\mathcal{F}_{targeted}$ 的提升效果则不是够好，但比随机的仍然要好很多。
这一点很重要，因为 RNet-DQN 学习策略能够在一个不相交的测试图集进行泛化，而贪婪方法则必须为这个集合中的每个图单独计算。
在所有类型的图和性能度量中，RNet-DQN 在统计学上明显优于随机方法。

**有效缩放到大图**

本文方法的最理想的特性是能够在小尺寸的图进行训练，在大尺寸的图上进行预测。在较小的图上进行学习，速度会快很多，因为构建学习表示的工作量减少了，可能的动作数也减少了，目标函数的评估也更快了。
因此，本文使用在 **Section 4** 中描述的大小为 $|V|=20$ 的图上训练的模型，并评估它们和基线在顶点数量高达 $|V|=100$ 的图上的性能 (由于计算开销，贪婪方法只能计算 $|V|=50$ 的图，见下一段)。

**图 2** 中展示了所得到的结果。
![]()

文章发现在考虑 BA 图时，RNet-DQN 的性能即两个目标函数下降相对较少。对于 ER 图，性能会迅速下降，对于大小为 $|V|=50$ 及以上的图，所学到的策略比随机策略执行得更差。这表明 ER 图的鲁棒性性质随着尺寸的增大而发生根本性的变化，而且 RNet-DQN 所学到的特征对于改进目标函数已经不再适用，超出了一个特定的尺寸 multiplier。

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
因此，贪婪的代理每步最多可以获取 $O(|V|^6)$，这对于 nontrivially 大小的图来说太昂贵了。
这种巨大的成本差异也被前面所展示的经验测量所捕捉。

值得注意的是，上面的分析没有考虑模型的训练成本，因为模型的运行时间取决于几个超参数以及问题，所以很难确定模型的复杂性。
尽管如此，训练还是包括了对每个训练图每一步评估一次目标函数。
因此，在需要对许多图进行预测或者模型可以缩放为计算目标函数开销很大的大图的情况下，这种方法是有优势的。

# 6 局限性和未来工作

在这一部分中，文章讨论了所提方法的一些局限性以及这项工作可能的未来方向。
因为我们使用的是深度强化学习方法，所以这类算法有一些典型的注意事项。caveats
文章观察到在相同的超参数下，网络权重的不同随机初始化的性能差异非常显著; 虽然图表报告了方法的平均性能，但有时会发现明显更好的解决方案。
我们有信心可以根据目标函数和图结构量身定制更聪明的探索策略，可以得到在不同的初始化下更加一致的解决方案。
我们也试验过每个 episode 增加更多数量的边缘，但是所涉及的噪音使得算法在这些场景中非常不稳定。

虽然我们的解决方案能够学习可泛化的策略，并且在计算复杂性方面被认为是有效的，但是与贪婪的解决方案相比，它必然会牺牲单个图实例的性能。
然而，这两种解决方案并不一定相互排斥: 基于模型的方法可以用来在贪婪搜索之前提供一个关于有希望的边的先验信息，减少必须要考虑的动作空间。
事实上，在一些真实世界的网络(例如，交通)中，有一个基本的几何图形，边的增加可能会受到一些实际问题的限制，例如对成对距离的约束。

所提框架的适用性并不局限于健壮性。
实际上，它支持定义在图上的任意目标函数。
在网络科学社区中经常使用的性质是可沟通性和效率[ Newman，2018]。

此外，当目标函数的评估成本很高时，这种方法可能是有利的——这是涉及流量或流行病等网络模拟的动态过程的情况[ Barrat 等人，2008]。
虽然在这项工作中，我们只考虑了拓扑结构，GNN 框架允许节点和边缘特性，如果可用的话。
如果与目标函数相关，这些方法有可能改进性能。
在目前的工作中，我们只研究了加边作为可能的行动。
人们也可以解决消除图中边的任务——布雷斯悖论 [Braess, 1968]
建议清除可能违反直觉导致提高效率。
允许去除边和添加可以给重新布线，这将大大增加可能的图的空间，可以用这种方式构建。

# 7 Related Work

**网络健壮性**

网络对随机错误和目标攻击的适应能力首先由 Albert 等人讨论[2000] ，他们检验了平均最短路径距离作为被移除节点数量的函数。
通过对两个无标度通信网络的分析，他们发现这种网络对随机故障具有很好的鲁棒性
但很容易受到有针对性的攻击。
Holme 等人的一个更广泛的调查[2002]分析了几个现实世界网络以及一些通过合成模型生成的网络的健壮性。
研究了不同的基于度和中间集中度的攻击策略，发现通常重新计算节点删除后的集中度可以产生更有效的攻击策略。
已经获得了描述两种攻击策略下网络模型的分解阈值的各种分析结果[ Cohen 等，2000; Cohen 等,
2001].
正如前面所讨论的，一些工作已经考虑到了提高现有网络弹性的问题[ Beygelzimer 等人，2005; Schneider 等人，2011;
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

相反，我们的方法是高度可伸缩的，在评估时提供了 $o (| v | ^ 3) $speed-up。

最后，值得注意的是，我们的贡献也是

方法的。

基于回弹的不同定义，或者考虑表示图的其他特征的完全不同的目标函数，本文提出的方法可以应用于其他问题。

学习如何提高图的鲁棒性可以作为本文提出的一般方法的一个案例研究。

# Reference

[Albert et al., 2000] Reka Albert, Hawoong Jeong, and  Albert-Laszlo Barabasi. Error and attack tolerance of com  plex networks. Nature, 406(6794):378–382, 2000.

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
