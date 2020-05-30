# Measuring Structural Similarities in Finite MDPs

标题：Measuring Structural Similarities in Finite MDPs

作者：Wang Hao，Dong Shaokang，Shao Ling

发表：IJCAI 2019

链接：https://www.ijcai.org/Proceedings/2019/0511.pdf

## 摘要

In this paper, we investigate the structural similarities within a finite Markov decision process (MDP).
We view a finite MDP as a heterogeneous directed bipartite graph and propose novel measures for the state and action similarities, in a mutually reinforced manner. 
We prove that the state similarity is a metric and the action similarity is a pseudometric. 
We also establish the connection between the proposed similarity measures and the optimal values of the MDP. Extensive experiments show that the proposed measures are effective.

## 1. Introduction

马尔可夫决策过程 (Markov Decision Process, MDP) 是一种支持决策的有效数学模型，在许多领域有实际应用，如智能控制系统、金融、能源管理、在线广告等等。

- [ [Cai et al.](Realtime_bidding_by_reinforcement_learning_in_display_advertising), WSDM, [2017](). ]
- [ [Han et al.](Dynamic_virtual_machine_management_via_approximate_Markov_decision_process), INFOCOM, [2016](). ]

有限 MDP 对智能体与外部环境之间的相互作用进行建模。环境被描述为一组离散的状态，其中，在每个状态上，智能体都有一组可用的动作。
通过观察当前状态并决定采取哪个动作，智能体可以改变环境的状态。
在一个设计得当的奖励方案的驱动下，智能体能够指定一个良好的策略，以便在环境中做出明智的决策。

本文研究了如何度量有限 MDP 中状态和动作的相似性。
这样的相似性很重要因为它们可能在其他各种研究领域上提供一种设计解决方案的原则方法。例如：

- [ [Taylor and Stone.](Transfer_learning_for_reinforcement_learning_domains:a_survey), JMLR, [2009](). ]，其目的是利用过去的学习经验来加速当前学习过程。
- [ [Abel.](A_theory_of_state_abstraction_for_reinforcement_learning) AAAI, [2019](). ]，其目标是构建具有较小的状态动作空间的抽象 MDP，同时仍然保持原始 MDP 的某些属性。

与本文关系最密切的研究是 [ [Ferns et al.](Metrics_for_finite_Markov_decision_processes), UAI, [2004](). ] 提出的互模拟度量 (Bisimulation metric)，顺利地扩展了 [ [Givan et al.](Equivalence_notions_and_model_minimization_in_Markovdecision_processes), Artificial Intelligence, [2003](). ] 提出的互模拟的概念。
简单地说，互模拟定义了状态之间的等价关系，从而将状态空间的划分归纳成等价类。无论何时采取了相同的动作，互模拟状态都会执行完全相同的 (概率) 转移到别的等价类，获得完全相同的奖励。
基于观察到如果相同的动作具有相似的效果那么两个状态是相似的 (two states are similar if
the same action has similar effects) 这一点，[ [Ferns et al.](Metrics_for_finite_Markov_decision_processes), UAI, [2004](). ] 建立起状态之间的互模拟度量，以扩展互模拟的严格概念。尽管如此，互模拟度量的一个问题是忽略了动作的作用。
考虑如 **图 1(a)** 所示的 MDP，互模拟度量无法捕捉到在状态 $u$ 上采取动作 $b$ 在某种程度上 (to some extent) 相似于在状态 $v$ 采取动作 $a$ 的事实。本文通过显式考虑不同动作之间的相似性来解决这个问题。

另一个相关概念是 [ [Ravindran and Barto.](SDMP_homomorphisms:an_algebraic_approach_to_abstraction_in_semi-Markov_decision_processes), IJCAI, [2003](). ] 提出的 MDP 同态 (MDP homomorphism) ，它由两个代数映射组成，一个是状态之间的映射，另一个是动作之间的。 这两个映射应当精确地保持奖励概率和转移概率。 
[ [Taylor and Stone.](An_introduction_to_intertask_transfer_for_reinforcement_learning), AI Magnize, [2011](). ] 指出同态通常太过严格，计算上也太困难，没有实际用处。 
[ [Sorg and Singh.](Transfer_via_soft_homomorphisms), AAMAS, [2009](). ] 提出了软同态 (soft homomorphism)，允许状态映射是概率的，只要它精确地保持期望的转移概率和奖励。 
本文认为在许多实际情况下，软同态仍然过于严格。例如，在 **图 1(a)** 的例子中没有非平凡 (软) 同态，即使状态 $u$ 和 $v$ 在直觉上非常相似。

本文不使用互模拟和同态所采用的代数观点，而是从图论的角度来揭示有限 MDP 内部的结构相似性。 本文方法论起源于度量一般图中节点之间的结构-上下文 (structual-contextual) 相似性的研究。 
[ [Jeh and Widom.](SimRank:a_measure_of_structural-context_similarity), KDD, [2002](). ] 建立了 SimRank，其基本思想是两个节点相似当且仅当它们的邻居相似 (two nodes are similar if and only if their neighbors are similar)。 虽然 SimRank 不能直接应用于 MDP，因为它不知道 MDP 特有的特征，但是仍然能够定制它的基本思想来建立 MDP 中的相似性度量。
综上所述，本文的主要贡献有以下几点。

1. 本文提出了有限 MDPs 的异构二部 (heterogenerous bipartite) 图表示。(Section 3)，该图包含了由决策和转移边连接的状态和动作节点。通过这样的表示，就能正确地捕获状态和动作所起的作用。
2. 本文提出了状态和动作相似性的递归 (recursive) 定义 (Section 4.1)，可以通过迭代算法 (Section 4.2) 来有效计算相似性。
3. 本文证明了 (induce) 距离度量有良好的度量性质 (Section 4.3)，还表明了该度量能够用于限制 MDP 最优值之间的差异 (Section 4.4)。
4. 通过在随机生成的 MDP 上的大量实验，本文证明该度量在捕获结构相似性上是有效的 (Section 5)。

## 2. Related Work

### 2.1 Transfer in Reinforcement Learning

迁移学习 (Transfer Learning) 的目的是利用过去的学习经验来加速当前任务的学习。
许多迁移学习技术都是基于任务相似性 (task similarity) 的概念。

- [ [Lazaric et al.](Transfer_of_samples_in_batch_reinforcement_learning), ICML, [2008](). ] 以样本为导向 (sample-oriented) 的观点来衡量两个 MDP 之间的相似性。
- [ [Ramamoorthy et al.](Clustering_Markov_decision_processes_for_continual_transfer), [2013](). ] 提出了在同一状态动作空间内两个 MDP 之间保值的李普希兹度量 (value-perserving Lipschitz metric)，通过计算奖励和状态转移函数之间可能的最大差异来得到该度量。
- [ [Ammar et al.](An_automated_measure_of_MDP_similarity_for_transfer_reinforcement_learning), AAAI [2014](). ] 使用一个受限玻尔兹曼机来度量来自两个 MDP 的两批样本之间的距离，这之后用作两个 MDP 之间的距离。
- [ [Sinapov et al.](Learning_inter-task_transferability_in_the_absence_of_target_task_samples), AAMAS, [2015](). ] 将 MDP 表示为特征向量，并训练回归模型来评估两个 MDP 的接近程度。
- [ [Song et al.](), [2016](). ] 最近应用几个度量标准来衡量在同一状态动作空间下两个 MDP 之间的距离。

以上这些工作大都集中在任务间 (inter-task) 的相似性而不是单独一个任务内部的结构相似性。

### 2.2 MDP Abstraction

MDP 抽象化这一研究领域同样与本文相关。 
[ [Li et al.](Towards_a_unified_theory_of_state_abstraction_for_MDPs), ISAIM, [2006](). ] 提出了一个状态抽象化的统一理论，包含了互模拟 [ [Givan et al.](Equivalence_notions_and_model_minimization_in_Markovdecision_processes), Artificial Intelligence, [2003](). ]，同态 [ [Ravindran and Barto.](SDMP_homomorphisms:an_algebraic_approach_to_abstraction_in_semi-Markov_decision_processes), IJCAI, [2003](). ]，策略不相关 [ Jong and Stone, 2005 ] 等等。
该理论的焦点在于在状态抽象化的过程中保持正确性。
结构相似性 (特别是动作间的相似性) 没有充分探索。

另一方面，已经有一些使用抽象 MDPs 进行规划和学习的研究工作。

- [ [Abel et al.](), [2019](). ]
- [ [Gopalan et al.](), [2017](). ]

这些工作的焦点和本文不同，但是本文工作引入的一个关于 MDP 抽象化的新颖观点可能有助于这一领域。

### 2.3 Structural Similarity in Graphs

除了 Section 1 中介绍的 SimRank 度量 [ [Jeh and Widom](), [2002](). ]，图中节点到节点近邻性质已经经过广泛的研究。
[ [Jeh and Widom](), [2003](). ] 提出了使用私人化 PageRank，essentially an asymmetric random walk distance。

因为 SimRank 在某些例子中是 counterintuitive 和 inflexible，许多研究工作尝试提升 SimRank。

- [ [Xi et al.](), [2005](). ] 提出了 SimFusion，基于一个图的统一关系矩阵表示
- [ [Antonellis et al.](), [2008](). ] 通过增加图节点邻居的权重将 SimRank 改为 SimRank++
- [ [Zhao et al.](), [2009](). ] 提出了 P-Rank 将 SimRank 推广到信息网络
- [ [Lin et al.](), [2012](). ] 将最大匹配引入到相似性度量，得到了 MatchSim。
- [ [Jin et al.](), [2014](). ] 提出了一个度量 RoleSim，由一组 admissible 性质集合 complies。

以上提到的度量不能直接用于本文问题，因为他们并没有定制到有限 MDPs。

## 3. Graph Representation of MDPs

MDP 通常被描述为一个元组 $M=(S,A,T,R)$：

- $S$ 有限状态集
- $A$ 有限动作集
- $T$ 状态转移函数 $S \times A \times S \rightarrow [0,1]$
- $R$ 奖励函数 $S \times A \times S \rightarrow [0,1]$

具体地说，对于每一个状态 $s\in S$，都有一个对应的可用动作集 (available actions) $A_s \ A$，状态转移函数 $T(s,a,s&#39;)$ 表示在状态 $s$ 采取动作 $a$ 得到状态 $s&#39;$ 的概率，而 $R(s,a,s&#39;)$ 则是这个状态转移的奖励。

这种 MDP 的代数表示的一个问题是它无法分辨出在不同状态下使用的同名动作。为了解决这一问题，本文考虑如下的 MDP 的图论表示。

一个 MDP $M=(S,A,T,R)$ 的 MDP 图 (MDP graph) 定义为 $G_M=(V,\Lambda,E,\Psi,p,r)$，一个由状态节点 $V$ 和动作节点 $\Lambda$ 构成的异构有向二部图。$E$ 是从状态节点到动作节点的决策边 (decision edges) 集，$\Psi$ 包含所有从动作节点到状态节点的转移边 (transition edges)。其中决策边是无权重的，每个转移边 $(\alpha,v)\in \Psi$ 的权重由转移概率 $p(\alpha, v)$ 和奖励 $r(\alpha,v)$ 决定。
下面的步骤表示从给定的 MDP $M$ 构造图 $G_M$ 的过程：

1. 对每一个 $s\in S$，创建一个新的节点 $v_s$ 加入 $V$
2. 对每一个 $s\in S$ 和 $a\in A_s$，创建一个新的节点 $\alpha_a$ 加入 $\Lambda$ 和一个新边 $(v_s,\alpha_a)$ 加入 $E$
3. 对于每一个使得 $T(s,a,s&#39;)&gt;0$ 的状态转移 $(s,a,s&#39;)\in S\times A\times S$，创建一个新边 $(\alpha_a,v_{s&#39;})$ 加入 $\Psi$，设置 $p(\alpha_a,v_{s&#39;})=T(s,a,s&#39;)$ 和 $r(\alpha_a,v_{s&#39;})=R(s,a,s&#39;)$

显然 MDP $M$ 和图 $G_M$ 之间存在一一对应短息。
图 $G_M$ 通常是二部的，$|V|=|S|,|\Lambda|=|E|=\sum_{s\in S}|A_s|$。
**图 1(b)** 展示了 **图 1(a)** 所示的 MDP 对应的图，由 6 个状态节点，4 个动作节点，4 个决策边和 9 个转移边构成。注意到 4 个动作节点区分了状态 $u$ 和 $v$ 上的两个动作 $a,b\in A$。

## 4. The Structural Similarities

对于给定的 MDP 图 $G_M=(V,\Lambda,E,\Psi,p,r)$ 中任意节点 $x\in V\cup\Lambda$，$N_x$ 表示 $x$ 所有的 out-邻居的集合。注意到 $G_M$ 是二部图，因此一个状态节点的 out-邻居总是动作节点，二动作节点的 out-邻居总是状态节点 (**图 1(b)**)。
现在定义状态相似性 $\sigma_S$ 和动作相似性 $\sigma_A$，目标是让 indcued distance 度量有需要的性质

$$
\sigma_S(u,v)\doteq 1-\sigma_S(u,v), \forall u, v\in V\\
\sigma_A(\alpha,\beta)\doteq 1-\sigma_A(\alpha, \beta), \forall \alpha,\beta \in A
$$

### 4.1 The Recursive Similarity Measure

本文修改了 SimRank [ [Jeh and Widom.](), [2002](). ] 的基本思想以定义 $\sigma_S$ 和 $\sigma_A$，即两个节点是相似的当且仅当它们的邻居是相似的。这一思想可以用 recursion 实现。

#### The Base Cases

作为基本情形，本文定义：

$$
\sigma_S(u,v)\doteq 0 
$$

一个状态是吸收的 (absorbing) 即没有输出邻居，实际上就是目标状态。
$d_{u,v}\in [0,1]$ 的 configuration 因此是一个目标状态间关系的应用依赖描述。两个特殊的情形是 $d_{u,v}=1$ 和 $d_{u,v}=0$，表示任意两个目标状态应该被认为完全不同或者相同。

#### The Recursion for State Similarity

对于任意两个状态节点 $u,v\in V$，相似性 $\sigma_S(u,v)$ 是 essentially 它们的输出邻居 $N_u$ 和 $N_v$ 之间的相似性，这两个输出邻居是两个动作节点集。
$N_u$ 和 $N_v$ 之间的相似度应该轮流基于 pairwise 相似性 $\sigma_A(\alpha,\beta)$，
注意对所有成对相似性 $\sigma_A(\alpha,\beta)$ 进行简单平均会使得 $\sigma_S$ 无法作为度量，因为三角不等式 is compromised。
因此本文考虑使用 Hausdorff 距离 [ [Delfour and Zolesio.](), [2011](). ]
具体地说，给定一个动作节点 $\alpha$ 和一组动作节点 $N$，$\alpha$ 与 $N$ 之间的距离，滥用符号 $\delta_A(\alpha,N)\doteq min_{\beta\in N}\delta_A(\alpha,\beta)$。
给定 $\delta_A$ 的 Hausdorff 距离是所有元素到集合距离的最大值:

$$
\delta_{Haus} (N_u,N_v;\delta_A)=max_{\alpha\in N_{u}\\ \beta\in N_v}\{\delta_A(\alpha,N_v),\delta_A(\beta,N_u)\}
$$

Hausdorff 距离 $\delta_{Haus} (N_u,N_v;\delta_A)=\Delta$ 提供了状态 $u$ 和 $v$ 上可用动作之间成对距离的上界，具体地说，在与 $N_u$ 中任何动作的距离为 $\Delta$ 之内，$N_v$ 总是存在另一个动作。 因此，$\Delta$ 界定了 $N_u$ 和 $N_v$ 之间的总体差异。利用 $\delta_{Haus}$，得到

$$
\delta_S(u，v) =C_s \cdot (1-\delta_{Haus}(N_u,N_v;\delta_A))
$$

其中 $u,v\in V$ 是两个不同的非吸收状态，$0&lt;C_s&lt;1$ 是一个常数，折扣化了邻居 $N_\alpha$ 和 $N_\beta$ 对状态节点 $(u,v)$ 的影响。

#### The Recursion for Action Similarity

状态相似性 $\delta_S$ 依赖于一个良好定义的 $\delta_A$ (公式 4)
如**图 1(b)** 所示，在 MDP 图中，动作节点本身传递了有限的信息——真正重要的是它的结果或效果。
一个动作节点 $\alpha$ 本质上既是状态节点上的 $p_\alpha = (\alpha,*)$ 分布 (概率转移) ，也是 [0,1] 上的 $r_\alpha=r(\alpha,*)$ 分布 (随机奖励)。 
奖励之间的距离相对简单:

$$ 
\delta_{rwd} (\alpha,\beta)=|\mathcal[r_\alpha]-\mathcal[r_\beta]|
$$

也就是说，它是他们期望值之间的差异。 
任务的其余部分涉及测量两个概率分布在状态节点上的距离。 
为此，我们利用 earth mover&#39;s distance (EMD)[ [Rubner et al.](),[1998]() ]

$$
\delta_{EMD}(p_\alpha,p_\beta;\delta_S)=min_F\sum_{u\in N_\alpha}\sum_{v\in N_\beta}f_{u,v}\cdot\delta_S(u,v)\\
 s.t. \forall u,v\in V:f_{u,v}\gep 0,\\ 
\forall u \in V: \sum_{v\in V}f_{u,v}=p(\alpha, u)\\ 
\forall v \in V: \sum_{u\in V}f_{u,v}=p(\beta, v)
$$

EMD 通过移动 earth (在我们的例子中是概率) 来量化将一个分布转化为另一个分布。 
矩阵 $F$ 中的值 $f_{u,v}$ 是从状态节点 $u\in N_\alpha$ 移动到状态节点 $v\in N_\beta$ 的。 
这种“地球”的运动需要花费 $\delta_S(u,v)$。 
因此，$\delta_{EMD}(p_\alpha,p_\beta; \delta_S)$ 是将分布 $p_\alpha$ 转化为分布 $p_\beta$ 的最小可能工作量。
基于经验模态分解($\delta_{EMD}$) ，$\sigma_A$ 的递归设计为

$$
\sigma_A(\alpha,\beta)=1-(1-C_A)\delta_{rwd}(\alpha,\beta)-C_A\delta_{EMD}(p_\alpha,p_\beta;\delta_S)
$$

其中 $0&lt;C_A&lt;1$ 为奖励相似性和转移相似性的权重加权参数。

### 4.2 Computation

算法 1 展示了模拟 4.1 节的 recursion 计算 $\sigma^*_S$ 和 $\sigma^*_A$ 的迭代算法。

算法 Structural similarities of an MDP graph

#### Space and time complexity

### 4.3 Mathematical Properties

#### Lemma 1 (Boundedness)

#### Lemma 2 (Monotonicity)

#### Theorem 1 (Unique Existence & Nontrivialness)

#### Lemma 3 (Triangle Inequality)

#### Theorem 2 (Metric Properties)

### 4.4 Bounding Differences Between Optimal Values

#### Theorem 3 Bounds on Differences of Optimal Values

## 5. Experiments

### 5.1 A Case Study

### 5.2 Distribution of Distance Values

### 5.3 Effect of Parameters

## 6. Conclusion and Future Work

文章研究了有限 MDP 中的结构相似性。 通过将 MDP 表示为图，文章通过度量图节点之间的接近度来描述状态和动作之间的相似性。 文章证明了所提出的测度的度量性质，并在此基础上导出了最优值差异的上界。 广泛的实验表明了文章方法的优点。 未来计划通过并行来加速所提出的度量的计算，这是处理大型图的常用方法。 文章还对 MDP 上的相似性搜索查询感兴趣，其目标是有效地识别最相似的状态或动作对。
