## 1. Introduction

The Arcade Learning Environment (ALE; Bellemare et al., [2013](https://arxiv.org/pdf/1207.4708.pdf)) was proposed as a platform for empirically assessing agents designed for general competency across a wide range of games. 
ALE offers an interface to a diverse set of Atari 2600 game environments designed to be engaging and challenging for human players. 
As Bellemare et al.(2013) put it, the Atari 2600 games are well suited for evaluating general competency in AI agents for three main reasons: (i) varied enough to claim generality, (ii) each interesting enough to be representative of settings that might be faced in practice, and (iii) each created by an independent party to be free of experimenter’s bias.

街机学习环境在 2013 年作为一个平台被提出，该平台用于经验性地评估为广泛的游戏的一般能力而设计的智能体。ALE 提供了一个多样化的雅达利 2600 游戏环境的界面，旨在吸引和挑战人类玩家。正如 Bellemare 所说，雅达利 2600 游戏非常适合评估人工智能智能体的一般能力，主要有三个原因: 
	1. 足够多样化以声称一般性;
	2. 每一个游戏都足够有趣，能够代表在实践中可能面临的环境;
	3. 每一个游戏都是由独立的一方创造出来的，没有实验者的偏见。

Agents are expected to perform well in as many games as possible making minimal assumptions about the domain at hand and without the use of game-specific information.
Deep Q-Networks (DQN ; Mnih et al., [2015]()) was the first algorithm to achieve human-level control in a large number of the Atari 2600 games, measured by human normalized scores (HNS). 
Subsequently, using HNS to assess performance on Atari games has become one of the most widely used benchmarks in deep reinforcement learning (RL), despite the human baseline scores potentially underestimating human performance relative to what is possible (Toromanoff et al., 2019). 
Nonetheless, human benchmark performance remains an oracle for “reasonable performance” across the 57 Atari games. 
Despite all efforts, no single RL algorithm has been able to achieve over 100% HNS on all 57 Atari games with one set of hyperparameters. 
Indeed, state of the art algorithms in model-based RL, MuZero (Schrittwieser et al., 2019), and in model-free RL, R2D2 (Kapturowski et al., 2018) surpass 100% HNS on 51 and 52 games, respectively. 
While these algorithms achieve well above average human-level performance on a large fraction of the games (e.g. achieving more than 1000% HNS), in the games they fail to do so, they often fail to learn completely. 
These games showcase particularly important issues that a general RL algorithm should be able to tackle. 
Firstly, long-term credit assignment: which decisions are most deserving of credit for the positive (or negative) outcomes that follow? 
This problem is particularly hard when rewards are delayed and credit needs to be assigned over long sequences of actions, such as in the games of Skiing or Solaris. 
The game of Skiing is a canonical example due to its peculiar reward structure. 
The goal of the game is to run downhill through all gates as fast as possible. 
A penalty of five seconds is given for each missed gate. 
The reward, given only at the end, is proportional to the time elapsed. 
Therefore long-term credit assignment is needed to understand why an action taken early in the game (e.g. missing a gate) has a negative impact in the obtained reward. 
Secondly, exploration: efficient exploration can be critical to effective learning in RL. 
Games like Private Eye, Montezuma’s Revenge, Pitfall! or Venture are widely considered hard exploration games (Bellemare et al., 2016; Ostrovski et al., 2017) as hundreds of actions may be required before a first positive reward is seen. 
In order to succeed, the agents need to keep exploring the environment despite the apparent impossibility of finding positive rewards. 
These problems are particularly challenging in large high dimensional state spaces where function approximation is required.

智能体被期望在尽可能多的游戏上表现良好，对当前域做出最小的假设，并且不使用游戏特定的信息。
以人类标准化得分 (HNS) 来衡量，深度 Q 网络是第一个在大量雅达利 2600 游戏上达到人类水平控制的算法。
随后，用人类标准化得分来评估在雅达利游戏上的性能已经变成在深度强化学习最广泛应用的基准之一，尽管这个基线得分可能低估了人类的表现。
尽管如此，人类的基准表现仍然是 57 款雅达利游戏中“合理性能”的预言。
尽管做了所有的努力，没有一个单一的强化学习算法能够用一组超参数在所有 57 个雅达利游戏上实现 100% 以上的HNS。
事实上，基于模型的强化学习 MuZero 和无模型强化学习 R2D2，作为最先进算法的代表分别在 51 款和 52 款游戏中超过 100% HNS。
虽然这些算法在很大一部分游戏中实现了远高于人类平均水平的性能 (例如，达到 1000% 以上的 HNS)，但在它们未能做到这一点的游戏中，它们往往无法完全学习。
这些游戏展示了一般的强化学习算法应当能解决的特别重要的问题。
首先，**长期信用分配**：对于随后的积极 (或消极) 的结果，哪些决策最值得信任？
当奖励被推迟，需要在长时间的动作序列中分配信用时，这个问题尤其困难，比如在滑雪或 Solaris 游戏中。
滑雪比赛是一个典型的例子，因为它有特殊的奖励结构。
这个游戏的目标是以尽可能快的速度下坡并通过所有的大门。
每错过一次大门将被处以 5 秒的处罚。
只有在结束时才给予的奖励，与所经过的时间成正比。
因此，需要长期的信用分配来理解为什么在游戏早期采取的行动 (例如，错过了一个门) 会对获得的奖励产生负面影响。
其次，**探索**：强化学习中有效的探索是有效学习的关键。
像“私家侦探”、“蒙特祖马的复仇”、“瀑布”之类的游戏被广泛认为是艰难的探索游戏，因为可能需要数百个行动才能看到第一个积极的奖励。
为了成功，智能体需要继续探索环境，尽管显然不可能找到积极的回报。
这些问题在需要函数逼近的大型高维状态空间中尤其具有挑战性。


Figure 1. Number of games where algorithms are better than the human benchmark throughout training for Agent57 and state-of-the-art baselines on the 57 Atari games.

Exploration algorithms in deep RL generally fall into three categories: 
* randomized value functions (Osband et al., 2016; Fortunato et al., 2017; Salimans et al., 2017; Plappert et al., 2017; Osband et al., 2018), 
* unsupervised policy learning (Gregor et al., 2016; Achiam et al., 2018; Eysenbach et al., 2018) 
* intrinsic motivation (Schmidhuber, 1991; Oudeyer et al., 2007; Barto, 2013; Bellemare et al., 2016; Ostrovski et al., 2017; Fu et al., 2017; Tang et al., 2017; Burda et al., 2018; Choi et al., 2018; Savinov et al., 2018; Puigdomènech Badia et al., 2020). 

Other work combines handcrafted features, domain-specific knowledge or privileged pre-training to side-step the exploration problem, sometimes only evaluating on a few Atari games (Aytar et al., 2018; Ecoffet et al., 2019). 
Despite the encouraging results, no algorithm has been able to significantly improve performance on challenging games without deteriorating performance on the remaining games without relying on human demonstrations (Pohlen et al., 2018). 
Notably, amongst all this work, intrinsic motivation, and in particular, Never Give Up (NGU;Puigdomènech Badia et al., 2020) has shown significant recent promise in improving performance on hard exploration games.
NGU achieves this by augmenting the reward signal with an internally generated intrinsic reward that is sensitive to novelty at two levels: short-term novelty within an episode and long-term novelty across episodes. 
It then learns a family of policies for exploring and exploiting (sharing the same parameters), with the end goal of obtain the highest score under the exploitative policy. 
However, NGU is not the most general agent: much like R2D2 and MuZero are able to perform strongly on all but few games, so too NGU suffers in that it performs strongly on a smaller, different set of games to agents such as MuZero and R2D2 (despite being based on R2D2). 
For example, in the game Surround R2D2 achieves the optimal score while NGU performs similar to a random policy.
One shortcoming of NGU is that it collects the same amount of experience following each of its policies, regardless of their contribution to the learning progress. 
Some games require a significantly different degree of exploration to others. Intuitively, one would want to allocate the shared resources (both network capacity and data collection) such that end performance is maximized.
We propose allowing NGU to adapt its exploration strategy over the course of an agent’s lifetime, enabling specialization to the particular game it is learning. 
This is the first significant improvement we make to NGU to allow it to be a more general agent.

深度强化学习中的探索算法通常分为三类：
* 随机化值函数
* 无监督策略学习
* 内在动机

其他工作结合了手动制作的特征、特定领域的知识或特权预训练，以回避探索问题，有时只对少数几款 Atari 游戏进行评估。 
尽管结果令人鼓舞，但没有任何算法能够在不依赖人类演示的情况下显著提高挑战性游戏的性能，而不会降低其余游戏的性能。
值得注意的是，在所有这些工作中，内在动机，特别是 Never Give Up 最近在改善探索难度大的游戏的表现方面显示出了巨大的希望。
NGU 通过用内部产生的内在奖励来增强奖励信号来实现这一点，这种奖励在两个层面上对新颖性很敏感：一个 episode 内的短期新颖性和 episodes 之间的长期新颖性。 
然后学习一系列用于探索和开发的策略(共享相同的参数)，最终目标是在开发策略下获得最高分数。
然而，NGU 并不是最通用的智能体：很像 R2D2 和 MuZero 能够在除了少数游戏之外的所有游戏上表现出色，而 NGU 也一样，因为它在较小的、不同于 MuZero 和 R2D2 的游戏上表现强劲 (尽管是基于 R2D2)。 例如，在游戏 Surround 中，R2D2 达到最优分数，而 NGU 更类似于随机策略。
NGU 的一个缺点是，它根据每一个策略收集了相同数量的经验，无论这些策略对学习进程的贡献如何。
有些游戏需要的探索程度与其他游戏有很大不同。 
直观地说，人们会希望分配共享资源 (网络容量和数据收集)，以使最终性能最大化。
文章建议允许 NGU 在智能体的整个生命周期内调整其探索策略，使其能够针对其正在学习的特定游戏进行专业化。 这是文章对 NGU 进行的第一个重大改进，使其成为更一般的智能体。

Recent work on long-term credit assignment can be categorized into roughly two types: ensuring that gradients correctly assign credit (Ke et al., 2017; Weber et al., 2019; Ferret et al., 2019; Fortunato et al., 2019) and using values or targets to ensure correct credit is assigned (ArjonaMedina et al., 2019; Hung et al., 2019; Liu et al., 2019; Harutyunyan et al., 2019). 
NGU is also unable to cope with long-term credit assignment problems such as Skiing or Solaris where it fails to reach 100% HNS. 
Advances in credit assignment in RL often involve a mixture of both approaches, as values and rewards form the loss whilst the flow of gradients through a model directs learning.

最近关于长期信用分配的工作可以大致分为两种类型：确保梯度正确分配信用和使用数值或目标来确保分配正确的信用。 NGU 也无法应对长期的信用分配问题，如滑雪或 Solaris，在这些问题上，它无法达到 100% 的HNS。 在强化学习中，信用分配的进步通常涉及两种方法的混合，因为价值和回报形成了损失，而通过模型的梯度流动指导了学习。

In this work, we propose tackling the long-term credit assignment problem by improving the overall training stability, dynamically adjusting the discount factor, and increasing the backprop through time window.
These are relatively simple changes compared to the approaches proposed in previous work, but we find them to be effective.
Much recent work has explored this problem of how to dynamically adjust hyperparameters of a deep RL agent, e.g., approaches based upon evolution (Jaderberg et al., 2017), gradients (Xu et al., 2018) or multi-armed bandits (Schaul et al., 2019). 
Inspired by Schaul et al. (2019), we propose using a simple non-stationary multi-armed bandit (Garivier & Moulines, 2008) to directly control the exploration rate and discount factor to maximize the episode return, and then provide this information to the value network of the agent as an input. 
Unlike Schaul et al. (2019), 1) it controls the exploration rate and discount factor (helping with longterm credit assignment), and 2) the bandit controls a family of state-action value functions that back up the effects of exploration and longer discounts, rather than linearly tilting a common value function by a fixed functional form.

在这项工作中，文章提出了通过提高整体训练稳定性、动态调整折扣因子、通过时间窗口增加向后传播来解决长期信用分配问题，这些与以前工作中提出的方法相比是相对简单的改变，但它们是有效的。
最近的工作探索了如何动态调整深度强化学习智能体的超参数的问题，例如基于进化算法、梯度或多臂老虎机的方法。
灵感来自 Schau l等人。 文章建议使用一个简单的非平稳多臂老虎机来直接控制探索率和折扣因子，以最大化回报，然后将这些信息作为输入提供给智能体的价值网络。 
与 Schaul 等人不同的是，1) 它控制探索率和折扣因子 (借助长期信贷分配)，2) 老虎机控制一族支持探索和较长折扣效果的状态动作值函数，而不是通过固定的函数形式线性倾斜公共价值函数。

In summary, our contributions are as follows: 
1. A new parameterization of the state-action value function that decomposes the contributions of the intrinsic and extrinsic rewards. As a result, we significantly increase the training stability over a large range of intrinsic reward scales.
2. A meta-controller: an adaptive mechanism to select which of the policies (parameterized by exploration rate and discount factors) to prioritize throughout the training process. This allows the agent to control the exploration/exploitation trade-off by dedicating more resources to one or the other.
3. Finally, we demonstrate for the first time performance that is above the human baseline across all Atari 57 games. As part of these experiments, we also find that simply re-tuning the backprop through time window to be twice the previously published window for R2D2 led to superior long-term credit assignment (e.g., in Solaris) while still maintaining or improving overall performance on the remaining games.

综上所述，本文贡献如下：
1. 对状态动作值函数进行了新的参数化，分解了内在奖励和外在奖励的贡献。因此，在很大范围的内在奖励范围内显著提高了训练的稳定性。
2. 元控制器：一种自适应机制，用于选择在整个训练过程中优先考虑哪些策略(由探索率和折扣因子参数化得到)。 这允许智能体通过将更多的资源分配给其中之一来控制探索/利用的权衡。
3. 最后，文章首次展示了在所有雅达利 57 游戏中高于人类基线的表现。作为这些实验的一部分，文章还发现，只需通过时间窗口将向后传播重新调整为以前发布的 R2D2 窗口的两倍，就可以获得优越的长期信用分配 (例如，在Solaris中)，同时保持或提高其余游戏的整体性能。

These improvements to NGU collectively transform it into the most general Atari 57 agent, enabling it to outperform the human baseline uniformly over all Atari 57 games. 
Thus, we call this agent: Agent57.

这些对 NGU 的改进共同将其转变为最通用的雅达利 57 智能体，使得它在所有雅达利 57 游戏中的表现都一致超过人类基线。
因此，本文称这个智能体为：**Agent57**。
