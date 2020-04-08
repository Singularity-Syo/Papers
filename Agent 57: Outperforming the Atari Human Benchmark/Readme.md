# Agent 57: Outperforming the Atari Human Benchmark

## Authors

* Adrià Puigdomènech Badia
* Bilal Piot
* Steven Kapturowski 
* Pablo Sprechmann 
* Alex Vitvitskyi
* Daniel Guo 
* Charles Blundell

## Abstract

Atari games have been a long-standing benchmark in the reinforcement learning (RL) community for the past decade. 
This benchmark was proposed to test general competency of RL algorithms. 
Previous work has achieved good average performance by doing outstandingly well on many games of the set, but very poorly in several of the most challenging games. 
We propose Agent57, the first deep RL agent that outperforms the standard human benchmark on all 57 Atari games. 
To achieve this result, we train a neural network which parameterizes a family of policies ranging from very exploratory to purely exploitative. 
We propose an adaptive mechanism to choose which policy to prioritize throughout the training process. 
Additionally, we utilize a novel parameterization of the architecture that allows for more consistent and stable learning.

雅达利游戏在过去十年里已经成为强化学习社区里的一个长期基准。这个基准被用于测试强化学习算法的一般 *competency*。先前的工作已经在该集合中的许多游戏上取得了良好的平均性能，但在几个最具挑战性的游戏里表现得非常糟糕。文章提出了 Agent 57，第一个在所有的 57 个 Atari 游戏上超过了标准人类基准的深度强化学习智能体。为了实现这一结果，文章训练了一个神经网络，它参数化了一系列从具有高探索性到纯粹利用性的策略。文章提出了一个自适应机制用于在整个训练过程中选择需要有限考虑的策略。此外，文章利用一个新颖的架构参数化方法使得学习过程能够更加一致和稳定。
