# Dream to Control: Learning Behaviors by Latent Imagination

Authors: Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi

Conference: ICLR 2020

## Abstract

Learned world models summarize an agent’s experience to facilitate learning complex behaviors. While learning world models from high-dimensional sensory inputs is becoming feasible through deep learning, there are many potential ways for deriving behaviors from them. We present Dreamer, a reinforcement learning agent that solves long-horizon tasks from images purely by latent imagination.
We efficiently learn behaviors by propagating analytic gradients of learned state
values back through trajectories imagined in the compact state space of a learned
world model. On 20 challenging visual control tasks, Dreamer exceeds existing
approaches in data-efficiency, computation time, and final performance.

## 1. Introduction

Notation：

- World Model
  
- Latent Dynamics Model
  
- Behavior
  
- Dreamer
  
- Bellman Consistency：强化学习建立在不动点基础上，一致性意思是策略和价值的对应。此处应该指贝尔曼最优方程。
  
- Analytic Gradient：
  

Summary：

## 2. Control with World Models

Notation：

- POMDP
- Observation
- Interleaved
- Markovian Transitions
- Latent Imagination
- non-linear Kalman filter
- HMM

Summary：

## 3. Learning Behaviors by Latent Imagination

Notation：

- tanh-transformed Gaussian
- Reparameterized Sampling
- Stochastic Backpropagation
- Straight-through Gradient
- A3C
- PPO
- DDPG
- SAC
- MVE
- STEVE
- D4PG

## 4. Learning Latent Dynamics

Notation：

- Contrastive Estimation
- PlaNet
- Variational Lower Bound (ELBO)
- Variational Information Bottleneck (VIB)
- KL Regularizer
- Recurrent State Space Model (RSSM)
- Transposed CNN
- Observation Marginal
- State Marginal
- Noise Contrastive Estimation (NCE)
- Pixel Prediction

## 5. Related Work

Notation:

- Derivative-free Policy
- E2C
- RCE
- SOLAR
- I2A
- Belief Representation
- VPN
- POLO
- PETS
- VisualMPC
- POPLIN
- SVG
- ME-TRPO
- DistGBP

## 6. Experiments

Notation:

- Orthogonal Choice
- DeepMind Control Suite
- DeepMind Lab levels
- TensorFlow Probability
- Nvidia V100 GPU
- SimPLe
- Rainbow
- IMPALA

## 7. Conclusion

We present Dreamer, an agent that learns long-horizon behaviors purely by latent imagination. For this, we propose an actor critic method that optimizes a parametric policy by propagating analytic gradients of multi-step values back through learned latent dynamics. Dreamer outperforms previous methods in data-efficiency, computation time, and final performance on a variety of challenging continuous control tasks with image inputs. We further show that Dreamer is applicable to tasks with discrete actions and early episode termination. Future research on representation learning can likely scale latent imagination to environments of higher visual complexity.

## A. Hyperparameters

## B. Derivations

## C. Discrete Control

## Blog

![](https://1.bp.blogspot.com/-4J0POdpDz8U/XnFFZ_POSXI/AAAAAAAAFfg/3Dzzf-nbPUQuYWiJuzbZK__vfjHTjYtrQCLcBGAsYHQ/s1600/image2.png)

![](https://1.bp.blogspot.com/-_zknTMFclfs/XnFFc0j_aLI/AAAAAAAAFgE/c1-Lzjr0SXA41bXGM99kaFWmvuy4IdnBACEwYBhgL/s640/image6.gif)
