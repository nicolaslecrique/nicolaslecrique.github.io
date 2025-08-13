---
title: 'RLHF Book, Reinforcement Learning from Human Feedback, Nathan Lambert'
date: 2025-08-06T10:48:07+02:00
draft: true
math: true
images: []
description: ''
resources:
- name: "featured-image"
  src: "featured-image.jpg"
---


# RLHF Book, Reinforcement Learning from Human Feedback, Nathan Lambert

## 1 Introduction

* Standard LLM training:
    1. Pretraining (self-supervised next token prediction): General knowledge acquisition
    2. Instruction / Supervised Finetuning (IFT/SFT): Learn to follow instructions, make it conversational and helpful. Starting point for RLHF.
    3. RLHF, possibly several loops reward training / model training
    4. RLVR (Verifiable Reward)

* Why RLHF:
    * Generalize better than SFT
    * Need less labeled data
    * Train on the entire answer than on the next token
    * Work well in the absence of a unique Ground Truth (only need preference)
    * ...But require a strong starting point (hence SFT)

## 3 Definitions & Background

* KL Divergence from P to Q: measure how different proba distribution Q is different from reference (true) distributions P on space $X$

$D_{KL}(P|Q)=\sum_{x \in X} P(x) log \frac{P(x)}{Q(x)}$


### NLP

* $x$ a prompt, $y$ a completion
* $\theta$: the parameters of the llm
* $\pi_{\theta}: \pi_{\theta}(y|x)$, the __policy__ (the llm), a probability distribution over all possible completions. In RL, it is the __Agent__ ($\pi(a|s)$).

### RL

 
* __State__: $s_i$, a prompt in our case
* __Action__: $a_i$, a completion in our case
* __Reward function__: $r:r(s_i,a_i)=r_i \in \real$. In our case it's a model, we note it $r_{\theta_r}$ with $\theta_r$ its parameters.
* __Trajectory__ $\tau:=(s_0, a_0, s_1...) \in \pi_\theta$. In our case it's only a pair prompt-completion of the probability space. 
* __Objective function__: $J(\pi_{\theta})=E_{\tau \sim \pi_{\theta}}[r_{\theta_r}(s,a)]$. What we want to maximize. Here it's just finding the best model (parameterized by $\theta$) to maximize the expectation of the reward for all pairs prompt / completion. In general it would be over all trajectories with a time horizon and discount.

* __Bradley-Terry model of preference__: We associate to each event $y_i$ the scalar $\beta_i$ such that $P(y_i > y_j)=\sigma(\beta_i-\beta_j)$, with $\sigma(x)=\frac{1}{1+e^{-x}}$ the sigmoid function.


## 4 Training Overview


* RL training loop adapted to RLHF on LM

![RLHF standard loop.png](rlhf_rl_standard_loop.png)


## 6 Preference Data

* __Likert scale__: ranking between 2 options with degrees (strongly prefer A, weakly...)

## 7 Reward Modeling

* __Standard architecture__: Linear head on top of an LLM, computed on the EOS token gives us a scalar $r_i$
* __Loss__: We use the __Bradley-Terry model of preference__ with $\beta$ being the output of the reward model: $P(y_i > y_j)=\sigma(r_{\theta_r}(x,y_i)-r_{\theta_r}(x,y_j))$ and derive the loss by minimizing the negative log-likelyhood to find the argmin $\theta_r^*$.

* For Reasoning tasks, when an output is either correct / incorrect, we can replace it with:
  * __Output Reward Model (ORM)__: We train the reward model with a classical cross-entropy loss against a ground-truth binary classification. Usually on all generated tokens (not only on EOS)
  * __Process Reward Model (PRM)__: We also use cross-entropy (or 3 classes classifications correct / incorrect / neutral), but independently for each token marking the end of a reasoning step to label if it is correct.
* __Generative Reward Model (GRM)__: Using a prompt and an llm as a RM to prevent costly labelling to train a RM. Still less performant than a dedicated RM.

## 8 Regularization