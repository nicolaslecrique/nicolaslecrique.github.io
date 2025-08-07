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
* $\pi_{\theta}: \pi_{\theta}[y|x]$, the __policy__ (the llm), a probability distribution over all possible completions. In RL, it is the __Agent__ ($\pi(a|s)$).

### RL


 
* __State__: $s_i$, a prompt in our case
* __Action__: $a_i$, a completion in our case
* __Reward function__: $r:r(s_i,a_i)=r_i \in \real$



## 4 Training Overview


* RL training loop adapted to RLHF on LM

![RLHF standard loop.png](rlhf_rl_standard_loop.png)

