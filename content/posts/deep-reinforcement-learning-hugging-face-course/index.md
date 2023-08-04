---
title: 'Notes from the Deep Reinforcement Learning Course by HuggingFace'
date: 2022-12-27T10:48:07+02:00
draft: true
math: true
images: []
description: 'Kubernetes conepts'
resources:
- name: "featured-image"
  src: "featured-image.jpg"
---

# Notes from the Deep Reinforcement Learning Course by HuggingFace

## UNIT 1. INTRODUCTION TO DEEP REINFORCEMENT LEARNING

### Definition

**Reinforcement learning** is a framework where an **agent** learns by doing **actions** within an **environment** and receiving **rewards**

### Return function

Let $r_t$ be the reward received at the time step $t$.The **cumulative reward**(=**return**) $R(\tau)$ received from time step $t$, following the **trajectory** $\tau$ (sequence of **states**/**actions**) with a **discount rate** $\gamma$ is defined by:

$R(\tau)=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}$

### Policy-based and Value-based methods

The policy $\pi$ associates to a state $s$ either:
* An action $a$: **deterministic** policy
* A probability distribution over actions: **stochastic** policy

In **policy-based** methods, we learn $\pi$ directly: the action we need to take.

In **Value-based** methods, we learn a **value function** that maps a state to the expected return and chooses the policy maximizing it.

Let $v_{\pi}$ be the *value function* when we follow the policy $\pi$, $R_t$ and $S_t$ random variables, respectively the reward and state at time $t$. We choose $\pi$ to maximize:

$v_{\pi}(s)=E_{\pi}[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}|S_t=s]$