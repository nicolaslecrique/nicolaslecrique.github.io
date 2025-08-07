## Basic concept

# Reinforcement learning

* Agent
* States S: the prompt (conv)
* Actions A: the next token
* trabsition model P: the fact that you append the generate token to the sequence.
* reward R: eval of llm output at the end of a sequence, evaluated by a pre-trained reward model. reward is often a llm itself. the reward is mapped to a loss function.
* discount gamma: OSEF
* policy pi: the LLM itself, update the state (the prompt)
* value functions: reward model itself.s
* ppo: update policy

# Why use reinforcement

- step 1: next token prediciton: self-supervised: billions of tokens, but no ability to follow instruction
- SFT to follow instruction, but very expensive to label, and only simple ibjective (next token prediction), compared to complex objectives for reward (cohesive, nuanced, safe...). it's easier to train a model to evaluate a given response that providing a ground truth

## Why use SFT before

* provide a better starting point with more consistant reponses, reduce variance in response
* reduce the need for human feedback
* solve issue of cold start problem with RL if output is really poor, feedback is meaningless.

# RLHF

reward trained on user

## RL

the problem in RL is: we have a model giving a score, but we cannot compute gradient of the score relative to the weights


## Basique solution

REINFORCE ?
MONTE CARLO

## PPO: plus stable

## DPO

reformulation plus simple
discussion on-policy off-policy

## GRPO

combine des avatanges

notion of verifiable rewards.


