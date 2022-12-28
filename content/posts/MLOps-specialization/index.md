---
title: 'DeepLearning.AI MLOps Specialization notes'
date: 2022-12-27T10:48:07+02:00
draft: false
math: true
images: []
description: 'DeepLearning.AI MLOps Specialization notes'
resources:
- name: "featured-image"
  src: "featured-image.jpg"
---

<!--more-->

# DeepLearning.AI MLOps Specialization notes

## Course 1: Introduction to Machine Learning in Production

### Week 1: Overview of the ML Lifecycle and Deployment

#### Concept drift / Data drift

When data in production diverges from training data.

#### Steps of a ML project

* Scoping
* Data
* modeling
* Deployment

This is not a linear but an iterative process.

#### Deployment patterns

* __gradual (Canary) deployment__: Send an increasing proportion of traffic to the new version.
* __blue-green deployment__: Add a router between old and new algorithm then switch, when we have to switch all users at once but are able to rollback instantly. 
* __Shadow mode__: execute new version in the background to compare predictions to a human or a previous version

#### levels of automation

1. Human
2. Shadow mode
3. AI assistance: user checks all ML results
4. Partial automation: user checks only uncertain or specific predictions
5. Full automation

Steps 2 and 3 are called __Human in the loop__ AI.

#### Monitoring

Three kinds of metrics to monitor (with graphs and alarms on thresholds to define):
* Software: server load...
* Input: missing values, statistical distribution...
* Output: rate at which users override ML prediction...

When there is a pipeline with several ML components, we must define metrics for each one of them.


### Week 2: Select and Train a Model


