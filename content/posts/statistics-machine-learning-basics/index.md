---
title: 'Statistics and Machine Learning, the basics'
date: 2022-07-15T10:48:07+02:00
draft: true
math: true
images: []
description: null
resources:
- name: "featured-image"
  src: "featured-image.jpg"
---

The objective of this post is to recapitulate the basics of statistics and machine learning, following a glossary layout, in the most compact form



# ---- Statistics Principles



## Regression, classification, clustering

* __Regression__: labels are continuous
* __Classification__: labels are categories
* __Clustering__: group data into clusters

## Supervised, unsupervised, self-supervised learning

* __Supervised__: data is labeled. e.g. *regression*, *classification*
* __Unsupervised__ : data is unlabeled, it is used to find patterns in data. e.g *clustering*
* __self-supervised__: label is in the data. e.g *language modeling*.

## Loss function / Cost function

A function (to minimize) that measures how well model predictions match labels. It should be *Lipschitz continuous* (derivative is bounded) for *gradient descent*.

## Gradient descent

Optimization algorithm used in Deep Learning.

Let *a* be the vector of all model parameters and *L(a)* the *loss function*

We compute *a* iteratively using:

a = a - learning_rate * gradient L(a)




##  ------------- TODO ------------

* Language modeling
* gradient descent
* Back propagation
* Learning rate
* Learning rate schedule


## Bias

## Variance

## Bias - Variance tradeoff

## Covariance

## Supervised, unsupervised, semi-supervised

## Test d'hypothese

## interval de confiance

## Cross-validation

## overfitting

## training-set, validation-set, test-set

# Machine learning principles

# ---- Algorithm

## ---- MCO