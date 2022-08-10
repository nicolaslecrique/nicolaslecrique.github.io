---
title: 'How to adapt the Transformer architecture to generate Sets'
date: 2022-07-15T10:48:07+02:00
draft: true
math: true
images: []
description: null
resources:
- name: "featured-image"
  src: "featured-image.jpg"
---

Generation is one of the most complex tasks in Machine Learning. Most of the literature is about images and text. But how do you make a generative model that creates sets?

<!--more-->

## The problem with set generation?

What I mean by set generation is the ability to generate a subset of a much bigger set following an implicit probability distribution.

An image is not a set of pixels, because there is a notion of geometry between them.

A text is not a set of words, it's a sequence, because there is a notion of order between words.

An example of a set is the list of ingredients in a dish. This is actually the use case for which I had to make a generative model.

This absence of natural order or geometry makes it impossible to directly use the models used for texts and images

## Potential architectures: VAE, GAN and AR

There are 3 broad classes of generative models commonly used:
* Variational Autoencoders
* Generative Adversarial Network
* Autoregressive Models

The first two are mainly used for image generation while the transformer architecture is mainly used for text generation.

In theory, all 3 can be used for sets generation:
* VAEs generate classical feature vectors. A subset can be represented as a vector of zeros and ones where each index represents an element of the set.
* GAN is more of a strategy to create a generator model than a specific model. As such, it can be adapted to any use case.
* AR can be used to generate sets by just ignoring the order of the generated sequences.





The big drawback here is its complexity. It demands a lot of tuning and data to make it work.

<!--

TODO:
* Explication generale
* Pourquoi pas GAN ou VAE
* Archi globale transformer => code de pytorch
* deux possibilités: ordre naturel ou pas d'ordre
* Suppression du positioning
* addaptation de la loss function






-- >