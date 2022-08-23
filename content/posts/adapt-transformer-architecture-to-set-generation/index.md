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

In practice,

* VAEs are sensitive to [posterior collapse]. it makes them unsuitable for sets generation. For example, when generating a set of ingredients for a pie, it will select a bit of puff, shortcrust, and shortbread pastry instead of choosing one.
* GANs demand a lot of tuning and a big dataset to work. It makes them unsuitable for a lot of use cases.

This makes Autoregressive Models the most suitable model class for set generation. Among them, the Transformer architecture is by far the most efficient one and completely replaced previous models like RNNs.


## Transformer architecture

If you need an introduction to the Transformer in the context of sequence generation, the best explanation I know is [Here](https://jalammar.github.io/illustrated-gpt2/), so I won't detail it here. Let's just recapitulate the different layers to see what we need to change.

1) Embedding
2) Positional encoding
3) N times:
    3.1) Masked self-attention
    3.2) Feed Forward Neural Network
4) Softmax

* The loss function is, as usual for a softmax head, the cross-entropy loss.
* At generation time, we generate tokens one by one up to a special "end of sequence" token.


## First, take the transformer as is

Nothing prevent us to reuse the previous architecture as is, make it generate a sequence, then just ignore the order to get a set. It will work, but we complexify the job of our network by asking him to find an ordering pattern when none exists. We can do better.

## Get rid of the positional encoding

The positional encoding is here to give information about the position of the previous tokens to generate the next one.

Here all we care about is the set of previous tokens, irrelative to there order. Because positional encoding is just an addition to the Embedding vectors, is extremely easy to remove it.

## adapt the loss function: Soft cross-entropy

Here it's a bit trickier. At training time, the standard loss function is the cross-entropy between the output probability distribution of the model and the one-hot vector the correct next token.

If we remove any notion of order, the next correct token is any token belonging to the set and not yet generated.

Thankfully, there exists an extension of this loss function that calculates the cross-entropy between two distributions. We can then set as a target vector a vector where the probability is shared between all remaining elements of the set.

## Data augmentation: shuffle the order

Now that our model is completely indifferent to order, we can apply one obvious data augmentation technic: shuffle the order of the elements in a set.

This still give a little boost to the resulting error.


## Recap

To adapt the transformer to the generation of set. we made three modification:
* On the Network architecture, by removing the positional encoding
* On the loss function, by replassing cross-entropy by soft cross-entropy
* On the dataset: by shuffling elements of the set between each epoch





<!--

TODO:
* Explication generale
* Pourquoi pas GAN ou VAE
* Archi globale transformer => code de pytorch
* deux possibilités: ordre naturel ou pas d'ordre
* Suppression du positioning
* addaptation de la loss function






-- >