---
title: "How to build a Transformer model from scratch with 4000 samples? A case study of ThisDishDoesNotExist."
date: 2022-07-15T10:48:07+02:00
draft: true
math: true
description: "Away from GAFAM and their trillion parameter models, how to harness the power of Deep Learning for the rest of us?"
images: []
resources:
- name: "featured-image"
  src: "featured-image.png"
---

Away from GAFAM and their trillion parameter models, how to harness the power of Deep Learning for the rest of us?

<!--more-->

> **NB**: This model was developed for the company Alantaya, which provides personalized meal plans to people with eating problems. I thank Alantaya and my manager Alain Bedu to let me use it in this article.


<!-- INTRODUCTION : what's in it for me -->

## Deep (Learning) into the void: a step-by-step guide to building a domain-specific model.

If you work on NLP or computer vision, building a new model usually follows the following script:

* Find a pre-trained model that does the closest thing to what you want
* fine-tune it on your limited data, and maybe add a few layers on top (the *head*) to make it do what you want

This is *Transfer Learning*, and [great libraries](https://huggingface.co/) exist to do just that.

But in my experience, outside of NLP and CV, there are a lot of domain-specific cases where Deep Learning can bring value, but where there is:

* No pre-trained or standard model
* No big Dataset

<!-- Genre de GIF SUR "I have a plan" -->

Building a model from scratch is a daunting task, so we need a plan. That's what we will do here, explaining at each step:
* The theory behind this step
* The actual implementation (in PyTorch) on the example of *ThisDishDoesNotExists*

## The Plan

1. Define the business objective
2. Explore the *scientific literature* for potential solutions
3. Define an *evaluation metric* to measure your progress
4. Implement a *baseline*: the simplest model that can work
5. Improve your model:
    * Architecture <!-- multi task learning -->
    * hyperparameters
    * Data augmentation
    * Loss function
    * Regularization <!-- label smoothing, dropout -->

## 1. Define the business objective

### Theory

Before going down the rabbit hole of Deep Learning, let's take a step back and think about what we want. This is important for two reasons:
* Based on your objective, a simpler model could do the trick, even a ruled-based one. And if so, you should NOT use Deep Learning as it will be more expensive to create, run, and maintain.
* While you will be working on the intricacies of your model, you might need to go back to your business objective to determine what is important and what is not.

### Implementation

For our use case, here is the business context:

##### Business context

* To create meal plans for people with eating problems, we need a large pool of recipes that are both tasty and meet harsh dietetic criteria.
* We already have 3000 recipes, but for a given person with its own set of constraints (e.g.: diabatic and gluten allergic), most of them are not usable. So we need even more recipes, but ideas are hard to come by.

##### Deliverable

To meet this challenge, we need a tool that generates recipe ideas. By recipe, we mean a set of ingredients with their respective quantities, plus its kind (starter, main course, dessert, snack). Those recipes can then be filtered and improved upon by Dietitians.

##### Non-goals

* We don't need cooking instructions. Only ingredients + quantities are needed to automatically select recipes with interesting nutritional properties.
* We don't need all generated recipes to be realistic, as they will be filtered manually afterward.

## 2. Explore the scientific literature for potential solutions

### Theory

You probably won't invent an entirely new deep learning architecture. The first thing you should do is look for standard machine learning tasks close to your use case.

Those are tasks for which there is an extensive literature on how to approach them.

> Here is the magic link to get an overview of standard tasks and models: [paperswithcode](https://paperswithcode.com/sota)


We might not be able to take the existing models as is, but you will draw inspiration from them.

> Good artists copy, great artists steal.

Once you get the most likely candidates, compare them with each other and to your task at hand.

* How complex are they?
* How much data do they need?
* Are there any blog posts or open source implementations to help you?
* How do they deviate from your task? 
* Can we manage those deviations?


### Implementation

In the case of *ThisDishDoesNotExists*, the most obvious feature of the task is that it's [generative](https://en.wikipedia.org/wiki/Generative_model). That is, we must generate new samples from an existing distribution.

##### Comparable tasks and models

A simple CTRL+F "generation" on [paperswithcode](https://paperswithcode.com/sota) will show you two common tasks on which there is plenty of research:
* [Image generation](https://paperswithcode.com/task/image-generation)
* [Text generation](https://paperswithcode.com/task/text-generation)

For those 2 tasks, there are 3 broad classes of models often used:
* Variational Autoencoders
* Generative Adversarial Network
* Autoregressive Transformer Models

The first two are mainly used for image generation while the transformer architecture is mainly used for text generation.

##### Specific constraints

* We only have access to a dataset of 4000 samples. This is a **very small dataset** for a generative model
* Unlike an image, if we model the recipe as a vector of quantities where each index of the vector corresponds to an ingredient, this vector is very **sparse**, about ten ingredients are activated (>0) out of a few thousand.
* Unlike an image, there is no geometry between ingredients (distance and continuity between pixels)
* Unlike a text, there is no order between ingredients. A Dish is a set, not a sequence.
* Unlike a text, a dish is not just a set of ingredients (like a sentence is a sequence of words), but a set of ingredients associated with quantities.
* On top of the set of ingredients and quantities, we must associate with each dish a type (starter...)


#### Comparison of model classes

There is no shortcut here. We must dig into the details of how models work to understand if and how we can adapt them to our needs.

Here is the gist of my conclusions:
* __Generative Adversarial Network (GAN)__ is more of a strategy to create a generator model than a specific model. As such, it can be adapted to any use case. The big drawback here is its complexity. It demands a lot of tuning to make it work and usually needs a huge amount of samples.
* __Variational Autoencoders (VAE)__ is a lot simpler and adapted to smaller datasets. but it's not well suited to sparse or sequential data.
* __Autoregressive Transformers__ is of intermediate complexity. While it is used in the literature in the context of huge datasets, it isn't intrinsically limited to those. It's also, by default, suited to sequential data (instead of sets) and cannot generate quantities, but a study of the model details shows that it can be adapted.

|                               | GAN       | VAE    | Transformer |
|-------------------------------|-----------|--------|-------------|
| Complexity                    | High      | Low    | Medium      |
| Adaptability to sparse data   | Medium    | Low    | High        |
| non sequential data           | Yes       | Yes    | Adaptable   |
| data without geometry         | Adaptable | Yes    | Adaptable   |
| generate quantities           | Yes       | Yes    | No          |


Because it's simpler than GANs and not subject to [Posterior Collapse](TODO) like VAEs, we will study further the Transformer model in the continuation of the article.

> Remark: because it's so simple to implement, I actually made a model based on a VAE architecture, but I could not allievate the Posterior Collapse issue enough to make it usefull. In practical terms, it means that  

## 3. Define an *evaluation metric* to measure your progress





Generate ideas of potential recipe

 of *ThisDishDoesNotExists*:




TODO NOT SEQUENTIEL, on passe iterativement par toutes les étapes.


EN COURS:
when you are left on your own, the task is a bit more daunting
What you need is a plan, and follows ideas if you cannot copies models
or ready-made solutions.




When possible 


The current process to build a deep learning is often the following



If you know one concept 

There is three kinds of jobs in the deep learning world.


Faire une intro en bullet points ?

RNN is dead. Transformer architecture took NLP by storm, improving state of the art in all NLP tasks by leaps and bounds. Computer vision is following the same path. 


Transformer architecture

By far, the most common usage 


parler de TRANSFERT LEARNING

<!-- description de la problématique -->

<!-- VERSION LA PLUS SIMPLE : vue haut niveau -->


<!-- VERSION SIMPLE: vue détaillée -->

<! -- Ajout par ajout: code + schema>







<!---

il faut trouver dans la litérature les idées ! puisque on ne peut pas pomper le model , il faut pomper les idées derrière le model.

IMPORTANCE DU COTE ITERATIF: Le DL EST UN ART AUANT QU'UNE SCIENCE

> Good artists copy, great artists steal.

accer la reflexion sur les adaptation necessaire pour un petit dataset:
-data augmentation
-technique pour prevenir l'over fitting: dropout, label smoothing, plus petite taille de reseau

- a chaque étape, on regarde comment evolue la validation loss
- a chaque étape on montre le code, on link vers un article "pour aller plus loin"



case study
pytorch
tought process
domain-specific constraints
small dataset
DL / Transformers for the rest of us


A la fin: en vrac 
* ce que j'ai essayé et ce qui n'a pas marché
* ce que j'aurais pu essayer d'autres

Small dataset, domain-specific constraints: Deep Learning and Transformers for the rest of us

tought process

why the user would read this article ?
google keyword ? headline from HN: what does he want to know ?


2 pistes pour utiliser transformers ouside google:
transfer learning, ou pb très specifique: pleins de ressources sur le 1er

- Description de l'objet du poste
https://www.youtube.com/watch?v=YODPgBadj80

1) Substance: write something that matters: always new original spins in topics: experience, language, humor; Normal to be hard
2) Packaging: important: headline, lead image, iamges, formatting, platform.
* A good headline makes all the difference
* people decide to read on the headline. extremelly important (A/B Test)
* Images are very important: one image every 350 words: pexels.com
* Formatting: Break with headlines and quotes: nice short paragraphes
3) Publicizing: last 10% of hte time, 90% of the work: HN lottery, specific target groups, linkedin, use other people audience


what the reader should gain from reading the article ?

- Define the purpose
- Titre: number, "you", "how to"

https://medium.com/quark-works/tips-on-how-to-write-your-first-successful-technical-blog-4cb65e5b4ce4


https://www.freecodecamp.org/news/how-to-write-a-great-technical-blog-post-414c414b67f6/


*Audience*: People knowing how to build DL models by making marginal modifications to existing models, and wanting to apply DL to an entirely new domain and with little data

*Goal*: a case study. Show How "I" did it. The iterative process. the *thought* process

====
Intro: give context, tell your audience what they will get


-->




# How are you

fine

## Subtitle

finefine

### Subtitle 2

 Some content

#### subtitle 3

**gras**
*italic*
~~barré~~
__lol__

>retrait
>retrait2


```python
def fix_hostname(hostname):
    hostname = re.sub(r'[\\/:"*?<>|]+', "-", hostname)
    hostname = hostname.lower()
    return hostname

hostname = fix_hostname(device["name"])
```


 ```js

func GetTitleFunc(style string) func(s string) string {
  switch strings.ToLower(style) {
  case "go":
    return strings.Title
  case "chicago":
    return transform.NewTitleConverter(transform.ChicagoStyle)
  default:
    return transform.NewTitleConverter(transform.APStyle)
  }
}

 ```


$\int_{-\infty}^{\infty} e^{-x^2} dx$.

$$\int_{-\infty}^{\infty} e^{-x^2} dx$$.