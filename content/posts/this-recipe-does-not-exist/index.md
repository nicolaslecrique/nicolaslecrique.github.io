---
title: "How to build a Transformer model from scratch with 3000 samples? A case study of ThisDishDoesNotExist."
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

## Deep (Learning) into the void: a step-by-step guide to build a domain-specific model.

If you work on NLP or computer vision, building a new model usually follows the following script:

* Find a pre-trained model that does the closest thing to what you want
* fine-tune it on your limited data, maybe add a few layers on top (the *head*) to make it do what you want

This is *Transfer Learning*, and [great libraries](https://huggingface.co/) exists to do just that.

But in my experience, ouside of NLP and CV, there is a lot of domain-specific cases where Deep Learning can bring value, but where there is:

* No pre-trained or standard model
* No big Dataset

<!-- Genre de GIF SUR "I have a plan" -->

Building a model from scratch is a daunting task, so we need a plan. That's what we will do here, explaining at each step:
* The theory behind this step
* The actual implementation (in PyTorch) on the example of *ThisDishDoesNotExists*

## The Plan

1. Define the business objective
2. Specify your technical constraints
2. Explore the *scientific literature* for potential solutions
3. Define a *evaluation metric* to measure your progress
4. Implement a *baseline*: the simplest model that can possibly work
5. Improve your model:
    * Architecture <!-- multi task learning -->
    * hyperparameters
    * Data augmentation
    * Loss function
    * Regularization <!-- label smoothing, dropout -->

## 1. Define the business objective

### Theory

Before going down the rabbit hole of Deep Learning, let's take a step back and think about what we actually want. This is important for two reasons:
* A simpler model could do the trick, even a ruled-based one. And if so, you should NOT use Deep Learning as it will be more expensive to create, run, and maintain.
* While you will be working on the intricacies of your model, you might need to go back to your business objective to determine what is important and what is not.

### Implementation

For our use case, here is the business context:

* To create meal plans for people with eating problems, we need a large pool of recipes that are both tasty and meet harsh dietetic criteria.
* We already have 3000 recipes, but for a given person with its own set of constraints (e.g.: diabatic and gluten allergic), most of them are not usable.
* We need even more recipes, but ideas are hard to come by.

> To meet this challenge, we need a tool that generates recipe ideas. Those ideas will then be filtered and improved upon by Dietitians.



## 2. Specify your technical constraints




============





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