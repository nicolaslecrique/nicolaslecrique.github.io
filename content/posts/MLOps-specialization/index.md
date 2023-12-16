---
title: 'DeepLearning.AI MLOps Specialization notes'
date: 2022-12-28T10:48:07+02:00
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
* __blue-green deployment__: Add a router between the old and new algorithm then switch, when we have to switch all users at once but are able to rollback instantly. 
* __Shadow mode__: Execute new version in the background to compare predictions to a human or a previous version

#### levels of automation

1. Human
2. Shadow mode: ML runs in parallel with a Human to assess its performance.
3. AI assistance: user checks all ML results
4. Partial automation: user checks only uncertain or specific predictions
5. Full automation

Steps 3 and 4 are called __Human in the loop__ AI.

#### Monitoring

Three kinds of metrics to monitor (with graphs and alarms on thresholds to define):
* Software: server load...
* Input: missing values, statistical distribution...
* Output: the rate at which users override ML prediction...

When there is a pipeline with several ML components, we must define metrics for each one of them.


### Week 2: Select and Train a Model

#### Literature overview

Look for existing or comparable solutions (open-source, papers...).

#### Model VS Data-centric approach

* __Model-centric__: Improve model while keeping data fixed, most literature uses this.
* __Data-centric__: Improve data while keeping the model. Often useful in practice, because good data on a correct model is often better than worst data on an excellent model

#### 3 Steps on modeling and common problems

1. Doing well on training set
2. Doing well on dev/test set
3. Doing well on business/project metrics

Common problems between 2 and 3.:
* Bad performance on disproportionately important cases
* Skewed data distribution: algorithm has a good average performance but is bad on rare classes.
* Bias / Discrimination

#### Establish a baseline

Possible baselines:
* Human Level Performance (HLP): often useful on unstructured data (text, images...), where Human are naturally good.
* Open-source solution
* Quick-and-dirty implementation
* Previous system

The baseline can be divided on some data categories, it is useful:
* To get an idea of what is possible. e.j: it's hard to beat HLP on some tasks.
* To know what to focus on.
* To get an idea of the irreducible (Bayes) error

#### Sanity check

Start by trying to overfit a small dataset (or even one sample) in training. 

It helps to find big issues or bugs quickly, it's useless to continue if it doesn't work on one sample.

#### Error analysis

Error analysis is used to understand how a model makes errors and how to improve it.

Here is the process:
1. Go manually through a random sample of bad predictions (10s, 100s)
2. Note in a spreadsheet some distinctive tags/characteristics for each one of them: "background noise", "bad labeling", "value X on a specific feature"...
3. Decide to work on a specific issue based on frequency, importance or ease of fix.

#### Skewed dataset

Loss is not always a good indicator of a skewed (unbalanced) dataset, you can compute [Precision, Recall and F1-Score](https://nicolaslecrique.github.io/posts/statistics-machine-learning-basics/#precision-recall-and-f1-score) to assess quality (for each class in a multi-class task).

#### Performance auditing & Data Slicing

1. brainstorm possible problems and subsets where the model could go wrong(bias, fairness...)
2. Slice data in several subsets and check performance metrics on each one.

#### Data Augmentation

Adding synthetic training samples (add background noise for speech recognition...). It's useful for unstructured data if it respects 3 conditions:
* Synthetic samples are realistic
* Algo does poorly on those
* Humans (or another baseline) do well on those.

It usually doesn't degrade performance on other data (very different from synthetically generated data) if the model is large enough (low bias).

### Adding feature

Adding features. it's mainly useful for structured (and limited) data. E.g: adding % of eaten meals containing meat to help a recommender system stop recommending meat restaurants to vegans.

### Experiment tracking

it's important to keep track of experiments, it can be through a shared spreadsheet or a dedicated experiment tracking system (like [Weights & Biases](https://wandb.ai/site))

Important features are:
* Infos for __replicability__: code version, dataset, hyperparameters...
* Results (metrics, trained model...)

#### Good data

A good dataset:
* has good coverage (covers all important cases)
* has consistent and unambiguous labeling 
* is monitored for data/concept drift


### Week 3: Why is data definition hard?

#### structured vs unstructured, small vs big data

* With unstructured data, you can often ask to label more data and use data augmentation.
* Consistent labeling is always useful, but crucial on a small dataset (<10k samples)
* On a small dataset, you can fix labels one by one, on a big dataset, you must improve the process (data extraction, labeling instructions)
* big data sets can also have small data problems due to a long tail of rare events (like in search or product recommendation)

#### Strategies to get consistent labeling?

* Define clear labeling instructions
* Add a new class to capture uncertainty
* Improve input (more or better info) if labeling is hard
* have several labelers label the same example to detect ambiguities in labeling
* Merge classes if the distinction is not clear and unnecessary
* If ambiguities cannot be fixed, use voting from multiple labelers

#### Human Level Performance (HLP)

* When __ground truth__ is defined by humans, HLP is often just a measure of ambiguity in labeling.
* Labeling inconsistencies make it both irrelevant and too easy to beat HLP 
* It's useful to improve labeling consistency to improve both the model performance and the HLP to compare with.


#### How much data to get?

* At the start, the minimum you need to iterate as soon as possible on the 'data->model->error analysis' loop.
* Then increase if needed, e.g. 2x (10x max), to not spend more time and money than needed. 


#### Data Inventory process

4 possible sources of data:
* Already owned
* Crowdsourced
* Pay for labelers (in-house or outsourced)
* Purchase existing dataset

Compare each possible source on:
* Cost
* Time
* Quantity
* Quality
* Privacy and regulatory constraints

#### Who will label?

* Do you need __SMEs (Subject Matter Experts)__?
* Good practice: ML Engineers should do a small part of the labeling to understand the data and possible labeling issues.

#### Data Pipeline

Data often needs pre-processing steps (cleanup...). After the POC phase, it's important to ensure that the data pipeline is replicable and identical between modeling (training data) and production (for inference).

#### Data lineage, data provenance, meta-data

The pipeline can be a complex graph of steps and ML algorithms. For future diagnostic purposes, it's important to keep track of:
* __Data provenance__: where does the data come from
* __Data lineage__: what are the steps that have been involved in the production of each final data/prediction
* __Meta-data__: what are the info associated with the production of the data (web browser, camera, localization...) 

#### Balanced train/dev/test set

On a small dataset, randomly splitting samples between train/dev/test can lead to a different proportion by class between train/dev/test sets. It's better to split sets such as each set has the same proportions.


#### Scoping

To choose what to work on:
1. Brainstorm __business__ problems
2. brainstorm potential AI solutions
3. Assess the technical feasibility (using literature, and benchmarks...) and business value of solutions (ROI)
4. Define key metrics (ML, Software and business). Find shared metrics on which business and tech people can agree.
5. budget resources.
6. Define a timeline and milestones

