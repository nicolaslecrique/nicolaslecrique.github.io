---
title: 'Statistics and Machine Learning, the basics (WIP)'
date: 2022-12-12T8:48:07+02:00
draft: false
math: true
images: []
description: null
resources:
- name: "featured-image"
  src: "featured-image.jpg"
---

The objective of this post is to recapitulate the basics of statistics and machine learning following a glossary form.

<!--more-->

# Basic Maths 

## Eigenvalues, Eigenvectors and Eigendecomposition (Singular Value Decomposition)

$v$ is an eigenvector and $\lambda$ an eigenvalue for a square matrix $M$ if:

$Mv=\lambda v$

$M$ is called *diagonalizable* if $\exists D$ diagonal, $P$ invertible such a:

$M=PDP^{-1}$

In such case, $P$ column vectors are the *eigenvectors* and $D$ diagonal values are the *eigenvalues*. This is the *eigendecomposition* of $M$.

Some interesting properties:
* All symmetric matrix (i.e. $M=M^T$) is diagonalizable by an orthogonal matrix (i.e. $P^TP=PP^T=I$, and so $P^T=P^{-1}$ and column/row vector norms =1)
* $M^n=PD^nP^{-1}$ and so $M^TM=PD^2P^{-1}$ and $M^{-1}=PD^{-1}P^{-1}$

## Sphering (Whitening) of a Matrix

Let $X$ be a $(p,n)$-dimensional matrix of $n$ samples and $p$ features, we suppose it's centered.

We are looking for a transformation (a basis) in which the dimensions are uncorrelated and of unit variance. that is, we are looking for a $(p,p)$-matrix $W$ such as

$Y=WX$

and

$Cov(Y):=\frac{1}{n}YY^T=I$

with $I$ the identity matrix. We have:

$Cov(X):=\frac{1}{n}XX^T=PDP^{-1}$

with $P$, $D$ the SVD decomposition of $Cov(X)$.

So by replacing $Y$ by $WX$, then $XX^T$ by the SVD, we are looking for $W$ such as:

$WPDP^{-1}W^T=I$

It is solved by taking $W:=D^{-\frac{1}{2}}P^{-1}$

It exists and is easy to compute because $D$ is diagonal.

[A good reference](https://theclevermachine.wordpress.com/2013/03/30/the-statistical-whitening-transform/)

# Descriptive statistics

## Hypothesis Testing and P-Value

We start with a null hypothesis H0. The alternate hypothesis is H1.

The *P-Value* is:

$P$(Observed Results are consistent with H1 | H0 is true).

We usually fix a threshold $p$ (e.g. 5%), and we reject the null hypothesis if P-value < $p$.

## Confidence interval 

For a given level $\gamma$ (e.g. 95%) and a estimated parameter $\theta$, a *confidence interval* for $\theta$ is a range $[\hat{\theta}^{min}, \hat{\theta}^{max}]$ such as

$P(\hat{\theta}^{min} < \theta < \hat{\theta}^{max})=\gamma$

$\theta$ is deterministic but unknown, the confidence interval is random and observed from an experience. It means that if we redo the experiment 100 times, the confidence interval will contain $\theta$ in average $\gamma$% of the times.

In Practice, $\theta$ might be the mean of the distribution of a random variable $X$, and we will observe a great number of independent realizations of $X$ and construct a confidence interval around the observed average.


# Statistical Learning Principles


## Regression, classification, clustering

* __Regression__: labels are continuous
* __Classification__: labels are categories
* __Clustering__: no labels, group data into clusters

## Supervised, unsupervised, self-supervised learning

* __Supervised__: data is labeled. e.g. *regression*, *classification*
* __Unsupervised__ : data is unlabeled, it is used to find patterns in data. e.g *clustering*
* __self-supervised__: labels are in the data itself. e.g *language modeling*.

## Loss function / Cost function

A function (to minimize) that measures how well model predictions match labels. It should be *Lipschitz continuous* (bounded derivative) for *gradient descent*.

### Mean Square Error

Usual loss function for regressions. It penalizes heavily big prediction errors.

$MSE=\frac{1}{n}||Y-\hat{Y}||^2=\frac{1}{n}\sum_{i=1}^n (Y_i-\hat{Y}_i)^2$

### Cross Entropy

Usual loss function for classifications. It's equivalent to maximizing the *likelihood* of the distribution.

$ CE=-\frac{1}{n}\sum_{i=1}^n  Y_i \cdot \ln{\hat{Y_i}}$

with $Y_i$ being a one-hot vector for the $i$-th sample with $1$ on the expected class

## Training set, Validation set, Test set

* __training set__: to train the model. ~75% of the data
* __validation set__: to check that you didn't *overfit* your model to the *training set*. ~20% of the data
* __test set__: to check that you didn't *overfit* your model hyper-parameters to your *validation set*. ~5% of the data.


## Bias, Variance and bias-variance trade-off

We take the hypothesis that we can decompose any output $y$ for an input $x$ as:

$y=f(x)+\epsilon$

with $f$ being the true function to approximate and $\epsilon$ a random variable (called *irreductible error*) such as $E[\epsilon]=0$ and $Var(\epsilon)=\sigma^2$

When computing *MSE* of the estimator $\hat{f}$, we get that:

$MSE=Bias(\hat{f})^2+Var(\hat{f})+\sigma^2$

with:

$Bias(\hat{f})=E[\hat{f}-f]$

$Var(\hat{f})=E[(\hat{f}-E[\hat{f}])^2]$

*Bias* is due to underfitting (lack of power of the algorithm), *Variance* is due to *overfitting*.

## Overfitting and underfitting

There is *overfitting* when the *loss* is a lot better on the *training set* than on the *validation set*. It means that the learned function doesn't generalize well to unseen data. Common solutions are *regularization* techniques, using a simpler model, *feature selection*, *dimensionality reduction*, *early stopping*, *dropout*...

There is *underfitting* when the *bias* is high and the *loss* is too high even on the training set. The common solutions are *feature engineering* and using a better more powerful model.

# Base algorithms

## Principal Component Analysis

Standard dimension reduction technique.

Let $X$ be a $(n,p)$-dimensional matrix of $n$ samples and $p$ features.

After having centered-reduced $X$ (to scale features), we want to find the sequence $(w_1, w_2...w_p)$ of $p$-dimensional orthogonal unit vectors (let's call it the $W_p$ orthogonal matrix) such that for any $q<p$, $W_q:=(w_1,..w_q)$ is the $q$-basis that minimizes the MSE of the distance between $X$ and it's projection on the suspace generated by $W_q$.

We want to minimize:

$\frac{1}{n}\sum_{i=0}^n |x_i-W_q<x_i,W_q>|^2=\frac{1}{n}\sum_{i=0}^n |x_i-W_q(W_q^Tx_i)|^2$   

We [can show](https://leimao.github.io/article/Principal-Component-Analysis/) that the optimal solution is found by applying *SVD* to the covariance matrix $\frac{1}{n}X^TX$ and then to choose $W$ as its *eigenvectors* by decreasing orders of *eigenvalue*.

The final reduced matrix is the matrix $XW_q$, it's a $(n,q)$-dimensional matrix. 


## Least squares

In a regression problem, let $Y$ be the vector of the $n$ expected outputs and $X$ the $(n,p)$ input matrix. We suppose that the output can be approximated by a linear combination of the inputs. We are looking for the $p$-dimensional vector $\beta$ to minimize the MSE:

$||Y-X \beta||^2$

If we place ourselves in a $n$-dimensional vector space and call $V$ the $p$-dimensional subspace generated by $X$. Minimizing MSE is equivalent to finding $\beta$ such as $X\beta$ is the projection of $Y$ on $V$. That is:

$<Y-X\beta, Xb>=0, \forall b \in \mathbb{R}^p$

$(Xb)^T(Y-X\beta)=0, \forall b$

$b^TX^T(Y-X\beta)=0, \forall b$

$X^T(Y-X\beta)=0$

$X^TY=X^TX\beta$

We conclude that

$\beta=(X^TX)^{-1}X^TY$

## Logistic regression

In a binary classification problem, let $Y$ be the vector of the $n$ expected outputs, with values $0$, $1$ for respectively the first and second class, and $X$ the $(n,p)$ input matrix.  We are looking for the $p$-dimensional vector $\beta$ to minimize the cross-entropy $CE(Y,\hat{Y})$ with:

$\hat{Y}=\frac{1}{1+\exp{X \beta}}$

We can show it's a convex optimization problem (the second derivatives of CE with respect to all $\beta_j$ is positive) and can be solved with convex optimization algorithms (Newton...)

## Linear/Quadratic Discriminant Analysis (LDA/QDA)

LDA is a supervised classification algorithm.

The hypothesis is:
* Each class $k$ has a multivariate gaussian distribution with density $f_k(x):=P(X=x|C=k)$ with mean $(p)$-vector $\mu_k$ and covariance $(p,p)$-matrix $\Sigma_k$
* $P(C=k)=\pi_k$
* __For LDA__: all classes have the same covariance matrix $\Sigma$, only the mean vectors $\mu_k$ are differents

For a given input vector $x$, we want to find the class $k$ such that:

$k:=arg max_{k=1..K} P(C=k|X=x)$

We get from *Bayes Theorem* that:

$P(C=k|X=x)=P(X=x|C=k)\frac{P(C=k)}{P(X=x)}=f_k(x)\frac{\pi_k}{\sum_{l=1}^{K}f_l(x)\pi_l}$

Since we just want to compare them, we can compute $\log (f_k(x)\pi_k)$ for every $k$ and remove factors common to all $k$. This gives us for every $k$ the *quadratic discriminant functions*:

$\delta_k(x)=-\frac{1}{2}\log|\Sigma_k|-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)+\log\pi_k$

To compute this function, we can see that we need to compute:
* The inverse of the covariance matrix
* The determinant $|\Sigma_k|$ of the covariance matrix

To simplify this computation, we would like to find a basis in which it's enough to compute the distance to the *centroid* ($\mu_k$) of the distributions

This is done by *Sphering* (*Whitening*) the data. Let:

$\Sigma_k=U_kD_kU_k^T$

be the SVD decomposition, we can derive the computation from the expression of $\delta_k(x)$ to see that:



$\delta_k(x)=-\frac{1}{2}\log|\Sigma_k|-\frac{1}{2}||X^\*-\mu_k^\*||^2+\log\pi_k$


with $X^\*$ and $\mu_k^\*$ be the transform of $X$ and $\mu_k$ in the sphered basis, that is:
* $X^*:=D^{-\frac{1}{2}}U^TX$
* $\mu_k^*:=D^{-\frac{1}{2}}U^T\mu_k$

To sum it up, the LDA (or QDA) process is:
* Compute $\Sigma$ ($\Sigma_k$) from the dataset 
* Compute the SVD decomposition of $\Sigma$ ($\Sigma_k$)
* Compute the determinant and the transformed class centroids $\mu_k^*$
* For a given $x$, compute $x^*$ and its distance to the centroids
* Classify $x$ to the class $k$ that minimizes $\delta_k(x)$

### Use LDA as a Dimension reduction algorithm

Suppose that we want to display the LDA classification in 2 dimensions. We want to find the axis that best split the different classes.

The usual criteria used is to take the axis that maximizes the variance of the centroids. This is what PCA does. So we do the following PCA decomposition:

Let $\mu^*:=(\mu_1^\*,..\mu_K^\*)$ be the $(p,k)$-matrix of the centroids in the transformed bases. We decompose its covariance matrix using PCA:

$Cov(\mu^\*)=U_{\mu}^\*D_{\mu}^\*U_{\mu}^{\*-1}$

The columns of $U_{\mu}^\*$ are the axis we are looking for. We get the coordinates of a vector $x$ by transforming to $x^*$ and taking the scalar products with $u_{\mu,0}^\*, u_{\mu,1}^\*$...


> NB: In ESL and most blog posts that seems to base their explanation on it, the dimension reduction algorithm uses [ZCA Whitening](https://towardsdatascience.com/pca-whitening-vs-zca-whitening-a-numpy-2d-visual-518b32033edf) instead of PCA whitening (that it uses before), this explains the reference to $W^{-\frac{1}{2}}$ everywhere to do the whitening.

#### Good references

* [computation details of LDA in ESL](https://stats.stackexchange.com/questions/405541/computation-of-lda-in-elements-of-statistical-learning-4-3-2)

* [Good explanation of dimension reduction](https://yangxiaozhou.github.io/data/2019/10/02/linear-discriminant-analysis.html)

* [Another good explanation of LDA not based directly on ESL](https://www.stat.cmu.edu/~ryantibs/datamining/lectures/21-clas2.pdf)


# Deep Learning

## Back propagation

Algorithm used to compute the gradient of the loss function $\nabla L(w)$ with respect to all model parameters $w$.

It's based on *dynamic programming* and the *chain rule*


## Gradient descent

Basic version of the optimization algorithms used in Deep Learning. Let $w$ be the vector of all model parameters and $L(w)$ the *loss function*. We compute $w$ iteratively using:

$
w = w - \lambda_{lr} * \nabla L(w)
$

$\lambda_{lr}$ is called the *learning rate*. It is a trade-off between:

* *Too high*: loss is unstable and diverges
* *Too low*: learning is slow because more step are needed before convergence

## Stochastic (batch) gradient descent

In deep learning setup, data size makes naïve GD unpractical. Instead SGD updates $w$ in a small subset called *batch* of *mini-batch*

$
w = w - \lambda_{lr} * \nabla L_i(w)
$

with $L_i$ loss of the $i$-th batch. SGD has also a regularizing effect (by making noisy update step and thus preventing local minima)

The sample number by batch is called *batch size*, usually ~64 and is a trade-off between:

* *Too high*: each step is too slow (gradient computation), high memory usage, less regularization
* *Too low*: bad usage of vectorization performance

NB:
* *Lerning rate* should decrease when *batch size* decreases (to compensate for more noisy gradients)
* A pass through the whole dataset is called an *epoch*. data is usually shuffled between epochs to prevent cycles in learning.

## Momentum, RMSProp and Adam

### Momentum

When computing the gradient, it will be noisy (it changes at each batch) on some dimensions, and more steady in another dimensions.

Momentum stabilizes gradient (removes noise) by replacing current batch gradient by an exponential moving average.



$\nabla_{mom}^{n} L = \lambda_{mom}\nabla_{mom}^{n-1}L + (1-\lambda_{mom})\nabla L(w)$


In practice:
* Momentum factor $\lambda_{mom}$ is usually ~0.9
* We omit the $(1-\lambda_{mom})$ factor by just adjusting the learning rate

### RMSProp (Root Mean Square Propagation)

*RMSProp* is a way to automatically adapt the learning rate to the current scale of the gradient for each parameter.

let $g_t^i$ the real gradient at step $t$ for the $i$-th parameter

At each step and for each parameter:
1) We evaluate the exponential moving average of the square of the size of the gradient $\bar{g}_t^{2,i}$. "$\bar{~}$" for the average, "$~^2$" for the value homogeneous to a square.

$\bar{g}\_t^{2,i}=\lambda\_{rms}\bar{g}\_{t-1}^{2,i}+(1-\lambda\_{rms})(g\_t^i)^2$

2) We scale le learning rate

$\lambda_{lr-rms,t}^i=\lambda_{lr}\frac{1}{\sqrt{\bar{g}_t^{2,i}+\epsilon}}$

3) We use this adapted learning rate for the next SGD step

$
w = w - \lambda_{lr-rms,t} * \nabla L(w)
$


### Adam (Adaptative Moment Optimization)

Let's recapitulate:
* *Momentum* smooth the gradient to reduce noise
* *RMSProp* adapt the learning rate to the average scale of the gradient

*Adam* combines both techniques.

NB: In practice, *Adam* often converges faster than pure momentum, but model generalization may be worst.


## Hyper parameters

All parameters externals (not learned) to the model (*batch size*, *learning rate*...). They are usually found by *grid search* or *random search*.

## Common Layers

### Sigmoid

$y=\frac{1}{1+e^{-x}}$

Usual output layer for binary classification. Rescale $]-\infty, +\infty[$ to $]0,1[$ so it's interpretable as a probability.

### Softmax

$y_i=\frac{e^{x_i}}{\sum_{i=1}^{n}e^{x_i}}$

Usual output layer for classification with $n$ classes. Sum to $1$ so it's interpretable as probabilities.

### ReLU

$y=max(0,x)$

Usual activation function of intermediate dense layers, it introduce a non linearity in the network so that models can learn complex representations.

There is a lot of varation of it (GELU, Leaky ReLu...)


## Model Heuristics

### Batch normalization

This technique improve convergence and generalization significantly, but we [don't know really why](https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338#3fb5).


It consists in renormalizing (center-reduce) outputs between two layers batch by batch, and rescaling them up based on new learned parameters.

With:
* $d$ dimension of the normalized layer output
* $m$ batch size
* $X$ $(d,m)$-matrix ou the previous layer output
* $X_{bn}$ output of the batch norm layer
* $b$ $d$-vector of learned bias
* $s$ $d$-vector of learned scaling
* $\bar{X}$ $d$-vector of average of $X$ along the batch axis
* $\sigma^2$ $d$-vector of variance of $X$ along the batch axis

$X_{bn}= s*\frac{X-\bar{X}}{\sqrt{\sigma^2+\epsilon}}+b
$



### Dropout (model, regularization)

Common regularization technique. It consists in adding a layer that will *drop* (i.e. set to $0$) intermediate dimensions randomly at a given probability $P$.

At training time, we rescale (i.e. divide by $1-P$) the remaining values to that expectation remains the same. 

At test time, dropout layer is ignored.

### Residual connection

Deep architectures suffer from a problem called *Vanishing gradient*: gradients become close to zero far from output layers.

Residual connections "skip" layers so every layers are closer to output.

For a given layer fonction $F(x)$, the residual connection is:

$y=F(x)+x$


## Data Heuristics

### Data augmentation (data)

Increasing the amount of training data by adding modified copies of raw data. e.j: various rotation of images.

## Loss Heuristics

### Weight decay (loss, regularization)

It has been proven that models generalize better when model parameters stay close to zero.

As for *Ridge regression*, we modify loss function to penalize the $L2$-norm of the vector $w$ of all models parameters, the penalization factor $\lambda_{wd}$ is called *weight decay*

$L_{wd}(w)=L(w)+\lambda_{wd}||w||^2$

We can directly plug it in the *Gradient descent* formula using

$\nabla L_{wd} = \nabla L + 2\lambda_{wd}w$

## Training Heuristics

### Early stopping (training, regularization)

On successive *epochs* of training, we sometime empirically see that past a certain point, *validation error* starts to increases while *training error* continue to decreases. Model is overfitting. *Early stopping* consists in stopping earning at this point (before training error stabilizes itself)



# Classifications

## metrics

### Precision, recall and F1 Score

Used for binary classification tasks.

|  | Positive Label | Negative Label|
|-------------------|----------------|----------------|
| Predicted Positive              | True Positive  | False Positive |
| Predicted Negative             | False Negative | True Negative |
|                   |                |                |

"__True__" means __Correctly classified__ by the model  (either as positive or negative), "__Positive/Negative__" corresponds to the model prediction. 

* __Precision__: $ \frac{true\ positives}{all\ predicted \ positives}$. Proportion of predicted positives that are really (labeled) positives. 1 when only relevant items are found. 

* __Recall__: $ \frac{true\ positives}{all\ labeled\ positives}$. Proportion of really (labeled) positives predicted positives. 1 when all relevant items are found.

* __F1 Score__: harmonic mean of recall and prevision (harmonic penalize imbalanced models between recall and precision): $\frac{2}{\frac{1}{recall}+\frac{1}{precision}}$

### Classification Threshold, ROC Curve and AUC

* __True Positive Rate (TPR)__: synonym of Recall
* __False Positive Rate (FPR)__: $\frac{false\ positives}{all\ labeled\ negatives}$. Proportion of labeled negatives misclassified as positives.


A binary classification model usually outputs a value between 0 and 1 (the output of a softmax) that can be interpreted as the probability of the example of being Positive. The __Classification (or Decision) Threshold__ is the minimal value for which an example is classified Positive.

A high threshold will lead to a low TPR and a low FPR, a low threshold will lead to a high TPR and a high FPR

The __ROC curve__ evaluates this trade-off at different threshold values by displaying TPR as a function of FPR.

We want that curve to be as high as possible (=1) for any FPR level on Abscissa. A standard metric of classification quality is the __Area Under the Curve (AUC)__, which sums up the ROC curve in only one number.


# Recommender systems

The goal is to recommend the most relevant items to users. Most of the time, "recommending" an item means guessing the ratings that a user would give to any item given its rating for other items. One example is movie recommendations.

## Collaborative filtering

This applies well when:
* We don't have natural feature vectors for items or users, and so we have to infer them from existing ratings (using some distance measure between users and between items)
* We have a way to solve the *cold-start* problems (getting initial ratings for a new item or a new user).

### Matrix Factorization

Let's model the ratings as a $(nbUsers,nbItems)$ matrix $R$ containing the ratings. This matrix is *sparse* and we want to fill in the holes.

We suppose that items and users can be modelized features called *latent factors*. Those features are the model parameters.

We compute a measure of the distance between a user and an item using a dot-product. We then use a sigmoid function to scale the rating as needed.

because a few lines of code are worth a thousand words, here is the model:

```python
import torch
from torch import nn


class MatrixFactorization(nn.Module):

    def __init__(self, nb_users: int, nb_items: int, nb_factors: int):
        super(MatrixFactorization, self).__init__()
        self.user_embeddings = nn.Embedding(nb_users, nb_factors)
        # some users might have a tendency to give only high or low ratings, this bias deal with it
        self.user_bias = nn.Embedding(nb_users, 1)
        self.item_embeddings = nn.Embedding(nb_items, nb_factors)
        # same as for users
        self.item_bias = nn.Embedding(nb_factors, 1)

    # user_item_pairs_batch is a (batch_size, 2) tensor
    def forward(self, user_item_pairs_batch: torch.Tensor):
        batch_user_idx_s = user_item_pairs_batch[:, 0]
        batch_item_idx_s = user_item_pairs_batch[:, 1]

        batch_user_embeddings = self.user_embeddings[batch_user_idx_s]
        batch_item_embeddings = self.item_embeddings[batch_item_idx_s]

        # raw ratings by batch entry before bias and sigmoid post-processing
        batch_raw_ratings = (batch_user_embeddings*batch_item_embeddings).sum(dim=1)
        raw_ratings_biased = batch_raw_ratings + self.user_bias[batch_user_idx_s] + self.movie_bias[batch_item_idx_s]

        # Scale ratings to [0,1]
        return torch.sigmoid(raw_ratings_biased)
```

The loss function is usually just MSE.

Of course, from this formulation, we can move to *Deep Learning Factorization* by using a more complex model than just a sigmoid of the embeddings product.


## Content-based filtering



<!---
##  ------------- TODO ------------

* KNN
* Naive Bayes
* Language modeling
* Learning rate
* Learning rate schedule
* Regularization
* Cross-validation
* confusion matrix
* erreur de 1ere / 2eme espèce
* model hyper-parameters
* accuracy, precision, recall
* stochastic grad desc (batch)
* intervall de confiance
* one hot vector
* TODO: parralel python / sklearn / pytorch sur tous les cas pertinants
* p-valeur
* layers
* batch, epochs
* Siamese network / Triplet loss

* feature selection
* regularization
* early stopping
*feature engineering

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
-->