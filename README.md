# latent_dirichlet_allocation
Project work in solidifying learnings in Latent Dirichlet Allocation for predicting words in documents.

## Project Organisation
Data is in the data folder, and each task has its own python file for that investigation.

## Task 1: Finding Maximum Likelihood Multinomial estimates over the words.
This is a simple task of finding the maximum likelihood estimates of the multinomial distribution over the words in the document by counting the number of times each word appears in the document and dividing by the total number of words.

## Task 2: Bayesian Inference using symmetric Dirichlet prior
This task involves using a symmetric Dirichlet prior to estimate the posterior distribution of the multinomial distribution over the words in the document. The posterior distribution is given by the Dirichlet distribution with parameters alpha + n_i, where n_i is the number of times word i appears in the document and alpha is the parameter of the Dirichlet prior (constant over all words).

## Task 3: Perplexity
Perplexity is a measure of how well a probability distribution or probability model predicts a sample. It may be used to compare probability models. A low perplexity indicates the probability distribution is good at predicting the sample. This task involves calculating the perplexity of the model on a test set of documents and comparing using the multinomial vs categorical distribution.

## Task 4: Bernoulli Mixture Model (BMM) a.k.a. Mixture of Multinomials Model
This task involves attemping to sample from the posterior distribution of latent topic assignments to documents using a collapsed Gibbs Sampler. 

## Task 5: Latent Dirichlet Allocation (LDA)
This task involves implementing the Latent Dirichlet Allocation algorithm to predict words in documents. The algorithm is a generative probabilistic model for collections of discrete data such as text corpora. It is also a three-level hierarchical Bayesian model, in which each item of a collection is modeled as a finite mixture over an underlying set of topics. Each topic is, in turn, modeled as an infinite mixture over an underlying set of topic probabilities.
