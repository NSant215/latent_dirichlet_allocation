import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.special import binom

def logfact(n):
    '''Return log(n!)'''
    return np.sum(np.log(np.arange(1, n+1)))

def multicoef(counts):
    val = logfact(np.sum(counts))
    for c in counts:
        val -= logfact(c)
    return val

def wordcount(M, W):
    wordcounts = np.zeros(W)

    for index in range(W):
        word_identified = np.where(M[:, 1] == index+1)
        occurences = np.array(M[word_identified, 2])
        wordcounts[index] = np.sum(occurences)
    
    return wordcounts

def single_doc(M, W, docnum):
    wordcounts = np.zeros(W)
    indices = np.where(M[:,0]==docnum)
    doc = M[indices]

    for index, val in enumerate(wordcounts):
        word_identified = np.where(doc[:, 1] == index+1)  # get all occurrences of document d in the training data
        occurences = np.array(M[word_identified, 2])
        wordcounts[index] = np.sum(occurences)
    
    return wordcounts

def mlm(wordcounts, alpha):   
    total_words = np.sum(wordcounts + alpha)
    probabilities = np.log(wordcounts + alpha) - np.log(total_words)
    assert(np.abs(np.sum(np.exp(probabilities)))-1 < 1e-6)
    return probabilities

def likelihood(P, wordcounts, coef):
    ll = np.sum(P * wordcounts)
    # for wordcount in wordcounts:
    #     ll += logfact(wordcount)
    return ll

if __name__ == '__main__':
    np.random.seed(1)
    # load data
    data = sio.loadmat('kos_doc_data.mat')
    A = np.array(data['A'])
    B = np.array(data['B'])
    V = data['V']
    W = np.max([np.max(A[:, 1]), np.max(B[:, 1])]) # total number of unique words
    print(W)

    A_wordcounts = wordcount(A, W)
    A_multicoef = multicoef(A_wordcounts)

    B_wordcounts = wordcount(B, W)
    B_multicoef = multicoef(B_wordcounts)

    docnum = 2001
    doc_wordcounts = single_doc(B, W, docnum)
    doc_multicoef = multicoef(doc_wordcounts)

    alpha = 2.5
    P = mlm(A_wordcounts,alpha)

    N_d = np.sum(doc_wordcounts)
    # print(N_d)
    ll_d = likelihood(P, doc_wordcounts, doc_multicoef)
    d_perplexity = np.exp(-ll_d/N_d)
    print("Likelihood: {}, Words: {}, Perplexity: {}".format(ll_d,N_d,d_perplexity))

    ll_b = likelihood(P, B_wordcounts, B_multicoef)
    N_b = np.sum(B_wordcounts)
    b_perplexity = np.exp(-ll_b/N_b)
    print("Likelihood: {}, Words: {}, Perplexity: {}".format(ll_b,N_b,b_perplexity))