import numpy as np
from numpy.core.fromnumeric import size
import scipy.io as sio
import matplotlib.pyplot as plt

def mlm(A, V):
    W = np.max(A[:, 1])  # total number of unique words
    wordcounts = np.zeros(W)
    word_identified = np.where(A[:, 1] == 1)

    for index in range(W):
        word_identified = np.where(A[:, 1] == index+1)  # get all occurrences of document d in the training data
        occurences = np.array(A[word_identified, 2])
        wordcounts[index] = np.sum(occurences)
    
    total_words = np.sum(wordcounts)
    probabilities = wordcounts/total_words
    indices = np.argsort(-probabilities)
    I = 20
    indices = np.flip(indices[:I])
    top_words = V[indices]
    return probabilities[indices], top_words

def barplot(P, W):
    M = len(P)
    xx = np.linspace(0, M, M)
    plt.figure(figsize=(12, 8))
    plt.barh(xx, P)
    plt.xlabel('Probability', fontsize = 14)
    plt.ylabel('Word', fontsize = 14)
    plt.yticks(xx, labels=W)
    plt.savefig('a.eps', format = 'eps', transparent=True, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    np.random.seed(1)
    # load data
    data = sio.loadmat('kos_doc_data.mat')
    A = np.array(data['A'])

    V = data['V']
    for index, value in enumerate(V):
        V[index] = value[0][0]
    V = V.flatten()

    P, W = mlm(A, V)
    barplot(P, W)
