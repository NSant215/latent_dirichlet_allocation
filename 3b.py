import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def wordcount(M, W):
    wordcounts = np.zeros(W)
    word_identified = np.where(M[:, 1] == 1)

    for index, val in enumerate(wordcounts):
        word_identified = np.where(M[:, 1] == index+1)  # get all occurrences of document d in the training data
        occurences = np.array(M[word_identified, 2])
        wordcounts[index] = np.sum(occurences)
    
    return wordcounts

def mlm(wordcounts, V, alpha):    
    wordcounts += alpha
    total_words = np.sum(wordcounts)
    probabilities = wordcounts/total_words
    assert(np.abs(np.sum(probabilities))-1 < 1e-6)
    indices = np.argsort(-probabilities)
    I = 20
    indices = np.flip(indices[:I])
    top_words = V[indices]
    return probabilities[indices], top_words

def barplot(P, W):
    M = len(P[0])
    xx = np.linspace(0, M, M)

    gs = gridspec.GridSpec(1,2)

    fig = plt.figure(figsize=(16, 8), dpi=80)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])

    ax1.barh(xx, P[0])
    ax1.set_yticks(xx)
    ax1.set_yticklabels(W[0])
    ax1.set_xlabel('probability', fontsize = 14)
    ax1.set_ylabel('word', fontsize = 14)
    ax1.set_title('(a) alpha = 2', fontsize = 16)

    ax2.barh(xx, P[1])
    ax2.set_yticks(xx)
    ax2.set_yticklabels(W[1])
    ax2.set_xlabel('probability', fontsize = 14)
    ax2.set_title('(b) alpha = 100000', fontsize = 16)
    fig.savefig('b1.eps', format = 'eps', transparent=True, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    np.random.seed(1)
    # load data
    data = sio.loadmat('kos_doc_data.mat')
    A = np.array(data['A'])
    B = np.array(data['B'])
    V = data['V']
    W = np.max([np.max(A[:, 1]), np.max(B[:, 1])]) # total number of unique words
    for index, value in enumerate(V):
        V[index] = value[0][0]
    V = V.flatten()
    A_wordcounts = wordcount(A,W)
    P1, W1 = mlm(A_wordcounts, V, alpha=2)
    P2, W2 = mlm(A_wordcounts, V, alpha=100000)

    barplot([P1,P2], [W1,W2])
