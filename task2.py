import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def logfact(n):
    """Return log(n!)"""
    return np.sum(np.log(np.arange(1, n+1)))

def multicoef(counts):
    """Return the multinomial coefficient of the counts"""
    val = logfact(np.sum(counts))
    for c in counts:
        val -= logfact(c)
    return val


def wordcount(M, W):
    """
    M: matrix of shape (N, 3) where N is the number of words in the document.
    M[:, 0] is the document number, M[:, 1] is the word number, M[:, 2] is the number of occurrences of the word in the document.
    W: total number of unique words.

    This function calculates the number of occurrences of each word in the document.
    """
    wordcounts = np.zeros(W)

    for index, val in enumerate(wordcounts):
        word_identified = np.where(M[:, 1] == index+1)  # get all occurrences of document d in the training data
        occurences = np.array(M[word_identified, 2])
        wordcounts[index] = np.sum(occurences)
    
    return wordcounts

def mlm(wordcounts, alpha): 
    """
    wordcounts: array of shape (W,) where W is the total number of unique words
    alpha: hyperparameter representing pseudocount of words.

    This function calculates the maximum likelihood estimate of the multinomial distribution of the words in the document.
    """  
    total_words = np.sum(wordcounts + alpha)
    probabilities = np.log(wordcounts + alpha) - np.log(total_words)
    assert(np.abs(np.sum(np.exp(probabilities)))-1 < 1e-6)
    return probabilities

def likelihood(P, wordcounts, coef):
    ll = np.sum(P * wordcounts)
    return ll

if __name__ == '__main__':
    np.random.seed(1)
    # load data
    data = sio.loadmat('Data/kos_doc_data.mat')
    A = np.array(data['A'])
    B = np.array(data['B'])
    V = data['V']
    W = np.max([np.max(A[:, 1]), np.max(B[:, 1])]) # total number of unique words

    A_wordcounts = wordcount(A, W)
    A_multicoef = multicoef(A_wordcounts)

    B_wordcounts = wordcount(B, W)
    B_multicoef = multicoef(B_wordcounts)

    for index, value in enumerate(V):
        V[index] = value[0][0]
    V = V.flatten()

    alphas = np.arange(0.05,5.05,0.05)
    likelihoods = np.zeros(100)

    for index, value in enumerate(alphas):
        P = mlm(A_wordcounts,value)
        ll = likelihood(P,B_wordcounts, B_multicoef)
        likelihoods[index] = ll
        print(ll)
    
    small_alphas_data = np.array([alphas, likelihoods])
    np.save('small_alphas_data.npy', small_alphas_data)

    alphas = np.arange(1,100001,1)
    likelihoods = np.zeros(100000)

    for index, value in enumerate(alphas):
        P = mlm(A_wordcounts,value)
        ll = likelihood(P,B_wordcounts, B_multicoef)
        likelihoods[index] = ll
        # print(ll)
    
    alphas_data = np.array([alphas, likelihoods])
    np.save('alphas_data.npy', alphas_data)

    gs = gridspec.GridSpec(1,2)

    data_1 = np.load('alphas_data.npy')
    data_2 = np.load('small_alphas_data.npy')

    print(data_1.shape)
    print(data_2.shape)

    fig = plt.figure(figsize=(16, 8), dpi=80)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    
    ax1.plot(data_1[0], data_1[1])
    ax1.set_xlabel('alpha', fontsize = 14)
    ax1.set_ylabel('log probability', fontsize = 14)
    ax1.set_title('(a) large range of alphas', fontsize = 16)

    ax2.plot(data_2[0], data_2[1])
    ax2.set_xlabel('alpha', fontsize = 14)
    ax2.set_ylabel('log probability', fontsize = 14)
    ax2.set_title('(b) optimum value', fontsize = 16)
    fig.savefig('b2.eps', format = 'eps', transparent=True, bbox_inches='tight')
    plt.show()
        
    
