import scipy.io as sio
import numpy as np
from scipy.sparse import coo_matrix as sparse
from sampleDiscrete import sampleDiscrete
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def LDA(A, B, K, alpha, gamma, num_gibbs_iters):

    """
    Latent Dirichlet Allocation

    :param A: Training data [D, 3]
    :param B: Test Data [D, 3]
    :param K: number of mixture components
    :param alpha: parameter of the Dirichlet over mixture components
    :param gamma: parameter of the Dirichlet over words
    :return: perplexity, multinomial over words
    """
    W = np.max([np.max(A[:, 1]), np.max(B[:, 1])])  # total number of unique words
    D = np.max(A[:, 0])  # number of documents in A

    # A's columns are doc_id, word_id, count
    swd = sparse((A[:, 2], (A[:, 1]-1, A[:, 0]-1))).tocsr() # [W,D] every element is a word 
    Swd = sparse((B[:, 2], (B[:, 1]-1, B[:, 0]-1))).tocsr()

    # Initialization
    skd = np.zeros((K, D))  # count of word assignments to topics for document d
    swk = np.zeros((W, K))  # unique word topic assignment counts across all documents

    s = []  # each element of the list corresponds to a document
    r = 0
    for d in range(D):  # iterate over the documents
        z = np.zeros((W, K))  # unique word topic assignment counts for doc d
        words_in_doc_d = A[np.where(A[:, 0] == d+1), 1][0]-1 # unique words
        for w in words_in_doc_d:  # loop over the unique words in doc d
            c = swd[w, d]  # number of occurrences for doc d
            for i in range(c):  # assign each occurrence of word w to a topic at random
                k = np.floor(K*np.random.rand())
                z[w, int(k)] += 1
                r += 1
        skd[:, d] = np.sum(z, axis=0)  # number of words in doc d assigned to each topic
        swk += z  # unique word topic assignment counts across all documents
        s.append(sparse(z))  # sparse representation: z contains many zero entries

    sk = np.sum(skd, axis=1)  # word to topic assignment counts accross all documents
    # This makes a number of Gibbs sampling sweeps through all docs and words, it may take a bit to run
    sk_array = np.zeros((K, num_gibbs_iters+1))
    sk_array[:,0] = sk
    for iter in range(num_gibbs_iters):
        for d in range(D):
            z = s[d].todense()  # unique word topic assigment counts for document d
            words_in_doc_d = A[np.where(A[:, 0] == d + 1), 1][0] - 1
            for w in words_in_doc_d:  # loop over unique words in doc d
                a = z[w, :].copy()  # number of times word w is assigned to each topic in doc d
                indices = np.where(a > 0)[1]  # topics with non-zero word counts for word w in doc d
                np.random.shuffle(indices)
                for k in indices:  # loop over topics in permuted order
                    k = int(k)
                    for i in range(int(a[0, k])):  # loop over counts for topic k
                        z[w, k] -= 1  # remove word from count matrices
                        swk[w, k] -= 1
                        sk[k] -= 1
                        skd[k, d] -= 1
                        b = (alpha + skd[:, d]) * (gamma + swk[w, :]) \
                            / (W * gamma + sk)
                        kk = sampleDiscrete(b, np.random.rand())  # Gibbs sample new topic assignment
                        z[w, kk] += 1  # add word with new topic to count matrices
                        swk[w, kk] += 1
                        sk[kk] += 1
                        skd[kk, d] += 1

            s[d] = sparse(z)  # store back into sparse structure
        sk_array[:,iter+1] = sk
        print(iter)

    return sk_array


if __name__ == '__main__':
    np.random.seed(0)
    # load data
    data = sio.loadmat('kos_doc_data.mat')
    A = np.array(data['A'])
    B = data['B']
    V = data['V']

    K = 20  # number of clusters
    alpha = .1  # parameter of the Dirichlet over mixture components
    gamma = .1  # parameter of the Dirichlet over words
    num_gibbs_iters = 50

    # sk_array = LDA(A, B, K, alpha, gamma, num_gibbs_iters)
    # np.save('sk_array_{}.npy'.format(num_gibbs_iters), sk_array)

    # sk_array = np.load('sk_array_{}.npy'.format(num_gibbs_iters))
    # posteriors = (sk_array+alpha)/(sk_array.sum(axis=0)+K*alpha)

    # for i in range(K):
    #     plt.plot(posteriors[i,:], label = 'k={}'.format(i))
    
    gs = gridspec.GridSpec(2,1)

    fig = plt.figure(figsize=(12, 15), dpi=80)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])

    data_50 = np.load('sk_array_{}.npy'.format(50))
    data_100 = np.load('sk_array_{}.npy'.format(100))

    posteriors_50 = (data_50+alpha)/(data_50.sum(axis=0)+K*alpha)
    posteriors_100 = (data_100+alpha)/(data_100.sum(axis=0)+K*alpha)

    for i in range(K):
        ax1.plot(posteriors_50[i,:], label = 'k={}'.format(i))
        ax2.plot(posteriors_100[i,:], label = 'k={}'.format(i))


    ax1.set_xlabel('Iteration', fontsize = 14)
    ax1.set_ylabel('Mixing Proportion', fontsize = 14)
    ax1.set_title('(a) 50 Iterations', fontsize = 16)

    ax2.set_xlabel('Iteration', fontsize = 14)
    ax2.set_ylabel('Mixing Proportion', fontsize = 14)
    ax2.set_title('(b) 100 Iterations', fontsize = 16)

    fig.savefig('e1.eps', format = 'eps', transparent=True, bbox_inches='tight')

    plt.show()

