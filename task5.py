import scipy.io as sio
import numpy as np
from scipy.sparse import coo_matrix as sparse
from sampleDiscrete import sampleDiscrete
import matplotlib.pyplot as plt

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
    
    # np.save('swk_{}.npy'.format(0), swk)

    sk = np.sum(skd, axis=1)  # word to topic assignment counts accross all documents
    # This makes a number of Gibbs sampling sweeps through all docs and words, it may take a bit to run
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
        np.save('swk_{}.npy'.format(iter+1), swk)
        print(iter)

        if iter in range(0,101,10):
            # compute the perplexity for all words in the test set B
            # We need the new Skd matrix, derived from corpus B
            lp, nd = 0, 0
            unique_docs_in_b = np.unique(B[:, 0])
            for d in unique_docs_in_b:  # loop over all documents in B
                print(d)
                # randomly assign topics to each word in test document d
                z = np.zeros((W, K))
                words_in_d = B[np.where(B[:, 0] == d), 1][0]-1
                for w in words_in_d:  # w are the words in doc d
                    c = Swd[w, d-1]
                    for i in range(c):
                        k = np.floor(K * np.random.rand())
                        z[w, int(k)] += 1

                Skd = np.sum(z, axis=0)
                # perform some iterations of Gibbs sampling for test document d
                for iters in range(num_gibbs_iters):
                    for w in words_in_d:  # w are the words in doc d
                        a = z[w, :].copy()  # number of times word w is assigned to each topic in doc d
                        indices = np.where(a > 0)[0]   # topics with non-zero word counts for word w in doc d
                        np.random.shuffle(indices)
                        for k in indices:
                            k = int(k)
                            for i in range(int(a[k])):
                                z[w, k] -= 1  # remove word from count matrix for doc d
                                Skd[k] -= 1
                                b = (alpha + Skd) * (gamma + swk[w, :]) \
                                    / (W * gamma + sk)
                                kk = sampleDiscrete(b, np.random.rand())
                                z[w, kk] += 1  # add word with new topic to count matrix for doc d
                                Skd[kk] += 1
                b1 = ((alpha + Skd) / np.sum(alpha + Skd))[:, None]
                b2 = (gamma + swk) / (W * gamma + sk)
                b = np.matmul(b2,b1)
                words_and_counts = B[np.where(B[:, 0] == d), 1:][0]
                lp += np.dot(np.log(b[words_and_counts[:, 0]-1]).T, words_and_counts[:, 1])  # log probability, doc d
                nd += np.sum(words_and_counts[:, 1])  # number of words, doc d

            perplexity = np.exp(-lp/nd)  # perplexity
            np.save('perplexity_{}.npy'.format(iter), np.array([perplexity]))

    return perplexity, swk


if __name__ == '__main__':
    np.random.seed(0)
    # load data
    data = sio.loadmat('kos_doc_data.mat')
    A = np.array(data['A'])
    B = data['B']
    V = data['V']
    W = np.max([np.max(A[:, 1]), np.max(B[:, 1])])  # total number of unique words


    K = 20  # number of clusters
    alpha = .1  # parameter of the Dirichlet over mixture components
    gamma = .1  # parameter of the Dirichlet over words

    num_gibbs_iters = 100
    perplexity, swk = LDA(A, B, K, alpha, gamma, num_gibbs_iters)
    print(perplexity)

    np.save('perplexity_{}.npy'.format(num_gibbs_iters), np.array([perplexity]))

    I = 20
    indices = np.argsort(-swk, axis=0)
    indices = indices[:20, :]
    top_words = V[indices]
    for topic in top_words[:, :, 0].T:
        print('\n')
        for word in topic:
            print(word[0])

    entropy_array = np.zeros((K, num_gibbs_iters+1))


    gamma = 1
    for i in range(num_gibbs_iters+1):
        swk = np.load('swk_{}.npy'.format(i))
        probabilities = (swk+gamma)/(swk.sum(axis=0)+W*gamma)
        log_probs = np.ma.log2(probabilities).filled(0)
        entropies = -np.sum(probabilities*log_probs, axis=0)
        entropy_array[:, i] = entropies
    
    for k in range(K):
        plt.plot(entropy_array[k], label='topic {}'.format(k))
    
    plt.xlabel('iteration', fontsize = 14)
    plt.ylabel('entropy', fontsize = 14)

    plt.savefig('e_entropy.eps', format = 'eps', transparent=True, bbox_inches='tight')
    
    # plt.legend()
    plt.grid()
    plt.show()