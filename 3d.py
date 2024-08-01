from sampleDiscrete import sampleDiscrete
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def BMM(A, B, K, alpha, gamma, num_iters_gibbs):

    """

    :param A: Training data [D, 3]
    :param B: Test Data [D, 3]
    :param K: number of mixture components
    :param alpha: parameter of the Dirichlet over mixture components
    :param gamma: parameter of the Dirichlet over words
    :return: test perplexity and multinomial weights over words
    """
    W = np.max([np.max(A[:, 1]), np.max(B[:, 1])])  # total number of unique words
    D = np.max(A[:, 0])  # number of documents in A

    # Initialization: assign each document a mixture component at random
    sd = np.floor(K * np.random.rand(D)).astype(int)   # mixture component assignment - randomly assigning topic to each doc
    swk = np.zeros((W, K))  # K multinomials over W unique words - our ß matrix
    sk_docs = np.zeros((K, 1), dtype=int)  # number of documents assigned to each mixture
    # Populate the count matrices by looping over documents
    for d in range(D):
        training_documents = np.where(A[:, 0] == d+1)  # get all occurrences of document d in the training data
        w = np.array(A[training_documents, 1])  # number of unique words in document d
        c = np.array(A[training_documents, 2])  # counts of words in document d
        k = sd[d]  # document d is in mixture k
        swk[w-1, k] += c  # number of times w is assigned to component k
        sk_docs[k] += 1

    sk_words = np.sum(swk, axis=0)  # number of words assigned to mixture k over all docs
    sk_docs_array = np.zeros((K,num_iters_gibbs+1))
    sk_docs_array[:,0] = sk_docs[:,0]
    sd_array = np.zeros((D,num_iters_gibbs+1))
    sd_array[:,0] = sd
    # Perform Gibbs sampling through all documents and words
    for iter in range(num_iters_gibbs):
        for d in range(D):

            training_documents = np.where(A[:, 0] == d+1)  # get all occurrences of document d in the training data
            w = A[training_documents, 1]  # number of unique words in document d
            c = A[training_documents, 2]  # counts of words in document d
            old_class = sd[d]  # document d is in mixture k
            # remove document from counts
            swk[w-1, old_class] -= c  # decrease number of times w is assigned to component k
            sk_docs[old_class] -= 1  # remove document d from count of docs
            sk_words[old_class] -= np.sum(c)  # remove word counts from mixture
            
            # resample class of document
            lb = np.zeros(K)  # log probability of doc d under mixture component k

            for k in range(K):
                ll = np.dot(np.log(swk[w-1, k] + gamma) - np.log(sk_words[k] + gamma * W), c.T)

                lb[k] = np.log(sk_docs[k] + alpha) + ll
            b = np.exp(lb - np.max(lb))  # exponentiation of log probability plus constant
            kk = sampleDiscrete(b, np.random.rand())  # sample from (un-normalized) multinomial distribution

            # update counts based on new class assignment
            swk[w-1, kk] += c  # number of times w is assigned to component k
            sk_docs[kk] += 1
            sk_words[kk] += np.sum(c)
            sd[d] = kk
        sk_docs_array[:,iter+1] = sk_docs[:,0]
        sd_array[:, iter+1] = sd

    return sk_docs_array, sd_array

if __name__ == '__main__':
    seed = 3
    np.random.seed(seed)
    # load data
    data = sio.loadmat('kos_doc_data.mat')
    A = np.array(data['A'])
    B = data['B']
    V = data['V']
    K = 20  # number of clusters
    D = np.max(A[:, 0])  # number of documents in A
    print(D)
    alpha = 10  # parameter of the Dirichlet over mixture components
    gamma = .1  # parameter of the Dirichlet over words
    num_iters_gibbs = 100
    gibbs_array, doc_array = BMM(A, B, K, alpha, gamma, num_iters_gibbs)
    np.save('d_mix{}_{}_{}.npy'.format(seed, num_iters_gibbs, K), gibbs_array)
    np.save('doc_assignments{}_{}_{}.npy'.format(seed, num_iters_gibbs, K), doc_array)

    gibbs_array= np.load('d_mix{}_{}_{}.npy'.format(seed, num_iters_gibbs, K))/D
    doc_array = np.load('doc_assignments{}_{}_{}.npy'.format(seed, num_iters_gibbs, K))

    gs = gridspec.GridSpec(1,2)

    fig = plt.figure(figsize=(16, 8), dpi=80)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])

    for i in range(K):
        ax1.plot(gibbs_array[i,:], label = 'k={}'.format(i))


    ax1.set_xlabel('Iteration', fontsize = 14)
    ax1.set_ylabel('Mixed Proportion of Category', fontsize = 14)
    ax1.set_title('(a) Category Mixing Proportions', fontsize = 16)


    for i in range(11):
        ax2.plot(doc_array[i,:], label = 'doc {}'.format(i))

    ax2.set_xlabel('Iteration', fontsize = 14)
    ax2.set_ylabel('Category Assigned', fontsize = 14)
    ax2.set_title('(b) Assignments of First 10 Documents over iterations', fontsize = 16)

    fig.savefig('d1.eps', format = 'eps', transparent=True, bbox_inches='tight')

    plt.show()


