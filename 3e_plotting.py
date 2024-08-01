import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# K = 20
# num_gibbs_iters = 100

# gs = gridspec.GridSpec(2,1)

# fig = plt.figure(figsize=(12, 15), dpi=80)
# ax1 = fig.add_subplot(gs[0,0])
# ax2 = fig.add_subplot(gs[1,0])

# entropy_array = np.zeros((K, num_gibbs_iters+1))

# gamma = 0.1

# for i in range(num_gibbs_iters+1):
#     swk = np.load('swk_{}.npy'.format(i))
#     probabilities = (swk+gamma)/(swk.sum(axis=0) + 6906 * gamma)
#     log_probs = np.ma.log2(probabilities).filled(0)
#     entropies = -np.sum(probabilities*log_probs, axis=0)
#     entropy_array[:, i] = entropies

# for k in range(K):
#     ax1.plot(entropy_array[k,:51], label='topic {}'.format(k))
#     ax2.plot(entropy_array[k])

# ax1.set_xlabel('Iteration', fontsize = 14)
# ax1.set_ylabel('Entropy (bits)', fontsize = 14)
# ax1.set_title('(a) 50 Iterations', fontsize = 16)

# ax2.set_xlabel('Iteration', fontsize = 14)
# ax2.set_ylabel('Entropy (bits)', fontsize = 14)
# ax2.set_title('(b) 100 Iterations', fontsize = 16)

# fig.savefig('e_entropy.eps', format = 'eps', transparent=True, bbox_inches='tight')

# plt.grid()
# plt.show()

perplexities = []
ranges = range(0,101,10)
for i in ranges:
    new = np.load('perplexity_{}.npy'.format(i))[0][0]
    perplexities.append(new)

print(perplexities)
plt.plot(ranges, perplexities)
plt.xlabel('iterations', fontsize = 14)
plt.ylabel('perplexity', fontsize = 14)
plt.savefig('e_perplexity.eps', format = 'eps', transparent=True, bbox_inches='tight')
plt.show()