import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


D = 2000
K=20
W=6906
alpha = 0
seed1_mix_50 = (np.load('d_mix{}_{}.npy'.format(1, 50))+alpha)/(D+W*alpha)
seed2_mix_50 = (np.load('d_mix{}_{}.npy'.format(2, 50))+alpha)/(D+W*alpha)
seed3_mix_50 = (np.load('d_mix{}_{}.npy'.format(3, 50))+alpha)/(D+W*alpha)

seed1_docs = np.load('doc_assignments{}_{}.npy'.format(1, 50, 20))
seed2_docs = np.load('doc_assignments{}_{}.npy'.format(2, 50, 20))
seed3_docs = np.load('doc_assignments{}_{}.npy'.format(3, 50, 20))

seed1_mix_100 = (np.load('d_mix{}_{}_{}.npy'.format(1, 100, 20))+alpha)/(D+W*alpha)
seed2_mix_100 = (np.load('d_mix{}_{}_{}.npy'.format(2, 100, 20))+alpha)/(D+W*alpha)
seed3_mix_100 = (np.load('d_mix{}_{}_{}.npy'.format(3, 100, 20))+alpha)/(D+W*alpha)


gs = gridspec.GridSpec(1,3)

fig = plt.figure(figsize=(13, 5), dpi=80)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

for i in range(K):
    ax1.plot(seed1_mix_100[i,:], label = 'k={}'.format(i))
    ax2.plot(seed2_mix_100[i,:], label = 'k={}'.format(i))
    ax3.plot(seed3_mix_100[i,:], label = 'k={}'.format(i))

# for i in range(11):
#     ax1.plot(seed1_docs[i,:], label = 'doc {}'.format(i))
#     ax2.plot(seed2_docs[i,:], label = 'doc {}'.format(i))
#     ax3.plot(seed3_docs[i,:], label = 'doc {}'.format(i))


ax1.set_xlabel('Iteration', fontsize = 14)
ax1.set_ylabel('Mixing Proportion', fontsize = 14)
ax1.set_title('(a) Seed 1', fontsize = 16)
# ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

ax2.set_xlabel('Iteration', fontsize = 14)
# ax2.set_ylabel('Category Assigned', fontsize = 14)
ax2.set_title('(b) Seed 2', fontsize = 16)
# ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

ax3.set_xlabel('Iteration', fontsize = 14)
# ax2.set_ylabel('Category Assigned', fontsize = 14)
ax3.set_title('(c) Seed 3', fontsize = 16)
# ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

fig.savefig('dseeds_100.eps', format = 'eps', transparent=True, bbox_inches='tight')

plt.show()