import numpy as np
from scipy.linalg import orth

import seaborn
import matplotlib.pyplot as plt


class VectorGenerator():
    A = None

    def __init__(self, m, n):
        np.random.seed(99)
        A = np.random.normal(size=(m, n))
        self.A = A

    def vec(self, k=3):
        A = self.A
        m, n = A.shape
        x = np.random.normal(size=(n, k))
        return np.dot(A, x).reshape((m, k))


if __name__ == '__main__':

    m, n = 10, 6
    vg = VectorGenerator(m, n)

    svals = []
    for k in range(1, 100 + 1):
        # initialize
        #k = 10
        B = vg.vec(k=k)
        Q0 = np.linalg.qr(B)[0]

        # sequentially generate vectors and update the subspace
        for i in range(200):
            v = vg.vec()
            #Q = orth(np.hstack((Q0, v)))
            Q = np.linalg.qr(np.hstack((Q0, v)))[0]
            Q0 = Q[:, :-1]

        # check that the subspace is good
        s = 0.
        for i in range(200):
            v = vg.vec()
            s += np.linalg.norm(v - np.dot(Q0, np.dot(Q0.transpose(), v))) ** 2
        svals.append(s)

        print('k: {:d}, s: {:6.4e}'.format(k, np.sqrt(s)))

    seaborn.set_style('white')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    ax.semilogy(np.arange(0, len(svals)), svals,
                'k-o', linewidth=4.0, markersize=10.0)
    ax.set_xlabel('Subspace Estimate Dimension')
    ax.set_ylabel('Subspace Estimate Goodness')

    ax.grid(True)
    plt.tight_layout()
    plt.show()