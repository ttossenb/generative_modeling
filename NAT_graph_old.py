import numpy as np
from networkx.algorithms import bipartite
#import array as arr
from scipy.sparse import coo_matrix


def normalize(a, order=2, axis=-1):
    #print(a[a == np.zeros(a[0].shape)])
    #a[a == np.zeros(a[0].shape)] = np.ones(a[0].shape)
    a_norms = np.expand_dims(np.linalg.norm(a, order, axis), axis)
    return a / a_norms


def binMatrix(d):
    binaries = np.array([[0], [1]], dtype=int)
    for i in range(1, d):
        binaries = np.vstack((np.concatenate((np.zeros((2 ** i, 1), dtype=int), binaries), axis=-1),
                              np.concatenate((np.ones((2 ** i, 1), dtype=int), binaries), axis=-1)))
    return binaries

#number of traing inputs
n = 50000
#latent dimension
d = 10
#cennectedness radius
r = 0.

#number of points to put in each space segment
k = n // (2 ** d)

#generating target points, normalizing them, taking the absolute value of each
targetPoints = np.absolute(normalize(np.random.normal(0, 1, (2 ** d, k, d))))

binaries = np.expand_dims(binMatrix(d), axis=1)

#0 -> 1 (don't flip sign), 1 -> -1 (flip sign)
targetPoints = ((-2) * binaries + 1) * targetPoints

#placeholder for input
latentPoints = normalize(np.random.normal(0, 1, (n, d)))

#find the close seqments for each latent point, and search on them,
#and also create the biadjacency sparse matrix of the desired bipartate graph
#closeLatentPoints = [[] for i in range(n)]
#create empty sparse matrix
B = coo_matrix((n, (2 ** d) * k), dtype=np.float16)
for i in range(n):
    if i % 100 == 0: print(i)
    closeSegments = np.array([0], dtype=int)
    for j in range(d):
        tempCol = latentPoints[i]
        if tempCol[j] > r:
            closeSegments = 2 * closeSegments
        elif tempCol[j] < (-r):
            closeSegments = 2 * closeSegments + 1
        else:
            closeSegments = np.concatenate((2 * closeSegments, 2 * closeSegments + 1))
    #closeTargets = np.reshape(targetPoints[closeSegments], (closeSegments.shape[0] * k, d))
    for j in range(closeSegments.shape[0]):
        norms = np.linalg.norm(targetPoints[closeSegments[j]] - latentPoints[i], axis=-1)
        smallNorms = norms < r
        data = norms[smallNorms]
        rows = np.repeat(i, data.shape[0])
        cols = np.arange(k)[smallNorms] + (closeSegments[j] * k)
        B = B + coo_matrix((data, (rows, cols)), shape=(n, (2 **d) * k), dtype=np.float16)
    #closeTargets = targetPoints[closeSegments]
    #norms = np.linalg.norm(closeTargets - latentPoints[i], axis=-1)

    #for j in range(containments.shape[0]):
    #    closeLatentPoints[containments[j]].extend([i])

#create the biadjacency sparse matrix of the desired bipartate graph
#for i in range(n):
#    segmentPointIndices = np.asarray(closeLatentPoints[i], dtype=int)
#    segmentPoints = latentPoints[segmentPointIndices]

G = bipartite.from_biadjacency_matrix(B)
