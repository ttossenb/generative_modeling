import time
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite

import autoencoder


def pairwiseSquaredDistances(clients, servers):
    cL2S = np.sum(clients ** 2, axis=-1)
    sL2S = np.sum(servers ** 2, axis=-1)
    cL2SM = np.tile(cL2S, (len(servers), 1))
    sL2SM = np.tile(sL2S, (len(clients), 1))
    squaredDistances = cL2SM + sL2SM.T - 2.0 * servers.dot(clients.T)
    return squaredDistances.T


class OOWrapper:
    def __init__(self, latentPoints, targetPoints):
        self.latentPoints = latentPoints
        self.targetPoints = targetPoints
        assert latentPoints.shape == targetPoints.shape
        self.n, self.d = latentPoints.shape
        self.matching = np.random.permutation(self.n)

    def updateBatch(self, latentIndices, latentBatch):
        targetIndices = self.matching[latentIndices]
        targetBatch = self.targetPoints[targetIndices]
        self.latentPoints[latentIndices] = latentBatch
        # pwd = np.sqrt(pairwiseSquaredDistances(latentBatch, targetBatch))
        localMatching = autoencoder.updateBipartiteGraphFromScratch(latentBatch, targetBatch)
        self.matching[latentIndices] = self.matching[latentIndices[localMatching]]

    # no position updates, just refining the matching
    def doEpoch(self, batch_size):
        assert self.n  % batch_size == 0
        random_permutation = np.random.permutation(self.n)
        for i in range(self.n // batch_size):
            indices = random_permutation[i * batch_size: (i + 1) * batch_size]
            latentBatch = self.latentPoints[indices]
            self.updateBatch(indices, latentBatch)

    def evaluateMatching(self):
        s = 0.0
        for i in range(self.n):
            s += np.linalg.norm(self.latentPoints[i] - self.targetPoints[self.matching[i]])
        return s

    def restart(self):
        # it's a no-op, dynamic algorithm rather than online.
        pass


def main():
    n = 1000
    d = 10

    np.random.seed(1)
    latentPoints = np.random.normal(0, 1, (n, d))
    targetPoints = np.random.normal(0, 1, (n, d))

    oo = OOWrapper(latentPoints=latentPoints, targetPoints=targetPoints)

    epochs = 10
    batch_size = 50
    start = time.clock()
    for epoch in range(epochs):
        oo.doEpoch(batch_size)
        print("epoch", epoch, "weight", oo.evaluateMatching())

    print('Elapsed time: ', time.clock() - start)


if __name__ == "__main__":
    main()
    # main_nonObjectOriented()
