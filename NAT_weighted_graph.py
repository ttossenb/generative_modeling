import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from scipy.sparse import coo_matrix
from annoy import AnnoyIndex
import time
from sortedcontainers import SortedSet
import directed_weighted_ES as ES
from math import sqrt, log, floor
from random import randint


def normalize(a, order=2, axis=-1):
    a_norms = np.expand_dims(np.linalg.norm(a, order, axis), axis)
    return a / a_norms


def createAnnoyIndex(d, targetPoints, n_trees):
    # create AnnoyIndex in R^d
    targetIndex = AnnoyIndex(d, metric='euclidean')
    # add each of the target points
    for i in range(targetPoints.shape[0]):
        targetIndex.add_item(i, targetPoints[i])

    # build the LSH-forest with the target points
    targetIndex.build(n_trees)

    # save and load with memory map
    targetIndex.save("LSHForest.ann")
    loadedIndex = AnnoyIndex(d, metric='euclidean')
    loadedIndex.load("LSHForest.ann")
    return loadedIndex


def createGraph(latentPoints, targetPoints, n_nbrs, n_rndms, loadedIndex):
    #find the closest neighbours for each latent point and their distances,
    #and also create the biadjacency sparse matrix of the desired bipartite graph

    #initialize closeIndices and closeDistances
    closeIndices = np.zeros((latentPoints.shape[0], n_nbrs), dtype=int)
    closeDistances = np.zeros((latentPoints.shape[0], n_nbrs), dtype=np.float16)

    #create the bipartite graph in networkx
    G = nx.Graph()
    G.add_nodes_from(range(0, latentPoints.shape[0]), bipartite=0)
    G.add_nodes_from(range(latentPoints.shape[0], latentPoints.shape[0] + targetPoints.shape[0]), bipartite=1)

    for i in range(latentPoints.shape[0]):
        (closeIndices[i], closeDistances[i]) = loadedIndex.get_nns_by_vector(latentPoints[i], n_nbrs, include_distances=True)
        for j in range(n_nbrs):
            G.add_edge(i, latentPoints.shape[0] + closeIndices[i, j], weight=closeDistances[i, j])
        for j in range(n_rndms):
            tempRandInt = randint(0, targetPoints.shape[0] - 1)
            G.add_edge(i, latentPoints.shape[0] + tempRandInt,
                       weight=np.linalg.norm(targetPoints[tempRandInt] - latentPoints[i])) #might add the same edge twice

    print(len(set(G)))

    client_nodes = SortedSet(range(latentPoints.shape[0]))
    server_nodes = SortedSet(range(latentPoints.shape[0], latentPoints.shape[0] + targetPoints.shape[0]))

    return G, client_nodes, server_nodes


def addAllClients(G, H, parents_by_level, levels, best_gains, M, F, client_nodes, server_nodes, max_level, source_node=-1):
    #add all the clients 1 by 1 while maintaining the ES structure
    #also modify M to be a maximal matching
    clients_to_add = client_nodes
    ES.addClients(G, clients_to_add, H, parents_by_level, levels, best_gains, M, source_node, max_level)

    #deepcopy M to F
    F.add_nodes_from(M.nodes)
    F.add_edges_from(M.edges)


def initializeESGraph(H, parents_by_level, levels, best_gains, client_nodes, server_nodes, source_node=-1):
    #H = directed ES graph
    H.add_node(source_node)
    levels[source_node] = 0
    #best_gains[source_node] = 0
    H.add_nodes_from(client_nodes)
    for c in client_nodes:
        levels[c] = 1
    H.add_nodes_from(server_nodes)
    for s in server_nodes:
        levels[s] = 1

    #connect the source node to the client nodes
    for c in client_nodes:
        H.add_edge(source_node, c, weight=0)
        parents_by_level[(c, 0)].add((source_node, 0))
        #best_gains[c] = 0

    #connect the source node to the server nodes
    for s in server_nodes:
        H.add_edge(source_node, s, weight=0)
        parents_by_level[(s, 0)].add((source_node, 0))
        #best_gains[s] = 0


def addBatchOfClients(G, batch_indices, H, parents_by_level, levels, best_gains, M, F, max_level, source_node=-1):
    ES.addClients(G, batch_indices, H, parents_by_level, levels, best_gains, M, source_node, max_level)
    #readjust F
    for c in batch_indices:
        for s in set(F.successors(c)):
            F.remove_edge(c, s)
        for s in M.successors(c):
            F.add_edge(c, s)


def pairwiseSquaredDistances(clients, servers):
    cL2S = np.sum(clients ** 2, axis=-1)
    sL2S = np.sum(servers ** 2, axis=-1)
    cL2SM = np.tile(cL2S, (len(servers), 1))
    sL2SM = np.tile(sL2S, (len(clients), 1))
    squaredDistances = cL2SM + sL2SM.T - 2.0 * servers.dot(clients.T)
    return squaredDistances.T


def initializeHelperStructures(client_nodes, server_nodes, max_level):
    #initialize H before the clients are being added iteratively
    H = nx.DiGraph()
    #initialize M digraph of the matching directed from C to S
    M = nx.DiGraph()
    M.add_nodes_from(client_nodes)
    M.add_nodes_from(server_nodes)
    #initialize parents, (node, level): SortedSet(parents of the node on the given level)
    parents_by_level = {}
    #for l in range(max_level + 2):
    #    parents_by_level[(source_node, l)] = SortedSet(key=lambda x: x[1])
    for c in client_nodes:
        for l in range(max_level + 2):
            parents_by_level[(c, l)] = SortedSet(key=lambda x: x[1])
    for s in server_nodes:
        for l in range(max_level + 2):
            parents_by_level[(s, l)] = SortedSet(key=lambda x: x[1])
    #initialize levels
    levels = {}
    best_gains = {}
    return H, M, parents_by_level, levels, best_gains


def clearStructures(G, H, parents_by_level, levels, best_gains, M, client_nodes, server_nodes, max_level, source_node):
    G.clear()
    H.clear()
    #for l in range(max_level + 2):
    #    parents_by_level[(source_node, l)].clear()
    for c in client_nodes:
        for l in range(max_level + 2):
            parents_by_level[(c, l)].clear()
    for s in server_nodes:
        for l in range(max_level + 2):
            parents_by_level[(s, l)].clear()
    M.clear()
    M.add_nodes_from(client_nodes)
    M.add_nodes_from(server_nodes)


def evaluateMatching(M, n):
    print('number of matching edges / n : ', len(M.edges()) / n)

    weight_of_matching = 0
    for (u, v, wt) in M.edges.data('weight'):
        weight_of_matching = weight_of_matching + wt
    print('Weight of the found matching: ', weight_of_matching)
    print('Average weight of an edge in the matching: ', weight_of_matching / len(M.edges()))


def matchingAsPermutation(F, n, client_nodes): #assumes that server_nodes == client_nodes + n
    matching = np.zeros((n, ), dtype=int)
    for c in client_nodes:
        if F.out_degree[c] == 1:
            matching[c] = list(F.successors(c))[0] - n
        else:
            matching[c] = np.random.random_integers(0, n-1)
    return matching


def calculateWeights(G, n, loadedIndex, batch_indices, latentBatch, targetPoints, n_nbrs, n_rndms, matching, fidelity=1):
    #initialize closeIndices and closeDistances
    closeIndices = np.zeros((n, n_nbrs), dtype=int)
    closeDistances = np.zeros((n, n_nbrs), dtype=np.float16)

    for i in range(len(batch_indices)):
        (closeIndices[i], closeDistances[i]) = loadedIndex.get_nns_by_vector(latentBatch[i], n_nbrs, include_distances=True)
        for j in range(n_nbrs):
            if matching[batch_indices[i]] == closeIndices[i, j]:
                wt = fidelity * closeDistances[i, j]
            else:
                wt = closeDistances[i, j]
            G.add_edge(batch_indices[i], n + closeIndices[i, j], weight=wt)
        for j in range(n_rndms):
            tempRandInt = randint(0, n - 1)
            if matching[batch_indices[i]] == tempRandInt:
                wt = fidelity * np.linalg.norm(targetPoints[tempRandInt] - latentBatch[i])
            else:
                wt = np.linalg.norm(targetPoints[tempRandInt] - latentBatch[i])
            G.add_edge(batch_indices[i], n + tempRandInt, weight=wt) #might add the same edge twice


class OOWrapper:
    def __init__(self, n, d, latentPoints, targetPoints, n_nbrs, n_rndms, max_level, n_trees=60):
        self.n = n
        self.d = d
        self.latentPoints = latentPoints
        self.targetPoints = targetPoints
        self.n_trees = n_trees
        self.n_nbrs = n_nbrs
        self.n_rndms = n_rndms
        self.max_level = max_level
        self.annoy_index = createAnnoyIndex(self.d, self.targetPoints, self.n_trees)
        self.G, self.client_nodes, self.server_nodes = createGraph(
            self.latentPoints, self.targetPoints, self.n_nbrs, self.n_rndms, self.annoy_index)
        #initialize F bipartate graph of forces
        self.F = nx.DiGraph()
        self.H, self.M, self.parents_by_level, self.levels, self.best_gains = initializeHelperStructures(
            self.client_nodes, self.server_nodes, self.max_level)
        initializeESGraph(self.H, self.parents_by_level, self.levels, self.best_gains, self.client_nodes,
                          self.server_nodes)
        self.buildMatching()
        self.evaluateMatching()
        self.matching = matchingAsPermutation(self.F, self.n, self.client_nodes)
        self.restart()

    def buildMatching(self):
        addAllClients(
            self.G, self.H, self.parents_by_level, self.levels, self.best_gains,
            self.M, self.F, self.client_nodes, self.server_nodes, self.max_level)

    def evaluateMatching(self):
        evaluateMatching(self.M, self.n)

    def restart(self):
        clearStructures(
            self.G, self.H, self.parents_by_level, self.levels, self.best_gains,
            self.M, self.client_nodes, self.server_nodes, self.max_level, source_node=-1)
        self.G, self.client_nodes, self.server_nodes = createGraph(
            self.latentPoints, self.targetPoints, self.n_nbrs, self.n_rndms, self.annoy_index)
        self.H, self.M, self.parents_by_level, self.levels, self.best_gains = initializeHelperStructures(
            self.client_nodes, self.server_nodes, self.max_level)
        initializeESGraph(self.H, self.parents_by_level, self.levels, self.best_gains, self.client_nodes,
                          self.server_nodes)

    def addBatch(self, batch_indices, latentBatch, fidelity=1):
        calculateWeights(self.G, self.n, self.annoy_index,
                         batch_indices, latentBatch,
                         self.targetPoints, self.n_nbrs, self.n_rndms, self.matching,
                         fidelity)
        addBatchOfClients(self.G,
                          batch_indices,
                          self.H, self.parents_by_level, self.levels, self.best_gains, self.M, self.F, self.max_level)

    def updateBatch(self, latentIndices, latentBatch, fidelity=1):
        self.latentPoints[latentIndices] = latentBatch
        self.addBatch(latentIndices, latentBatch, fidelity)
        for c in latentIndices:
            if self.M.out_degree[c] == 1:
                self.matching[c] = list(self.M.successors(c))[0] - self.n


# hopefully obsolete
def main_nonObjectOriented():
    n = 50000
    d = 10
    # max_level = floor(sqrt(n) * sqrt(log(n))) #=735 for n=50000
    max_level = 4
    n_nbrs = 11
    n_rndms = 0
    n_trees = 60
    source_node = -1

    latentPoints = normalize(np.random.normal(0, 1, (n, d)))
    targetPoints = normalize(np.random.normal(0, 1, (n, d)))

    start = time.clock()
    annoy_index = createAnnoyIndex(d, targetPoints, n_trees)
    G, client_nodes, server_nodes = createGraph(
                latentPoints=latentPoints, targetPoints=targetPoints,
                n_nbrs=n_nbrs, n_rndms=n_rndms, loadedIndex=annoy_index)
    print('Created G bipartite graph. Elapsed time: ', time.clock() - start)

    #initialize F bipartate graph of forces
    F = nx.DiGraph()
    H, M, parents_by_level, levels, best_gains = initializeHelperStructures(
                client_nodes, server_nodes, max_level)

    start = time.clock()
    addAllClients(G, H, parents_by_level, levels, best_gains, M, F, client_nodes, server_nodes, max_level)
    print('Created the initial ES graph. Elapsed time: ', time.clock() - start)

    evaluateMatching(M, n)

    start = time.clock()
    clearStructures(G, H, parents_by_level, levels, best_gains, M, client_nodes, server_nodes, max_level, source_node)
    print('Rebuilt ES graph. Elapsed time: ', time.clock() - start)

    #---simulate training by batches on an epoch---
    batch_size = 200
    start = time.clock()
    for i in range(n // batch_size):
        print(i)
        batch_indices = range(i * batch_size, (i+1) * batch_size)
        newLatentBatch = np.random.normal(0, 1, size=(batch_size, d))
        latentBatch = latentPoints[i * batch_size : (i+1) * batch_size]
        latentBatch[:] = newLatentBatch
        addBatch(
            G, n, annoy_index, batch_indices, latentBatch, targetPoints,
            H, parents_by_level, levels, best_gains,
            M, F, max_level, n_nbrs=n_nbrs, n_rndms=n_rndms)
        if i == n // batch_size // 2:
            print("At halftime:")
            evaluateMatching(M, n)

    print('Modified on one epoch. Elapsed time: ', time.clock() - start)

    evaluateMatching(M, n)


def main():
    n = 5000
    d = 10
    # max_level = floor(sqrt(n) * sqrt(log(n))) #=735 for n=50000
    max_level = 4
    n_nbrs = 11
    n_rndms = 0
    source_node = -1

    #latentPoints = normalize(np.random.normal(0, 1, (n, d)))
    #targetPoints = normalize(np.random.normal(0, 1, (n, d)))
    latentPoints = np.random.normal(0, 1, (n, d))
    targetPoints = np.random.normal(0, 1, (n, d))

    start = time.clock()
    oo = OOWrapper(
        n=n, d=d, latentPoints=latentPoints, targetPoints=targetPoints,
        n_trees=60, n_nbrs=n_nbrs, n_rndms=n_rndms, max_level=max_level)
    print('Created G bipartite graph. Elapsed time: ', time.clock() - start)

    start = time.clock()
    oo.buildMatching()
    print('Created the initial ES graph. Elapsed time: ', time.clock() - start)

    oo.evaluateMatching()

    start = time.clock()
    oo.restart()
    print('Rebuilt ES graph. Elapsed time: ', time.clock() - start)

    oo.buildMatching()
    oo.evaluateMatching()
    oo.restart()

    #---simulate training by batches on an epoch---
    batch_size = 200
    start = time.clock()
    for i in range(n // batch_size):
        print(i)
        batch_indices = range(i * batch_size, (i+1) * batch_size)
        #newLatentBatch = np.random.normal(0, 1, size=(batch_size, d))
        #latentBatch = latentPoints[i * batch_size : (i+1) * batch_size]
        #latentBatch[:] = newLatentBatch
        #oo.updateBatch(batch_indices, latentBatch)
        oo.updateBatch(batch_indices, latentPoints[batch_indices])
        if i == n // batch_size // 2:
            print("At halftime:")
            oo.evaluateMatching()

    print('Modified on one epoch. Elapsed time: ', time.clock() - start)

    oo.evaluateMatching()


if __name__ == "__main__":
    main()
    #main_nonObjectOriented()
