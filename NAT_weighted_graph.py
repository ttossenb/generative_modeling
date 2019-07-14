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


def binMatrix(d):
    binaries = np.array([[0], [1]], dtype=int)
    for i in range(1, d):
        binaries = np.vstack((np.concatenate((np.zeros((2 ** i, 1), dtype=int), binaries), axis=-1),
                              np.concatenate((np.ones((2 ** i, 1), dtype=int), binaries), axis=-1)))
    return binaries


def createGraph(n, d, latentPoints, targetPoints, n_trees, n_nbrs, n_rndms):
    #create AnnoyIndex in R^d
    targetIndex = AnnoyIndex(d, metric='euclidean')
    #add each of the target points
    for i in range(targetPoints.shape[0]):
        targetIndex.add_item(i, targetPoints[i])

    #build the LSH-forest with the target points
    targetIndex.build(n_trees)

    #save and load with memory map
    targetIndex.save("LSHForest.ann")
    loadedIndex = AnnoyIndex(d, metric='euclidean')
    loadedIndex.load("LSHForest.ann")

    #end1 = time.clock()
    #start2 = time.clock()

    #find the closest neighbours for each latent point and their distances,
    #and also create the biadjacency sparse matrix of the desired bipartite graph
    #initialize with an empty sparse matrix
    #B = coo_matrix((latentPoints.shape[0], targetPoints.shape[0]), dtype=np.float16)
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

    #end2 = time.clock()
    #start3 = time.clock()

    #create the biadjacency matrix
    #arg1=data=(data, (rows, cols))
    #B = coo_matrix((closeDistances.flatten(), (np.repeat(np.arange(latentPoints.shape[0]), n_nbrs), closeIndices.flatten())))

    #create the bipartite graph in networkx
    #G = bipartite.from_biadjacency_matrix(B)
    #G.add_nodes_from(range(n, n + (2 ** d) * k), )
    print(len(set(G)))

    client_nodes = SortedSet(range(latentPoints.shape[0]))
    server_nodes = SortedSet(range(latentPoints.shape[0], latentPoints.shape[0] + targetPoints.shape[0]))

    #client_nodes = SortedSet({n for n, d in G.nodes(data=True) if d['bipartite']==0}) #0 .. n-1
    #server_nodes = SortedSet(set(G) - client_nodes)

    #end3 = time.clock()

    #print("Elapsed time during the initialization of the targets: ", end1-start1)
    #print("Elapsed time during the initialization of the LSH-forest: ", end2-start2)
    #print("Elapsed time during the creation of the bipartite graph: ", end3-start3)

    return G, client_nodes, server_nodes, loadedIndex


def createESGraph(G, H, parents_by_level, levels, best_gains, M, F, client_nodes, server_nodes, max_level, source_node=-1):
    #H = directed ES graph
    #n = len(client_nodes)

    H.add_node(source_node)
    levels[source_node] = 0
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


def deleteBatchOfClients(clients_to_delete, H, parents_by_level, levels, best_gains, M, max_level, source_node=-1):
    #clients_to_delete = sorted set of the indices of the batch elements to delete
    ES.deleteClients(clients_to_delete, H, M, source_node, max_level)


def updateBatch(G, n, annoy_index, batch_indices, latentBatch, targetPoints, H, parents_by_level, levels, best_gains, M, max_level, n_nbrs, n_rndms=0, source_node=-1):
    #latentBatch=latentPoints["batch_indices as np.array"] (must be sorted)
    #delete the nodes (and the edges from these nodes)
    deleteBatchOfClients(batch_indices, H, parents_by_level, levels, best_gains, M, max_level)
    G.remove_nodes_from(batch_indices)
    #add back the nodes
    G.add_nodes_from(batch_indices)
    #initialize closeIndices and closeDistances
    closeIndices = np.zeros((latentBatch.shape[0], n_nbrs), dtype=int)
    closeDistances = np.zeros((latentBatch.shape[0], n_nbrs), dtype=np.float16)
    #add new weighted edges to G
    for i in range(len(batch_indices)):
        (closeIndices[i], closeDistances[i]) = annoy_index.get_nns_by_vector(latentBatch[i], n_nbrs, include_distances=True)
        for j in range(n_nbrs):
            G.add_edge(batch_indices[i], closeIndices[i][j] + n, weight=closeDistances[i][j])
        for j in range(n_rndms):
            tempRandInt = randint(0, n-1) #n as number of SERVER nodes, might need to change it!!!
            G.add_edge(batch_indices[i], n + tempRandInt,
                       weight=np.linalg.norm(latentBatch[i] - targetPoints[tempRandInt])) #might add the same edge twice
    ES.addClients(G, batch_indices, H, M, source_node, max_level)


def addBatch(G, n, annoy_index, batch_indices, latentBatch, targetPoints, H, parents_by_level, levels, best_gains, M, F, max_level, n_nbrs, n_rndms=0, source_node=-1):
    #latentBatch=latentPoints["batch_indices as np.array"] (must be sorted)
    #delete the nodes (and the edges from these nodes)
    #G.remove_nodes_from(batch_indices)
    #add back the nodes
    batch_indices = SortedSet(batch_indices)
    G.add_nodes_from(batch_indices)
    #initialize closeIndices and closeDistances
    closeIndices = np.zeros((latentBatch.shape[0], n_nbrs), dtype=int)
    closeDistances = np.zeros((latentBatch.shape[0], n_nbrs), dtype=np.float16)
    #add new weighted edges to G
    for i in range(len(batch_indices)):
        (closeIndices[i], closeDistances[i]) = annoy_index.get_nns_by_vector(latentBatch[i], n_nbrs, include_distances=True)
        for j in range(n_nbrs):
            G.add_edge(batch_indices[i], closeIndices[i][j] + n, weight=closeDistances[i][j])
        for j in range(n_rndms):
            tempRandInt = randint(0, n-1) #n as the number of SERVER nodes, might need to change it!!!
            G.add_edge(batch_indices[i], n + tempRandInt,
                       weight=np.linalg.norm(latentBatch[i] - targetPoints[tempRandInt])) #might add the same edge twice
    ES.addClients(G, batch_indices, H, parents_by_level, levels, best_gains, M, source_node, max_level)
    #readjust F
    for c in batch_indices:
        for s in set(F.successors(c)):
            F.remove_edge(c, s)
        for s in M.successors(c):
            F.add_edge(c, s)


def initializeHelperStructures(client_nodes, server_nodes, max_level):
    #initialize H before the clients are being added iteratively
    H = nx.DiGraph()
    #initialize M digraph of the matching directed from C to S
    M = nx.DiGraph()
    M.add_nodes_from(client_nodes)
    M.add_nodes_from(server_nodes)
    #initialize F bipartate graph of forces
    F = nx.DiGraph()
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
    return H, M, F, parents_by_level, levels, best_gains


def restart(G, H, parents_by_level, levels, best_gains, M, client_nodes, server_nodes, max_level, source_node):
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
    initializeESGraph(H, parents_by_level, levels, best_gains, client_nodes, server_nodes, source_node)
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


class OOWrapper:
    def __init__(self, n, d, latentPoints, targetPoints, n_trees, n_nbrs, n_rndms, max_level):
        self.n = n
        self.d = d
        self.latentPoints = latentPoints
        self.targetPoints = targetPoints
        self.n_trees = n_trees
        self.n_nbrs = n_nbrs
        self.n_rndms = n_rndms
        self.max_level = max_level
        self.G, self.client_nodes, self.server_nodes, self.annoy_index = createGraph(
            n, d, latentPoints, targetPoints, n_trees, n_nbrs, n_rndms)
        self.H, self.M, self.F, self.parents_by_level, self.levels, self.best_gains = initializeHelperStructures(
            self.client_nodes, self.server_nodes, self.max_level)

    def buildMatching(self):
        createESGraph(
            self.G, self.H, self.parents_by_level, self.levels, self.best_gains,
            self.M, self.F, self.client_nodes, self.server_nodes, self.max_level)

    def evaluateMatching(self):
        evaluateMatching(self.M, self.n)

    def restart(self):
        restart(
            self.G, self.H, self.parents_by_level, self.levels, self.best_gains,
            self.M, self.client_nodes, self.server_nodes, self.max_level, source_node=-1)

    def addBatch(self, batch_indices, latentBatch):
        addBatch(
            self.G, self.n, self.annoy_index,
            batch_indices, latentBatch,
            self.targetPoints, self.H, self.parents_by_level, self.levels, self.best_gains,
            self.M, self.F, self.max_level, self.n_nbrs, self.n_rndms)


# hopefully obsolete
def main_nonObjectOriented():
    n = 50000
    d = 10
    # max_level = floor(sqrt(n) * sqrt(log(n))) #=735 for n=50000
    max_level = 4
    n_nbrs = 11
    n_rndms = 0
    source_node = -1

    latentPoints = normalize(np.random.normal(0, 1, (n, d)))
    targetPoints = normalize(np.random.normal(0, 1, (n, d)))

    start = time.clock()
    G, client_nodes, server_nodes, annoy_index = createGraph(
                n=n, d=d, latentPoints=latentPoints, targetPoints=targetPoints,
                n_trees=60, n_nbrs=n_nbrs, n_rndms=n_rndms)
    print('Created G bipartite graph. Elapsed time: ', time.clock() - start)

    H, M, F, parents_by_level, levels, best_gains = initializeHelperStructures(
                client_nodes, server_nodes, max_level)

    start = time.clock()
    createESGraph(G, H, parents_by_level, levels, best_gains, M, F, client_nodes, server_nodes, max_level)
    print('Created the initial ES graph. Elapsed time: ', time.clock() - start)

    evaluateMatching(M, n)

    start = time.clock()
    restart(G, H, parents_by_level, levels, best_gains, M, client_nodes, server_nodes, max_level, source_node)
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
    n = 50000
    d = 10
    # max_level = floor(sqrt(n) * sqrt(log(n))) #=735 for n=50000
    max_level = 4
    n_nbrs = 11
    n_rndms = 0
    source_node = -1

    latentPoints = normalize(np.random.normal(0, 1, (n, d)))
    targetPoints = normalize(np.random.normal(0, 1, (n, d)))

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

    #---simulate training by batches on an epoch---
    batch_size = 200
    start = time.clock()
    for i in range(n // batch_size):
        print(i)
        batch_indices = range(i * batch_size, (i+1) * batch_size)
        newLatentBatch = np.random.normal(0, 1, size=(batch_size, d))
        latentBatch = latentPoints[i * batch_size : (i+1) * batch_size]
        latentBatch[:] = newLatentBatch
        oo.addBatch(batch_indices, latentBatch)
        if i == n // batch_size // 2:
            print("At halftime:")
            oo.evaluateMatching()

    print('Modified on one epoch. Elapsed time: ', time.clock() - start)

    oo.evaluateMatching()


if __name__ == "__main__":
    main()
    #main_nonObjectOriented()
