import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from scipy.sparse import coo_matrix
from annoy import AnnoyIndex
import time
from sortedcontainers import SortedSet
import directed_unweighted_ES as ES
from math import sqrt, log, floor
from random import randint

def normalize(a, order=2, axis=-1):
    a_norms = np.expand_dims(np.linalg.norm(a, order, axis), axis)
    return a / a_norms

def createGraph(n, d, latentPoints, n_trees, n_nbrs, n_rndms):
    targetPoints = normalize(np.random.normal(0, 1, (n, d)))

    targetIndex = AnnoyIndex(d)
    #add each of the target points
    for i in range(targetPoints.shape[0]):
        targetIndex.add_item(i, targetPoints[i])

    #build the LSH-forest with the target points
    targetIndex.build(n_trees)

    #save and load with memory map
    targetIndex.save("LSHForest.ann")
    loadedIndex = AnnoyIndex(d)
    loadedIndex.load("LSHForest.ann")

    closeIndices = np.zeros((latentPoints.shape[0], n_nbrs), dtype=int)
    closeDistances = np.zeros((latentPoints.shape[0], n_nbrs), dtype=np.float16)

    #create the bipartite graph in networkx
    G = nx.Graph()
    G.add_nodes_from(range(0, latentPoints.shape[0]), bipartite=0)
    G.add_nodes_from(range(latentPoints.shape[0], latentPoints.shape[0] + targetPoints.shape[0]), bipartite=1)

    for i in range(latentPoints.shape[0]):
        (closeIndices[i], closeDistances[i]) = loadedIndex.get_nns_by_vector(latentPoints[i], n_nbrs, include_distances=True)
        for j in range(n_nbrs):
            G.add_edge(i, latentPoints.shape[0] + closeIndices[i, j], weight=(-1) * closeDistances[i, j])
        for j in range(n_rndms):
            tempRandInt = randint(0, targetPoints.shape[0] - 1)
            G.add_edge(i, latentPoints.shape[0] + tempRandInt,
                       weight=(-1) * np.linalg.norm(targetPoints[tempRandInt] - latentPoints[i])) #might add the same edge twice

    #print(len(set(G)))

    client_nodes = SortedSet(range(latentPoints.shape[0]))
    server_nodes = SortedSet(range(latentPoints.shape[0], latentPoints.shape[0] + targetPoints.shape[0]))

    return G, client_nodes, server_nodes, loadedIndex, targetPoints


def main():
    n = 500
    d = 10
    n_nbrs = 7
    n_rndms = 0
    n_trees = 60

    latentPoints = normalize(np.random.normal(0, 1, (n, d)))

    (G, client_node, server_nodes, loadedIndex, targetPoints) = createGraph(n, d, latentPoints, n_trees, n_nbrs, n_rndms)
    print(len(G.nodes))
    print(len(G.edges))

    start = time.clock()
    matching = nx.algorithms.max_weight_matching(G, maxcardinality=True)
    end = time.clock()
    w = 0
    for m in matching:
        w = w + G.edges[m]['weight']
    print("Sum of weights:", -w)
    print("Number of matched edges / n: ", len(matching)/n)
    print("Average length of edge in the matching:" , -w/n)
    print("Elapsed time:", end - start)



if __name__ == "__main__":
    main()