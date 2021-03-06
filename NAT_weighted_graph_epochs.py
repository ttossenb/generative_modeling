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


def buildServers(n, d):
    #number of trees in the LSH-forest
    n_trees = 60

    targetPoints = normalize(np.random.normal(0, 1, size=(n, d)))
    #create AnnoyIndex in R^d
    targetIndex = AnnoyIndex(d, metric='euclidean')
    #add each of the target points
    for i in range(targetPoints.shape[0]):
        targetIndex.add_item(i, targetPoints[i])
    targetIndex.build(n_trees)
    B = nx.Graph()
    B.add_nodes_from(range(n, 2*n), bipartite=1)
    return targetPoints, B, targetIndex


def createGraph(n, d, latentPoints, n_trees, n_nbrs, n_rndms):
    #n = number of traing inputs
    #n = 50000
    #d = latent dimension
    #d = 10
    #n_trees = number of trees in the LSH-forest
    #n_trees = 60
    #n_nbrs = number of closest neighbourers to check
    #n_nbrs = 10

    #number of points to put in each space segment
    #k = n // (2 ** d)

    #start1 = time.clock()

    #generating target points, normalizing them, taking the absolute value of each
    #targetPoints = np.absolute(normalize(np.random.normal(0, 1, (2 ** d, k, d))))

    targetPoints = normalize(np.random.normal(0, 1, (n, d)))

    #binaries = np.expand_dims(binMatrix(d), axis=1)

    #0 -> 1 (don't flip sign), 1 -> -1 (flip sign)
    #targetPoints = np.reshape(((-2) * binaries + 1) * targetPoints, ((2 ** d) * k, d))
    #np.random.shuffle(targetPoints)

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
            #print("closeDistances: ", closeDistances[i, j])
            #print(np.linalg.norm(latentPoints[i] - targetPoints[closeIndices[i, j]]))
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

    return G, client_nodes, server_nodes, loadedIndex, targetPoints


def createESGraph(G, H, parents_by_level, levels, best_gains, M, client_nodes, server_nodes, max_level, source_node=-1):
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


def addBatch(G, n, annoy_index, batch_indices, latentBatch, targetPoints, H, parents_by_level, levels, best_gains, M, max_level, n_nbrs, n_rndms=0, source_node=-1):
    #latentBatch=latentPoints["batch_indices as np.array"] (must be sorted)
    #delete the nodes (and the edges from these nodes)
    #G.remove_nodes_from(batch_indices)
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
            tempRandInt = randint(0, n-1) #n as the number of SERVER nodes, might need to change it!!!
            G.add_edge(batch_indices[i], n + tempRandInt,
                       weight=np.linalg.norm(latentBatch[i] - targetPoints[tempRandInt])) #might add the same edge twice
    ES.addClients(G, batch_indices, H, parents_by_level, levels, best_gains, M, source_node, max_level)
    #readjust F
    #for c in batch_indices:
    #    for s in set(F.successors(c)):
    #        F.remove_edge(c, s)
    #    for s in M.successors(c):
    #        F.add_edge(c, s)


def main():
    n = 50000
    d = 10
    #max_level = floor(sqrt(n) * sqrt(log(n))) #=735 for n=50000
    max_level = 6
    n_nbrs = 11
    n_rndms = 0
    source_node = -1
    n_epochs = 1
    batch_size = 200

    #placeholder for input
    latentPoints = normalize(np.random.normal(0, 1, (n, d)))
    #latentPoints = normalize(np.load('latent_points_10.npy')[:n])

    start1 = time.clock()
    #---before training---
    (G, client_nodes, server_nodes, annoy_index, targetPoints) = createGraph(n=n, d=10, latentPoints=latentPoints,
                                                               n_trees=60, n_nbrs=n_nbrs, n_rndms=n_rndms)
    end1 = time.clock()
    print('Created G bipartate graph. Elapsed time: ', end1 - start1)

    #initialize H before the clients are being added iteratively
    H = nx.DiGraph()
    #initialize M digraph of the matching directed from C to S
    M = nx.DiGraph()
    M.add_nodes_from(client_nodes)
    M.add_nodes_from(server_nodes)
    #initialize F bipartate graph of forces
    #F = nx.DiGraph()
    #initialize levels
    levels = {}
    best_gains = {}
    free_gains = SortedSet(key=lambda x: x[1])
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

    for e in range(n_epochs): #simulate epochs
        start2 = time.clock()
        initializeESGraph(H, parents_by_level, levels, best_gains, client_nodes, server_nodes, source_node=source_node)
        for i in range(n // batch_size): #simulate training by batches on an epoch
            batch_indices = SortedSet(range(i * batch_size, (i+1) * batch_size)) #placeholder
            #latentBatch = normalize(np.random.normal(0, 1, size=(batch_size, 10))) #placeholder
            latentBatch = latentPoints[i * batch_size : (i+1) * batch_size]  # placeholder
            addBatch(G, n, annoy_index, batch_indices, latentBatch, targetPoints, H, parents_by_level, levels, best_gains, M, max_level, n_nbrs=n_nbrs, n_rndms=n_rndms)
        end2 = time.clock()

        print('number of matching edges / n : ', len(M.edges()) / n)

        #deepcopy M to F
        #F.add_nodes_from(M.nodes)
        #F.add_edges_from(M.edges)
        print('Created the initial ES graph. Elapsed time: ', end2 - start2)

        weight_of_matching = 0
        for (u, v, wt) in M.edges.data('weight'):
            weight_of_matching = weight_of_matching + wt
        print('Weight of the found matching: ', weight_of_matching)
        print('Average weight of an edge in the matching: ', weight_of_matching / len(M.edges()))

        improvable = True
        while improvable:
            for s in server_nodes:
                if levels[s] > 1 and levels[s] <= max_level:
                    #print('s: ', s)
                    #print(levels[s])
                    #print((s, parents_by_level[(s, levels[s] - 1)][0][1]))
                    free_gains.add((s, parents_by_level[(s, levels[s] - 1)][0][1]))
            #print(len(free_gains))
            print(free_gains[0])
            if free_gains[0][1] >= 0:
                improvable = False
            else:
                (path, found_short_path) = ES.findSAP(H, parents_by_level, levels, best_gains, free_gains[0][0])
                ES.applyServerPath(H, parents_by_level, levels, best_gains, M, path, max_level, found_short_path, source_node)
            free_gains.clear()

        weight_of_matching = 0
        for (u, v, wt) in M.edges.data('weight'):
            weight_of_matching = weight_of_matching + wt
        print('Weight of the improved matching: ', weight_of_matching)
        print('Average weight of an edge in the matching: ', weight_of_matching / len(M.edges()))

        #reset for the next epoch
        G.clear()
        H.clear()
        M.clear()
        M.add_nodes_from(client_nodes)
        M.add_nodes_from(server_nodes)

        #for l in range(max_level + 2):
        #    parents_by_level[(source_node, l)].clear()
        for c in client_nodes:
            for l in range(max_level + 2):
                parents_by_level[(c, l)].clear()
        for s in server_nodes:
            for l in range(max_level + 2):
                parents_by_level[(s, l)].clear()

        #placeholder for latent points
        latentPoints = normalize(np.random.normal(0, 1, (n, d)))
        #latentPoints = normalize(np.load('latent_points_10.npy')[:n])


if __name__ == "__main__":
    main()
