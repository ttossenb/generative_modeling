import numpy as np
import time
import networkx as nx

import NAT_weighted_graph
import NAT_straight
import autoencoder


def main():
    n = 1000
    d = 10
    # max_level = floor(sqrt(n) * sqrt(log(n))) #=735 for n=50000
    max_level = 4
    n_nbrs = 10
    n_rndms = 10
    source_node = -1

    latentPoints = np.random.normal(0, 1, (n, d))

    start = time.clock()
    G, client_nodes, server_nodes, annoy_index, targetPoints = NAT_weighted_graph.createGraph(
                n=n, d=d, latentPoints=latentPoints, n_trees=60, n_nbrs=n_nbrs, n_rndms=n_rndms)
    print('Created G bipartate graph. Elapsed time: ', time.clock() - start)

    H, M, F, parents_by_level, levels, best_gains = NAT_weighted_graph.initializeHelperStructures(
                client_nodes, server_nodes, max_level)

    start = time.clock()
    NAT_weighted_graph.createESGraph(G, H, parents_by_level, levels, best_gains, M, F, client_nodes, server_nodes, max_level)
    print('Created the initial ES graph. Elapsed time: ', time.clock() - start)

    NAT_weighted_graph.evaluateMatching(M, n)

    start = time.clock()
    NAT_weighted_graph.restart(G, H, parents_by_level, levels, best_gains, M, client_nodes, server_nodes, max_level, source_node)
    print('Rebuilt ES graph. Elapsed time: ', time.clock() - start)

    #---simulate training by batches on an epoch---
    batch_size = 200
    start = time.clock()
    for i in range(n // batch_size):
        print(i)
        batch_indices = range(i * batch_size, (i+1) * batch_size)
        newLatentBatch = np.random.normal(0, 1, size=(batch_size, d)) + 1
        latentBatch = latentPoints[i * batch_size : (i+1) * batch_size]
        latentBatch[:] = newLatentBatch
        NAT_weighted_graph.addBatch(
            G, n, annoy_index, batch_indices, latentBatch, targetPoints,
            H, parents_by_level, levels, best_gains,
            M, F, max_level, n_nbrs=n_nbrs, n_rndms=n_rndms)
        if i == n // batch_size // 2:
            print("At halftime:")
            NAT_weighted_graph.evaluateMatching(M, n)

    print('Modified on one epoch. Elapsed time: ', time.clock() - start)

    NAT_weighted_graph.evaluateMatching(M, n)


# hard to directly compare, because this does not use sparse graph approximation.
def natBaseline(latentPoints, targetPoints, batch_size, epoch_count):
    import autoencoder # TODO should have its own .py
    n = len(latentPoints)
    assert n == len(targetPoints)
    assert n % batch_size == 0
    batch_count = n // batch_size
    matching = np.random.permutation(n)
    for e in range(epoch_count):
        random_permutation = np.random.permutation(n)
        global_wom = autoencoder.weight_of_matching(matching, latentPoints, targetPoints)
        print("epoch %d weight %f" % (e, global_wom))
        for i in range(batch_count):
            latentIndices = random_permutation[range(i*batch_size, (i+1)*batch_size)]
            targetIndices = matching[latentIndices]
            latentBatch = latentPoints[latentIndices]
            targetBatch = targetPoints[targetIndices]

            localMatching = autoencoder.updateBipartiteGraphFromScratch(latentBatch, targetBatch)

            # see https://github.com/danielvarga/repulsive-autoencoder/blob/223d2b96f16e6916c4e4398883a0eb8d1eac88b9/earthMoverNAT.py#L260
            matching[latentIndices] = matching[latentIndices[localMatching]]


def natBaselineOO(latentPoints, targetPoints, batch_size, epoch_count):
    oo = NAT_straight.OOWrapper(latentPoints=latentPoints, targetPoints=targetPoints)
    print("start", 0, "weight", oo.evaluateMatching())
    for epoch in range(epoch_count):
        oo.doEpoch(batch_size)
        print("epoch", epoch + 1, "weight", oo.evaluateMatching())


def natBaselineTest():
    n = 1000
    d = 10

    np.random.seed(1)
    latentPoints = np.random.normal(0, 1, (n, d))
    targetPoints = np.random.normal(0, 1, (n, d))
    # classes = 10
    # targetPoints = np.random.normal(0, 1.0 / classes, (classes, n // classes, d))
    # targetPoints += np.linspace(0, 1, classes)[:, np.newaxis, np.newaxis]
    # targetPoints = targetPoints.reshape((n, d))

    print("Classic minibatch-oriented NaT.")
    batch_size = 50
    epoch_count = 10

    natBaselineOO(latentPoints, targetPoints, batch_size, epoch_count)

    print("Full bloom matching on sparse graph.")
    G = nx.Graph()
    G.add_nodes_from(range(n), bipartite=0)
    G.add_nodes_from(range(n, 2 * n), bipartite=1)

    # TODO n_nbrs n_rndms are hardwired now.
    matching = autoencoder.updateBipartiteGraph(range(n), latentPoints, targetPoints, G, annoyIndex=None)
    print("Weight:", autoencoder.weight_of_matching(matching, latentPoints, targetPoints))

    print("Toszi's algorithm.")

    n_nbrs = 50
    n_rndms = 0
    max_level = 4
    oo = NAT_weighted_graph.OOWrapper(
        n=n, d=d, latentPoints=latentPoints, targetPoints=targetPoints, n_trees=60, n_nbrs=n_nbrs, n_rndms=n_rndms, max_level=max_level)
    oo.buildMatching()
    oo.evaluateMatching()






def mainOO():
    n = 1000
    d = 10
    # max_level = floor(sqrt(n) * sqrt(log(n))) #=735 for n=50000
    max_level = 4
    n_nbrs = 10
    n_rndms = 10
    source_node = -1

    latentPoints = np.random.normal(0, 1, (n, d))
    targetPoints = np.random.normal(0, 1, (n, d))

    start = time.clock()
    oo = NAT_weighted_graph.OOWrapper(
        n=n, d=d, latentPoints=latentPoints, targetPoints=targetPoints, n_trees=60, n_nbrs=n_nbrs, n_rndms=n_rndms, max_level=max_level)
    print('Created G bipartate graph. Elapsed time: ', time.clock() - start)

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
        newLatentBatch = np.random.normal(0, 1, size=(batch_size, d)) + 1
        latentBatch = latentPoints[i * batch_size : (i+1) * batch_size]
        latentBatch[:] = newLatentBatch
        oo.addBatch(batch_indices, latentBatch)
        if i == n // batch_size // 2:
            print("At halftime:")
            oo.evaluateMatching()

    print('Modified on one epoch. Elapsed time: ', time.clock() - start)

    oo.evaluateMatching()


if __name__ == "__main__":
    natBaselineTest()
    # main()
    # mainOO()
