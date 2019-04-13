import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
import numpy as np
import networkx as nx


def pairwiseSquaredDistances(clients, servers):
    cL2S = np.sum(clients ** 2, axis=-1)
    sL2S = np.sum(servers ** 2, axis=-1)
    cL2SM = np.tile(cL2S, (len(servers), 1))
    sL2SM = np.tile(sL2S, (len(clients), 1))
    squaredDistances = cL2SM + sL2SM.T - 2.0 * servers.dot(clients.T)
    return squaredDistances.T


def distance_test():
    n = 1000
    d = 10

    clients = np.random.normal(size=(n, d))
    servers = np.random.normal(size=(n, d)) + 1
    squaredDistances = pairwiseSquaredDistances(clients, servers)
    for i in range(5):
        for j in range(5):
            print(squaredDistances[i, j], np.linalg.norm(clients[i] - servers[j]) ** 2)


def main():
    n = 100
    d = 10

    np.random.seed(1)
    clients = np.random.normal(size=(n, d))
    servers = np.random.normal(size=(n, d)) + 1
    distances = np.sqrt(pairwiseSquaredDistances(clients, servers))
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, bipartite=0)
        G.add_node(n + i, bipartite=1)
    mode = "sparse"
    if mode == "sparse":
        k = 20
        for i in range(n):
            neighbors = np.argsort(distances[i, :])[:k]
            for j in neighbors:
                G.add_edge(i, n+j, weight=-distances[i, j])
    elif mode == "dense":
        for i in range(n):
            for j in range(n):
                G.add_edge(i, n+j, weight=-distances[i, j])
    elif mode == "random":
        k = 10
        for i in range(n):
            randoms = np.random.choice(n, k)
            for j in randoms:
                G.add_edge(i, n+j, weight=-distances[i, j])
    elif mode == "mixed":
        k = 10
        for i in range(n):
            neighbors = np.argsort(distances[i, :])[:k]
            for j in neighbors:
                G.add_edge(i, n+j, weight=-distances[i, j])
            randoms = np.random.choice(n, k)
            for j in randoms:
                G.add_edge(i, n+j, weight=-distances[i, j])

    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True, weight='weight')
    print(len(matching), "/", n)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(clients[:, 0], clients[:, 1], c='red')
    ax.scatter(servers[:, 0], servers[:, 1], c='blue')
    totalDistance = 0.0
    for a, b in matching:
        if a >= n:
            b, a = a, b
        client = a
        server = b - n
        xs = [clients[client, 0], servers[server, 0]]
        ys = [clients[client, 1], servers[server, 1]]
        l = Line2D(xs, ys)
        ax.add_line(l)
        totalDistance += distances[client, server]
    print(totalDistance)

    plt.savefig(sys.argv[1])


if __name__ == "__main__":
    main()
