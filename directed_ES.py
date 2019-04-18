import networkx as nx
#import itertools
from sortedcontainers import SortedSet
import numpy as np


G = nx.Graph() #<----TODO input from NAT_graph, placeholder atm
max_level = 485 #int: sqrt(n)*sqrt(log(n))
source_node = 0 #int: index of the source node
#M = nx.DiGraph() #matching, directed from C to S
#n = 50000


def readjustGraph(H, affected_nodes, max_level):
    lvl = H.nodes(affected_nodes[0])['level']
    new_affected_nodes = SortedSet()
    for v in affected_nodes:
        if len(H.nodes(v)['witnesses']) == 0 and lvl < max_level:
            H.nodes(v)['level'] += 1
            for w in H.successors(v):
                H.remove_edge(v, w)
                H.nodes(w)['witnesses'].discard(v)
                new_affected_nodes.add(w)
            new_affected_nodes.add(v)
    if len(new_affected_nodes) > 0:
        readjustGraph(H, new_affected_nodes, max_level)


def deleteEdge(H, u, v, max_level):
    H.remove_edge(u, v)
    H.nodes(v)['witnesses'].discard(u)
    affected_nodes = SortedSet()
    affected_nodes.add(v) #affected nodes are on the same level by definition
    readjustGraph(H, affected_nodes, max_level)


def addOneClient(H, c, source_node, max_level):
    for s in G.neighbors(c):
        H.add_edge(s, c)
    deleteEdge(H, source_node, c, max_level)


def findSAP(H, c, source_node):
    if len(H.nodes(c)['witnesses']) > 0: #exists a SAP to c with length <= max_level
        p = c
        lvl = H.nodes(c)['level']
        path = np.zeros((lvl + 1, ))
        path[lvl] = c
        for i in range(H.nodes(c)['level'] - 1, 0, -1):
            p = H.nodes(p)['witnesses'][0] #could take any!
            path[i] = p
            #P.add_edge(H.nodes(p)['witnesses'][0], p)
    else: #have to find a SAP with BFS
        path = np.array(nx.algorithms.shortest_path(H, source_node, c, weight=None))[1:] #array of consecutive path-nodes
        #for i in range(len(path) - 1):
        #    P.add_edge(path[i], path[i+1])
    return path


def applyPath(H, M, path, max_level, source_node):
    for i in range(len(path) - 1, 1, -2): #len(path)-1 .. 3, since len(path)-1 is odd
        M.add_edge(path[i], path[i-1])
        M.remove_edge(path[i-2], path[i-1])
    M.add_edge(path[1], path[0])

    for i in range(len(path) - 1, 0, -1):
        H.add_edge(path[i], path[i-1])
        deleteEdge(H, path[i-1], path[i], max_level)
    deleteEdge(H, source_node, path[0], max_level)


def addClients(clients_to_add, H, M, source_node, max_level):
    for c in clients_to_add:
        addOneClient(H, c, source_node, max_level)
        path = findSAP(H, c, source_node)
        applyPath(H, M, path, max_level, source_node)


def deleteClients(clients_to_delete, H, M, source_node, max_level):
    for c in clients_to_delete:
        for s in H.predecessors(c):
            deleteEdge(H, s, c, max_level)
        for s in H.successors(c):
            deleteEdge(H, c, s, max_level)

        for s in M.successors(c):
            deleteEdge(M, c, s, max_level)

    for c in clients_to_delete:
        H.add_edge(source_node, c)


server_nodes = SortedSet({n for n, d in G.nodes(data=True) if d['bipartite']==0})
client_nodes = SortedSet(set(G) - server_nodes)

#unmatched_arrived_clients = (client_nodes - not_arrived_clients) - set(M)

#D = nx.DiGraph()
#D.add_nodes_from(G)
#for s in server_nodes:
#    for c in set(G.adj[s]) - not_arrived_clients:
#        D.add_edge(s, c)
#
#D.remove_edges_from(M.reverse())
#D.add_nodes_from(M)


#initialize H before the clients being added iteratively
H = nx.DiGraph()

H.add_node(source_node, level=0)
H.add_nodes_from(client_nodes, level=1)

for s in server_nodes:
    H.add_edge(source_node, s)
    H.nodes(s)['witnesses'] = SortedSet()
    H.nodes(s)['witnesses'].add(source_node)

for c in client_nodes:
    H.add_edge(source_node, c)
    H.nodes(c)['witnesses'] = SortedSet()
    H.nodes(c)['witnesses'].add(source_node)

#initialize M
M = nx.DiGraph()

#initialize path graph
#P = nx.DiGraph()

#free_servers = SortedSet(server_nodes - set(M))

#add all the clients 1 by 1 while maintaining the ES structure
clients_to_add = client_nodes
addClients(clients_to_add, H, M, source_node, max_level)

#add (a batch of) clients
clients_to_add=... #Todo from network
addClients(clients_to_add, H, M, source_node, max_level)

#delete (a batch of) clients
clients_to_delete=... #Todo from network
deleteClients(clients_to_delete, H, M, source_node, max_level)

