import networkx as nx
#import itertools
from sortedcontainers import SortedSet
import numpy as np

#G = nx.Graph() #<----TODO input from NAT_graph, placeholder atm
#max_level = 735 #int: sqrt(n)*sqrt(log(n))
#source_node = -1 #int: index of the source node
#M = nx.DiGraph() #matching, directed from C to S
#n = 50000


def readjustGraph(H, affected_nodes, max_level):
    #print('affected nodes: ', affected_nodes)
    lvl = H.nodes[affected_nodes[0]]['level']
    #print('level of affected nodes: ', lvl)
    new_affected_nodes = SortedSet()
    increased_nodes = SortedSet()
    for v in affected_nodes:
        if len(H.nodes[v]['witnesses']) == 0 and lvl < max_level:
            H.nodes[v]['level'] += 1
            increased_nodes.add(v)
    for v in increased_nodes:
        new_affected_nodes.add(v)
        for w in set(H.successors(v)): #changes dictionary during iteration so we use set()
            H.remove_edge(v, w)
            H.nodes[w]['witnesses'].discard(v)
            new_affected_nodes.add(w)

        H.nodes[v]['witnesses'].clear()
        for u in H.predecessors(v):
            if H.nodes[u]['level'] == lvl:
                #print(u, ' is a new witness of the increased node ', v)
                H.nodes[v]['witnesses'].add(u)
    if len(new_affected_nodes) > 0:
        readjustGraph(H, new_affected_nodes, max_level)


def deleteEdge(H, u, v, max_level):
    #print('delete: ', u, ', ', v)
    H.remove_edge(u, v)
    H.nodes[v]['witnesses'].discard(u)
    affected_nodes = SortedSet()
    affected_nodes.add(v) #affected nodes are on the same level by definition
    #print('before readjustment')
    readjustGraph(H, affected_nodes, max_level)
    #print('after readjustment')


def addOneClient(G, H, c, source_node, max_level):
    if c % 1000 == 0: print(c)
    for s in G.neighbors(c):
        H.add_edge(s, c)
        #print('added: ', s, ', ', c)
    deleteEdge(H, source_node, c, max_level)


def findSAP(H, c, source_node):
    if len(H.nodes[c]['witnesses']) > 0: #exists a SAP to c with length <= max_level
        found_short_path = True
        p = c
        lvl = H.nodes[c]['level']
        path = np.zeros((lvl, ), dtype=int)
        path[lvl - 1] = c
        for i in range(lvl - 2, -1, -1): #lvl-2 .. 0
            p = H.nodes[p]['witnesses'][0] #could take any of the witnesses!!!
            path[i] = p
            #P.add_edge(H.nodes[p]['witnesses'][0], p)
#    else: #have to find a SAP with BFS
#        path = np.array(nx.algorithms.shortest_path(H, source_node, c, weight=None))[1:] #array of consecutive path-nodes
#        #for i in range(len(path) - 1):
#        #    P.add_edge(path[i], path[i+1])
    else:
        found_short_path = False
        path = np.array([])
    return path, found_short_path


def applyPath(H, M, path, max_level, found_short_path, source_node):
    if found_short_path:
        path_length = len(path) - 1
        for i in range(path_length, 1, -2): #path_length .. 3, since path_length is odd
            M.add_edge(path[i], path[i-1])
            #print('added matching edge: ', path[i], ', ', path[i-1])
            M.remove_edge(path[i-2], path[i-1])
            #print('removed matching edge: ', path[i-2], ', ', path[i-1])
        M.add_edge(path[1], path[0])

        for i in range(path_length, 0, -1): #path_length .. 1
            H.add_edge(path[i], path[i-1])
            #print('added reversed path edge: ', path[i], ', ', path[i-1])
            deleteEdge(H, path[i-1], path[i], max_level)
            #print('removed path edge: ', path[i-1], ', ', path[i])
        deleteEdge(H, source_node, path[0], max_level)


def addClients(G, clients_to_add, H, M, source_node, max_level):
    for c in clients_to_add:
        #print('client to add: ', c)
        addOneClient(G, H, c, source_node, max_level)
        #print('added client: ', c)
        (path, found_short_path) = findSAP(H, c, source_node)
        #print('found alternating path: ', path)
        #print('applying path')
        applyPath(H, M, path, max_level, found_short_path, source_node)
        #print('applied path')


def deleteClients(clients_to_delete, H, M, source_node, max_level):
    for c in clients_to_delete:
        for s in set(H.predecessors(c)):
            deleteEdge(H, s, c, max_level)
        for s in set(H.successors(c)):
            deleteEdge(H, c, s, max_level)

        for s in set(M.successors(c)):
            M.remove_edge(c, s)

    for c in clients_to_delete:
        H.add_edge(source_node, c)
        H.nodes[c]['witnesses'] = SortedSet()
        H.nodes[c]['witnesses'].add(source_node)
        H.nodes[c]['level'] = 1


#server_nodes = SortedSet({n for n, d in G.nodes(data=True) if d['bipartite']==0})
#client_nodes = SortedSet(set(G) - server_nodes)

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
#H = nx.DiGraph()

#H.add_node(source_node, level=0)
#H.add_nodes_from(client_nodes, level=1)

#for s in server_nodes:
#    H.add_edge(source_node, s)
#    H.nodes[s]['witnesses'] = SortedSet()
#    H.nodes[s]['witnesses'].add(source_node)

#for c in client_nodes:
#    H.add_edge(source_node, c)
#    H.nodes[c]['witnesses'] = SortedSet()
#    H.nodes[c]['witnesses'].add(source_node)

#initialize M
#M = nx.DiGraph()

#initialize path graph
#P = nx.DiGraph()

#free_servers = SortedSet(server_nodes - set(M))

#add all the clients 1 by 1 while maintaining the ES structure
#clients_to_add = client_nodes
#addClients(G, clients_to_add, H, M, source_node, max_level)

#add (a batch of) clients
#clients_to_add=... #Todo from network
#addClients(G, clients_to_add, H, M, source_node, max_level)

#delete (a batch of) clients
#clients_to_delete=... #Todo from network
#deleteClients(clients_to_delete, H, M, source_node, max_level)
