import networkx as nx
#import itertools
from sortedcontainers import SortedSet
import numpy as np

#G = nx.Graph() #<----TODO input from NAT_graph, placeholder atm
#max_level = 735 #int: sqrt(n)*sqrt(log(n))
#source_node = -1 #int: index of the source node
#M = nx.DiGraph() #matching, directed from C to S
#n = 50000


def readjustGraph(H, parents_by_level, levels, best_gains, affected_nodes, gain_changed, max_level):
    lvl = levels[gain_changed[0]]
    if lvl > max_level:
        #print(lvl, "túllépés")
        return
    new_affected_nodes = SortedSet()
    #might_be_affected = SortedSet()
    new_gain_changed = SortedSet()
    might_change_gain = SortedSet()
    new_old_gains = {}
    #neighbours = SortedSet()
    #print('affected nodes: ', affected_nodes)

    for v in gain_changed:
        for w in H.successors(v):
            if not levels[w] == lvl + 1:
                #readjust for the changed best gain of v
                l = [item for item in parents_by_level[(w, lvl)] if item[0] == v]
                if len(l) > 0:
                    parents_by_level[(w, lvl)].discard(l[0])
                if v not in affected_nodes:
                    parents_by_level[(w, lvl)].add((v, parents_by_level[(v, lvl - 1)][0][1] + H.edges[v, w]['weight']))
            else:
                might_change_gain.add(w)

    for w in might_change_gain: #levels[w] == lvl + 1
        if parents_by_level[(w, lvl)][0][0] in gain_changed:
            new_gain_changed.add(w)
            #new_old_gains[parents_by_level[(w, lvl)][0][0]] = parents_by_level[(w, lvl)][0][1] #store old best gain

    for v in gain_changed:
        for w in H.successors(v):
            if levels[w] == lvl + 1:
                #readjust for the changed best gain of v
                l = [item for item in parents_by_level[(w, lvl)] if item[0] == v]
                if len(l) > 0:
                    parents_by_level[(w, lvl)].discard(l[0])
                if v not in affected_nodes:
                    parents_by_level[(w, lvl)].add((v, parents_by_level[(v, lvl - 1)][0][1] + H.edges[v, w]['weight']))

    #raise levels
    for v in affected_nodes:
        new_gain_changed.add(v)
        levels[v] = levels[v] + 1

    for w in new_gain_changed:
        has_witness = False
        for v in H.predecessors(w):
            if levels[v] == lvl:
                has_witness = True
        if has_witness: #node stops falling
            for x in H.successors(w):
                parents_by_level[(x, lvl + 1)].add((w, parents_by_level[(w, lvl)][0][1] + H.edges[w, x]['weight']))
        else: #node will fall further
            new_affected_nodes.add(w)

    if len(new_gain_changed) > 0:
        readjustGraph(H, parents_by_level, levels, best_gains, new_affected_nodes, new_gain_changed, max_level)


def deleteEdge(H, parents_by_level, levels, best_gains, u, v, max_level):
    #print('delete: ', u, ', ', v)
    gain_changed = SortedSet()
    affected_nodes = SortedSet()
    #old_gains = {}
    wt = H.edges[u, v]['weight']
    H.remove_edge(u, v)
    if levels[v] == levels[u] + 1:
        if parents_by_level[(v, levels[u])][0][0] == u:
            gain_changed.add(v)
            #old_gains[v] = parents_by_level[(v, levels[u])][0][1]
    #print((u, levels[u] - 1))
    #print(parents_by_level[(u, levels[u] - 1)][0][1] + wt)
    if levels[u] == 0: #u == source_node
        parents_by_level[(v, 0)].discard((u, 0))
    else:
        parents_by_level[(v, levels[u])].discard((u, parents_by_level[(u, levels[u] - 1)][0][1] + wt))  #roundabout way
    if levels[v] == levels[u] + 1:
        if len(parents_by_level[(v, levels[v] - 1)]) == 0:
            affected_nodes.add(v) #affected nodes are on the same level by definition; always a subset of gain_changed
    if not len(gain_changed) == 0:
        #print('before readjustment')
        readjustGraph(H, parents_by_level, levels, best_gains, affected_nodes, gain_changed, max_level)
        #print('after readjustment')


def addOneClient(G, H, parents_by_level, levels, best_gains, c, source_node, max_level):
    if c % 1000 == 0: print(c)
    for s in G.neighbors(c):
        H.add_edge(s, c, weight=G.edges[c, s]['weight'])
        if levels[s] <= max_level:
            parents_by_level[(c, levels[s])].add((s, parents_by_level[(s, levels[s] - 1)][0][1] + H.edges[s, c]['weight']))
        #print('added: ', s, ', ', c)
    deleteEdge(H, parents_by_level, levels, best_gains, source_node, c, max_level)
    #if levels[c] == 4: print(levels[c])


def findSAP(H, parents_by_level, levels, best_gains, c):
    if len(parents_by_level[(c, levels[c] - 1)]) > 0: #exists a SAP to c with length <= max_level
        found_short_path = True
        p = c
        lvl = levels[c]
        #if lvl > 2: print(lvl)
        path = np.zeros((lvl, ), dtype=int)
        path[lvl - 1] = c
        for i in range(lvl - 2, -1, -1): #lvl-2 .. 0
            #if lvl > 2: print("lehetőség: ", len(parents_by_level[(p, i+1)]), "p: ", p, "szint: ", levels[p])
            #if len(parents_by_level[(p, i+1)]) == 0:
            #    print(p, levels[p])
            #    print(list(H.successors(p)))
            #    for y in H.successors(p):
            #        print(y, levels[y])
            #    print(list(H.predecessors(p)))
            #    for y in H.predecessors(p):
            #        print(y, levels[y])
            p = parents_by_level[(p, i + 1)][0][0] #taking the smallest gain
            #print(p)
            path[i] = p
        #if lvl > 2: print(path)
        #if lvl == 4:
        #    print('4 long path')
            #P.add_edge(H.nodes[p]['witnesses'][0], p)
    #else: #have to find a SAP with BFS
    #    path = np.array(nx.algorithms.shortest_path(H, source_node, c, weight=None))[1:] #array of consecutive path-nodes
    #     #for i in range(len(path) - 1):
    #     #    P.add_edge(path[i], path[i+1])
    else:
        found_short_path = False
        path = np.array([])
    return path, found_short_path


def applyPath(H, parents_by_level, levels, best_gains, M, path, max_level, found_short_path, source_node):
    if found_short_path:
        path_length = len(path) - 1
        #if path_length > 2: print(path_length)
        for i in range(path_length, 1, -2): #path_length .. 3, since path_length is odd
            M.add_edge(path[i], path[i-1], weight=H.edges[path[i-1], path[i]]['weight'])
            #print('added matching edge: ', path[i], ', ', path[i-1])
            M.remove_edge(path[i-2], path[i-1])
            #print('removed matching edge: ', path[i-2], ', ', path[i-1])
        M.add_edge(path[1], path[0], weight=H.edges[path[0], path[1]]['weight'])

        for i in range(path_length, 0, -1): #path_length .. 1
            H.add_edge(path[i], path[i-1], weight=-(H.edges[path[i-1], path[i]]['weight']))
            parents_by_level[(path[i-1], levels[path[i]])].add((path[i], parents_by_level[(path[i], levels[path[i]] - 1)][0][1] + H.edges[path[i], path[i-1]]['weight']))
            #print('added reversed path edge: ', path[i], ', ', path[i-1])
            deleteEdge(H, parents_by_level, levels, best_gains, path[i-1], path[i], max_level)
            #print('removed path edge: ', path[i-1], ', ', path[i])
        #print('path: ', path)
        deleteEdge(H, parents_by_level, levels, best_gains, source_node, path[0], max_level)


def applyServerPath(H, parents_by_level, levels, best_gains, M, path, max_level, found_short_path, source_node):
    if found_short_path:
        path_length = len(path) - 1
        #if path_length > 2: print(path_length)
        for i in range(path_length, 0, -2): #path_length .. 2, since path_length is even
            M.add_edge(path[i-1], path[i-2], weight=H.edges[path[i-2], path[i-1]]['weight'])
            #print('added matching edge: ', path[i], ', ', path[i-1])
            M.remove_edge(path[i-1], path[i])
            #print('removed matching edge: ', path[i-2], ', ', path[i-1])

        for i in range(path_length, 0, -1): #path_length .. 2
            H.add_edge(path[i], path[i-1], weight=-(H.edges[path[i-1], path[i]]['weight']))
            parents_by_level[(path[i-1], levels[path[i]])].add((path[i], parents_by_level[(path[i], levels[path[i]] - 1)][0][1] + H.edges[path[i], path[i-1]]['weight']))
            #print('added reversed path edge: ', path[i], ', ', path[i-1])
            deleteEdge(H, parents_by_level, levels, best_gains, path[i-1], path[i], max_level)
            #print('removed path edge: ', path[i-1], ', ', path[i])
        #print('path: ', path)
        deleteEdge(H, parents_by_level, levels, best_gains, source_node, path[0], max_level)


def addClients(G, clients_to_add, H, parents_by_level, levels, best_gains, M, source_node, max_level):
    for c in clients_to_add:
        #print('client to add: ', c)
        addOneClient(G, H, parents_by_level, levels, best_gains, c, source_node, max_level)
        #print('added client: ', c)
        #lvldistr = [0, 0, 0, 0, 0]
        #for i in range(50000):
        #    lvldistr[levels[i] - 1] = lvldistr[levels[i] - 1] + 1
        #    if levels[i] == 2:
        #        print("lvl 2: ", i)
        #    if levels[i] == 5:
        #        print("lvl 5: ", i)
        #print("client level distribution: ", lvldistr)
        #lvldistr = [0, 0, 0, 0, 0]
        #for i in range(50000, 100000):
        #    lvldistr[levels[i] - 1] = lvldistr[levels[i] - 1] + 1
        #    if levels[i] == 5:
        #        print("lvl 5: ", i)
        #print("server level distribution", lvldistr)
        (path, found_short_path) = findSAP(H, parents_by_level, levels, best_gains, c)
        #print('found alternating path: ', path)
        #print('applying path')
        applyPath(H, parents_by_level, levels, best_gains, M, path, max_level, found_short_path, source_node)
        #print('applied path')


def deleteClients(clients_to_delete, H, parents_by_level, levels, best_gains, M, source_node, max_level):
    for c in clients_to_delete:
        for s in set(H.predecessors(c)):
            deleteEdge(H, parents_by_level, levels, best_gains, s, c, max_level)
        for s in set(H.successors(c)):
            deleteEdge(H, parents_by_level, levels, best_gains, c, s, max_level)

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
