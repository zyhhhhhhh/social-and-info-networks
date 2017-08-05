from heapq import heappush, heappop
from itertools import count
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
path = "WattsStrogatz.txt"
# path = "ErdosRenyi.txt"
# path = "Barabasi.txt"
# path = "sample.txt"
# path = "testgraph.txt"

print(path)
f = open(path,'r')
edge_list = f.read().splitlines()
edge_list = [eval(x) for x in edge_list]
g =nx.Graph()
g.add_edges_from(edge_list)
startEdges = g.number_of_edges()

def approximate_calculate_edge_betweenness(G):
    betweenness = dict.fromkeys(G.edges(), 0.0)
    V = G.number_of_nodes()
    K = V/10
    CV = 5*V
    nodes = G.nodes()
    min_key = min(betweenness, key=betweenness.get)
    counter = 0
    while betweenness[min_key] < CV and counter < K:
        index_random = random.randrange(len(nodes))
        s = nodes[index_random]
        S, P, total = bfs(G, s)
        betweenness = sum_edges(betweenness, S, P,total)
        min_key = min(betweenness, key=betweenness.get)
        counter += 1
    for v in betweenness:
        betweenness[v] *= (V/K)
    return betweenness

def bfs(G, s):
    S = []
    P = {}
    for v in G:
        P[v] = []
    D = {}
    D[s] = 0
    Q = [s]
    total = dict.fromkeys(G, 0)
    total[s] = 1
    while Q:
        v = Q.pop(0)
        if v not in total:
            total[v] = 0
        S.append(v)
        for w in G.neighbors(v):
            if w not in D:
                Q.append(w)
                D[w] = D[v] + 1
            if D[w] == D[v] + 1:
                total[w] += total[v]
                P[w].append(v)
    return S, P,total


def sum_edges(betweenness, S, P,total):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        k = (1.0 + delta[w]) / total[w]
        for v in P[w]:
            d = total[v] * k
            if (v, w) not in betweenness:
                betweenness[(w, v)] += d
            else:
                betweenness[(v, w)] += d
            delta[v] += d
    return betweenness


def calculate_modularity(part_G,orig_G,m):
    # print("m",m)
    E = part_G.number_of_edges()
    # print("E",E)
    nodes = part_G.nodes()
    s = 0
    for i in nodes:
        element = orig_G.degree(i)
        s+= element
        # print("element",element)
    result = E/m - (s/(2*m))**2
    # print("result",result)
    return result

for i in range(1,6):
    print(i, " connected part")
    M = g.number_of_edges()
    parts = list(nx.connected_component_subgraphs(g))
    sum = 0
    for item in parts:
        print(item.nodes())
        sum += calculate_modularity(item, g, M)
    print("Modularity score",sum)
    print("edges removed:", startEdges-M)
    if i != 5:
        while(nx.number_connected_components(g)==i):
            b = approximate_calculate_edge_betweenness(g)
            maxedge = (max(b,key = b.get))
            g.remove_edge(*maxedge)



