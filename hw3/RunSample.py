from heapq import heappush, heappop
from itertools import count
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import sys

path = "sample.txt"

print("------------------------------------------------------------")
print("using exact betweenness")
print("------------------------------------------------------------")
f = open(path,'r')
edge_list = f.read().splitlines()
edge_list = [eval(x) for x in edge_list]
g =nx.Graph()
g.add_edges_from(edge_list)
startEdges = g.number_of_edges()
def calculate_edge_betweenness(G):
    betweenness = dict.fromkeys(G.edges(), 0.0)
    for s in G.nodes():
        S, P,total = bfs(G, s)
        betweenness = sum_edges(betweenness, S, P, s,total)
    for v in betweenness:
        betweenness[v] *= 0.5
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

def sum_edges(betweenness, S, P,s,total):
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


result = []
for i in range(1,6):
    temp = [i]
    M = g.number_of_edges()
    parts = list(nx.connected_component_subgraphs(g))
    sum = 0
    for item in parts:
        sum += calculate_modularity(item, g, M)
    temp.append(startEdges-M)
    temp.append(sum)
    result.append(temp)
    if i != 5:
        while(nx.number_connected_components(g)==i):
            b = calculate_edge_betweenness(g)
            maxedge = (max(b,key = b.get))
            g.remove_edge(*maxedge)
print("Number of community     Cumulative Edges Removed   Modularity")
for element in result:
    print(element)





