from igraph import *
import random
import numpy as np
import matplotlib.pyplot as plt
# generate Graph
def generate_graph():
    g = Graph.Full(10, directed=False)
    g.es["weight"] = 0
    for item in g.es:
        item["weight"] = random.getrandbits(1)
    return g

def balance_graph(g,num_b):
    # balance the graph(Algorithm)
    a = list(range(0,10))
    nodes = random.sample(a, 3)
    eids = [g.get_eid(nodes[0],nodes[1]),g.get_eid(nodes[0],nodes[2]),g.get_eid(nodes[1],nodes[2])]
    # check if balanced
    w_tol = g.es[eids[0]]["weight"]+g.es[eids[1]]["weight"]+g.es[eids[2]]["weight"]
    if w_tol==0 or w_tol ==2:
        i = random.randint(0,2)
        if g.es[eids[i]]["weight"] == 0:
            g.es[eids[i]]["weight"] = 1
        else:
            g.es[eids[i]]["weight"] = 0
        num_b = balance_triads(g)
    return g,num_b

def balance_triads(g):
    sum = 0
    for i in range(0, 8):
        for j in range(i + 1, 9):
            for k in range(j + 1, 10):
                eids = [g.get_eid(i, j), g.get_eid(i, k), g.get_eid(j, k)]
                w_tol = g.es[eids[0]]["weight"] + g.es[eids[1]]["weight"] + g.es[eids[2]]["weight"]
                if w_tol == 1 or w_tol == 3:
                    sum += 1
    return sum

total_list = []
for i in range(0,100):
    g = generate_graph()
    num_balanced = balance_triads(g)
    b_list = []
    for j in range(0,1000000):
        g,num_balanced = balance_graph(g,num_balanced)
        b_list.append(num_balanced)
    total_list.append(b_list)
average_list = np.mean(total_list, axis=0)/120
x = np.arange(0,1000000)
plt.plot(x,average_list)
plt.xscale('log')
plt.title("Average fraction of balanced triads at iteration i")
plt.xlabel("iteration")
plt.ylabel("fraction of balanced triads")
plt.show()


