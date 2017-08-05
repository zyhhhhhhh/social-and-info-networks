from csv import reader
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import csv
np.set_printoptions(threshold=np.inf)

def calculate_potential(p_buyer,p_seller):
    return np.sum(p_buyer+p_seller)


with open('preference.csv', 'r') as f:
    buyer = list(reader(f))

# initialize seller
seller = np.zeros(len(buyer[0])-1)
count = 0
potential = []
# loop
while count != 100:
    # initialize prefer
    prefer = np.zeros((len(buyer),len(buyer[0])-1))
    # construct preferred seller matrix
    p_buyer = np.zeros(len(buyer))
    G=nx.Graph()
    for i in range(len(buyer)):
        payoff = np.array(buyer[i][1:],dtype = int) - seller
        maxpayoff = max(payoff)
        p_buyer[i] = maxpayoff
        for j in range(len(buyer[0])-1):
            G.add_node(i, bipartite=0)
            G.add_node("seller"+str(j),bipartite=1)
            if payoff[j] == maxpayoff:
                prefer[i][j] = 1
                G.add_edge(i,"seller"+str(j))
    potential.append(calculate_potential(p_buyer,seller))
    bipart_result = nx.bipartite.maximum_matching(G)
    match_edge = [ [k,v] for k, v in bipart_result.items()]
    best_match_edge = np.repeat(-1,len(buyer))
    no_match_edge = copy.deepcopy(prefer)
    # construct best_match_edge
    for item in match_edge:
        if isinstance( item[0], int ):
            best_match_edge[int(item[0])]=int(item[1].split("seller")[1])
        else:
            best_match_edge[int(item[1])]=int(item[0].split("seller")[1])
    count = 0
    for item in best_match_edge:
        if item != -1:
            count += 1
    print(count)
    # construct no_match_edge
    for i in range(len(buyer)):
        if best_match_edge[i]!= -1:
            no_match_edge[i,best_match_edge[i]] = 0
    # bfs
    constricted = []
    for i in range(len(buyer)):
        if best_match_edge[i] == -1:
            constricted.append(i)
            break
    # print(constricted)
    # print(no_match_edge)
    flag = 0
    bfs_s_list = []
    bfs_b_list = []
    # for seller, use no match edge
    for j in range(100):
        if no_match_edge[i][j] == 1:
            bfs_s_list.append(j)
    no_match = copy.deepcopy(no_match_edge)
    while flag!= 1:
        # for buyer, use matched edge
        for item in bfs_s_list:
            for c in range(len(best_match_edge)):
                if best_match_edge[c] == item:
                    bfs_b_list.append(c)
        bfs_s_list[:] = []
        # check if all leaf
        for item in bfs_b_list:
            for j in range(100):
                if no_match[item][j] == 1:
                    bfs_s_list.append(j)
                    no_match[item][j] = -1
                if (j == 99 and no_match[item][j] != 1):
                    constricted.append(item)
        bfs_b_list[:] = []
        if not bfs_s_list:
            flag = 1
    constricted = set(constricted)
    s = []
    while constricted:
        b = constricted.pop()
        for i in range(100):
            if prefer[b][i] == 1:
                s.append(i)
    s = set(s)
    while s:
        seller[s.pop()] += 1
    if min(seller) > 0:
        seller = seller-min(seller)

print(potential)
print(best_match_edge)
print(seller)
payoff_final = []
for i in range(100):
    btemp = buyer[i][1:]
    temp = int(btemp[best_match_edge[i]])-seller[best_match_edge[i]]
    payoff_final.append(temp)
print(payoff_final)
# csv
with open('market-clearing.csv', 'w') as csvfile:
    fieldnames = ['buyer', 'house_name',"buyer's payoff"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(100):
        writer.writerow({'buyer': i, 'house_name': ('house'+str(best_match_edge[i])) , "buyer's payoff": p_buyer[i]})
# plot
plt.plot(potential)
plt.xlabel("Round")
plt.ylabel("Potential-energy")
plt.title("Potential-energy vs Round")
plt.show()
# X, Y = nx.bipartite.sets(G)
# pos = dict()
# pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
# pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
# nx.draw(G, pos=pos)
# plt.show()
# nx.draw(G)
# plt.show()
