import random
import matplotlib.pyplot as plt
def random_pick(N):
    random.seed()
    return random.sample(range(1000),N)

def degree_pick(N,degree_dict):
    temp = sorted(degree_dict, key=degree_dict.get,reverse = True)
    return temp[:N]
def central_pick(N,adj_dict,num_nodes):
    distance = {}
    # do bfs from i to all
    for i in range(num_nodes):
        S= {}
        S[i] = 0
        Q = adj_dict[i]
        d = 0
        while Q:
            d += 1
            temp = []
            for node in Q:
                if node not in S:
                    S[node] = d
                    temp = temp+ adj_dict[node]
            Q = temp
        s = 0
        for j in range(len(adj_dict)):
            if j in S:
                s+=S[j]
            else:
                s+= 5*len(adj_dict)
        s/=num_nodes
        distance[i] = s
    temp = sorted(distance, key = distance.get)
    return temp[0:N]

def linear_threshold(start, degree,theta, adj,nodes):
    influenced = set(start)
    process = []
    for i in range(1000):
        for node in nodes:
            active = 0
            for n in adj[node]:
                if n in influenced:
                    active += 1
            b = active / degree[node]
            if b > theta[node]:
                influenced.add(node)
        process.append(len(influenced))
    # print(process)
    # print(influenced)
    return len(influenced)


def main(inputname, thetaname):
    # read theta
    f = open(thetaname, 'r')
    temp = f.readlines()
    theta = []
    for index, item in enumerate(temp):
        a =item.split("\t")
        a = a[1][:-1]
        a = eval(a)
        theta.append(a)
    # print(theta)
    nodes = range(len(theta))
    # read network
    adj = {}
    f = open(inputname,'r')
    temp = f.read().split("\n")
    for index,item in enumerate(temp):
        if item:
            (node1,node2) = item.split('\t')
            node1 = eval(node1)
            node2 = eval(node2)
            if node1 not in adj:
                adj[node1] = [node2]
            else:
                adj[node1].append(node2)

    # first: pick 5 random
    degree = {}
    for i in range(len(nodes)):
        degree[i] = len(adj[i])
    # print(random_pick(5))
    # print(degree_pick(5,degree))
    # print(central_pick(5,adj,len(nodes)))
    # print(central_pick(10, adj, len(nodes)))
    # print(central_pick(20, adj, len(nodes)))
    # print(central_pick(25, adj, len(nodes)))
    # print(central_pick(30, adj, len(nodes)))

    sizes  = [5,10,20,25,30]
    random_result = []
    for size in sizes:
        start = random_pick(size)
        random_result.append(linear_threshold(start,degree,theta,adj,nodes))
    print(random_result)
    degree_result = []
    for size in sizes:
        start = degree_pick(size,degree)
        degree_result.append(linear_threshold(start,degree,theta,adj,nodes))
    print(degree_result)
    central_result = []
    for size in sizes:
        start = central_pick(size,adj,len(nodes))
        central_result.append(linear_threshold(start,degree,theta,adj,nodes))
    print(central_result)
    line1, = plt.plot(sizes,random_result, label='random')
    line2, = plt.plot(sizes, degree_result, label='degree')
    line3, = plt.plot(sizes, central_result, label='central')
    plt.legend(handles=[line1, line2, line3],loc = 2)
    plt.savefig('influence-result.png')
    plt.show()
    print("number of active nodes for RANDOM selection with initial size of 20")
    print(random_result[3])
    print("number of active nodes for DEGREE selection with initial size of 20")
    print(degree_result[3])
    print("number of active nodes for CENTRAL selection with initial size of 20")
    print(central_result[3])

main("network.txt","theta.txt")
# main("input.txt","theta.txt")


