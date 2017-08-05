import networkx as nx
from collections import defaultdict
import numpy as np
class k_clique():

    def __init__(self):
        pass

    def train(self,networks):
        """
        No training actually takes place in this class.
        :param networks: list of egonet objects
        :return: Predictions on training networks
        """
        # Normally some training would take place here...

        return self.predict(networks)



    # This is the main clustering function
    # Our network is built on networkx so I will need it for the basic graph concepts
    # Like get nodes and edges
    # This will generate the prediction communities
    def predict(self,networks):

        # this function is the clique percolate method, it finds the max clique and discover the adjacent circles.
        # it is a generator that generates a set of clique communities, the return value is a set
        # it takes a graph g and a clique size k as input
        def find_k_clique(g, k):
            clique_set = bronKerbosch(g)
            clique_set = [frozenset(c) for c in clique_set if len(c) >= k]
            c_dic = defaultdict(list)
            for c in clique_set:
                for node in c:
                    c_dic[node].append(c)
            tempG = nx.Graph()
            tempG.add_nodes_from(clique_set)
            for c in clique_set:
                adjacent_cliques = set()
                for n in c:
                    for adj_clique in c_dic[n]:
                        if c != adj_clique:
                            adjacent_cliques.add(adj_clique)
                for adj_clique in adjacent_cliques:
                    if len(c.intersection(adj_clique)) >= (k - 1):
                        tempG.add_edge(c, adj_clique)
            for component in connectedComponents(tempG):
                yield frozenset.union(*component)


        # this function uses the bron kerbosch method to find the the maximal clique in the graph
        # this uses pivot optimization to optimize the runtime
        # this takes a graph g as input and generates a set of nodes
        def bronKerbosch(g):
            max = -1
            num_neighbors = {}
            num_pivot = set()
            S = []
            calculated = set()
            result = []
            for n, neighbor in iter(g.adj.items()):
                neighbor = set(neighbor)
                neighbor.discard(n)
                l_neighbors = len(neighbor)
                if l_neighbors > max:
                    num_neighbors[n] = num_pivot = neighbor
                    max = l_neighbors
                else:
                    num_neighbors[n] = neighbor
            temp = set(num_neighbors)
            temp2 = set(temp - num_pivot)
            while temp2 or S:
                if temp2:
                    n = temp2.pop()
                else:
                    temp, calculated, temp2 = S.pop()
                    result.pop()
                    continue
                result.append(n)
                temp.remove(n)
                calculated.add(n)
                neighbors = num_neighbors[n]
                temp_n = temp.intersection(neighbors)
                calculated_n = calculated & neighbors
                if not temp_n:
                    if not calculated_n:
                        yield result[:]
                    result.pop()
                    continue
                if not calculated_n and len(temp_n) == 1:
                    yield result + list(temp_n)
                    result.pop()
                    continue
                l_temp_n = len(temp_n)
                maxcalculated = -1
                for n in calculated_n:
                    cn = temp_n.intersection(num_neighbors[n])
                    l_neighbors = len(cn)
                    if l_neighbors > maxcalculated:
                        pivotdonenbrs = cn
                        maxcalculated = l_neighbors
                        if maxcalculated == l_temp_n:
                            break
                if maxcalculated == l_temp_n:
                    result.pop()
                    continue
                max = -1
                for n in temp_n:
                    cn = temp_n.intersection(num_neighbors[n])
                    l_neighbors = len(cn)
                    if l_neighbors > max:
                        num_pivot = cn
                        max = l_neighbors
                        if max == l_temp_n - 1:
                            break
                if maxcalculated > max:
                    num_pivot = pivotdonenbrs
                calculated = calculated_n
                S.append((temp, calculated, temp2))
                temp = temp_n
                temp2 = temp - num_pivot

        # this function is the page rank function it will return the nodes with page ranks associated with it
        # It takes a graph g and do matrix calculation on the graph
        # returns a dict of nodes with their pageRank
        def pagerank(g):
            if len(g) == 0:
                return {}
            N = g.nodes()
            nlen = len(N)
            a = zip(N, range(nlen))
            index = dict(a)
            mat = np.zeros((nlen, nlen))
            for v, neighbor in iter(g.adj.items()):
                for node, node2 in neighbor.items():
                    temp_t = node2.get(1, 1)
                    mat[index[v], index[node]] = temp_t
            mat = np.asmatrix(mat)
            (n, m) = (len(mat), len(mat[0]))
            if n == 0:
                return mat
            d = np.where(mat.sum(axis=1) == 0)
            for node2 in d[0]:
                mat[node2] = 1.0 / n
            mat = mat / mat.sum(axis=1)
            arr = np.ones((n))
            node = arr
            node = node/node.sum()
            mat = 0.8*mat+(1-0.8)*np.outer(arr,node)
            eigenvalues, eigenvectors = np.linalg.eig(mat.T)
            ind = eigenvalues.argsort()
            prime = np.array(eigenvectors[:, ind[-1]]).flatten().real
            total = prime.sum()
            total = float(total)
            m = map(float,prime/total)
            t = zip(N, m)
            return dict(t)

        # this function computes the connected components and returns the components.
        # It uses bfs to detect connected components
        def connectedComponents(g):
            V = {}
            for node in g:
                if node not in V:
                    S = {}
                    D = 0
                    Q = {node: 1}
                    while Q:
                        temp = Q
                        Q = {}
                        for node in temp:
                            if node not in S:
                                S[node] = D
                                Q.update(g[node])
                        D = D + 1
                    yield list(S)
                    V.update(S)

        # start from here we run the training on the given graphs and obtain the communities
        # initial clique size = 5 if node size too large change clique size to 7 to find more dense graph
        # first test the node num and apply pageRank Then use cliques to detect communities
        # It returns a list that has every network and the element in the list is a dictionary of predictions for that egoNet
        predictions = []
        for n in networks:
            cliqueSize = 5
            maxNodes = 50
            if len(n.G.nodes()) > maxNodes*6:
                cliqueSize = 7
            dictOfCircles = {}
            while len(n.G.nodes()) > maxNodes:
                temprankDictionary = pagerank(n.G)
                sortedlist = sorted(temprankDictionary, key = temprankDictionary.get,reverse = True)
                for ind, item in enumerate(sortedlist):
                    if ind >= maxNodes:
                        n.G.remove_node(item)


            k_clique_result = list(find_k_clique(n.G, cliqueSize))
            count = 0
            for community in k_clique_result:
                dictOfCircles[count] = (list(community))
                count += 1
            predictions.append(dictOfCircles)
            print(dictOfCircles)
        return predictions




