import igraph as ig
import community as cm
import networkx as nx
import pandas as pd


class infomap:

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


    def predict(self,networks):
        """
        something
        Randomly constract valid circles given initialized hyper parameters
        :param networks:
        :return:
        """
        predictions = []
        # f = open('yzhng127result', 'w')
        # f.write('hi there\n')  # python will convert \n to os.linesep
        for n in networks:
            index = 0
            cliqueSize = 6
            tooLittleFriendsInCircleThreshold = 6
            # tooManyNodesThreshold = 120
            # if len(n.G.nodes()) > tooManyNodesThreshold:
            #     print 'skipping user ' + str(n.egoid)
            #     continue
            # else:
            #     print 'predicting for user ' + str(n.egoid)

            # find comunities using k_clique_communities()
            # kCliqueComunities = list(community_infomap(n.G, cliqueSize))
            dictOfCircles = {}
            count = 0
            # g2 = ig.Graph.Adjacency((nx.to_numpy_matrix(n.G) > 0).tolist())
            # print(n.G.nodes())
            g1 = ig.Graph()
            # g1.add_vertex(n.G.nodes())
            # print(g1.vs)
            for item in n.G.edges():
                g1.add_edge(eval(item[0]),eval(item[1]))
            # g1.add_edges(n.G.edges())
            result = list(g1.community_infomap())
        #
            for community in result:
                # leave only relativly large communities
                if len(community) >= tooLittleFriendsInCircleThreshold:
                    dictOfCircles[count] = (list(community))
                    count += 1
            predictions.append(dictOfCircles)
            print(dictOfCircles)
        # #     f.write(predicted_circles)
        # # f.close()
        return predictions