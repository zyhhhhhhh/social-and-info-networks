import community as cm
import networkx as nx
class Louvain:
    """
    Louvain community detection
    """

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
        Randomly constract valid circles given initialized hyper parameters
        :param networks:
        :return:
        """


        predictions = []
        for n in networks:

            #if not issubclass(n,egonet):
            #    raise TypeError('Given a non egonet item: %s'%type(n))
            if len(n.G.nodes()) > 200:
                num_remove = int(0.2*len(n.G.nodes()))
                temprankDictionary = nx.pagerank(n.G)
                sortedlist = sorted(temprankDictionary, key=temprankDictionary.get)
                for ind in range(num_remove):
                    n.G.remove_node(sortedlist[ind])
            communities = cm.best_partition(n.G)
            predicted_circles = {}
            for node,comm in communities.items():
                if comm not in predicted_circles:
                    predicted_circles[comm] = [node]
                else:
                    predicted_circles[comm].append(node)

            predictions.append(predicted_circles)
            print("circles")
            print(predicted_circles)

        return predictions



