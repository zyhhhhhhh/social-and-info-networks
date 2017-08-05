from collections import Counter
import random

class LPA:
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
        for egonet in networks:
            n = egonet.G
            # Initially each node is labeled by themselves
            label = dict(zip(n.nodes(),n.nodes()))

            isConverged = False
            while not isConverged:
                isConverged = True
                for node in n.nodes():
                    c = Counter(label[_] for _ in n.edge[node])
                    most = max(c.values())
                    # When the label of the node is not among the most of its neighbors' labels
                    if c[label[node]] != max(c.values()):
                        # Update the label to be one of the most choosed label of its beighborhood
                        label[node] = random.choice([_ for _ in c if c[_] == most])
                        isConverged = False

            predicted_circles = {}
            for node in n.nodes():
                if label[node] not in predicted_circles:
                    predicted_circles[label[node]] = [node]
                else:
                    predicted_circles[label[node]].append(node)

            predictions.append(predicted_circles)
            print("circles")
            print(predicted_circles)
        return predictions



